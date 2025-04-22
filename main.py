"""
Copyright (c) 2024 Siemens AG Author: Steffen Limmer
SPDX-License-Identifier: MIT
"""

import itertools as it
import json
import os
import time

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import uqtestfuns as uqtf
import yaml
from jax.example_libraries import stax
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

from model import MLP

jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig):
    # Print the configuration
    print(cfg)
    folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logfile = os.path.join(folder, 'opt_log.txt')

    # Load and prepare the data
    if isinstance(cfg.dataset_id, int):
        dataset = fetch_ucirepo(id=cfg.dataset_id)
        X = jnp.array(dataset.data.features)
        y = jnp.array(dataset.data.targets)
    elif isinstance(cfg.dataset_id, str):  # load data from uqtf
        dataset = dict()
        dataset["metadata"] = dict()
        dataset["metadata"]["name"] = cfg.dataset_id

        num_points = 10_000
        if cfg.dataset_id == 'piston':
            testfun = uqtf.Piston()
        elif cfg.dataset_id == 'ishigami':
            testfun = uqtf.Ishigami()
        elif cfg.dataset_id == 'circuit':
            testfun = uqtf.OTLCircuit()
        else:
            raise NotImplementedError
        xx_sample = testfun.prob_input.get_sample(num_points)
        yy_sample = testfun(xx_sample)
        X = jnp.array(xx_sample)
        y = jnp.array(yy_sample).reshape(-1, 1)
    else:
        raise NotImplementedError

    if y.ndim > 1:
        y = jnp.atleast_2d(y[:, 0]).T
    input_dim = X.shape[1]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=cfg.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=cfg.seed)

    scaler_x = MinMaxScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_val = scaler_x.transform(X_val)
    X = scaler_x.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train).flatten()
    y_train += cfg.noise_var * jax.random.normal(jax.random.key(cfg.seed), y_train.shape)
    y_val = scaler_y.transform(y_val).flatten()
    y_test = scaler_y.transform(y_test).flatten()
    y = scaler_y.transform(y).flatten()

    # define activation map
    act_map = {'sigmoid': stax.Sigmoid,
               'swish': stax.elementwise(jax.nn.swish),
               'sin': stax.elementwise(jax.numpy.sin),
               'elu': stax.Elu,
               'relu': stax.Relu,
               'rep': stax.elementwise(lambda x: jnp.power(jax.nn.relu(x), input_dim + 1)
                                                 / jax.scipy.special.factorial(input_dim + 1)),
               }

    # train model
    start_train = time.time()
    # Instantiate MLP
    model = MLP([cfg.num_neurons] * (cfg.num_layers - 1), input_dim, cfg.mode, activation=act_map[cfg.act_str],
                logfile=logfile)
    best_params = model.train((X_train, y_train), (X_val, y_val), batch_size=128, max_epochs=cfg.max_epochs,
                              optimizer='bfgs', lr=1e-3, l2input=cfg.l2input, l2output=cfg.l2output, patience=10,
                              verbose=cfg.verbose)
    stop_train = time.time() - start_train
    num_params = sum([param.size for tup in best_params for param in tup])

    # predict the anova model
    if cfg.mode == 'anova':
        # obtain dictionary of predictions
        anova_preds, sigmas = model.predict_anova(model.params, X, verbose=cfg.verbose)

    # evaluate rmse for different truncated sums
    train_rmse = dict()
    val_rmse = dict()
    test_rmse = dict()

    # superposition dimension only applies for anova
    superpos_dims = list(range(input_dim + 1)) if cfg.mode == 'anova' else [input_dim]
    for superpos_dim in superpos_dims:
        if cfg.mode == 'anova':
            preds = sum([val for subset, val in anova_preds.items() if sum(subset) <= superpos_dim])
        else:
            preds = model.predict(model.params, X)
        _, X_temp, y_train_pred, y_temp = train_test_split(X, preds, test_size=0.4, random_state=cfg.seed)
        _, _, y_val_pred, y_test_pred = train_test_split(X_temp, y_temp, test_size=0.5, random_state=cfg.seed)
        train_rmse[f'sup_{superpos_dim}'] = jnp.sqrt(jnp.mean((y_train_pred.flatten() - y_train.flatten()) ** 2))
        val_rmse[f'sup_{superpos_dim}'] = jnp.sqrt(jnp.mean((y_val_pred.flatten() - y_val.flatten()) ** 2))
        test_rmse[f'sup_{superpos_dim}'] = jnp.sqrt(jnp.mean((y_test_pred.flatten() - y_test.flatten()) ** 2))

    # truncation for anova and mean imputation for mlp
    trunc_vars_list = [(i,) for i in range(input_dim)] + list(it.combinations(range(input_dim), 2))
    for trunc_vars in trunc_vars_list:
        if cfg.mode == 'anova':
            preds = sum(
                [val for subset, val in anova_preds.items() if all([subset[ii] != 1 for ii in trunc_vars])])
        else:
            X_trunc = jnp.array(X)
            for trunc_var in trunc_vars:
                X_trunc = X_trunc.at[:, trunc_var].set(jnp.mean(X[:, trunc_var]))
            preds = model.predict(model.params, X_trunc)
        _, X_temp, y_train_pred, y_temp = train_test_split(X, preds, test_size=0.4, random_state=cfg.seed)
        _, _, y_val_pred, y_test_pred = train_test_split(X_temp, y_temp, test_size=0.5, random_state=cfg.seed)
        train_rmse[f'trunc_{trunc_vars}'] = jnp.sqrt(jnp.mean((y_train_pred.flatten() - y_train.flatten()) ** 2))
        val_rmse[f'trunc_{trunc_vars}'] = jnp.sqrt(jnp.mean((y_val_pred.flatten() - y_val.flatten()) ** 2))
        test_rmse[f'trunc_{trunc_vars}'] = jnp.sqrt(jnp.mean((y_test_pred.flatten() - y_test.flatten()) ** 2))

    results = {
        'dataset_name': dataset["metadata"]["name"],
        'dataset_num_feat': X.shape[1],
        'dataset_num_samples': X.shape[0],
        'train_time': stop_train,
        'num_params': num_params,
        **cfg,
        **{f'train_rmse_order_{key}': float(val) for key, val in train_rmse.items()},
        **{f'val_rmse_order_{key}': float(val) for key, val in val_rmse.items()},
        **{f'test_rmse_order_{key}': float(val) for key, val in test_rmse.items()}
    }
    print(results)
    # Create a dataframe and store the results
    results_df = pd.DataFrame(results, index=[0])
    results_df.to_json(os.path.join(folder, "results.json"))
    with open(os.path.join(folder, "results.yaml"), 'w') as file:
        yaml.dump({'result': json.loads(results_df.to_json(orient='records'))}, file, default_flow_style=False)
    return True


if __name__ == '__main__':
    train()

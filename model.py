"""
Copyright (c) 2024 Siemens AG Author: Steffen Limmer
SPDX-License-Identifier: MIT
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import stax, optimizers
from jax.random import PRNGKey
from jax.experimental.jet import jet
import dill
import math
import time
import functools as ft
import itertools as it
from collections import defaultdict
import optax
from functools import partial
import jaxonloader

jax.config.update("jax_enable_x64", True)


class logger:
    def __init__(self, file):
        self.epoch = 0
        self.file = file
        self.train_hist = []
        self.val_hist = []
        self.time_hist = []
        self.best_params = []
        with open(self.file, "w") as file:
            file.write(f"")

    def __call__(self, train_err, val_err, params):
        with open(self.file, "a") as file:
            file.write(f"train_err {train_err}, val_err {val_err} \n")
        if val_err < min(self.val_hist + [math.inf]):
            self.best_params = params
        self.train_hist.append(train_err)
        self.val_hist.append(val_err)
        self.time_hist.append(time.time())


class MLP:
    def __init__(self, layer_sizes, input_dim, mode='default', diff_mode='nested', activation=stax.Sigmoid, seed=0,
                 logfile='opt_log.txt'):
        # Define the network
        W_init = jax.nn.initializers.variance_scaling(scale=len(layer_sizes), mode="fan_avg", distribution="uniform")
        b_init = jax.nn.initializers.truncated_normal(stddev=0.5)
        self.init_random_params, self.predict_fn = stax.serial(
            *(sum([[stax.Dense(size, W_init=W_init, b_init=b_init), activation] for size in layer_sizes], [])) +
             [stax.Dense(1, W_init=W_init, b_init=b_init)]
        )
        self.val_hist = []
        self.train_hist = []
        self.time_hist = []
        self.start_time = None
        self.mode = mode
        self.diff_mode = diff_mode
        self.input_dim = input_dim
        key = PRNGKey(seed)
        self.seed = seed
        self.key = key
        self.converged = False
        self.l2input = 0.
        self.l2output = 0.
        self.layer_sizes = layer_sizes
        _, self.params = self.init_random_params(key, (input_dim,))
        self._predict_ddx_fn = None
        # build nested derivative w.r.t. all variables (cf. 16)
        self._predict_ddx_fn = self.compose(*[ft.partial(jax.grad, argnums=i) for i in range(1, self.input_dim + 1)])\
            (self.predict_with_expanded_args)
        self.logfile = logfile
        self.logger = logger(file=logfile)
        self._activ_ind = list(it.product(*[[0, 1] for _ in range(input_dim)]))
        self._activ_ind = sorted(self._activ_ind, key=lambda tup: sum(tup))
        self._get_bounds = lambda dim: list(it.product(*[[0.0, 1.0] for _ in range(dim)]))
        self._get_signs = lambda dim: jnp.prod(jnp.array(self._get_bounds(dim)) * 2 - 1, axis=1)
        self._get_subsets = lambda curr_ind: list(it.product(*[(0, 1) if i == 1 else (0,) for i in curr_ind]))[:-1]

    def __getstate__(self):
        """ Remove useless objects before pickling. """
        keep_dict = dict((k, v) for k, v in self.__dict__.items()
                         if not isinstance(getattr(self, k), type(self.compose) | type(self._get_bounds)
                                           | type(self._activ_ind)))
        return keep_dict

    def compose(self, *functions):
        """ Helper to obtain nested derivatives. """
        return ft.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

    def predict_with_expanded_args(self, params, *args):
        """ Wrapper to predict with unstacked arguments and allow indexed derivatives. """
        out = self.predict_fn(params, jnp.stack(args)).reshape(-1)[0]
        return out

    def predict(self, params, inputs):
        """ Wrapper for the predict function. """
        if self.mode == 'anova' and self.diff_mode == 'nested':
            preds = jax.vmap(lambda x: self._predict_ddx_fn(params, *x))(inputs)
            return preds
        else:
            return self.predict_fn(params, inputs)

    def _predict_partial_ddx_fn(self, x, diff_args, subs_args):
        """ Wrapper for the predict function. Differentiates w.r.t. diff_args, substitutes subs_args and sums."""
        if self.diff_mode == 'nested':
            diff_fun = self.compose(*[ft.partial(jax.grad, argnums=i + 1) for i in diff_args]) \
                (self.predict_with_expanded_args)
        if len(subs_args) > 0.:
            subs_args = jnp.array(subs_args)
            return jnp.sum(
                jnp.array([sgn * diff_fun(self.params, *x.at[subs_args].set(jnp.array(bnds))) for bnds, sgn in
                           zip(self._get_bounds(self.input_dim - len(diff_args)),
                               self._get_signs(self.input_dim - len(diff_args)))]))
        else:
            return diff_fun(self.params, *x)

    def get_diff_subs_args(self, curr_idx):
        """ Utility function to obtain diff_args and subs_args at current index """
        diff_args = tuple([idx for idx, val in enumerate(curr_idx) if val == 1])
        subs_args = tuple([idx for idx, val in enumerate(curr_idx) if val == 0])
        return {'diff_args': diff_args, 'subs_args': subs_args}

    def predict_anova(self, params, inputs, max_order=jnp.inf, verbose=False):
        """ Prediction function for n-anova network. Returns dict of subset predictions, and dict of sigmas. Handles
        temporary sums to avoid recursive evaluation (23) """
        self.params = params
        anova_fast_tmp = dict()
        anova_fast = dict()
        sigmas_fast = dict()
        for curr_idx in self._activ_ind:
            if verbose:
                print(f'calculating subset: {curr_idx}')
            if sum(curr_idx) > max_order:
                sigmas_fast[curr_idx] = 0.
                anova_fast[curr_idx] = 0.
                continue
            subsets = self._get_subsets(curr_idx)
            subset_signs = [(-1) ** jnp.sum(jnp.array(curr_idx) - jnp.array(subset)) for subset in subsets]
            if len(subsets) > 0:
                summed_subsets = jnp.sum(
                    jnp.stack([sign * anova_fast_tmp[subset] for subset, sign in zip(subsets, subset_signs)], axis=1),
                    axis=1)
            else:
                summed_subsets = 0.
            anova_fast_tmp[curr_idx] = jax.vmap(
                ft.partial(self._predict_partial_ddx_fn, **self.get_diff_subs_args(curr_idx)))(inputs)
            anova_fast[curr_idx] = anova_fast_tmp[curr_idx] + summed_subsets
            if curr_idx == (0,) * self.input_dim:
                sigmas_fast[curr_idx] = 0.
            else:
                sigmas_fast[curr_idx] = jnp.mean(anova_fast[curr_idx] ** 2)
        return anova_fast, sigmas_fast

    def mse_loss(self, params, batch):
        """ Standard mse loss. """
        X_train, y_train = batch
        train_err = jnp.mean((self.predict(params, X_train).flatten() - y_train) ** 2)
        return train_err

    def rmse_loss_optx(self, params, batch):
        """ Adapted mse loss with l2-reg and logging callback to allow early stopping for bfgs. """
        batch_train, batch_val = batch
        X_train, y_train = batch_train
        X_val, y_val = batch_val
        train_err = jnp.mean((self.predict(params, X_train).flatten() - y_train) ** 2)
        val_err = jnp.mean((self.predict(params, X_val).flatten() - y_val) ** 2)

        l2_per_layer = [
            jnp.mean(jnp.square(p)) for p in list(jax.tree_util.tree_leaves(params)) if p.ndim > 1]
        l2_input = self.l2input * l2_per_layer[0]
        l2_output = self.l2output * l2_per_layer[-1]

        reg_loss = l2_input + l2_output
        jax.debug.print("train_err {train_err}, val_err {val_err}, reg_loss {reg_loss}",
                        train_err=train_err, val_err=val_err, reg_loss=reg_loss)
        jax.debug.callback(lambda x, y, prms: self.logger(x, y, prms), train_err, val_err, params)
        return jnp.sqrt(train_err) * (1.0 + reg_loss)

    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self = dill.load(f)
        return self

    def train(self, train_data, val_data, batch_size, max_epochs, optimizer, lr, l2input=0., l2output=0., seed=0,
              patience=10, verbose=False):
        """ Train function. Evaluated mostly with bfgs training to reduce runtime. Adam and others should also work"""
        self.start_time = time.time()
        self.l2output = l2output
        self.l2input = l2input
        params = self.params

        train_dl = jaxonloader.JaxonDataLoader(
            jaxonloader.DataTargetDataset(*train_data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Define optimizer
        if optimizer == 'sgd':
            optimizer = optax.sgd(lr)
        elif optimizer == 'adam':
            optimizer = optax.adam(lr)
        elif optimizer == 'rmsprop':
            optimizer = optax.rmsprop(lr)
        elif optimizer == 'adagrad':
            optimizer = optax.adagrad(lr)
        elif optimizer == 'none':
            return self.params
        elif optimizer == 'bfgs':
            import optimistix as optx
            bfgs_tol = 1e-12
            solver = optx.BFGS(rtol=bfgs_tol, atol=bfgs_tol)
            self.converged = False
            sol = optx.minimise(self.rmse_loss_optx, solver, self.params, max_steps=max_epochs, throw=False,
                                args=(train_data, val_data))
            self.params = self.logger.best_params
            if self.mode == 'default':
                self.converged = True
            else:
                anova_preds, _ = self.predict_anova(self.params, val_data[0], max_order=2, verbose=False)
                preds0 = sum([val for subset, val in anova_preds.items() if sum(subset) <= 0]).flatten()
                self.converged = jnp.mean((preds0 - val_data[1].flatten()) ** 2) <= 10.
            self.train_hist = self.logger.train_hist
            self.val_hist = self.logger.val_hist
            self.time_hist = self.logger.time_hist
            return self.params

        # classical sgd training loop, slow but should also work
        opt_state = optimizer.init(params)

        @jit
        def train_step(opt_state, params, batch):
            """Train for a single epoch."""
            grads = grad(self.mse_loss)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params

        # Run optimizer and do early stopping
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(max_epochs):
            for train_batch in train_dl:
                opt_state, params = train_step(opt_state, params, train_batch)
            val_loss = self.mse_loss(params, val_data)
            train_loss = self.mse_loss(params, train_data)
            if verbose:
                print(f"Epoch {epoch + 1}, Train {train_loss}, Val: {val_loss}")
            self.val_hist.append(val_loss)
            self.train_hist.append(train_loss)
            self.time_hist.append(time.time() - self.start_time)
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        self.params = best_params
        return self.params

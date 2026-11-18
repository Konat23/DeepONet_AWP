import torch
import numpy as np
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
from deepxde.nn.pytorch.mionet import MIONetCartesianProd


class FullCartesianProd(Data):
    """Dataset for the full Cartesian product of two branch inputs and a trunk input.

    X_train and X_test each contain three arrays:
        branch1_inputs: (n_branch1, branch1_input_dim)
        branch2_inputs: (n_branch2, branch2_input_dim)
        trunk_inputs:   (n_trunk, trunk_input_dim)

    Within each split, targets must have shape (n_branch1, n_branch2, n_trunk).
    targets[i, j, k] corresponds to the input combination
    (branch1_inputs[i], branch2_inputs[j], trunk_inputs[k]).

    Batch sizes:
        None: Return the full training set.
        int: Sample the first branch and keep the other two axes complete.
        tuple or list: Sample each axis independently using the three sizes
            (branch1_batch_size, branch2_batch_size, trunk_batch_size).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = self._validate_split(
            X_train, y_train, "training"
        )
        self.test_x, self.test_y = self._validate_split(
            X_test, y_test, "test"
        )

        n_branch1, n_branch2, n_trunk = self.train_y.shape
        self.branch1_sampler = BatchSampler(n_branch1, shuffle=True)
        self.branch2_sampler = BatchSampler(n_branch2, shuffle=True)
        self.trunk_sampler = BatchSampler(n_trunk, shuffle=True)

    @staticmethod
    def _validate_split(inputs, targets, split):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 3:
            raise ValueError(
                f"The {split} inputs must contain three arrays in the order "
                "(branch1_inputs, branch2_inputs, trunk_inputs)."
            )

        inputs = tuple(np.asarray(array) for array in inputs)
        targets = np.asarray(targets)

        for axis, array in zip(("branch1", "branch2", "trunk"), inputs):
            if array.ndim != 2 or array.shape[0] == 0:
                raise ValueError(
                    f"The {split} {axis} input must be a two-dimensional "
                    f"array with at least one row. Got shape {array.shape}."
                )

        expected_shape = tuple(array.shape[0] for array in inputs)
        if targets.shape != expected_shape:
            raise ValueError(
                f"The {split} targets must have shape {expected_shape} "
                f"in (branch1, branch2, trunk) order. Got {targets.shape}."
            )

        return inputs, targets

    @staticmethod
    def _validate_batch_size(size, available, axis):
        if (
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or not 1 <= size <= available
        ):
            raise ValueError(
                f"The batch size for {axis} must be an integer between "
                f"1 and {available}. Got {size!r}."
            )

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y

        branch1_inputs, branch2_inputs, trunk_inputs = self.train_x

        if not isinstance(batch_size, (tuple, list)):
            self._validate_batch_size(batch_size, len(branch1_inputs), "branch1")
            branch1_idx = self.branch1_sampler.get_next(batch_size)
            batch_inputs = (branch1_inputs[branch1_idx], branch2_inputs, trunk_inputs)
            return batch_inputs, self.train_y[branch1_idx]

        if len(batch_size) != 3:
            raise ValueError(
                "Expected three batch sizes in (branch1, branch2, trunk) order."
            )

        # Validate every axis before advancing any sampler.
        for size, available, axis in zip(
            batch_size, self.train_y.shape, ("branch1", "branch2", "trunk")
        ):
            self._validate_batch_size(size, available, axis)

        branch1_idx = self.branch1_sampler.get_next(batch_size[0])
        branch2_idx = self.branch2_sampler.get_next(batch_size[1])
        trunk_idx = self.trunk_sampler.get_next(batch_size[2])

        batch_inputs = (
            branch1_inputs[branch1_idx],
            branch2_inputs[branch2_idx],
            trunk_inputs[trunk_idx],
        )
        batch_targets = self.train_y[np.ix_(branch1_idx, branch2_idx, trunk_idx)]
        return batch_inputs, batch_targets

    def test(self):
        return self.test_x, self.test_y

class MIONetFullCartesianProd(MIONetCartesianProd):
    """Evaluate a MIONet over the full Cartesian product of three input sets.

    Input shapes:
        branch1_inputs: (n_branch1, branch1_input_dim)
        branch2_inputs: (n_branch2, branch2_input_dim)
        trunk_inputs:   (n_trunk, trunk_input_dim)

    Output shape:
        (n_branch1, n_branch2, n_trunk)

    All three networks must produce feature vectors of the same width.
    This variant uses multiplicative fusion with no optional merger networks.
    """

    def forward(self, inputs):
        if (
            self.merge_operation != "mul"
            or self.merger is not None
            or self.output_merger is not None
        ):
            raise ValueError(
                "MIONetFullCartesianProd requires merge_operation='mul', "
                "layer_sizes_merger=None, and layer_sizes_output_merger=None."
            )

        branch1_inputs, branch2_inputs, trunk_inputs = inputs

        branch1_features = self.branch1(branch1_inputs)
        branch2_features = self.branch2(branch2_inputs)

        if self._input_transform is not None:
            trunk_inputs = self._input_transform(trunk_inputs)

        trunk_features = self.trunk(trunk_inputs)
        if self.trunk_last_activation:
            trunk_features = self.activation_trunk(trunk_features)

        if not (
            branch1_features.shape[-1]
            == branch2_features.shape[-1]
            == trunk_features.shape[-1]
        ):
            raise ValueError(
                "Both branch networks and the trunk network "
                "must have the same output width."
            )

        # Keep the three input axes independent and sum over the feature axis.
        outputs = torch.einsum(
            "ip,jp,kp->ijk",
            branch1_features,
            branch2_features,
            trunk_features,
        ) + self.b

        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)

        return outputs
  
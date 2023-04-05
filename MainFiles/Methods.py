import numpy as np
import torch

"""
MDS, MMF and last methods used by the windowing class in the other file
"""


class Methods:
    # initialize results dictionary
    results = {}
    results['MDS'] = {}
    results['MMF'] = {}
    results['Last'] = {}

    def MDS(self, k=4, deprecated_weights=False, include_binary=False):
        """
        Applies MDS algorithm on all windows to retrieve the regression predictions.
        Saves the absolute error in the prediction. Also saves MDS version where you
        scale /adapt curves before picking them. Original version scales after picking
        """
        data = np.copy(self.data)
        data[..., self.triu[1], self.triu[0]] = 0

        if deprecated_weights:
            weights = np.arange(1, self.s + 1) ** 2
        else:
            weights = (2 ** np.arange(self.s))[:, None]

        # We have the following shape [new dataset, meta dataset, learner, (point on curve),... extra dims]
        scalar = np.sum(weights * data[None] * data[:, None], axis=3) / np.sum(weights * (data ** 2), axis=2)[None]
        adapted_curves = scalar[:, :, :, None] * data
        adapted_target = scalar[..., None] * self.target[..., None, :]
        distance = np.sum((data - data[:, None]) ** 2, axis=3)
        distance_adapted = np.sum((adapted_curves - data) ** 2, axis=3)

        # extra dim for moving target
        distance = np.repeat(distance[..., None], len(self.target_anchors), axis=-1)
        distance_adapted = np.repeat(distance_adapted[..., None], len(self.target_anchors), axis=-1)

        # Remove curves that can't predict at target
        ind = np.isnan(self.target).nonzero()
        distance[:, ind[0], ind[1], ..., ind[2]] = np.nan
        distance_adapted[:, ind[0], ind[1], ..., ind[2]] = np.nan

        # So that it doesn't pick itself
        np.einsum('ii...->i...', distance)[...] = np.nan
        np.einsum('ii...->i...', distance_adapted)[...] = np.nan

        if include_binary:
            # scale after, just like in the paper
            distance_sum = distance[:, :, None] + distance[:, :, :, None]

        # Take k closest
        part_scale_after = np.argpartition(distance, k, axis=1)[:, :k]
        part_scale_before = np.argpartition(distance_adapted, k, axis=1)[:, :k]
        k_closest_curves_scale_after = np.take_along_axis(adapted_target, part_scale_after, axis=1)
        k_closest_curves_scale_before = np.take_along_axis(adapted_target, part_scale_before, axis=1)

        if include_binary:
            # We need an extra dimension for the rival learner
            adapted_target_binary = np.repeat(adapted_target[:, :, :, None], adapted_target.shape[2], axis=3)
            part_binary = np.argpartition(distance_sum, k, axis=1)[:, :k]
            k_closest_curves_binary = np.take_along_axis(adapted_target_binary, part_binary, axis=1)

        # Predicted target is just the mean of the k closest curves at the target point
        prediction_scale_after = np.mean(k_closest_curves_scale_after, axis=1)
        prediction_scale_before = np.mean(k_closest_curves_scale_before, axis=1)

        # For binary it is a little more complicated, we need the prediction of both learners
        # then we check which one is larger
        if include_binary:
            # For rival learner we can just swap the learner axis
            k_closest_curves_binary_rival = np.swapaxes(k_closest_curves_binary, 2, 3)

            prediction_l1 = np.mean(k_closest_curves_binary, axis=1)
            prediction_l2 = np.mean(k_closest_curves_binary_rival, axis=1)

            # Check when l1 wins over l2
            prediction_binary = (prediction_l1 > prediction_l2).astype(float)

            # Check for ties
            prediction_binary[prediction_l1 == prediction_l2] = 0.5

            # Return nans as they get removed
            prediction_binary[np.isnan(prediction_l1)] = np.nan
            prediction_binary[np.isnan(prediction_l2)] = np.nan

            # Don't compete against same learner
            np.einsum('ijj...->ij...', prediction_binary)[...] = np.nan

        # Remove predictions for anchor points that are in front of the window
        ind = np.triu_indices(len(self.target_anchors), 1)
        prediction_scale_after[..., ind[1], ind[0]] = np.nan
        prediction_scale_before[..., ind[1], ind[0]] = np.nan
        if include_binary:
            prediction_binary[..., ind[1], ind[0]] = np.nan

        self.results['MDS']['test error'] = np.abs(prediction_scale_after - self.target[:, :, None])
        self.results['MDS']['(scale before) test error'] = np.abs(prediction_scale_before - self.target[:, :, None])
        self.results['MDS']['regression'] = prediction_scale_after
        self.results['MDS']['(scale before) regression'] = prediction_scale_before
        self.results['MDS']['k closest curves ID'] = part_scale_after
        self.results['MDS']['(scale before) k closest curves ID'] = part_scale_before
        self.results['MDS']['scalar'] = scalar
        if include_binary:
            self.results['MDS']['binary'] = prediction_binary

    def MMF(self, steps=500):
        '''
        Applies MMF algorithm. Tensorized with pytorch so that it calculates quickly
        '''

        # Tensorise variables
        target = torch.tensor(self.target, dtype=torch.float32)
        data = torch.tensor(self.data, dtype=torch.float32)
        train_anchors = torch.tensor(self.train_anchors, dtype=torch.float32)[:, None]
        weights = (2 ** torch.arange(self.s))[:, None]
        params = torch.tensor([0.5, 1, 1, -1], dtype=torch.float32)
        shp = np.hstack((np.array(data.shape[:2]), np.array(data.shape[3])))

        params = params.repeat(*shp, 1)[:, :, None]  # since len(dim[2]) = len(dim[3])

        params.requires_grad_()
        gr = torch.ones(*shp)  # For .backward(), usually only takes a scalar but with this it doesn't need  to

        # Func to optimise

        def mmf_func(beta, x):
            return (beta[..., 0] * beta[..., 1] + beta[..., 2] * x ** beta[..., 3]) / (beta[..., 1] + x ** beta[..., 3])

        optimizer = torch.optim.Adam([params], lr=0.1)  # SGD did weird things, Adam works well!
        y = data
        nan = torch.isnan(y)
        y = torch.where(nan, torch.tensor(0.0), y)

        for i in range(steps):
            optimizer.zero_grad()
            # Sum over last dim, watch out with nan's maybe?
            out = mmf_func(params, train_anchors)
            out = torch.where(nan, torch.tensor(0.0), out)
            loss = (((out - y) ** 2) * weights).sum(2)
            # Each curve has separate .backward() with this
            loss.backward(gradient=gr)
            optimizer.step()

        params.requires_grad = False  # To not waste computation
        target_points = torch.tensor(self.target_anchors, dtype=torch.float32)[:, None]
        prediction = mmf_func(params, target_points)

        # Remove predictions for anchor points that are in front of the window
        ind = torch.triu_indices(len(target_points), len(target_points), 1)
        prediction[:, :, ind[0], ind[1]] = torch.nan
        prediction = torch.einsum('ijkl->ijlk', prediction)
        self.results['MMF']['test error'] = (prediction - target[:, :, None]).abs().numpy()
        self.results['MMF']['regression'] = prediction.numpy()
        self.results['MMF']['parameters'] = params.numpy()

    def Last(self):

        prediction = np.diagonal(self.data, axis1=-2, axis2=-1)
        error = np.abs(prediction[..., None] - self.target[..., None, :])

        # Remove predictions for anchor points that are in front of the window
        ind = np.triu_indices(len(self.target_anchors), 1)
        error[:, :, ind[1], ind[0]] = np.nan

        self.results['Last']['test error'] = error
        self.results['Last']['regression'] = np.repeat(prediction[..., None], len(self.target_anchors), axis=-1)

    def MDS_low_budget_binary(self, k=4, deprecated_weights=False):
        """
        Binary MDS takes too much memory, we thus make a lower budget version which is slower but uses less memory
        """
        data = np.copy(self.data)
        data[..., self.triu[1], self.triu[0]] = 0

        if deprecated_weights:
            weights = np.arange(1, self.s + 1) ** 2
        else:
            weights = (2 ** np.arange(self.s))[:, None]

        # We have the following shape [new dataset, meta dataset, learner, (point on curve),... extra dims]
        scalar = np.sum(weights * data[None] * data[:, None], axis=3) / np.sum(weights * (data ** 2), axis=2)[None]
        adapted_target = scalar[..., None] * self.target[..., None, :]
        distance = np.sum((data - data[:, None]) ** 2, axis=3)

        # extra dim for moving target
        distance = np.repeat(distance[..., None], len(self.target_anchors), axis=-1)

        # Remove curves that can't predict at target
        ind = np.isnan(self.target).nonzero()
        distance[:, ind[0], ind[1], ..., ind[2]] = np.nan

        # So that it doesn't pick itself
        np.einsum('ii...->i...', distance)[...] = np.nan

        # Take k closest, for loop to save memory
        prediction_binary = []  # init

        # We loop over meta dataset
        for i in range(distance.shape[0]):
            distance_temp = distance[i]
            distance_sum = distance_temp[:, None] + distance_temp[:, :, None]

            # We need an extra dimension for the rival learner
            adapted_target_temp = adapted_target[i]
            adapted_target_temp = np.repeat(adapted_target_temp[:, :, None], adapted_target_temp.shape[1], axis=2)
            part_binary = np.argpartition(distance_sum, k, axis=0)[:k]
            k_closest_curves_binary = np.take_along_axis(adapted_target_temp, part_binary, axis=0)

            # For binary we need the prediction of both learners then we check which one is larger
            # For rival learner we can just swap the learner axis
            k_closest_curves_binary_rival = np.swapaxes(k_closest_curves_binary, 1, 2)

            prediction_l1 = np.mean(k_closest_curves_binary, axis=0)
            prediction_l2 = np.mean(k_closest_curves_binary_rival, axis=0)

            # Check when l1 wins over l2
            prediction_binary_temp = (prediction_l1 > prediction_l2).astype(float)

            # Check for ties
            prediction_binary_temp[prediction_l1 == prediction_l2] = 0.5

            # Return nans as they get removed
            prediction_binary_temp[np.isnan(prediction_l1)] = np.nan
            prediction_binary_temp[np.isnan(prediction_l2)] = np.nan

            # Don't compete against same learner
            np.einsum('jj...->j...', prediction_binary_temp)[...] = np.nan

            prediction_binary.append(prediction_binary_temp)

        prediction_binary = np.array(prediction_binary)
        # Remove predictions for anchor points that are in front of the window
        ind = np.triu_indices(len(self.target_anchors), 1)
        prediction_binary[..., ind[1], ind[0]] = np.nan

        self.results['MDS']['binary'] = prediction_binary

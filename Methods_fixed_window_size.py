import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from Preprocessing import *


class Windowing:
    Slearners = np.array(
        ['SVCl', 'SVCp', 'SVCr', 'SVCs', 'xTrs', 'GrBo', 'rFor', 'LogR', 'PaAg', 'Perc', 'Ridg', 'SGD', 'Bern', 'MuNo',
         'KNei', 'MLP', 'dTre', 'xTre', 'linD', 'QuaD']).astype(str)

    def __init__(self, agg_dataframe):
        self.orig_dataframe = agg_dataframe
        self.results = {}

        # To pick what anchor points we want to analyse the distribution of data on each anchor point.
        train_size_count = \
            self.orig_dataframe.groupby(['openmlid', 'size_train']).first().reset_index().groupby(['size_train'])[
                ['size_train']].count().rename(columns={'size_train': 'Count'})
        display(train_size_count)
        self.train_size = train_size_count.index.values
        self.train_size_count = train_size_count.values.reshape(-1)

        # Turn dataframe into numpy
        self.learners = agg_dataframe['learner'].unique()
        self.datasets = agg_dataframe['openmlid'].unique()
        self.points = np.sort(agg_dataframe['size_train'].unique())

        name = ['openmlid', 'learner', 'size_train']
        multind = pd.MultiIndex.from_product([self.datasets, self.learners, self.points])

        _ = agg_dataframe.set_index(name)
        _ = _.reindex(multind)

        # shape is [dataset,learner,point]
        self.dataframe = _['score_test'].values.reshape(len(self.datasets), len(self.learners), len(self.points))

    def set_points(self, anchor_points, window_size=5):
        self.window_size = window_size
        self.anchor_points = np.array([anchor_points]).reshape(-1)  # To get right shape
        self.train_anchors = np.copy(sliding_window_view(self.anchor_points[:-1], window_size))
        self.train_anchors = np.einsum('ij->ji', self.train_anchors)
        self.target_anchors = self.anchor_points[window_size:]
        self.s = len(self.anchor_points)
        self.results = {}

        # Grab the indices for numpy array
        indices = np.where(np.in1d(self.points, self.anchor_points))[0]

        # Grab the segment values
        self.segment = self.dataframe[..., indices]

        # Split all possible windows and target points
        # Use the sliding_window_view() function to create a view of the array with sliding windows
        self.target = self.segment[:, :, window_size:]
        self.data = np.copy(sliding_window_view(self.segment[:, :, :-1], window_size, axis=-1))
        self.data = np.einsum('ijkl -> ijlk', self.data)

        # method 2
        endpoints = np.arange(window_size - 1, len(self.anchor_points) - 1)
        ind = np.ceil(np.linspace(0, endpoints, self.window_size)).astype(int)

        self.train_anchors2 = self.anchor_points[ind]
        self.data2 = self.segment[:, :, ind]

    def MDS(self, k=4):
        '''
        Applies MDS algorithm on all windows to retrieve the regression predictions.
        Saves the absolute error in the prediction. Also saves MDS version where you
        scale /adapt curves before picking them. Original version scales after picking

        Uses method 1: Naively pick last in window
        '''
        data = np.copy(self.data)
        weights = (2 ** np.arange(self.window_size))[:, None]

        # We have the following shape [new dataset, meta dataset, learner, (point on curve),... extra dims]scalar = np.sum(weights * data[None]*data[:,None], axis = 3) / np.sum(weights*(data**2), axis = 2)[None]
        scalar = np.sum(weights * data[None] * data[:, None], axis=3) / np.sum(weights * (data ** 2), axis=2)[None]
        adapted_curves = scalar[:, :, :, None] * data
        adapted_target = scalar[..., None] * self.target[..., None, :]
        distance = np.sum((data - data[:, None]) ** 2, axis=3)
        distance_adapted = np.sum((adapted_curves - data) ** 2, axis=3)

        # Expand for next step
        distance = np.repeat(distance[..., None], self.target.shape[-1], axis=-1)
        distance_adapted = np.repeat(distance_adapted[..., None], self.target.shape[-1], axis=-1)

        # Remove curves that can't predict at target
        ind = np.isnan(self.target).nonzero()
        distance[:, ind[0], ind[1], ..., ind[2]] = np.nan
        distance_adapted[:, ind[0], ind[1], ..., ind[2]] = np.nan

        # So that it doesn't pick itself
        np.einsum('ii...->i...', distance)[...] = np.nan
        np.einsum('ii...->i...', distance_adapted)[...] = np.nan

        # Take k closest
        part_scale_after = np.argpartition(distance, k, axis=1)[:, :k]
        part_scale_before = np.argpartition(distance_adapted, k, axis=1)[:, :k, ]
        k_closest_curves_scale_after = np.take_along_axis(adapted_target, part_scale_after, axis=1)
        k_closest_curves_scale_before = np.take_along_axis(adapted_target, part_scale_before, axis=1)

        # Predicted target is just the mean of the k closest curves at the target point
        prediction_scale_after = np.mean(k_closest_curves_scale_after, axis=1)
        prediction_scale_before = np.mean(k_closest_curves_scale_before, axis=1)

        # Remove predictions for anchor points that are in front of the window
        ind = np.triu_indices(len(self.target_anchors), 1)
        prediction_scale_after[:, :, ind[1], ind[0]] = np.nan
        prediction_scale_before[:, :, ind[1], ind[0]] = np.nan

        self.results['MDS'] = {}
        self.results['MDS']['test error'] = np.abs(prediction_scale_after - self.target[:, :, None])
        self.results['MDS']['(scale before) test error'] = np.abs(prediction_scale_before - self.target[:, :, None])
        self.results['MDS']['regression'] = prediction_scale_after
        self.results['MDS']['(scale before) regression'] = prediction_scale_before
        self.results['MDS']['k closest curves ID'] = part_scale_after
        self.results['MDS']['(scale before) k closest curves ID'] = part_scale_before
        self.results['MDS']['scalar'] = scalar

    def MMF(self, steps=500):
        """
        Applies MMF algorithm. Tensorized with pytorch so that it calculates quickly

        Uses method 2: Pick uniformly in window
        """

        # Tensorise variables
        target = torch.tensor(self.target, dtype=torch.float32)
        data = torch.tensor(self.data2, dtype=torch.float32)
        train_anchors = torch.tensor(self.train_anchors2, dtype=torch.float32)
        weights = (2 ** torch.arange(self.window_size))[:, None]
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
            # Sum over last dimension
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

        self.results['MMF'] = {}
        self.results['MMF']['test error'] = (prediction - target[:, :, None]).abs().numpy()
        self.results['MMF']['parameters'] = params.numpy()

    def Last(self):
        """
        Uses the last value of the curve as a prediction
        """
        prediction = self.data[..., -1, :]

        error = np.abs(prediction[..., None] - self.target[..., None, :])

        # Remove predictions for anchor points that are in front of the window
        ind = np.triu_indices(len(self.target_anchors), 1)
        error[:, :, ind[1], ind[0]] = np.nan

        self.results['Last'] = {}
        self.results['Last']['test error'] = error

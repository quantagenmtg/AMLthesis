import pandas as pd
import torch
from matplotlib import pyplot as plt
from Plotting import Boxplots

from Preprocessing import *

"""
NOTE: This class only works with windows = TRUE
"""


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

    def set_points(self, window_points, target_points, windows=True):

        self.windows = windows
        self.window_points = np.array([window_points]).reshape(-1)  # To get right shape
        self.target_points = np.array([target_points]).reshape(-1)
        self.anchor_points = np.hstack((window_points, target_points[:-1]))
        self.s = len(self.anchor_points)
        self.results = {}

        # Grab the indices for numpy array
        indices_window = np.where(np.in1d(self.points, self.window_points))[0]
        indices_target = np.where(np.in1d(self.points, self.target_points))[0]

        # Grab the segment and the target values
        self.segment = self.dataframe[..., np.hstack((indices_window, indices_target[:-1]))]
        self.target = self.dataframe[..., indices_target]

        if self.windows:
            # Make all the windows, another dimension is added at the end for this [..., window]
            self.data = np.repeat(self.segment[..., None], self.segment.shape[2], -1)
            self.triu = np.triu_indices(self.segment.shape[2], 1)
            self.data[..., self.triu[1], self.triu[0]] = np.nan
        else:
            self.data = np.repeat(self.segment[..., None], self.target.shape[2], -1)
            self.triu = np.triu_indices(self.target.shape[2], self.segment.shape[2] - self.target.shape[2] + 1,
                                        self.segment.shape[2])
            self.data[..., self.triu[1], self.triu[0]] = np.nan

    def MDS(self, k=4):
        '''
        Applies MDS algorithm on all windows to retrieve the regression predictions.
        Saves the absolute error in the prediction. Also saves MDS version where you
        scale /adapt curves before picking them. Original version scales after picking
        '''
        data = np.copy(self.data)
        data[..., self.triu[1], self.triu[0]] = 0
        # weights = np.arange(1, self.s + 1) ** 2
        weights = (2 ** np.arange(self.s))[:, None]

        # We have the following shape [new dataset, meta dataset, learner, (point on curve),... extra dims]
        scalar = np.sum(weights * data[None] * data[:, None], axis=3) / np.sum(weights * (data ** 2), axis=2)[None]
        adapted_curves = scalar[:, :, :, None] * data
        adapted_target = scalar[..., None] * self.target[..., None, :]
        distance = np.sum((data - data[:, None]) ** 2, axis=3)
        distance_adapted = np.sum((adapted_curves - data) ** 2, axis=3)

        distance = np.repeat(distance[..., None], len(self.target_points), axis=-1)
        distance_adapted = np.repeat(distance_adapted[..., None], len(self.target_points), axis=-1)

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

        target = np.copy(self.target)
        if self.windows:
            target = target[:, :, None]

        self.results['MDS'] = {}
        self.results['MDS']['test error'] = np.abs(prediction_scale_after - target)
        self.results['MDS']['(scale before) test error'] = np.abs(prediction_scale_before - target)
        self.results['MDS']['regression'] = prediction_scale_after
        self.results['MDS']['(scale before) regression'] = prediction_scale_before
        self.results['MDS']['k closest curves ID'] = part_scale_after
        self.results['MDS']['(scale before) k closest curves ID'] = part_scale_before
        self.results['MDS']['scalar'] = scalar

    def MMF(self, steps=500):
        '''
        Applies MMF algorithm. Tensorized with pytorch so that it calculates quickly
        '''

        # Tensorise variables
        target = torch.tensor(self.target)
        data = torch.tensor(self.data)
        anchor_points = torch.tensor(self.anchor_points)
        weights = (2 ** torch.arange(self.s))[:, None]
        params = torch.tensor([0.5, 1, 1, -1])
        shp = np.hstack((np.array(data.shape[:2]), np.array(data.shape[3])))

        params = params.repeat(*shp, 1)[:, :, None]  # since len(dim[2]) = len(dim[3])

        params.requires_grad_()
        gr = torch.ones(*shp)  # For .backward(), usually only takes a scalar but with this it doesn't need  to

        # Func to optimise

        def mmf_func(beta, x):
            x = x[:, None]
            return (beta[..., 0] * beta[..., 1] + beta[..., 2] * x ** beta[..., 3]) / (beta[..., 1] + x ** beta[..., 3])

        optimizer = torch.optim.Adam([params], lr=0.1)  # SGD did weird things, Adam works well!
        y = data
        nan = torch.isnan(y)
        y = torch.where(nan, torch.tensor(0.0), y)

        for i in range(steps):
            optimizer.zero_grad()
            # Sum over last dim, watch out with nan's maybe?
            out = mmf_func(params, anchor_points)
            out = torch.where(nan, torch.tensor(0.0), out)
            loss = (((out - y) ** 2) * weights).sum(2)
            # Each curve has separate .backward() with this
            loss.backward(gradient=gr)
            optimizer.step()

        params.requires_grad = False  # To not waste computation
        target_points = torch.tensor(self.target_points)
        prediction = mmf_func(params, target_points)

        if self.windows:
            ind = torch.triu_indices(len(target_points), len(target_points), 1)
            prediction[:, :, ind[0], ind[1]] = torch.nan
            prediction = torch.einsum('ijkl->ijlk', prediction)
            target = target[:, :, None]

        self.results['MMF'] = {}
        self.results['MMF']['test error'] = (target - prediction).abs().numpy()
        self.results['MMF']['parameters'] = params.numpy()

    def Last(self):

        target = np.copy(self.target)
        if self.windows:
            prediction = self.data[:, :, np.arange(self.s), np.arange(self.s)]
            target = target[:, :, None]
        else:
            prediction = self.data[:, :, -1]

        self.results['Last'] = {}
        self.results['Last']['test error'] = np.abs(prediction[..., None] - target)

    def _returnIDs(self, dataset=slice(None), learner=slice(None), window=slice(None), target=slice(None)):

        if type(window) == str:
            windowID = np.where(self.anchor_points.astype(str) == window)[0][0]
        else:
            windowID = window

        if type(learner) == str:
            learnerID = np.where(self.Slearners == learner)[0][0]
        else:
            learnerID = learner

        if type(dataset) == str:
            datasetID = np.where(self.datasets.astype(str) == dataset)[0][0]
        else:
            datasetID = dataset

        if type(target) == str:
            targetID = np.where(self.target_points.astype(str) == target)[0][0]
        else:
            targetID = target

        return datasetID, learnerID, windowID, targetID

    def PlotCurveAndPrediction(self, dataset, learner, window, target, scaled=True, ylabel=None, xlabel=None,
                               title=None):
        '''
        Given the learner, dataset and window this will return a scatter plot of the
        actual curve, the MMF predicted curve, the k-nearest curves (scaled) and the
        MDS prediction. It even plots the missing anchor points that are not included
        when tensorizing the calculations.
        '''
        fig, ax = plt.subplots(1, 1, figsize=[6.4, 4.8])
        datasetID, learnerID, windowID, targetID = self._returnIDs(dataset, learner, window, target)

        if title:
            fig.suptitle(title, y=0.9)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)

            # Curve points, we also take the regularised points and put them back in to see
        ax.scatter(self.anchor_points[:windowID + 1], self.segment[datasetID, learnerID][:windowID + 1], color='blue',
                   marker='o')
        ax.scatter(self.target_points[targetID], self.target[datasetID, learnerID, targetID], color='red')

        stopID = np.where(self.points == self.target_points[targetID])[0][0]
        ax.scatter(self.points[:stopID], self.dataframe[datasetID, learnerID, :stopID], color='blue', marker='x')

        # MMF prediction plot
        x = np.linspace(self.anchor_points[0], self.target_points[targetID], 10000)
        a, b, c, d = self.results['MMF']['parameters'][datasetID, learnerID, 0, windowID]
        fun = lambda x: (a * b + c * x ** d) / (b + x ** d)
        ax.plot(x, fun(x))

        # MDS regression k-nearest neighbors, also put the regularised points back in
        knn = self.results['MDS']['k closest curves ID'][datasetID, :, learnerID, windowID, targetID]

        for k in knn:
            c = self.results['MDS']['scalar'][datasetID, k, learnerID, windowID]
            ax.scatter(self.points[:stopID + 1], c * self.dataframe[k, learnerID, :stopID + 1], s=5)

        # MDS regression prediction
        ax.scatter(self.target_points[targetID],
                   self.results['MDS']['regression'][datasetID, learnerID, windowID, targetID], color='red', marker='x')

        # Last One prediction
        ax.plot(x, np.full(x.shape, self.segment[datasetID, learnerID, windowID]), color='green')

    def Boxplots(self, dataset=slice(None), learner=slice(None), window=slice(None), target=slice(None), xaxis='window',
                 ylim=(0, 1), figsize=(3, 1), ylabel='Absolute Test Error', title=None):
        datasetID, learnerID, windowID, targetID = self._returnIDs(dataset, learner, window, target)
        plots = {}
        if xaxis == 'dataset':
            ind = np.arange(len(self.datasets))[datasetID]
            for key in self.results.keys():
                plots[key] = {str(self.datasets[i]): self.results[key]['test error'][i, learnerID, windowID, targetID]
                              for i in ind}
            xlabel = 'ID of dataset'

        if xaxis == 'learner':
            ind = np.arange(len(self.Slearners))[learnerID]
            for key in self.results.keys():
                plots[key] = {self.Slearners[i]: self.results[key]['test error'][datasetID, i, windowID, targetID] for i
                              in ind}
            xlabel = 'Abbreviated name of learner'

        if xaxis == 'window':
            ind = np.arange(len(self.anchor_points))[windowID]
            for key in self.results.keys():
                plots[key] = {
                    str(self.anchor_points[i]): self.results[key]['test error'][datasetID, learnerID, i, targetID] for i
                    in ind}
            xlabel = 'Value of last anchor point in the window'

        if xaxis == 'target':
            ind = np.arange(len(self.target_points))[targetID]
            for key in self.results.keys():
                plots[key] = {
                    str(self.target_points[i]): self.results[key]['test error'][datasetID, learnerID, windowID, i] for i
                    in ind}
            xlabel = 'Value of anchor point at which the prediction hapens'

        if not title:
            title = f'Test error per {xaxis}'
            sup = ' for '
            for val, name in [(dataset, 'dataset(s)'), (learner, 'learner(s)'), (window, 'last point in window(s)'),
                              (target, 'target(s)')]:
                if val == slice(None):
                    continue
                if name == 'dataset(s)':
                    val = self.datasets[datasetID]
                if name == 'learner(s)':
                    val = self.Slearners[learnerID]
                if name == 'last point in window(s)':
                    val = self.anchor_points[windowID]
                if name == 'target(s)':
                    val = self.target_points[targetID]
                sup += f'{name}: {val}, '
            sup = sup[0:-2]
            if sup != ' fo': title += sup
        Boxplots(plots, ylabel=ylabel, xlabel=xlabel, ylim=ylim, figsize=figsize, title=title)
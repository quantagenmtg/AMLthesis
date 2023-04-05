import pandas as pd
import torch
from matplotlib import pyplot as plt
from HelperFiles.Plotting import Boxplots

from HelperFiles.Preprocessing import *
from Methods import Methods



class Windowing(Methods):
    """
    Main Windwing class. This is used to run the experiments. The methods are given in the Method file.

    Included in this class are some rudimentary plotting functions to showcase the results.

    Use help function to get more detail.
    """

    Slearners = np.array(
        ['SVCl', 'SVCp', 'SVCr', 'SVCs', 'xTrs', 'GrBo', 'rFor', 'LogR', 'PaAg', 'Perc', 'Ridg', 'SGD', 'Bern', 'MuNo',
         'KNei', 'MLP', 'dTre', 'xTre', 'linD', 'QuaD']).astype(str)

    def __init__(self, agg_dataframe):
        self.orig_dataframe = agg_dataframe

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

    def set_points(self, anchor_points):

        self.anchor_points = np.array([anchor_points]).reshape(-1)  # To get right shape
        self.train_anchors = self.anchor_points[:-1]
        self.target_anchors = self.anchor_points[1:]
        self.s = len(self.train_anchors)

        # Grab the indices for numpy array
        indices = np.where(np.in1d(self.points, self.anchor_points))[0]

        # Grab the segment values
        self.segment = self.dataframe[..., indices]

        # Split all possible windows and target points
        # Use the sliding_window_view() function to create a view of the array with sliding windows
        self.train = self.segment[:, :, :-1]
        self.target = self.segment[:, :, 1:]

        # Make all the windows, another dimension is added at the end for this [..., window]
        self.data = np.repeat(self.train[..., None], self.train.shape[2], -1)
        self.triu = np.triu_indices(self.train.shape[2], 1)
        self.data[..., self.triu[1], self.triu[0]] = np.nan

    def _returnIDs(self, dataset=slice(None), learner=slice(None), window=slice(None), target=slice(None)):

        if type(window) == str:
            windowID = np.where(self.train_anchors.astype(str) == window)[0][0]
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
            targetID = np.where(self.target_anchors.astype(str) == target)[0][0]
        else:
            targetID = target

        return datasetID, learnerID, windowID, targetID

    def PlotCurveAndPrediction(self, dataset, learner, window, target, scaled=True, legend=True):
        '''
        Given the learner, dataset and window this will return a scatter plot of the
        actual curve, the MMF predicted curve, the k-nearest curves (scaled) and the
        MDS prediction. It even plots the missing anchor points that are not included
        when tensorizing the calculations.
        '''
        fig, ax = plt.subplots(1, 1, figsize=[6.4, 4.8])
        datasetID, learnerID, windowID, targetID = self._returnIDs(dataset, learner, window, target)

        # Curve points, we also take the regularised points and put them back in to see
        ax.scatter(self.train_anchors[:windowID + 1], self.segment[datasetID, learnerID][:windowID + 1], color='blue',
                   marker='o', label='Curve points (given)')
        ax.scatter(self.target_anchors[targetID], self.target[datasetID, learnerID, targetID], color='red', label='Target point')

        if np.isnan(self.target[datasetID, learnerID, targetID]):
            warnings.warn("Given curve has no value at target point. Predictions can still be made and shown "
                          "but are not used in the error calculations.", UserWarning)

        stopID = np.where(self.points == self.target_anchors[targetID])[0][0]
        ax.scatter(self.points[:stopID], self.dataframe[datasetID, learnerID, :stopID], color='blue', marker='x',
                   label='Curve points')

        # MMF prediction plot
        x = np.linspace(self.train_anchors[0], self.target_anchors[targetID], 10000)
        a, b, c, d = self.results['MMF']['parameters'][datasetID, learnerID, 0, windowID]
        fun = lambda x: (a * b + c * x ** d) / (b + x ** d)
        ax.plot(x, fun(x), label='MMF prediction')

        # MDS regression k-nearest neighbors, also put the regularised points back in
        knn = self.results['MDS']['k closest curves ID'][datasetID, :, learnerID, windowID, targetID]

        for k in knn:
            if scaled:
                c = self.results['MDS']['scalar'][datasetID, k, learnerID, windowID]
                ax.scatter(self.points[:stopID + 1], c * self.dataframe[k, learnerID, :stopID + 1], s=5)
            else:
                ax.scatter(self.points[:stopID + 1], self.dataframe[k, learnerID, :stopID + 1], s=5)

        # MDS regression prediction
        ax.scatter(self.target_anchors[targetID],
                   self.results['MDS']['regression'][datasetID, learnerID, windowID, targetID], color='red', marker='x',
                   label='MDS prediction')

        # Last One prediction
        ax.plot(x, np.full(x.shape, self.segment[datasetID, learnerID, windowID]), color='green',
                label='Last one prediction')

        # set y axis and x axis name
        ax.set_ylabel('Metric value')
        ax.set_xlabel('Train set size')

        # make a title
        ax.set_title('Dataset: {}, Learner: {}, Last anchor in window: {}, Target: {}'.format(self.datasets[datasetID],
                                                                                         self.Slearners[learnerID],
                                                                                              self.train_anchors[
                                                                                                  windowID],
                                                                                              self.target_anchors[
                                                                                                  targetID]))

        if legend:
            ax.legend()

    def Boxplots(self, dataset=slice(None), learner=slice(None), window=slice(None), target=slice(None), xaxis='window',
                 ylim=(0, 1), figsize=(3, 1), ylabel='Error of Extrapolation', title=None, save=None, dpi = 800):
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
            ind = np.arange(len(self.train_anchors))[windowID]
            for key in self.results.keys():
                plots[key] = {
                    str(self.train_anchors[i]): self.results[key]['test error'][datasetID, learnerID, i, targetID] for i
                    in ind}
            xlabel = 'Value of last anchor point in the window'

        if xaxis == 'target':
            ind = np.arange(len(self.target_anchors))[targetID]
            for key in self.results.keys():
                plots[key] = {
                    str(self.target_anchors[i]): self.results[key]['test error'][datasetID, learnerID, windowID, i] for
                    i
                    in ind}
            xlabel = 'Value of anchor point at which the prediction hapens'

        if not title:
            title = f'Error of extrapolation per {xaxis}'
            sup = ' for '
            for val, name in [(dataset, 'dataset(s)'), (learner, 'learner(s)'), (window, 'last point in window(s)'),
                              (target, 'target anchor point')]:
                if val == slice(None):
                    continue
                if name == 'dataset(s)':
                    val = self.datasets[datasetID]
                if name == 'learner(s)':
                    val = self.Slearners[learnerID]
                if name == 'last point in window(s)':
                    val = self.train_anchors[windowID]
                if name == 'target anchor point':
                    val = self.target_anchors[targetID]
                sup += f'{name}: {val}, '
            sup = sup[0:-2]
            if sup != ' fo': title += sup
        Boxplots(plots, ylabel=ylabel, xlabel=xlabel, ylim=ylim, figsize=figsize, title=title, save=save, dpi=dpi)

    def help(self):
        print("This class is used to analyse the results of different learning curve regression methods.")
        print("The results are stored in a dictionary called 'results'. To access them use the following syntax: "
              "self.results['method']['test error'][datasetID, learnerID, windowID, targetID]")
        print("To plot the results of a specific method, use the following functions: \n")
        print("PlotCurveAndPrediction: plots the predictions and true values for a specific dataset, learner, "
              "window and target. This gives a zoomed in view of what is happening.\n")
        print("Boxplots: Given a specific slice of the dataset, learner, window and target, this function makes "
              "the corresponding boxplots. You can specify what you want to plot the boxplots over, by setting "
              "the xaxis parameter to 'dataset', 'learner', 'window' or 'target'.\n")

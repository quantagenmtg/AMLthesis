import lcdb
import numpy as np

# Trying out lcdb
curves = lcdb.get_all_curves()
aggCurves = curves.groupby(['openmlid', 'learner', 'size_train']).mean().reset_index()

"""
# Checking if there is consistent train_sizes
last = aggCurves.groupby(['openmlid', 'learner']).last().reset_index()
unique_train_size_learners = aggCurves.groupby(['openmlid', 'learner']).count().reset_index().groupby(['openmlid','size_train']).first().reset_index()
consistentIndex = unique_train_size_learners.groupby(['openmlid']).tail(1).index
inconsistent = unique_train_size_learners.drop(consistentIndex)

# For each dataset each learner has the same final size_train, sometimes up to 1 learner doesn't have some of the earlier size_train's
display(inconsistent)
"""

# We drop inconsistent train_size learners in their respective dataset where they are inconsistent
# A smarter regularization could be done but this is simpler for now
aggCurves['count'] = aggCurves.groupby(['openmlid', 'learner'])['size_train'].transform('count')
normCount = aggCurves.groupby(['openmlid'])['count'].transform(lambda x: x / np.max(x))
ind = np.array(normCount[normCount != 1].index)
aggCurves = aggCurves.drop(ind)

# This will show how many datasets have the given train_size (so not per learner, as every learner in the dataset has
# the same train_sizes. We thus check how many db have the train_size)
train_size_count = aggCurves.groupby(['openmlid', 'size_train']).first().reset_index().groupby(['size_train'])[
    ['size_train']].count().rename(columns={'size_train': 'Count'})
learners = aggCurves['learner'].unique()
for i, learner in enumerate(learners):
    if '.' in learner:
        learners[i] = learner[learner.rfind('.') + 1:]
learners = list(learners)

Slearners = ['SVCl', 'SVCp', 'SVCr', 'SVCs', 'xTrs', 'GrBo', 'rFor', 'LogR', 'PaAg', 'Perc', 'Ridg', 'SGD', 'Bern',
             'MuNo', 'KNei', 'MLP', 'dTre', 'xTre', 'linD', 'QuaD']
# AMLthesis
Thesis project in automated machine learning. The project is about comparing various extrapolation techniques employed for the extrapolation of learning curves and comparing them in different situations.
This should provide an understanding of which extrapolation technique is best suited for which situation.
To this extent we use the best parametric model: MMF and the best meta-learning model: MDS.
MMF has been shown to be the best parametric model for extrapolation by Mohr et al. in [LCDB 1.0: An Extensive Learning Curves
Database for Classification Tasks](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_1317.pdf).
MDS is the only known extrapolation technique that employs meta-learning, it was introduced by Leite
and Brazdil in [Predicting relative performance of classifiers from samples](https://www.researchgate.net/publication/221346339_Predicting_relative_performance_of_classifiers_from_samples).

# TLDR for the paper
- [MainFiles/Methods.py](MainFiles/Methods.py) contains the code for applying the extrapolation techniques used in the paper to LCDB.
- [Plots.ipynb](Plots.ipynb) contains the code to make the plots in the paper.

# Files
- Experiments: Contains the jupyter notebooks where the experiments are run and plots are made
- HelperFiles: Contains files for plotting and preprocessing of Learning Curve Database (LCDB)
- MainFiles: Contains the classes that perform the extrapolation techniques
- Plots: Contains the plots that are made in the experiments

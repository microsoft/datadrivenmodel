from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model


from tune_sklearn import TuneSearchCV
from tune_sklearn import TuneGridSearchCV
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV

# param_grid = {"learning_rate": (0.01, 0.1), "n_estimators": (25, 250), "subsample": [False, True]}

pipe = Pipeline(
    [
        # the reduce_dim stage is populated by the param_grid
        ("reduce_dim", "passthrough"),
        ("classify", LinearSVC(dual=False, max_iter=10000)),
    ]
)

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10]
param_grid = [
    {
        "reduce_dim": [PCA(iterated_power=7), NMF()],
        "reduce_dim__n_components": N_FEATURES_OPTIONS,
        "classify__C": C_OPTIONS,
    },
    {
        "reduce_dim": [SelectKBest(chi2)],
        "reduce_dim__k": N_FEATURES_OPTIONS,
        "classify__C": C_OPTIONS,
    },
]

random = TuneSearchCV(pipe, param_grid, search_optimization="random")
X, y = load_digits(return_X_y=True)
random.fit(X, y)
print(random.cv_results_)

grid = TuneGridSearchCV(pipe, param_grid=param_grid)
grid.fit(X, y)
print(grid.cv_results_)


if __name__ == "__main__":

    """Example using an sklearn Pipeline with TuneGridSearchCV.

    Example taken and modified from
    https://scikit-learn.org/stable/auto_examples/compose/
    plot_compare_reduction.html
    """


    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import load_digits
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.decomposition import PCA, NMF
    from sklearn.feature_selection import SelectKBest, chi2

    from tune_sklearn import TuneSearchCV
    from tune_sklearn import TuneGridSearchCV

    pipe = Pipeline(
        [
            # the reduce_dim stage is populated by the param_grid
            ("reduce_dim", "passthrough"),
            ("classify", LinearSVC(dual=False, max_iter=10000)),
        ]
    )

    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10]
    param_grid = [
        {
            "reduce_dim": [PCA(iterated_power=7), NMF()],
            "reduce_dim__n_components": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        },
        {
            "reduce_dim": [SelectKBest(chi2)],
            "reduce_dim__k": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        },
    ]

    random = TuneSearchCV(pipe, param_grid, search_optimization="random")
    X, y = load_digits(return_X_y=True)
    random.fit(X, y)
    print(random.cv_results_)

    grid = TuneGridSearchCV(pipe, param_grid=param_grid)
    grid.fit(X, y)
    print(grid.cv_results_)

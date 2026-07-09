"""Train a single random forest over all (non-marine) environments.

Environment membership is supplied as one-hot input features; model + scalers are
pickled. The shared training routine lives in ``stratml.modeling.core.train_all``.
"""

from stratml import config
from stratml.modeling.core import train_all


def main():
    train_all(config.FEATURES_ALL, config.SAMPLE_N_ALL, 'all_data_model.pkl')


if __name__ == "__main__":
    main()

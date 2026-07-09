"""Train a single random forest over all (non-marine) environments, with the
``High_Erosion`` tag added as an input feature and a larger sample.

Model + scalers are pickled. The shared training routine lives in
``stratml.modeling.core.train_all``.
"""

from stratml import config
from stratml.modeling.core import train_all


def main():
    train_all(config.FEATURES_ALL_TAGGED, config.SAMPLE_N_TAGGED, 'all_data_tagged_model.pkl')


if __name__ == "__main__":
    main()

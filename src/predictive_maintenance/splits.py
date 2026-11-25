"""Reproducible train, validation, and test partitions."""

from sklearn.model_selection import train_test_split

SEED = 42


def split_dataset(features, target):
    develop_x, test_x, develop_y, test_y = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        develop_x, develop_y, test_size=0.2, stratify=develop_y, random_state=SEED
    )
    return train_x, validation_x, test_x, train_y, validation_y, test_y

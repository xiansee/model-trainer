import pytest
from pydantic import BaseModel, ValidationError

from model_trainer.core.hyperparam_tuning import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
)


def test_hyperparameter_base_class():
    """Test Hyperparameter base class."""

    class FooHyperparameter(Hyperparameter):
        foo_var: str

    with pytest.raises(ValidationError):
        # Missing required "name" argument
        FooHyperparameter(foo_var="")

    try:
        # Correct initialization
        FooHyperparameter(name="Foo", foo_var="")

    except ValidationError:
        pytest.fail(
            "Hyperparameter child class failed to instantiate with correct set of inputs."
        )


def test_integer_hyperparameter():
    """Test initialization of IntegerHyperparameter"""

    with pytest.raises(ValidationError):
        # Missing name argument
        IntegerHyperparameter(low=1, high=5)

        # Incorrect argument type
        IntegerHyperparameter(low=2.5, high=5)

        # Missing "high" argument
        IntegerHyperparameter(name="FooHyperparam", low=1)

    try:
        # Correct initialization
        IntegerHyperparameter(name="FooHyperparam", low=1, high=5)
        IntegerHyperparameter(name="FooHyperparam", low=1, high=5, log=False)

    except ValidationError:
        pytest.fail(
            "IntegerHyperparameter failed to instantiate a correct set of inputs."
        )


def test_float_hyperparameter():
    """Test initialization of FloatHyperparameter"""

    with pytest.raises(ValidationError):
        # Missing name argument
        FloatHyperparameter(low=1.2, high=5.5)

        # Incorrect argument type
        FloatHyperparameter(name="FooHyperparam", low=1.2, high=5)

        # Missing high argument
        FloatHyperparameter(name="FooHyperparam", low=1.2)

    try:
        # Correct initialization
        FloatHyperparameter(name="FooHyperparam", low=1.2, high=5.5)
        FloatHyperparameter(name="FooHyperparam", low=1.2, high=5.5, log=True)

    except ValidationError:
        pytest.fail(
            "IntegerHyperparameter failed to instantiate a correct set of inputs."
        )


def test_categorical_hyperparameter():
    """Test initialization of categorical hyperparameter."""

    with pytest.raises(ValidationError):
        # Missing name argument
        CategoricalHyperparameter(choices=[1, 2, 3])

    try:
        # Correct initializations
        CategoricalHyperparameter(name="FooHyperparam", choices=[1, 2, 3])
        CategoricalHyperparameter(name="FooHyperparam", choices=[1.3, 2.1, 3.7])
        CategoricalHyperparameter(name="FooHyperparam", choices=["a", "b", "c"])
        CategoricalHyperparameter(name="FooHyperparam", choices=[True, False])

    except ValidationError:
        pytest.fail(
            "CategoricalHyperparameter failed to instantiate with a correct set of inputs."
        )

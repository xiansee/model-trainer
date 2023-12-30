import pytest
from pydantic import ValidationError
from torch import optim

from model_trainer.core.optimizer import (
    OptimizerConfig,
    _OptimizerChoices,
    get_optimizer,
)


def test_get_optimizer():
    """Test get optimizer function."""

    with pytest.raises(ValueError):
        # Unsupported option
        get_optimizer(name="foo")

    # Verify all supported options are implemented
    for supported_option in _OptimizerChoices:
        try:
            get_optimizer(name=supported_option.value)
        except NotImplementedError:
            pytest.fail(
                f"{supported_option.value} optimizer provided as an option but not implemented."
            )

    assert get_optimizer(name="adam") == optim.Adam
    assert get_optimizer(name="ADAM") == optim.Adam
    assert get_optimizer(name="sgd") == optim.SGD


def test_optimizer_config():
    """Test initialization of OptimizerConfig."""
    with pytest.raises(ValidationError):
        # Missing required input "lr"
        OptimizerConfig(weight_decay=0.001)

    try:
        # Correct inputs
        OptimizerConfig(lr=0.1)

    except ValidationError:
        pytest.fail("OptimizerConfig failed to initialize with correct set of inputs.")

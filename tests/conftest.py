import warnings

from pydantic.warnings import PydanticDeprecatedSince20

warnings.filterwarnings(
    "ignore",
    category=PydanticDeprecatedSince20,
)
warnings.filterwarnings(
    "ignore",
    message=".*Downcasting object dtype arrays on .fillna.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*DataFrameGroupBy.apply operated on the grouping columns.*",
    category=FutureWarning,
)

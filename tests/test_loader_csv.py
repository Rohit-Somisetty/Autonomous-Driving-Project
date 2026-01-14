import pandas as pd
import pytest
from pandas.api import types as pdt

from av_eval.data.loaders import CsvFrameLoader, validate_schema
from av_eval.data.schema import REQUIRED_COLUMNS


CSV_CONTENT = """trip_id,timestamp_s,dt_s,speed_mps,accel_mps2,lane_offset_m,ttc,engaged,weather,time_of_day,traffic_density,date,driver
trip_a,0,0.5,12.0,0.1,0.05,6.0,1,sunny,day,low,2025-01-01,baseline
trip_a,0.5,0.5,14.0,0.2,0.04,4.0,True,sunny,day,low,2025-01-01,baseline
trip_b,0,1.0,8.0,-0.1,0.02,7.5,false,rain,night,high,2025-01-02,candidate
"""


@pytest.fixture()
def sample_csv(tmp_path):
    csv_path = tmp_path / "logs.csv"
    csv_path.write_text(CSV_CONTENT)
    return csv_path


def test_csv_loader_returns_canonical_schema(sample_csv):
    loader = CsvFrameLoader()
    df = loader.load(sample_csv)

    for column in REQUIRED_COLUMNS:
        assert column in df.columns

    assert pdt.is_bool_dtype(df["engaged"])  # booleans parsed
    assert pdt.is_datetime64_any_dtype(df["log_date"])  # dates parsed
    assert pdt.is_numeric_dtype(df["distance_m"])  # derived field added
    assert set(df["driver_version"].unique()) == {"baseline", "candidate"}


def test_validate_schema_complains_on_missing_columns():
    bad_df = pd.DataFrame({"trip_id": ["trip_a"]})
    with pytest.raises(ValueError):
        validate_schema(bad_df)

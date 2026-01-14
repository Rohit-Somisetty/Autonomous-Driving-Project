import pandas as pd

from av_eval.metrics.top_events import rank_top_events


def _event_row(event_name, trip_id, severity, duration):
    return {
        "event_name": event_name,
        "trip_id": trip_id,
        "start_time_s": 0.0,
        "end_time_s": duration,
        "duration_s": duration,
        "severity": severity,
    }


def test_rank_top_events_orders_by_custom_severity():
    frames = pd.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3"],
            "weather": ["sun", "rain", "fog"],
            "time_of_day": ["day", "night", "dusk"],
            "traffic_density": ["low", "med", "high"],
            "driver_version": ["baseline", "baseline", "candidate"],
        }
    )

    near_miss = pd.DataFrame(
        [
            _event_row("near_miss", "t1", 0.5, 1.0),  # severity score = 2.0
            _event_row("near_miss", "t2", 0.2, 1.0),  # severity score = 5.0 (highest)
        ]
    )
    hard_brake = pd.DataFrame(
        [
            _event_row("hard_braking", "t3", -6.0, 0.5),  # score = 6.0 second highest
        ]
    )
    event_tables = {
        "near_miss": near_miss,
        "hard_braking": hard_brake,
        "lane_deviation": pd.DataFrame(columns=near_miss.columns),
    }

    top = rank_top_events(frames, event_tables, top_n=3)
    assert list(top["trip_id"]) == ["t3", "t2", "t1"]
    assert list(top["event_name"]) == ["hard_braking", "near_miss", "near_miss"]
    assert top.loc[0, "severity"] == 6.0  # abs acceleration
    assert top.loc[1, "severity"] == 5.0  # 1 / ttc


def test_rank_top_events_omits_driver_when_missing():
    frames = pd.DataFrame(
        {
            "trip_id": ["t1"],
            "weather": ["sun"],
            "time_of_day": ["day"],
            "traffic_density": ["low"],
        }
    )
    table = pd.DataFrame(
        [
            _event_row("lane_deviation", "t1", 0.8, 2.0),
        ]
    )
    event_tables = {"lane_deviation": table}

    result = rank_top_events(frames, event_tables, top_n=1)
    assert "driver_version" not in result.columns
    assert result.loc[0, "severity"] == 0.8
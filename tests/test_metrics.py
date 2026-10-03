import pytest
from ai4sts2.metrics import compare_evaluations, summarise


def episode(floor, *, victory=False, truncated=False, character="IRONCLAD", act=0):
    return {
        "floor": floor,
        "victory": victory,
        "truncated": truncated,
        "character": character,
        "act": act,
    }


def test_progress_distinguishes_zero_win_candidates_and_keeps_wins_primary():
    shallow = summarise([episode(2)] * 5)
    deeper = summarise([episode(floor) for floor in (9, 4, 5, 6, 8)])
    winning = summarise([episode(1, victory=True)] + [episode(0)] * 4)
    extreme = summarise([episode(10**16)] * 5)
    assert shallow["selection_score"] < deeper["selection_score"] < winning["selection_score"]
    assert extreme["selection_score"] < winning["selection_score"]
    assert deeper["mean_floor"] == 6.4
    assert deeper["median_floor"] == 6
    assert deeper["mean_death_floor"] == 6.4


def test_truncation_has_no_death_floor_and_cannot_gain_selection_credit():
    result = summarise([episode(2), episode(100, truncated=True)])
    assert result["mean_floor"] == 51
    assert result["mean_death_floor"] == 2
    assert result["truncated_episodes"] == 1
    assert result["selection_score"] == -1
    assert not result["eligible"]


def test_character_and_act_statistics_do_not_hide_uneven_performance():
    result = summarise(
        [
            episode(3),
            episode(20, act=1),
            episode(50, act=2, character="SILENT", victory=True),
        ]
    )
    assert result["characters"]["IRONCLAD"]["win_rate"] == 0
    assert result["characters"]["SILENT"]["win_rate"] == 1
    assert result["characters"]["SILENT"]["mean_death_floor"] is None
    assert result["act_reach_rates"] == {"1": 1, "2": 2 / 3, "3": 1 / 3}


@pytest.mark.parametrize("floor", [None, -1, 1.5, float("nan"), True])
def test_missing_or_invalid_progress_is_not_silently_replaced(floor):
    with pytest.raises(ValueError, match="floor"):
        summarise([episode(floor)])


def test_comparison_requires_matching_cases_and_complete_episodes():
    candidate = {
        "evaluation_id": "shared",
        "eligible": True,
        "wins": 0,
        "episodes": [{"character": "IRONCLAD", "seed": "ABC", "floor": 4}],
    }
    baseline = candidate | {"episodes": [candidate["episodes"][0] | {"floor": 2}]}
    difference = compare_evaluations(candidate, baseline)
    assert difference["mean_floor_difference"] == 2
    assert difference["deeper_episodes"] == 1
    assert compare_evaluations(candidate | {"eligible": False}, baseline) == {"eligible": False}
    with pytest.raises(ValueError, match="limits"):
        compare_evaluations(candidate, baseline | {"evaluation_id": "other"})
    with pytest.raises(ValueError, match="episodes"):
        compare_evaluations(
            candidate, baseline | {"episodes": [{"character": "DEFECT", "seed": 1}]}
        )

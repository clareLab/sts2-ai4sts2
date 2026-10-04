import statistics


def progress(episodes):
    if not episodes:
        raise ValueError("Evaluation requires at least one episode.")
    scopes = {episode.get("scope", "run") for episode in episodes}
    if len(scopes) != 1:
        raise ValueError("Episode scopes do not match.")
    for episode in episodes:
        for field in ("floor", "act"):
            value = episode[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Invalid episode {field}.")
        if episode["victory"] and episode["truncated"]:
            raise ValueError("A truncated episode cannot be a victory.")
        success = episode.get("task_success", episode["victory"])
        if success and episode["truncated"]:
            raise ValueError("A truncated episode cannot be a task success.")
        if episode.get("scope") == "act1" and success != (episode["act"] >= 1):
            raise ValueError("Act 1 success requires entering Act 2.")
    floors = [episode["floor"] for episode in episodes]
    deaths = [
        e["floor"]
        for e in episodes
        if not e.get("task_success", e["victory"])
        and not e["truncated"]
        and not e.get("goal_success", False)
    ]
    wins = sum(episode["victory"] for episode in episodes)
    successes = sum(e.get("task_success", e["victory"]) for e in episodes)
    return {
        "scope": next(iter(scopes)),
        "episodes": len(episodes),
        "wins": wins,
        "win_rate": wins / len(episodes),
        "task_successes": successes,
        "task_success_rate": successes / len(episodes),
        "boss_reach_rate": sum(e.get("boss_reached", False) for e in episodes) / len(episodes),
        "truncated_episodes": sum(episode["truncated"] for episode in episodes),
        "curriculum_episodes": sum("goal_floor" in episode for episode in episodes),
        "curriculum_successes": sum(episode.get("goal_success", False) for episode in episodes),
        "mean_floor": statistics.mean(floors),
        "median_floor": statistics.median(floors),
        "min_floor": min(floors),
        "max_floor": max(floors),
        "mean_death_floor": statistics.mean(deaths) if deaths else None,
        "act_reach_rates": {
            str(act + 1): sum(e["act"] >= act for e in episodes) / len(episodes)
            for act in range(max(3, max(e["act"] for e in episodes) + 1))
        },
    }


def summarise(episodes):
    summary = progress(episodes)
    eligible = summary["truncated_episodes"] == summary["curriculum_episodes"] == 0
    return summary | {
        "eligible": eligible,
        "selection_score": summary["task_success_rate"] if eligible else -1.0,
        "characters": {
            character: progress([e for e in episodes if e["character"] == character])
            for character in sorted({e["character"] for e in episodes})
        },
    }


def compare_evaluations(candidate, baseline):
    if candidate["evaluation_id"] != baseline["evaluation_id"]:
        raise ValueError("Evaluation cases or limits do not match.")
    pairs = list(zip(candidate["episodes"], baseline["episodes"], strict=True))
    for current, reference in pairs:
        if (current["character"], current["seed"]) != (reference["character"], reference["seed"]):
            raise ValueError("Evaluation episodes do not match.")
    if not candidate["eligible"] or not baseline["eligible"]:
        return {"eligible": False}
    differences = [current["floor"] - reference["floor"] for current, reference in pairs]
    return {
        "eligible": True,
        "wins_difference": candidate["wins"] - baseline["wins"],
        "task_successes_difference": candidate.get("task_successes", candidate["wins"])
        - baseline.get("task_successes", baseline["wins"]),
        "mean_floor_difference": statistics.mean(differences),
        "deeper_episodes": sum(delta > 0 for delta in differences),
        "equal_floor_episodes": sum(delta == 0 for delta in differences),
        "shallower_episodes": sum(delta < 0 for delta in differences),
    }

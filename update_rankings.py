"""Automated College Football 2026 ranking update.

Cleaned from CFB_2026_Rankings notebook.
Set CFBD_API_KEY in the environment before running.
"""

import os
import json
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

SEASON = int(os.getenv("CFB_SEASON", "2026"))
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing CFBD_API_KEY environment variable")

def fetch_cfbd(endpoint, params):
    url = f"https://api.collegefootballdata.com/{endpoint}?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode())

# Pull FBS regular-season games
games = fetch_cfbd("games", {"year": SEASON, "classification": "fbs"})
df_games = pd.DataFrame(games)
if df_games.empty:
    raise SystemExit(f"No {SEASON} games available yet.")
df_games = df_games[df_games["seasonType"] == "regular"].copy()
# Rankings use completed games only; future scheduled games must not count as games played.
if "completed" in df_games.columns:
    df_games = df_games[df_games["completed"] == True].copy()
if df_games.empty:
    print(f"No completed {SEASON} regular-season FBS games available yet.")
    raise SystemExit(0)



# --- Notebook cell 6 ---
games_clean = df_games[
    [
        "id",
        "week",
        "seasonType",
        "neutralSite",
        "homeTeam",
        "homeConference",
        "homePoints",
        "awayTeam",
        "awayConference",
        "awayPoints"
    ]
].copy()

games_clean.head(10)



# --- Notebook cell 7 ---
home_games = games_clean.rename(columns={
    "id": "game_id",
    "homeTeam": "team",
    "homeConference": "conference",
    "homePoints": "points_for",
    "awayTeam": "opponent",
    "awayPoints": "points_against"
})[
    [
        "game_id",
        "week",
        "seasonType",
        "neutralSite",
        "team",
        "conference",
        "opponent",
        "points_for",
        "points_against"
    ]
].copy()

home_games["location"] = home_games["neutralSite"].map({
    True: "neutral",
    False: "home"
})


away_games = games_clean.rename(columns={
    "id": "game_id",
    "awayTeam": "team",
    "awayConference": "conference",
    "awayPoints": "points_for",
    "homeTeam": "opponent",
    "homePoints": "points_against"
})[
    [
        "game_id",
        "week",
        "seasonType",
        "neutralSite",
        "team",
        "conference",
        "opponent",
        "points_for",
        "points_against"
    ]
].copy()

away_games["location"] = away_games["neutralSite"].map({
    True: "neutral",
    False: "away"
})


team_games = pd.concat(
    [home_games, away_games],
    ignore_index=True
)

team_games = team_games.drop(
    columns=["neutralSite"]
)

team_games.head(10)



# --- Notebook cell 8 ---
team_games["margin"] = (
    team_games["points_for"] - team_games["points_against"]
)

team_games["win"] = (
    team_games["points_for"] > team_games["points_against"]
).astype(int)

team_games["loss"] = (
    team_games["points_for"] < team_games["points_against"]
).astype(int)

team_games.head(10)



# --- Notebook cell 9 ---
team_stats = team_games.groupby(
    ["team", "conference"]
).agg(
    games=("team", "count"),
    wins=("win", "sum"),
    losses=("loss", "sum"),
    points_for=("points_for", "sum"),
    points_against=("points_against", "sum"),
    avg_margin=("margin", "mean")
).reset_index()

team_stats["win_pct"] = (
    team_stats["wins"] / team_stats["games"]
)

team_stats["points_per_game"] = (
    team_stats["points_for"] / team_stats["games"]
)

team_stats["points_allowed_per_game"] = (
    team_stats["points_against"] / team_stats["games"]
)

team_stats.head(20)



# --- Notebook cell 10 ---
fbs_home = df_games.loc[
    df_games["homeClassification"] == "fbs",
    "homeTeam"
]

fbs_away = df_games.loc[
    df_games["awayClassification"] == "fbs",
    "awayTeam"
]

fbs_teams = set(
    pd.concat([fbs_home, fbs_away]).dropna().unique()
)

fbs_stats = team_stats[
    team_stats["team"].isin(fbs_teams)
].copy()

fbs_stats["record"] = (
    fbs_stats["wins"].astype(str)
    + "-"
    + fbs_stats["losses"].astype(str)
)

fbs_stats = fbs_stats.sort_values(
    ["win_pct", "avg_margin"],
    ascending=False
).reset_index(drop=True)

fbs_stats.head(25)



# --- Notebook cell 11 ---
win_pct_lookup = fbs_stats.set_index("team")["win_pct"].to_dict()

team_games["opponent_win_pct"] = team_games["opponent"].map(win_pct_lookup)

sos = (
    team_games[
        team_games["team"].isin(fbs_teams)
    ]
    .groupby("team")["opponent_win_pct"]
    .mean()
    .reset_index()
)

sos = sos.rename(columns={
    "opponent_win_pct": "sos"
})

fbs_stats = fbs_stats.merge(
    sos,
    on="team",
    how="left"
)

fbs_stats = fbs_stats.sort_values(
    "sos",
    ascending=False
).reset_index(drop=True)

fbs_stats[
    ["team", "conference", "record", "win_pct", "sos"]
].head(25)



# --- Notebook cell 12 ---
def min_max_score(series):
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=series.index)
    return 100 * ((series - lo) / (hi - lo))

fbs_stats["record_score"] = min_max_score(
    fbs_stats["win_pct"]
)

fbs_stats["sos_score"] = min_max_score(
    fbs_stats["sos"]
)

fbs_stats["margin_score"] = min_max_score(
    fbs_stats["avg_margin"]
)

fbs_stats[
    [
        "team",
        "record",
        "record_score",
        "sos_score",
        "margin_score"
    ]
].head(20)



# --- Notebook cell 13 ---
fbs_stats["rating"] = (
    0.40 * fbs_stats["record_score"]
    + 0.35 * fbs_stats["sos_score"]
    + 0.25 * fbs_stats["margin_score"]
)

rankings = fbs_stats.sort_values(
    "rating",
    ascending=False
).reset_index(drop=True)

rankings.index = rankings.index + 1
rankings.index.name = "rank"

rankings[
    [
        "team",
        "conference",
        "record",
        "rating",
        "record_score",
        "sos_score",
        "margin_score"
    ]
].head(25)



# --- Notebook cell 14 ---
team_games["opponent_final_win_pct"] = team_games["opponent"].map(win_pct_lookup)

team_games["quality_win"] = (
    (team_games["win"] == 1)
    & (team_games["opponent_final_win_pct"] >= 0.75)
).astype(int)

team_games["bad_loss"] = (
    (team_games["loss"] == 1)
    & (team_games["opponent_final_win_pct"] < 0.50)
).astype(int)

quality_metrics = (
    team_games[
        team_games["team"].isin(fbs_teams)
    ]
    .groupby("team")
    .agg(
        quality_wins=("quality_win", "sum"),
        bad_losses=("bad_loss", "sum")
    )
    .reset_index()
)

fbs_stats = fbs_stats.merge(
    quality_metrics,
    on="team",
    how="left"
)

fbs_stats[
    [
        "team",
        "record",
        "quality_wins",
        "bad_losses"
    ]
].sort_values(
    "quality_wins",
    ascending=False
).head(25)



# --- Notebook cell 15 ---
# Convert quality wins and bad losses to 0-100 scores

fbs_stats["quality_win_score"] = min_max_score(
    fbs_stats["quality_wins"]
)

fbs_stats["bad_loss_score"] = min_max_score(
    fbs_stats["bad_losses"]
)

# New rating formula
fbs_stats["rating_v2"] = (
    0.35 * fbs_stats["record_score"]
    + 0.25 * fbs_stats["sos_score"]
    + 0.20 * fbs_stats["margin_score"]
    + 0.15 * fbs_stats["quality_win_score"]
    - 0.05 * fbs_stats["bad_loss_score"]
)

rankings_v2 = fbs_stats.sort_values(
    "rating_v2",
    ascending=False
).reset_index(drop=True)

rankings_v2.index = rankings_v2.index + 1
rankings_v2.index.name = "rank"

rankings_v2[
    [
        "team",
        "conference",
        "record",
        "rating_v2",
        "quality_wins",
        "bad_losses",
        "sos_score",
        "margin_score"
    ]
].head(25)



# --- Notebook cell 16 ---
team_games["opponent_strength"] = team_games["opponent"].map(win_pct_lookup)

team_games["win_value"] = (
    team_games["win"] * team_games["opponent_strength"]
)

team_games["loss_penalty"] = (
    team_games["loss"] * (1 - team_games["opponent_strength"])
)

resume_metrics = (
    team_games[
        team_games["team"].isin(fbs_teams)
    ]
    .groupby("team")
    .agg(
        weighted_win_value=("win_value", "sum"),
        weighted_loss_penalty=("loss_penalty", "sum")
    )
    .reset_index()
)

fbs_stats = fbs_stats.merge(
    resume_metrics,
    on="team",
    how="left"
)

fbs_stats[
    [
        "team",
        "record",
        "weighted_win_value",
        "weighted_loss_penalty"
    ]
].sort_values(
    "weighted_win_value",
    ascending=False
).head(25)



# --- Notebook cell 17 ---
fbs_stats["weighted_win_score"] = min_max_score(
    fbs_stats["weighted_win_value"]
)

fbs_stats["weighted_loss_score"] = min_max_score(
    fbs_stats["weighted_loss_penalty"]
)

fbs_stats["rating_v3"] = (
    0.30 * fbs_stats["record_score"]
    + 0.25 * fbs_stats["sos_score"]
    + 0.20 * fbs_stats["margin_score"]
    + 0.20 * fbs_stats["weighted_win_score"]
    - 0.05 * fbs_stats["weighted_loss_score"]
)

rankings_v3 = fbs_stats.sort_values(
    "rating_v3",
    ascending=False
).reset_index(drop=True)

rankings_v3.index = rankings_v3.index + 1
rankings_v3.index.name = "rank"

rankings_v3[
    [
        "team",
        "conference",
        "record",
        "rating_v3",
        "weighted_win_value",
        "weighted_loss_penalty",
        "sos_score",
        "margin_score"
    ]
].head(25)



# --- Notebook cell 18 ---
# First-level opponent strength
opp_win_pct_lookup = fbs_stats.set_index("team")["win_pct"].to_dict()

team_games["opp_win_pct"] = team_games["opponent"].map(opp_win_pct_lookup)

# Calculate each team's average opponent win percentage
opp_wp = (
    team_games[team_games["team"].isin(fbs_teams)]
    .groupby("team")["opp_win_pct"]
    .mean()
    .to_dict()
)

# Map opponent's own schedule strength
team_games["opp_opp_win_pct"] = team_games["opponent"].map(opp_wp)

# Calculate upgraded SOS
sos_v2 = (
    team_games[team_games["team"].isin(fbs_teams)]
    .groupby("team")
    .agg(
        opp_win_pct=("opp_win_pct", "mean"),
        opp_opp_win_pct=("opp_opp_win_pct", "mean")
    )
    .reset_index()
)

sos_v2["sos_v2"] = (
    (2 / 3) * sos_v2["opp_win_pct"]
    + (1 / 3) * sos_v2["opp_opp_win_pct"]
)

fbs_stats = fbs_stats.merge(
    sos_v2[["team", "sos_v2"]],
    on="team",
    how="left"
)

fbs_stats[
    ["team", "record", "sos", "sos_v2"]
].sort_values(
    "sos_v2",
    ascending=False
).head(25)



# --- Notebook cell 19 ---
fbs_stats["sos_v2_score"] = min_max_score(
    fbs_stats["sos_v2"]
)

fbs_stats["rating_v4"] = (
    0.30 * fbs_stats["record_score"]
    + 0.25 * fbs_stats["sos_v2_score"]
    + 0.20 * fbs_stats["margin_score"]
    + 0.20 * fbs_stats["weighted_win_score"]
    - 0.05 * fbs_stats["weighted_loss_score"]
)

rankings_v4 = fbs_stats.sort_values(
    "rating_v4",
    ascending=False
).reset_index(drop=True)

rankings_v4.index = rankings_v4.index + 1
rankings_v4.index.name = "rank"

rankings_v4[
    [
        "team",
        "conference",
        "record",
        "rating_v4",
        "sos_v2_score",
        "margin_score",
        "weighted_win_value",
        "weighted_loss_penalty"
    ]
].head(25)



# --- Notebook cell 20 ---
team_games["location_multiplier"] = (
    team_games["location"].map({
        "home": 1.00,
        "neutral": 1.08,
        "away": 1.15
    })
)

team_games[
    [
        "game_id",
        "team",
        "opponent",
        "location",
        "location_multiplier"
    ]
].head(20)



# --- Notebook cell 21 ---
team_games["location_adjusted_win_value"] = (
    team_games["win"]
    * team_games["opponent_strength"]
    * team_games["location_multiplier"]
)

location_resume = (
    team_games[
        team_games["team"].isin(fbs_teams)
    ]
    .groupby("team")
    .agg(
        location_adjusted_win_value=("location_adjusted_win_value", "sum")
    )
    .reset_index()
)

fbs_stats = fbs_stats.merge(
    location_resume,
    on="team",
    how="left"
)

fbs_stats[
    [
        "team",
        "record",
        "weighted_win_value",
        "location_adjusted_win_value"
    ]
].sort_values(
    "location_adjusted_win_value",
    ascending=False
).head(25)



# --- Notebook cell 22 ---
fbs_stats["location_adjusted_win_score"] = min_max_score(
    fbs_stats["location_adjusted_win_value"]
)

fbs_stats["rating_v5"] = (
    0.30 * fbs_stats["record_score"]
    + 0.25 * fbs_stats["sos_v2_score"]
    + 0.20 * fbs_stats["margin_score"]
    + 0.20 * fbs_stats["location_adjusted_win_score"]
    - 0.05 * fbs_stats["weighted_loss_score"]
)

rankings_v5 = fbs_stats.sort_values(
    "rating_v5",
    ascending=False
).reset_index(drop=True)

rankings_v5.index = rankings_v5.index + 1
rankings_v5.index.name = "rank"

rankings_v5[
    [
        "team",
        "conference",
        "record",
        "rating_v5",
        "sos_v2_score",
        "margin_score",
        "location_adjusted_win_value",
        "weighted_loss_penalty"
    ]
].head(25)



# --- Notebook cell 23 ---
team_games["capped_margin"] = team_games["margin"].clip(
    lower=-28,
    upper=28
)

capped_margin = (
    team_games[
        team_games["team"].isin(fbs_teams)
    ]
    .groupby("team")["capped_margin"]
    .mean()
    .reset_index()
)

capped_margin = capped_margin.rename(
    columns={"capped_margin": "avg_capped_margin"}
)

fbs_stats = fbs_stats.merge(
    capped_margin,
    on="team",
    how="left"
)

fbs_stats["capped_margin_score"] = min_max_score(
    fbs_stats["avg_capped_margin"]
)

fbs_stats[
    [
        "team",
        "record",
        "avg_margin",
        "avg_capped_margin",
        "margin_score",
        "capped_margin_score"
    ]
].sort_values(
    "avg_capped_margin",
    ascending=False
).head(25)



# --- Notebook cell 24 ---
fbs_stats["rating_v6"] = (
    0.30 * fbs_stats["record_score"]
    + 0.25 * fbs_stats["sos_v2_score"]
    + 0.20 * fbs_stats["capped_margin_score"]
    + 0.20 * fbs_stats["location_adjusted_win_score"]
    - 0.05 * fbs_stats["weighted_loss_score"]
)

rankings_v6 = fbs_stats.sort_values(
    "rating_v6",
    ascending=False
).reset_index(drop=True)

rankings_v6.index = rankings_v6.index + 1
rankings_v6.index.name = "rank"

rankings_v6[
    [
        "team",
        "conference",
        "record",
        "rating_v6",
        "sos_v2_score",
        "capped_margin_score",
        "location_adjusted_win_value",
        "weighted_loss_penalty"
    ]
].head(25)



# --- Notebook cell 25 ---
# Get the current Top 25 teams from our model
top25_teams = set(rankings_v6.head(25)["team"])

# Identify games against Top 25 opponents
team_games["elite_opponent"] = team_games["opponent"].isin(top25_teams)

# Top 25 wins and losses
elite_results = (
    team_games[
        team_games["team"].isin(fbs_teams)
        & team_games["elite_opponent"]
    ]
    .groupby("team")
    .agg(
        elite_games=("win", "count"),
        elite_wins=("win", "sum")
    )
    .reset_index()
)

elite_results["elite_losses"] = (
    elite_results["elite_games"]
    - elite_results["elite_wins"]
)

fbs_stats = fbs_stats.merge(
    elite_results,
    on="team",
    how="left"
)

fbs_stats[
    ["elite_games", "elite_wins", "elite_losses"]
] = fbs_stats[
    ["elite_games", "elite_wins", "elite_losses"]
].fillna(0)

fbs_stats[
    [
        "team",
        "record",
        "elite_games",
        "elite_wins",
        "elite_losses"
    ]
].sort_values(
    ["elite_wins", "elite_losses"],
    ascending=[False, True]
).head(25)



# --- Notebook cell 26 ---
fbs_stats["elite_win_score"] = min_max_score(
    fbs_stats["elite_wins"]
)

fbs_stats["elite_loss_score"] = min_max_score(
    fbs_stats["elite_losses"]
)

fbs_stats["rating_v7"] = (
    0.27 * fbs_stats["record_score"]
    + 0.22 * fbs_stats["sos_v2_score"]
    + 0.18 * fbs_stats["capped_margin_score"]
    + 0.18 * fbs_stats["location_adjusted_win_score"]
    + 0.10 * fbs_stats["elite_win_score"]
    - 0.05 * fbs_stats["weighted_loss_score"]
    - 0.05 * fbs_stats["elite_loss_score"]
)

rankings_v7 = fbs_stats.sort_values(
    "rating_v7",
    ascending=False
).reset_index(drop=True)

rankings_v7.index = rankings_v7.index + 1
rankings_v7.index.name = "rank"

rankings_v7[
    [
        "team",
        "conference",
        "record",
        "rating_v7",
        "elite_wins",
        "elite_losses",
        "sos_v2_score",
        "capped_margin_score",
        "location_adjusted_win_value"
    ]
].head(25)



# --- Notebook cell 27 ---
def calculate_rankings_with_elite_iterations(
    fbs_stats,
    team_games,
    fbs_teams,
    iterations=5
):
    current_rankings = fbs_stats.copy()

    for _ in range(iterations):

        current_rankings = current_rankings.sort_values(
            "rating_v6",
            ascending=False
        ).reset_index(drop=True)

        top25_teams = set(
            current_rankings.head(25)["team"]
        )

        team_games["elite_opponent_iter"] = (
            team_games["opponent"].isin(top25_teams)
        )

        elite_results_iter = (
            team_games[
                team_games["team"].isin(fbs_teams)
                & team_games["elite_opponent_iter"]
            ]
            .groupby("team")
            .agg(
                elite_games_iter=("win", "count"),
                elite_wins_iter=("win", "sum")
            )
            .reset_index()
        )

        elite_results_iter["elite_losses_iter"] = (
            elite_results_iter["elite_games_iter"]
            - elite_results_iter["elite_wins_iter"]
        )

        current_rankings = current_rankings.drop(
            columns=[
                "elite_games_iter",
                "elite_wins_iter",
                "elite_losses_iter"
            ],
            errors="ignore"
        )

        current_rankings = current_rankings.merge(
            elite_results_iter,
            on="team",
            how="left"
        )

        current_rankings[
            [
                "elite_games_iter",
                "elite_wins_iter",
                "elite_losses_iter"
            ]
        ] = current_rankings[
            [
                "elite_games_iter",
                "elite_wins_iter",
                "elite_losses_iter"
            ]
        ].fillna(0)

        current_rankings["elite_win_score_iter"] = min_max_score(
            current_rankings["elite_wins_iter"]
        )

        current_rankings["elite_loss_score_iter"] = min_max_score(
            current_rankings["elite_losses_iter"]
        )

        current_rankings["rating_final"] = (
            0.27 * current_rankings["record_score"]
            + 0.22 * current_rankings["sos_v2_score"]
            + 0.18 * current_rankings["capped_margin_score"]
            + 0.18 * current_rankings["location_adjusted_win_score"]
            + 0.10 * current_rankings["elite_win_score_iter"]
            - 0.05 * current_rankings["weighted_loss_score"]
            - 0.05 * current_rankings["elite_loss_score_iter"]
        )

        current_rankings = current_rankings.sort_values(
            "rating_final",
            ascending=False
        ).reset_index(drop=True)

    return current_rankings


final_rankings = calculate_rankings_with_elite_iterations(
    fbs_stats,
    team_games,
    fbs_teams
)

final_rankings.index = final_rankings.index + 1
final_rankings.index.name = "rank"

final_rankings[
    [
        "team",
        "conference",
        "record",
        "rating_final",
        "elite_wins_iter",
        "elite_losses_iter",
        "sos_v2_score",
        "capped_margin_score"
    ]
].head(25)


# Pull regular-season betting lines
lines_data = fetch_cfbd("lines", {"year": SEASON})
df_lines = pd.DataFrame(lines_data)
if not df_lines.empty and "seasonType" in df_lines.columns:
    df_lines = df_lines[df_lines["seasonType"] == "regular"].copy()



# --- Notebook cell 33 ---
def average_spread(lines):
    spreads = [
        line["spread"]
        for line in lines
        if line.get("spread") is not None
    ]
    
    if len(spreads) == 0:
        return None
    
    return sum(spreads) / len(spreads)


df_lines["consensus_spread"] = df_lines["lines"].apply(average_spread)

df_lines[
    [
        "homeTeam",
        "awayTeam",
        "homeScore",
        "awayScore",
        "consensus_spread"
    ]
].dropna().head(20)



# --- Notebook cell 34 ---
df_lines["home_margin"] = (
    df_lines["homeScore"] - df_lines["awayScore"]
)

df_lines["home_vs_spread"] = (
    df_lines["home_margin"] + df_lines["consensus_spread"]
)

df_lines[
    [
        "homeTeam",
        "awayTeam",
        "homeScore",
        "awayScore",
        "consensus_spread",
        "home_margin",
        "home_vs_spread"
    ]
].dropna().head(20)



# --- Notebook cell 35 ---
# Home team ATS results
home_ats = df_lines[
    ["homeTeam", "consensus_spread", "home_vs_spread"]
].copy()

home_ats = home_ats.rename(columns={
    "homeTeam": "team",
    "home_vs_spread": "vs_spread"
})

# Away team's result is the exact opposite of the home team's result
away_ats = df_lines[
    ["awayTeam", "consensus_spread", "home_vs_spread"]
].copy()

away_ats = away_ats.rename(columns={
    "awayTeam": "team"
})

away_ats["vs_spread"] = -away_ats["home_vs_spread"]
away_ats = away_ats.drop(columns=["home_vs_spread"])

# Combine home and away
ats_games = pd.concat(
    [home_ats, away_ats],
    ignore_index=True
)

# Remove games without betting lines
ats_games = ats_games.dropna(subset=["vs_spread"])

# Determine cover / loss / push
ats_games["ats_win"] = (ats_games["vs_spread"] > 0).astype(int)
ats_games["ats_loss"] = (ats_games["vs_spread"] < 0).astype(int)
ats_games["ats_push"] = (ats_games["vs_spread"] == 0).astype(int)

# Team-level ATS statistics
ats_stats = (
    ats_games
    .groupby("team")
    .agg(
        ats_wins=("ats_win", "sum"),
        ats_losses=("ats_loss", "sum"),
        ats_pushes=("ats_push", "sum"),
        avg_vs_spread=("vs_spread", "mean")
    )
    .reset_index()
)

ats_stats["ats_games"] = (
    ats_stats["ats_wins"] +
    ats_stats["ats_losses"] +
    ats_stats["ats_pushes"]
)

ats_stats["ats_pct"] = (
    ats_stats["ats_wins"] /
    (ats_stats["ats_wins"] + ats_stats["ats_losses"])
)

ats_stats.sort_values(
    "avg_vs_spread",
    ascending=False
).head(25)



# --- Notebook cell 36 ---
# Merge ATS / Vegas performance into FBS stats
fbs_stats = fbs_stats.merge(
    ats_stats[
        [
            "team",
            "ats_wins",
            "ats_losses",
            "ats_pushes",
            "ats_pct",
            "avg_vs_spread"
        ]
    ],
    on="team",
    how="left"
)

# Fill missing betting data
fbs_stats[
    [
        "ats_wins",
        "ats_losses",
        "ats_pushes",
        "ats_pct",
        "avg_vs_spread"
    ]
] = fbs_stats[
    [
        "ats_wins",
        "ats_losses",
        "ats_pushes",
        "ats_pct",
        "avg_vs_spread"
    ]
].fillna(0)

fbs_stats[
    [
        "team",
        "record",
        "ats_wins",
        "ats_losses",
        "ats_pushes",
        "ats_pct",
        "avg_vs_spread"
    ]
].head(25)



# --- Notebook cell 37 ---
# Convert average performance vs Vegas to a 0-100 score
fbs_stats["vegas_score"] = (
    100 *
    (fbs_stats["avg_vs_spread"] - fbs_stats["avg_vs_spread"].min()) /
    (fbs_stats["avg_vs_spread"].max() - fbs_stats["avg_vs_spread"].min())
)

# Add Vegas performance as 10% of the ranking
# Existing rating_v6 keeps 90% of its influence
fbs_stats["rating_v7"] = (
    0.90 * fbs_stats["rating_v6"] +
    0.10 * fbs_stats["vegas_score"]
)

rankings_v7 = (
    fbs_stats
    .sort_values("rating_v7", ascending=False)
    .reset_index(drop=True)
)

rankings_v7.index = rankings_v7.index + 1
rankings_v7.index.name = "rank"

rankings_v7[
    [
        "team",
        "conference",
        "record",
        "rating_v7",
        "ats_wins",
        "ats_losses",
        "ats_pushes",
        "avg_vs_spread",
        "vegas_score"
    ]
].head(25)



# --- Notebook cell 38 ---
conference_quality = (
    fbs_stats
    .groupby("conference")
    .agg(
        avg_team_rating=("rating_v7", "mean"),
        median_team_rating=("rating_v7", "median"),
        avg_sos=("sos_v2_score", "mean"),
        teams=("team", "count")
    )
    .reset_index()
)

conference_quality = conference_quality.sort_values(
    "avg_team_rating",
    ascending=False
).reset_index(drop=True)

conference_quality



# --- Notebook cell 39 ---
# Create opponent conference lookup
conference_lookup = (
    fbs_stats
    .set_index("team")["conference"]
    .to_dict()
)

team_games["opponent_conference"] = (
    team_games["opponent"].map(conference_lookup)
)

# Keep only games between teams from different conferences
non_conf_games = team_games[
    (team_games["conference"] != team_games["opponent_conference"]) &
    (team_games["opponent_conference"].notna())
].copy()

# Win indicator
non_conf_games["win"] = (
    non_conf_games["points_for"] > non_conf_games["points_against"]
).astype(int)

# Margin
non_conf_games["margin"] = (
    non_conf_games["points_for"] -
    non_conf_games["points_against"]
)

# Conference non-conference performance
conference_nonconf = (
    non_conf_games
    .groupby("conference")
    .agg(
        nonconf_games=("win", "count"),
        nonconf_wins=("win", "sum"),
        nonconf_win_pct=("win", "mean"),
        nonconf_avg_margin=("margin", "mean")
    )
    .reset_index()
)

conference_nonconf = conference_nonconf.sort_values(
    "nonconf_win_pct",
    ascending=False
).reset_index(drop=True)

conference_nonconf



# --- Notebook cell 40 ---
# Conference quality lookup from our earlier conference ratings
conf_rating_lookup = (
    conference_quality
    .set_index("conference")["avg_team_rating"]
    .to_dict()
)

# Add opponent conference strength to each non-conference game
non_conf_games["opponent_conf_rating"] = (
    non_conf_games["opponent_conference"]
    .map(conf_rating_lookup)
)

# Only use games where we have a conference rating
conf_adjusted_games = non_conf_games[
    non_conf_games["opponent_conf_rating"].notna()
].copy()

# Reward wins over stronger conferences more
conf_adjusted_games["adjusted_win_value"] = (
    conf_adjusted_games["win"] *
    conf_adjusted_games["opponent_conf_rating"]
)

# Summarize by conference
conference_adjusted = (
    conf_adjusted_games
    .groupby("conference")
    .agg(
        games=("win", "count"),
        wins=("win", "sum"),
        win_pct=("win", "mean"),
        avg_opponent_conf_strength=("opponent_conf_rating", "mean"),
        adjusted_win_value=("adjusted_win_value", "mean"),
        avg_margin=("margin", "mean")
    )
    .reset_index()
)

conference_adjusted = conference_adjusted.sort_values(
    "adjusted_win_value",
    ascending=False
).reset_index(drop=True)

conference_adjusted



# --- Notebook cell 41 ---
# Merge overall conference quality with non-conference performance
conference_model = conference_quality.merge(
    conference_adjusted[
        [
            "conference",
            "adjusted_win_value",
            "avg_margin"
        ]
    ],
    on="conference",
    how="left"
)

# Only conferences with enough teams to be meaningful
conference_model = conference_model[
    conference_model["teams"] >= 8
].copy()

# Fill missing values
conference_model[
    ["adjusted_win_value", "avg_margin"]
] = conference_model[
    ["adjusted_win_value", "avg_margin"]
].fillna(0)

# Function to scale a column from 0-100
def scale_0_100(series):
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=series.index)
    return ((series - lo) / (hi - lo)) * 100

# Scale the three conference-strength components
conference_model["team_strength_score"] = scale_0_100(
    conference_model["avg_team_rating"]
)

conference_model["nonconf_score"] = scale_0_100(
    conference_model["adjusted_win_value"]
)

conference_model["nonconf_margin_score"] = scale_0_100(
    conference_model["avg_margin"]
)

# Final conference strength score
conference_model["conference_score"] = (
    0.40 * conference_model["team_strength_score"]
    + 0.35 * conference_model["nonconf_score"]
    + 0.25 * conference_model["nonconf_margin_score"]
)

conference_model = conference_model.sort_values(
    "conference_score",
    ascending=False
).reset_index(drop=True)

conference_model[
    [
        "conference",
        "conference_score",
        "team_strength_score",
        "nonconf_score",
        "nonconf_margin_score"
    ]
]



# --- Notebook cell 42 ---
# Map conference strength back to each team
conference_score_lookup = (
    conference_model
    .set_index("conference")["conference_score"]
    .to_dict()
)

fbs_stats["conference_score"] = (
    fbs_stats["conference"].map(conference_score_lookup)
)

# Fill independents / missing conference scores with neutral value
fbs_stats["conference_score"] = (
    fbs_stats["conference_score"].fillna(50)
)

# Version 8:
# Keep 95% of the existing team rating,
# add 5% conference context
fbs_stats["rating_v8"] = (
    0.95 * fbs_stats["rating_v7"]
    + 0.05 * fbs_stats["conference_score"]
)

rankings_v8 = (
    fbs_stats
    .sort_values("rating_v8", ascending=False)
    .reset_index(drop=True)
)

rankings_v8.index = rankings_v8.index + 1
rankings_v8.index.name = "rank"

rankings_v8[
    [
        "team",
        "conference",
        "record",
        "rating_v8",
        "conference_score",
        "ats_wins",
        "ats_losses",
        "avg_vs_spread"
    ]
].head(25)



# --- Notebook cell 44 ---
# Version 8B - reduce conference adjustment from 5% to 3%

fbs_stats["rating_v8b"] = (
    0.97 * fbs_stats["rating_v7"]
    + 0.03 * fbs_stats["conference_score"]
)

rankings_v8b = (
    fbs_stats
    .sort_values("rating_v8b", ascending=False)
    .reset_index(drop=True)
)

rankings_v8b.index = rankings_v8b.index + 1
rankings_v8b.index.name = "rank"

rankings_v8b[
    [
        "team",
        "conference",
        "record",
        "rating_v8b",
        "conference_score",
        "ats_wins",
        "ats_losses",
        "avg_vs_spread"
    ]
].head(25)



# --- Notebook cell 46 ---
import numpy as np

# Use our current rating as the opponent-strength scale
rating_mean = fbs_stats["rating_v8b"].mean()
rating_std = fbs_stats["rating_v8b"].std()

rating_lookup = (
    fbs_stats
    .set_index("team")["rating_v8b"]
    .to_dict()
)

team_games["opponent_rating"] = (
    team_games["opponent"].map(rating_lookup)
)

# Convert opponent strength to a standardized value
team_games["opponent_z"] = (
    (team_games["opponent_rating"] - rating_mean) /
    rating_std
)

# Location adjustment for an average FBS team
location_logit = {
    "home": 0.20,
    "neutral": 0.00,
    "away": -0.20
}

team_games["location_logit"] = (
    team_games["location"]
    .map(location_logit)
    .fillna(0)
)

# Probability an average FBS team would win each game
team_games["avg_team_win_prob"] = (
    1 /
    (
        1 +
        np.exp(
            team_games["opponent_z"]
            - team_games["location_logit"]
        )
    )
)

team_games[
    [
        "team",
        "opponent",
        "location",
        "opponent_rating",
        "avg_team_win_prob"
    ]
].dropna().head(20)



# --- Notebook cell 47 ---
expected_wins = (
    team_games[
        team_games["team"].isin(fbs_teams)
    ]
    .groupby("team")
    .agg(
        expected_wins=("avg_team_win_prob", "sum")
    )
    .reset_index()
)

fbs_stats = fbs_stats.merge(
    expected_wins,
    on="team",
    how="left"
)

fbs_stats["sor_overachievement"] = (
    fbs_stats["wins"] -
    fbs_stats["expected_wins"]
)

fbs_stats[
    [
        "team",
        "conference",
        "record",
        "wins",
        "expected_wins",
        "sor_overachievement"
    ]
].sort_values(
    "sor_overachievement",
    ascending=False
).head(25)



# --- Notebook cell 49 ---
fbs_stats["sor_score"] = scale_0_100(
    fbs_stats["sor_overachievement"]
)

fbs_stats["rating_v9"] = (
    0.90 * fbs_stats["rating_v8b"]
    + 0.10 * fbs_stats["sor_score"]
)

rankings_v9 = (
    fbs_stats
    .sort_values("rating_v9", ascending=False)
    .reset_index(drop=True)
)

rankings_v9.index = rankings_v9.index + 1
rankings_v9.index.name = "rank"

rankings_v9[
    [
        "team",
        "conference",
        "record",
        "rating_v9",
        "expected_wins",
        "sor_overachievement",
        "sor_score"
    ]
].head(25)



# --- Notebook cell 55 ---
# VERSION 10
# Resume-first model

fbs_stats["rating_v10"] = (
    0.25 * fbs_stats["sor_score"]
    + 0.20 * fbs_stats["record_score"]
    + 0.20 * fbs_stats["sos_v2_score"]
    + 0.17 * fbs_stats["location_adjusted_win_score"]
    + 0.08 * fbs_stats["capped_margin_score"]
    + 0.05 * fbs_stats["vegas_score"]
    + 0.03 * fbs_stats["conference_score"]
    - 0.02 * fbs_stats["weighted_loss_score"]
)

rankings_v10 = (
    fbs_stats
    .sort_values("rating_v10", ascending=False)
    .reset_index(drop=True)
)

rankings_v10.index = rankings_v10.index + 1
rankings_v10.index.name = "rank"

rankings_v10[
    [
        "team",
        "conference",
        "record",
        "rating_v10",
        "sor_score",
        "sos_v2_score",
        "location_adjusted_win_score",
        "capped_margin_score",
        "vegas_score",
        "conference_score",
        "weighted_loss_score"
    ]
].head(25)



# --- Notebook cell 57 ---
# VERSION 11 - percentile-based scaling

def percentile_score(series, higher_is_better=True):
    if higher_is_better:
        return series.rank(pct=True) * 100
    else:
        return (1 - series.rank(pct=True)) * 100


# Re-scale major inputs
fbs_stats["sor_pct_score"] = percentile_score(
    fbs_stats["sor_overachievement"]
)

fbs_stats["record_pct_score"] = percentile_score(
    fbs_stats["win_pct"]
)

fbs_stats["sos_pct_score"] = percentile_score(
    fbs_stats["sos_v2"]
)

fbs_stats["location_win_pct_score"] = percentile_score(
    fbs_stats["location_adjusted_win_value"]
)

fbs_stats["margin_pct_score"] = percentile_score(
    fbs_stats["avg_capped_margin"]
)

fbs_stats["vegas_pct_score"] = percentile_score(
    fbs_stats["avg_vs_spread"]
)

fbs_stats["conference_pct_score"] = percentile_score(
    fbs_stats["conference_score"]
)

fbs_stats["loss_pct_score"] = percentile_score(
    fbs_stats["weighted_loss_penalty"],
    higher_is_better=False
)



# --- Notebook cell 58 ---
fbs_stats["rating_v11"] = (
    0.25 * fbs_stats["sor_pct_score"]
    + 0.20 * fbs_stats["record_pct_score"]
    + 0.20 * fbs_stats["sos_pct_score"]
    + 0.17 * fbs_stats["location_win_pct_score"]
    + 0.08 * fbs_stats["margin_pct_score"]
    + 0.05 * fbs_stats["vegas_pct_score"]
    + 0.03 * fbs_stats["conference_pct_score"]
    + 0.02 * fbs_stats["loss_pct_score"]
)

rankings_v11 = (
    fbs_stats
    .sort_values("rating_v11", ascending=False)
    .reset_index(drop=True)
)

rankings_v11.index = rankings_v11.index + 1
rankings_v11.index.name = "rank"

rankings_v11[
    [
        "team",
        "conference",
        "record",
        "rating_v11",
        "sor_pct_score",
        "sos_pct_score",
        "location_win_pct_score",
        "margin_pct_score",
        "vegas_pct_score",
        "conference_pct_score",
        "loss_pct_score"
    ]
].head(25)



# --- Notebook cell 66 ---
# V12 LOSS QUALITY SYSTEM
# Evaluates losses based on:
# 1. Opponent quality
# 2. Margin of loss
# 3. Game location

loss_games = team_games[
    team_games["loss"] == 1
].copy()

# -------------------------
# OPPONENT QUALITY
# -------------------------
# Strong opponent = smaller penalty
# Weak opponent = larger penalty

loss_games["opponent_loss_factor"] = (
    (100 - loss_games["opponent_rating"]) / 100
)

# -------------------------
# MARGIN OF LOSS
# -------------------------
# Close losses receive limited punishment.
# Blowouts become increasingly costly.

loss_games["loss_margin"] = (
    loss_games["points_against"] -
    loss_games["points_for"]
)

loss_games["margin_loss_factor"] = (
    loss_games["loss_margin"].clip(upper=28) / 28
)

# -------------------------
# LOCATION
# -------------------------
# Home losses hurt more.
# Road losses hurt less.

loss_games["loss_location_factor"] = (
    loss_games["location"].map({
        "home": 1.15,
        "neutral": 1.00,
        "away": 0.85
    })
)

# -------------------------
# COMBINE LOSS SEVERITY
# -------------------------

loss_games["loss_severity"] = (
    0.50 * loss_games["opponent_loss_factor"] +
    0.35 * loss_games["margin_loss_factor"] +
    0.15 * (loss_games["loss_location_factor"] - 0.85) / 0.30
)

# Keep it between 0 and 1
loss_games["loss_severity"] = (
    loss_games["loss_severity"]
    .clip(0, 1)
)

# Show individual losses
loss_games[
    [
        "team",
        "opponent",
        "location",
        "loss_margin",
        "opponent_rating",
        "loss_severity"
    ]
].sort_values(
    "loss_severity",
    ascending=False
).head(30)



# --- Notebook cell 67 ---
# Aggregate loss quality by team

team_loss_quality = (
    loss_games
    .groupby("team")
    .agg(
        losses=("loss_severity", "count"),
        avg_loss_severity=("loss_severity", "mean"),
        worst_loss_severity=("loss_severity", "max"),
        total_loss_severity=("loss_severity", "sum")
    )
    .reset_index()
)

# Teams with no losses need zeros
all_teams = pd.DataFrame({
    "team": team_games["team"].unique()
})

team_loss_quality = all_teams.merge(
    team_loss_quality,
    on="team",
    how="left"
).fillna(0)

team_loss_quality.sort_values(
    "total_loss_severity",
    ascending=False
).head(30)



# --- Notebook cell 68 ---
contenders = [
    "Indiana",
    "Ohio State",
    "Georgia",
    "Oregon",
    "Texas Tech",
    "BYU",
    "Alabama",
    "Ole Miss",
    "Oklahoma",
    "Notre Dame",
    "Miami",
    "Texas A&M"
]

team_loss_quality[
    team_loss_quality["team"].isin(contenders)
].sort_values("total_loss_severity")



# --- Notebook cell 69 ---
# V12 TEAM LOSS QUALITY SCORE
# 100 = best possible loss profile
# Undefeated teams receive 100

team_loss_quality["loss_quality_raw"] = (
    0.65 * team_loss_quality["avg_loss_severity"] +
    0.35 * team_loss_quality["worst_loss_severity"]
)

# Convert to 0-100 score
team_loss_quality["loss_quality_score"] = (
    100 * (1 - team_loss_quality["loss_quality_raw"])
).clip(0, 100)

# Undefeated teams automatically get 100
team_loss_quality.loc[
    team_loss_quality["losses"] == 0,
    "loss_quality_score"
] = 100

team_loss_quality[
    team_loss_quality["team"].isin(contenders)
][
    [
        "team",
        "losses",
        "avg_loss_severity",
        "worst_loss_severity",
        "loss_quality_score"
    ]
].sort_values("loss_quality_score", ascending=False)



# --- Notebook cell 70 ---
rankings_v12 = rankings_v11.merge(
    team_loss_quality[
        ["team", "loss_quality_score"]
    ],
    on="team",
    how="left"
)

rankings_v12["loss_quality_score"] = (
    rankings_v12["loss_quality_score"].fillna(100)
)



# --- Notebook cell 72 ---
# VERSION 12
# Resume-first percentile model
# Improved loss-quality system replaces old loss_pct_score

rankings_v12 = rankings_v11.merge(
    team_loss_quality[
        ["team", "loss_quality_score"]
    ],
    on="team",
    how="left"
)

# Undefeated / missing teams get perfect loss quality
rankings_v12["loss_quality_score"] = (
    rankings_v12["loss_quality_score"].fillna(100)
)

rankings_v12["rating_v12"] = (
    0.24235 * rankings_v12["sor_pct_score"]
    + 0.19388 * rankings_v12["record_pct_score"]
    + 0.19388 * rankings_v12["sos_pct_score"]
    + 0.16480 * rankings_v12["location_win_pct_score"]
    + 0.07755 * rankings_v12["margin_pct_score"]
    + 0.04847 * rankings_v12["vegas_pct_score"]
    + 0.02907 * rankings_v12["conference_pct_score"]
    + 0.05000 * rankings_v12["loss_quality_score"]
)

rankings_v12 = (
    rankings_v12
    .sort_values("rating_v12", ascending=False)
    .reset_index(drop=True)
)

rankings_v12.index = rankings_v12.index + 1
rankings_v12.index.name = "rank"

rankings_v12[
    [
        "team",
        "conference",
        "record",
        "rating_v12",
        "sor_pct_score",
        "sos_pct_score",
        "location_win_pct_score",
        "margin_pct_score",
        "vegas_pct_score",
        "conference_pct_score",
        "loss_quality_score"
    ]
].head(25)



# --- Notebook cell 75 ---
# QUALITY WIN STRENGTH
# Rewards wins based on opponent quality and game location

quality_win_games = team_games[
    team_games["win"] == 1
].copy()

# Normalize opponent rating to 0-1
rating_min = quality_win_games["opponent_rating"].min()
rating_max = quality_win_games["opponent_rating"].max()

quality_win_games["opponent_quality"] = (
    (quality_win_games["opponent_rating"] - rating_min) /
    (rating_max - rating_min)
)

# Location multiplier:
# road wins > neutral wins > home wins
quality_win_games["quality_location_multiplier"] = (
    quality_win_games["location"].map({
        "home": 1.00,
        "neutral": 1.08,
        "away": 1.15
    })
)

# Continuous quality-win value
quality_win_games["quality_win_value"] = (
    quality_win_games["opponent_quality"]
    * quality_win_games["quality_location_multiplier"]
)

quality_win_games[
    [
        "team",
        "opponent",
        "location",
        "opponent_rating",
        "quality_win_value"
    ]
].sort_values(
    "quality_win_value",
    ascending=False
).head(30)



# --- Notebook cell 76 ---
quality_win_summary = (
    quality_win_games
    .groupby("team")
    .agg(
        total_quality_win_value=("quality_win_value", "sum"),
        avg_quality_win_value=("quality_win_value", "mean"),
        best_win_value=("quality_win_value", "max"),
        quality_wins=("quality_win_value", lambda x: (x >= 0.70).sum())
    )
    .reset_index()
)

quality_win_summary.sort_values(
    "total_quality_win_value",
    ascending=False
).head(30)



# --- Notebook cell 77 ---
quality_win_summary["quality_win_strength_score"] = (
    quality_win_summary["total_quality_win_value"]
    .rank(pct=True) * 100
)

quality_win_summary.sort_values(
    "quality_win_strength_score",
    ascending=False
).head(25)



# --- Notebook cell 78 ---
# ==========================================
# VERSION 13 - QUALITY WIN STRENGTH
# ==========================================

rankings_v13 = rankings_v12.merge(
    quality_win_summary[
        ["team", "quality_win_strength_score"]
    ],
    on="team",
    how="left"
)

rankings_v13["quality_win_strength_score"] = (
    rankings_v13["quality_win_strength_score"].fillna(0)
)

rankings_v13["rating_v13"] = (
    0.24 * rankings_v13["sor_pct_score"]
    + 0.18 * rankings_v13["record_pct_score"]
    + 0.18 * rankings_v13["sos_pct_score"]
    + 0.08 * rankings_v13["location_win_pct_score"]
    + 0.12 * rankings_v13["quality_win_strength_score"]
    + 0.07 * rankings_v13["margin_pct_score"]
    + 0.05 * rankings_v13["vegas_pct_score"]
    + 0.03 * rankings_v13["conference_pct_score"]
    + 0.05 * rankings_v13["loss_quality_score"]
)

rankings_v13 = (
    rankings_v13
    .sort_values("rating_v13", ascending=False)
    .reset_index(drop=True)
)

rankings_v13.index = rankings_v13.index + 1
rankings_v13.index.name = "rank"

rankings_v13[
    [
        "team",
        "conference",
        "record",
        "rating_v13",
        "sor_pct_score",
        "sos_pct_score",
        "quality_win_strength_score",
        "loss_quality_score",
        "margin_pct_score",
        "vegas_pct_score"
    ]
].head(25)



# --- Notebook cell 79 ---
# ==========================================
# VERSION 14 - REDUCED SOS DOUBLE COUNTING
# ==========================================

rankings_v14 = rankings_v13.copy()

rankings_v14["rating_v14"] = (
    0.27 * rankings_v14["sor_pct_score"]
    + 0.18 * rankings_v14["record_pct_score"]
    + 0.10 * rankings_v14["sos_pct_score"]
    + 0.15 * rankings_v14["quality_win_strength_score"]
    + 0.08 * rankings_v14["location_win_pct_score"]
    + 0.07 * rankings_v14["margin_pct_score"]
    + 0.05 * rankings_v14["vegas_pct_score"]
    + 0.07 * rankings_v14["loss_quality_score"]
    + 0.03 * rankings_v14["conference_pct_score"]
)

rankings_v14 = (
    rankings_v14
    .sort_values("rating_v14", ascending=False)
    .reset_index(drop=True)
)

rankings_v14.index = rankings_v14.index + 1
rankings_v14.index.name = "rank"

rankings_v14[
    [
        "team",
        "conference",
        "record",
        "rating_v14",
        "sor_pct_score",
        "sos_pct_score",
        "quality_win_strength_score",
        "loss_quality_score",
        "margin_pct_score",
        "vegas_pct_score"
    ]
].head(25)



# --- Notebook cell 81 ---
# ==========================================
# V15 - BETTER QUALITY WIN RESUME
# ==========================================

quality_win_games = team_games[
    team_games["win"] == 1
].copy()

# Normalize opponent rating to 0-1
rating_min = quality_win_games["opponent_rating"].min()
rating_max = quality_win_games["opponent_rating"].max()

quality_win_games["opponent_quality"] = (
    (quality_win_games["opponent_rating"] - rating_min) /
    (rating_max - rating_min)
)

# Road > neutral > home
quality_win_games["quality_location_multiplier"] = (
    quality_win_games["location"].map({
        "home": 1.00,
        "neutral": 1.08,
        "away": 1.15
    })
)

quality_win_games["quality_win_value"] = (
    quality_win_games["opponent_quality"]
    * quality_win_games["quality_location_multiplier"]
)

# Only stronger wins count heavily here
quality_win_games["strong_win_value"] = (
    quality_win_games["quality_win_value"]
    .where(quality_win_games["opponent_quality"] >= 0.60, 0)
)

quality_win_games["elite_win"] = (
    quality_win_games["opponent_quality"] >= 0.75
).astype(int)



# --- Notebook cell 82 ---
win_resume = (
    quality_win_games
    .groupby("team")
    .agg(
        strong_win_total=("strong_win_value", "sum"),
        best_win_value=("quality_win_value", "max"),
        elite_wins=("elite_win", "sum"),
        quality_wins=("opponent_quality", lambda x: (x >= 0.60).sum())
    )
    .reset_index()
)



# --- Notebook cell 83 ---
win_resume["strong_win_score"] = (
    win_resume["strong_win_total"]
    .rank(pct=True) * 100
)

win_resume["best_win_score"] = (
    win_resume["best_win_value"]
    .rank(pct=True) * 100
)

win_resume["quality_depth_score"] = (
    win_resume["quality_wins"]
    .rank(pct=True, method="average") * 100
)

win_resume["win_resume_score"] = (
    0.50 * win_resume["strong_win_score"]
    + 0.30 * win_resume["best_win_score"]
    + 0.20 * win_resume["quality_depth_score"]
)



# --- Notebook cell 85 ---
rankings_v15 = rankings_v14.merge(
    win_resume[
        ["team", "win_resume_score"]
    ],
    on="team",
    how="left"
)

rankings_v15["win_resume_score"] = (
    rankings_v15["win_resume_score"].fillna(0)
)

rankings_v15["rating_v15"] = (
    0.27 * rankings_v15["sor_pct_score"]
    + 0.18 * rankings_v15["record_pct_score"]
    + 0.10 * rankings_v15["sos_pct_score"]
    + 0.15 * rankings_v15["win_resume_score"]
    + 0.08 * rankings_v15["location_win_pct_score"]
    + 0.07 * rankings_v15["margin_pct_score"]
    + 0.05 * rankings_v15["vegas_pct_score"]
    + 0.07 * rankings_v15["loss_quality_score"]
    + 0.03 * rankings_v15["conference_pct_score"]
)

rankings_v15 = (
    rankings_v15
    .sort_values("rating_v15", ascending=False)
    .reset_index(drop=True)
)

rankings_v15.index = rankings_v15.index + 1
rankings_v15.index.name = "rank"

rankings_v15[
    [
        "team",
        "conference",
        "record",
        "rating_v15",
        "sor_pct_score",
        "sos_pct_score",
        "win_resume_score",
        "loss_quality_score",
        "margin_pct_score",
        "vegas_pct_score"
    ]
].head(25)



# --- Notebook cell 93 ---
# ==========================================
# V16 - CONTENDER HEAD-TO-HEAD ADJUSTMENT
# ==========================================

rankings_v16 = rankings_v15.copy()

rankings_v16["rating_pre_h2h"] = rankings_v16["rating_v15"]

# Identify Top 30 teams BEFORE applying H2H
top30_teams = set(
    rankings_v16
    .sort_values("rating_pre_h2h", ascending=False)
    .head(30)["team"]
)

rating_lookup = (
    rankings_v16
    .set_index("team")["rating_pre_h2h"]
    .to_dict()
)

h2h_adjustment = {
    team: 0.0
    for team in rankings_v16["team"]
}

# Actual head-to-head wins
h2h_games = (
    team_games[
        team_games["win"] == 1
    ][["team", "opponent"]]
    .drop_duplicates()
)

for _, game in h2h_games.iterrows():

    winner = game["team"]
    loser = game["opponent"]

    # BOTH teams must be Top 30 contenders
    if winner not in top30_teams or loser not in top30_teams:
        continue

    winner_rating = rating_lookup[winner]
    loser_rating = rating_lookup[loser]

    rating_gap = abs(winner_rating - loser_rating)

    # H2H only matters when resumes are reasonably close
    if rating_gap <= 5.0:

        h2h_adjustment[winner] += 0.75
        h2h_adjustment[loser] -= 0.75


# Cap total H2H influence
rankings_v16["h2h_adjustment"] = (
    rankings_v16["team"]
    .map(h2h_adjustment)
    .fillna(0)
    .clip(-1.5, 1.5)
)

rankings_v16["rating_v16"] = (
    rankings_v16["rating_pre_h2h"]
    + rankings_v16["h2h_adjustment"]
)

rankings_v16 = (
    rankings_v16
    .sort_values("rating_v16", ascending=False)
    .reset_index(drop=True)
)

rankings_v16.index = rankings_v16.index + 1
rankings_v16.index.name = "rank"



# --- Notebook cell 95 ---
# COMPONENT RANKS FOR EXPLANATIONS

explain_cols = {
    "sor_pct_score": "SOR",
    "sos_pct_score": "SOS",
    "win_resume_score": "Win Resume",
    "loss_quality_score": "Loss Quality",
    "margin_pct_score": "Margin",
    "vegas_pct_score": "Vegas",
    "conference_pct_score": "Conference",
    "location_win_pct_score": "Location Wins"
}

for col, label in explain_cols.items():
    rankings_v16[f"{col}_rank"] = (
        rankings_v16[col]
        .rank(ascending=False, method="min")
        .astype(int)
    )



# --- Notebook cell 96 ---
def build_team_explanation(row):

    strengths = []
    weaknesses = []

    metrics = [
        ("SOR", row["sor_pct_score"], row["sor_pct_score_rank"]),
        ("SOS", row["sos_pct_score"], row["sos_pct_score_rank"]),
        ("Win Resume", row["win_resume_score"], row["win_resume_score_rank"]),
        ("Loss Quality", row["loss_quality_score"], row["loss_quality_score_rank"]),
        ("Margin", row["margin_pct_score"], row["margin_pct_score_rank"]),
        ("Vegas", row["vegas_pct_score"], row["vegas_pct_score_rank"]),
        ("Location Wins", row["location_win_pct_score"], row["location_win_pct_score_rank"]),
    ]

    # strongest components
    metrics_sorted = sorted(
        metrics,
        key=lambda x: x[1],
        reverse=True
    )

    for name, score, metric_rank in metrics_sorted[:3]:
        strengths.append(
            f"{name} #{metric_rank}"
        )

    # weakest component
    weakest = min(
        metrics,
        key=lambda x: x[1]
    )

    weaknesses.append(
        f"{weakest[0]} #{weakest[2]}"
    )

    explanation = (
        f"{row['record']} record; "
        f"strengths: {', '.join(strengths)}; "
        f"biggest weakness: {weaknesses[0]}"
    )

    if row["h2h_adjustment"] > 0:
        explanation += "; head-to-head boost"

    elif row["h2h_adjustment"] < 0:
        explanation += "; head-to-head penalty"

    return explanation



# --- Notebook cell 97 ---
rankings_v16["explanation"] = (
    rankings_v16.apply(
        build_team_explanation,
        axis=1
    )
)

rankings_v16[
    [
        "team",
        "conference",
        "record",
        "rating_v16",
        "h2h_adjustment",
        "explanation"
    ]
].head(25)



# --- Notebook cell 98 ---
final_rankings = rankings_v16[
    [
        "team",
        "conference",
        "record",
        "rating_v16",
        "h2h_adjustment",
        "sor_pct_score",
        "sos_pct_score",
        "win_resume_score",
        "loss_quality_score",
        "margin_pct_score",
        "vegas_pct_score",
        "conference_pct_score",
        "location_win_pct_score",
        "explanation"
    ]
].copy()

final_rankings = final_rankings.rename(columns={
    "team": "Team",
    "conference": "Conference",
    "record": "Record",
    "rating_v16": "Rating",
    "h2h_adjustment": "H2H Adjustment",
    "sor_pct_score": "SOR Score",
    "sos_pct_score": "SOS Score",
    "win_resume_score": "Win Resume",
    "loss_quality_score": "Loss Quality",
    "margin_pct_score": "Margin Score",
    "vegas_pct_score": "Vegas Score",
    "conference_pct_score": "Conference Score",
    "location_win_pct_score": "Location Win Score",
    "explanation": "Explanation"
})

final_rankings["Rating"] = final_rankings["Rating"].round(2)
final_rankings["SOR Score"] = final_rankings["SOR Score"].round(1)
final_rankings["SOS Score"] = final_rankings["SOS Score"].round(1)
final_rankings["Win Resume"] = final_rankings["Win Resume"].round(1)
final_rankings["Loss Quality"] = final_rankings["Loss Quality"].round(1)
final_rankings["Margin Score"] = final_rankings["Margin Score"].round(1)
final_rankings["Vegas Score"] = final_rankings["Vegas Score"].round(1)

final_rankings.head(25)



# --- Notebook cell 99 ---
final_rankings.to_csv(
    f"CFB_{SEASON}_Final_Rankings.csv",
    index=True,
    index_label="Rank"
)



# CFP-eligible export. Configure exclusions with a comma-separated environment variable.
# Example: CFP_INELIGIBLE_TEAMS="Team A,Team B"
_raw_ineligible = os.getenv("CFP_INELIGIBLE_TEAMS", "")
CFP_INELIGIBLE_TEAMS = {x.strip() for x in _raw_ineligible.split(",") if x.strip()}
cfp_rankings = final_rankings[~final_rankings["Team"].isin(CFP_INELIGIBLE_TEAMS)].copy()
cfp_rankings = cfp_rankings.reset_index(drop=True)
cfp_rankings.index = cfp_rankings.index + 1
cfp_rankings.index.name = "Rank"
cfp_rankings.to_csv(f"CFB_{SEASON}_CFP_Eligible_Final.csv", index=True, index_label="Rank")

print(f"Saved {len(final_rankings)} teams to CFB_{SEASON}_Final_Rankings.csv")
print(f"Saved {len(cfp_rankings)} CFP-eligible teams to CFB_{SEASON}_CFP_Eligible_Final.csv")
print(cfp_rankings.head(25).to_string())

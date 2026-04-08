# letterboxd_analysis
As I was very disappointed at letterboxd wrapped and I wanted to have more insights about my movie-watching persona I did this little project for fun :)

# Letterboxd Data Dashboard

A personal data project that transforms my Letterboxd watch history into an interactive Tableau dashboard — enriched with TMDB metadata and a content-based recommendation engine built in Python.

---

## Overview

I exported my full watch history from Letterboxd (925 films) and built a pipeline that:

1. Enriches each film with metadata from the TMDB API (director, genres, country, language, runtime, ratings)
2. Cross-references supplementary Letterboxd exports (ratings, diary, reviews) for complete data
3. Computes a **Taste Match Score** for every film in a curated TMDB catalog — identifying unseen films most aligned with my viewing profile
4. Outputs analysis-ready CSVs imported into a multi-panel Tableau dashboard

---

## Files

| File | Description |
|------|-------------|
| `enrich_letterboxd.py` | Combines Letterboxd export CSVs and enriches each film via the TMDB API |
| `letterboxd_recommend.py` | Builds a taste profile, scores the TMDB catalog, and outputs recommendations |
| `letterboxd_dashboard.twbx` | Tableau packaged workbook with all 5 dashboard panels |
| `screenshots/` | Dashboard preview images |

---

## Dashboard Panels

- **Overview** — key metrics, rating distribution, genre breakdown, full film log with taglines
- **Your Taste** — avg rating by decade, divergence from TMDB consensus, films that define your profile
- **Watching Habits** — consumption over time, release year distribution, map by production country, top directors
- **You vs. The Crowd** — your ratings vs. TMDB by genre, overrated and underrated scatter plot
- **What to Watch Next** — personalized recommendations scored by taste match, filterable by genre, country, decade and runtime

---

## Recommendation Engine

The content-based recommender builds a weighted taste profile from films rated ≥ 3.5★, then scores every unseen film in the TMDB catalog by attribute similarity.

**Attributes and weights:**

| Attribute | Weight |
|-----------|--------|
| Primary genre | 30% |
| Director | 20% |
| Decade | 15% |
| Secondary genre | 15% |
| Production country | 10% |
| Original language | 10% |

Films liked on Letterboxd receive a 1.5× boost in the profile. The final score is normalized to 0–100.

---

## Tech Stack

- **Python** — pandas, requests, scikit-learn, tqdm, python-dotenv
- **TMDB API** — film metadata and catalog
- **Tableau** — interactive dashboard and visualizations
- **Data sources** — Letterboxd export (watched, ratings, diary, reviews)

---

## Setup

```bash
pip install requests python-dotenv pandas tqdm
```

Create a `.env` file in the project root:

```
TMDB_API_KEY=your_key_here
```

Place your Letterboxd export files in the same folder (`watched.csv`, `ratings.csv`, `diary.csv`, `reviews.csv`), then run:

```bash
# Step 1 — enrich your watch history
python enrich_letterboxd.py

# Step 2 — generate scores and recommendations
python letterboxd_recommend.py
```

Import the output CSVs into Tableau:
- `letterboxd_enriched_scored.csv` → historical analysis panels
- `letterboxd_recommendations.csv` → recommendations panel

---

## Key Findings

- **925 films** watched across all time
- **668 rated** — avg rating of 3.34★
- **636 films** rated above the TMDB global average — consistent pattern of generosity toward films I choose to watch
- Top genres: Drama (210), Comedy (181), Action (87)
- Strongest divergence from TMDB consensus in Horror and Thriller

---

*Data extracted March 2026. TMDB catalog collected April 2026.*

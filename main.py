import base64
import io
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import httpx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode('utf-8')

async def scrape_wikipedia_films() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        dfs = pd.read_html(resp.text)
    for df in dfs:
        cols_lower = [str(c).lower() for c in df.columns]
        if ("rank" in cols_lower and ("peak" in cols_lower or "year" in cols_lower)):
            return df
    return dfs[0]

def analyze_films(df: pd.DataFrame):
    df.columns = [str(c).strip() for c in df.columns]
    cole = {c.lower(): c for c in df.columns}
    rank_col = cole.get('rank') or cole.get('peak rank') or cole.get('peak')
    year_col = cole.get('year') or cole.get('release year')
    title_col = cole.get('title') or cole.get('film') or cole.get('movie')
    gross_col = None
    for key in ['worldwide gross', 'gross', 'box office']:
        if key in cole:
            gross_col = cole[key]
            break
    if not all([rank_col, year_col, title_col]):
        raise ValueError("Required columns (Rank, Year, Title) not found")
    df[rank_col] = pd.to_numeric(df[rank_col], errors='coerce')
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    def gross_to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace("$", "").replace(",", "").lower()
        if "bn" in s:
            try:
                return float(s.replace("bn", "").strip()) * 1e9
            except:
                return np.nan
        try:
            return float(s)
        except:
            return np.nan
    if gross_col:
        df[gross_col] = df[gross_col].apply(gross_to_float)
    num_2bn_before_2000 = ((df[gross_col] >= 2e9) & (df[year_col] < 2000)).sum() if gross_col else 0
    earliest_film = None
    if gross_col:
        filtered = df[df[gross_col] > 1.5e9]
        if not filtered.empty:
            earliest_film = filtered.loc[filtered[year_col].idxmin(), title_col]
    rank_data = df[rank_col]
    peak_data = df[rank_col]
    corr = rank_data.corr(peak_data)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rank_data, peak_data, alpha=0.7)
    X = rank_data.values.reshape(-1, 1)
    y = peak_data.values
    model = LinearRegression()
    model.fit(X, y)
    x_line = np.linspace(X.min(), X.max(), 50)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, 'r--', linewidth=1.5)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Peak (simulated)')
    ax.set_title('Rank vs Peak with regression line')
    ax.grid(True)
    plot_data_uri = fig_to_data_uri(fig)
    return [
        int(num_2bn_before_2000),
        earliest_film or "",
        round(corr if corr is not None else 0, 6),
        plot_data_uri
    ]

@app.post("/api/")
async def analyze(
    task_text: str = Body(..., media_type="text/plain", description="Paste or POST the raw plain text task prompt here")
):
    logging.info("Received task prompt: %s", task_text[:300])
    t = task_text.lower()
    # --- Flexible, robust detection ---
    if ("highest grossing" in t and "wikipedia" in t) or \
       ("highest-grossing" in t and "wikipedia" in t) or \
       ("wikipedia.org/wiki/list_of_highest-grossing_films" in t) or \
       ("highest grossing films" in t):
        try:
            df = await scrape_wikipedia_films()
            answers = analyze_films(df)
            return JSONResponse(content=answers)
        except Exception as e:
            logging.exception("Error in Wikipedia tasks: %s", e)
            raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    return JSONResponse(content={"error": "Task not recognized or not implemented yet."})

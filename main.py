from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv  # Make sure you've installed: pip install python-dotenv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import io
import base64
import numpy as np  # Ensure numpy is imported
from scipy.stats import pearsonr, linregress
import re
from typing import List, Dict, Union, Any  # Added 'Any' for some generic types if needed, or remove if not used

# --- LLM Integration Placeholder (Example using OpenAI) ---
# If you plan to use an LLM for task interpretation, uncomment and configure:
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- DuckDB Integration Setup (Active) ---
import duckdb

def get_duckdb_connection():
    """Initializes and returns a DuckDB connection with httpfs and parquet extensions loaded."""
    conn = duckdb.connect(database=':memory:', read_only=False)
    conn.execute("INSTALL httpfs; LOAD httpfs;")  # For S3 access
    conn.execute("INSTALL parquet; LOAD parquet;")  # For Parquet files
    return conn

# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# Initialize FastAPI application
app = FastAPI()

# Define the request body model for the API
class DataAnalysisTask(BaseModel):
    task: str  # This field will receive the content of question.txt

# --- Helper Functions for Data Analysis Tasks ---

def _generate_plot_base64(
    df_plot: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    regression_line: bool = False,
    x_label: str = "",
    y_label: str = ""
) -> str:
    """
    Generates a scatter plot with an optional red dotted regression line and returns it
    as a base64 encoded PNG data URI, ensuring size is under 100KB.
    """
    img_base64 = ""
    if not df_plot.empty and len(df_plot) > 1:
        plt.figure(figsize=(6, 4))
        x_vals = df_plot[x_col].astype(float).to_numpy()
        y_vals = df_plot[y_col].astype(float).to_numpy()
        plt.scatter(x_vals, y_vals, alpha=0.7)
        plt.xlabel(x_label or x_col)
        plt.ylabel(y_label or y_col)
        plt.title(title)
        if regression_line:
            clean_x = df_plot[x_col].dropna().astype(float).to_numpy()
            clean_y = df_plot[y_col].dropna().astype(float).to_numpy()
            if len(clean_x) > 1 and len(clean_x) == len(clean_y):
                slope, intercept, r_value, _, _ = linregress(clean_x, clean_y)
                regression_line_data = intercept + slope * x_vals
                plt.plot(x_vals, regression_line_data, "r:", 
                         label=f"R={r_value:.2f}" if x_col == "Rank" else f"Slope={slope:.2f}")
                plt.legend()
            else:
                print(f"Warning: Not enough valid data points for regression in plot '{title}'. Skipping regression line.")
        plt.grid(True, linestyle="--", alpha=0.6)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        buf.seek(0)
        img_base64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        if len(img_base64) > 100000:
            print(f"Warning: Plot size ({len(img_base64)} bytes) exceeds 100,000 bytes for '{title}'. Consider lower DPI or different format/compression.")
    return img_base64

def _handle_highest_grossing_films_task(task_description: str) -> List[Union[int, str, float, None]]:
    """
    Handles the task for scraping and analyzing the highest-grossing films from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    try:
        html = requests.get(url, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find('table', {'class': [
            'wikitable', 
            'sortable', 
            'plainrowheaders', 
            'sticky-header', 
            'jquery-tablesorter'
        ]})
        if table is None:
            table = soup.find('table', {'class': ['wikitable', 'sortable', 'plainrowheaders']})
            if table is None:
                raise ValueError("Could not find the highest grossing films table on Wikipedia. Table structure might have changed or required classes are different.")
        df = pd.read_html(str(table))[0]
        df.columns = [re.sub(r"\[\d+\]", "", str(col)).strip() for col in df.columns]
        if 'Title' in df.columns:
            df.rename(columns={"Title": "Film"}, inplace=True)
        df.rename(columns={"Worldwide gross": "Worldwide_gross", "Year": "Release_Year"}, inplace=True)
        df["Worldwide_gross"] = df["Worldwide_gross"].replace(r"[\$,]", "", regex=True)
        df["Worldwide_gross_numeric"] = pd.to_numeric(df["Worldwide_gross"], errors="coerce")
        df["Release_Year"] = pd.to_numeric(df["Release_Year"], errors="coerce")
        df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
        df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")
        df.dropna(subset=["Worldwide_gross_numeric", "Release_Year", "Rank", "Peak"], inplace=True)
        billion_2_before_2020 = int(df[(df["Worldwide_gross_numeric"] >= 2.0) & (df["Release_Year"] < 2020)].shape[0])
        over_1_5 = df[df["Worldwide_gross_numeric"] >= 1.5].sort_values("Release_Year")
        earliest = over_1_5.iloc[0]["Film"] if "Film" in over_1_5.columns and not over_1_5.empty else "N/A"
        correlation: Union[float, None] = None
        corr_df = df[["Rank", "Peak"]].dropna()
        if not corr_df.empty and len(corr_df) > 1:
            rank_np = corr_df["Rank"].astype(float).to_numpy()
            peak_np = corr_df["Peak"].astype(float).to_numpy()
            corr_val_tuple = pearsonr(rank_np, peak_np)
            correlation = round(float(corr_val_tuple[0]), 6)
        else:
            correlation = None 
        plot_img = _generate_plot_base64(
            corr_df, "Rank", "Peak", 
            "Rank vs Peak Grossing Films",
            regression_line=True,
            x_label="Rank",
            y_label="Peak"
        )
        return [billion_2_before_2020, earliest, correlation, plot_img]
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Network error during web scraping: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing film data: {e}")

def _handle_indian_high_court_task(task_description: str) -> Dict[str, Union[str, float]]:
    """
    Handles the task for analyzing the Indian High Court Judgement dataset using DuckDB and S3.
    """
    try:
        conn = get_duckdb_connection()
        # Query 1: High court with most cases (2019-2022)
        sql_most_cases = """
        SELECT court, COUNT(*) as case_count
        FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
        WHERE year >= 2019 AND year <= 2022
        GROUP BY court
        ORDER BY case_count DESC
        LIMIT 1;
        """
        most_cases_result = conn.execute(sql_most_cases).fetchone()
        most_cases_court = most_cases_result[0] if most_cases_result else "N/A"
        # Query 2 & 3: Regression and plotting (e.g. for court 33_10)
        sql_delay_data = """
        SELECT year, date_of_registration, decision_date
        FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=33_10/bench=*/metadata.parquet?s3_region=ap-south-1')
        WHERE year >= 2019 AND year <= 2023
        ;
        """
        delay_df = conn.execute(sql_delay_data).fetchdf()
        delay_df['date_of_registration'] = pd.to_datetime(delay_df['date_of_registration'], errors='coerce', dayfirst=True)
        delay_df['decision_date'] = pd.to_datetime(delay_df['decision_date'], errors='coerce')
        delay_df['delay_days'] = (delay_df['decision_date'] - delay_df['date_of_registration']).dt.days
        delay_df.dropna(subset=['year', 'delay_days'], inplace=True)
        regression_slope_val = None
        if not delay_df.empty and len(delay_df) > 1:
            slope, _, _, _, _ = linregress(delay_df['year'].to_numpy(dtype=float), delay_df['delay_days'].to_numpy(dtype=float))
            regression_slope_val = round(float(slope), 6)
        img_base64_court_plot_val = _generate_plot_base64(
            delay_df, 'year', 'delay_days',
            'Year vs Days of Delay (Court 33_10)',
            regression_line=True, x_label='Year', y_label='# of Days of Delay'
        )
        conn.close()
        return {
            "Which high court disposed the most cases from 2019 - 2022?": most_cases_court,
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": regression_slope_val,
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": img_base64_court_plot_val
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing high court data: {str(e)}")

def _interpret_task_with_llm(task_description: str) -> str:
    """
    (Placeholder) This function would use an LLM to interpret the incoming task
    and determine which specific data analysis function needs to be called.
    For now, it uses simple keyword matching.
    """
    # --- LLM INTEGRATION (COMMENTED OUT; NOT NEEDED) ---
    # See the placeholder in your original code.
    # Active: simple keyword matching
    task_lower = task_description.lower()
    if "highest grossing films" in task_lower:
        return "HANDLE_FILM_SCRAPING"
    elif "indian high court" in task_lower and "judgement" in task_lower:
        return "HANDLE_HIGH_COURT_DATA"
    else:
        return "UNKNOWN_TASK"

@app.post("/api/")
async def analyze_data(task_payload: DataAnalysisTask):
    """
    Main API endpoint for the Data Analyst Agent.
    Interprets the incoming data analysis task and dispatches it
    to the appropriate specialized handler function.
    """
    task_description = task_payload.task.strip()
    interpreted_command = _interpret_task_with_llm(task_description)
    if interpreted_command == "HANDLE_FILM_SCRAPING":
        return _handle_highest_grossing_films_task(task_description)
    elif interpreted_command == "HANDLE_HIGH_COURT_DATA":
        return _handle_indian_high_court_task(task_description)
    else:
        raise HTTPException(status_code=400, detail="Unknown data analysis task provided. The agent could not interpret your request.")

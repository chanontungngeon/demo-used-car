import pandas as pd
import numpy as np
from typing import Optional

class DataHandler:
    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        """Load and cache the dataset"""
        df = pd.read_csv(path)
        df = DataHandler.clean_data(df)
        return df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the dataset"""
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Clean numeric columns
        numeric_columns = ['Mileage', 'SoldPrice', 'YearDatePriced', 'Year', 'MSRP']
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Drop rows with missing values in key columns
        df_clean = df_clean.dropna(subset=['Mileage', 'Age', 'SoldPrice'])
        
        return df_clean

    @staticmethod
    def filter_data(df: pd.DataFrame, make: str, model: str, trim: str, year: int) -> pd.DataFrame:
        """Filter data based on selections"""
        mask = (
            (df.Model_I == model) &
            (df.Version_Name_jato == trim) &
            (df.Year == year)
        )
        return df[mask].copy()
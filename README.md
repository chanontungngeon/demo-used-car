# RV Value Prediction App

A Streamlit application for predicting residual values of vehicles.

## Setup

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone <your-repo-url>
cd rv-prediction-app
```

3. Install dependencies:
```bash
poetry install
```

4. Run the application:
```bash
poetry run streamlit run app/main.py
```

## Features

- Interactive vehicle selection filters
- Residual value visualization
- Trend analysis with prediction intervals
- Detailed data view

## Project Structure

```
rv-prediction-app/
├── app/                # Application code
│   ├── main.py        # Main application file
│   ├── data_handler.py # Data processing
│   └── utils.py       # Utility functions
├── .gitignore         # Git ignore file
├── README.md          # Project documentation
├── pyproject.toml     # Poetry configuration
└── requirements.txt   # Python dependencies
```
# Customer Persona Segmenter

An unsupervised machine learning project that groups customers into meaningful segments using K-Means clustering. The application helps explore customer behavior and derive data-driven customer personas.

## Overview

The Customer Persona Segmenter analyzes customer data to identify groups with similar characteristics. By applying unsupervised learning, it supports customer segmentation and helps businesses better understand different customer groups.

## Features

- **Data Preprocessing:** Cleans and prepares customer data for clustering.
- **K-Means Clustering:** Groups customers based on selected features.
- **Feature Scaling:** Uses a scaler to prepare numerical features for clustering.
- **Cluster Analysis:** Examines cluster characteristics to understand customer groups.
- **Customer Persona Generation:** Helps summarize the characteristics of each segment.
- **REST API:** Provides backend endpoints for interacting with the clustering model.
- **Visualization Support:** Enables cluster exploration through a dashboard or visualization layer, if configured.

## Tech Stack

- **Programming Language:** Python
- **Backend Framework:** FastAPI
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Clustering Algorithm:** K-Means
- **Model Storage:** Pickle (`.pkl`)
- **API Testing:** Swagger UI / Postman
- **Dashboard:** Streamlit, if included in the application

## My Role — Backend Developer

As a Backend Developer, I contributed to building the FastAPI backend and integrating the unsupervised machine learning model.

- Developed backend API endpoints using FastAPI.
- Integrated the trained K-Means clustering model.
- Connected the preprocessing and feature-scaling components to the prediction workflow.
- Implemented the backend workflow for assigning customer records to clusters.
- Supported the delivery of clustering results to the application interface.
- Worked with saved model artifacts and API testing.

## Project Workflow

1. Load and inspect the customer dataset.
2. Clean the data and prepare the relevant features.
3. Scale the features as required.
4. Train the K-Means clustering model.
5. Analyze the resulting clusters and identify their characteristics.
6. Save the trained model and preprocessing components.
7. Integrate the model with FastAPI.
8. Return cluster assignments through the API and use the results to support customer persona analysis.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version
- pip
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/nethanreddyp/Customer-Persona-Segmenter.git
cd Customer-Persona-Segmenter
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run the Backend

If the backend entry point is `main_unsupervised.py` and the FastAPI instance is named `app`, run:

```bash
python -m uvicorn main_unsupervised:app --reload --port 8000
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

**Note:** Check the actual filenames and application configuration in the repository before running these commands.

## Applications

- Customer segmentation and behavioral analysis
- Customer group identification
- Marketing strategy support
- Data-driven customer persona development
- Exploration of unsupervised machine learning techniques

## Future Enhancements

- Add interactive cluster visualizations.
- Improve persona descriptions using cluster statistics.
- Add a dedicated dashboard for customer insights.
- Deploy the API and dashboard.
- Evaluate different clustering configurations.

## Disclaimer

This project is intended for educational and demonstration purposes. Customer segments are analytical groupings and should be interpreted in the context of the dataset and business requirements.

## Author

**Nethan Reddy**

GitHub: [@nethanreddyp](https://github.com/nethanreddyp)

## Acknowledgements

Developed as part of an AI/ML training project, with collaborative contributions across data preparation, machine learning, backend integration, and customer segment analysis.

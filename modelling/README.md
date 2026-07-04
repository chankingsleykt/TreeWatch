# 🌳 About
Welcome to ForestWatch! This project predicts upcoming deforestation events by training ML models on geospatial data from the Earth Engine Data Catalog. Soon to be an app where users can interact with a map to select regions and make live deforestation predictions.

# 📋 Stages
- [x] Data Collection: Gather and preprocess datasets with the Earth Engine API
- [x] Model Evaluation: Train three models on the three types of forest and evaluate out-of-sample performance
- [ ] (In progress) Application Development: Deploy a web application serving predictions on current data

# 📊 Data Sources
- [Hansen Global Forest Change](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12#dois): Forest Loss by years 2000-2024. Source of labels (was this pixel deforested in 20XX?)
- [USGS Landsat 8 Level 2, Collection 2, Tier 1](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2): Spectral image bands recorded by the Landsat 8 Satellite. Source of features

# 🚀 Installation and Usage
1. Download the libraries in requirements.txt
2. Check out classical_ml.ipynb to compare models on the datasets I extracted. If you'd like to explore how I gathered the data, check out data_collection.ipynb, though you'll need a Google Cloud Project ID to run the code

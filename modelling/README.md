# 🌳 About
Welcome to TreeWatch! This project detects deforestation events by training ML models on geospatial data from the Earth Engine Data Catalog. The best ML model is deployed in a web application where users can interact with a map to select regions and make live deforestation predictions.

# 📋 Stages
- [x] Data Collection: Gather and preprocess datasets with the Earth Engine API
- [x] Model Evaluation: Train three models on the three types of forest and evaluate out-of-sample performance
- [x] Application Development: Deploy a web application serving predictions on current data

# 📊 Data Sources
- [Hansen Global Forest Change](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13): Forest Loss by years 2000-2025. Source of labels (was this pixel deforested in 20XX?)
- [USGS Landsat 8 Level 2, Collection 2, Tier 1](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2): Spectral image bands recorded by the Landsat 8 Satellite. Source of features

# 🚀 Installation and Usage

## App 
Check it out here! https://treewatch.onrender.com
Note that as I used the Free tier, the app will spin down after 15 minutes of inactivity. You may have to wait for around one minute for the app to start up.

## Model Evaluation
1. Download the libraries in requirements.txt
2. Check out classical_ml.ipynb to compare models on the datasets I extracted. If you'd like to explore how I gathered the data, check out data_collection.ipynb, though you'll need a Google Cloud Project ID to run the code

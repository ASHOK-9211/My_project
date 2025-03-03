import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import os

# Load Data
destinations = pd.read_csv("dataset/Destinations.csv")
users = pd.read_csv("dataset/Users.csv")

# Trim whitespace from column names
destinations.columns = destinations.columns.str.strip()
users.columns = users.columns.str.strip()

# Validate Required Columns
required_columns = {
    "Users.csv": ["TravelPreferences", "UserID"],
    "Destinations.csv": ["Name", "Category", "State", "BestTimeToVisit", "PopularityScore"]
}


# Clean and prepare data
destinations['Category'] = destinations['Category'].str.strip()
users['TravelPreferences'] = users['TravelPreferences'].str.strip()

# Create label encoder for travel preferences
le_preferences = LabelEncoder()
unique_preferences = sorted(list(set([
    pref.strip() 
    for pref in destinations['Category'].str.split(',').explode().unique()
])))
le_preferences.fit(unique_preferences)

# Create feature matrix for destinations
vectorizer = TfidfVectorizer(stop_words='english')
destination_features = vectorizer.fit_transform(
    destinations['Name'] + ' ' + 
    destinations['Category'] + ' ' + 
    destinations['State']
)

# Calculate content similarity
content_similarity = cosine_similarity(destination_features)

# Create category matching matrix
def get_category_match_score(dest_category, preference):
    """Calculate how well a preference matches destination categories"""
    dest_cats = [cat.strip().lower() for cat in dest_category.split(',')]
    return 1.0 if preference.lower() in dest_cats else 0.0

category_match_matrix = np.zeros((len(destinations), len(unique_preferences)))
for i, dest in destinations.iterrows():
    for j, pref in enumerate(unique_preferences):
        category_match_matrix[i, j] = get_category_match_score(dest['Category'], pref)

# Normalize matrices
def normalize_matrix(matrix):
    """Normalize matrix values to [0, 1] range"""
    min_val = matrix.min()
    max_val = matrix.max()
    if max_val - min_val > 0:
        return (matrix - min_val) / (max_val - min_val)
    return matrix

content_similarity = normalize_matrix(content_similarity)
category_match_matrix = normalize_matrix(category_match_matrix)

# Combine into hybrid similarity
hybrid_similarity = 0.6 * content_similarity + 0.4 * np.dot(category_match_matrix, category_match_matrix.T)
hybrid_similarity = normalize_matrix(hybrid_similarity)

# Store index mapping and models
destination_indices = pd.Series(destinations.index, index=destinations['Name']).to_dict()

# Save all necessary data
model_dir = "models/"
os.makedirs(model_dir, exist_ok=True)

models_to_save = {
    "destination_indices.pkl": destination_indices,
    "hybrid_similarity.pkl": hybrid_similarity,
}

for filename, model in models_to_save.items():
    with open(os.path.join(model_dir, filename), "wb") as f:
        pickle.dump(model, f)

print("✅ Model trained and Saved Successfully!")
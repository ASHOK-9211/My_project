from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import json

app = Flask(__name__)

# Load data
try:
    destinations_df = pd.read_csv("dataset/Destinations.csv")
    users_df = pd.read_csv("dataset/Users.csv")
    
    # Clean column names by stripping whitespace
    destinations_df.columns = destinations_df.columns.str.strip()
    users_df.columns = users_df.columns.str.strip()
    
    # Add popularity score if not exists
    if 'PopularityScore' not in destinations_df.columns:
        destinations_df['PopularityScore'] = np.random.uniform(0.1, 1.0, len(destinations_df))
    
    # Process categories and preferences
    mlb = MultiLabelBinarizer()
    dest_categories = destinations_df['Category'].str.split(',').map(lambda x: [i.strip() for i in x])
    dest_encoded = mlb.fit_transform(dest_categories)
    
except Exception as e:
    print(f"Error loading data: {e}")
    raise

@app.route('/')
def home():
    try:
        all_preferences = set()
        for categories in destinations_df['Category'].dropna():
            all_preferences.update(cat.strip() for cat in categories.split(','))
        
        # Convert UserID to string in the dictionary
        users_list = users_df.to_dict('records')
        for user in users_list:
            user['UserID'] = str(user['UserID'])
        
        return render_template('index.html',
                             destinations=destinations_df.to_dict('records'),
                             users=users_list,
                             preferences=sorted(all_preferences),
                             states=sorted(destinations_df['State'].unique()))
    
    except Exception as e:
        print(f"Error in home route: {e}")
        return "An error occurred", 500

@app.route('/get_destination_details')
def get_destination_details():
    try:
        name = request.args.get('name')
        if not name:
            return jsonify({'error': 'Destination name is required'}), 400
            
        dest = destinations_df[destinations_df['Name'] == name]
        
        if dest.empty:
            return jsonify({'error': 'Destination not found'}), 404
            
        dest_data = dest.iloc[0]
        return jsonify({
            'District': str(dest_data['District']),
            'State': str(dest_data['State']),
            'Category': str(dest_data['Category']),
            'BestTimeToVisit': str(dest_data['BestTimeToVisit'])
        })
        
    except Exception as e:
        print(f"Error getting destination details: {e}")
        return jsonify({'error': 'Failed to retrieve destination details'}), 500

@app.route('/get_user_details')
def get_user_details():
    try:
        user_id = request.args.get('id')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Convert user_id to string for comparison
        user = users_df[users_df['UserID'].astype(str) == str(user_id)]
        
        if user.empty:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user.iloc[0]
        return jsonify({
            'Name': str(user_data['Name']),
            'Gender': str(user_data['Gender']),
            'Location': str(user_data['Location']),
            'TravelPreferences': str(user_data['TravelPreferences']),
            'NumberOfAdults': int(user_data['Number of Adults']),  # Ensure this matches the CSV column name
            'NumberOfChildren': int(user_data['Number of Children'])  # Ensure this matches the CSV column name
        })
        
    except Exception as e:
        print(f"Error getting user details: {e}")
        return jsonify({'error': 'Failed to retrieve user details'}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        search_type = data.get('type')
        
        if search_type == 'destination':
            destination = data.get('destination')
            if not destination:
                return jsonify({'error': 'Destination is required'}), 400
            recommendations = get_recommendations_by_destination(destination)
            
        elif search_type == 'user':
            user_id = data.get('userId')
            if not user_id:
                return jsonify({'error': 'User ID is required'}), 400
            recommendations = get_recommendations_by_user(user_id)
            
        elif search_type == 'custom':
            preferences = data.get('preferences')
            if not preferences:
                return jsonify({'error': 'Preferences are required'}), 400
            state = data.get('state')
            recommendations = get_custom_recommendations(preferences, state)
            
        else:
            return jsonify({'error': 'Invalid search type'}), 400
            
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)}), 500

def calculate_similarity_score(query_categories, destination_categories):
    try:
        query_cats = set(cat.strip().lower() for cat in query_categories.split(','))
        dest_cats = set(cat.strip().lower() for cat in destination_categories.split(','))
        
        if not query_cats:
            return 0
        
        common_cats = query_cats.intersection(dest_cats)
        return len(common_cats) / len(query_cats)
    except Exception as e:
        print(f"Error calculating similarity score: {e}")
        return 0

def get_recommendations_by_destination(destination_name):
    try:
        selected_dest = destinations_df[destinations_df['Name'] == destination_name]
        if selected_dest.empty:
            return []
            
        selected_dest = selected_dest.iloc[0]
        recommendations = []
        
        for _, dest in destinations_df.iterrows():
            if dest['Name'] != destination_name:
                score = calculate_similarity_score(selected_dest['Category'], dest['Category'])
                if score > 0:  # Only include if there's some similarity
                    recommendations.append({
                        'Name': str(dest['Name']),
                        'Category': str(dest['Category']),
                        'District': str(dest['District']),
                        'State': str(dest['State']),
                        'BestTimeToVisit': str(dest['BestTimeToVisit']),
                        'MatchScore': float(score),
                        'PopularityScore': float(dest['PopularityScore'])
                    })
        
        recommendations.sort(key=lambda x: (x['MatchScore'], x['PopularityScore']), reverse=True)
        return recommendations[:10]
    
    except Exception as e:
        print(f"Error in destination recommendations: {e}")
        return []

def get_recommendations_by_user(user_id):
    try:
        user = users_df[users_df['UserID'].astype(str) == str(user_id)]
        if user.empty:
            return []
            
        user = user.iloc[0]
        recommendations = []
        
        for _, dest in destinations_df.iterrows():
            score = calculate_similarity_score(user['TravelPreferences'], dest['Category'])
            if score > 0:  # Only include if there's some similarity
                recommendations.append({
                    'Name': str(dest['Name']),
                    'Category': str(dest['Category']),
                    'District': str(dest['District']),
                    'State': str(dest['State']),
                    'BestTimeToVisit': str(dest['BestTimeToVisit']),
                    'MatchScore': float(score),
                    'PopularityScore': float(dest['PopularityScore'])
                })
        
        recommendations.sort(key=lambda x: (x['MatchScore'], x['PopularityScore']), reverse=True)
        return recommendations[:10]
    
    except Exception as e:
        print(f"Error in user recommendations: {e}")
        return []

def get_custom_recommendations(preferences, state=None):
    try:
        recommendations = []
        filtered_df = destinations_df
        
        if state:
            filtered_df = destinations_df[destinations_df['State'] == state]
        
        for _, dest in filtered_df.iterrows():
            score = calculate_similarity_score(preferences, dest['Category'])
            if score > 0:  # Only include if there's some similarity
                recommendations.append({
                    'Name': str(dest['Name']),
                    'Category': str(dest['Category']),
                    'District': str(dest['District']),
                    'State': str(dest['State']),
                    'BestTimeToVisit': str(dest['BestTimeToVisit']),
                    'MatchScore': float(score),
                    'PopularityScore': float(dest['PopularityScore'])
                })
        
        recommendations.sort(key=lambda x: (x['MatchScore'], x['PopularityScore']), reverse=True)
        return recommendations[:10]
    
    except Exception as e:
        print(f"Error in custom recommendations: {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True, port=5000)
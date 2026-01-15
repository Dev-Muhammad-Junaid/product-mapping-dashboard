"""
Initialize sample data for the Nav App
Run this script to populate the app with example destinations and lists
"""

import json
import os
import uuid
from datetime import datetime

DATA_DIR = 'data'
DESTINATIONS_FILE = os.path.join(DATA_DIR, 'destinations.json')
LISTS_FILE = os.path.join(DATA_DIR, 'lists.json')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Sample destinations matching the screenshot
sample_destinations = [
    {
        'id': str(uuid.uuid4()),
        'name': 'Work',
        'emoji': '💼',
        'color': '#FF6B6B',
        'x': 55,
        'y': 20,
        'category': 'work',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Home',
        'emoji': '🏠',
        'color': '#45B7D1',
        'x': 50,
        'y': 35,
        'category': 'general',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'School',
        'emoji': '🏫',
        'color': '#FFA07A',
        'x': 70,
        'y': 35,
        'category': 'general',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Train',
        'emoji': '🚆',
        'color': '#FFA07A',
        'x': 35,
        'y': 40,
        'category': 'travel',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Airport',
        'emoji': '✈️',
        'color': '#4ECDC4',
        'x': 25,
        'y': 52,
        'category': 'travel',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Grocery',
        'emoji': '🏪',
        'color': '#98D8C8',
        'x': 30,
        'y': 65,
        'category': 'shopping',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Gym',
        'emoji': '🏋️',
        'color': '#F7DC6F',
        'x': 38,
        'y': 78,
        'category': 'leisure',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Drama',
        'emoji': '🎭',
        'color': '#98D8C8',
        'x': 50,
        'y': 82,
        'category': 'leisure',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Church',
        'emoji': '⛪',
        'color': '#BB8FCE',
        'x': 65,
        'y': 78,
        'category': 'general',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Soccer',
        'emoji': '⚽',
        'color': '#85929E',
        'x': 75,
        'y': 65,
        'category': 'leisure',
        'created_at': datetime.now().isoformat()
    }
]

# Sample lists
sample_lists = [
    {
        'id': str(uuid.uuid4()),
        'name': 'Best Coffee Shops in Town',
        'description': 'A curated list of the best places to get your caffeine fix',
        'category': 'food',
        'destinations': [],
        'shared': True,
        'creator': 'CoffeeEnthusiast',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Weekend Getaway Spots',
        'description': 'Perfect destinations for a quick weekend escape',
        'category': 'travel',
        'destinations': [],
        'shared': True,
        'creator': 'TravelBug',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Must-Visit Restaurants',
        'description': 'Top-rated restaurants you cannot miss',
        'category': 'food',
        'destinations': [],
        'shared': True,
        'creator': 'Foodie',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': str(uuid.uuid4()),
        'name': 'Family Entertainment',
        'description': 'Fun places to visit with the whole family',
        'category': 'entertainment',
        'destinations': [],
        'shared': True,
        'creator': 'FamilyFun',
        'created_at': datetime.now().isoformat()
    }
]

def initialize_sample_data():
    """Initialize the app with sample data"""
    
    # Save destinations
    with open(DESTINATIONS_FILE, 'w') as f:
        json.dump(sample_destinations, f, indent=2)
    print(f"✓ Created {len(sample_destinations)} sample destinations")
    
    # Save lists
    with open(LISTS_FILE, 'w') as f:
        json.dump(sample_lists, f, indent=2)
    print(f"✓ Created {len(sample_lists)} sample lists")
    
    print("\n✨ Sample data initialized successfully!")
    print("Run 'python3 app.py' to start the app")

if __name__ == '__main__':
    initialize_sample_data()

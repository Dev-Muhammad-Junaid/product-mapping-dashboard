from flask import Flask, render_template, jsonify
from datetime import datetime

app = Flask(__name__)

# Sample timeline data
TIMELINES = {
    'le-mans': {
        'id': 'le-mans',
        'title': 'Le Mans Racing Evolution',
        'description': 'The legendary 24 Hours of Le Mans race evolution from its inception to modern era',
        'category': 'Motorsport',
        'image': 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=800',
        'events': [
            {
                'year': 1923,
                'title': 'First Le Mans Race',
                'description': 'The inaugural 24 Hours of Le Mans race held on May 26-27. Won by André Lagache and René Léonard in a Chenard-Walcker.',
                'image': 'https://images.unsplash.com/photo-1566023888849-e87e069816e0?w=600',
                'tags': ['Origin', 'Historic']
            },
            {
                'year': 1935,
                'title': 'Speed Era Begins',
                'description': 'Introduction of streamlined racing cars. Speeds begin to exceed 100 mph on the Mulsanne Straight.',
                'image': 'https://images.unsplash.com/photo-1583267746897-c45e6d9d6e87?w=600',
                'tags': ['Innovation', 'Speed']
            },
            {
                'year': 1955,
                'title': 'The Tragic Accident',
                'description': 'The deadliest accident in motorsport history leads to major safety reforms across all racing.',
                'image': 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=600',
                'tags': ['Safety', 'Historic']
            },
            {
                'year': 1966,
                'title': 'Ford vs Ferrari',
                'description': 'Ford GT40 breaks Ferrari\'s dominance with its first victory, beginning a four-year winning streak.',
                'image': 'https://images.unsplash.com/photo-1580273916550-e323be2ae537?w=600',
                'tags': ['Competition', 'Legendary']
            },
            {
                'year': 1988,
                'title': 'Group C Golden Age',
                'description': 'Peak of Group C prototypes. Porsche 962, Jaguar XJR-9, and others produce legendary races.',
                'image': 'https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=600',
                'tags': ['Technology', 'Golden Age']
            },
            {
                'year': 1999,
                'title': 'Modern Safety Measures',
                'description': 'Major circuit modifications including chicanes on Mulsanne Straight to reduce speeds and improve safety.',
                'image': 'https://images.unsplash.com/photo-1540148426574-c47bfebd75e2?w=600',
                'tags': ['Safety', 'Modern']
            },
            {
                'year': 2012,
                'title': 'Hybrid Era Begins',
                'description': 'Introduction of hybrid powertrains in LMP1 class. Audi leads with e-tron quattro technology.',
                'image': 'https://images.unsplash.com/photo-1551830820-330a71b99659?w=600',
                'tags': ['Technology', 'Sustainability']
            },
            {
                'year': 2023,
                'title': 'Hypercar Revolution',
                'description': 'New Hypercar class brings manufacturers like Ferrari, Porsche, Toyota, and Cadillac back to Le Mans.',
                'image': 'https://images.unsplash.com/photo-1600712242805-5f78671b24da?w=600',
                'tags': ['Modern', 'Innovation']
            }
        ]
    },
    'mobile-devices': {
        'id': 'mobile-devices',
        'title': 'Mobile Device Evolution',
        'description': 'From the first telephone to modern smartphones - the complete journey of mobile communication',
        'category': 'Technology',
        'image': 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=800',
        'events': [
            {
                'year': 1876,
                'title': 'The First Telephone',
                'description': 'Alexander Graham Bell patents the telephone. First words transmitted: "Mr. Watson, come here, I want to see you."',
                'image': 'https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?w=600',
                'tags': ['Origin', 'Historic']
            },
            {
                'year': 1973,
                'title': 'First Mobile Phone Call',
                'description': 'Martin Cooper of Motorola makes the first handheld mobile phone call on a prototype DynaTAC.',
                'image': 'https://images.unsplash.com/photo-1605236453806-6ff36851218e?w=600',
                'tags': ['Innovation', 'Wireless']
            },
            {
                'year': 1983,
                'title': 'Motorola DynaTAC 8000X',
                'description': 'First commercially available handheld mobile phone. Weighed 2 pounds, cost $3,995, 30-minute battery life.',
                'image': 'https://images.unsplash.com/photo-1599999190850-8ea883d665f7?w=600',
                'tags': ['Commercial', 'Breakthrough']
            },
            {
                'year': 1992,
                'title': 'First SMS Message',
                'description': 'Neil Papworth sends the first SMS text message: "Merry Christmas" on the Vodafone network.',
                'image': 'https://images.unsplash.com/photo-1596526131083-e8c633c948d2?w=600',
                'tags': ['Communication', 'Innovation']
            },
            {
                'year': 1999,
                'title': 'BlackBerry Revolution',
                'description': 'BlackBerry 850 introduces mobile email. Physical keyboard and instant messaging change business communication.',
                'image': 'https://images.unsplash.com/photo-1519241047957-be31d7379a5d?w=600',
                'tags': ['Business', 'Email']
            },
            {
                'year': 2007,
                'title': 'iPhone Launch',
                'description': 'Steve Jobs unveils the iPhone. Multi-touch interface, mobile internet, and App Store ecosystem revolutionize smartphones.',
                'image': 'https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5?w=600',
                'tags': ['Revolutionary', 'Touchscreen']
            },
            {
                'year': 2008,
                'title': 'Android Emerges',
                'description': 'First Android phone (HTC Dream/G1) launches. Open-source platform enables diverse smartphone ecosystem.',
                'image': 'https://images.unsplash.com/photo-1607252650355-f7fd0460ccdb?w=600',
                'tags': ['Competition', 'Open Source']
            },
            {
                'year': 2010,
                'title': '4G LTE Era',
                'description': 'High-speed mobile internet enables streaming, video calls, and cloud services on mobile devices.',
                'image': 'https://images.unsplash.com/photo-1556656793-08538906a9f8?w=600',
                'tags': ['Speed', 'Internet']
            },
            {
                'year': 2017,
                'title': 'Edge-to-Edge Displays',
                'description': 'iPhone X and Samsung Galaxy S8 introduce nearly bezel-less designs with OLED displays and facial recognition.',
                'image': 'https://images.unsplash.com/photo-1556656793-08538906a9f8?w=600',
                'tags': ['Design', 'Innovation']
            },
            {
                'year': 2024,
                'title': 'AI-Powered Smartphones',
                'description': 'On-device AI, computational photography, real-time translation, and advanced voice assistants become standard.',
                'image': 'https://images.unsplash.com/photo-1592286927505-ed0baa7cac1c?w=600',
                'tags': ['AI', 'Modern']
            }
        ]
    }
}

@app.route('/')
def index():
    """Render the main timeline interface"""
    return render_template('index.html', timelines=TIMELINES)

@app.route('/api/timelines')
def get_timelines():
    """Get all available timelines"""
    return jsonify({
        'timelines': [
            {
                'id': t['id'],
                'title': t['title'],
                'description': t['description'],
                'category': t['category'],
                'image': t['image'],
                'eventCount': len(t['events']),
                'yearRange': {
                    'start': min(e['year'] for e in t['events']),
                    'end': max(e['year'] for e in t['events'])
                }
            }
            for t in TIMELINES.values()
        ]
    })

@app.route('/api/timeline/<timeline_id>')
def get_timeline(timeline_id):
    """Get detailed timeline data"""
    timeline = TIMELINES.get(timeline_id)
    if not timeline:
        return jsonify({'error': 'Timeline not found'}), 404
    return jsonify(timeline)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

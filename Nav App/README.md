# Here 2 There - Nav App

A beautiful destination exploration app that lets you visually plan your day, create routes, and share destination lists.

## Features

- 🗺️ **Interactive Destination Map**: Drag and drop destination bubbles on a beautiful canvas
- 🎯 **Route Planning**: Draw paths between locations to plan your journey
- 📝 **List Management**: Create and share curated lists of destinations (best foods, travel spots, etc.)
- 📅 **Day Planner**: Order your destinations to plan your day with drag-and-drop functionality
- 🎨 **Beautiful UI**: Modern, colorful interface inspired by iOS design
- 💾 **Save & Share**: Save your plans and share them with others

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Initialize with sample data:
```bash
python3 init_sample_data.py
```

3. Run the application:
```bash
python3 app.py
```

4. Open your browser and navigate to:
```
http://localhost:5001
```

That's it! You're ready to start exploring and planning your destinations.

## Usage

### Adding Destinations
- Click the "+" button to add a new destination
- Choose an emoji icon and color
- Drag the bubble to position it on the canvas

### Creating Routes
- Select a destination as your starting point
- Click on other destinations to create a path
- The app will draw a visual route between locations

### Managing Lists
- Create themed lists (e.g., "Best Coffee Shops", "Weekend Getaways")
- Add destinations to your lists
- Share lists with others

### Day Planning
- Switch to List View to see your destinations in order
- Drag to reorder destinations by priority
- Plan your day efficiently

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Storage**: JSON files
- **UI**: Custom CSS with modern design principles

## API Endpoints

- `GET/POST /api/destinations` - Manage destinations
- `GET/POST /api/routes` - Manage routes
- `GET/POST /api/lists` - Manage destination lists
- `GET/POST /api/planner` - Day planner functionality

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License

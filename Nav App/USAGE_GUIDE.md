# Here 2 There - Complete Usage Guide

## Overview

Here 2 There is an interactive destination exploration app that helps you visualize, plan, and organize your travels. Whether you're planning your daily commute, a weekend trip, or exploring new food spots, this app makes it fun and intuitive.

## Features at a Glance

### 🗺️ Map View
The heart of the app - an interactive canvas where you can:
- Add destination bubbles
- Drag and position them anywhere
- Create visual routes between locations
- See your entire journey at a glance

### 📋 List View (Day Planner)
Organize your destinations in a prioritized list:
- Drag to reorder destinations
- Number each stop in your journey
- Perfect for planning your day step-by-step

### 🌟 Shared Lists
Discover and share curated destination collections:
- Browse lists created by others
- Share your own favorite spots
- Categories: Food, Travel, Shopping, Entertainment

## Getting Started

### Initial Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Sample Data** (Optional but recommended)
   ```bash
   python3 init_sample_data.py
   ```
   This creates 10 sample destinations and 4 shared lists to help you get started.

3. **Launch the App**
   ```bash
   python3 app.py
   ```
   The app will be available at `http://localhost:5001`

## Using the Map View

### Adding a Destination

1. Click the **"➕ Add Destination"** button
2. Fill in the details:
   - **Name**: Give it a memorable name (e.g., "Morning Coffee Spot")
   - **Emoji**: Choose an icon that represents the location
   - **Color**: Pick a color for visual organization
   - **Category**: Select the type (Food, Travel, Work, etc.)
3. Click **"Add Destination"**

Your new destination appears on the canvas as a colorful bubble!

### Moving Destinations

- Simply **click and drag** any destination bubble
- Position it wherever makes sense on your map
- The app saves the position automatically

### Creating Routes

1. Click **"🛤️ Create Route"** button (it turns purple)
2. Click destinations in the order you want to visit them
3. Watch as animated paths appear connecting your stops
4. Click the button again to save your route

Routes show:
- Curved paths between locations
- Arrows indicating direction
- Animation to visualize the journey

### Clearing Routes

Click **"🗑️ Clear Routes"** to remove all route visualizations and start fresh.

## Using the Day Planner

### Accessing the Planner

Click the **"📋 List View"** tab at the top of the app.

### Organizing Your Day

1. See all your destinations listed in order
2. Each destination shows:
   - Position number
   - Emoji icon
   - Name
   - Category
3. **Drag and drop** items to reorder them
4. Your priorities are automatically saved

**Pro Tip**: Use the planner to:
- Order stops by importance
- Arrange locations geographically to minimize travel time
- Group similar activities together

## Creating and Sharing Lists

### Making a List

1. Switch to **"🌟 Shared Lists"** view
2. Click **"➕ Create New List"**
3. Fill in the details:
   - **Name**: e.g., "Best Brunch Spots"
   - **Description**: Tell others what makes this list special
   - **Category**: Choose the appropriate type
   - **Share**: Check the box to make it public
4. Click **"Create List"**

### List Ideas

- **Food & Dining**: Best pizza, coffee shops, late-night eats
- **Travel**: Weekend getaways, scenic drives, hidden gems
- **Shopping**: Vintage stores, farmer's markets, bookshops
- **Entertainment**: Live music venues, theaters, parks

## Tips & Best Practices

### Organizing Your Map

- **Use colors strategically**: Group related destinations with similar colors
- **Spread out bubbles**: Avoid overlap for a cleaner view
- **Arrange geographically**: Position destinations roughly where they are in real life

### Efficient Route Planning

- Start with your current location (the center blue bubble)
- Plan routes in a logical order to minimize backtracking
- Create multiple routes for different purposes (work commute, weekend fun, etc.)

### Managing Lists

- Be specific in descriptions to help others understand your recommendations
- Update lists as you discover new places
- Create themed lists for easy sharing

### Daily Workflow

1. **Morning**: Open the app and check your day planner
2. **Add new spots**: As you discover places, add them immediately
3. **Create routes**: Before heading out, plan your route visually
4. **Share**: After a great day, create a list to share your journey

## Keyboard Shortcuts

While the app is primarily mouse-driven, here are some tips:

- **Click once**: Select a destination for routes
- **Click and drag**: Move destinations
- **Double-click**: Quick view of destination details (coming soon!)

## Data Management

### Where Data is Stored

All your data is stored locally in JSON files within the `data/` directory:
- `destinations.json`: Your destination bubbles
- `lists.json`: Shared and personal lists
- `routes.json`: Saved routes
- `planner.json`: Day planner order

### Backing Up

To backup your data, simply copy the `data/` folder to a safe location.

### Resetting

To start fresh:
```bash
rm -rf data/
python3 init_sample_data.py  # Optional: reload sample data
```

## Troubleshooting

### App Won't Start

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that port 5001 isn't already in use
- Try: `python3 app.py` instead of `python app.py`

### Destinations Won't Move

- Make sure you're clicking and dragging the colored bubble, not the label
- Try refreshing the page
- Check browser console for errors (F12)

### Routes Not Appearing

- Ensure you've clicked at least 2 destinations
- Verify the "Create Route" button is purple (active)
- Try clearing routes and starting over

## Advanced Features (Coming Soon)

- Real map integration with actual locations
- Time estimates between destinations
- Weather integration for travel planning
- Mobile app version
- Collaborative planning with friends
- Export routes to GPS apps

## Privacy & Data

- All data is stored locally on your machine
- Shared lists are only shared within your local instance
- No data is sent to external servers
- You have complete control over your information

## Support & Feedback

Found a bug or have a feature request? 
- Check the GitHub repository
- Open an issue with details
- Contribute to the project!

## Credits

Inspired by the need to make destination planning fun, visual, and intuitive. Built with Flask, JavaScript, and a love for travel.

---

**Happy Exploring! 🗺️✨**

# Ingredient Mapping System

A local ingredient mapping system that automatically maps product ingredients to a standardized database using fuzzy matching algorithms. This system processes CSV data entirely offline without requiring external APIs.

## Features

- **Automatic Fuzzy Matching**: Uses RapidFuzz algorithms to match ingredient variations
- **Synonym Management**: Persistent synonym mapping with JSON storage
- **Manual Correction Interface**: Web-based UI for manual ingredient mapping
- **Confidence Scoring**: Configurable confidence thresholds for matching
- **Comprehensive Processing**: Handles complex ingredient strings with dosages and allergen info
- **Export Functionality**: Export mapped/unmapped results to CSV
- **Processing Logs**: Track all operations with detailed logging
- **Modern Web Interface**: Responsive dashboard built with Tailwind CSS

## Requirements

- Python 3.7+
- Flask 3.1.1
- pandas 2.2.3
- rapidfuzz 3.13.0

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv ingredient_mapper_env
source ingredient_mapper_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure Data Files

Make sure you have these CSV files in the project directory:
- `products_raw.csv` - Contains product data with ingredient information
- `ingredients_db.csv` - Master ingredient database with ID mappings

### 4. Run the Application

```bash
python app.py
```

Note: If port 5000 is in use (common on macOS due to AirPlay), run on a different port:

```bash
python -c "from app import app; app.run(debug=True, host='0.0.0.0', port=5001)"
```

### 5. Access the Web Interface

Open your browser and navigate to:
- `http://localhost:5000` (default)
- `http://localhost:5001` (if using alternate port)

## Usage Guide

### 1. Load Data
- Click "Load Data Files" to import your CSV files
- The system will show counts of products and ingredients loaded

### 2. Configure Settings
- Adjust the confidence threshold (60-95%) using the slider
- Higher values require more exact matches

### 3. Process Ingredients
- Click "Process Ingredients" to start the fuzzy matching
- The system will:
  - Normalize ingredient strings (remove dosages, allergens)
  - Apply synonym mappings
  - Perform fuzzy matching with confidence scoring
  - Generate comprehensive processing logs

### 4. Review Results
- **Mapping Results Tab**: View successfully mapped ingredients with confidence scores
- **Unmapped Ingredients Tab**: Manually map ingredients that couldn't be automatically matched
- **Processing Logs Tab**: Review detailed processing information

### 5. Manual Mapping
- Click "Map Manually" for unmapped ingredients
- Search the ingredient database in the modal
- Select the correct match to create permanent mappings

### 6. Export Results
- Click "Export Results" to generate CSV files:
  - `mapped_ingredients_[timestamp].csv` - Successfully mapped ingredients
  - `unmapped_ingredients_[timestamp].csv` - Ingredients requiring review
  - `processing_logs_[timestamp].csv` - Complete processing logs

## Data Format Requirements

### products_raw.csv
Expected columns:
- Product identifier columns
- `ai_ingredients_JSON_fixed` - JSON-like array of ingredient strings

### ingredients_db.csv
Required columns:
- `ingredient_id` - Unique identifier for each ingredient
- `name` - Standardized ingredient name

## Algorithm Details

### Ingredient Normalization
- Removes dosage patterns (e.g., "15 mg", "0.5%")
- Strips allergen information in parentheses
- Normalizes whitespace and punctuation
- Converts to lowercase for matching

### Matching Strategy
1. **Exact Match**: Direct database lookup
2. **Synonym Mapping**: Check pre-defined synonyms
3. **Fuzzy Matching**: RapidFuzz token_sort_ratio algorithm
4. **Confidence Scoring**: Configurable threshold filtering

### Synonym Management
- Automatically builds synonym mappings from manual corrections
- Persists to `ingredient_synonyms.json`
- Includes default mappings for common variations

## File Structure

```
Project Mapping System/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── products_raw.csv               # Product data (user-provided)
├── ingredients_db.csv             # Ingredient database (user-provided)
├── ingredient_synonyms.json       # Auto-generated synonym mappings
├── templates/
│   └── dashboard.html             # Web interface template
└── ingredient_mapper_env/         # Virtual environment
```

## API Endpoints

- `GET /` - Main dashboard
- `POST /api/load-data` - Load CSV files
- `POST /api/process` - Process ingredients with fuzzy matching
- `GET /api/results` - Get mapping results
- `POST /api/manual-map` - Manual ingredient mapping
- `POST /api/add-synonym` - Add synonym mappings
- `POST /api/export` - Export results to CSV
- `GET /api/ingredients/search` - Search ingredient database
- `GET/POST /api/settings` - Configure algorithm parameters

## Troubleshooting

### Port 5000 Already in Use
On macOS, disable AirPlay Receiver or use a different port:
```bash
python -c "from app import app; app.run(debug=True, host='0.0.0.0', port=5001)"
```

### Missing Dependencies
Ensure you're in the virtual environment and all packages are installed:
```bash
source ingredient_mapper_env/bin/activate
pip install -r requirements.txt
```

### Data Loading Issues
- Verify CSV files exist in the project directory
- Check that `ingredients_db.csv` has required columns: `ingredient_id`, `name`
- Ensure `products_raw.csv` has the `ai_ingredients_JSON_fixed` column

### Performance Issues
- For large datasets, consider increasing the confidence threshold
- Use processing logs to identify bottlenecks
- Monitor memory usage during processing

## License

This project is designed for local ingredient mapping tasks and includes comprehensive fuzzy matching capabilities for standardizing ingredient databases. 
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response
from rapidfuzz import fuzz, process
import json
import re
from datetime import datetime
import logging
from collections import defaultdict, Counter
import queue
import threading

# Add file upload support
from werkzeug.utils import secure_filename
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'csv'}

# Global queue for real-time log streaming
log_queue = queue.Queue()

# Custom logging handler to capture logs for streaming
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'level': record.levelname,
            'message': record.getMessage(),
            'type': record.levelname.lower()
        }
        self.log_queue.put(log_entry)

# Add the queue handler to the app logger
queue_handler = QueueHandler(log_queue)
app.logger.addHandler(queue_handler)
app.logger.setLevel(logging.INFO)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv_structure(df, file_type):
    """Validate CSV structure based on file type"""
    if file_type == 'products':
        # Check for required columns (flexible column naming)
        required_cols = ['product_name', 'ingredients']
        actual_cols = df.columns.str.lower()
        
        # Try to find columns that might contain product and ingredient data
        product_col = None
        ingredient_col = None
        
        for col in actual_cols:
            if any(keyword in col for keyword in ['product', 'name', 'title']):
                product_col = col
            if any(keyword in col for keyword in ['ingredient', 'composition', 'content']):
                ingredient_col = col
        
        if product_col is None or ingredient_col is None:
            return False, "Products CSV must contain columns for product name and ingredients"
        
        return True, {"product_col": product_col, "ingredient_col": ingredient_col}
        
    elif file_type == 'ingredients':
        # Check for ingredient database structure
        required_cols = ['ingredient_id', 'name']
        actual_cols = df.columns.str.lower()
        
        id_col = None
        name_col = None
        
        for col in actual_cols:
            if any(keyword in col for keyword in ['id', 'identifier']):
                id_col = col
            if any(keyword in col for keyword in ['name', 'ingredient', 'title']):
                name_col = col
        
        if id_col is None or name_col is None:
            return False, "Ingredients CSV must contain columns for ID and name"
        
        return True, {"id_col": id_col, "name_col": name_col}
    
    return False, "Unknown file type"

class IngredientMapper:
    def __init__(self):
        self.products_df = None
        self.ingredients_db = None
        self.mapping_results = []
        self.unmapped_ingredients = []
        self.processing_logs = []
        self.confidence_threshold = 85
        self.synonym_map = {}
        
        # Column mapping for flexible CSV structure
        self.product_col_mapping = None
        self.ingredient_col_mapping = None
        
        # File paths for uploaded files
        self.products_file_path = None
        self.ingredients_file_path = None
        
        self.load_synonyms()
        
        # Enhanced debugging tracking
        self.parsing_stats = defaultdict(int)
        self.confidence_distribution = []
        self.unmatched_patterns = Counter()
        self.database_coverage_analysis = {}
        
    def load_synonyms(self):
        """Load or create synonym mapping"""
        try:
            if os.path.exists('ingredient_synonyms.json'):
                with open('ingredient_synonyms.json', 'r') as f:
                    self.synonym_map = json.load(f)
        except:
            pass
        
        # Enhanced default synonyms for supplements
        self.synonym_map = {
            "ascorbic acid": ["vitamin c", "l-ascorbate", "vitamin c (ascorbic acid)", "l-ascorbic acid"],
            "niacin": ["vitamin b3", "nicotinic acid", "nicotinamide", "niacinamide"],
            "whey protein isolate": ["whey protein isolate (milk)", "whey isolate", "hydrolyzed whey protein isolate"],
            "whey protein concentrate": ["whey protein concentrate (milk)", "whey concentrate", "whey protein concentrate (milk)"],
            "soy lecithin": ["lecithin (soy)", "soy lecithin (emulsifier)", "emulsifier (soy lecithin)"],
            "sunflower lecithin": ["lecithin (sunflower)", "sunflower lecithin (emulsifier)", "emulsifier (sunflower lecithin)"],
            "magnesium oxide": ["magnesium (as magnesium oxide)", "heavy magnesium oxide", "magnesium oxide (magnesium)"],
            "calcium carbonate": ["calcium (as calcium carbonate)", "calcium carbonate (calcium)"],
            "sucralose": ["sweetener (sucralose)", "artificial sweetener sucralose", "sweetener 955"],
            "acesulfame potassium": ["acesulfame-potassium", "sweetener (acesulfame potassium)", "acesulfame potassium"],
            "thiamine": ["vitamin b1", "thiamine hydrochloride", "thiamine nitrate", "thiamin"],
            "riboflavin": ["vitamin b2", "riboflavin sodium phosphate"],
            "pyridoxine": ["vitamin b6", "pyridoxine hydrochloride"],
            "cyanocobalamin": ["vitamin b12", "cobalamin"],
            "folic acid": ["folate", "vitamin b9"],
            "biotin": ["vitamin h", "vitamin b7"],
            "colecalciferol": ["vitamin d3", "cholecalciferol"],
            "tocopherol": ["vitamin e", "dl-alpha-tocopherol", "alpha-tocopherol"],
            "calcium": ["calcium citrate", "calcium phosphate", "calcium carbonate"],
            "magnesium": ["magnesium citrate", "magnesium chloride", "magnesium carbonate hydrate"],
            "zinc": ["zinc oxide", "zinc sulfate", "zinc citrate"],
            "iron": ["ferrous fumarate", "ferric pyrophosphate", "iron amino acid chelate"]
        }
    
    def save_synonyms(self):
        """Save current synonym mapping"""
        with open('ingredient_synonyms.json', 'w') as f:
            json.dump(self.synonym_map, f, indent=2)
    
    def analyze_ingredient_complexity(self, ingredient_str):
        """Analyze and log complexity patterns in ingredient strings"""
        if pd.isna(ingredient_str):
            self.parsing_stats['empty_ingredients'] += 1
            return "empty"
        
        ingredient_str = str(ingredient_str).strip()
        
        # Check for various complexity patterns
        complexity_flags = []
        
        if re.search(r'\d+\.?\d*\s*(mg|g|micrograms?|iu|mcg)\b', ingredient_str, re.IGNORECASE):
            complexity_flags.append('has_dosage')
            self.parsing_stats['ingredients_with_dosage'] += 1
            
        if re.search(r'\([^)]+\)', ingredient_str):
            complexity_flags.append('has_parentheses')
            self.parsing_stats['ingredients_with_parentheses'] += 1
            
        if re.search(r'allergens?:', ingredient_str, re.IGNORECASE):
            complexity_flags.append('has_allergens')
            self.parsing_stats['ingredients_with_allergens'] += 1
            
        if re.search(r'equiv\.?|equivalent', ingredient_str, re.IGNORECASE):
            complexity_flags.append('has_equivalents')
            self.parsing_stats['ingredients_with_equivalents'] += 1
            
        if re.search(r'[,;]\s*', ingredient_str):
            complexity_flags.append('compound_ingredient')
            self.parsing_stats['compound_ingredients'] += 1
            
        if len(ingredient_str) > 50:
            complexity_flags.append('long_string')
            self.parsing_stats['long_ingredient_strings'] += 1
            
        return complexity_flags if complexity_flags else ['simple']
    
    def normalize_ingredient(self, ingredient_str):
        """Enhanced normalize ingredient string for better matching"""
        if pd.isna(ingredient_str):
            return ""
        
        # Analyze complexity before normalization
        complexity = self.analyze_ingredient_complexity(ingredient_str)
        
        # Convert to string and lowercase
        normalized = str(ingredient_str).lower().strip()
        
        # Enhanced patterns to remove - more comprehensive for supplements
        patterns_to_remove = [
            r'\([^)]*\d+\.?\d*\s*(mg|g|micrograms?|iu|mcg)\)',  # Remove dosage in parentheses
            r'\d+\.?\d*\s*(mg|g|micrograms?|iu|mcg)\b',  # Remove standalone dosages
            r'allergens?:.*$',  # Remove allergen information
            r'traces?:.*$',     # Remove trace information
            r'contains?:.*$',   # Remove contains information
            r'equiv\.?.*?(?=,|$)',  # Remove equivalent information
            r'\(as [^)]+\)',    # Remove "as compound" specifications
            r'\([^)]*%\)',      # Remove percentage specifications
            r'processed\s+with\s+[^,]+',  # Remove processing descriptions
            r'standardised\s+to\s+contain[^,]+',  # Remove standardization info
            r'from\s+\d+\.?\d*\s*(mg|g)\s+dry[^,]*',  # Remove extraction ratios
        ]
        
        original_normalized = normalized
        for pattern in patterns_to_remove:
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Log if significant changes were made
        if len(original_normalized) - len(normalized) > 20:
            self.parsing_stats['heavily_normalized'] += 1
        
        # Clean up extra whitespace and punctuation
        normalized = re.sub(r'[,;]+', ',', normalized)  # Normalize separators
        normalized = re.sub(r'\s+', ' ', normalized)    # Normalize whitespace
        normalized = normalized.strip(' ,-.()')
        
        # Log very short results that might indicate over-normalization
        if len(normalized) < 3 and len(original_normalized) > 10:
            self.parsing_stats['over_normalized'] += 1
            logger.warning(f"Possible over-normalization: '{str(ingredient_str)[:50]}...' -> '{normalized}'")
        
        return normalized
    
    def extract_individual_ingredients(self, ingredient_list):
        """Enhanced extract individual ingredients from complex strings"""
        if pd.isna(ingredient_list) or not ingredient_list:
            self.parsing_stats['empty_ingredient_lists'] += 1
            return []
        
        # Handle JSON-like arrays with better error handling
        if isinstance(ingredient_list, str) and ingredient_list.startswith('['):
            try:
                # Try multiple parsing approaches
                parsed = None
                
                # First try direct JSON parsing
                try:
                    parsed = json.loads(ingredient_list.replace("'", '"'))
                except:
                    # Try fixing common JSON issues
                    fixed = ingredient_list.replace("'", '"').replace('",]', '"]').replace(',]', ']')
                    parsed = json.loads(fixed)
                
                if isinstance(parsed, list):
                    ingredient_list = parsed
                    self.parsing_stats['successful_json_parse'] += 1
                else:
                    self.parsing_stats['json_parse_not_list'] += 1
                    
            except Exception as e:
                self.parsing_stats['failed_json_parse'] += 1
                logger.warning(f"JSON parsing failed for: {ingredient_list[:100]}... Error: {str(e)}")
                # Fall back to string splitting
                ingredient_list = ingredient_list.strip('[]').replace("'", "").split(',')
        
        # If it's already a list
        if isinstance(ingredient_list, list):
            ingredients = ingredient_list
            self.parsing_stats['list_ingredients'] += 1
        else:
            # Split by common separators with enhanced patterns
            split_patterns = [r'[,;]\s*', r'\n', r'(?<=[a-z])\s+(?=[A-Z])']
            ingredients = [ingredient_list]
            
            for pattern in split_patterns:
                new_ingredients = []
                for ing in ingredients:
                    new_ingredients.extend(re.split(pattern, str(ing)))
                ingredients = new_ingredients
            
            self.parsing_stats['string_split_ingredients'] += 1
        
        # Process each ingredient with enhanced validation
        processed = []
        for ing in ingredients:
            if pd.isna(ing) or not str(ing).strip():
                continue
                
            original = str(ing).strip()
            
            # Skip obvious non-ingredients
            if re.match(r'^\d+\.?\d*\s*(mg|g|micrograms?|iu|mcg)$', original, re.IGNORECASE):
                self.parsing_stats['skipped_dosage_only'] += 1
                continue
                
            if re.match(r'^allergens?:', original, re.IGNORECASE):
                self.parsing_stats['skipped_allergen_labels'] += 1
                continue
            
            normalized = self.normalize_ingredient(original)
            
            # Enhanced length validation
            if normalized and len(normalized) > 2:
                processed.append({
                    'original': original,
                    'normalized': normalized,
                    'complexity': self.analyze_ingredient_complexity(original)
                })
                self.parsing_stats['successfully_processed'] += 1
            else:
                self.parsing_stats['too_short_after_normalization'] += 1
        
        return processed
    
    def analyze_database_coverage(self):
        """Analyze what types of ingredients are in our database"""
        if self.ingredients_db is None:
            return
        
        ingredient_types = {
            'vitamins': 0,
            'minerals': 0,
            'proteins': 0,
            'sweeteners': 0,
            'emulsifiers': 0,
            'flavors': 0,
            'preservatives': 0,
            'water_analysis': 0,
            'other': 0
        }
        
        vitamin_keywords = ['vitamin', 'ascorbic', 'thiamine', 'riboflavin', 'niacin', 'pyridoxine', 'cobalamin', 'folic', 'biotin', 'tocopherol', 'colecalciferol']
        mineral_keywords = ['calcium', 'magnesium', 'iron', 'zinc', 'potassium', 'sodium', 'phosphate', 'chloride', 'sulfate']
        protein_keywords = ['protein', 'whey', 'casein', 'isolate', 'concentrate']
        sweetener_keywords = ['sucralose', 'aspartame', 'acesulfame', 'stevia', 'sweetener']
        emulsifier_keywords = ['lecithin', 'emulsifier', 'carrageenan']
        water_keywords = ['chlorine', 'fluoride', 'bromate', 'trihalomethanes', 'haloacetic', 'pfas', 'benzene', 'asbestos']
        
        for _, row in self.ingredients_db.iterrows():
            name = row['name'].lower()
            
            if any(keyword in name for keyword in vitamin_keywords):
                ingredient_types['vitamins'] += 1
            elif any(keyword in name for keyword in mineral_keywords):
                ingredient_types['minerals'] += 1
            elif any(keyword in name for keyword in protein_keywords):
                ingredient_types['proteins'] += 1
            elif any(keyword in name for keyword in sweetener_keywords):
                ingredient_types['sweeteners'] += 1
            elif any(keyword in name for keyword in emulsifier_keywords):
                ingredient_types['emulsifiers'] += 1
            elif any(keyword in name for keyword in water_keywords):
                ingredient_types['water_analysis'] += 1
            else:
                ingredient_types['other'] += 1
        
        self.database_coverage_analysis = ingredient_types
        logger.info(f"Database coverage analysis: {ingredient_types}")
    
    def find_best_match(self, ingredient_normalized, original_ingredient):
        """Enhanced find best match with detailed logging and alternatives"""
        if not ingredient_normalized:
            return None, 0, "Empty ingredient", []
        
        # Check synonyms first with enhanced logging
        for standard_name, synonyms in self.synonym_map.items():
            if ingredient_normalized == standard_name or ingredient_normalized in synonyms:
                # Find exact match in database
                db_match = self.ingredients_db[
                    self.ingredients_db['name'].str.lower() == standard_name
                ]
                if not db_match.empty:
                    logger.debug(f"Synonym match: '{original_ingredient}' -> '{standard_name}'")
                    return db_match.iloc[0], 100, "Synonym match", []
                else:
                    logger.warning(f"Synonym '{standard_name}' not found in database for '{original_ingredient}'")
        
        # Fuzzy match against database with enhanced scoring
        ingredient_names = self.ingredients_db['name'].tolist()
        
        # Try multiple scoring methods and collect all results
        scorers = [
            ('token_sort_ratio', fuzz.token_sort_ratio),
            ('token_set_ratio', fuzz.token_set_ratio), 
            ('partial_ratio', fuzz.partial_ratio),
            ('ratio', fuzz.ratio)
        ]
        
        all_matches = []
        best_match = None
        best_score = 0
        best_scorer = None
        
        for scorer_name, scorer in scorers:
            # Get top 5 matches for this scorer
            matches = process.extract(
                ingredient_normalized,
                ingredient_names,
                scorer=scorer,
                limit=5
            )
            
            for match_name, score, _ in matches:
                all_matches.append({
                    'name': match_name,
                    'score': score,
                    'scorer': scorer_name
                })
                
                if score > best_score:
                    best_match = (match_name, score, _)
                    best_score = score
                    best_scorer = scorer_name
        
        # Sort all matches by score and remove duplicates while keeping best scorer info
        unique_matches = {}
        for match in all_matches:
            name = match['name']
            if name not in unique_matches or match['score'] > unique_matches[name]['score']:
                unique_matches[name] = match
        
        # Get top 5 unique alternatives sorted by score
        alternatives = sorted(unique_matches.values(), key=lambda x: x['score'], reverse=True)[:5]
        
        if best_match:
            matched_name, score, _ = best_match
            matched_row = self.ingredients_db[
                self.ingredients_db['name'] == matched_name
            ].iloc[0]
            
            # Log confidence distribution for analysis
            self.confidence_distribution.append(score)
            
            # Enhanced logging with alternatives
            alt_info = []
            for alt in alternatives[:3]:  # Show top 3 alternatives in logs
                if alt['name'] != matched_name:  # Don't repeat the best match
                    alt_info.append(f"{alt['name']} ({alt['score']:.1f}% via {alt['scorer']})")
            
            alt_text = f" | Alternatives: {', '.join(alt_info)}" if alt_info else ""
            logger.debug(f"Fuzzy match ({best_scorer}): '{original_ingredient}' -> '{matched_name}' (score: {score}){alt_text}")
            
            return matched_row, score, f"Fuzzy match ({best_scorer})", alternatives
        
        # Track unmatched patterns for analysis
        self.unmatched_patterns[ingredient_normalized] += 1
        logger.debug(f"No match found for: '{original_ingredient}' (normalized: '{ingredient_normalized}')")
        return None, 0, "No match found", []
    
    def safe_get_str(self, val, default=''):
        import pandas as pd
        if isinstance(val, pd.Series):
            val = val.dropna()
            if not val.empty:
                return str(val.iloc[0])
            else:
                return default
        if pd.isna(val):
            return default
        return str(val)
    
    def process_products(self):
        """Process all products and map ingredients"""
        if self.products_df is None or self.ingredients_db is None:
            app.logger.error("❌ No data loaded. Please load CSV files first.")
            return
        
        self.mapping_results = []
        self.unmapped_ingredients = []
        self.processing_logs = []
        
        total_products = len(self.products_df)
        app.logger.info(f"🚀 Starting ingredient mapping for {total_products} products...")
        
        for idx, row in self.products_df.iterrows():
            product_name = self.safe_get_str(row.get('product_name', f'Product {idx}'), f'Product {idx}')
            product_company = self.safe_get_str(row.get('product_company', ''))
            ingredient_column = 'ingredients'
            
            app.logger.info(f"📦 Processing product {idx+1}/{total_products}: {product_name}")
            
            if ingredient_column not in row or pd.isna(row[ingredient_column]):
                app.logger.warning(f"⚠️ No ingredients found for {product_name}")
                self.processing_logs.append({
                    'product': product_name,
                    'message': 'No ingredients found',
                    'type': 'warning'
                })
                continue
            
            # Extract individual ingredients
            app.logger.info(f"🔍 Extracting ingredients from {product_name}...")
            ingredients = self.extract_individual_ingredients(row[ingredient_column])
            
            product_mappings = []
            product_unmapped = []
            
            total_ingredients = len(ingredients)
            app.logger.info(f"📋 Found {total_ingredients} ingredients to process")
            
            for ing_idx, ing_data in enumerate(ingredients):
                original = ing_data['original']
                normalized = ing_data['normalized']
                
                app.logger.info(f"  🧪 [{ing_idx+1}/{total_ingredients}] Processing: '{original[:50]}{'...' if len(original) > 50 else ''}'")
                
                # Find best match
                match_result, confidence, match_type, alternatives = self.find_best_match(normalized, original)
                
                mapping_entry = {
                    'product_name': product_name,
                    'product_company': product_company,
                    'original_ingredient': original,
                    'normalized_ingredient': normalized,
                    'confidence': confidence,
                    'match_type': match_type,
                    'alternatives': alternatives,  # Store alternatives for later use
                    'timestamp': datetime.now().isoformat()
                }
                
                if match_result is not None and confidence >= self.confidence_threshold:
                    mapping_entry.update({
                        'ingredient_id': match_result['ingredient_id'],
                        'matched_name': match_result['name'],
                        'status': 'mapped'
                    })
                    product_mappings.append(mapping_entry)
                    
                    # Enhanced success logging with alternatives
                    alt_summary = ""
                    if alternatives and len(alternatives) > 1:
                        # Show top 2 alternatives (excluding the matched one)
                        other_alts = [alt for alt in alternatives if alt['name'] != match_result['name']][:2]
                        if other_alts:
                            alt_texts = [f"{alt['name']} ({alt['score']:.1f}%)" for alt in other_alts]
                            alt_summary = f" | Other options: {', '.join(alt_texts)}"
                    
                    app.logger.info(f"    ✅ MAPPED: '{original}' → '{match_result['name']}' ({confidence:.1f}% confidence){alt_summary}")
                    
                else:
                    mapping_entry.update({
                        'ingredient_id': None,
                        'matched_name': match_result['name'] if match_result is not None else '',
                        'status': 'unmapped'
                    })
                    product_unmapped.append(mapping_entry)
                    
                    if match_result is not None:
                        # Enhanced failure logging with top alternatives
                        alt_summary = ""
                        if alternatives:
                            # Show top 3 alternatives for failed matches
                            alt_texts = [f"{alt['name']} ({alt['score']:.1f}% via {alt['scorer']})" for alt in alternatives[:3]]
                            alt_summary = f" | Consider: {', '.join(alt_texts)}"
                        
                        app.logger.warning(f"    ❌ LOW CONFIDENCE: '{original}' → '{match_result['name']}' ({confidence:.1f}% < {self.confidence_threshold}%){alt_summary}")
                    else:
                        app.logger.warning(f"    ❌ NO MATCH: '{original}' (normalized: '{normalized}')")
                        
                        # For completely unmatched ingredients, try to suggest some possibilities
                        if alternatives:
                            top_alts = alternatives[:3]
                            alt_texts = [f"{alt['name']} ({alt['score']:.1f}%)" for alt in top_alts]
                            app.logger.info(f"      💡 SUGGESTIONS: {', '.join(alt_texts)}")
            
            self.mapping_results.extend(product_mappings)
            self.unmapped_ingredients.extend(product_unmapped)
            
            # Log processing summary
            mapped_count = len(product_mappings)
            unmapped_count = len(product_unmapped)
            total_ing = mapped_count + unmapped_count
            success_rate = (mapped_count / total_ing * 100) if total_ing > 0 else 0
            
            app.logger.info(f"📊 {product_name} SUMMARY: {mapped_count} mapped, {unmapped_count} unmapped ({success_rate:.1f}% success rate)")
            
            self.processing_logs.append({
                'product': product_name,
                'message': f'Processed {mapped_count + unmapped_count} ingredients. Mapped: {mapped_count}, Unmapped: {unmapped_count}',
                'type': 'info'
            })
        
        # Final summary
        total_mapped = len(self.mapping_results)
        total_unmapped = len(self.unmapped_ingredients)
        overall_total = total_mapped + total_unmapped
        overall_success = (total_mapped / overall_total * 100) if overall_total > 0 else 0
        
        app.logger.info(f"🎉 PROCESSING COMPLETE!")
        app.logger.info(f"📈 FINAL RESULTS: {total_mapped} mapped, {total_unmapped} unmapped ({overall_success:.1f}% overall success rate)")
        app.logger.info(f"🏁 Processed {total_products} products with {overall_total} total ingredients")
    
    def load_data(self, products_file=None, ingredients_file=None):
        """Load CSV data files (from uploads or default paths)"""
        try:
            # Load products file
            if products_file:
                self.products_df = pd.read_csv(products_file)
                self.products_file_path = products_file
            else:
                self.products_df = pd.read_csv('products_raw.csv')
                self.products_file_path = 'products_raw.csv'
            
            # Validate and map product columns
            valid, result = validate_csv_structure(self.products_df, 'products')
            if not valid:
                self.processing_logs.append({
                    'product': 'System',
                    'message': f'Products file validation failed: {result}',
                    'type': 'error'
                })
                return False
            
            self.product_col_mapping = result
            
            # Standardize column names for processing
            original_cols = self.products_df.columns.tolist()
            for original_col in original_cols:
                if original_col.lower() == self.product_col_mapping['product_col']:
                    self.products_df = self.products_df.rename(columns={original_col: 'product_name'})
                elif original_col.lower() == self.product_col_mapping['ingredient_col']:
                    self.products_df = self.products_df.rename(columns={original_col: 'ingredients'})
            
            # Load ingredients database
            if ingredients_file:
                self.ingredients_db = pd.read_csv(ingredients_file)
                self.ingredients_file_path = ingredients_file
            else:
                self.ingredients_db = pd.read_csv('ingredients_db.csv')
                self.ingredients_file_path = 'ingredients_db.csv'
            
            # Validate and map ingredient columns
            valid, result = validate_csv_structure(self.ingredients_db, 'ingredients')
            if not valid:
                self.processing_logs.append({
                    'product': 'System',
                    'message': f'Ingredients file validation failed: {result}',
                    'type': 'error'
                })
                return False
            
            self.ingredient_col_mapping = result
            
            # Standardize column names for processing
            original_cols = self.ingredients_db.columns.tolist()
            for original_col in original_cols:
                if original_col.lower() == self.ingredient_col_mapping['id_col']:
                    self.ingredients_db = self.ingredients_db.rename(columns={original_col: 'ingredient_id'})
                elif original_col.lower() == self.ingredient_col_mapping['name_col']:
                    self.ingredients_db = self.ingredients_db.rename(columns={original_col: 'name'})
            
            # Normalize ingredient names in database for better matching
            self.ingredients_db['name_normalized'] = self.ingredients_db['name'].str.lower().str.strip()
            
            self.processing_logs.append({
                'product': 'System',
                'message': f'Loaded {len(self.products_df)} products and {len(self.ingredients_db)} ingredients from database',
                'type': 'success'
            })
            return True
        except Exception as e:
            self.processing_logs.append({
                'product': 'System',
                'message': f'Error loading data: {str(e)}',
                'type': 'error'
            })
            return False

    def reset(self):
        self.products_df = None
        self.ingredients_db = None
        self.mapping_results = []
        self.unmapped_ingredients = []
        self.processing_logs = []
        self.product_col_mapping = None
        self.ingredient_col_mapping = None
        self.products_file_path = None
        self.ingredients_file_path = None
        self.confidence_threshold = 85
        self.synonym_map = {}
        self.load_synonyms()
        self.parsing_stats = defaultdict(int)
        self.confidence_distribution = []
        self.unmatched_patterns = Counter()
        self.database_coverage_analysis = {}

# Initialize mapper
mapper = IngredientMapper()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data from CSV files (uploaded or default)"""
    products_file = None
    ingredients_file = None
    
    # Check if this is a JSON request with file paths
    if request.is_json:
        data = request.get_json()
        products_file = data.get('products_file')
        ingredients_file = data.get('ingredients_file')
    else:
        # Check if files were uploaded in this request
        if 'products_file' in request.files:
            file = request.files['products_file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"products_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
                file.save(filepath)
                products_file = filepath
        
        if 'ingredients_file' in request.files:
            file = request.files['ingredients_file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"ingredients_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
                file.save(filepath)
                ingredients_file = filepath

    success = mapper.load_data(products_file, ingredients_file)
    return jsonify({
        'success': success,
        'products_count': len(mapper.products_df) if mapper.products_df is not None else 0,
        'ingredients_count': len(mapper.ingredients_db) if mapper.ingredients_db is not None else 0,
        'logs': mapper.processing_logs[-10:]  # Last 10 logs
    })

@app.route('/api/process', methods=['POST'])
def process_ingredients():
    """Process ingredients with fuzzy matching"""
    data = request.get_json()
    mapper.confidence_threshold = data.get('confidence_threshold', 85)
    
    mapper.process_products()
    
    return jsonify({
        'success': True,
        'mapped_count': len(mapper.mapping_results),
        'unmapped_count': len(mapper.unmapped_ingredients),
        'logs': mapper.processing_logs[-20:]  # Last 20 logs
    })

@app.route('/api/results')
def get_results():
    """Get mapping results"""
    # Convert data to JSON-serializable format
    def convert_to_json_serializable(data):
        import pandas as pd
        import numpy as np
        try:
            print(f'[DEBUG] Serializing type: {type(data)}, value: {str(data)[:200]}')
            if isinstance(data, list):
                return [convert_to_json_serializable(item) for item in data]
            elif isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    try:
                        result[key] = convert_to_json_serializable(value)
                    except Exception as e:
                        print(f'[SERIALIZATION ERROR] Key: {key}, Value: {value}, Error: {e}')
                        result[key] = str(value)
                return result
            elif isinstance(data, pd.Series):
                if len(data) == 1:
                    return convert_to_json_serializable(data.iloc[0])
                else:
                    return data.tolist()
            elif isinstance(data, pd.DataFrame):
                return data.to_dict(orient='records')
            elif hasattr(data, 'item'):
                try:
                    return data.item()
                except Exception as e:
                    print(f'[DEBUG] .item() failed: {e}, fallback to str')
                    return str(data)
            elif isinstance(data, (np.generic,)):
                return data.item()
            else:
                return data
        except Exception as e:
            print(f'[SERIALIZATION FATAL ERROR] {e} for data: {str(data)[:200]}')
            return str(data)

    # Log the lengths and samples
    print(f'[RESULTS API] mapping_results length: {len(mapper.mapping_results)}')
    print(f'[RESULTS API] unmapped_ingredients length: {len(mapper.unmapped_ingredients)}')
    if mapper.mapping_results:
        print(f'[RESULTS API] mapping_results sample: {mapper.mapping_results[0]}')
    if mapper.unmapped_ingredients:
        print(f'[RESULTS API] unmapped_ingredients sample: {mapper.unmapped_ingredients[0]}')

    try:
        mapped = convert_to_json_serializable(mapper.mapping_results)
        unmapped = convert_to_json_serializable(mapper.unmapped_ingredients)
        return jsonify({
            'mapped': mapped,
            'unmapped': unmapped
        })
    except Exception as e:
        print(f'[RESULTS API ERROR] {e}')
        return jsonify({
            'mapped': [],
            'unmapped': [],
            'error': str(e)
        }), 500

@app.route('/api/manual-map', methods=['POST'])
def manual_map():
    """Manually map an unmapped ingredient"""
    data = request.get_json()
    original_ingredient = data['original_ingredient']
    ingredient_id = data['ingredient_id']
    
    # Find the ingredient in database
    matched_ingredient = mapper.ingredients_db[
        mapper.ingredients_db['ingredient_id'] == ingredient_id
    ]
    
    if matched_ingredient.empty:
        return jsonify({'success': False, 'error': 'Ingredient ID not found'})
    
    matched_ingredient = matched_ingredient.iloc[0]
    
    # Update unmapped list
    for i, item in enumerate(mapper.unmapped_ingredients):
        if item['original_ingredient'] == original_ingredient:
            # Move to mapped
            mapping_entry = item.copy()
            mapping_entry.update({
                'ingredient_id': ingredient_id,
                'matched_name': matched_ingredient['name'],
                'status': 'mapped',
                'confidence': 100,
                'match_type': 'manual'
            })
            mapper.mapping_results.append(mapping_entry)
            mapper.unmapped_ingredients.pop(i)
            break
    
    return jsonify({'success': True})

@app.route('/api/add-synonym', methods=['POST'])
def add_synonym():
    """Add a synonym mapping"""
    data = request.get_json()
    standard_name = data['standard_name'].lower()
    synonym = data['synonym'].lower()
    
    if standard_name not in mapper.synonym_map:
        mapper.synonym_map[standard_name] = []
    
    if synonym not in mapper.synonym_map[standard_name]:
        mapper.synonym_map[standard_name].append(synonym)
        mapper.save_synonyms()
    
    return jsonify({'success': True})

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export mapping results to CSV"""
    try:
        # Create final mapped products DataFrame
        mapped_df = pd.DataFrame(mapper.mapping_results)
        unmapped_df = pd.DataFrame(mapper.unmapped_ingredients)
        
        # Export mapped results
        mapped_filename = f'mapped_ingredients_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        mapped_df.to_csv(mapped_filename, index=False)
        
        # Export unmapped for review
        unmapped_filename = f'unmapped_ingredients_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        unmapped_df.to_csv(unmapped_filename, index=False)
        
        # Export processing logs
        logs_filename = f'processing_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        logs_df = pd.DataFrame(mapper.processing_logs)
        logs_df.to_csv(logs_filename, index=False)
        
        return jsonify({
            'success': True,
            'files': {
                'mapped': mapped_filename,
                'unmapped': unmapped_filename,
                'logs': logs_filename
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ingredients/search')
def search_ingredients():
    """Search ingredients database"""
    query = request.args.get('q', '').lower()
    
    if not query or len(query) < 2:
        return jsonify([])
    
    if mapper.ingredients_db is None:
        return jsonify([])
    
    # Search in ingredient names
    matches = mapper.ingredients_db[
        mapper.ingredients_db['name'].str.lower().str.contains(query, na=False)
    ].head(20)
    
    return jsonify([{
        'id': row['ingredient_id'],
        'name': row['name']
    } for _, row in matches.iterrows()])

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update settings"""
    if request.method == 'POST':
        data = request.get_json()
        mapper.confidence_threshold = data.get('confidence_threshold', 85)
        return jsonify({'success': True})
    
    return jsonify({
        'confidence_threshold': mapper.confidence_threshold,
        'synonyms': mapper.synonym_map
    })

@app.route('/api/logs')
def get_logs():
    """Get real-time logs"""
    def generate():
        while True:
            log_entry = log_queue.get()
            yield f"data: {json.dumps(log_entry)}\n\n"
    
    return Response(generate(), content_type='text/event-stream')

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Upload and validate a CSV file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    file_type = request.form.get('type')  # 'products' or 'ingredients'
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Only CSV files are allowed'})
    
    try:
        # Read file into pandas DataFrame for validation
        df = pd.read_csv(file)
        
        # Validate structure
        valid, result = validate_csv_structure(df, file_type)
        if not valid:
            return jsonify({'success': False, 'error': result})
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_type}_{timestamp}_{filename}")
        file.seek(0)  # Reset file pointer
        file.save(filepath)
        
        # Return preview data
        preview_data = {
            'filename': file.filename,
            'filepath': filepath,
            'rows': len(df),
            'columns': df.columns.tolist(),
            'column_mapping': result,
            'sample_data': df.head(5).to_dict('records')
        }
        
        return jsonify({'success': True, 'preview': preview_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'})

@app.route('/api/preview-file', methods=['POST'])
def preview_file():
    """Preview uploaded file data"""
    data = request.get_json()
    filepath = data.get('filepath')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})
    
    try:
        df = pd.read_csv(filepath)
        
        # Get basic stats
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'sample_data': df.head(10).to_dict('records'),
            'column_types': df.dtypes.astype(str).to_dict(),
            'null_counts': df.isnull().sum().to_dict()
        }
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error reading file: {str(e)}'})

@app.route('/api/reset', methods=['POST'])
def reset_all():
    """Reset all loaded data and state"""
    mapper.reset()
    return jsonify({'success': True})

@app.route('/api/preview-mapped', methods=['GET'])
def preview_mapped():
    """Preview mapped results and available fields"""
    mapped_df = pd.DataFrame(mapper.mapping_results)
    preview = mapped_df.head(100).to_dict(orient='records')
    fields = list(mapped_df.columns)
    return jsonify({'preview': preview, 'fields': fields})

@app.route('/api/export-custom', methods=['POST'])
def export_custom():
    """Export mapped results with only selected fields as CSV"""
    data = request.get_json()
    fields = data.get('fields', [])
    filename = data.get('filename', None)
    mapped_df = pd.DataFrame(mapper.mapping_results)
    if not fields:
        return jsonify({'success': False, 'error': 'No fields selected'}), 400
    try:
        export_df = mapped_df[fields]
        if filename:
            if not filename.endswith('.csv'):
                filename += '.csv'
        else:
            filename = f'custom_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        export_df.to_csv(filename, index=False)
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 
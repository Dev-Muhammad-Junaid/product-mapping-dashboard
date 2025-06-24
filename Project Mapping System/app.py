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
        self.confidence_threshold = 70  # Lowered from 85 to improve matching
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
        
        # Add custom pattern matching system
        self.custom_patterns = {}
        self.exact_mappings = {}
        self.pattern_rules = {}
        self.load_custom_patterns()
        
        # Improved scoring weights based on analysis
        self.scorer_weights = {
            'exact_match': 100,
            'custom_pattern': 99,
            'partial_ratio': 0.4,      # Higher weight - better for compound ingredients
            'token_sort_ratio': 0.25,  # Lower weight - poor performance on compounds
            'token_set_ratio': 0.25,   # Lower weight - similar issues
            'ratio': 0.1               # Lowest weight - too strict
        }
        
    def safe_get_str(self, value, default=""):
        """Safely convert value to string with fallback"""
        try:
            if pd.isna(value) or value is None:
                return default
            return str(value).strip()
        except Exception:
            return default
        
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
    
    def load_custom_patterns(self):
        """Load comprehensive custom pattern mappings based on CSV analysis"""
        
        # Load from file if exists
        try:
            if os.path.exists('custom_patterns.json'):
                with open('custom_patterns.json', 'r') as f:
                    data = json.load(f)
                    self.exact_mappings = data.get('exact_mappings', {})
                    self.pattern_rules = data.get('pattern_rules', {})
                    logger.info(f"Loaded {len(self.exact_mappings)} exact mappings and {len(self.pattern_rules)} pattern rules")
                    return
        except Exception as e:
            logger.warning(f"Could not load custom patterns: {e}")
        
        # EXACT MAPPINGS - High confidence direct matches based on your analysis
        self.exact_mappings = {
            # CRITICAL FIX: Protein Blend should map to ID 108 (exact match)
            "protein blend": 108,  # Protein Blend (exact match)
            
            # Specific Protein Blends - map to the detailed one when context suggests it
            "whey protein blend": 163,  # "Protein blend (whey, casein, milk protein concentrate)"
            "micro filtered whey protein blend": 163,
            "micro-filtered whey protein blend": 163,
            "vitalstrength protein blend": 163,
            "musashi high protein blend": 163,
            "athena protein blend": 163,
            "burn protein blend": 163,
            "milk protein blend": 163,
            "premium protein blend": 163,
            "genetix nutrition proprietary protein blend": 163,
            "anabolix proprietary protein blend": 163,
            
            # CRITICAL FIX: Whey Protein Issues - Your main complaint
            "whey protein isolate milk": 801,  # Whey Protein Isolate (NOT soy protein!)
            "whey protein concentrate milk": 229,  # Whey Protein Concentrate
            "hydrolysed whey protein isolate": 806,  # Hydrolysed Whey Protein Isolate
            "hydrolyzed whey protein isolate": 830,  # Hydrolyzed Whey Protein Isolate
            "ultra filtered whey protein isolate": 853,  # Ultra Filtered Whey Protein Isolate (WPI)
            "ultra filtered whey protein concentrate": 854,  # Ultra Filtered Whey Protein Concentrate (WPC)
            "micro-filtered whey protein isolate": 857,  # Micro-filtered Whey Protein Isolate
            "micro-filtered whey protein concentrate": 858,  # Micro-filtered Whey Protein Concentrate
            "grass feed protein whey": 801,  # Should map to Whey Protein Isolate
            "whey protein isolate": 801,  # Direct mapping
            "whey protein concentrate": 229,  # Direct mapping
            "whey protein": 104,  # Generic Whey Protein
            
            # Plant Protein Blends
            "plant protein blend": 108,  # Generic Protein Blend for now
            "organic yellow pea protein": 257,  # Pumpkin seed protein (closest match)
            "organic brown rice protein": 258,  # Organic sunflower seed protein (closest)
            "faba bean protein": 259,  # Alfalfa protein (closest legume)
            
            # Specific Compound Issues from your examples
            "protein water blend": 163,  # Should be protein blend
            "isolate protein": 801,  # Whey Protein Isolate
            "peptide blend": 801,  # Whey Protein Isolate (peptides are processed whey)
            
            # Lecithin Issues
            "soy lecithin": 46,  # Soy Lecithin (assuming this ID exists)
            "sunflower lecithin": 46,  # Map to Soy Lecithin for now
            "lecithin": 46,  # Soy Lecithin
            
            # Milk Issues
            "milk": 247,  # Non-Fat Milk Powder
            "whole milk powder": 247,
            "skim milk powder": 247,
            "buttermilk powder": 247,
            
            # Casein Issues
            "casein": 251,  # Calcium Caseinate (Milk Derivative)
            "calcium caseinate": 251,
            "sodium caseinate": 252,  # Sodium Caseinate (Milk Derivative)
            "micellar casein": 251,  # Map to calcium caseinate
            "milk protein concentrate": 251,  # Map to calcium caseinate
            "milk protein isolate": 251,
            
            # Common Mismatches from analysis
            "flavour": 266,  # Strawberry flavoring (generic)
            "flavours": 266,
            "natural flavour": 269,  # Organic and Natural Flavors
            "natural flavours": 269,
            "artificial flavors": 266,
            "flavoring": 266,
            "natural flavouring": 269,
            
            # Salt Issues
            "salt": 19,  # Salt (assuming this ID)
            "sodium chloride": 19,
            "himalayan pink salt": 19,
            
            # Sweetener Issues
            "sweeteners": 955,  # Sucralose (most common)
            "natural sweetener": 955,
            "natural sweeteners": 955,
            "sweetener": 955,
            "sucralose": 955,
            "acesulfame potassium": 950,
            
            # Vitamin Issues
            "vitamin c": 238,  # Ascorbic Acid
            "ascorbic acid": 238,
            
            # Emulsifier Issues
            "emulsifier": 46,  # Soy Lecithin (most common emulsifier)
            "emulsifiers": 46,
        }
        
        # PATTERN RULES - Regex-based intelligent matching
        self.pattern_rules = {
            # Protein Type Patterns - Most Important
            r".*whey.*protein.*isolate.*": 801,  # Any whey protein isolate variant
            r".*whey.*protein.*concentrate.*": 229,  # Any whey protein concentrate variant
            r".*whey.*protein.*(?!isolate|concentrate)": 104,  # Generic whey protein
            r".*protein.*blend.*whey.*": 163,  # Whey-containing protein blends
            r".*protein.*blend.*": 108,  # Generic protein blend (your fix!)
            r".*casein.*": 251,  # Any casein variant
            r".*milk.*protein.*": 251,  # Milk protein variants
            
            # Vitamin Patterns
            r".*vitamin.*c.*": 238,  # Ascorbic Acid
            r".*ascorbic.*acid.*": 238,
            r".*vitamin.*e.*": 240,  # Vitamin E Acetate
            
            # Sweetener Patterns
            r".*sucralose.*": 955,  # Sucralose
            r".*acesulfame.*": 950,  # Acesulfame Potassium
            r".*stevia.*": 278,  # Stevia Extract (if exists)
            
            # Lecithin Patterns
            r".*lecithin.*": 46,  # Any lecithin variant
        }
        
        # Save the default patterns
        self.save_custom_patterns()
    
    def apply_custom_patterns(self, ingredient_normalized, original_ingredient):
        """Apply custom pattern matching with high priority"""
        
        # 1. Check exact mappings first
        if ingredient_normalized in self.exact_mappings:
            ingredient_id = self.exact_mappings[ingredient_normalized]
            matched_row = self.ingredients_db[
                self.ingredients_db['ingredient_id'] == ingredient_id
            ]
            if not matched_row.empty:
                logger.info(f"🎯 EXACT PATTERN MATCH: '{original_ingredient}' → '{matched_row.iloc[0]['name']}' (ID: {ingredient_id})")
                return matched_row.iloc[0], 99, "Custom exact match", []
        
        # 2. Check regex patterns
        import re
        for pattern, ingredient_id in self.pattern_rules.items():
            if re.match(pattern, ingredient_normalized, re.IGNORECASE):
                matched_row = self.ingredients_db[
                    self.ingredients_db['ingredient_id'] == ingredient_id
                ]
                if not matched_row.empty:
                    logger.info(f"🎯 PATTERN RULE MATCH: '{original_ingredient}' → '{matched_row.iloc[0]['name']}' (Pattern: {pattern})")
                    return matched_row.iloc[0], 98, f"Pattern rule match", []
        
        return None, 0, "No custom pattern match", []
    
    def find_exact_match_no_normalization(self, original_ingredient):
        """
        Tier 1: Exact 1-on-1 match without any normalization
        Highest priority - most accurate matches
        """
        if not original_ingredient or pd.isna(original_ingredient):
            return None, 0, "Empty ingredient", []
        
        # Clean minimal whitespace but no normalization
        cleaned_original = str(original_ingredient).strip()
        
        # Try exact case-insensitive match first
        exact_matches = self.ingredients_db[
            self.ingredients_db['name'].str.lower() == cleaned_original.lower()
        ]
        
        if not exact_matches.empty:
            app.logger.info(f"🎯 TIER 1 - EXACT MATCH: '{original_ingredient}' → '{exact_matches.iloc[0]['name']}' (100%)")
            return exact_matches.iloc[0], 100, "Tier 1: Exact match (no normalization)", []
        
        # Try with minimal cleaning (just extra whitespace)
        minimal_clean = re.sub(r'\s+', ' ', cleaned_original).strip()
        if minimal_clean != cleaned_original:
            minimal_matches = self.ingredients_db[
                self.ingredients_db['name'].str.lower() == minimal_clean.lower()
            ]
            if not minimal_matches.empty:
                app.logger.info(f"🎯 TIER 1 - EXACT MATCH (minimal clean): '{original_ingredient}' → '{minimal_matches.iloc[0]['name']}' (100%)")
                return minimal_matches.iloc[0], 100, "Tier 1: Exact match (minimal cleaning)", []
        
        app.logger.debug(f"❌ TIER 1: No exact match for '{original_ingredient}'")
        return None, 0, "No exact match found", []
    
    def find_normalized_fuzzy_match(self, original_ingredient, normalized_ingredient):
        """
        Tier 2: Normalized method with fuzzy search and closest alternatives
        Medium priority - handles variations and compound ingredients
        """
        if not normalized_ingredient:
            return None, 0, "Empty normalized ingredient", []
        
        # First check synonyms with normalized input
        for standard_name, synonyms in self.synonym_map.items():
            if normalized_ingredient == standard_name or normalized_ingredient in synonyms:
                db_match = self.ingredients_db[
                    self.ingredients_db['name'].str.lower() == standard_name
                ]
                if not db_match.empty:
                    app.logger.info(f"🎯 TIER 2 - SYNONYM MATCH: '{original_ingredient}' → '{standard_name}' (100%)")
                    return db_match.iloc[0], 100, "Tier 2: Synonym match", []
        
        # Enhanced fuzzy matching with weighted scoring
        ingredient_names = self.ingredients_db['name'].tolist()
        
        # Use weighted approach prioritizing partial_ratio for compound ingredients
        scorers = [
            ('partial_ratio', fuzz.partial_ratio, self.scorer_weights['partial_ratio']),
            ('token_sort_ratio', fuzz.token_sort_ratio, self.scorer_weights['token_sort_ratio']),
            ('token_set_ratio', fuzz.token_set_ratio, self.scorer_weights['token_set_ratio']),
            ('ratio', fuzz.ratio, self.scorer_weights['ratio'])
        ]
        
        all_matches = []
        weighted_scores = {}
        
        for scorer_name, scorer, weight in scorers:
            try:
                matches = process.extract(
                    normalized_ingredient,
                    ingredient_names,
                    scorer=scorer,
                    limit=10  # Get more matches for better analysis
                )
                
                for match_name, score, _ in matches:
                    if match_name not in weighted_scores:
                        weighted_scores[match_name] = {'total': 0, 'count': 0, 'scores': {}}
                    
                    weighted_score = score * weight
                    weighted_scores[match_name]['total'] += weighted_score
                    weighted_scores[match_name]['count'] += 1
                    weighted_scores[match_name]['scores'][scorer_name] = score
                    
                    all_matches.append({
                        'name': match_name,
                        'score': score,
                        'weighted_score': weighted_score,
                        'scorer': scorer_name
                    })
            except Exception as e:
                app.logger.warning(f"Error in fuzzy matching with {scorer_name}: {e}")
                continue
        
        if not weighted_scores:
            app.logger.debug(f"❌ TIER 2: No fuzzy matches found for '{original_ingredient}'")
            return None, 0, "No fuzzy matches found", []
        
        # Calculate final weighted scores
        final_scores = []
        for name, data in weighted_scores.items():
            try:
                # Weighted average with bonus for multiple high scores
                avg_weighted = data['total'] / len(scorers)  # Normalize by number of scorers
                
                # Bonus for consistency across scorers
                high_scores = sum(1 for score in data['scores'].values() if score > 80)
                consistency_bonus = high_scores * 2
                
                final_score = min(100, avg_weighted + consistency_bonus)
                
                final_scores.append({
                    'name': name,
                    'final_score': final_score,
                    'raw_scores': data['scores'],
                    'best_scorer': max(data['scores'].items(), key=lambda x: x[1])[0]
                })
            except Exception as e:
                app.logger.warning(f"Error calculating final score for {name}: {e}")
                continue
        
        # Sort by final weighted score
        final_scores.sort(key=lambda x: x['final_score'], reverse=True)
        
        if final_scores:
            best_match = final_scores[0]
            
            # Apply confidence threshold - only return matches above threshold
            if best_match['final_score'] >= self.confidence_threshold:
                try:
                    matched_row = self.ingredients_db[
                        self.ingredients_db['name'] == best_match['name']
                    ].iloc[0]
                    
                    # Prepare alternatives with detailed scoring info (up to 5 alternatives)
                    alternatives = []
                    for match in final_scores[:5]:
                        alternatives.append({
                            'name': match['name'],
                            'score': match['final_score'],
                            'scorer': match['best_scorer'],
                            'raw_scores': match['raw_scores']
                        })
                    
                    app.logger.info(f"🎯 TIER 2 - FUZZY MATCH: '{original_ingredient}' → '{best_match['name']}' "
                                   f"(score: {best_match['final_score']:.1f}%, via: {best_match['best_scorer']})")
                    
                    return matched_row, best_match['final_score'], f"Tier 2: Fuzzy match ({best_match['best_scorer']})", alternatives
                except Exception as e:
                    app.logger.error(f"Error retrieving matched row for {best_match['name']}: {e}")
                    return None, 0, "Error in match retrieval", []
            else:
                app.logger.debug(f"❌ TIER 2: Best score {best_match['final_score']:.1f}% below threshold {self.confidence_threshold}% for '{original_ingredient}'")
                
                # Return top alternatives even if below threshold for manual review
                alternatives = []
                for match in final_scores[:3]:
                    alternatives.append({
                        'name': match['name'],
                        'score': match['final_score'],
                        'scorer': match['best_scorer'],
                        'raw_scores': match['raw_scores']
                    })
                
                return None, best_match['final_score'], f"Below threshold (best: {best_match['final_score']:.1f}%)", alternatives
        
        app.logger.debug(f"❌ TIER 2: No valid fuzzy matches for '{original_ingredient}'")
        return None, 0, "No fuzzy matches found", []
    
    def find_pattern_match(self, original_ingredient, normalized_ingredient):
        """
        Tier 3: Pattern matching for remaining ingredients
        Lowest priority - handles special cases and complex patterns
        """
        if not normalized_ingredient:
            return None, 0, "Empty ingredient for pattern matching", []
        
        try:
            # 1. Check exact mappings first
            if normalized_ingredient in self.exact_mappings:
                ingredient_id = self.exact_mappings[normalized_ingredient]
                matched_row = self.ingredients_db[
                    self.ingredients_db['ingredient_id'] == ingredient_id
                ]
                if not matched_row.empty:
                    app.logger.info(f"🎯 TIER 3 - EXACT PATTERN: '{original_ingredient}' → '{matched_row.iloc[0]['name']}' (ID: {ingredient_id})")
                    return matched_row.iloc[0], 99, "Tier 3: Custom exact pattern", []
            
            # 2. Check regex patterns
            for pattern, ingredient_id in self.pattern_rules.items():
                try:
                    if re.match(pattern, normalized_ingredient, re.IGNORECASE):
                        matched_row = self.ingredients_db[
                            self.ingredients_db['ingredient_id'] == ingredient_id
                        ]
                        if not matched_row.empty:
                            app.logger.info(f"🎯 TIER 3 - REGEX PATTERN: '{original_ingredient}' → '{matched_row.iloc[0]['name']}' (Pattern: {pattern})")
                            return matched_row.iloc[0], 98, f"Tier 3: Regex pattern match", []
                except re.error as e:
                    app.logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                    continue
                except Exception as e:
                    app.logger.warning(f"Error applying pattern '{pattern}': {e}")
                    continue
            
            # 3. Advanced pattern matching for complex cases
            advanced_match = self.apply_advanced_patterns(original_ingredient, normalized_ingredient)
            if advanced_match:
                return advanced_match
            
        except Exception as e:
            app.logger.error(f"Error in pattern matching for '{original_ingredient}': {e}")
        
        app.logger.debug(f"❌ TIER 3: No pattern matches for '{original_ingredient}'")
        return None, 0, "No pattern matches found", []
    
    def apply_advanced_patterns(self, original_ingredient, normalized_ingredient):
        """
        Advanced pattern matching for complex ingredient cases
        """
        try:
            # Pattern for vitamin compounds (e.g., "vitamin c as ascorbic acid")
            vitamin_pattern = r'vitamin\s+([a-z0-9]+)(?:\s+as\s+(.+))?'
            vitamin_match = re.search(vitamin_pattern, normalized_ingredient, re.IGNORECASE)
            if vitamin_match:
                vitamin_type = vitamin_match.group(1).lower()
                compound = vitamin_match.group(2) if vitamin_match.group(2) else None
                
                # Look for vitamin matches in database
                vitamin_candidates = self.ingredients_db[
                    self.ingredients_db['name'].str.contains(f'vitamin.*{vitamin_type}', case=False, regex=True)
                ]
                
                if not vitamin_candidates.empty:
                    best_candidate = vitamin_candidates.iloc[0]
                    app.logger.info(f"🎯 TIER 3 - VITAMIN PATTERN: '{original_ingredient}' → '{best_candidate['name']}'")
                    return best_candidate, 95, "Tier 3: Vitamin pattern match", []
            
            # Pattern for mineral compounds (e.g., "calcium as calcium carbonate")
            mineral_pattern = r'(calcium|magnesium|iron|zinc|potassium|sodium)(?:\s+as\s+(.+))?'
            mineral_match = re.search(mineral_pattern, normalized_ingredient, re.IGNORECASE)
            if mineral_match:
                mineral_type = mineral_match.group(1).lower()
                
                mineral_candidates = self.ingredients_db[
                    self.ingredients_db['name'].str.contains(mineral_type, case=False)
                ]
                
                if not mineral_candidates.empty:
                    best_candidate = mineral_candidates.iloc[0]
                    app.logger.info(f"🎯 TIER 3 - MINERAL PATTERN: '{original_ingredient}' → '{best_candidate['name']}'")
                    return best_candidate, 94, "Tier 3: Mineral pattern match", []
            
            # Pattern for protein types
            protein_pattern = r'(whey|casein|soy|pea|rice|hemp).*protein'
            protein_match = re.search(protein_pattern, normalized_ingredient, re.IGNORECASE)
            if protein_match:
                protein_type = protein_match.group(1).lower()
                
                protein_candidates = self.ingredients_db[
                    self.ingredients_db['name'].str.contains(f'{protein_type}.*protein', case=False, regex=True)
                ]
                
                if not protein_candidates.empty:
                    best_candidate = protein_candidates.iloc[0]
                    app.logger.info(f"🎯 TIER 3 - PROTEIN PATTERN: '{original_ingredient}' → '{best_candidate['name']}'")
                    return best_candidate, 93, "Tier 3: Protein pattern match", []
            
        except Exception as e:
            app.logger.warning(f"Error in advanced pattern matching: {e}")
        
        return None
    
    def find_best_match_three_tier(self, original_ingredient):
        """
        New three-tiered matching algorithm:
        1. Exact 1-on-1 match without normalization
        2. Normalized fuzzy search with alternatives 
        3. Pattern matching for remaining ingredients
        """
        if not original_ingredient or pd.isna(original_ingredient):
            return None, 0, "Empty ingredient", []
        
        # Tier 1: Exact match without normalization
        tier1_result = self.find_exact_match_no_normalization(original_ingredient)
        if tier1_result[0] is not None:
            return tier1_result
        
        # Tier 2: Normalized fuzzy search
        normalized_ingredient = self.normalize_ingredient(original_ingredient)
        tier2_result = self.find_normalized_fuzzy_match(original_ingredient, normalized_ingredient)
        if tier2_result[0] is not None:
            return tier2_result
        
        # Tier 3: Pattern matching
        tier3_result = self.find_pattern_match(original_ingredient, normalized_ingredient)
        if tier3_result[0] is not None:
            return tier3_result
        
        # No matches found in any tier
        app.logger.info(f"❌ ALL TIERS FAILED: No match found for '{original_ingredient}'")
        
        # For completely unmatched ingredients, still try to provide some alternatives from Tier 2
        if tier2_result[3]:  # If tier2 had alternatives below threshold
            return None, 0, "No match found - see alternatives", tier2_result[3]
        
        return None, 0, "No match found in any tier", []

    # Keep the old method name for backward compatibility but use new algorithm
    def find_best_match_enhanced(self, ingredient_normalized, original_ingredient):
        """
        Legacy method name - now uses the new three-tier algorithm
        """
        return self.find_best_match_three_tier(original_ingredient)
    
    def get_pattern_override_interface(self):
        """Return current patterns for UI display and editing"""
        return {
            'exact_mappings': self.exact_mappings,
            'pattern_rules': self.pattern_rules,
            'total_patterns': len(self.exact_mappings) + len(self.pattern_rules)
        }
    
    def add_custom_pattern(self, pattern_type, pattern, ingredient_id, description=""):
        """Add or update a custom pattern"""
        if pattern_type == "exact":
            self.exact_mappings[pattern.lower()] = ingredient_id
        elif pattern_type == "regex":
            self.pattern_rules[pattern] = ingredient_id
        
        # Save to file for persistence
        self.save_custom_patterns()
        
        logger.info(f"Added custom pattern: {pattern_type} '{pattern}' -> ID {ingredient_id}")
    
    def save_custom_patterns(self):
        """Save custom patterns to file for persistence"""
        patterns_data = {
            'exact_mappings': self.exact_mappings,
            'pattern_rules': self.pattern_rules,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open('custom_patterns.json', 'w') as f:
                json.dump(patterns_data, f, indent=2)
            logger.info("Custom patterns saved successfully")
        except Exception as e:
            logger.error(f"Failed to save custom patterns: {e}")
    
    def process_products(self):
        """Process all products and map ingredients"""
        if self.products_df is None or self.ingredients_db is None:
            app.logger.error("[IngredientMapper] No data loaded. Please load CSV files first.")
            return
        
        app.logger.info(f"[IngredientMapper] Starting mapping for {len(self.products_df)} products and {len(self.ingredients_db)} ingredients in DB.")
        self.mapping_results = []
        self.unmapped_ingredients = []
        self.processing_logs = []
        
        total_products = len(self.products_df)
        app.logger.info(f"🚀 Starting ingredient mapping for {total_products} products...")
        
        for idx, row in self.products_df.iterrows():
            product_name = self.safe_get_str(row.get('product_name', f'Product {idx}'), f'Product {idx}')
            product_company = self.safe_get_str(row.get('product_company', ''))
            ingredient_column = 'ingredients'
            
            app.logger.info(f"[IngredientMapper] Processing product {idx+1}/{total_products}: {product_name}")
            
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
                
                # Find best match using new three-tier algorithm with safety wrapper
                match_result, confidence, match_type, alternatives = self.find_best_match_three_tier_safe(original)
                
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
        
        app.logger.info(f"[IngredientMapper] Finished mapping. Total mapped: {total_mapped}, Total unmapped: {total_unmapped}")
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
            
            # Optimize database for three-tier matching
            self.optimize_database_for_matching()
            
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
        self.confidence_threshold = 70  # Lowered from 85 to improve matching
        self.synonym_map = {}
        self.load_synonyms()
        self.parsing_stats = defaultdict(int)
        self.confidence_distribution = []
        self.unmatched_patterns = Counter()
        self.database_coverage_analysis = {}
        self.custom_patterns = {}
        self.exact_mappings = {}
        self.pattern_rules = {}
        self.load_custom_patterns()

    def validate_ingredient_input(self, ingredient):
        """
        Validate and sanitize ingredient input before processing
        Handles edge cases and malformed data
        """
        if ingredient is None or pd.isna(ingredient):
            return None, "Null or NaN ingredient"
        
        # Convert to string and basic cleaning
        ingredient_str = str(ingredient).strip()
        
        # Check for empty or whitespace-only
        if not ingredient_str or ingredient_str.isspace():
            return None, "Empty or whitespace-only ingredient"
        
        # Check for obviously invalid ingredients
        invalid_patterns = [
            r'^\d+$',  # Numbers only
            r'^[^\w\s]+$',  # Special characters only
            r'^(.)\1{10,}',  # Repeated character spam
            r'^[xX]+$',  # Just X's (placeholder text)
            r'^test\s*\d*$',  # Test entries
            r'^example\b',  # Example entries
            r'^lorem\s+ipsum',  # Lorem ipsum text
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, ingredient_str, re.IGNORECASE):
                return None, f"Invalid ingredient pattern: {pattern}"
        
        # Check for excessively long ingredients (likely corrupted data)
        if len(ingredient_str) > 500:
            return None, f"Ingredient too long ({len(ingredient_str)} chars)"
        
        # Check for binary or encoded data
        if not ingredient_str.isprintable():
            return None, "Non-printable characters detected"
        
        return ingredient_str, None
    
    def get_tier_statistics(self):
        """
        Get statistics about which tier is being used most often
        Useful for algorithm optimization
        """
        if not hasattr(self, 'tier_stats'):
            self.tier_stats = {
                'tier1_exact': 0,
                'tier2_synonym': 0, 
                'tier2_fuzzy': 0,
                'tier3_pattern': 0,
                'tier3_advanced': 0,
                'no_match': 0,
                'total_processed': 0
            }
        return self.tier_stats
    
    def update_tier_statistics(self, match_type):
        """Update tier usage statistics"""
        if not hasattr(self, 'tier_stats'):
            self.tier_stats = {
                'tier1_exact': 0,
                'tier2_synonym': 0, 
                'tier2_fuzzy': 0,
                'tier3_pattern': 0,
                'tier3_advanced': 0,
                'no_match': 0,
                'total_processed': 0
            }
        
        self.tier_stats['total_processed'] += 1
        
        if 'Tier 1' in match_type:
            self.tier_stats['tier1_exact'] += 1
        elif 'Tier 2' in match_type:
            if 'Synonym' in match_type:
                self.tier_stats['tier2_synonym'] += 1
            else:
                self.tier_stats['tier2_fuzzy'] += 1
        elif 'Tier 3' in match_type:
            if 'pattern' in match_type.lower():
                self.tier_stats['tier3_pattern'] += 1
            else:
                self.tier_stats['tier3_advanced'] += 1
        else:
            self.tier_stats['no_match'] += 1
    
    def find_best_match_three_tier_safe(self, original_ingredient):
        """
        Safe wrapper around the three-tier algorithm with comprehensive error handling
        """
        try:
            # Input validation
            validated_ingredient, error_msg = self.validate_ingredient_input(original_ingredient)
            if validated_ingredient is None:
                app.logger.debug(f"❌ INPUT VALIDATION FAILED: {original_ingredient} - {error_msg}")
                return None, 0, f"Invalid input: {error_msg}", []
            
            # Call the main algorithm
            result = self.find_best_match_three_tier(validated_ingredient)
            
            # Update statistics
            if len(result) >= 3:
                self.update_tier_statistics(result[2])
            
            return result
            
        except Exception as e:
            app.logger.error(f"🚨 CRITICAL ERROR in three-tier matching for '{original_ingredient}': {str(e)}")
            # Try to provide some fallback alternatives
            try:
                # Emergency fallback: simple fuzzy search without tiers
                if self.ingredients_db is not None and not self.ingredients_db.empty:
                    ingredient_names = self.ingredients_db['name'].tolist()
                    if ingredient_names:
                        matches = process.extract(
                            str(original_ingredient), 
                            ingredient_names, 
                            limit=3
                        )
                        if matches:
                            alternatives = [{'name': match[0], 'score': match[1], 'scorer': 'emergency_fallback'} for match in matches]
                            return None, 0, f"Error occurred - emergency fallback", alternatives
            except:
                pass
            
            return None, 0, f"Critical error: {str(e)}", []
    
    def optimize_database_for_matching(self):
        """
        Pre-process the ingredients database for faster matching
        Create lookup indices and normalized versions
        """
        if self.ingredients_db is None or self.ingredients_db.empty:
            return
        
        try:
            # Create lowercase lookup dictionary for Tier 1 exact matching
            if not hasattr(self, 'exact_lookup'):
                self.exact_lookup = {}
                for idx, row in self.ingredients_db.iterrows():
                    name_lower = str(row['name']).lower().strip()
                    self.exact_lookup[name_lower] = row
                
                app.logger.info(f"📊 Created exact lookup index with {len(self.exact_lookup)} entries")
            
            # Create normalized lookup for faster Tier 2 matching
            if not hasattr(self, 'normalized_lookup'):
                self.normalized_lookup = {}
                for idx, row in self.ingredients_db.iterrows():
                    normalized = self.normalize_ingredient(row['name'])
                    if normalized and len(normalized) > 2:
                        self.normalized_lookup[normalized] = row
                
                app.logger.info(f"📊 Created normalized lookup index with {len(self.normalized_lookup)} entries")
                
        except Exception as e:
            app.logger.warning(f"⚠️ Failed to optimize database for matching: {e}")
    
    def find_exact_match_no_normalization_optimized(self, original_ingredient):
        """
        Optimized version of Tier 1 matching using pre-built indices
        """
        if not original_ingredient or pd.isna(original_ingredient):
            return None, 0, "Empty ingredient", []
        
        # Use optimized lookup if available
        if hasattr(self, 'exact_lookup'):
            cleaned_original = str(original_ingredient).strip().lower()
            
            # Direct lookup
            if cleaned_original in self.exact_lookup:
                matched_row = self.exact_lookup[cleaned_original]
                app.logger.info(f"🎯 TIER 1 - EXACT MATCH (optimized): '{original_ingredient}' → '{matched_row['name']}' (100%)")
                return matched_row, 100, "Tier 1: Exact match (optimized)", []
            
            # Try with minimal cleaning
            minimal_clean = re.sub(r'\s+', ' ', cleaned_original).strip()
            if minimal_clean != cleaned_original and minimal_clean in self.exact_lookup:
                matched_row = self.exact_lookup[minimal_clean]
                app.logger.info(f"🎯 TIER 1 - EXACT MATCH (optimized, minimal clean): '{original_ingredient}' → '{matched_row['name']}' (100%)")
                return matched_row, 100, "Tier 1: Exact match (optimized, minimal cleaning)", []
        
        # Fallback to original method if optimization not available
        return self.find_exact_match_no_normalization(original_ingredient)

    def get_algorithm_summary(self):
        """
        Get a summary of the three-tier algorithm implementation
        
        TIER 1: EXACT MATCH (No Normalization) - 100% Confidence
        - Direct case-insensitive string matching
        - Minimal whitespace cleaning only
        - Optimized with pre-built lookup indices
        - Handles exact ingredient names from database
        
        TIER 2: NORMALIZED FUZZY SEARCH - Variable Confidence
        - Synonym matching (100% confidence)
        - Comprehensive ingredient normalization
        - Weighted fuzzy matching algorithms:
          * partial_ratio (40% weight) - best for compound ingredients
          * token_sort_ratio (25% weight) - handles word order
          * token_set_ratio (25% weight) - handles partial matches
          * ratio (10% weight) - strict character matching
        - Confidence threshold filtering
        - Alternative suggestions for manual review
        
        TIER 3: PATTERN MATCHING - 98-93% Confidence
        - Custom exact pattern mappings (99% confidence)
        - Regex pattern rules (98% confidence)
        - Advanced pattern recognition:
          * Vitamin compounds (95% confidence)
          * Mineral compounds (94% confidence) 
          * Protein types (93% confidence)
        
        EDGE CASE HANDLING:
        - Input validation and sanitization
        - Error recovery with emergency fallback
        - Performance optimization with lookup indices
        - Comprehensive logging and statistics tracking
        - Database structure validation
        - Memory-efficient processing
        
        STATISTICS TRACKING:
        - Tier usage distribution
        - Confidence score analysis
        - Processing performance metrics
        - Pattern effectiveness monitoring
        """
        stats = self.get_tier_statistics()
        total = stats.get('total_processed', 0)
        
        summary = {
            "algorithm_name": "Three-Tier Ingredient Matching Algorithm",
            "tiers": {
                "tier1": {
                    "name": "Exact Match (No Normalization)",
                    "confidence": "100%",
                    "usage_count": stats.get('tier1_exact', 0),
                    "usage_percent": (stats.get('tier1_exact', 0) / total * 100) if total > 0 else 0,
                    "description": "Direct case-insensitive matching with minimal cleaning"
                },
                "tier2": {
                    "name": "Normalized Fuzzy Search", 
                    "confidence": "Variable (threshold-based)",
                    "usage_count": stats.get('tier2_synonym', 0) + stats.get('tier2_fuzzy', 0),
                    "usage_percent": ((stats.get('tier2_synonym', 0) + stats.get('tier2_fuzzy', 0)) / total * 100) if total > 0 else 0,
                    "description": "Synonym matching and weighted fuzzy algorithms"
                },
                "tier3": {
                    "name": "Pattern Matching",
                    "confidence": "98-93%",
                    "usage_count": stats.get('tier3_pattern', 0) + stats.get('tier3_advanced', 0),
                    "usage_percent": ((stats.get('tier3_pattern', 0) + stats.get('tier3_advanced', 0)) / total * 100) if total > 0 else 0,
                    "description": "Custom patterns and advanced ingredient recognition"
                }
            },
            "optimization_features": [
                "Pre-built lookup indices for Tier 1",
                "Normalized lookup cache for Tier 2", 
                "Input validation and sanitization",
                "Error recovery mechanisms",
                "Performance statistics tracking"
            ],
            "total_processed": total,
            "unmatched_count": stats.get('no_match', 0),
            "success_rate": ((total - stats.get('no_match', 0)) / total * 100) if total > 0 else 0
        }
        
        return summary

# Initialize mapper
mapper = IngredientMapper()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load the most recently uploaded products and ingredients files into the backend."""
    try:
        if not mapper.products_file_path or not mapper.ingredients_file_path:
            return jsonify({'success': False, 'error': 'Files not uploaded yet.'}), 400
        loaded = mapper.load_data(mapper.products_file_path, mapper.ingredients_file_path)
        if loaded:
            return jsonify({'success': True, 'products_count': len(mapper.products_df), 'ingredients_count': len(mapper.ingredients_db), 'logs': mapper.processing_logs[-10:]})
        else:
            return jsonify({'success': False, 'error': 'Failed to load data.'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_ingredients():
    """Process ingredients with fuzzy matching"""
    try:
        data = request.get_json()
        app.logger.info(f"[API] /api/process called with data: {data}")
        
        # Validate mapper state
        app.logger.info(f"[API] Mapper validation - products_df: {mapper.products_df is not None}, ingredients_db: {mapper.ingredients_db is not None}")
        app.logger.info(f"[API] Custom patterns loaded: exact={len(mapper.exact_mappings)}, rules={len(mapper.pattern_rules)}")
        
        mapper.confidence_threshold = data.get('confidence_threshold', 70)
        app.logger.info(f"[API] Set confidence_threshold to {mapper.confidence_threshold}")
        
        if mapper.products_df is None or mapper.ingredients_db is None:
            app.logger.error("[API] No data loaded. Cannot process ingredients.")
            return jsonify({'success': False, 'error': 'No data loaded.'}), 400
            
        # Test enhanced method with a simple case
        app.logger.info("[API] Testing enhanced matching method...")
        test_result = mapper.find_best_match_enhanced("protein", "protein")
        app.logger.info(f"[API] Test result: {test_result}")
        
        mapper.process_products()
        app.logger.info(f"[API] Mapping complete. mapped_count={len(mapper.mapping_results)}, unmapped_count={len(mapper.unmapped_ingredients)}")
        return jsonify({
            'success': True,
            'mapped_count': len(mapper.mapping_results),
            'unmapped_count': len(mapper.unmapped_ingredients),
            'logs': mapper.processing_logs[-20:]  # Last 20 logs
        })
    except Exception as e:
        app.logger.error(f"[API] Processing failed: {str(e)}")
        import traceback
        app.logger.error(f"[API] Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/results')
def get_results():
    """Get mapping results"""
    # Convert data to JSON-serializable format
    def convert_to_json_serializable(data):
        import pandas as pd
        import numpy as np
        try:
            if isinstance(data, list):
                return [convert_to_json_serializable(item) for item in data]
            elif isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    try:
                        result[key] = convert_to_json_serializable(value)
                    except Exception as e:
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
                    return str(data)
            elif isinstance(data, (np.generic,)):
                return data.item()
            else:
                return data
        except Exception as e:
            return str(data)

    try:
        mapped = convert_to_json_serializable(mapper.mapping_results)
        unmapped = convert_to_json_serializable(mapper.unmapped_ingredients)
        return jsonify({
            'mapped': mapped,
            'unmapped': unmapped
        })
    except Exception as e:
        return jsonify({
            'mapped': [],
            'unmapped': [],
            'error': str(e)
        }), 500

@app.route('/api/manual-map', methods=['POST'])
def manual_map():
    """Manually map an unmapped ingredient or batch of ingredients"""
    data = request.get_json()
    original_ingredient = data['original_ingredient']
    ingredient_id = data['ingredient_id']
    note = data.get('note', '')
    # Find the ingredient in database
    matched_ingredient = mapper.ingredients_db[
        mapper.ingredients_db['ingredient_id'] == ingredient_id
    ]
    if matched_ingredient.empty:
        return jsonify({'success': False, 'error': 'Ingredient ID not found'})
    matched_ingredient = matched_ingredient.iloc[0]
    # Support batch mapping
    if isinstance(original_ingredient, list):
        count = 0
        for orig in original_ingredient:
            for i, item in enumerate(mapper.unmapped_ingredients):
                if item['original_ingredient'] == orig:
                    mapping_entry = item.copy()
                    mapping_entry.update({
                        'ingredient_id': ingredient_id,
                        'matched_name': matched_ingredient['name'],
                        'status': 'mapped',
                        'confidence': 100,
                        'match_type': 'manual',
                        'mapping_source': 'manual',
                        'note': note
                    })
                    mapper.mapping_results.append(mapping_entry)
                    mapper.unmapped_ingredients.pop(i)
                    count += 1
                    break
        return jsonify({'success': True, 'mapped_count': count})
    # Single mapping
    for i, item in enumerate(mapper.unmapped_ingredients):
        if item['original_ingredient'] == original_ingredient:
            mapping_entry = item.copy()
            mapping_entry.update({
                'ingredient_id': ingredient_id,
                'matched_name': matched_ingredient['name'],
                'status': 'mapped',
                'confidence': 100,
                'match_type': 'manual',
                'mapping_source': 'manual',
                'note': note
            })
            mapper.mapping_results.append(mapping_entry)
            mapper.unmapped_ingredients.pop(i)
            break
    return jsonify({'success': True})

@app.route('/api/remap-ingredient', methods=['POST'])
def remap_ingredient():
    """Remap an already mapped ingredient to a different ingredient"""
    data = request.get_json()
    original_ingredient = data['original_ingredient']
    new_ingredient_id = data['new_ingredient_id']
    product_name = data.get('product_name', '')
    note = data.get('note', '')
    
    # Find the new ingredient in database
    new_matched_ingredient = mapper.ingredients_db[
        mapper.ingredients_db['ingredient_id'] == new_ingredient_id
    ]
    if new_matched_ingredient.empty:
        return jsonify({'success': False, 'error': 'New ingredient ID not found'})
    new_matched_ingredient = new_matched_ingredient.iloc[0]
    
    # Find the existing mapping in mapped results
    for i, item in enumerate(mapper.mapping_results):
        if (item['original_ingredient'] == original_ingredient and 
            (not product_name or item.get('product_name') == product_name)):
            # Update the existing mapping
            mapper.mapping_results[i].update({
                'ingredient_id': new_ingredient_id,
                'matched_name': new_matched_ingredient['name'],
                'status': 'mapped',
                'confidence': 100,
                'match_type': 'manual_remap',
                'mapping_source': 'manual',
                'note': note,
                'timestamp': datetime.now().isoformat()
            })
            return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Original mapping not found'})

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
    """Export mapping results to CSV, including product mapping status."""
    try:
        mapped_df = pd.DataFrame(mapper.mapping_results)
        unmapped_df = pd.DataFrame(mapper.unmapped_ingredients)
        # Ensure mapping_source and note columns exist
        if 'mapping_source' not in mapped_df.columns:
            mapped_df['mapping_source'] = mapped_df.get('match_type', '').apply(lambda x: 'manual' if x == 'manual' else 'auto')
        if 'note' not in mapped_df.columns:
            mapped_df['note'] = ''
        # Add product-level mapping status
        mapped_df['fully_mapped'] = False
        if not mapped_df.empty:
            prod_status = {}
            for pname in mapped_df['product_name'].unique():
                total = len(mapped_df[mapped_df['product_name'] == pname]) + len(unmapped_df[unmapped_df['product_name'] == pname])
                unmapped = len(unmapped_df[unmapped_df['product_name'] == pname])
                prod_status[pname] = (unmapped == 0 and total > 0)
            mapped_df['fully_mapped'] = mapped_df['product_name'].map(prod_status)
        import tempfile
        import os
        
        # Create files in temp directory to avoid permission issues
        temp_dir = tempfile.gettempdir()
        
        mapped_filename = os.path.join(temp_dir, f'mapped_ingredients_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        mapped_df.to_csv(mapped_filename, index=False)
        
        unmapped_filename = os.path.join(temp_dir, f'unmapped_ingredients_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        unmapped_df.to_csv(unmapped_filename, index=False)
        
        logs_filename = os.path.join(temp_dir, f'processing_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        logs_df = pd.DataFrame(mapper.processing_logs)
        logs_df.to_csv(logs_filename, index=False)
        return jsonify({
            'success': True,
            'files': {
                'mapped': os.path.basename(mapped_filename),
                'unmapped': os.path.basename(unmapped_filename),
                'logs': os.path.basename(logs_filename)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ingredients/search')
def search_ingredients():
    """Search ingredients database"""
    try:
        query = request.args.get('q', '').lower()
        ingredient_id = request.args.get('id')
        limit = int(request.args.get('limit', 20))
        
        if mapper.ingredients_db is None:
            return jsonify([])
        
        # Search by ID if provided
        if ingredient_id:
            try:
                ingredient_id = int(ingredient_id)
                matches = mapper.ingredients_db[
                    mapper.ingredients_db['ingredient_id'] == ingredient_id
                ]
                
                return jsonify([{
                    'ingredient_id': row['ingredient_id'],
                    'name': row['name']
                } for _, row in matches.iterrows()])
            except ValueError:
                return jsonify([])
        
        # Search by query text
        if not query or len(query) < 2:
            return jsonify([])
        
        # Escape special regex characters
        import re
        escaped_query = re.escape(query)
        
        # Search in ingredient names
        matches = mapper.ingredients_db[
            mapper.ingredients_db['name'].str.lower().str.contains(escaped_query, na=False, regex=True)
        ].head(limit)
        
        return jsonify([{
            'ingredient_id': row['ingredient_id'],
            'name': row['name']
        } for _, row in matches.iterrows()])
        
    except Exception as e:
        app.logger.error(f"Search error: {str(e)}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update settings"""
    if request.method == 'POST':
        data = request.get_json()
        mapper.confidence_threshold = data.get('confidence_threshold', 70)
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
        # Set the file path on the mapper for later loading
        if file_type == 'products':
            mapper.products_file_path = filepath
        elif file_type == 'ingredients':
            mapper.ingredients_file_path = filepath
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
    """Export mapped results with only selected fields as CSV, including product mapping status."""
    data = request.get_json()
    fields = data.get('fields', [])
    filename = data.get('filename', None)
    mapped_df = pd.DataFrame(mapper.mapping_results)
    unmapped_df = pd.DataFrame(mapper.unmapped_ingredients)
    # Ensure mapping_source and note columns exist
    if 'mapping_source' not in mapped_df.columns:
        mapped_df['mapping_source'] = mapped_df.get('match_type', '').apply(lambda x: 'manual' if x == 'manual' else 'auto')
    if 'note' not in mapped_df.columns:
        mapped_df['note'] = ''
    # Add product-level mapping status
    mapped_df['fully_mapped'] = False
    if not mapped_df.empty:
        prod_status = {}
        for pname in mapped_df['product_name'].unique():
            total = len(mapped_df[mapped_df['product_name'] == pname]) + len(unmapped_df[unmapped_df['product_name'] == pname])
            unmapped = len(unmapped_df[unmapped_df['product_name'] == pname])
            prod_status[pname] = (unmapped == 0 and total > 0)
        mapped_df['fully_mapped'] = mapped_df['product_name'].map(prod_status)
    if not fields:
        return jsonify({'success': False, 'error': 'No fields selected'}), 400
    try:
        # Filter fields to only include those that actually exist in the DataFrame
        available_fields = [field for field in fields if field in mapped_df.columns]
        if not available_fields:
            return jsonify({'success': False, 'error': 'None of the selected fields exist in the data'}), 400
        
        export_df = mapped_df[available_fields]
        if filename:
            if not filename.endswith('.csv'):
                filename += '.csv'
        else:
            filename = f'custom_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # Create file in temp directory
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        export_df.to_csv(file_path, index=False)
        
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    """Return the mapped results as a CSV string (for export preview). Only mapped rows."""
    import pandas as pd
    try:
        mapped_df = pd.DataFrame(mapper.mapping_results)
        if mapped_df.empty:
            csv_str = ''
        else:
            csv_str = mapped_df.to_csv(index=False)
        return jsonify({'csv': csv_str})
    except Exception as e:
        return jsonify({'csv': '', 'error': str(e)}), 500

def to_python_type(val):
    if isinstance(val, dict):
        return {k: to_python_type(v) for k, v in val.items()}
    if isinstance(val, list):
        return [to_python_type(v) for v in val]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val

@app.route('/api/products-mapping-status', methods=['GET'])
def products_mapping_status():
    """Return mapping status for all products, including mapped/unmapped ingredients and summary info."""
    products = {}
    for row in mapper.mapping_results:
        pname = row.get('product_name')
        if pname not in products:
            products[pname] = {'product_name': pname, 'ingredients': [], 'mapped': [], 'unmapped': []}
        row_copy = row.copy()
        row_copy['is_mapped'] = True
        products[pname]['ingredients'].append(row_copy)
        products[pname]['mapped'].append(row_copy)
    for row in mapper.unmapped_ingredients:
        pname = row.get('product_name')
        if pname not in products:
            products[pname] = {'product_name': pname, 'ingredients': [], 'mapped': [], 'unmapped': []}
        row_copy = row.copy()
        row_copy['is_mapped'] = False
        products[pname]['ingredients'].append(row_copy)
        products[pname]['unmapped'].append(row_copy)
    result = []
    for pname, info in products.items():
        total = int(len(info['ingredients']))
        mapped = int(len(info['mapped']))
        unmapped = int(len(info['unmapped']))
        fully_mapped = unmapped == 0 and total > 0
        info['total_ingredients'] = total
        info['mapped_count'] = mapped
        info['unmapped_count'] = unmapped
        info['fully_mapped'] = fully_mapped
        result.append(info)
    # Robust conversion for all nested fields
    result = to_python_type(result)
    return jsonify({'products': result})

@app.route('/api/add-ingredient', methods=['POST'])
def add_ingredient():
    """Add a new ingredient to a product as unmapped."""
    data = request.get_json()
    product_name = data.get('product_name')
    ingredient_name = data.get('ingredient_name')
    if not product_name or not ingredient_name:
        return jsonify({'success': False, 'error': 'Missing product or ingredient name'}), 400
    # Add to unmapped_ingredients
    new_row = {
        'product_name': product_name,
        'original_ingredient': ingredient_name,
        'normalized_ingredient': ingredient_name.lower(),
        'matched_name': '',
        'confidence': 0,
        'match_type': '',
        'mapping_source': '',
        'note': '',
    }
    mapper.unmapped_ingredients.append(new_row)
    return jsonify({'success': True, 'added': new_row})

@app.route('/api/bulk-ingredients', methods=['GET'])
def get_bulk_ingredients():
    """Get ingredients grouped by their normalized name for bulk mapping"""
    try:
        # Group unmapped ingredients by normalized name
        ingredient_groups = {}
        
        for item in mapper.unmapped_ingredients:
            normalized = item.get('normalized_ingredient', '').lower().strip()
            if not normalized:
                continue
                
            if normalized not in ingredient_groups:
                ingredient_groups[normalized] = {
                    'normalized_ingredient': normalized,
                    'original_ingredients': [],
                    'products': set(),
                    'count': 0,
                    'suggested_mapping': None,
                    'confidence': 0
                }
            
            ingredient_groups[normalized]['original_ingredients'].append(item['original_ingredient'])
            ingredient_groups[normalized]['products'].add(item.get('product_name', 'Unknown'))
            ingredient_groups[normalized]['count'] += 1
            
            # Use the best suggested mapping from any of the instances
            if item.get('matched_name') and item.get('confidence', 0) > ingredient_groups[normalized]['confidence']:
                ingredient_groups[normalized]['suggested_mapping'] = item['matched_name']
                ingredient_groups[normalized]['confidence'] = item.get('confidence', 0)
        
        # Convert to list and sort by count (most common first)
        bulk_ingredients = []
        for group in ingredient_groups.values():
            group['products'] = list(group['products'])  # Convert set to list for JSON
            bulk_ingredients.append(group)
        
        bulk_ingredients.sort(key=lambda x: x['count'], reverse=True)
        
        # Only return ingredients that appear in multiple products or multiple times
        bulk_ingredients = [ing for ing in bulk_ingredients if ing['count'] > 1 or len(ing['products']) > 1]
        
        return jsonify({'bulk_ingredients': bulk_ingredients})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bulk-map', methods=['POST'])
def bulk_map_ingredient():
    """Map all instances of a normalized ingredient to a target ingredient"""
    try:
        data = request.get_json()
        normalized_ingredient = data.get('normalized_ingredient')
        target_ingredient_id = data.get('target_ingredient_id')
        note = data.get('note', 'Bulk Mapped')
        
        if not normalized_ingredient or not target_ingredient_id:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Find the target ingredient in database
        target_ingredient = mapper.ingredients_db[
            mapper.ingredients_db['ingredient_id'] == target_ingredient_id
        ]
        if target_ingredient.empty:
            return jsonify({'success': False, 'error': 'Target ingredient ID not found'}), 400
        target_ingredient = target_ingredient.iloc[0]
        
        # Find all unmapped instances of this normalized ingredient
        instances_to_map = []
        for i, item in enumerate(mapper.unmapped_ingredients):
            if item.get('normalized_ingredient', '').lower().strip() == normalized_ingredient.lower().strip():
                instances_to_map.append((i, item))
        
        if not instances_to_map:
            return jsonify({'success': False, 'error': 'No unmapped instances found for this ingredient'}), 400
        
        # Map all instances
        mapped_count = 0
        for i, item in reversed(instances_to_map):  # Reverse to maintain indices when removing
            # Create mapped record
            mapped_record = {
                'product_name': item.get('product_name', ''),
                'original_ingredient': item['original_ingredient'],
                'normalized_ingredient': item['normalized_ingredient'],
                'ingredient_id': target_ingredient_id,
                'matched_name': target_ingredient['name'],
                'status': 'mapped',
                'confidence': 100,
                'match_type': 'bulk_manual',
                'mapping_source': 'manual',
                'note': note,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to mapped results
            mapper.mapping_results.append(mapped_record)
            
            # Remove from unmapped
            mapper.unmapped_ingredients.pop(i)
            mapped_count += 1
        
        return jsonify({
            'success': True, 
            'mapped_count': mapped_count,
            'message': f'Successfully mapped {mapped_count} instances of "{normalized_ingredient}" to "{target_ingredient["name"]}"'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/product-status/<product_name>', methods=['GET'])
def get_single_product_status(product_name):
    """Get mapping status for a single product (optimized for performance)"""
    try:
        product_info = {'product_name': product_name, 'ingredients': [], 'mapped': [], 'unmapped': []}
        
        # Get mapped ingredients for this product
        for row in mapper.mapping_results:
            if row.get('product_name') == product_name:
                row_copy = row.copy()
                row_copy['is_mapped'] = True
                product_info['ingredients'].append(row_copy)
                product_info['mapped'].append(row_copy)
        
        # Get unmapped ingredients for this product
        for row in mapper.unmapped_ingredients:
            if row.get('product_name') == product_name:
                row_copy = row.copy()
                row_copy['is_mapped'] = False
                product_info['ingredients'].append(row_copy)
                product_info['unmapped'].append(row_copy)
        
        # Calculate summary stats
        total = len(product_info['ingredients'])
        mapped = len(product_info['mapped'])
        unmapped = len(product_info['unmapped'])
        fully_mapped = unmapped == 0 and total > 0
        
        product_info.update({
            'total_ingredients': total,
            'mapped_count': mapped,
            'unmapped_count': unmapped,
            'fully_mapped': fully_mapped
        })
        
        return jsonify({'product': to_python_type(product_info)})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-note', methods=['POST'])
def save_note():
    """Save or update note for any ingredient (mapped or unmapped)"""
    data = request.get_json()
    original_ingredient = data['original_ingredient']
    product_name = data.get('product_name', '')
    note = data.get('note', '')
    
    print(f"[SAVE-NOTE DEBUG] Looking for ingredient: '{original_ingredient}' in product: '{product_name}'")
    
    # First, try to find in MAPPED ingredients
    for i, item in enumerate(mapper.mapping_results):
        if (item.get('original_ingredient') == original_ingredient and 
            item.get('product_name') == product_name):
            print(f"[SAVE-NOTE DEBUG] Found in MAPPED list at index {i}, updating note")
            mapper.mapping_results[i]['note'] = note
            return jsonify({'success': True, 'location': 'mapped'})
    
    # If not found in mapped, try UNMAPPED ingredients
    for i, item in enumerate(mapper.unmapped_ingredients):
        if (item.get('original_ingredient') == original_ingredient and 
            item.get('product_name') == product_name):
            print(f"[SAVE-NOTE DEBUG] Found in UNMAPPED list at index {i}, updating note")
            mapper.unmapped_ingredients[i]['note'] = note
            return jsonify({'success': True, 'location': 'unmapped'})
    
    # If still not found, provide detailed debug info
    print(f"[SAVE-NOTE DEBUG] Ingredient not found in either list!")
    print("Available MAPPED ingredients (first 3):")
    for i, item in enumerate(mapper.mapping_results[:3]):
        print(f"  {i}: '{item.get('original_ingredient', '')}' in '{item.get('product_name', '')}'")
    print("Available UNMAPPED ingredients (first 3):")
    for i, item in enumerate(mapper.unmapped_ingredients[:3]):
        print(f"  {i}: '{item.get('original_ingredient', '')}' in '{item.get('product_name', '')}'")
    
    return jsonify({
        'success': False, 
        'error': f'Ingredient "{original_ingredient}" not found in either mapped or unmapped lists for product "{product_name}"'
    })

@app.route('/api/custom-patterns', methods=['GET'])
def get_custom_patterns():
    """Get current custom patterns for UI display"""
    try:
        logger.info("[PATTERN DEBUG] GET /api/custom-patterns called")
        patterns = mapper.get_pattern_override_interface()
        logger.info(f"[PATTERN DEBUG] Retrieved patterns: exact={len(patterns['exact_mappings'])}, rules={len(patterns['pattern_rules'])}")
        
        # Add ingredient names for display
        enhanced_exact = {}
        for pattern, ing_id in patterns['exact_mappings'].items():
            ing_row = mapper.ingredients_db[
                mapper.ingredients_db['ingredient_id'] == ing_id
            ]
            enhanced_exact[pattern] = {
                'id': ing_id,
                'name': ing_row.iloc[0]['name'] if not ing_row.empty else 'Unknown'
            }
        
        enhanced_rules = {}
        for pattern, ing_id in patterns['pattern_rules'].items():
            ing_row = mapper.ingredients_db[
                mapper.ingredients_db['ingredient_id'] == ing_id
            ]
            enhanced_rules[pattern] = {
                'id': ing_id,
                'name': ing_row.iloc[0]['name'] if not ing_row.empty else 'Unknown'
            }
        
        return jsonify({
            'success': True,
            'patterns': {
                'exact_mappings': enhanced_exact,
                'pattern_rules': enhanced_rules,
                'total_patterns': patterns['total_patterns']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/custom-patterns', methods=['POST'])
def add_custom_pattern():
    """Add a new custom pattern"""
    try:
        logger.info("[PATTERN DEBUG] POST /api/custom-patterns called")
        data = request.get_json()
        logger.info(f"[PATTERN DEBUG] Received data: {data}")
        pattern_type = data.get('type')  # 'exact' or 'regex'
        pattern = data.get('pattern', '').strip().lower()
        ingredient_id = data.get('ingredient_id')
        description = data.get('description', '')
        
        if not pattern or not ingredient_id:
            return jsonify({'success': False, 'error': 'Pattern and ingredient ID required'}), 400
        
        # Validate ingredient exists
        ing_row = mapper.ingredients_db[
            mapper.ingredients_db['ingredient_id'] == ingredient_id
        ]
        if ing_row.empty:
            return jsonify({'success': False, 'error': 'Ingredient ID not found'}), 400
        
        mapper.add_custom_pattern(pattern_type, pattern, ingredient_id, description)
        
        return jsonify({
            'success': True,
            'message': f'Pattern added: {pattern} -> {ing_row.iloc[0]["name"]}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/custom-patterns', methods=['DELETE'])
def delete_custom_pattern():
    """Delete a custom pattern"""
    try:
        logger.info("[PATTERN DEBUG] DELETE /api/custom-patterns called")
        data = request.get_json()
        pattern_type = data.get('type')  # 'exact' or 'regex'
        pattern = data.get('pattern', '').strip()
        
        logger.info(f"[PATTERN DEBUG] Deleting {pattern_type} pattern: {pattern}")
        
        if not pattern or not pattern_type:
            return jsonify({'success': False, 'error': 'Pattern and type required'}), 400
        
        # Remove from appropriate dictionary
        removed = False
        if pattern_type == "exact":
            pattern_key = pattern.lower()
            if pattern_key in mapper.exact_mappings:
                del mapper.exact_mappings[pattern_key]
                removed = True
        elif pattern_type == "regex":
            if pattern in mapper.pattern_rules:
                del mapper.pattern_rules[pattern]
                removed = True
        
        if not removed:
            return jsonify({'success': False, 'error': 'Pattern not found'}), 404
        
        # Save updated patterns to file
        mapper.save_custom_patterns()
        
        logger.info(f"[PATTERN DEBUG] Successfully deleted {pattern_type} pattern: {pattern}")
        return jsonify({
            'success': True,
            'message': f'Pattern deleted: {pattern}'
        })
    except Exception as e:
        logger.error(f"[PATTERN DEBUG] Error deleting pattern: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test-pattern', methods=['POST'])
def test_pattern():
    """Test a pattern against sample ingredients"""
    try:
        app.logger.info("[TEST-PATTERN] Starting test pattern API call")
        
        # Validate mapper state first
        app.logger.info(f"[TEST-PATTERN] Mapper validation:")
        app.logger.info(f"  - ingredients_db loaded: {mapper.ingredients_db is not None}")
        app.logger.info(f"  - products_df loaded: {mapper.products_df is not None}")
        app.logger.info(f"  - exact_mappings count: {len(mapper.exact_mappings)}")
        app.logger.info(f"  - pattern_rules count: {len(mapper.pattern_rules)}")
        
        if mapper.ingredients_db is None:
            app.logger.warning("[TEST-PATTERN] No ingredients database loaded - providing basic pattern testing")
            # Provide basic pattern testing without full database matching
            results = []
            for ingredient in test_ingredients:
                # Test only custom patterns without database matching
                exact_match = mapper.exact_mappings.get(ingredient.lower())
                pattern_match = None
                
                # Test regex patterns
                for pattern, ingredient_id in mapper.pattern_rules.items():
                    try:
                        import re
                        if re.search(pattern, ingredient.lower()):
                            pattern_match = {'pattern': pattern, 'ingredient_id': ingredient_id}
                            break
                    except Exception:
                        continue
                
                result = {
                    'original': str(ingredient),
                    'matched': f"Pattern Test (ID: {exact_match or (pattern_match['ingredient_id'] if pattern_match else 'None')})" if exact_match or pattern_match else None,
                    'matched_id': int(exact_match) if exact_match else (int(pattern_match['ingredient_id']) if pattern_match else None),
                    'confidence': 100 if exact_match or pattern_match else 0,
                    'match_type': 'exact_pattern' if exact_match else ('regex_pattern' if pattern_match else 'no_match'),
                    'alternatives': []
                }
                results.append(result)
            
            return jsonify({
                'success': True,
                'results': results,
                'note': 'Basic pattern testing only - load data for full matching'
            })
        
        data = request.get_json()
        test_ingredients = data.get('ingredients', [])
        app.logger.info(f"[TEST-PATTERN] Testing {len(test_ingredients)} ingredients: {test_ingredients}")
        
        results = []
        for i, ingredient in enumerate(test_ingredients):
            app.logger.info(f"[TEST-PATTERN] Processing ingredient {i+1}/{len(test_ingredients)}: '{ingredient}'")
            
            try:
                # Test with enhanced matching - note the method signature
                match_result, confidence, match_type, alternatives = mapper.find_best_match_enhanced(
                    ingredient.lower(), ingredient
                )
                
                app.logger.info(f"[TEST-PATTERN] Match result for '{ingredient}':")
                app.logger.info(f"  - match_result: {match_result}")
                app.logger.info(f"  - confidence: {confidence}")
                app.logger.info(f"  - match_type: {match_type}")
                app.logger.info(f"  - alternatives count: {len(alternatives) if alternatives else 0}")
                
                # Safely serialize alternatives
                safe_alternatives = []
                if alternatives:
                    for alt in alternatives[:3]:  # Top 3 alternatives
                        try:
                            safe_alt = {
                                'name': str(alt.get('name', '')),
                                'score': float(alt.get('score', 0)) if alt.get('score') is not None else 0,
                                'scorer': str(alt.get('scorer', ''))
                            }
                            safe_alternatives.append(safe_alt)
                        except Exception as alt_error:
                            app.logger.warning(f"[TEST-PATTERN] Error serializing alternative: {alt_error}")
                
                result_entry = {
                    'original': str(ingredient),
                    'matched': str(match_result['name']) if match_result is not None else None,
                    'matched_id': int(match_result['ingredient_id']) if match_result is not None else None,
                    'confidence': float(confidence) if confidence is not None else 0,
                    'match_type': str(match_type) if match_type is not None else 'unknown',
                    'alternatives': safe_alternatives
                }
                
                results.append(result_entry)
                app.logger.info(f"[TEST-PATTERN] Successfully processed '{ingredient}'")
                
            except Exception as ingredient_error:
                app.logger.error(f"[TEST-PATTERN] Error processing ingredient '{ingredient}': {ingredient_error}")
                import traceback
                app.logger.error(f"[TEST-PATTERN] Traceback: {traceback.format_exc()}")
                
                # Add error result for this ingredient
                results.append({
                    'original': ingredient,
                    'matched': None,
                    'matched_id': None,
                    'confidence': 0,
                    'match_type': f'error: {str(ingredient_error)}',
                    'alternatives': []
                })
        
        app.logger.info(f"[TEST-PATTERN] Completed testing {len(results)} ingredients")
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        app.logger.error(f"[TEST-PATTERN] Critical error: {str(e)}")
        import traceback
        app.logger.error(f"[TEST-PATTERN] Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pattern-samples', methods=['GET'])
def get_pattern_samples():
    """Get sample problematic ingredients for testing"""
    try:
        # Sample ingredients from your analysis
        samples = [
            "Protein Blend",
            "Whey Protein Isolate MILK",
            "Whey Protein Concentrate MILK", 
            "Micro Filtered Whey Protein Blend",
            "Premium Protein Blend",
            "Hydrolysed Whey Protein Isolate",
            "Plant Protein Blend",
            "Milk Protein Concentrate",
            "Soy Lecithin",
            "Natural Flavours",
            "Sweeteners",
            "Vitamin C"
        ]
        
        return jsonify({
            'success': True,
            'samples': samples
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patterns/export-csv', methods=['GET'])
def export_patterns_csv():
    """Export custom patterns as CSV for easy editing"""
    try:
        import tempfile
        import csv
        
        # Create temporary CSV file
        temp_dir = tempfile.gettempdir()
        filename = f'custom_patterns_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Type', 'Pattern', 'Target_Ingredient_ID', 'Target_Ingredient_Name', 'Description'])
            
            # Write exact mappings
            for pattern, ingredient_id in mapper.exact_mappings.items():
                # Get ingredient name from ID
                ingredient_name = "Unknown"
                if mapper.ingredients_db is not None:
                    match = mapper.ingredients_db[mapper.ingredients_db['ingredient_id'] == ingredient_id]
                    if not match.empty:
                        ingredient_name = match.iloc[0]['name']
                
                writer.writerow(['exact', pattern, ingredient_id, ingredient_name, 'Exact text match'])
            
            # Write regex patterns
            for pattern, ingredient_id in mapper.pattern_rules.items():
                # Get ingredient name from ID
                ingredient_name = "Unknown"
                if mapper.ingredients_db is not None:
                    match = mapper.ingredients_db[mapper.ingredients_db['ingredient_id'] == ingredient_id]
                    if not match.empty:
                        ingredient_name = match.iloc[0]['name']
                
                writer.writerow(['regex', pattern, ingredient_id, ingredient_name, 'Regular expression pattern'])
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        app.logger.error(f"Error exporting patterns: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patterns/import-csv', methods=['POST'])
def import_patterns_csv():
    """Import custom patterns from CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400
        
        # Backup current patterns before import
        backup_patterns()
        
        # Read and parse CSV
        import csv
        import io
        
        # Read file content
        file_content = file.read().decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(file_content))
        
        new_exact_mappings = {}
        new_pattern_rules = {}
        imported_count = 0
        errors = []
        
        for row_num, row in enumerate(csv_reader, start=2):
            try:
                pattern_type = row.get('Type', '').strip().lower()
                pattern = row.get('Pattern', '').strip()
                ingredient_id = row.get('Target_Ingredient_ID', '').strip()
                
                if not pattern or not ingredient_id:
                    errors.append(f"Row {row_num}: Missing pattern or ingredient ID")
                    continue
                
                # Validate ingredient ID
                try:
                    ingredient_id = int(ingredient_id)
                except ValueError:
                    errors.append(f"Row {row_num}: Invalid ingredient ID '{ingredient_id}'")
                    continue
                
                # Validate ingredient exists in database
                if mapper.ingredients_db is not None:
                    match = mapper.ingredients_db[mapper.ingredients_db['ingredient_id'] == ingredient_id]
                    if match.empty:
                        errors.append(f"Row {row_num}: Ingredient ID {ingredient_id} not found in database")
                        continue
                
                # Add to appropriate collection
                if pattern_type == 'exact':
                    new_exact_mappings[pattern.lower()] = ingredient_id
                    imported_count += 1
                elif pattern_type == 'regex':
                    # Validate regex pattern
                    try:
                        import re
                        re.compile(pattern)
                        new_pattern_rules[pattern] = ingredient_id
                        imported_count += 1
                    except re.error as e:
                        errors.append(f"Row {row_num}: Invalid regex pattern '{pattern}': {str(e)}")
                else:
                    errors.append(f"Row {row_num}: Invalid type '{pattern_type}'. Must be 'exact' or 'regex'")
                    
            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")
        
        # If there are critical errors, don't import
        if len(errors) > imported_count:
            return jsonify({
                'success': False, 
                'error': f'Too many errors in CSV file. Please fix and try again.',
                'errors': errors[:10]  # Show first 10 errors
            }), 400
        
        # Replace patterns with imported ones
        mapper.exact_mappings = new_exact_mappings
        mapper.pattern_rules = new_pattern_rules
        
        # Save to file
        mapper.save_custom_patterns()
        
        # Reload patterns to ensure consistency
        mapper.load_custom_patterns()
        
        result = {
            'success': True,
            'message': f'Successfully imported {imported_count} patterns',
            'imported_count': imported_count,
            'exact_count': len(new_exact_mappings),
            'regex_count': len(new_pattern_rules)
        }
        
        if errors:
            result['warnings'] = errors
            result['warning_count'] = len(errors)
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error importing patterns: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patterns/backup', methods=['POST'])
def create_patterns_backup():
    """Create a backup of current patterns"""
    try:
        backup_filename = f'custom_patterns_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        patterns_data = {
            'exact_mappings': mapper.exact_mappings,
            'pattern_rules': mapper.pattern_rules,
            'backup_created': datetime.now().isoformat(),
            'total_patterns': len(mapper.exact_mappings) + len(mapper.pattern_rules)
        }
        
        with open(backup_filename, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Backup created: {backup_filename}',
            'filename': backup_filename
        })
        
    except Exception as e:
        app.logger.error(f"Error creating backup: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def backup_patterns():
    """Internal function to backup patterns before risky operations"""
    try:
        backup_filename = f'auto_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        patterns_data = {
            'exact_mappings': mapper.exact_mappings,
            'pattern_rules': mapper.pattern_rules,
            'backup_created': datetime.now().isoformat(),
            'backup_type': 'automatic'
        }
        
        with open(backup_filename, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        app.logger.info(f"Auto-backup created: {backup_filename}")
        
    except Exception as e:
        app.logger.error(f"Auto-backup failed: {str(e)}")

@app.route('/api/patterns/restore', methods=['POST'])
def restore_patterns():
    """Restore patterns from a backup file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No backup file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Create current backup before restoring
        backup_patterns()
        
        # Read and parse backup file
        file_content = file.read().decode('utf-8')
        backup_data = json.loads(file_content)
        
        # Validate backup structure
        if 'exact_mappings' not in backup_data or 'pattern_rules' not in backup_data:
            return jsonify({'success': False, 'error': 'Invalid backup file format'}), 400
        
        # Restore patterns
        mapper.exact_mappings = backup_data['exact_mappings']
        mapper.pattern_rules = backup_data['pattern_rules']
        
        # Save restored patterns
        mapper.save_custom_patterns()
        
        # Reload to ensure consistency
        mapper.load_custom_patterns()
        
        return jsonify({
            'success': True,
            'message': 'Patterns restored successfully',
            'exact_count': len(mapper.exact_mappings),
            'regex_count': len(mapper.pattern_rules),
            'backup_date': backup_data.get('backup_created', 'Unknown')
        })
        
    except Exception as e:
        app.logger.error(f"Error restoring patterns: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/algorithm-stats', methods=['GET'])
def get_algorithm_stats():
    """Get statistics about the three-tier algorithm performance"""
    try:
        tier_stats = mapper.get_tier_statistics()
        
        # Calculate percentages
        total = tier_stats.get('total_processed', 0)
        if total > 0:
            percentages = {
                'tier1_exact_pct': (tier_stats.get('tier1_exact', 0) / total) * 100,
                'tier2_synonym_pct': (tier_stats.get('tier2_synonym', 0) / total) * 100,
                'tier2_fuzzy_pct': (tier_stats.get('tier2_fuzzy', 0) / total) * 100,
                'tier3_pattern_pct': (tier_stats.get('tier3_pattern', 0) / total) * 100,
                'tier3_advanced_pct': (tier_stats.get('tier3_advanced', 0) / total) * 100,
                'no_match_pct': (tier_stats.get('no_match', 0) / total) * 100
            }
        else:
            percentages = {
                'tier1_exact_pct': 0,
                'tier2_synonym_pct': 0, 
                'tier2_fuzzy_pct': 0,
                'tier3_pattern_pct': 0,
                'tier3_advanced_pct': 0,
                'no_match_pct': 0
            }
        
        # Get database optimization status
        optimization_status = {
            'exact_lookup_available': hasattr(mapper, 'exact_lookup'),
            'normalized_lookup_available': hasattr(mapper, 'normalized_lookup'),
            'exact_lookup_size': len(getattr(mapper, 'exact_lookup', {})),
            'normalized_lookup_size': len(getattr(mapper, 'normalized_lookup', {}))
        }
        
        return jsonify({
            'success': True,
            'tier_stats': tier_stats,
            'percentages': percentages,
            'optimization_status': optimization_status,
            'confidence_threshold': mapper.confidence_threshold,
            'total_patterns': len(mapper.exact_mappings) + len(mapper.pattern_rules),
            'exact_patterns': len(mapper.exact_mappings),
            'regex_patterns': len(mapper.pattern_rules)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/algorithm-summary', methods=['GET'])
def get_algorithm_summary():
    """Get comprehensive summary of the three-tier algorithm"""
    try:
        summary = mapper.get_algorithm_summary()
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import sys
    port = 5000
    # Check for --port argument
    for i, arg in enumerate(sys.argv):
        if arg in ('--port', '-p') and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except Exception:
                pass
    # Allow PORT env var override
    import os
    port = int(os.environ.get('PORT', port))
    app.run(debug=False, host='0.0.0.0', port=port) 
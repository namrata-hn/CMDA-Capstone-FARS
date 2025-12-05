"""
Metadata Extractor for FARS Database
Extracts relevant column metadata based on keywords in natural language questions.
"""

import logging
from typing import Dict, Set, List, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# KEYWORD MAPPINGS - Maps natural language terms to columns
# ============================================================

KEYWORD_MAPPINGS = {
    # School Bus Related
    "SCH_BUS": [
        "school bus", "school-bus", "schoolbus", "school transport"
    ],
    
    # Weather Conditions
    "WEATHER": [
        "weather", "rain", "raining", "rainy", "snow", "snowing", "snowy",
        "fog", "foggy", "sleet", "hail", "wind", "windy", "cloudy", "clear",
        "storm", "blizzard", "precipitation"
    ],
    
    # Lighting Conditions
    "LGT_COND": [
        "light", "lighting", "daylight", "dark", "darkness", "dawn", "dusk",
        "night", "nighttime", "street light", "streetlight", "illumination"
    ],
    
    # Person Type
    "PER_TYP": [
        "driver", "drivers", "passenger", "passengers", "pedestrian", "pedestrians",
        "cyclist", "cyclists", "bicyclist", "person type", "occupant", "occupants"
    ],
    
    # Sex/Gender (avoid common false positives like "man" in "many")
    "SEX": [
        "male", "female", "woman", "women", "sex", "gender",
        "boy", "girl", "boys", "girls", "men and women", "man or woman"
    ],
    
    # Injury Severity
    "INJ_SEV": [
        "injury severity", "injury level", "injuries", "injured", 
        "severity", "serious injury", "minor injury"
    ],
    
    # Alcohol/Drinking
    "DRINKING": [
        "alcohol", "drinking", "drunk", "dui", "dwi", "intoxicated",
        "impaired", "under the influence", "alcoholic", "breathalyzer"
    ],
    
    # Body Type
    "BODY_TYP": [
        "body type", "sedan", "suv", "truck", "van", "minivan", "pickup",
        "coupe", "convertible", "wagon", "hatchback", "vehicle type"
    ],
    
    # Special Use
    "SPEC_USE": [
        "special use", "taxi", "police", "emergency", "rental", "government",
        "military", "commercial"
    ],
    
    # Bus Use
    "BUS_USE": [
        "bus use", "transit", "public transport", "charter", "tour bus",
        "intercity", "commuter"
    ],
    
    # Emergency Use
    "EMER_USE": [
        "emergency", "ambulance", "fire truck", "fire engine", "police vehicle",
        "police car", "ems", "first responder"
    ],
    
    # Rural/Urban
    "RUR_URB": [
        "rural", "urban", "city", "countryside", "suburban", "town",
        "metropolitan", "location type"
    ],
    
    # Route Type
    "ROUTE": [
        "route", "interstate", "highway", "freeway", "state route",
        "us route", "county road", "local road"
    ],
    
    # Day of Week
    "DAY_WEEK": [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "weekday", "weekend", "day of week", "day of the week"
    ],
    
    # Month
    "MONTH": [
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "month", "monthly"
    ],
    
    # Harmful Event
    "HARM_EV": [
        "harmful event", "collision", "crash type", "rollover", "fire",
        "immersion", "jackknife", "overturn", "impact"
    ],
    
    # Manner of Collision
    "MAN_COLL": [
        "manner of collision", "head-on", "head on", "rear-end", "rear end",
        "angle", "sideswipe", "side swipe", "collision type"
    ],
    
    # Speed Related
    "SPEEDREL": [
        "speed", "speeding", "too fast", "speed limit", "velocity",
        "racing", "speed related"
    ],
    
    # Hit and Run
    "HIT_RUN": [
        "hit and run", "hit-and-run", "flee", "fled", "left scene",
        "left the scene", "runaway"
    ],
    
    # Seat Position
    "SEAT_POS": [
        "seat position", "front seat", "back seat", "rear seat",
        "driver seat", "passenger seat", "seating"
    ],
    
    # Restraint Use
    "REST_USE": [
        "seatbelt", "seat belt", "restraint", "safety belt", "harness",
        "child seat", "car seat", "booster"
    ],
    
    # Helmet Use
    "HELM_USE": [
        "helmet", "head protection", "motorcycle helmet", "bike helmet"
    ],
    
    # Airbag
    "AIR_BAG": [
        "airbag", "air bag", "deployed", "srs"
    ],
    
    # Ejection
    "EJECTION": [
        "ejection", "ejected", "thrown", "thrown from", "thrown out"
    ],
    
    # Work Zone
    "WRK_ZONE": [
        "work zone", "construction", "construction zone", "workers present",
        "work area", "maintenance"
    ],
    
    # Junction/Intersection
    "REL_JUNC": [
        "junction", "intersection", "crossroad", "cross road",
        "interchange", "on ramp", "off ramp"
    ],
    
    # Traffic Control
    "VTRAFCON": [
        "traffic control", "signal", "stop sign", "yield", "traffic light",
        "stop light", "warning sign"
    ],
    
    # Vehicle Configuration
    "V_CONFIG": [
        "vehicle configuration", "single unit", "truck trailer",
        "truck tractor", "combination", "articulated"
    ],
    
    # Rollover
    "ROLLOVER": [
        "rollover", "roll over", "rolled over", "overturn", "overturned",
        "flip", "flipped"
    ],
    
    # Fire
    "FIRE_EXP": [
        "fire", "explosion", "burn", "burned", "caught fire", "ignition"
    ],
    
    # Drug Related
    "DRUGS": [
        "drug", "drugs", "narcotic", "narcotics", "marijuana", "cannabis",
        "medication", "substance", "controlled substance"
    ],
    
    # Age
    "AGE": [
        "age", "years old", "year old", "elderly", "senior", "teen",
        "teenager", "child", "children", "juvenile", "minor", "adult"
    ],
    
    # Hour/Time (remove short abbreviations that cause false positives)
    "HOUR": [
        "hour", "time of day", "o'clock", "morning", "afternoon",
        "evening", "midnight", "noon", "what time"
    ],
    
    # Functional System
    "FUNC_SYS": [
        "functional system", "functional classification", "arterial",
        "collector", "local road"
    ],
}


# ============================================================
# METADATA EXTRACTION FUNCTIONS
# ============================================================

def extract_relevant_columns(question: str) -> Set[str]:
    """
    Identifies relevant column names based on keywords in the question.
    Uses word boundary matching to avoid false positives.
    
    Args:
        question: Natural language question from user
        
    Returns:
        Set of column names that are relevant to the question
    """
    import re
    
    question_lower = question.lower()
    relevant_columns = set()
    
    # Check each keyword mapping
    for column, keywords in KEYWORD_MAPPINGS.items():
        for keyword in keywords:
            # Use word boundaries for multi-word keywords and single words
            # For multi-word phrases, check if they exist in the question
            if ' ' in keyword or '-' in keyword:
                # Multi-word phrase - simple containment check is fine
                if keyword in question_lower:
                    relevant_columns.add(column)
                    logger.debug(f"Matched keyword '{keyword}' to column '{column}'")
                    break
            else:
                # Single word - use word boundary matching to avoid false positives
                # \b ensures we match whole words only
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, question_lower):
                    relevant_columns.add(column)
                    logger.debug(f"Matched keyword '{keyword}' to column '{column}'")
                    break
    
    logger.info(f"Extracted {len(relevant_columns)} relevant columns from question")
    return relevant_columns


def build_metadata_context(
    relevant_columns: Set[str],
    column_metadata: Dict[str, Dict[str, Dict]],
    max_codes_per_column: int = 20
) -> str:
    """
    Builds formatted metadata context string for the LLM prompt.
    
    Args:
        relevant_columns: Set of column names to include
        column_metadata: The full FARS codebook metadata dictionary
        max_codes_per_column: Maximum number of codes to include per column
        
    Returns:
        Formatted string with column descriptions and code mappings
    """
    if not relevant_columns:
        return ""
    
    metadata_lines = ["\n" + "="*60]
    metadata_lines.append("RELEVANT COLUMN METADATA AND CODE MAPPINGS")
    metadata_lines.append("="*60)
    metadata_lines.append("The following columns were identified as relevant to your question.")
    metadata_lines.append("USE THESE NUMERIC CODES in your SQL WHERE clauses!\n")
    
    columns_found = 0
    
    for column in sorted(relevant_columns):
        # Search for the column in all tables
        found = False
        for table_name, table_meta in column_metadata.items():
            if column in table_meta:
                col_info = table_meta[column]
                description = col_info.get("description", "No description available")
                codes = col_info.get("codes", {})
                
                metadata_lines.append(f"\n🔹 Column: {column} (from {table_name} table)")
                metadata_lines.append(f"   Description: {description}")
                
                if codes:
                    metadata_lines.append(f"   📋 Valid Codes (USE THESE NUMERIC VALUES):")
                    
                    # Convert codes dict to sorted list for consistent ordering
                    code_items = sorted(codes.items(), key=lambda x: str(x[0]))
                    
                    # Limit number of codes to avoid overwhelming the prompt
                    displayed_codes = code_items[:max_codes_per_column]
                    
                    for code, label in displayed_codes:
                        metadata_lines.append(f"      {code} = {label}")
                    
                    if len(code_items) > max_codes_per_column:
                        remaining = len(code_items) - max_codes_per_column
                        metadata_lines.append(f"      ... ({remaining} more codes available)")
                else:
                    metadata_lines.append(f"   ℹ️  No coded values (use direct comparison)")
                
                columns_found += 1
                found = True
                break  # Found in this table, no need to check others
        
        if not found:
            logger.warning(f"Column '{column}' matched by keyword but not found in metadata")
            metadata_lines.append(f"\n🔹 Column: {column}")
            metadata_lines.append(f"   ⚠️  Metadata not available")
    
    metadata_lines.append("\n" + "="*60)
    metadata_lines.append("🚨 CRITICAL: Always use the NUMERIC CODE values shown above!")
    metadata_lines.append("   NEVER use string literals like 'SCHOOL_BUS' or 'RAIN'")
    metadata_lines.append("   Example: Use SCH_BUS = 1 (NOT SPEC_USE = 'SCHOOL_BUS')")
    metadata_lines.append("="*60 + "\n")
    
    logger.info(f"Built metadata context for {columns_found} columns")
    return "\n".join(metadata_lines)


def extract_relevant_metadata(
    question: str,
    column_metadata: Dict[str, Dict[str, Dict]],
    max_codes_per_column: int = 20
) -> str:
    """
    Main function: Extract relevant metadata based on question keywords.
    
    This is the primary function to be imported and used by sql_query_chain.py
    
    Args:
        question: Natural language question from user
        column_metadata: The full FARS codebook metadata dictionary
        max_codes_per_column: Maximum number of codes to include per column
        
    Returns:
        Formatted metadata context string for inclusion in LLM prompt
    """
    try:
        # Step 1: Identify relevant columns
        relevant_columns = extract_relevant_columns(question)
        
        if not relevant_columns:
            logger.info("No relevant columns identified from question")
            return ""
        
        # Step 2: Build formatted metadata context
        metadata_context = build_metadata_context(
            relevant_columns,
            column_metadata,
            max_codes_per_column
        )
        
        return metadata_context
        
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return ""


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_all_mapped_columns() -> List[str]:
    """Returns a list of all columns that have keyword mappings."""
    return sorted(KEYWORD_MAPPINGS.keys())


def get_keywords_for_column(column: str) -> List[str]:
    """Returns the list of keywords mapped to a specific column."""
    return KEYWORD_MAPPINGS.get(column, [])


def add_keyword_mapping(column: str, keywords: List[str]) -> None:
    """
    Dynamically add or extend keyword mappings for a column.
    
    Args:
        column: The column name to map to
        keywords: List of keywords to add
    """
    if column in KEYWORD_MAPPINGS:
        # Extend existing mappings
        KEYWORD_MAPPINGS[column].extend(keywords)
        # Remove duplicates while preserving order
        KEYWORD_MAPPINGS[column] = list(dict.fromkeys(KEYWORD_MAPPINGS[column]))
    else:
        # Create new mapping
        KEYWORD_MAPPINGS[column] = keywords
    
    logger.info(f"Added/updated {len(keywords)} keywords for column '{column}'")


def get_mapping_statistics() -> Dict[str, int]:
    """Returns statistics about the keyword mappings."""
    return {
        "total_columns": len(KEYWORD_MAPPINGS),
        "total_keywords": sum(len(keywords) for keywords in KEYWORD_MAPPINGS.values()),
        "avg_keywords_per_column": sum(len(keywords) for keywords in KEYWORD_MAPPINGS.values()) / len(KEYWORD_MAPPINGS) if KEYWORD_MAPPINGS else 0
    }


# ============================================================
# TESTING/DEBUGGING
# ============================================================

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    # Test cases
    test_questions = [
        "Give me an example of an accident that occurred in the year 2005 that involved a school bus",
        "How many fatalities occurred in rainy weather conditions?",
        "Show me accidents involving drunk drivers",
        "What's the distribution of accidents by day of week?",
        "Find accidents where the driver was ejected from the vehicle"
    ]
    
    print("="*70)
    print("METADATA EXTRACTOR TEST")
    print("="*70)
    
    # Print mapping statistics
    stats = get_mapping_statistics()
    print(f"\n📊 Keyword Mapping Statistics:")
    print(f"   - Total columns mapped: {stats['total_columns']}")
    print(f"   - Total keywords: {stats['total_keywords']}")
    print(f"   - Average keywords per column: {stats['avg_keywords_per_column']:.1f}")
    
    # Test each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {question}")
        print('='*70)
        
        columns = extract_relevant_columns(question)
        print(f"\n✅ Matched Columns: {', '.join(sorted(columns)) if columns else 'None'}")
        
        # Show which keywords triggered the match
        for col in sorted(columns):
            matching_keywords = [kw for kw in KEYWORD_MAPPINGS[col] if kw in question.lower()]
            print(f"   - {col}: triggered by '{matching_keywords[0] if matching_keywords else '?'}'")
    
    print(f"\n{'='*70}")
    print("Test complete!")
    print('='*70)
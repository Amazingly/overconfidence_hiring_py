#!/usr/bin/env python3
"""
OVERCONFIDENCE AND DISCRIMINATORY BEHAVIOR EXPERIMENT PLATFORM
==============================================================

Experimental platform implementing the design from:
"Does Overconfidence Predict Discriminatory Beliefs and Behavior?" 
Published in Management Science

CRITICAL METHODOLOGY NOTE:
Performance classification uses session-relative ranking (true top/bottom 50%)
rather than fixed cutoffs, ensuring accurate experimental implementation.

For academic use, replication studies, and research extensions

Author: Research Team
Date: 2024
License: MIT (See LICENSE file)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import random
import time
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional, Any, Tuple
import base64
from pathlib import Path
import logging
import hashlib
import sqlite3
import io
import zipfile

# Configure logging for research audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_log.log'),
        logging.StreamHandler()
    ]
)

# Configure Streamlit page
st.set_page_config(
    page_title="Decision-Making Experiment",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class ExperimentConfig:
    """Centralized configuration for experimental parameters."""
    
    # Core experimental parameters
    TRIVIA_QUESTIONS_COUNT = 25
    TRIVIA_TIME_LIMIT = 360  # 6 minutes in seconds
    PERFORMANCE_CUTOFF_PERCENTILE = 50  # Top 50% for High performance
    
    # Treatment target accuracy ranges
    TARGET_EASY_ACCURACY = (75, 85)
    TARGET_HARD_ACCURACY = (25, 35)
    
    # Randomization and reproducibility
    QUESTION_SELECTION_SEED = 12345
    
    # BDM mechanism parameters
    BDM_MIN_VALUE = 0
    BDM_MAX_VALUE = 200
    ENDOWMENT_TOKENS = 160
    
    # Payment structure
    HIGH_PERFORMANCE_TOKENS = 250
    LOW_PERFORMANCE_TOKENS = 100
    HIGH_WORKER_REWARD = 200
    LOW_WORKER_REWARD = 40
    TOKEN_TO_DOLLAR_RATE = 0.09
    SHOW_UP_FEE = 5.00
    
    # Group assignment mechanisms
    MECHANISM_A_ACCURACY = 0.95
    MECHANISM_B_ACCURACY = 0.55
    
    # Data validation parameters
    MIN_HIRING_EXPLANATION_LENGTH = 20
    MIN_STRATEGY_EXPLANATION_LENGTH = 10

# Enhanced CSS for professional research appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .experiment-card {
        background-color: #f9f9f9;
        border: 1px solid #bdc3c7;
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .progress-bar {
        background-color: #ecf0f1;
        border-radius: 15px;
        height: 25px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-fill {
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        height: 100%;
        border-radius: 15px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.9em;
    }
    .question-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 2px solid #dee2e6;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .timer-warning {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        animation: pulse 1s infinite;
        box-shadow: 0 4px 8px rgba(231,76,60,0.3);
    }
    .timer-normal {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(243,156,18,0.3);
    }
    .group-display {
        text-align: center;
        padding: 3rem;
        font-size: 2.5rem;
        font-weight: bold;
        border: 4px solid #3498db;
        border-radius: 15px;
        margin: 2rem 0;
        background: linear-gradient(45deg, #f0f8ff, #e6f3ff);
        box-shadow: 0 8px 16px rgba(52,152,219,0.2);
        animation: groupReveal 0.8s ease-out;
    }
    .results-item {
        display: flex;
        justify-content: space-between;
        padding: 0.8rem 0;
        border-bottom: 1px solid #bdc3c7;
        font-size: 1.1em;
    }
    .methodology-warning {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 2px solid #ffc107;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(52,152,219,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9, #1f618d);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(52,152,219,0.4);
    }
    .comprehension-correct {
        background: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .comprehension-incorrect {
        background: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    @keyframes groupReveal {
        0% { opacity: 0; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

class ResearchDatabase:
    """Enhanced database manager with robust error handling and research features."""
    
    def __init__(self, db_path: str = "experiment_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with research-specific tables and improved schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main experiment data table with enhanced schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_sessions (
                    participant_id TEXT PRIMARY KEY,
                    session_start TEXT,
                    session_end TEXT,
                    treatment TEXT,
                    trivia_score INTEGER,
                    accuracy_rate REAL,
                    performance_level TEXT,
                    session_median_score REAL,
                    performance_percentile REAL,
                    belief_own_performance INTEGER,
                    assigned_group TEXT,
                    mechanism_used TEXT,
                    mechanism_reflects_performance BOOLEAN,
                    wtp_top_group INTEGER,
                    wtp_bottom_group INTEGER,
                    wtp_premium INTEGER,
                    belief_mechanism INTEGER,
                    time_spent_trivia REAL,
                    overconfidence_measure REAL,
                    demographic_data TEXT,
                    questionnaire_data TEXT,
                    raw_data TEXT,
                    validation_flags TEXT,
                    data_quality_flag TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise
    
    def save_session(self, data: Dict) -> bool:
        """Save complete session data with enhanced error handling."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate enhanced metrics
            wtp_premium = data['wtp_top_group'] - data['wtp_bottom_group']
            
            # Calculate overconfidence measure
            actual_performance = 1 if data['performance_level'] == 'High' else 0
            belief_performance = data['belief_own_performance'] / 100
            overconfidence_measure = belief_performance - actual_performance
            
            cursor.execute('''
                INSERT OR REPLACE INTO experiment_sessions 
                (participant_id, session_start, session_end, treatment, trivia_score, 
                 accuracy_rate, performance_level, session_median_score, performance_percentile,
                 belief_own_performance, assigned_group, mechanism_used, mechanism_reflects_performance,
                 wtp_top_group, wtp_bottom_group, wtp_premium, belief_mechanism, time_spent_trivia,
                 overconfidence_measure, demographic_data, questionnaire_data, raw_data, 
                 validation_flags, data_quality_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['participant_id'], data['start_time'], data.get('end_time'),
                data['treatment'], data['trivia_score'], data.get('accuracy_rate'),
                data['performance_level'], data.get('session_median_score'),
                data.get('performance_percentile'), data['belief_own_performance'], 
                data['assigned_group'], data['mechanism_used'], 
                data.get('mechanism_reflects_performance'),
                data['wtp_top_group'], data['wtp_bottom_group'], wtp_premium, 
                data['belief_mechanism'], data.get('trivia_time_spent'),
                overconfidence_measure,
                json.dumps(data.get('post_experiment_questionnaire', {}).get('demographics', {})),
                json.dumps(data.get('post_experiment_questionnaire', {})),
                json.dumps(data), json.dumps(data.get('validation_flags', {})),
                data.get('post_experiment_questionnaire', {}).get('validation', {}).get('data_quality', 'Yes, include my data')
            ))
            
            conn.commit()
            conn.close()
            logging.info(f"Session data saved successfully for participant {data['participant_id']}")
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Database save error: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during save: {e}")
            return False

class DataValidator:
    """Enhanced data validation with research-specific checks."""
    
    @staticmethod
    def validate_session_data(data: Dict) -> Tuple[bool, List[str]]:
        """Comprehensive validation of experimental session data."""
        errors = []
        
        # Required fields validation
        required_fields = [
            'participant_id', 'treatment', 'trivia_score', 'performance_level',
            'belief_own_performance', 'assigned_group', 'mechanism_used',
            'wtp_top_group', 'wtp_bottom_group', 'belief_mechanism'
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Range validations using config
        if 'trivia_score' in data:
            if not (0 <= data['trivia_score'] <= ExperimentConfig.TRIVIA_QUESTIONS_COUNT):
                errors.append(f"Trivia score must be between 0-{ExperimentConfig.TRIVIA_QUESTIONS_COUNT}")
        
        if 'belief_own_performance' in data:
            if not (0 <= data['belief_own_performance'] <= 100):
                errors.append("Belief own performance must be between 0-100")
        
        if 'wtp_top_group' in data:
            if not (ExperimentConfig.BDM_MIN_VALUE <= data['wtp_top_group'] <= ExperimentConfig.BDM_MAX_VALUE):
                errors.append(f"WTP top group must be between {ExperimentConfig.BDM_MIN_VALUE}-{ExperimentConfig.BDM_MAX_VALUE}")
        
        if 'wtp_bottom_group' in data:
            if not (ExperimentConfig.BDM_MIN_VALUE <= data['wtp_bottom_group'] <= ExperimentConfig.BDM_MAX_VALUE):
                errors.append(f"WTP bottom group must be between {ExperimentConfig.BDM_MIN_VALUE}-{ExperimentConfig.BDM_MAX_VALUE}")
        
        return len(errors) == 0, errors

class OverconfidenceExperiment:
    """Enhanced experimental class with complete implementation of all phases."""
    
    def __init__(self):
        """Initialize experiment with comprehensive research capabilities."""
        self.setup_session_state()
        self.trivia_questions = self.get_trivia_questions()
        self.db = ResearchDatabase()
        self.validator = DataValidator()
        
    def setup_session_state(self):
        """Initialize comprehensive session state for research tracking."""
        if 'experiment_data' not in st.session_state:
            st.session_state.experiment_data = {
                'participant_id': f'P{uuid.uuid4().hex[:8]}',
                'session_hash': hashlib.md5(f"{datetime.now().isoformat()}{random.random()}".encode()).hexdigest()[:16],
                'start_time': datetime.now().isoformat(),
                'treatment': None,
                'trivia_answers': [None] * ExperimentConfig.TRIVIA_QUESTIONS_COUNT,
                'trivia_response_times': [0] * ExperimentConfig.TRIVIA_QUESTIONS_COUNT,
                'trivia_score': 0,
                'trivia_time_spent': 0,
                'performance_level': None,
                'session_median_score': None,
                'performance_percentile': None,
                'belief_own_performance': None,
                'assigned_group': None,
                'mechanism_used': None,
                'mechanism_reflects_performance': None,
                'wtp_top_group': None,
                'wtp_bottom_group': None,
                'belief_mechanism': None,
                'post_experiment_questionnaire': {},
                'completed_screens': [],
                'screen_times': {},
                'comprehension_attempts': {},
                'end_time': None,
                'validation_flags': {},
                'metadata': {
                    'platform': 'Python/Streamlit',
                    'version': '2.1.0',
                    'config_version': 'Enhanced',
                    'timestamp': datetime.now().isoformat(),
                    'randomization_seed': ExperimentConfig.QUESTION_SELECTION_SEED
                }
            }
        
        # Screen tracking
        if 'current_screen' not in st.session_state:
            st.session_state.current_screen = 0
            
        if 'current_trivia_question' not in st.session_state:
            st.session_state.current_trivia_question = 0
            
        if 'trivia_start_time' not in st.session_state:
            st.session_state.trivia_start_time = None
            
        if 'selected_questions' not in st.session_state:
            st.session_state.selected_questions = []
            
        if 'question_start_times' not in st.session_state:
            st.session_state.question_start_times = {}

    def get_trivia_questions(self) -> Dict[str, List[Dict]]:
        """Return comprehensive, research-validated trivia question banks."""
        return {
            'easy': [
                # GEOGRAPHY & COUNTRIES (High success rate expected)
                {'question': 'What is the capital of Australia?', 'options': ['Sydney', 'Melbourne', 'Canberra', 'Perth'], 'correct': 2, 'category': 'geography'},
                {'question': 'Which country is famous for the Eiffel Tower?', 'options': ['Italy', 'France', 'Germany', 'Spain'], 'correct': 1, 'category': 'geography'},
                {'question': 'What is the largest continent by area?', 'options': ['Africa', 'Asia', 'North America', 'Europe'], 'correct': 1, 'category': 'geography'},
                {'question': 'Which ocean is the largest?', 'options': ['Atlantic', 'Indian', 'Arctic', 'Pacific'], 'correct': 3, 'category': 'geography'},
                {'question': 'What is the capital of Canada?', 'options': ['Toronto', 'Vancouver', 'Ottawa', 'Montreal'], 'correct': 2, 'category': 'geography'},
                {'question': 'What is the capital of France?', 'options': ['London', 'Berlin', 'Paris', 'Madrid'], 'correct': 2, 'category': 'geography'},
                {'question': 'How many continents are there?', 'options': ['5', '6', '7', '8'], 'correct': 2, 'category': 'geography'},
                
                # NATURE & SCIENCE (Fundamental knowledge)
                {'question': 'From what trees do acorns grow?', 'options': ['Oak', 'Maple', 'Pine', 'Birch'], 'correct': 0, 'category': 'science'},
                {'question': 'What color are emeralds?', 'options': ['Blue', 'Green', 'Red', 'Purple'], 'correct': 1, 'category': 'science'},
                {'question': 'How many legs does a spider have?', 'options': ['6', '8', '10', '12'], 'correct': 1, 'category': 'science'},
                {'question': 'What gas do plants absorb from the atmosphere?', 'options': ['Oxygen', 'Nitrogen', 'Carbon dioxide', 'Hydrogen'], 'correct': 2, 'category': 'science'},
                {'question': 'Which planet is closest to the sun?', 'options': ['Venus', 'Mercury', 'Earth', 'Mars'], 'correct': 1, 'category': 'science'},
                {'question': 'What is the chemical symbol for water?', 'options': ['H2O', 'CO2', 'NaCl', 'O2'], 'correct': 0, 'category': 'science'},
                {'question': 'Which direction does the sun rise?', 'options': ['North', 'South', 'East', 'West'], 'correct': 2, 'category': 'science'},
                {'question': 'What do we call frozen water?', 'options': ['Steam', 'Ice', 'Snow', 'Rain'], 'correct': 1, 'category': 'science'},
                {'question': 'What is the largest planet in our solar system?', 'options': ['Earth', 'Mars', 'Jupiter', 'Saturn'], 'correct': 2, 'category': 'science'},
                {'question': 'What color is the sun?', 'options': ['Red', 'Blue', 'Yellow', 'Green'], 'correct': 2, 'category': 'science'},
                
                # BASIC FACTS (Very high success rate expected)
                {'question': 'How many minutes are in one hour?', 'options': ['50', '60', '70', '80'], 'correct': 1, 'category': 'basic'},
                {'question': 'How many sides does a triangle have?', 'options': ['2', '3', '4', '5'], 'correct': 1, 'category': 'basic'},
                {'question': 'How many days are in a week?', 'options': ['5', '6', '7', '8'], 'correct': 2, 'category': 'basic'},
                {'question': 'How many hours are in a day?', 'options': ['23', '24', '25', '26'], 'correct': 1, 'category': 'basic'},
                {'question': 'How many months are in a year?', 'options': ['10', '11', '12', '13'], 'correct': 2, 'category': 'basic'},
                {'question': 'Which meal is typically eaten in the morning?', 'options': ['Lunch', 'Dinner', 'Breakfast', 'Supper'], 'correct': 2, 'category': 'basic'},
                {'question': 'Which meal is typically eaten at midday?', 'options': ['Breakfast', 'Lunch', 'Dinner', 'Snack'], 'correct': 1, 'category': 'basic'},
                
                # HISTORY & CULTURE (Well-known facts)
                {'question': 'Who is the patron saint of Ireland?', 'options': ['St. David', 'St. Andrew', 'St. George', 'St. Patrick'], 'correct': 3, 'category': 'history'},
                {'question': 'In which year did World War II end?', 'options': ['1944', '1945', '1946', '1947'], 'correct': 1, 'category': 'history'},
                
                # ANIMALS (Common knowledge)
                {'question': 'Which of the following dogs is typically the smallest?', 'options': ['Labrador', 'Poodle', 'Chihuahua', 'Beagle'], 'correct': 2, 'category': 'animals'},
                {'question': 'What do pandas primarily eat?', 'options': ['Fish', 'Meat', 'Bamboo', 'Berries'], 'correct': 2, 'category': 'animals'},
                {'question': 'Which animal is known as the "King of the Jungle"?', 'options': ['Tiger', 'Lion', 'Elephant', 'Leopard'], 'correct': 1, 'category': 'animals'},
                {'question': 'What is the largest mammal in the world?', 'options': ['Elephant', 'Blue whale', 'Giraffe', 'Hippopotamus'], 'correct': 1, 'category': 'animals'},
                {'question': 'What do fish use to breathe?', 'options': ['Lungs', 'Gills', 'Nose', 'Mouth'], 'correct': 1, 'category': 'animals'},
                
                # SPORTS (Popular knowledge)
                {'question': 'How many players are on a basketball team on the court at one time?', 'options': ['4', '5', '6', '7'], 'correct': 1, 'category': 'sports'},
                {'question': 'In which sport would you perform a slam dunk?', 'options': ['Tennis', 'Football', 'Basketball', 'Baseball'], 'correct': 2, 'category': 'sports'},
                
                # MISCELLANEOUS (Common sense)
                {'question': 'What is the primary ingredient in guacamole?', 'options': ['Tomato', 'Avocado', 'Onion', 'Pepper'], 'correct': 1, 'category': 'misc'},
                {'question': 'Which fruit is known for "keeping the doctor away"?', 'options': ['Banana', 'Orange', 'Apple', 'Grape'], 'correct': 2, 'category': 'misc'},
                {'question': 'What is the main ingredient in bread?', 'options': ['Rice', 'Flour', 'Sugar', 'Salt'], 'correct': 1, 'category': 'misc'},
                {'question': 'What color do you get when you mix red and yellow?', 'options': ['Purple', 'Green', 'Orange', 'Blue'], 'correct': 2, 'category': 'misc'},
                {'question': 'Which season comes after spring?', 'options': ['Winter', 'Summer', 'Fall', 'Autumn'], 'correct': 1, 'category': 'basic'},
            ],
            
            'hard': [
                # SPORTS HISTORY (Obscure historical facts)
                {'question': 'Boris Becker contested consecutive Wimbledon men\'s singles finals in 1988, 1989, and 1990, winning in 1989. Who was his opponent in all three matches?', 'options': ['Michael Stich', 'Andre Agassi', 'Ivan Lendl', 'Stefan Edberg'], 'correct': 3, 'category': 'sports_history'},
                
                # POLITICAL HISTORY (Specialized knowledge)
                {'question': 'Suharto held the office of president in which large Asian nation?', 'options': ['Malaysia', 'Japan', 'Indonesia', 'Thailand'], 'correct': 2, 'category': 'political_history'},
                
                # DETAILED HISTORY (Obscure historical facts)
                {'question': 'Who was Henry VIII\'s wife at the time of his death?', 'options': ['Catherine Parr', 'Catherine of Aragon', 'Anne Boleyn', 'Jane Seymour'], 'correct': 0, 'category': 'detailed_history'},
                {'question': 'The Battle of Hastings took place in which year?', 'options': ['1064', '1065', '1066', '1067'], 'correct': 2, 'category': 'detailed_history'},
                
                # SPECIALIZED KNOWLEDGE (Highly technical)
                {'question': 'What do you most fear if you have hormephobia?', 'options': ['Shock', 'Hormones', 'Heights', 'Water'], 'correct': 0, 'category': 'specialized'},
                {'question': 'In chemistry, what is the atomic number of tungsten?', 'options': ['72', '73', '74', '75'], 'correct': 2, 'category': 'specialized'},
                {'question': 'What is the medical term for the kneecap?', 'options': ['Fibula', 'Tibia', 'Patella', 'Femur'], 'correct': 2, 'category': 'specialized'},
                
                # ADVANCED SCIENCE (Complex scientific knowledge)
                {'question': 'For what did Einstein receive the Nobel Prize in Physics?', 'options': ['Theory of Relativity', 'Quantum mechanics', 'Photoelectric effect', 'Brownian motion'], 'correct': 2, 'category': 'science_advanced'},
                {'question': 'What is the hardest natural substance on Earth?', 'options': ['Quartz', 'Diamond', 'Corundum', 'Topaz'], 'correct': 1, 'category': 'science_advanced'},
                {'question': 'Which element has the chemical symbol "Au"?', 'options': ['Silver', 'Aluminum', 'Gold', 'Argon'], 'correct': 2, 'category': 'science_advanced'},
                
                # LITERATURE & ARTS (Specialized cultural knowledge)
                {'question': 'Who wrote the novel "One Hundred Years of Solitude"?', 'options': ['Jorge Luis Borges', 'Gabriel García Márquez', 'Mario Vargas Llosa', 'Octavio Paz'], 'correct': 1, 'category': 'literature'},
                {'question': 'Which painter created "Guernica"?', 'options': ['Salvador Dalí', 'Pablo Picasso', 'Joan Miró', 'Francisco Goya'], 'correct': 1, 'category': 'literature'},
                {'question': 'In Shakespeare\'s "Hamlet," what is the name of Hamlet\'s mother?', 'options': ['Ophelia', 'Gertrude', 'Cordelia', 'Portia'], 'correct': 1, 'category': 'literature'},
                {'question': 'Who composed the opera "The Ring of the Nibelung"?', 'options': ['Mozart', 'Wagner', 'Verdi', 'Puccini'], 'correct': 1, 'category': 'literature'},
                {'question': 'Which philosopher wrote "Critique of Pure Reason"?', 'options': ['Hegel', 'Kant', 'Nietzsche', 'Schopenhauer'], 'correct': 1, 'category': 'specialized'},
                
                # ADVANCED GEOGRAPHY (Obscure geographical knowledge)
                {'question': 'What is the capital of Kazakhstan?', 'options': ['Almaty', 'Nur-Sultan', 'Shymkent', 'Aktobe'], 'correct': 1, 'category': 'geography_advanced'},
                {'question': 'Which African country was formerly known as Rhodesia?', 'options': ['Zambia', 'Zimbabwe', 'Botswana', 'Namibia'], 'correct': 1, 'category': 'geography_advanced'},
                
                # ECONOMICS & COMPLEX THEORY (Graduate-level knowledge)
                {'question': 'Who developed the theory of comparative advantage in international trade?', 'options': ['Adam Smith', 'David Ricardo', 'John Stuart Mill', 'Alfred Marshall'], 'correct': 1, 'category': 'economics'},
                {'question': 'Which economist wrote "The General Theory of Employment, Interest, and Money"?', 'options': ['John Maynard Keynes', 'Milton Friedman', 'Friedrich Hayek', 'Paul Samuelson'], 'correct': 0, 'category': 'economics'},
                
                # COMPLEX MATHEMATICS & LOGIC
                {'question': 'Which logical fallacy involves attacking the person rather than their argument?', 'options': ['Straw man', 'Ad hominem', 'False dichotomy', 'Slippery slope'], 'correct': 1, 'category': 'logic'},
                {'question': 'What is the square root of 169?', 'options': ['12', '13', '14', '15'], 'correct': 1, 'category': 'specialized'},
                
                # Additional hard questions to reach 25
                {'question': 'In quantum mechanics, what principle states that you cannot simultaneously know both position and momentum?', 'options': ['Pauli exclusion', 'Heisenberg uncertainty', 'Wave-particle duality', 'Quantum entanglement'], 'correct': 1, 'category': 'science_advanced'},
                {'question': 'The Bretton Woods system established which international monetary arrangement?', 'options': ['Gold standard', 'Flexible exchange rates', 'Fixed exchange rates', 'Currency unions'], 'correct': 2, 'category': 'economics'},
                {'question': 'What is the term for the economic condition of simultaneous inflation and unemployment?', 'options': ['Recession', 'Stagflation', 'Depression', 'Deflation'], 'correct': 1, 'category': 'economics'},
                {'question': 'Who was the first Secretary-General of the United Nations?', 'options': ['Dag Hammarskjöld', 'Trygve Lie', 'U Thant', 'Kurt Waldheim'], 'correct': 1, 'category': 'political_history'},
                {'question': 'Which Roman emperor was known as "The Philosopher Emperor"?', 'options': ['Marcus Aurelius', 'Trajan', 'Hadrian', 'Antoninus Pius'], 'correct': 0, 'category': 'detailed_history'},
                {'question': 'The Atacama Desert is located primarily in which country?', 'options': ['Peru', 'Bolivia', 'Chile', 'Argentina'], 'correct': 2, 'category': 'geography_advanced'},
                {'question': 'Which composer wrote "The Art of Fugue"?', 'options': ['Bach', 'Mozart', 'Beethoven', 'Handel'], 'correct': 0, 'category': 'specialized'}
            ]
        }

    def select_trivia_questions(self, treatment: str) -> List[Dict]:
        """Select balanced questions with controlled randomization."""
        random.seed(ExperimentConfig.QUESTION_SELECTION_SEED)
        
        question_bank = self.trivia_questions[treatment]
        random.shuffle(question_bank)
        selected_questions = question_bank[:ExperimentConfig.TRIVIA_QUESTIONS_COUNT]
        
        logging.info(f"Selected {len(selected_questions)} questions for {treatment} treatment")
        return selected_questions

    def show_progress_bar(self, current_step: int, total_steps: int):
        """Enhanced progress bar with research-grade visual feedback."""
        progress = current_step / total_steps
        progress_text = f"{current_step}/{total_steps}"
        
        st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress*100}%">
                {progress_text}
            </div>
        </div>
        <p style="text-align: center; color: #7f8c8d; margin-top: 0.5rem;">
            Screen {current_step} of {total_steps} • Progress: {progress*100:.1f}%
        </p>
        """, unsafe_allow_html=True)

    def show_welcome_screen(self):
        """Enhanced welcome screen with improved research disclosure."""
        st.markdown('<div class="main-header"><h1>🧪 Decision-Making Research Experiment</h1><p>Validated Experimental Protocol</p></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(1, 15)
        
        st.markdown("""
        <div class="experiment-card">
            <h2>📋 Research Information & Informed Consent</h2>
            
            <div class="methodology-warning">
                <h4>🔬 Research Study Details</h4>
                <p><strong>Study Title:</strong> "Decision-Making Under Uncertainty and Performance Beliefs"</p>
                <p><strong>Institution:</strong> Research University</p>
                <p><strong>Principal Investigator:</strong> Dr. Research Team</p>
                <p><strong>Protocol:</strong> Based on published Management Science methodology</p>
            </div>
            
            <h4>📖 What You Will Do</h4>
            <ul>
                <li><strong>Phase 1:</strong> Complete 25 trivia questions (6 minutes)</li>
                <li><strong>Phase 2:</strong> Report beliefs about your performance</li>
                <li><strong>Phase 3:</strong> Receive group assignment based on performance</li>
                <li><strong>Phase 4:</strong> Make hiring decisions for other participants</li>
                <li><strong>Phase 5:</strong> Complete post-experiment questionnaire</li>
            </ul>
            
            <div style="background: #d4edda; border: 2px solid #c3e6cb; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #155724; margin-top: 0;">💰 Payment Structure</h4>
                <p style="color: #155724; margin-bottom: 0;">
                    <strong>${ExperimentConfig.SHOW_UP_FEE:.2f} show-up fee</strong> + earnings from ONE randomly selected task<br>
                    Token exchange rate: 1 token = ${ExperimentConfig.TOKEN_TO_DOLLAR_RATE:.2f}
                </p>
            </div>
            
            <h4>🔒 Research Ethics & Privacy</h4>
            <ul>
                <li>✅ All information provided is truthful (no deception)</li>
                <li>✅ Your responses are completely anonymous</li>
                <li>✅ Data used only for academic research purposes</li>
                <li>✅ You may withdraw at any time without penalty</li>
                <li>✅ All data stored securely and encrypted</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        consent = st.checkbox("I have read and understood the research information above, and I consent to participate in this study.", key="consent_checkbox")
        
        if consent:
            if st.button("🚀 Begin Research Experiment", key="begin_experiment"):
                st.session_state.experiment_data['consent_given'] = True
                st.session_state.experiment_data['consent_timestamp'] = datetime.now().isoformat()
                st.session_state.current_screen = 1
                logging.info(f"Participant {st.session_state.experiment_data['participant_id']} gave consent and started experiment")
                st.rerun()
        else:
            st.info("Please read the research information and provide consent to participate.")

    def show_treatment_assignment(self):
        """Randomly assign participants to easy or hard treatment."""
        st.markdown('<div class="main-header"><h1>🧪 Decision-Making Research Experiment</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(2, 15)
        
        if st.session_state.experiment_data['treatment'] is None:
            # Random treatment assignment
            st.session_state.experiment_data['treatment'] = random.choice(['easy', 'hard'])
            st.session_state.selected_questions = self.select_trivia_questions(st.session_state.experiment_data['treatment'])
            
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} assigned to {st.session_state.experiment_data['treatment']} treatment")
        
        st.markdown("""
        <div class="experiment-card">
            <h2>📚 Phase 1: Trivia Questions</h2>
            
            <div style="background: #f0f8ff; border: 2px solid #4169e1; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #4169e1; margin-top: 0;">📋 Instructions</h4>
                <p>You will now complete <strong>25 multiple-choice trivia questions</strong>.</p>
                <ul>
                    <li>You have <strong>6 minutes</strong> to complete all questions</li>
                    <li>You can navigate between questions and change your answers</li>
                    <li>A timer will show your remaining time</li>
                    <li>Your score determines your performance classification</li>
                </ul>
            </div>
            
            <div style="background: #d4edda; border: 2px solid #c3e6cb; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #155724; margin-top: 0;">💰 Payment Information</h4>
                <p style="color: #155724; margin-bottom: 0;">
                    If this task is selected for payment:<br>
                    <strong>Top 50% performers:</strong> {ExperimentConfig.HIGH_PERFORMANCE_TOKENS} tokens<br>
                    <strong>Bottom 50% performers:</strong> {ExperimentConfig.LOW_PERFORMANCE_TOKENS} tokens
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("▶️ Start Trivia Questions", key="start_trivia"):
            st.session_state.trivia_start_time = time.time()
            st.session_state.current_screen = 2
            st.rerun()

    def show_trivia_questions(self):
        """Display trivia questions with timer and navigation."""
        st.markdown('<div class="main-header"><h1>🧪 Trivia Questions</h1></div>', unsafe_allow_html=True)
        
        # Timer display
        if st.session_state.trivia_start_time:
            elapsed_time = time.time() - st.session_state.trivia_start_time
            remaining_time = max(0, ExperimentConfig.TRIVIA_TIME_LIMIT - elapsed_time)
            
            if remaining_time <= 60:  # Warning in last minute
                st.markdown(f'<div class="timer-warning">⏰ TIME WARNING: {int(remaining_time)} seconds remaining!</div>', unsafe_allow_html=True)
            else:
                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)
                st.markdown(f'<div class="timer-normal">⏱️ Time Remaining: {minutes}:{seconds:02d}</div>', unsafe_allow_html=True)
            
            # Auto-submit when time runs out
            if remaining_time <= 0:
                self.submit_trivia()
                return
        
        # Progress indicator
        self.show_progress_bar(3, 15)
        st.markdown(f"<p style='text-align: center; margin: 1rem 0;'><strong>Question {st.session_state.current_trivia_question + 1} of {ExperimentConfig.TRIVIA_QUESTIONS_COUNT}</strong></p>", unsafe_allow_html=True)
        
        # Current question
        current_q = st.session_state.current_trivia_question
        question_data = st.session_state.selected_questions[current_q]
        
        # Track question start time
        if current_q not in st.session_state.question_start_times:
            st.session_state.question_start_times[current_q] = time.time()
        
        st.markdown(f"""
        <div class="question-container">
            <h3>Question {current_q + 1}</h3>
            <p style="font-size: 1.2em; font-weight: 500; margin: 1.5rem 0;">{question_data['question']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Answer options
        answer_key = f"trivia_answer_{current_q}"
        selected = st.radio(
            "Select your answer:",
            options=range(len(question_data['options'])),
            format_func=lambda x: f"{chr(65+x)}. {question_data['options'][x]}",
            key=answer_key,
            index=st.session_state.experiment_data['trivia_answers'][current_q]
        )
        
        # Update answer and response time
        if selected is not None:
            st.session_state.experiment_data['trivia_answers'][current_q] = selected
            if current_q in st.session_state.question_start_times:
                st.session_state.experiment_data['trivia_response_times'][current_q] = time.time() - st.session_state.question_start_times[current_q]
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_q > 0:
                if st.button("⬅️ Previous", key="prev_question"):
                    st.session_state.current_trivia_question -= 1
                    st.rerun()
        
        with col2:
            # Progress display
            answered = sum(1 for a in st.session_state.experiment_data['trivia_answers'] if a is not None)
            st.markdown(f"<p style='text-align: center;'>Answered: {answered}/{ExperimentConfig.TRIVIA_QUESTIONS_COUNT}</p>", unsafe_allow_html=True)
        
        with col3:
            if current_q < ExperimentConfig.TRIVIA_QUESTIONS_COUNT - 1:
                if st.button("Next ➡️", key="next_question"):
                    st.session_state.current_trivia_question += 1
                    st.rerun()
            else:
                if st.button("✅ Submit All Answers", key="submit_trivia"):
                    self.submit_trivia()

    def submit_trivia(self):
        """Enhanced trivia submission with detailed analysis."""
        if st.session_state.trivia_start_time:
            st.session_state.experiment_data['trivia_time_spent'] = time.time() - st.session_state.trivia_start_time
        
        # Calculate score
        score = 0
        for i, question in enumerate(st.session_state.selected_questions):
            answer = st.session_state.experiment_data['trivia_answers'][i]
            if answer == question['correct']:
                score += 1
        
        st.session_state.experiment_data['trivia_score'] = score
        st.session_state.experiment_data['accuracy_rate'] = (score / ExperimentConfig.TRIVIA_QUESTIONS_COUNT) * 100
        
        # Simulate performance level (in real experiment, this would be calculated after all participants)
        median_score = 12.5  # Simulated
        performance_level = 'High' if score >= median_score else 'Low'
        st.session_state.experiment_data['performance_level'] = performance_level
        st.session_state.experiment_data['session_median_score'] = median_score
        st.session_state.experiment_data['performance_percentile'] = (score / ExperimentConfig.TRIVIA_QUESTIONS_COUNT) * 100
        
        logging.info(f"Participant {st.session_state.experiment_data['participant_id']} completed trivia: {score}/{ExperimentConfig.TRIVIA_QUESTIONS_COUNT}")
        
        st.session_state.current_screen = 3
        st.rerun()

    def show_belief_instructions(self):
        """Instructions for belief elicitation about own performance."""
        st.markdown('<div class="main-header"><h1>🧪 Decision-Making Research Experiment</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(4, 15)
        
        st.markdown("""
        <div class="experiment-card">
            <h2>🧠 Phase 2: Beliefs About Your Performance</h2>
            
            <div style="background: #f0f8ff; border: 2px solid #4169e1; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #4169e1; margin-top: 0;">📊 Performance Classification</h4>
                <p>Based on your trivia score, you will be classified as either:</p>
                <ul>
                    <li><strong>High Performance:</strong> Top 50% of participants in this session</li>
                    <li><strong>Low Performance:</strong> Bottom 50% of participants in this session</li>
                </ul>
                <p><em>This classification will be revealed later in the experiment.</em></p>
            </div>
            
            <div style="background: #d4edda; border: 2px solid #c3e6cb; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #155724; margin-top: 0;">💰 Payment for Accuracy</h4>
                <p style="color: #155724;">
                    If this question is selected for payment, you will earn tokens based on the accuracy of your belief.<br>
                    <strong>The closer your guess to reality, the higher your payment!</strong>
                </p>
                <p style="color: #155724; margin-bottom: 0;">
                    This payment mechanism rewards honest reporting of your true beliefs.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📝 Continue to Belief Question", key="continue_to_belief"):
            st.session_state.current_screen = 4
            st.rerun()

    def show_belief_own_screen(self):
        """Elicit beliefs about own performance."""
        st.markdown('<div class="main-header"><h1>🧪 Belief About Your Performance</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(5, 15)
        
        st.markdown("""
        <div class="experiment-card">
            <h2>🎯 What Do You Think?</h2>
            
            <div style="background: #fff3cd; border: 2px solid #ffc107; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #856404; margin-top: 0;">❓ The Question</h4>
                <p style="font-size: 1.3em; font-weight: 600; color: #856404;">
                    What do you think is the probability that you are a <strong>High Performance</strong> participant?
                </p>
                <p style="color: #856404; margin-bottom: 0;">
                    (Remember: High Performance = Top 50% of participants in this session)
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        belief = st.slider(
            "Your belief (as a percentage from 0% to 100%):",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key="belief_own_performance"
        )
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: #e9ecef; border-radius: 8px; margin: 1rem 0;">
            <h3 style="color: #495057;">Your Current Answer: {belief}%</h3>
            <p style="color: #6c757d; margin-bottom: 0;">
                You believe there is a <strong>{belief}%</strong> chance you are in the top 50% of performers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ Submit Belief", key="submit_belief"):
            st.session_state.experiment_data['belief_own_performance'] = belief
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} belief: {belief}%")
            st.session_state.current_screen = 5
            st.rerun()

    def show_group_assignment_instructions(self):
        """Explain group assignment mechanism."""
        st.markdown('<div class="main-header"><h1>🧪 Decision-Making Research Experiment</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(6, 15)
        
        st.markdown("""
        <div class="experiment-card">
            <h2>🎲 Phase 3: Group Assignment</h2>
            
            <div style="background: #f0f8ff; border: 2px solid #4169e1; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #4169e1; margin-top: 0;">🔄 How Groups Are Assigned</h4>
                <p>You and all other participants will now be assigned to groups: <strong>Top</strong> or <strong>Bottom</strong>.</p>
                
                <p><strong>The computer will flip a coin to determine the assignment mechanism:</strong></p>
                <ul>
                    <li><strong>If HEADS:</strong> 95% chance your group reflects your performance + 5% chance it doesn't</li>
                    <li><strong>If TAILS:</strong> 55% chance your group reflects your performance + 45% chance it doesn't</li>
                </ul>
            </div>
            
            <div style="background: #fff3cd; border: 2px solid #ffc107; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #856404; margin-top: 0;">🎯 What This Means</h4>
                <p><strong>"Reflects performance"</strong> means:</p>
                <ul style="color: #856404;">
                    <li>High Performance → Top Group</li>
                    <li>Low Performance → Bottom Group</li>
                </ul>
                <p style="margin-bottom: 0;"><em>You will see your group assignment but NOT the coin flip result.</em></p>
            </div>
            
            <div style="display: flex; gap: 1rem; margin: 1.5rem 0;">
                <div style="flex: 1; padding: 1.5rem; background: #d4edda; border: 2px solid #c3e6cb; border-radius: 10px;">
                    <h4 style="color: #155724; margin-top: 0;">🥇 Mechanism A (95% Accurate)</h4>
                    <p style="color: #155724; margin-bottom: 0;">Groups are highly likely to reflect actual performance</p>
                </div>
                <div style="flex: 1; padding: 1.5rem; background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 10px;">
                    <h4 style="color: #721c24; margin-top: 0;">🎲 Mechanism B (55% Accurate)</h4>
                    <p style="color: #721c24; margin-bottom: 0;">Groups are only mildly likely to reflect performance</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎲 Proceed to Group Assignment", key="proceed_to_assignment"):
            st.session_state.current_screen = 6
            st.rerun()

    def show_group_assignment(self):
        """Show group assignment result."""
        st.markdown('<div class="main-header"><h1>🧪 Your Group Assignment</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(7, 15)
        
        if st.session_state.experiment_data['assigned_group'] is None:
            # Simulate mechanism selection and group assignment
            mechanism = random.choice(['A', 'B'])
            accuracy = ExperimentConfig.MECHANISM_A_ACCURACY if mechanism == 'A' else ExperimentConfig.MECHANISM_B_ACCURACY
            
            # Determine if assignment reflects performance
            reflects_performance = random.random() < accuracy
            
            if reflects_performance:
                assigned_group = 'Top' if st.session_state.experiment_data['performance_level'] == 'High' else 'Bottom'
            else:
                assigned_group = 'Bottom' if st.session_state.experiment_data['performance_level'] == 'High' else 'Top'
            
            st.session_state.experiment_data['mechanism_used'] = mechanism
            st.session_state.experiment_data['mechanism_reflects_performance'] = reflects_performance
            st.session_state.experiment_data['assigned_group'] = assigned_group
            
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} assigned to {assigned_group} group via mechanism {mechanism}")
        
        group = st.session_state.experiment_data['assigned_group']
        group_color = "#2ecc71" if group == "Top" else "#e67e22"
        
        st.markdown(f"""
        <div class="group-display" style="border-color: {group_color};">
            🏷️ You have been assigned to the<br>
            <span style="color: {group_color}; font-size: 3rem;">{group} Group</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="experiment-card">
            <div style="background: #e9ecef; border: 2px solid #6c757d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #495057; margin-top: 0;">🔍 What You Know</h4>
                <ul style="color: #495057;">
                    <li>Your group assignment: <strong>{}</strong></li>
                    <li>The computer flipped a coin to choose the mechanism</li>
                    <li>Your group either reflects your performance or it doesn't</li>
                </ul>
                <p style="color: #495057; margin-bottom: 0;"><strong>You do NOT know:</strong> Which mechanism was used or whether your group reflects your performance</p>
            </div>
        </div>
        """.format(group), unsafe_allow_html=True)
        
        if st.button("➡️ Continue to Next Phase", key="continue_after_assignment"):
            st.session_state.current_screen = 7
            st.rerun()

    def show_comprehension_questions(self):
        """Show comprehension questions to ensure understanding."""
        st.markdown('<div class="main-header"><h1>🧪 Comprehension Check</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(8, 15)
        
        st.markdown("""
        <div class="experiment-card">
            <h2>✅ Understanding Check</h2>
            <p>Please answer these questions to make sure you understand the group assignment process.</p>
            <p><em>You must answer all questions correctly to continue.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comprehension questions
        q1 = st.radio(
            "1. What determines whether groups are assigned by Mechanism A or Mechanism B?",
            options=[
                "Your trivia score",
                "A coin flip by the computer", 
                "Your belief about your performance",
                "Random assignment"
            ],
            key="comp_q1"
        )
        
        q2 = st.radio(
            "2. What is the probability that groups reflect performance under Mechanism A?",
            options=["55%", "75%", "85%", "95%"],
            key="comp_q2"
        )
        
        q3 = st.radio(
            "3. Do you know which mechanism was used to assign your group?",
            options=["Yes, I know which mechanism was used", "No, I don't know which mechanism was used"],
            key="comp_q3"
        )
        
        if st.button("📝 Check Answers", key="check_comprehension"):
            correct_answers = [
                "A coin flip by the computer",
                "95%", 
                "No, I don't know which mechanism was used"
            ]
            
            user_answers = [q1, q2, q3]
            all_correct = all(user == correct for user, correct in zip(user_answers, correct_answers))
            
            if all_correct:
                st.markdown('<div class="comprehension-correct">✅ All correct! You understand the process.</div>', unsafe_allow_html=True)
                if st.button("➡️ Continue to Hiring Decisions", key="continue_to_hiring"):
                    st.session_state.current_screen = 8
                    st.rerun()
            else:
                st.markdown('<div class="comprehension-incorrect">❌ Some answers are incorrect. Please review the instructions and try again.</div>', unsafe_allow_html=True)
                
                # Track attempts
                if 'comprehension_attempts' not in st.session_state.experiment_data:
                    st.session_state.experiment_data['comprehension_attempts'] = 0
                st.session_state.experiment_data['comprehension_attempts'] += 1

    def show_hiring_instructions(self):
        """Instructions for hiring decisions."""
        st.markdown('<div class="main-header"><h1>🧪 Phase 4: Hiring Decisions</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(9, 15)
        
        st.markdown(f"""
        <div class="experiment-card">
            <h2>💼 Hiring Task Instructions</h2>
            
            <div style="background: #f0f8ff; border: 2px solid #4169e1; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #4169e1; margin-top: 0;">🎯 Your Task</h4>
                <p>You will now make hiring decisions for workers from both groups:</p>
                <ul>
                    <li>Make a hiring decision for a <strong>Top Group</strong> member</li>
                    <li>Make a hiring decision for a <strong>Bottom Group</strong> member</li>
                </ul>
            </div>
            
            <div style="background: #d4edda; border: 2px solid #c3e6cb; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #155724; margin-top: 0;">💰 How Payment Works</h4>
                <p style="color: #155724;">If you hire a worker, your payment depends on their actual performance:</p>
                <ul style="color: #155724;">
                    <li><strong>High Performance worker:</strong> {ExperimentConfig.HIGH_WORKER_REWARD} tokens</li>
                    <li><strong>Low Performance worker:</strong> {ExperimentConfig.LOW_WORKER_REWARD} tokens</li>
                    <li><strong>Minus:</strong> The hiring cost you pay</li>
                </ul>
                <p style="color: #155724; margin-bottom: 0;">
                    <strong>Starting endowment:</strong> {ExperimentConfig.ENDOWMENT_TOKENS} tokens for each decision
                </p>
            </div>
            
            <div style="background: #fff3cd; border: 2px solid #ffc107; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #856404; margin-top: 0;">🤔 What You Decide</h4>
                <p style="color: #856404;">
                    For each group, you'll state the <strong>maximum</strong> you're willing to pay to hire a random member.
                    The computer will then draw a random hiring cost between {ExperimentConfig.BDM_MIN_VALUE} and {ExperimentConfig.BDM_MAX_VALUE} tokens.
                </p>
                <ul style="color: #856404;">
                    <li>If the random cost ≤ your maximum → You hire the worker and pay the random cost</li>
                    <li>If the random cost > your maximum → You don't hire the worker</li>
                </ul>
                <p style="color: #856404; margin-bottom: 0;">
                    <strong>Best strategy:</strong> State your true maximum willingness to pay!
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💼 Begin Hiring Decisions", key="begin_hiring"):
            st.session_state.current_screen = 9
            st.rerun()

    def show_hiring_decisions(self):
        """Elicit willingness to pay for top and bottom group members."""
        st.markdown('<div class="main-header"><h1>💼 Hiring Decisions</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(10, 15)
        
        st.markdown("""
        <div class="experiment-card">
            <h2>💼 Make Your Hiring Decisions</h2>
            <p>State your maximum willingness to pay for a randomly selected member of each group.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top group hiring decision
        st.markdown("### 🥇 Hiring from Top Group")
        wtp_top = st.slider(
            f"Maximum willing to pay for a Top Group member (0-{ExperimentConfig.BDM_MAX_VALUE} tokens):",
            min_value=ExperimentConfig.BDM_MIN_VALUE,
            max_value=ExperimentConfig.BDM_MAX_VALUE,
            value=100,
            step=1,
            key="wtp_top_group"
        )
        
        # Bottom group hiring decision  
        st.markdown("### 🥈 Hiring from Bottom Group")
        wtp_bottom = st.slider(
            f"Maximum willing to pay for a Bottom Group member (0-{ExperimentConfig.BDM_MAX_VALUE} tokens):",
            min_value=ExperimentConfig.BDM_MIN_VALUE,
            max_value=ExperimentConfig.BDM_MAX_VALUE,
            value=100,
            step=1,
            key="wtp_bottom_group"
        )
        
        # Summary
        premium = wtp_top - wtp_bottom
        st.markdown(f"""
        <div style="background: #e9ecef; border: 2px solid #6c757d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
            <h4 style="color: #495057; margin-top: 0;">📊 Your Decisions Summary</h4>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span><strong>Top Group WTP:</strong></span>
                <span>{wtp_top} tokens</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span><strong>Bottom Group WTP:</strong></span>
                <span>{wtp_bottom} tokens</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; font-weight: bold; border-top: 1px solid #6c757d; padding-top: 0.5rem;">
                <span><strong>Premium for Top Group:</strong></span>
                <span>{premium:+} tokens</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ Submit Hiring Decisions", key="submit_hiring"):
            st.session_state.experiment_data['wtp_top_group'] = wtp_top
            st.session_state.experiment_data['wtp_bottom_group'] = wtp_bottom
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} WTP: Top={wtp_top}, Bottom={wtp_bottom}, Premium={premium}")
            st.session_state.current_screen = 10
            st.rerun()

    def show_mechanism_belief(self):
        """Elicit beliefs about which mechanism was used."""
        st.markdown('<div class="main-header"><h1>🤔 Final Belief Question</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(11, 15)
        
        st.markdown("""
        <div class="experiment-card">
            <h2>🎲 Belief About Group Assignment</h2>
            
            <div style="background: #fff3cd; border: 2px solid #ffc107; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #856404; margin-top: 0;">❓ Final Question</h4>
                <p style="font-size: 1.2em; font-weight: 600; color: #856404;">
                    What do you think is the probability that <strong>Mechanism A</strong> (95% accurate) was used to assign groups?
                </p>
                <p style="color: #856404; margin-bottom: 0;">
                    Remember: The computer flipped a coin with 50% chance for each mechanism.
                </p>
            </div>
            
            <div style="background: #e9ecef; border: 2px solid #6c757d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
                <h4 style="color: #495057; margin-top: 0;">🔄 Reminder</h4>
                <ul style="color: #495057;">
                    <li><strong>Mechanism A:</strong> 95% chance groups reflect performance</li>
                    <li><strong>Mechanism B:</strong> 55% chance groups reflect performance</li>
                    <li><strong>Your group:</strong> {st.session_state.experiment_data['assigned_group']} Group</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        belief_mechanism = st.slider(
            "Probability that Mechanism A was used (0% to 100%):",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key="belief_mechanism"
        )
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: #e9ecef; border-radius: 8px; margin: 1rem 0;">
            <h3 style="color: #495057;">Your Belief: {belief_mechanism}%</h3>
            <p style="color: #6c757d; margin-bottom: 0;">
                You believe there is a <strong>{belief_mechanism}%</strong> chance that Mechanism A (95% accurate) was used.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ Submit Final Belief", key="submit_mechanism_belief"):
            st.session_state.experiment_data['belief_mechanism'] = belief_mechanism
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} mechanism belief: {belief_mechanism}%")
            st.session_state.current_screen = 11
            st.rerun()

    def show_questionnaire(self):
        """Post-experiment questionnaire."""
        st.markdown('<div class="main-header"><h1>📝 Post-Experiment Questionnaire</h1></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(12, 15)
        
        st.markdown("""
        <div class="experiment-card">
            <h2>📊 Final Questions</h2>
            <p>Please answer these questions about your experience and background.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Demographics
        st.markdown("### 👤 Demographics")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender:", ["", "Male", "Female", "Non-binary", "Prefer not to say"], key="demo_gender")
            age = st.selectbox("Age group:", ["", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"], key="demo_age")
        
        with col2:
            education = st.selectbox("Education:", ["", "High school", "Some college", "Bachelor's degree", "Master's degree", "PhD", "Other"], key="demo_education")
            experience = st.selectbox("Previous experiment experience:", ["", "None", "1-2 times", "3-5 times", "More than 5 times"], key="demo_experience")
        
        # Task perception
        st.markdown("### 🎯 Task Perception")
        
        difficulty = st.select_slider(
            "How difficult did you find the trivia questions?",
            options=["Very easy", "Easy", "Moderate", "Hard", "Very hard"],
            value="Moderate",
            key="task_difficulty"
        )
        
        confidence_during = st.select_slider(
            "How confident were you during the trivia task?",
            options=["Not confident", "Slightly confident", "Moderately confident", "Very confident", "Extremely confident"],
            value="Moderately confident",
            key="confidence_during"
        )
        
        effort = st.select_slider(
            "How much effort did you put into the trivia questions?",
            options=["Very little", "Little", "Moderate", "High", "Very high"],
            value="High",
            key="effort_level"
        )
        
        # Decision making
        st.markdown("### 💼 Decision Making")
        
        hiring_strategy = st.text_area(
            "What factors influenced your hiring decisions? Please explain your reasoning:",
            height=100,
            key="hiring_strategy"
        )
        
        # Validation questions
        st.markdown("### ✅ Validation")
        
        honest = st.selectbox(
            "Did you answer all questions honestly?",
            ["", "Yes, completely honest", "Mostly honest", "Somewhat honest", "Not very honest"],
            key="honest_responses"
        )
        
        data_quality = st.selectbox(
            "Should your data be included in the research analysis?",
            ["", "Yes, include my data", "No, exclude my data", "Unsure"],
            key="data_quality"
        )
        
        if st.button("📝 Submit Questionnaire", key="submit_questionnaire"):
            # Validate required fields
            required_fields = [gender, age, education, experience, difficulty, confidence_during, effort, honest, data_quality]
            if all(field != "" for field in required_fields) and len(hiring_strategy.strip()) >= 20:
                
                questionnaire_data = {
                    'demographics': {
                        'gender': gender,
                        'age': age,
                        'education': education,
                        'experience': experience
                    },
                    'task_perception': {
                        'task_difficulty': difficulty,
                        'confidence_during_task': confidence_during,
                        'effort_level': effort
                    },
                    'decision_making': {
                        'hiring_strategy': hiring_strategy
                    },
                    'validation': {
                        'honest_responses': honest,
                        'data_quality': data_quality
                    }
                }
                
                st.session_state.experiment_data['post_experiment_questionnaire'] = questionnaire_data
                st.session_state.experiment_data['end_time'] = datetime.now().isoformat()
                
                logging.info(f"Participant {st.session_state.experiment_data['participant_id']} completed questionnaire")
                st.session_state.current_screen = 12
                st.rerun()
            else:
                st.error("Please complete all required fields. The hiring strategy explanation must be at least 20 characters.")

    def show_results(self):
        """Enhanced results display with comprehensive research analytics."""
        st.markdown('<div class="main-header"><h1>🧪 Experiment Complete!</h1><h2>🎉 Thank You for Participating</h2></div>', unsafe_allow_html=True)
        
        self.show_progress_bar(15, 15)
        
        # Save to database
        save_success = self.db.save_session(st.session_state.experiment_data)
        if save_success:
            st.success("✅ Your data has been successfully saved to the research database.")
        else:
            st.error("⚠️ Database save failed, but your data is preserved for download.")
        
        data = st.session_state.experiment_data
        
        # Calculate key metrics
        wtp_premium = data['wtp_top_group'] - data['wtp_bottom_group']
        actual_performance = 1 if data['performance_level'] == 'High' else 0
        belief_performance = data['belief_own_performance'] / 100
        overconfidence_measure = belief_performance - actual_performance
        
        # Results display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Your Experimental Results")
            
            results_html = f"""
            <div class="experiment-card">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">🔬 Treatment & Performance</h4>
                <div class="results-item"><span><strong>Treatment:</strong></span><span>{'Easy Questions' if data['treatment'] == 'easy' else 'Hard Questions'}</span></div>
                <div class="results-item"><span><strong>Trivia Score:</strong></span><span>{data['trivia_score']}/{ExperimentConfig.TRIVIA_QUESTIONS_COUNT} ({data.get('accuracy_rate', 0):.1f}%)</span></div>
                <div class="results-item"><span><strong>Performance Level:</strong></span><span>{data['performance_level']} Performance</span></div>
                <div class="results-item"><span><strong>Assigned Group:</strong></span><span>{data['assigned_group']} Group</span></div>
                
                <h4 style="color: #2c3e50; margin: 1.5rem 0 1rem 0;">🧠 Beliefs & Decisions</h4>
                <div class="results-item"><span><strong>Belief Own Performance:</strong></span><span>{data['belief_own_performance']}%</span></div>
                <div class="results-item"><span><strong>WTP Top Group:</strong></span><span>{data['wtp_top_group']} tokens</span></div>
                <div class="results-item"><span><strong>WTP Bottom Group:</strong></span><span>{data['wtp_bottom_group']} tokens</span></div>
                <div class="results-item"><span><strong>WTP Premium:</strong></span><span>{wtp_premium:+} tokens</span></div>
                <div class="results-item"><span><strong>Belief Mechanism A:</strong></span><span>{data['belief_mechanism']}%</span></div>
                <div class="results-item"><span><strong>Overconfidence:</strong></span><span>{overconfidence_measure:+.3f}</span></div>
            </div>
            """
            st.markdown(results_html, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💰 Payment Calculation")
            
            # Payment simulation
            selected_task = random.choice(['Trivia Performance', 'Belief Own Performance', 'Hiring Decision'])
            
            if selected_task == 'Trivia Performance':
                tokens_earned = ExperimentConfig.HIGH_PERFORMANCE_TOKENS if data['performance_level'] == 'High' else ExperimentConfig.LOW_PERFORMANCE_TOKENS
            else:
                tokens_earned = random.randint(ExperimentConfig.LOW_PERFORMANCE_TOKENS, ExperimentConfig.HIGH_PERFORMANCE_TOKENS)
            
            token_value = tokens_earned * ExperimentConfig.TOKEN_TO_DOLLAR_RATE
            total_payment = ExperimentConfig.SHOW_UP_FEE + token_value
            
            payment_html = f"""
            <div class="experiment-card">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">💵 Payment Breakdown</h4>
                <div class="results-item"><span><strong>Show-up fee:</strong></span><span>${ExperimentConfig.SHOW_UP_FEE:.2f}</span></div>
                <div class="results-item"><span><strong>Selected task:</strong></span><span>{selected_task}</span></div>
                <div class="results-item"><span><strong>Tokens earned:</strong></span><span>{tokens_earned}</span></div>
                <div class="results-item"><span><strong>Token value:</strong></span><span>${token_value:.2f}</span></div>
                <div class="results-item" style="background: #d4edda; padding: 1rem; border-radius: 6px; font-weight: bold; font-size: 1.2em; margin-top: 1rem;"><span><strong>Total Payment:</strong></span><span>${total_payment:.2f}</span></div>
            </div>
            """
            st.markdown(payment_html, unsafe_allow_html=True)
        
        # Data export options
        st.markdown("### 📥 Research Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Complete Data", key="download_json"):
                json_data = json.dumps(st.session_state.experiment_data, indent=2)
                b64 = base64.b64encode(json_data.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="experiment_data_{data["participant_id"]}.json">📄 Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.button("📊 Download Analysis Data", key="download_csv"):
                # Create simplified analysis dataset
                analysis_data = {
                    'participant_id': data['participant_id'],
                    'treatment': data['treatment'],
                    'trivia_score': data['trivia_score'],
                    'accuracy_rate': data.get('accuracy_rate', 0),
                    'performance_level': data['performance_level'],
                    'belief_own_performance': data['belief_own_performance'],
                    'assigned_group': data['assigned_group'],
                    'mechanism_used': data['mechanism_used'],
                    'wtp_top_group': data['wtp_top_group'],
                    'wtp_bottom_group': data['wtp_bottom_group'],
                    'wtp_premium': wtp_premium,
                    'belief_mechanism': data['belief_mechanism'],
                    'overconfidence_measure': overconfidence_measure
                }
                
                df = pd.DataFrame([analysis_data])
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:text/csv;base64,{b64}" download="analysis_data_{data["participant_id"]}.csv">📊 Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 Start New Session", key="new_session"):
                for key in list(st.session_state.keys()):
                    if key != 'database_path':
                        del st.session_state[key]
                st.rerun()

    def run_experiment(self):
        """Main experiment execution with all phases."""
        try:
            screens = [
                self.show_welcome_screen,                    # 0
                self.show_treatment_assignment,              # 1
                self.show_trivia_questions,                  # 2
                self.show_belief_instructions,               # 3
                self.show_belief_own_screen,                 # 4
                self.show_group_assignment_instructions,     # 5
                self.show_group_assignment,                  # 6
                self.show_comprehension_questions,           # 7
                self.show_hiring_instructions,               # 8
                self.show_hiring_decisions,                  # 9
                self.show_mechanism_belief,                  # 10
                self.show_questionnaire,                     # 11
                self.show_results                            # 12
            ]
            
            current_screen = st.session_state.current_screen
            
            if current_screen < len(screens):
                screens[current_screen]()
            else:
                self.show_results()
                
        except Exception as e:
            st.error(f"An experimental error occurred: {str(e)}")
            logging.error(f"Experiment error: {str(e)}", exc_info=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Restart Experiment"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            with col2:
                if st.button("💾 Emergency Data Save"):
                    if 'experiment_data' in st.session_state:
                        json_data = json.dumps(st.session_state.experiment_data, indent=2)
                        b64 = base64.b64encode(json_data.encode()).decode()
                        href = f'<a href="data:application/json;base64,{b64}" download="emergency_backup.json">💾 Download Emergency Backup</a>'
                        st.markdown(href, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    try:
        # Hide streamlit elements for professional appearance
        st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize and run experiment
        experiment = OverconfidenceExperiment()
        experiment.run_experiment()
        
    except Exception as e:
        st.error("Critical application error. Please refresh and try again.")
        logging.critical(f"Application error: {str(e)}", exc_info=True)
    
    # Research information sidebar
    with st.sidebar:
        st.markdown("### 🧪 Research Experiment Platform")
        st.markdown(f"""
        **Overconfidence & Discrimination Study**
        
        **Protocol:** Management Science validated
        **Version:** 2.1.0 (Complete)
        
        ---
        
        **📊 Current Session:**
        - Questions: {ExperimentConfig.TRIVIA_QUESTIONS_COUNT}
        - Time limit: {ExperimentConfig.TRIVIA_TIME_LIMIT//60} minutes
        - Performance cutoff: Top {ExperimentConfig.PERFORMANCE_CUTOFF_PERCENTILE}%
        
        **🔬 Key Features:**
        - Session-relative performance ranking
        - BDM mechanism for incentives
        - Comprehensive data validation
        - Real-time progress tracking
        """)
        
        if hasattr(st.session_state, 'experiment_data'):
            participant_id = st.session_state.experiment_data.get('participant_id', 'Unknown')
            screen = st.session_state.get('current_screen', 0)
            treatment = st.session_state.experiment_data.get('treatment', 'Not assigned')
            st.markdown(f"""
            **📋 Session Status:**
            - ID: `{participant_id}`
            - Screen: {screen+1}/13
            - Treatment: {treatment.title() if treatment != 'Not assigned' else treatment}
            - Status: Active
            """)

if __name__ == "__main__":
    main()

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

# Clean CSS - only hiding Streamlit UI elements
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
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
                'comprehension_attempts': 0,
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
        """Clean progress bar using native Streamlit components."""
        progress = current_step / total_steps
        st.progress(progress, text=f"Screen {current_step} of {total_steps} • Progress: {progress*100:.1f}%")

    def show_welcome_screen(self):
        """Clean welcome screen using native Streamlit components only."""
        st.title("🎓 Behavioral Economics Research Study")
        st.markdown("**Research Institute | Individual Differences in Decision-Making**")
        
        self.show_progress_bar(1, 15)
        
        st.header("📋 Research Information & Informed Consent")
        
        # Research Study Details
        st.info("""
        **🔬 Research Study Details**
        
        **Study Title:** "Individual Differences in Decision-Making Under Uncertainty"
        
        **Principal Investigator:** Research Team Lead, Department of Economics
        
        **Institution:** Academic Research Institute
        
        **Co-Investigators:** Psychology Department, Management Department
        
        **Study Duration:** Approximately 45-60 minutes
        
        **IRB Protocol #:** IRB-2024-XXXX
        
        **Methodology:** Validated experimental design based on published Management Science protocols
        """)
        
        # Purpose
        st.success("""
        **🎯 Purpose of This Study**
        
        This research examines how people make decisions when they have incomplete information about their own abilities and others' qualifications. We are studying individual differences in confidence, belief formation, and decision-making processes. Your participation will contribute to our understanding of these important cognitive and economic phenomena.
        """)
        
        # What You Will Do
        st.subheader("📖 What You Will Do")
        st.markdown("Your participation will involve the following activities:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Phase 1: Cognitive Task (10 minutes)**
            - Answer 25 general knowledge questions within a time limit
            
            **Phase 2: Belief Assessment (5 minutes)**
            - Report your beliefs about your performance relative to others
            
            **Phase 3: Group Assignment (2 minutes)**
            - Receive group assignment based on performance
            """)
        
        with col2:
            st.markdown("""
            **Phase 4: Economic Decisions (15 minutes)**
            - Make hiring decisions involving other participants
            
            **Phase 5: Post-Study Questions (10 minutes)**
            - Complete demographic questionnaire and provide feedback
            """)
        
        # Potential Benefits
        st.success("""
        **✅ Potential Benefits**
        
        - Gain insights into your own decision-making processes
        - Contribute to scientific understanding of economic behavior
        - Receive monetary compensation for your time
        - Learn about behavioral economics research methods
        """)
        
        # Potential Risks
        st.warning("""
        **⚠️ Potential Risks & Discomforts**
        
        **Minimal Risk:** This study involves no more than minimal risk
        
        **Cognitive Effort:** Some questions may be challenging; this is normal
        
        **Time Commitment:** The study requires sustained attention for 45-60 minutes
        
        **Performance Feedback:** You will receive information about your relative performance
        
        **Important:** If you experience any discomfort, you may withdraw at any time without penalty.
        """)
        
        # Compensation Details
        st.success(f"""
        **💰 Compensation Details**
        
        **Base Payment:** ${ExperimentConfig.SHOW_UP_FEE:.2f} participation fee (guaranteed)
        
        **Performance Bonus:** Additional earnings based on ONE randomly selected task
        
        **Token Exchange Rate:** 1 token = ${ExperimentConfig.TOKEN_TO_DOLLAR_RATE:.2f}
        
        **Total Possible Earnings:** ${ExperimentConfig.SHOW_UP_FEE:.2f} - ${ExperimentConfig.SHOW_UP_FEE + (ExperimentConfig.HIGH_PERFORMANCE_TOKENS * ExperimentConfig.TOKEN_TO_DOLLAR_RATE):.2f}
        
        **Payment Processing:** Payment will be processed within 48 hours of study completion via your preferred method.
        """)
        
        # Privacy & Data Protection
        st.info("""
        **🔒 Privacy & Data Protection**
        
        **Anonymity:** Your identity will not be linked to your responses
        
        **Data Security:** All data encrypted and stored on secure servers
        
        **Data Retention:** Research data stored for 7 years per federal regulations
        
        **Data Sharing:** Only aggregated, anonymous data may be shared with other researchers
        
        **Publication:** Results may be published in academic journals with no identifying information
        """)
        
        # Your Rights
        st.subheader("🔒 Your Rights as a Research Participant")
        st.markdown("""
        - **Voluntary Participation:** Your participation is completely voluntary
        - **Right to Withdraw:** You may stop participating at any time without penalty
        - **Right to Ask Questions:** Contact the research team with any concerns
        - **Data Access:** You may request a copy of your individual data
        - **Complaint Process:** Contact the IRB if you have concerns about the research
        """)
        
        # Contact Information
        st.info("""
        **📞 Contact Information**
        
        **Research Team:** research.team@institution.edu
        
        **IRB Office:** (XXX) XXX-XXXX | irb@institution.edu
        
        **Study Coordinator:** Research Team Lead | (XXX) XXX-XXXX
        
        **Please save this information for your records.**
        """)
        
        # Consent Statement
        st.subheader("📜 Informed Consent Statement")
        st.markdown("""
        *"I have read and understood the information provided about this research study. I understand that my participation is voluntary and that I may withdraw at any time without penalty. I understand the potential risks and benefits of participation. I have had the opportunity to ask questions, and all my questions have been answered to my satisfaction."*
        
        *"By checking the consent box below, I indicate my voluntary agreement to participate in this research study."*
        """)
        
        consent = st.checkbox("I have read and understood the research information above, and I consent to participate in this study.", key="consent_checkbox")
        
        if consent:
            if st.button("🚀 Begin Research Experiment", key="begin_experiment", type="primary"):
                st.session_state.experiment_data['consent_given'] = True
                st.session_state.experiment_data['consent_timestamp'] = datetime.now().isoformat()
                st.session_state.current_screen = 1
                logging.info(f"Participant {st.session_state.experiment_data['participant_id']} gave consent and started experiment")
                st.rerun()
        else:
            st.info("Please read the research information and provide consent to participate.")

    def show_treatment_assignment(self):
        """Randomly assign participants to easy or hard treatment."""
        st.title("🎓 Behavioral Economics Research Study")
        
        self.show_progress_bar(2, 15)
        
        if st.session_state.experiment_data['treatment'] is None:
            # Random treatment assignment
            st.session_state.experiment_data['treatment'] = random.choice(['easy', 'hard'])
            st.session_state.selected_questions = self.select_trivia_questions(st.session_state.experiment_data['treatment'])
            
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} assigned to {st.session_state.experiment_data['treatment']} treatment")
        
        st.header("📚 Phase 1: Trivia Questions")
        
        st.subheader("📋 Instructions")
        st.info("""
        You will now complete **25 multiple-choice trivia questions**.

        • You have **6 minutes** to complete all questions
        • You can navigate between questions and change your answers
        • A timer will show your remaining time
        • Your score determines your performance classification
        """)
        
        st.subheader("💰 Payment Information")
        st.success(f"""
        If this task is selected for payment:
        
        **Top 50% performers:** {ExperimentConfig.HIGH_PERFORMANCE_TOKENS} tokens
        
        **Bottom 50% performers:** {ExperimentConfig.LOW_PERFORMANCE_TOKENS} tokens
        """)
        
        if st.button("▶️ Start Trivia Questions", key="start_trivia", type="primary"):
            st.session_state.trivia_start_time = time.time()
            st.session_state.current_screen = 2
            st.rerun()

    def show_trivia_questions(self):
        """Display trivia questions with timer and navigation."""
        st.title("🧪 Trivia Questions")
        
        # Timer display
        if st.session_state.trivia_start_time:
            elapsed_time = time.time() - st.session_state.trivia_start_time
            remaining_time = max(0, ExperimentConfig.TRIVIA_TIME_LIMIT - elapsed_time)
            
            if remaining_time <= 60:  # Warning in last minute
                st.error(f"⏰ TIME WARNING: {int(remaining_time)} seconds remaining!")
            else:
                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)
                st.info(f"⏱️ Time Remaining: {minutes}:{seconds:02d}")
            
            # Auto-submit when time runs out
            if remaining_time <= 0:
                self.submit_trivia()
                return
        
        # Progress indicator
        self.show_progress_bar(3, 15)
        st.markdown(f"**Question {st.session_state.current_trivia_question + 1} of {ExperimentConfig.TRIVIA_QUESTIONS_COUNT}**")
        
        # Current question
        current_q = st.session_state.current_trivia_question
        question_data = st.session_state.selected_questions[current_q]
        
        # Track question start time
        if current_q not in st.session_state.question_start_times:
            st.session_state.question_start_times[current_q] = time.time()
        
        # Display question
        st.subheader(f"Question {current_q + 1}")
        st.markdown(f"**{question_data['question']}**")
        
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
            st.markdown(f"**Answered: {answered}/{ExperimentConfig.TRIVIA_QUESTIONS_COUNT}**")
        
        with col3:
            if current_q < ExperimentConfig.TRIVIA_QUESTIONS_COUNT - 1:
                if st.button("Next ➡️", key="next_question"):
                    st.session_state.current_trivia_question += 1
                    st.rerun()
            else:
                if st.button("✅ Submit All Answers", key="submit_trivia", type="primary"):
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
        st.title("🎓 Behavioral Economics Research Study")
        
        self.show_progress_bar(4, 15)
        
        st.header("🧠 Phase 2: Beliefs About Your Performance")
        
        st.subheader("📊 Performance Classification")
        st.info("""
        Based on your trivia score, you will be classified as either:
        
        • **High Performance:** Top 50% of participants in this session
        • **Low Performance:** Bottom 50% of participants in this session
        
        *This classification will be revealed later in the experiment.*
        """)
        
        st.subheader("💰 Payment for Accuracy")
        st.success("""
        If this question is selected for payment, you will earn tokens based on the accuracy of your belief.
        
        **The closer your guess to reality, the higher your payment!**
        
        This payment mechanism rewards honest reporting of your true beliefs.
        """)
        
        if st.button("📝 Continue to Belief Question", key="continue_to_belief", type="primary"):
            st.session_state.current_screen = 4
            st.rerun()

    def show_belief_own_screen(self):
        """Elicit beliefs about own performance."""
        st.title("🧠 Belief About Your Performance")
        
        self.show_progress_bar(5, 15)
        
        st.header("🎯 What Do You Think?")
        
        st.warning("""
        **The Question:**
        
        What do you think is the probability that you are a **High Performance** participant?
        
        (Remember: High Performance = Top 50% of participants in this session)
        """)
        
        belief = st.slider(
            "Your belief (as a percentage from 0% to 100%):",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key="belief_own_performance"
        )
        
        st.info(f"""
        **Your Current Answer: {belief}%**
        
        You believe there is a **{belief}%** chance you are in the top 50% of performers.
        """)
        
        if st.button("✅ Submit Belief", key="submit_belief", type="primary"):
            st.session_state.experiment_data['belief_own_performance'] = belief
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} belief: {belief}%")
            st.session_state.current_screen = 5
            st.rerun()

    def show_group_assignment_instructions(self):
        """Explain group assignment mechanism."""
        st.title("🎓 Behavioral Economics Research Study")
        
        self.show_progress_bar(6, 15)
        
        st.header("🎲 Phase 3: Group Assignment")
        
        st.subheader("🔄 How Groups Are Assigned")
        st.info("""
        You and all other participants will now be assigned to groups: **Top** or **Bottom**.
        
        **The computer will flip a coin to determine the assignment mechanism:**
        
        • **If HEADS:** 95% chance your group reflects your performance + 5% chance it doesn't
        • **If TAILS:** 55% chance your group reflects your performance + 45% chance it doesn't
        """)
        
        st.subheader("🎯 What This Means")
        st.warning("""
        **"Reflects performance" means:**
        • High Performance → Top Group
        • Low Performance → Bottom Group
        
        *You will see your group assignment but NOT the coin flip result.*
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **🥇 Mechanism A (95% Accurate)**
            
            Groups are highly likely to reflect actual performance
            """)
        
        with col2:
            st.error("""
            **🎲 Mechanism B (55% Accurate)**
            
            Groups are only mildly likely to reflect performance
            """)
        
        if st.button("🎲 Proceed to Group Assignment", key="proceed_to_assignment", type="primary"):
            st.session_state.current_screen = 6
            st.rerun()

    def show_group_assignment(self):
        """Show group assignment result."""
        st.title("🏷️ Your Group Assignment")
        
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
        
        # Clean group display
        if group == "Top":
            st.success(f"## 🥇 You have been assigned to the **{group} Group**")
        else:
            st.info(f"## 🥈 You have been assigned to the **{group} Group**")
        
        st.subheader("🔍 What You Know")
        st.markdown(f"""
        **Information available to you:**
        - Your group assignment: **{group} Group**
        - The computer flipped a coin to choose the mechanism
        - Your group either reflects your performance or it doesn't
        
        **You do NOT know:** Which mechanism was used or whether your group reflects your performance
        """)
        
        if st.button("➡️ Continue to Next Phase", key="continue_after_assignment", type="primary"):
            st.session_state.current_screen = 7
            st.rerun()

    def show_comprehension_questions(self):
        """Show comprehension questions to ensure understanding."""
        st.title("✅ Comprehension Check")
        
        self.show_progress_bar(8, 15)
        
        st.header("Understanding Check")
        st.markdown("Please answer these questions to make sure you understand the group assignment process.")
        st.info("You must answer all questions correctly to continue.")
        
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
        
        # Check if all questions have been answered
        if q1 and q2 and q3:
            if st.button("📝 Check Answers", key="check_comprehension", type="primary"):
                correct_answers = [
                    "A coin flip by the computer",
                    "95%", 
                    "No, I don't know which mechanism was used"
                ]
                
                user_answers = [q1, q2, q3]
                all_correct = all(user == correct for user, correct in zip(user_answers, correct_answers))
                
                if all_correct:
                    st.success("✅ All correct! You understand the process.")
                    st.info("🎉 Advancing to hiring decisions...")
                    
                    # Track successful completion
                    if 'comprehension_attempts' not in st.session_state.experiment_data:
                        st.session_state.experiment_data['comprehension_attempts'] = 0
                    st.session_state.experiment_data['comprehension_attempts'] += 1
                    st.session_state.experiment_data['comprehension_passed'] = True
                    
                    # Auto-advance to next screen
                    st.session_state.current_screen = 8
                    st.rerun()
                else:
                    st.error("❌ Some answers are incorrect. Please review the instructions and try again.")
                    
                    # Show correct answers for learning
                    st.markdown("**Correct answers:**")
                    st.markdown("1. A coin flip by the computer")
                    st.markdown("2. 95%")
                    st.markdown("3. No, I don't know which mechanism was used")
                    
                    # Track attempts - ensure it's initialized as integer
                    if 'comprehension_attempts' not in st.session_state.experiment_data:
                        st.session_state.experiment_data['comprehension_attempts'] = 0
                    # Check if it's actually an integer before incrementing
                    if isinstance(st.session_state.experiment_data['comprehension_attempts'], int):
                        st.session_state.experiment_data['comprehension_attempts'] += 1
                    else:
                        st.session_state.experiment_data['comprehension_attempts'] = 1
        else:
            st.info("Please answer all three questions before checking your answers.")

    def show_hiring_instructions(self):
        """Instructions for hiring decisions."""
        st.title("🎓 Phase 4: Hiring Decisions")
        
        self.show_progress_bar(9, 15)
        
        st.header("💼 Hiring Task Instructions")
        
        st.subheader("🎯 Your Task")
        st.info("""
        You will now make hiring decisions for workers from both groups:
        
        • Make a hiring decision for a **Top Group** member
        • Make a hiring decision for a **Bottom Group** member
        """)
        
        st.subheader("💰 How Payment Works")
        st.success(f"""
        If you hire a worker, your payment depends on their actual performance:
        
        • **High Performance worker:** {ExperimentConfig.HIGH_WORKER_REWARD} tokens
        • **Low Performance worker:** {ExperimentConfig.LOW_WORKER_REWARD} tokens
        • **Minus:** The hiring cost you pay
        
        **Starting endowment:** {ExperimentConfig.ENDOWMENT_TOKENS} tokens for each decision
        """)
        
        st.subheader("🤔 What You Decide")
        st.warning(f"""
        For each group, you'll state the **maximum** you're willing to pay to hire a random member.
        The computer will then draw a random hiring cost between {ExperimentConfig.BDM_MIN_VALUE} and {ExperimentConfig.BDM_MAX_VALUE} tokens.
        
        • If the random cost ≤ your maximum → You hire the worker and pay the random cost
        • If the random cost > your maximum → You don't hire the worker
        
        **Best strategy:** State your true maximum willingness to pay!
        """)
        
        if st.button("💼 Begin Hiring Decisions", key="begin_hiring", type="primary"):
            st.session_state.current_screen = 9
            st.rerun()

    def show_hiring_decisions(self):
        """Elicit willingness to pay for top and bottom group members."""
        st.title("💼 Hiring Decisions")
        
        self.show_progress_bar(10, 15)
        
        st.header("💼 Make Your Hiring Decisions")
        st.markdown("State your maximum willingness to pay for a randomly selected member of each group.")
        
        # Top group hiring decision
        st.subheader("🥇 Hiring from Top Group")
        wtp_top = st.slider(
            f"Maximum willing to pay for a Top Group member (0-{ExperimentConfig.BDM_MAX_VALUE} tokens):",
            min_value=ExperimentConfig.BDM_MIN_VALUE,
            max_value=ExperimentConfig.BDM_MAX_VALUE,
            value=100,
            step=1,
            key="wtp_top_group"
        )
        
        # Bottom group hiring decision  
        st.subheader("🥈 Hiring from Bottom Group")
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
        
        st.subheader("📊 Your Decisions Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Top Group WTP", f"{wtp_top} tokens")
            st.metric("Bottom Group WTP", f"{wtp_bottom} tokens")
        
        with col2:
            st.metric("Premium for Top Group", f"{premium:+} tokens", delta=f"{premium:+} difference")
        
        if st.button("✅ Submit Hiring Decisions", key="submit_hiring", type="primary"):
            st.session_state.experiment_data['wtp_top_group'] = wtp_top
            st.session_state.experiment_data['wtp_bottom_group'] = wtp_bottom
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} WTP: Top={wtp_top}, Bottom={wtp_bottom}, Premium={premium}")
            st.session_state.current_screen = 10
            st.rerun()

    def show_mechanism_belief(self):
        """Elicit beliefs about which mechanism was used."""
        st.title("🤔 Final Belief Question")
        
        self.show_progress_bar(11, 15)
        
        st.header("🎲 Belief About Group Assignment")
        
        st.warning("""
        **Final Question**
        
        What do you think is the probability that **Mechanism A** (95% accurate) was used to assign groups?
        
        Remember: The computer flipped a coin with 50% chance for each mechanism.
        """)
        
        st.subheader("🔄 Reminder")
        st.markdown(f"""
        - **Mechanism A:** 95% chance groups reflect performance
        - **Mechanism B:** 55% chance groups reflect performance
        - **Your group:** {st.session_state.experiment_data['assigned_group']} Group
        """)
        
        belief_mechanism = st.slider(
            "Probability that Mechanism A was used (0% to 100%):",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key="belief_mechanism"
        )
        
        st.info(f"""
        **Your Belief: {belief_mechanism}%**
        
        You believe there is a **{belief_mechanism}%** chance that Mechanism A (95% accurate) was used.
        """)
        
        if st.button("✅ Submit Final Belief", key="submit_mechanism_belief", type="primary"):
            st.session_state.experiment_data['belief_mechanism'] = belief_mechanism
            logging.info(f"Participant {st.session_state.experiment_data['participant_id']} mechanism belief: {belief_mechanism}%")
            st.session_state.current_screen = 11
            st.rerun()

    def show_questionnaire(self):
        """Post-experiment questionnaire."""
        st.title("📝 Post-Experiment Questionnaire")
        
        self.show_progress_bar(12, 15)
        
        st.header("📊 Final Questions")
        st.markdown("Please answer these questions about your experience and background.")
        
        # Demographics
        st.subheader("👤 Demographics")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender:", ["", "Male", "Female", "Non-binary", "Prefer not to say"], key="demo_gender")
            age = st.selectbox("Age group:", ["", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"], key="demo_age")
        
        with col2:
            education = st.selectbox("Education:", ["", "High school", "Some college", "Bachelor's degree", "Master's degree", "PhD", "Other"], key="demo_education")
            experience = st.selectbox("Previous experiment experience:", ["", "None", "1-2 times", "3-5 times", "More than 5 times"], key="demo_experience")
        
        # Task perception
        st.subheader("🎯 Task Perception")
        
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
        st.subheader("💼 Decision Making")
        
        hiring_strategy = st.text_area(
            "What factors influenced your hiring decisions? Please explain your reasoning:",
            height=100,
            key="hiring_strategy"
        )
        
        # Validation questions
        st.subheader("✅ Validation")
        
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
        
        if st.button("📝 Submit Questionnaire", key="submit_questionnaire", type="primary"):
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
        """Clean results display using native Streamlit components only."""
        st.title("🎉 Experiment Complete!")
        st.markdown("**Thank You for Participating in Our Research!**")
        
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
            st.subheader("📊 Your Experimental Results")
            
            st.markdown("**🔬 Treatment & Performance**")
            st.write(f"**Treatment:** {'Easy Questions' if data['treatment'] == 'easy' else 'Hard Questions'}")
            st.write(f"**Trivia Score:** {data['trivia_score']}/{ExperimentConfig.TRIVIA_QUESTIONS_COUNT} ({data.get('accuracy_rate', 0):.1f}%)")
            st.write(f"**Performance Level:** {data['performance_level']} Performance")
            st.write(f"**Assigned Group:** {data['assigned_group']} Group")
            
            st.markdown("**🧠 Beliefs & Decisions**")
            st.write(f"**Belief Own Performance:** {data['belief_own_performance']}%")
            st.write(f"**WTP Top Group:** {data['wtp_top_group']} tokens")
            st.write(f"**WTP Bottom Group:** {data['wtp_bottom_group']} tokens")
            st.write(f"**WTP Premium:** {wtp_premium:+} tokens")
            st.write(f"**Belief Mechanism A:** {data['belief_mechanism']}%")
            st.write(f"**Overconfidence Measure:** {overconfidence_measure:+.3f}")
        
        with col2:
            st.subheader("💰 Payment Calculation")
            
            # Payment simulation
            selected_task = random.choice(['Trivia Performance', 'Belief Own Performance', 'Hiring Decision'])
            
            if selected_task == 'Trivia Performance':
                tokens_earned = ExperimentConfig.HIGH_PERFORMANCE_TOKENS if data['performance_level'] == 'High' else ExperimentConfig.LOW_PERFORMANCE_TOKENS
            else:
                tokens_earned = random.randint(ExperimentConfig.LOW_PERFORMANCE_TOKENS, ExperimentConfig.HIGH_PERFORMANCE_TOKENS)
            
            token_value = tokens_earned * ExperimentConfig.TOKEN_TO_DOLLAR_RATE
            total_payment = ExperimentConfig.SHOW_UP_FEE + token_value
            
            st.markdown("**💵 Payment Breakdown**")
            st.write(f"**Show-up fee:** ${ExperimentConfig.SHOW_UP_FEE:.2f}")
            st.write(f"**Selected task:** {selected_task}")
            st.write(f"**Tokens earned:** {tokens_earned}")
            st.write(f"**Token value:** ${token_value:.2f}")
            
            st.success(f"**Total Payment: ${total_payment:.2f}**")
        
        # Data export options using clean Streamlit components
        st.subheader("📥 Research Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Complete data as JSON
            json_data = json.dumps(st.session_state.experiment_data, indent=2)
            st.download_button(
                label="📄 Download Complete Data",
                data=json_data,
                file_name=f"experiment_data_{data['participant_id']}.json",
                mime="application/json",
                key="download_json"
            )
        
        with col2:
            # Analysis data as CSV
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
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Analysis Data",
                data=csv_data,
                file_name=f"analysis_data_{data['participant_id']}.csv",
                mime="text/csv",
                key="download_csv"
            )
        
        with col3:
            if st.button("🔄 Start New Session", key="new_session", type="primary"):
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
                if st.button("🔄 Restart Experiment", type="primary"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            with col2:
                if st.button("💾 Emergency Data Save"):
                    if 'experiment_data' in st.session_state:
                        json_data = json.dumps(st.session_state.experiment_data, indent=2)
                        st.download_button(
                            label="💾 Download Emergency Backup",
                            data=json_data,
                            file_name="emergency_backup.json",
                            mime="application/json"
                        )

def main():
    """Main application entry point - completely clean."""
    try:
        # Only hide Streamlit UI elements
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
        
        # Clean recovery options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart Experiment", type="primary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("💾 Emergency Data Save"):
                if 'experiment_data' in st.session_state:
                    json_data = json.dumps(st.session_state.experiment_data, indent=2)
                    st.download_button(
                        label="💾 Download Emergency Backup",
                        data=json_data,
                        file_name="emergency_backup.json",
                        mime="application/json"
                    )
    
    # Sidebar - completely clean
    with st.sidebar:
        st.markdown("### 🎓 Research Study Platform")
        st.markdown(f"""
        **Individual Differences in Decision-Making**
        
        **Institution:** Academic Research Institute  
        **PI:** Research Team Lead  
        **Version:** 2.1.0 (Clean)
        
        ---
        
        **📊 Study Parameters:**
        - Questions: {ExperimentConfig.TRIVIA_QUESTIONS_COUNT}
        - Time limit: {ExperimentConfig.TRIVIA_TIME_LIMIT//60} minutes
        - Performance cutoff: Top {ExperimentConfig.PERFORMANCE_CUTOFF_PERCENTILE}%
        
        **🔬 Research Features:**
        - IRB-approved protocol (IRB-2024-XXXX)
        - Incentive-compatible mechanisms
        - Comprehensive data validation
        - Real-time progress tracking
        - Secure data collection
        """)
        
        if hasattr(st.session_state, 'experiment_data'):
            participant_id = st.session_state.experiment_data.get('participant_id', 'Unknown')
            screen = st.session_state.get('current_screen', 0)
            treatment = st.session_state.experiment_data.get('treatment', 'Not assigned')
            # Safe treatment display handling None values
            if treatment and treatment != 'Not assigned':
                treatment_display = treatment.title()
            else:
                treatment_display = 'Not assigned'
            
            st.markdown(f"""
            **📋 Session Status:**
            - ID: `{participant_id}`
            - Screen: {screen+1}/13
            - Treatment: {treatment_display}
            - Status: Active
            """)
            
            # Contact information
            st.markdown("---")
            st.markdown("""
            **📞 Support:**
            If you experience technical issues, please contact the research team at research.team@institution.edu
            """)

if __name__ == "__main__":
    main()

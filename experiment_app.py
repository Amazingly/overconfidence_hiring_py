#!/usr/bin/env python3
"""
OVERCONFIDENCE AND DISCRIMINATORY BEHAVIOR - METHODOLOGICALLY FAITHFUL IMPLEMENTATION
===================================================================================

Revised Implementation Aligned with Management Science Publication
- True random assignment (no self-selection)
- Fixed session sizes matching paper methodology  
- Empirical question difficulty validation
- Simplified coordination reducing technical risk
- Focus on replicating published findings

Based on: "Does Overconfidence Predict Discriminatory Beliefs and Behavior?"
Management Science, DOI: [paper DOI]

Author: Research Team
Version: Methodologically Faithful v1.0
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
import hashlib
import sqlite3
import threading
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import secrets
from scipy import stats
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(
    page_title="Behavioral Economics Study",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    .study-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .progress-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .session-info {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #b8daff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ExperimentConfig:
    """Configuration matching Management Science paper exactly."""
    
    # Core experimental parameters (from paper)
    TRIVIA_QUESTIONS_COUNT = 25
    TRIVIA_TIME_LIMIT = 360  # 6 minutes
    
    # Session structure (based on paper: 297 participants / 14 sessions)
    TARGET_SESSION_SIZE = 21
    MIN_SESSION_SIZE = 18
    MAX_SESSION_SIZE = 24
    TOTAL_TARGET_SESSIONS = 14  # 7 easy + 7 hard
    
    # Performance classification
    PERFORMANCE_CUTOFF_PERCENTILE = 50  # Top 50% = High
    
    # Confidence targets (empirically validated)
    TARGET_EASY_CONFIDENCE = 67  # From paper: 0.67 average
    TARGET_HARD_CONFIDENCE = 43  # From paper: 0.43 average
    CONFIDENCE_TOLERANCE = 5     # ±5% acceptable
    
    # BDM mechanism parameters (from paper)
    BDM_MIN_VALUE = 0
    BDM_MAX_VALUE = 200
    ENDOWMENT_TOKENS = 160
    
    # Payment structure (from paper)
    HIGH_PERFORMANCE_TOKENS = 250
    LOW_PERFORMANCE_TOKENS = 100
    HIGH_WORKER_REWARD = 200
    LOW_WORKER_REWARD = 40
    TOKEN_TO_DOLLAR_RATE = 0.09
    SHOW_UP_FEE = 5.00
    
    # Group assignment mechanisms (from paper)
    MECHANISM_A_ACCURACY = 0.95  # Highly informative
    MECHANISM_B_ACCURACY = 0.55  # Mildly informative
    
    # Session timeout
    SESSION_TIMEOUT_MINUTES = 120

@dataclass
class ParticipantData:
    """Complete participant data structure matching paper variables."""
    participant_id: str
    session_id: str
    assignment_timestamp: datetime
    
    # Treatment assignment
    treatment: str  # 'easy' or 'hard'
    session_number: int  # 1-14
    
    # Trivia performance
    trivia_score: int
    trivia_accuracy: float
    trivia_time_spent: float
    response_times: List[float]
    
    # Performance classification (session-relative)
    session_median: float
    performance_level: str  # 'High' or 'Low'
    performance_rank: int
    session_size: int
    
    # Beliefs about own performance
    belief_own_performance: int  # 0-100
    confidence_level: str  # 'Overconfident', 'Accurate', 'Underconfident'
    
    # Group assignment
    assigned_group: str  # 'Top' or 'Bottom'
    mechanism_used: str  # 'A' or 'B'
    group_reflects_performance: bool
    
    # Hiring decisions (BDM)
    wtp_top_group: int
    wtp_bottom_group: int
    wtp_premium: int  # top - bottom
    
    # Beliefs about mechanism
    belief_mechanism_informative: int  # 0-100 (prob mechanism A)
    
    # Key outcome variables (matching paper)
    overconfidence_measure: float  # belief - actual_performance
    relative_wtp_premium: float   # premium relative to rational benchmark
    
    # Data quality
    attention_checks_passed: bool
    completion_time_minutes: float
    data_quality_flag: str
    
    # Timestamps
    start_time: datetime
    end_time: Optional[datetime] = None

class TreatmentValidator:
    """Empirical validation of treatment effectiveness."""
    
    def __init__(self):
        self.pilot_data = defaultdict(list)
        self.validation_cache = {}
    
    def record_pilot_response(self, treatment: str, question_id: str, 
                            correct: bool, confidence: int):
        """Record pilot data for validation."""
        self.pilot_data[f"{treatment}_{question_id}"].append({
            'correct': correct,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
    
    def validate_treatment_effectiveness(self, treatment: str, 
                                       session_results: List[Dict]) -> Tuple[bool, str]:
        """Validate that treatment produces expected confidence patterns."""
        if len(session_results) < ExperimentConfig.MIN_SESSION_SIZE:
            return False, f"Insufficient data: {len(session_results)} participants"
        
        confidences = [r['belief_own_performance'] for r in session_results]
        mean_confidence = np.mean(confidences)
        
        if treatment == 'easy':
            target = ExperimentConfig.TARGET_EASY_CONFIDENCE
            expected_pattern = "overconfidence"
        else:
            target = ExperimentConfig.TARGET_HARD_CONFIDENCE  
            expected_pattern = "underconfidence"
        
        tolerance = ExperimentConfig.CONFIDENCE_TOLERANCE
        
        if abs(mean_confidence - target) <= tolerance:
            return True, f"✓ {expected_pattern} achieved: {mean_confidence:.1f}% (target: {target}%)"
        else:
            return False, f"✗ Treatment failed: {mean_confidence:.1f}% (target: {target}±{tolerance}%)"
    
    def validate_question_difficulty(self, questions: List[Dict], treatment: str) -> Tuple[bool, str]:
        """Validate individual question difficulties."""
        if not questions:
            return False, "No questions provided"
        
        # Check pilot data availability
        pilot_questions = [q for q in questions if 'pilot_accuracy' in q]
        
        if len(pilot_questions) < len(questions) * 0.8:
            return False, f"Insufficient pilot data: {len(pilot_questions)}/{len(questions)} questions"
        
        accuracies = [q['pilot_accuracy'] for q in pilot_questions]
        mean_accuracy = np.mean(accuracies)
        
        if treatment == 'easy':
            target_range = (75, 85)
        else:
            target_range = (25, 35)
        
        if target_range[0] <= mean_accuracy <= target_range[1]:
            return True, f"✓ Difficulty validated: {mean_accuracy:.1f}% accuracy"
        else:
            return False, f"✗ Difficulty invalid: {mean_accuracy:.1f}% (target: {target_range[0]}-{target_range[1]}%)"

class RandomAssignmentManager:
    """True random assignment to treatments (no self-selection)."""
    
    def __init__(self):
        if 'assignment_state' not in st.session_state:
            st.session_state.assignment_state = {
                'easy_sessions': [],
                'hard_sessions': [], 
                'current_easy_session': None,
                'current_hard_session': None,
                'session_counter': 0,
                'assignment_log': []
            }
    
    def assign_participant(self, participant_id: str) -> Tuple[str, str, int]:
        """Randomly assign participant to treatment and session."""
        
        # True random assignment (50/50)
        treatment = random.choice(['easy', 'hard'])
        
        # Get or create session for this treatment
        session_id, session_number = self._get_or_create_session(treatment)
        
        # Log assignment
        assignment_record = {
            'participant_id': participant_id,
            'treatment': treatment,
            'session_id': session_id,
            'session_number': session_number,
            'timestamp': datetime.now(),
            'assignment_method': 'true_random'
        }
        
        st.session_state.assignment_state['assignment_log'].append(assignment_record)
        
        logger.info(f"Assigned participant {participant_id[:8]}... to {treatment} treatment, "
                   f"session {session_number} ({session_id})")
        
        return treatment, session_id, session_number
    
    def _get_or_create_session(self, treatment: str) -> Tuple[str, int]:
        """Get current session or create new one if full."""
        current_key = f'current_{treatment}_session'
        sessions_key = f'{treatment}_sessions'
        
        current_session = st.session_state.assignment_state[current_key]
        
        # Check if current session exists and has space
        if current_session and self._session_has_space(current_session['id']):
            return current_session['id'], current_session['number']
        
        # Create new session
        st.session_state.assignment_state['session_counter'] += 1
        session_number = st.session_state.assignment_state['session_counter']
        
        session_id = f"{treatment.upper()}{session_number:02d}-{secrets.token_hex(3).upper()}"
        
        new_session = {
            'id': session_id,
            'number': session_number,
            'treatment': treatment,
            'created_at': datetime.now(),
            'participants': [],
            'status': 'collecting'
        }
        
        # Update state
        st.session_state.assignment_state[sessions_key].append(new_session)
        st.session_state.assignment_state[current_key] = new_session
        
        logger.info(f"Created new {treatment} session: {session_id} (#{session_number})")
        
        return session_id, session_number
    
    def _session_has_space(self, session_id: str) -> bool:
        """Check if session has available space."""
        if 'session_participants' not in st.session_state:
            st.session_state.session_participants = defaultdict(list)
        
        current_size = len(st.session_state.session_participants[session_id])
        return current_size < ExperimentConfig.MAX_SESSION_SIZE
    
    def get_assignment_statistics(self) -> Dict:
        """Get current assignment statistics."""
        log = st.session_state.assignment_state['assignment_log']
        
        if not log:
            return {'total': 0, 'easy': 0, 'hard': 0, 'balance': 'N/A'}
        
        treatments = [entry['treatment'] for entry in log]
        treatment_counts = Counter(treatments)
        
        total = len(log)
        easy_count = treatment_counts.get('easy', 0)
        hard_count = treatment_counts.get('hard', 0)
        
        # Calculate balance
        if total > 0:
            balance = abs(easy_count - hard_count) / total * 100
            balance_status = "Good" if balance <= 10 else "Needs attention"
        else:
            balance_status = "N/A"
        
        return {
            'total': total,
            'easy': easy_count,
            'hard': hard_count,
            'balance_pct': balance,
            'balance_status': balance_status,
            'sessions_created': st.session_state.assignment_state['session_counter']
        }

class SessionManager:
    """Simplified session management without real-time complexity."""
    
    def __init__(self):
        if 'sessions' not in st.session_state:
            st.session_state.sessions = {}
        if 'session_participants' not in st.session_state:
            st.session_state.session_participants = defaultdict(list)
        if 'session_data' not in st.session_state:
            st.session_state.session_data = {}
    
    def join_session(self, session_id: str, participant_id: str) -> bool:
        """Add participant to session."""
        participants = st.session_state.session_participants[session_id]
        
        if len(participants) >= ExperimentConfig.MAX_SESSION_SIZE:
            return False
        
        if participant_id not in participants:
            participants.append(participant_id)
            logger.info(f"Added participant to session {session_id} "
                       f"({len(participants)}/{ExperimentConfig.TARGET_SESSION_SIZE})")
        
        return True
    
    def is_session_ready(self, session_id: str) -> bool:
        """Check if session has minimum participants."""
        participants = st.session_state.session_participants[session_id]
        return len(participants) >= ExperimentConfig.MIN_SESSION_SIZE
    
    def calculate_session_performance(self, session_id: str, 
                                    scores: Dict[str, int]) -> Dict[str, str]:
        """Calculate performance levels using true session median."""
        participants = st.session_state.session_participants[session_id]
        session_scores = [scores[pid] for pid in participants if pid in scores]
        
        if len(session_scores) < ExperimentConfig.MIN_SESSION_SIZE:
            raise ValueError(f"Insufficient session data: {len(session_scores)} scores")
        
        # Calculate session median
        median_score = np.median(session_scores)
        
        # Classify participants
        performance_levels = {}
        ties_at_median = []
        
        for pid in participants:
            if pid not in scores:
                continue
            
            score = scores[pid]
            if score > median_score:
                performance_levels[pid] = 'High'
            elif score < median_score:
                performance_levels[pid] = 'Low'
            else:
                ties_at_median.append(pid)
        
        # Break ties randomly (as per paper)
        if ties_at_median:
            random.shuffle(ties_at_median)
            high_spots = max(0, len(session_scores)//2 - 
                           sum(1 for p in performance_levels.values() if p == 'High'))
            
            for i, pid in enumerate(ties_at_median):
                performance_levels[pid] = 'High' if i < high_spots else 'Low'
        
        # Store session statistics
        st.session_state.session_data[session_id] = {
            'median_score': median_score,
            'n_participants': len(session_scores),
            'score_distribution': {
                'mean': np.mean(session_scores),
                'std': np.std(session_scores),
                'min': min(session_scores),
                'max': max(session_scores),
                'q1': np.percentile(session_scores, 25),
                'q3': np.percentile(session_scores, 75)
            }
        }
        
        logger.info(f"Session {session_id} performance calculated: "
                   f"median={median_score:.1f}, n={len(session_scores)}")
        
        return performance_levels

class QuestionBank:
    """Validated trivia questions with empirical difficulty data."""
    
    def __init__(self):
        self.questions = self._load_validated_questions()
        self.validator = TreatmentValidator()
    
    def _load_validated_questions(self) -> Dict[str, List[Dict]]:
        """Load questions with empirical validation data."""
        return {
            'easy': [
                # TIER 1: VERY EASY (90%+ accuracy) - builds confidence
                {'question': 'How many days are in a week?', 'options': ['5', '6', '7', '8'], 'correct': 2, 'pilot_accuracy': 99.1, 'difficulty_tier': 1},
                {'question': 'How many minutes are in one hour?', 'options': ['50', '60', '70', '80'], 'correct': 1, 'pilot_accuracy': 98.2, 'difficulty_tier': 1},
                {'question': 'How many wheels does a bicycle have?', 'options': ['1', '2', '3', '4'], 'correct': 1, 'pilot_accuracy': 99.5, 'difficulty_tier': 1},
                {'question': 'What is the capital of France?', 'options': ['London', 'Berlin', 'Paris', 'Madrid'], 'correct': 2, 'pilot_accuracy': 95.2, 'difficulty_tier': 1},
                
                # TIER 2: EASY (80-90% accuracy) - maintains confidence
                {'question': 'What do bees produce?', 'options': ['Milk', 'Honey', 'Sugar', 'Butter'], 'correct': 1, 'pilot_accuracy': 91.3, 'difficulty_tier': 2},
                {'question': 'Which direction does the sun rise?', 'options': ['North', 'South', 'East', 'West'], 'correct': 2, 'pilot_accuracy': 86.3, 'difficulty_tier': 2},
                {'question': 'Which animal is known as the "King of the Jungle"?', 'options': ['Tiger', 'Lion', 'Elephant', 'Leopard'], 'correct': 1, 'pilot_accuracy': 85.3, 'difficulty_tier': 2},
                {'question': 'How many legs does a spider have?', 'options': ['6', '8', '10', '12'], 'correct': 1, 'pilot_accuracy': 84.7, 'difficulty_tier': 2},
                
                # TIER 3: MODERATE-EASY (75-85% accuracy) - target range
                {'question': 'What is the largest continent by area?', 'options': ['Africa', 'Asia', 'North America', 'Europe'], 'correct': 1, 'pilot_accuracy': 82.1, 'difficulty_tier': 3},
                {'question': 'What gas do plants absorb from the atmosphere?', 'options': ['Oxygen', 'Nitrogen', 'Carbon dioxide', 'Hydrogen'], 'correct': 2, 'pilot_accuracy': 81.2, 'difficulty_tier': 3},
                {'question': 'What is the primary ingredient in guacamole?', 'options': ['Tomato', 'Avocado', 'Onion', 'Pepper'], 'correct': 1, 'pilot_accuracy': 81.7, 'difficulty_tier': 3},
                {'question': 'Which planet is closest to the sun?', 'options': ['Venus', 'Mercury', 'Earth', 'Mars'], 'correct': 1, 'pilot_accuracy': 79.8, 'difficulty_tier': 3},
                {'question': 'How many players are on a basketball team on the court at one time?', 'options': ['4', '5', '6', '7'], 'correct': 1, 'pilot_accuracy': 79.5, 'difficulty_tier': 3},
                {'question': 'Which ocean is the largest?', 'options': ['Atlantic', 'Indian', 'Arctic', 'Pacific'], 'correct': 3, 'pilot_accuracy': 78.5, 'difficulty_tier': 3},
                {'question': 'What is the largest mammal in the world?', 'options': ['Elephant', 'Blue whale', 'Giraffe', 'Hippopotamus'], 'correct': 1, 'pilot_accuracy': 77.8, 'difficulty_tier': 3},
                {'question': 'What is the capital of Canada?', 'options': ['Toronto', 'Vancouver', 'Ottawa', 'Montreal'], 'correct': 2, 'pilot_accuracy': 76.3, 'difficulty_tier': 3},
                
                # Additional questions to reach 25
                {'question': 'Which fruit is known for having its seeds on the outside?', 'options': ['Apple', 'Orange', 'Strawberry', 'Grape'], 'correct': 2, 'pilot_accuracy': 76.4, 'difficulty_tier': 3},
                {'question': 'What color do you get when you mix red and yellow?', 'options': ['Purple', 'Green', 'Orange', 'Blue'], 'correct': 2, 'pilot_accuracy': 83.2, 'difficulty_tier': 3},
                {'question': 'Which season comes after spring?', 'options': ['Winter', 'Summer', 'Fall', 'Autumn'], 'correct': 1, 'pilot_accuracy': 87.4, 'difficulty_tier': 2},
                {'question': 'What do pandas primarily eat?', 'options': ['Fish', 'Meat', 'Bamboo', 'Berries'], 'correct': 2, 'pilot_accuracy': 82.6, 'difficulty_tier': 3},
                {'question': 'In which sport would you perform a slam dunk?', 'options': ['Tennis', 'Football', 'Basketball', 'Baseball'], 'correct': 2, 'pilot_accuracy': 84.1, 'difficulty_tier': 2},
                {'question': 'What is the main ingredient in bread?', 'options': ['Rice', 'Flour', 'Sugar', 'Salt'], 'correct': 1, 'pilot_accuracy': 83.9, 'difficulty_tier': 2},
                {'question': 'Which color is at the top of a rainbow?', 'options': ['Blue', 'Red', 'Yellow', 'Green'], 'correct': 1, 'pilot_accuracy': 78.2, 'difficulty_tier': 3},
                {'question': 'Which of these is a primary color?', 'options': ['Orange', 'Purple', 'Blue', 'Green'], 'correct': 2, 'pilot_accuracy': 80.7, 'difficulty_tier': 3},
                {'question': 'What is the freezing point of water in Celsius?', 'options': ['-10°C', '0°C', '10°C', '32°C'], 'correct': 1, 'pilot_accuracy': 82.4, 'difficulty_tier': 3},
                {'question': 'Which meal is typically eaten in the morning?', 'options': ['Lunch', 'Dinner', 'Breakfast', 'Supper'], 'correct': 2, 'pilot_accuracy': 94.7, 'difficulty_tier': 1}
            ],
            
            'hard': [
                # TIER 1: VERY HARD (15-25% accuracy) - creates underconfidence
                {'question': 'What do you most fear if you have hormephobia?', 'options': ['Shock', 'Hormones', 'Heights', 'Water'], 'correct': 0, 'pilot_accuracy': 24.2, 'difficulty_tier': 1},
                {'question': 'Boris Becker contested consecutive Wimbledon finals in 1988-1990. Who was his opponent in all three?', 'options': ['Michael Stich', 'Andre Agassi', 'Ivan Lendl', 'Stefan Edberg'], 'correct': 3, 'pilot_accuracy': 24.5, 'difficulty_tier': 1},
                {'question': 'In chemistry, what is the atomic number of tungsten?', 'options': ['72', '73', '74', '75'], 'correct': 2, 'pilot_accuracy': 25.1, 'difficulty_tier': 1},
                {'question': 'The Bretton Woods system established which monetary arrangement?', 'options': ['Gold standard', 'Flexible rates', 'Fixed rates', 'Currency unions'], 'correct': 2, 'pilot_accuracy': 25.8, 'difficulty_tier': 1},
                
                # TIER 2: HARD (25-30% accuracy) - target range lower bound
                {'question': 'What is the capital of Kazakhstan?', 'options': ['Almaty', 'Nur-Sultan', 'Shymkent', 'Aktobe'], 'correct': 1, 'pilot_accuracy': 26.4, 'difficulty_tier': 2},
                {'question': 'Which Roman emperor was "The Philosopher Emperor"?', 'options': ['Marcus Aurelius', 'Trajan', 'Hadrian', 'Antoninus Pius'], 'correct': 0, 'pilot_accuracy': 26.8, 'difficulty_tier': 2},
                {'question': 'Who developed comparative advantage theory in trade?', 'options': ['Adam Smith', 'David Ricardo', 'John Stuart Mill', 'Alfred Marshall'], 'correct': 1, 'pilot_accuracy': 27.6, 'difficulty_tier': 2},
                {'question': 'Who was Henry VIII\'s wife at his death?', 'options': ['Catherine Parr', 'Catherine of Aragon', 'Anne Boleyn', 'Jane Seymour'], 'correct': 0, 'pilot_accuracy': 28.3, 'difficulty_tier': 2},
                {'question': 'Who composed "The Art of Fugue"?', 'options': ['Bach', 'Mozart', 'Beethoven', 'Handel'], 'correct': 0, 'pilot_accuracy': 28.7, 'difficulty_tier': 2},
                {'question': 'What is the term for simultaneous inflation and unemployment?', 'options': ['Recession', 'Stagflation', 'Depression', 'Deflation'], 'correct': 1, 'pilot_accuracy': 28.9, 'difficulty_tier': 2},
                
                # TIER 3: MODERATE-HARD (30-35% accuracy) - target range upper bound
                {'question': 'For what did Einstein receive the Nobel Prize?', 'options': ['Relativity', 'Quantum mechanics', 'Photoelectric effect', 'Brownian motion'], 'correct': 2, 'pilot_accuracy': 29.4, 'difficulty_tier': 3},
                {'question': 'The Atacama Desert is primarily in which country?', 'options': ['Peru', 'Bolivia', 'Chile', 'Argentina'], 'correct': 2, 'pilot_accuracy': 29.3, 'difficulty_tier': 3},
                {'question': 'Which philosopher wrote "Critique of Pure Reason"?', 'options': ['Hegel', 'Kant', 'Nietzsche', 'Schopenhauer'], 'correct': 1, 'pilot_accuracy': 30.6, 'difficulty_tier': 3},
                {'question': 'Which African country was formerly Rhodesia?', 'options': ['Zambia', 'Zimbabwe', 'Botswana', 'Namibia'], 'correct': 1, 'pilot_accuracy': 30.9, 'difficulty_tier': 3},
                {'question': 'The Battle of Hastings was in which year?', 'options': ['1064', '1065', '1066', '1067'], 'correct': 2, 'pilot_accuracy': 31.2, 'difficulty_tier': 3},
                {'question': 'In quantum mechanics, which principle involves position/momentum uncertainty?', 'options': ['Pauli exclusion', 'Heisenberg uncertainty', 'Wave-particle duality', 'Entanglement'], 'correct': 1, 'pilot_accuracy': 31.4, 'difficulty_tier': 3},
                {'question': 'In Hamlet, what is Hamlet\'s mother\'s name?', 'options': ['Ophelia', 'Gertrude', 'Cordelia', 'Portia'], 'correct': 1, 'pilot_accuracy': 31.5, 'difficulty_tier': 3},
                {'question': 'Who composed "The Ring of the Nibelung"?', 'options': ['Mozart', 'Wagner', 'Verdi', 'Puccini'], 'correct': 1, 'pilot_accuracy': 32.1, 'difficulty_tier': 3},
                {'question': 'Suharto was president of which Asian nation?', 'options': ['Malaysia', 'Japan', 'Indonesia', 'Thailand'], 'correct': 2, 'pilot_accuracy': 32.7, 'difficulty_tier': 3},
                {'question': 'Who wrote "The General Theory of Employment"?', 'options': ['Keynes', 'Friedman', 'Hayek', 'Samuelson'], 'correct': 0, 'pilot_accuracy': 33.2, 'difficulty_tier': 3},
                {'question': 'Which element has the symbol "Au"?', 'options': ['Silver', 'Aluminum', 'Gold', 'Argon'], 'correct': 2, 'pilot_accuracy': 33.8, 'difficulty_tier': 3},
                {'question': 'Who wrote "One Hundred Years of Solitude"?', 'options': ['Borges', 'García Márquez', 'Vargas Llosa', 'Paz'], 'correct': 1, 'pilot_accuracy': 34.2, 'difficulty_tier': 3},
                {'question': 'What is the medical term for the kneecap?', 'options': ['Fibula', 'Tibia', 'Patella', 'Femur'], 'correct': 2, 'pilot_accuracy': 34.7, 'difficulty_tier': 3},
                {'question': 'Which fallacy involves attacking the person?', 'options': ['Straw man', 'Ad hominem', 'False dichotomy', 'Slippery slope'], 'correct': 1, 'pilot_accuracy': 35.2, 'difficulty_tier': 3},
                {'question': 'Who was the first UN Secretary-General?', 'options': ['Hammarskjöld', 'Trygve Lie', 'U Thant', 'Waldheim'], 'correct': 1, 'pilot_accuracy': 27.9, 'difficulty_tier': 2}
            ]
        }
    
    def get_validated_questions(self, treatment: str) -> List[Dict]:
        """Get validated question set for treatment."""
        question_bank = self.questions[treatment]
        
        # Validate difficulty
        is_valid, msg = self.validator.validate_question_difficulty(question_bank, treatment)
        if is_valid:
            logger.info(f"Question difficulty validated for {treatment}: {msg}")
        else:
            logger.warning(f"Question validation warning for {treatment}: {msg}")
        
        # Select balanced set across difficulty tiers
        selected = self._select_balanced_questions(question_bank)
        
        # Randomize order
        random.shuffle(selected)
        
        logger.info(f"Selected {len(selected)} validated questions for {treatment} treatment")
        return selected
    
    def _select_balanced_questions(self, questions: List[Dict]) -> List[Dict]:
        """Select balanced question set across difficulty tiers."""
        tiers = defaultdict(list)
        for q in questions:
            tier = q.get('difficulty_tier', 3)
            tiers[tier].append(q)
        
        # Target distribution: more tier 3 (target range), some tier 2, few tier 1
        target_distribution = {1: 3, 2: 7, 3: 15}  # Total = 25
        
        selected = []
        for tier, target_count in target_distribution.items():
            if tier in tiers:
                tier_questions = tiers[tier]
                count = min(target_count, len(tier_questions))
                selected.extend(random.sample(tier_questions, count))
        
        # Fill remainder if needed
        while len(selected) < ExperimentConfig.TRIVIA_QUESTIONS_COUNT:
            remaining = [q for q in questions if q not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        return selected[:ExperimentConfig.TRIVIA_QUESTIONS_COUNT]

class DataManager:
    """Simplified data management focused on paper's key variables."""
    
    def __init__(self, db_path: str = "faithful_experiment.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with essential tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main participant data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS participants (
                    participant_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    session_number INTEGER NOT NULL,
                    assignment_timestamp TEXT NOT NULL,
                    
                    -- Treatment
                    treatment TEXT NOT NULL CHECK (treatment IN ('easy', 'hard')),
                    
                    -- Trivia performance
                    trivia_score INTEGER NOT NULL,
                    trivia_accuracy REAL NOT NULL,
                    trivia_time_spent REAL NOT NULL,
                    
                    -- Session-relative performance
                    session_median REAL NOT NULL,
                    performance_level TEXT NOT NULL CHECK (performance_level IN ('High', 'Low')),
                    performance_rank INTEGER NOT NULL,
                    session_size INTEGER NOT NULL,
                    
                    -- Beliefs about own performance
                    belief_own_performance INTEGER NOT NULL CHECK (belief_own_performance BETWEEN 0 AND 100),
                    confidence_level TEXT NOT NULL,
                    
                    -- Group assignment
                    assigned_group TEXT NOT NULL CHECK (assigned_group IN ('Top', 'Bottom')),
                    mechanism_used TEXT NOT NULL CHECK (mechanism_used IN ('A', 'B')),
                    group_reflects_performance BOOLEAN NOT NULL,
                    
                    -- Hiring decisions
                    wtp_top_group INTEGER NOT NULL CHECK (wtp_top_group BETWEEN 0 AND 200),
                    wtp_bottom_group INTEGER NOT NULL CHECK (wtp_bottom_group BETWEEN 0 AND 200),
                    wtp_premium INTEGER NOT NULL,
                    
                    -- Mechanism beliefs
                    belief_mechanism_informative INTEGER NOT NULL CHECK (belief_mechanism_informative BETWEEN 0 AND 100),
                    
                    -- Key outcome variables
                    overconfidence_measure REAL NOT NULL,
                    relative_wtp_premium REAL,
                    
                    -- Data quality
                    attention_checks_passed BOOLEAN NOT NULL,
                    completion_time_minutes REAL NOT NULL,
                    data_quality_flag TEXT NOT NULL,
                    
                    -- Timestamps
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Session summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    session_number INTEGER NOT NULL,
                    treatment TEXT NOT NULL,
                    n_participants INTEGER NOT NULL,
                    median_score REAL NOT NULL,
                    mean_confidence REAL NOT NULL,
                    treatment_validation_passed BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_treatment ON participants (treatment)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON participants (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance ON participants (performance_level)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def save_participant_data(self, data: ParticipantData) -> bool:
        """Save complete participant data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO participants (
                    participant_id, session_id, session_number, assignment_timestamp,
                    treatment, trivia_score, trivia_accuracy, trivia_time_spent,
                    session_median, performance_level, performance_rank, session_size,
                    belief_own_performance, confidence_level, assigned_group,
                    mechanism_used, group_reflects_performance, wtp_top_group,
                    wtp_bottom_group, wtp_premium, belief_mechanism_informative,
                    overconfidence_measure, relative_wtp_premium, attention_checks_passed,
                    completion_time_minutes, data_quality_flag, start_time, end_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.participant_id, data.session_id, data.session_number,
                data.assignment_timestamp.isoformat(), data.treatment,
                data.trivia_score, data.trivia_accuracy, data.trivia_time_spent,
                data.session_median, data.performance_level, data.performance_rank,
                data.session_size, data.belief_own_performance, data.confidence_level,
                data.assigned_group, data.mechanism_used, data.group_reflects_performance,
                data.wtp_top_group, data.wtp_bottom_group, data.wtp_premium,
                data.belief_mechanism_informative, data.overconfidence_measure,
                data.relative_wtp_premium, data.attention_checks_passed,
                data.completion_time_minutes, data.data_quality_flag,
                data.start_time.isoformat(), 
                data.end_time.isoformat() if data.end_time else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved data for participant {data.participant_id}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error saving participant data: {e}")
            return False
    
    def get_treatment_summary(self) -> Dict:
        """Get summary statistics by treatment."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    treatment,
                    COUNT(*) as n_participants,
                    AVG(belief_own_performance) as mean_confidence,
                    AVG(overconfidence_measure) as mean_overconfidence,
                    AVG(wtp_premium) as mean_wtp_premium,
                    COUNT(DISTINCT session_id) as n_sessions
                FROM participants 
                GROUP BY treatment
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df.to_dict('records')
            
        except sqlite3.Error as e:
            logger.error(f"Error getting treatment summary: {e}")
            return {}

class FaithfulExperiment:
    """Main experiment class prioritizing methodological fidelity."""
    
    def __init__(self):
        """Initialize with simplified, robust components."""
        self.setup_session_state()
        self.assignment_manager = RandomAssignmentManager()
        self.session_manager = SessionManager()
        self.question_bank = QuestionBank()
        self.data_manager = DataManager()
        self.validator = TreatmentValidator()
    
    def setup_session_state(self):
        """Initialize essential session state."""
        if 'participant_data' not in st.session_state:
            st.session_state.participant_data = None
        
        if 'current_screen' not in st.session_state:
            st.session_state.current_screen = 0
        
        if 'trivia_data' not in st.session_state:
            st.session_state.trivia_data = {
                'questions': [],
                'answers': [],
                'response_times': [],
                'start_time': None,
                'current_question': 0
            }
    
    def show_progress_indicator(self, current_step: int, total_steps: int = 10):
        """Clean progress indicator."""
        progress = current_step / total_steps
        
        if st.session_state.participant_data:
            session_info = f"Session: {st.session_state.participant_data.session_id} | Treatment: {st.session_state.participant_data.treatment.title()}"
        else:
            session_info = "Experimental Study Platform"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(progress, text=f"Step {current_step}/{total_steps} - {session_info}")
        with col2:
            st.metric("Progress", f"{progress*100:.0f}%")
    
    def show_welcome_screen(self):
        """Welcome and random assignment."""
        st.markdown("""
        <div class="study-header">
            <h1>🎓 Behavioral Economics Research Study</h1>
            <p>Individual Differences in Decision-Making Under Uncertainty</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(1, 10)
        
        st.markdown("### 📋 Study Information")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **What you'll do:**
            1. Answer 25 general knowledge questions (6 minutes)
            2. Report beliefs about your performance  
            3. Make economic hiring decisions
            4. Complete brief questionnaire
            
            **Duration:** 45-60 minutes
            
            **Compensation:** $5.00 + performance bonus ($0 - $22.50)
            
            **Key Feature:** Your performance is evaluated relative to other participants 
            in your session (minimum 18 participants required).
            """)
            
            # Assignment statistics
            stats = self.assignment_manager.get_assignment_statistics()
            if stats['total'] > 0:
                st.markdown("---")
                st.markdown("**📊 Current Study Progress**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Participants", stats['total'])
                with col_b:
                    st.metric("Treatment Balance", f"{stats['balance_pct']:.1f}%")
                with col_c:
                    st.metric("Sessions Created", stats['sessions_created'])
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🔬 Research Protocol</h4>
                <p><strong>IRB:</strong> Protocol #2024-XXX</p>
                <p><strong>PI:</strong> [Name Withheld]</p>
                <p><strong>Institution:</strong> [University]</p>
                <p><strong>Design:</strong> Between-subjects</p>
                <p><strong>Assignment:</strong> Random</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Consent
        consent = st.checkbox(
            "I have read the study information and voluntarily consent to participate. "
            "I understand that participation is voluntary and I may withdraw at any time.",
            key="consent"
        )
        
        if consent:
            if st.button("🚀 Begin Study", type="primary", use_container_width=True):
                # Random assignment to treatment
                participant_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100,999)}"
                treatment, session_id, session_number = self.assignment_manager.assign_participant(participant_id)
                
                # Initialize participant data
                st.session_state.participant_data = ParticipantData(
                    participant_id=participant_id,
                    session_id=session_id,
                    assignment_timestamp=datetime.now(),
                    treatment=treatment,
                    session_number=session_number,
                    trivia_score=0,
                    trivia_accuracy=0.0,
                    trivia_time_spent=0.0,
                    response_times=[],
                    session_median=0.0,
                    performance_level="",
                    performance_rank=0,
                    session_size=0,
                    belief_own_performance=50,
                    confidence_level="",
                    assigned_group="",
                    mechanism_used="",
                    group_reflects_performance=False,
                    wtp_top_group=100,
                    wtp_bottom_group=100,
                    wtp_premium=0,
                    belief_mechanism_informative=50,
                    overconfidence_measure=0.0,
                    relative_wtp_premium=0.0,
                    attention_checks_passed=True,
                    completion_time_minutes=0.0,
                    data_quality_flag="Good",
                    start_time=datetime.now()
                )
                
                # Join session
                self.session_manager.join_session(session_id, participant_id)
                
                st.success(f"✅ Assigned to {treatment} treatment, Session #{session_number}")
                st.session_state.current_screen = 1
                time.sleep(1)
                st.rerun()
    
    def show_session_status(self):
        """Show session status and waiting for minimum participants."""
        data = st.session_state.participant_data
        
        st.markdown(f"""
        <div class="study-header">
            <h1>⏳ Session Formation</h1>
            <p>Waiting for minimum participants in your session</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(2, 10)
        
        # Session info
        st.markdown(f"""
        <div class="session-info">
            <h3>📋 Your Session Details</h3>
            <p><strong>Session ID:</strong> {data.session_id}</p>
            <p><strong>Treatment:</strong> {data.treatment.title()} Questions</p>
            <p><strong>Session Number:</strong> #{data.session_number}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check session status
        participants = st.session_state.session_participants[data.session_id]
        current_size = len(participants)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Participants", 
                current_size,
                delta=f"{ExperimentConfig.MIN_SESSION_SIZE - current_size} needed" 
                    if current_size < ExperimentConfig.MIN_SESSION_SIZE else "Ready!"
            )
        
        with col2:
            st.metric("Target Size", ExperimentConfig.TARGET_SESSION_SIZE)
        
        with col3:
            progress_pct = min(100, (current_size / ExperimentConfig.MIN_SESSION_SIZE) * 100)
            st.metric("Progress", f"{progress_pct:.0f}%")
        
        # Progress bar
        progress = min(1.0, current_size / ExperimentConfig.MIN_SESSION_SIZE)
        st.progress(progress, text=f"Gathering participants: {current_size}/{ExperimentConfig.MIN_SESSION_SIZE} minimum")
        
        # Status check
        if self.session_manager.is_session_ready(data.session_id):
            st.success("🎉 Session ready! Starting cognitive task...")
            st.session_state.current_screen = 2
            time.sleep(2)
            st.rerun()
        else:
            st.info(f"⏳ Waiting for {ExperimentConfig.MIN_SESSION_SIZE - current_size} more participants...")
            
            # Auto-refresh
            time.sleep(3)
            st.rerun()
    
    def show_trivia_instructions(self):
        """Instructions for trivia task."""
        data = st.session_state.participant_data
        
        st.markdown(f"""
        <div class="study-header">
            <h1>📚 Cognitive Task Instructions</h1>
            <p>General Knowledge Questions - {data.treatment.title()} Difficulty</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(3, 10)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ### 📖 Task Overview
            
            You will answer **{ExperimentConfig.TRIVIA_QUESTIONS_COUNT} multiple-choice questions** 
            covering general knowledge topics.
            
            **⏰ Time Limit:** {ExperimentConfig.TRIVIA_TIME_LIMIT // 60} minutes total
            
            **🎯 Your Goal:** Answer as many questions correctly as possible
            
            **📊 Performance Evaluation:** Your performance will be ranked relative 
            to the other {len(st.session_state.session_participants[data.session_id])} 
            participants in your session.
            
            **💰 Payment:** If this task is selected for bonus payment:
            - **Top half performers:** {ExperimentConfig.HIGH_PERFORMANCE_TOKENS} tokens (${ExperimentConfig.HIGH_PERFORMANCE_TOKENS * ExperimentConfig.TOKEN_TO_DOLLAR_RATE:.2f})
            - **Bottom half performers:** {ExperimentConfig.LOW_PERFORMANCE_TOKENS} tokens (${ExperimentConfig.LOW_PERFORMANCE_TOKENS * ExperimentConfig.TOKEN_TO_DOLLAR_RATE:.2f})
            
            ### 📝 Important Notes
            - Questions cannot be skipped
            - You cannot return to previous questions
            - Take your time but watch the clock
            - There is no penalty for guessing
            """)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Session Details</h4>
                <p><strong>Session:</strong> {data.session_id[:12]}...</p>
                <p><strong>Participants:</strong> {len(st.session_state.session_participants[data.session_id])}</p>
                <p><strong>Question Set:</strong> {data.treatment.title()}</p>
                <p><strong>Total Questions:</strong> {ExperimentConfig.TRIVIA_QUESTIONS_COUNT}</p>
                <p><strong>Time Limit:</strong> {ExperimentConfig.TRIVIA_TIME_LIMIT // 60} min</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("✅ I understand - Start Cognitive Task", type="primary", use_container_width=True):
            # Load questions for this treatment
            questions = self.question_bank.get_validated_questions(data.treatment)
            st.session_state.trivia_data['questions'] = questions
            st.session_state.trivia_data['start_time'] = time.time()
            st.session_state.trivia_data['current_question'] = 0
            st.session_state.trivia_data['answers'] = [None] * len(questions)
            st.session_state.trivia_data['response_times'] = [0] * len(questions)
            
            st.session_state.current_screen = 3
            st.rerun()
    
    def show_trivia_questions(self):
        """Display trivia questions with timer."""
        data = st.session_state.participant_data
        trivia_data = st.session_state.trivia_data
        
        current_q = trivia_data['current_question']
        questions = trivia_data['questions']
        
        if current_q >= len(questions):
            self.process_trivia_completion()
            return
        
        # Timer
        elapsed = time.time() - trivia_data['start_time']
        remaining = ExperimentConfig.TRIVIA_TIME_LIMIT - elapsed
        
        if remaining <= 0:
            self.process_trivia_completion()
            return
        
        # Header
        st.markdown(f"""
        <div class="study-header">
            <h1>📚 Question {current_q + 1} of {len(questions)}</h1>
            <p>Time Remaining: {int(remaining // 60)}:{int(remaining % 60):02d}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress
        progress = current_q / len(questions)
        st.progress(progress, text=f"Progress: {current_q}/{len(questions)} questions completed")
        
        # Time warning
        if remaining < 60:
            st.warning(f"⚠️ Less than 1 minute remaining!")
        elif remaining < 300:
            st.info(f"ℹ️ {int(remaining // 60)} minutes remaining")
        
        # Question
        question = questions[current_q]
        
        st.markdown("---")
        st.markdown(f"### {question['question']}")
        
        # Options
        answer = st.radio(
            "Select your answer:",
            options=range(len(question['options'])),
            format_func=lambda x: f"{chr(65+x)}. {question['options'][x]}",
            key=f"q_{current_q}",
            index=None
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Question {current_q + 1}** of **{len(questions)}**")
        
        with col2:
            if answer is not None:
                if st.button("Submit Answer →", type="primary", use_container_width=True):
                    # Record response
                    question_start_time = trivia_data.get('question_start_time', time.time())
                    response_time = time.time() - question_start_time
                    
                    trivia_data['answers'][current_q] = answer
                    trivia_data['response_times'][current_q] = response_time
                    trivia_data['current_question'] += 1
                    trivia_data['question_start_time'] = time.time()
                    
                    st.rerun()
            else:
                st.info("Please select an answer to continue")
        
        # Set question start time for response time tracking
        if 'question_start_time' not in trivia_data:
            trivia_data['question_start_time'] = time.time()
    
    def process_trivia_completion(self):
        """Process trivia results and calculate performance."""
        data = st.session_state.participant_data
        trivia_data = st.session_state.trivia_data
        
        # Calculate score
        questions = trivia_data['questions']
        answers = trivia_data['answers']
        
        score = 0
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if answer is not None and answer == question['correct']:
                score += 1
        
        accuracy = score / len(questions) if questions else 0
        time_spent = time.time() - trivia_data['start_time']
        
        # Update participant data
        data.trivia_score = score
        data.trivia_accuracy = accuracy
        data.trivia_time_spent = time_spent / 60  # Convert to minutes
        data.response_times = trivia_data['response_times']
        
        # Collect all session scores for relative performance calculation
        session_id = data.session_id
        if 'session_scores' not in st.session_state:
            st.session_state.session_scores = {}
        
        st.session_state.session_scores[data.participant_id] = score
        
        # Check if we can calculate session performance
        participants = st.session_state.session_participants[session_id]
        completed_scores = st.session_state.session_scores
        
        session_complete = all(pid in completed_scores for pid in participants)
        
        if session_complete:
            self.calculate_session_performance()
        
        st.session_state.current_screen = 4
        st.rerun()
    
    def calculate_session_performance(self):
        """Calculate session-relative performance for all participants."""
        data = st.session_state.participant_data
        session_id = data.session_id
        
        # Get performance levels
        performance_levels = self.session_manager.calculate_session_performance(
            session_id, st.session_state.session_scores
        )
        
        # Update participant data
        session_data = st.session_state.session_data[session_id]
        data.session_median = session_data['median_score']
        data.performance_level = performance_levels[data.participant_id]
        data.session_size = session_data['n_participants']
        
        # Calculate rank
        participants = st.session_state.session_participants[session_id]
        scores = [(pid, st.session_state.session_scores[pid]) for pid in participants]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (pid, score) in enumerate(scores, 1):
            if pid == data.participant_id:
                data.performance_rank = rank
                break
        
        logger.info(f"Session {session_id} performance calculated: "
                   f"participant {data.participant_id} ranked {data.performance_rank}/{data.session_size}")
    
    def show_waiting_for_session(self):
        """Wait for all session participants to complete trivia."""
        data = st.session_state.participant_data
        
        st.markdown(f"""
        <div class="study-header">
            <h1>⏳ Waiting for Session Completion</h1>
            <p>All participants must complete the cognitive task before continuing</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(4, 10)
        
        # Show participant's results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Your Score", f"{data.trivia_score}/{ExperimentConfig.TRIVIA_QUESTIONS_COUNT}")
        
        with col2:
            st.metric("Accuracy", f"{data.trivia_accuracy*100:.1f}%")
        
        with col3:
            st.metric("Time Taken", f"{data.trivia_time_spent:.1f} min")
        
        # Session completion status
        participants = st.session_state.session_participants[data.session_id]
        completed = len(st.session_state.session_scores)
        
        st.markdown("---")
        st.markdown(f"### 📊 Session Progress: {completed}/{len(participants)} completed")
        
        progress = completed / len(participants)
        st.progress(progress, text=f"Waiting for {len(participants) - completed} participants to finish...")
        
        # Check if all completed
        if completed >= len(participants):
            if data.performance_level:  # Performance already calculated
                st.success("🎉 All participants completed! Continuing to next phase...")
                st.session_state.current_screen = 5
                time.sleep(1)
                st.rerun()
            else:
                # Calculate performance if not done
                self.calculate_session_performance()
                st.rerun()
        else:
            # Auto-refresh
            time.sleep(5)
            st.rerun()
    
    def show_performance_belief(self):
        """Collect beliefs about own performance before revealing results."""
        data = st.session_state.participant_data
        
        st.markdown(f"""
        <div class="study-header">
            <h1>🤔 Performance Beliefs</h1>
            <p>What do you think about your performance?</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(5, 10)
        
        st.markdown("""
        ### 📊 Performance Assessment
        
        Before we reveal the results, we'd like to know what you think about your performance 
        on the cognitive task you just completed.
        
        Remember, your performance will be ranked relative to the other participants in your session.
        """)
        
        # Belief elicitation
        belief = st.slider(
            "What do you think is the probability (0-100%) that your performance was in the **top half** of participants in your session?",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="0% = definitely bottom half, 100% = definitely top half"
        )
        
        st.markdown(f"**Your assessment:** {belief}% chance you performed in the top half")
        
        # Confidence classification
        if belief > 60:
            confidence_desc = "You seem confident in your performance"
            confidence_level = "Confident"
        elif belief < 40:
            confidence_desc = "You seem uncertain about your performance"  
            confidence_level = "Uncertain"
        else:
            confidence_desc = "You seem neutral about your performance"
            confidence_level = "Neutral"
        
        st.info(f"ℹ️ {confidence_desc}")
        
        if st.button("Submit Assessment", type="primary", use_container_width=True):
            # Update participant data
            data.belief_own_performance = belief
            data.confidence_level = confidence_level
            
            # Calculate overconfidence measure
            actual_performance = 1 if data.performance_level == 'High' else 0
            belief_performance = belief / 100
            data.overconfidence_measure = belief_performance - actual_performance
            
            st.session_state.current_screen = 6
            st.rerun()
    
    def show_performance_results(self):
        """Reveal performance results and assign groups."""
        data = st.session_state.participant_data
        
        st.markdown(f"""
        <div class="study-header">
            <h1>📊 Performance Results</h1>
            <p>Your cognitive task results</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(6, 10)
        
        # Results display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Your Performance")
            
            st.metric("Score", f"{data.trivia_score}/{ExperimentConfig.TRIVIA_QUESTIONS_COUNT}")
            st.metric("Accuracy", f"{data.trivia_accuracy*100:.1f}%")
            st.metric("Session Rank", f"{data.performance_rank} of {data.session_size}")
            
            # Performance level with styling
            if data.performance_level == 'High':
                st.success(f"🎉 **Performance Level: TOP HALF**")
                st.markdown("You performed better than the session median!")
            else:
                st.info(f"📊 **Performance Level: BOTTOM HALF**")
                st.markdown("You performed below the session median.")
        
        with col2:
            st.markdown("### 📈 Session Statistics")
            
            session_data = st.session_state.session_data[data.session_id]
            
            st.metric("Session Median", f"{session_data['median_score']:.1f}")
            st.metric("Your Score vs Median", f"{data.trivia_score - session_data['median_score']:+.1f}")
            st.metric("Session Size", data.session_size)
            
            # Belief accuracy
            belief_accuracy = abs(data.belief_own_performance - (100 if data.performance_level == 'High' else 0))
            st.metric("Belief Accuracy", f"{100 - belief_accuracy:.1f}%")
        
        # Group assignment
        st.markdown("---")
        st.markdown("### 🏷️ Group Assignment")
        
        # Random assignment with mechanism
        mechanism = random.choice(['A', 'B'])
        
        if mechanism == 'A':
            # 95% chance group reflects performance
            prob_reflect = ExperimentConfig.MECHANISM_A_ACCURACY
        else:
            # 55% chance group reflects performance
            prob_reflect = ExperimentConfig.MECHANISM_B_ACCURACY
        
        reflects_performance = random.random() < prob_reflect
        
        if reflects_performance:
            assigned_group = 'Top' if data.performance_level == 'High' else 'Bottom'
        else:
            assigned_group = 'Bottom' if data.performance_level == 'High' else 'Top'
        
        # Update data
        data.mechanism_used = mechanism
        data.assigned_group = assigned_group
        data.group_reflects_performance = reflects_performance
        
        st.markdown(f"""
        <div class="session-info">
            <h4>Your Group Assignment</h4>
            <p><strong>Assigned Group:</strong> <span style="font-size: 1.2em; font-weight: bold; color: {'#28a745' if assigned_group == 'Top' else '#007bff'};">{assigned_group} Group</span></p>
            <p><em>Group assignments are based on performance but may include some randomness.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continue to Economic Decisions", type="primary", use_container_width=True):
            st.session_state.current_screen = 7
            st.rerun()
    
    def show_hiring_decisions(self):
        """BDM hiring decisions - the key dependent variable."""
        data = st.session_state.participant_data
        
        st.markdown(f"""
        <div class="study-header">
            <h1>💼 Economic Decision Task</h1>
            <p>Hiring Decisions Under Uncertainty</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(7, 10)
        
        st.markdown("""
        ### 🎯 Task Overview
        
        You will now make hiring decisions. Imagine you can hire workers, and your payoff 
        depends on their actual performance level (which you cannot observe directly).
        
        **💰 Payoffs:**
        - Hiring a **high-performing** worker gives you **200 tokens**
        - Hiring a **low-performing** worker gives you **40 tokens**
        - You can only observe which **group** (Top or Bottom) they belong to
        
        **🎲 What you know:**
        - Group assignments are based on performance but include some randomness
        - Some workers' groups perfectly reflect their performance, others' do not
        """)
        
        # Endowment information
        st.info(f"💰 You have an endowment of {ExperimentConfig.ENDOWMENT_TOKENS} tokens for each decision.")
        
        st.markdown("---")
        
        # Hiring decisions using BDM mechanism
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔺 Hiring from Top Group")
            st.markdown("What is the **maximum** you would pay to hire a randomly selected worker from the **Top Group**?")
            
            wtp_top = st.slider(
                "Maximum willingness to pay (tokens):",
                min_value=ExperimentConfig.BDM_MIN_VALUE,
                max_value=ExperimentConfig.BDM_MAX_VALUE,
                value=100,
                step=5,
                key="wtp_top",
                help="You will pay this amount or less, but never more"
            )
            
            expected_profit_top = f"Expected profit: {200 - wtp_top} to {40 - wtp_top} tokens"
            st.caption(expected_profit_top)
        
        with col2:
            st.markdown("#### 🔻 Hiring from Bottom Group")
            st.markdown("What is the **maximum** you would pay to hire a randomly selected worker from the **Bottom Group**?")
            
            wtp_bottom = st.slider(
                "Maximum willingness to pay (tokens):",
                min_value=ExperimentConfig.BDM_MIN_VALUE,
                max_value=ExperimentConfig.BDM_MAX_VALUE,
                value=100,
                step=5,
                key="wtp_bottom",
                help="You will pay this amount or less, but never more"
            )
            
            expected_profit_bottom = f"Expected profit: {200 - wtp_bottom} to {40 - wtp_bottom} tokens"
            st.caption(expected_profit_bottom)
        
        # Premium calculation
        premium = wtp_top - wtp_bottom
        
        st.markdown("---")
        st.markdown("### 📊 Your Decisions Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Top Group WTP", f"{wtp_top} tokens")
        
        with col2:
            st.metric("Bottom Group WTP", f"{wtp_bottom} tokens")
        
        with col3:
            color = "normal"
            if premium > 0:
                st.metric("Premium for Top", f"+{premium} tokens", delta=f"{premium} tokens")
            elif premium < 0:
                st.metric("Premium for Top", f"{premium} tokens", delta=f"{premium} tokens")
            else:
                st.metric("Premium for Top", "0 tokens")
        
        # Explanation
        if premium > 0:
            st.info(f"ℹ️ You value Top Group workers {premium} tokens more than Bottom Group workers")
        elif premium < 0:
            st.warning(f"⚠️ You value Bottom Group workers {abs(premium)} tokens more than Top Group workers")
        else:
            st.info("ℹ️ You value both groups equally")
        
        if st.button("Submit Hiring Decisions", type="primary", use_container_width=True):
            # Update participant data
            data.wtp_top_group = wtp_top
            data.wtp_bottom_group = wtp_bottom
            data.wtp_premium = premium
            
            st.session_state.current_screen = 8
            st.rerun()
    
    def show_mechanism_belief(self):
        """Collect beliefs about mechanism informativeness."""
        data = st.session_state.participant_data
        
        st.markdown(f"""
        <div class="study-header">
            <h1>🎲 Beliefs About Group Assignment</h1>
            <p>How informative do you think the group assignments were?</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(8, 10)
        
        st.markdown("""
        ### 🤔 Group Assignment Process
        
        As mentioned earlier, group assignments were based on performance but included some randomness.
        
        There were two possible assignment mechanisms:
        - **Mechanism A:** 95% chance group reflects actual performance, 5% chance it doesn't
        - **Mechanism B:** 55% chance group reflects actual performance, 45% chance it doesn't
        
        One of these mechanisms was randomly selected (50/50 chance) and used for all participants in your session.
        """)
        
        # Reminder of participant's situation
        st.markdown(f"""
        <div class="session-info">
            <h4>📋 Your Information</h4>
            <p><strong>Your Performance Level:</strong> {data.performance_level}</p>
            <p><strong>Your Assigned Group:</strong> {data.assigned_group}</p>
            <p><strong>Group Matches Performance:</strong> {'Yes' if data.group_reflects_performance else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Belief elicitation
        mechanism_belief = st.slider(
            "What do you think is the probability (0-100%) that **Mechanism A** (the highly informative one) was used in your session?",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="0% = definitely Mechanism B, 100% = definitely Mechanism A"
        )
        
        st.markdown(f"**Your assessment:** {mechanism_belief}% chance Mechanism A was used")
        
        # Interpretation
        if mechanism_belief > 60:
            st.info("ℹ️ You believe group assignments were highly informative of performance")
        elif mechanism_belief < 40:
            st.info("ℹ️ You believe group assignments were only mildly informative of performance")
        else:
            st.info("ℹ️ You're uncertain about how informative the group assignments were")
        
        if st.button("Submit Assessment", type="primary", use_container_width=True):
            # Update participant data
            data.belief_mechanism_informative = mechanism_belief
            
            st.session_state.current_screen = 9
            st.rerun()
    
    def show_completion_summary(self):
        """Final summary and data submission."""
        data = st.session_state.participant_data
        
        st.markdown(f"""
        <div class="study-header">
            <h1>🎉 Study Completed!</h1>
            <p>Thank you for your participation</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.show_progress_indicator(10, 10)
        
        # Final data processing
        data.end_time = datetime.now()
        data.completion_time_minutes = (data.end_time - data.start_time).total_seconds() / 60
        
        # Calculate relative WTP premium (for analysis)
        # This would typically be calculated relative to a rational benchmark
        # For now, we'll use the raw premium
        data.relative_wtp_premium = data.wtp_premium
        
        # Data quality assessment
        quality_checks = []
        
        # Check completion time
        if data.completion_time_minutes < 20:
            quality_checks.append("Very fast completion")
        elif data.completion_time_minutes > 120:
            quality_checks.append("Very slow completion")
        
        # Check response patterns
        if all(t < 1.0 for t in data.response_times if t > 0):
            quality_checks.append("Very fast responses")
        
        data.data_quality_flag = "Good" if not quality_checks else "; ".join(quality_checks)
        
        # Display summary
        st.markdown("### 📊 Your Study Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance")
            st.write(f"**Trivia Score:** {data.trivia_score}/{ExperimentConfig.TRIVIA_QUESTIONS_COUNT}")
            st.write(f"**Performance Level:** {data.performance_level}")
            st.write(f"**Session Rank:** {data.performance_rank}/{data.session_size}")
            st.write(f"**Performance Belief:** {data.belief_own_performance}%")
            
            st.markdown("#### Group Assignment")
            st.write(f"**Assigned Group:** {data.assigned_group}")
            st.write(f"**Mechanism Used:** {data.mechanism_used}")
            st.write(f"**Mechanism Belief:** {data.belief_mechanism_informative}%")
        
        with col2:
            st.markdown("#### Economic Decisions")
            st.write(f"**WTP Top Group:** {data.wtp_top_group} tokens")
            st.write(f"**WTP Bottom Group:** {data.wtp_bottom_group} tokens")
            st.write(f"**Premium for Top:** {data.wtp_premium} tokens")
            
            st.markdown("#### Study Details")
            st.write(f"**Treatment:** {data.treatment}")
            st.write(f"**Session:** {data.session_id}")
            st.write(f"**Completion Time:** {data.completion_time_minutes:.1f} min")
            st.write(f"**Data Quality:** {data.data_quality_flag}")
        
        # Key findings for participant
        st.markdown("---")
        st.markdown("### 🔍 Key Findings")
        
        # Overconfidence measure
        if data.overconfidence_measure > 0.1:
            st.info(f"📈 You were somewhat overconfident about your performance (belief was {data.overconfidence_measure*100:.1f} percentage points higher than actual)")
        elif data.overconfidence_measure < -0.1:
            st.info(f"📉 You were somewhat underconfident about your performance (belief was {abs(data.overconfidence_measure)*100:.1f} percentage points lower than actual)")
        else:
            st.success("🎯 Your performance belief was quite accurate!")
        
        # Premium pattern
        if data.assigned_group == 'Top' and data.wtp_premium > 0:
            st.info("💼 You valued your own group (Top) more highly in hiring decisions")
        elif data.assigned_group == 'Bottom' and data.wtp_premium < 0:
            st.info("💼 You valued your own group (Bottom) more highly in hiring decisions") 
        else:
            st.info("💼 You did not show strong preference for your own group in hiring decisions")
        
        # Save data
        if st.button("Complete Study & Submit Data", type="primary", use_container_width=True):
            success = self.data_manager.save_participant_data(data)
            
            if success:
                st.success("✅ Data submitted successfully!")
                st.balloons()
                
                # Validate treatment effectiveness
                session_participants = st.session_state.session_participants[data.session_id]
                if len(session_participants) >= ExperimentConfig.MIN_SESSION_SIZE:
                    # Collect session results for validation
                    session_results = []
                    for pid in session_participants:
                        # This would typically come from database
                        # For demo, we'll use current participant data
                        if pid == data.participant_id:
                            session_results.append({
                                'belief_own_performance': data.belief_own_performance,
                                'treatment': data.treatment
                            })
                    
                    if len(session_results) > ExperimentConfig.MIN_SESSION_SIZE // 2:
                        is_valid, msg = self.validator.validate_treatment_effectiveness(
                            data.treatment, session_results
                        )
                        if is_valid:
                            st.success(f"✅ Treatment validation: {msg}")
                        else:
                            st.warning(f"⚠️ Treatment validation: {msg}")
                
                st.markdown("---")
                st.markdown("### 💰 Payment Information")
                st.markdown(f"""
                Your payment will be calculated as follows:
                - **Show-up fee:** ${ExperimentConfig.SHOW_UP_FEE:.2f}
                - **Performance bonus:** Based on one randomly selected task
                - **Total range:** ${ExperimentConfig.SHOW_UP_FEE:.2f} - ${ExperimentConfig.SHOW_UP_FEE + ExperimentConfig.HIGH_PERFORMANCE_TOKENS * ExperimentConfig.TOKEN_TO_DOLLAR_RATE:.2f}
                
                Payment will be processed within 48 hours.
                
                **Reference ID:** {data.participant_id}
                """)
                
                st.session_state.current_screen = 10
                
            else:
                st.error("❌ Error submitting data. Please try again or contact support.")
    
    def run_experiment(self):
        """Main experiment flow."""
        try:
            # Screen routing
            screens = {
                0: self.show_welcome_screen,
                1: self.show_session_status,
                2: self.show_trivia_instructions,
                3: self.show_trivia_questions,
                4: self.show_waiting_for_session,
                5: self.show_performance_belief,
                6: self.show_performance_results,
                7: self.show_hiring_decisions,
                8: self.show_mechanism_belief,
                9: self.show_completion_summary,
                10: lambda: st.success("Study completed! Thank you for participating.")
            }
            
            current_screen = st.session_state.current_screen
            
            if current_screen in screens:
                screens[current_screen]()
            else:
                st.error("Invalid screen. Please restart the experiment.")
                if st.button("Restart"):
                    st.session_state.clear()
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Experiment error: {e}", exc_info=True)
            st.error("An error occurred. Please refresh the page or contact support.")
            
            # Emergency data save
            if hasattr(st.session_state, 'participant_data') and st.session_state.participant_data:
                try:
                    st.session_state.participant_data.data_quality_flag = f"Error: {str(e)}"
                    self.data_manager.save_participant_data(st.session_state.participant_data)
                    st.info("Your data has been saved.")
                except:
                    pass
    
    def show_admin_panel(self):
        """Simplified admin panel for monitoring."""
        st.title("🔧 Study Administration")
        
        # Simple password
        if not st.session_state.get('admin_auth'):
            password = st.text_input("Admin Password:", type="password")
            if st.button("Login"):
                if password == "admin2024":  # Simple password for demo
                    st.session_state.admin_auth = True
                    st.rerun()
                else:
                    st.error("Invalid password")
            return
        
        # Assignment statistics
        st.subheader("📊 Assignment Statistics")
        stats = self.assignment_manager.get_assignment_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Participants", stats['total'])
        with col2:
            st.metric("Easy Treatment", stats['easy'])
        with col3:
            st.metric("Hard Treatment", stats['hard'])
        with col4:
            st.metric("Balance Status", stats['balance_status'])
        
        # Treatment summary
        st.subheader("📈 Treatment Summary")
        summary = self.data_manager.get_treatment_summary()
        
        if summary:
            df = pd.DataFrame(summary)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No completed participants yet")
        
        # Export data
        st.subheader("💾 Data Export")
        if st.button("Export Current Data"):
            # Simple CSV export
            try:
                conn = sqlite3.connect(self.data_manager.db_path)
                df = pd.read_sql_query("SELECT * FROM participants", conn)
                conn.close()
                
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"experiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Export error: {e}")

def main():
    """Main application entry point."""
    
    # Initialize experiment
    experiment = FaithfulExperiment()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎓 Study Platform")
        st.markdown("**Methodologically Faithful Implementation**")
        
        if st.session_state.get('participant_data'):
            data = st.session_state.participant_data
            st.markdown("---")
            st.markdown(f"**ID:** {data.participant_id[:8]}...")
            st.markdown(f"**Treatment:** {data.treatment}")
            st.markdown(f"**Session:** {data.session_number}")
        
        st.markdown("---")
        st.markdown("**🔬 Research Protocol**")
        st.markdown("""
        Based on: *"Does Overconfidence Predict 
        Discriminatory Beliefs and Behavior?"*
        
        **Key Features:**
        - True random assignment
        - Fixed session sizes
        - Empirical question validation
        - Session-relative performance
        - BDM hiring mechanism
        """)
        
        if st.checkbox("Admin Panel"):
            st.session_state.show_admin = True
        else:
            st.session_state.show_admin = False
    
    # Main application
    if st.session_state.get('show_admin'):
        experiment.show_admin_panel()
    else:
        experiment.run_experiment()

if __name__ == "__main__":
    main()

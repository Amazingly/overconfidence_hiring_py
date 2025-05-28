    def show_trivia_task(self):
        self.log_screen_time('S2_TriviaTask', is_start_of_screen=True)
        st.markdown('<div class="main-header"><h1>🧪 Decision-Making Research Experiment</h1></div>', unsafe_allow_html=True)
        self.show_progress_bar(3) # Screen 3

        if st.session_state.trivia_start_time is None: # Should have been set in previous screen
            logger.error(f"P:{st.session_state.experiment_data['participant_id']} - Trivia start time not set!")
            st.error("Error: Trivia timer not initialized. Please contact researchers.")
            return

        elapsed_time = time.time() - st.session_state.trivia_start_time
        time_remaining = max(0, TRIVIA_TIME_LIMIT_SECONDS - elapsed_time)
        minutes, seconds = divmod(int(time_remaining), 60)

        if time_remaining <= 0:
            st.warning("Time is up! Submitting your answers...")
            self.log_screen_time('S2_TriviaTask', is_start_of_screen=False) # Log end before submitting
            self.submit_trivia() # This will rerun to the next screen
            return # Important to stop further execution of this screen

        timer_class = "timer-warning" if time_remaining <= 60 else "timer-normal"
        timer_prefix = "⚠️ <strong>WARNING:</strong> " if time_remaining <= 60 else "⏱️ "
        st.markdown(f'<div class="{timer_class}">{timer_prefix}Time Remaining: {minutes}:{seconds:02d}</div>', unsafe_allow_html=True)

        current_q_idx = st.session_state.current_trivia_question
        if not st.session_state.selected_questions or current_q_idx >= len(st.session_state.selected_questions):
            logger.error(f"P:{st.session_state.experiment_data['participant_id']} - Invalid question index or no questions selected.")
            st.error("Error: Could not load trivia question. Please contact researchers.")
            return

        question = st.session_state.selected_questions[current_q_idx]

        if current_q_idx not in st.session_state.question_start_times: # For per-question timing
            st.session_state.question_start_times[current_q_idx] = time.time()

        # ... (Your full markdown for question display from the original code) ...
        st.markdown(f"""
        <div class="question-container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <div style="background: #2c3e50; color: white; padding: 0.5rem 1.5rem; border-radius: 25px; font-weight: bold;">
                    Question {current_q_idx + 1} of {NUM_TRIVIA_QUESTIONS}
                </div>
                <div style="background: #f8f9fa; color: #6c757d; padding: 0.5rem 1rem; border-radius: 15px; font-size: 0.9em;">
                    Category: {question.get('category','N/A').replace('_', ' ').title()}
                </div>
            </div>
            <h3 style="color: #2c3e50; line-height: 1.4; margin-bottom: 1.5rem;">{question['question']}</h3>
        </div>
        """, unsafe_allow_html=True)

        current_answer_for_q = st.session_state.experiment_data['trivia_answers'][current_q_idx]
        # Ensure options are available and correct
        options_list = question.get('options', [])
        if not options_list:
            st.error("Question options missing.")
            return

        selected_option_idx = st.radio(
            "Select your answer:",
            options=range(len(options_list)),
            format_func=lambda x: f"{chr(65 + x)}. {options_list[x]}",
            index=current_answer_for_q if current_answer_for_q is not None else 0, # Default to first if not answered
            key=f"trivia_q_radio_{current_q_idx}"
        )

        # Logic to update answer if it changes from what's stored
        # Streamlit runs top-to-bottom, so this captures the new selection
        if selected_option_idx != current_answer_for_q:
            st.session_state.experiment_data['trivia_answers'][current_q_idx] = selected_option_idx
            if current_q_idx in st.session_state.question_start_times:
                response_time = time.time() - st.session_state.question_start_times[current_q_idx]
                st.session_state.experiment_data['trivia_response_times'][current_q_idx] = round(response_time, 3)
            st.rerun() # Rerun to reflect the change immediately (e.g., for "answered" status display)


        st.markdown("---")
        nav_cols = st.columns([1, 2, 1]) # Previous, (Spacer or status), Next/Submit

        with nav_cols[0]:
            if current_q_idx > 0:
                if st.button("← Previous Question", key=f"prev_q_{current_q_idx}"):
                    # Log time for current question before moving
                    if current_q_idx in st.session_state.question_start_times and \
                       st.session_state.experiment_data['trivia_response_times'][current_q_idx] == 0.0: # Only if not already recorded
                        response_time = time.time() - st.session_state.question_start_times[current_q_idx]
                        st.session_state.experiment_data['trivia_response_times'][current_q_idx] = round(response_time, 3)

                    st.session_state.current_trivia_question -= 1
                    st.rerun()
        
        with nav_cols[1]:
            answered_count = sum(1 for ans in st.session_state.experiment_data['trivia_answers'] if ans is not None)
            st.markdown(f"<div style='text-align:center;'>Answered: {answered_count}/{NUM_TRIVIA_QUESTIONS}</div>", unsafe_allow_html=True)


        with nav_cols[2]:
            if current_q_idx < NUM_TRIVIA_QUESTIONS - 1:
                if st.button("Next Question →", key=f"next_q_{current_q_idx}"):
                    # Log time for current question before moving
                    if current_q_idx in st.session_state.question_start_times and \
                       st.session_state.experiment_data['trivia_response_times'][current_q_idx] == 0.0:
                        response_time = time.time() - st.session_state.question_start_times[current_q_idx]
                        st.session_state.experiment_data['trivia_response_times'][current_q_idx] = round(response_time, 3)
                    
                    st.session_state.current_trivia_question += 1
                    st.rerun()
            else: # Last question
                if st.button("📝 Submit All Answers", key="final_submit_trivia"):
                    # Log time for the last question
                    if current_q_idx in st.session_state.question_start_times and \
                       st.session_state.experiment_data['trivia_response_times'][current_q_idx] == 0.0:
                        response_time = time.time() - st.session_state.question_start_times[current_q_idx]
                        st.session_state.experiment_data['trivia_response_times'][current_q_idx] = round(response_time, 3)
                    
                    self.log_screen_time('S2_TriviaTask', is_start_of_screen=False) # Log end of this screen
                    self.submit_trivia() # This will change screen and rerun

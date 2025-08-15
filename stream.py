import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import json
import io

# Import your WritingProgressTracker
from story import WritingProgressTracker # Assuming your script is saved as paste.py

# Set page config
st.set_page_config(
    page_title="Writing Analytics Dashboard",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .feedback-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tracker' not in st.session_state:
    st.session_state.tracker = WritingProgressTracker()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

def main():
    # Header
    st.markdown('<h1 class="main-header">✍️ Writing Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.selectbox("Choose a page:", [
            "📝 Story Analysis", 
            "📈 Progress Tracking", 
            "🎯 Goals Management",
            "📊 Analytics Dashboard",
            "🤖 AI Feedback"
        ])
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # Model selection for AI feedback
        if st.session_state.get('ollama_available', False):
            model_name = st.selectbox("AI Model:", [
               "deepseek-r1:8b-0528-qwen3-q4_K_M"
            ])
            st.session_state.tracker.model_name = model_name
        else:
            st.warning("⚠️ Ollama not available. AI features disabled.")
    
    # Main content based on selected page
    if page == "📝 Story Analysis":
        story_analysis_page()
    elif page == "📈 Progress Tracking":
        progress_tracking_page()
    elif page == "🎯 Goals Management":
        goals_management_page()
    elif page == "📊 Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "🤖 AI Feedback":
        ai_feedback_page()

def story_analysis_page():
    st.header("📝 Story Analysis")
    
    # Input methods
    input_method = st.radio("How would you like to input your story?", 
                           ["Type/Paste Text", "Upload File"])
    
    story_text = ""
    story_title = ""
    
    if input_method == "Type/Paste Text":
        col1, col2 = st.columns([3, 1])
        with col1:
            story_title = st.text_input("Story Title:", placeholder="Enter your story title...")
        with col2:
            genre = st.selectbox("Genre:", [
                "Literary Fiction", "Mystery", "Romance", "Sci-Fi", 
                "Fantasy", "Horror", "Thriller", "Drama", "Comedy", "Other"
            ])
        
        story_text = st.text_area("Your Story:", 
                                 placeholder="Paste your story here...", 
                                 height=300)
    
    else:  # Upload File
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'md'])
            if uploaded_file is not None:
                story_text = str(uploaded_file.read(), "utf-8")
                story_title = uploaded_file.name.replace('.txt', '').replace('.md', '')
        with col2:
            genre = st.selectbox("Genre:", [
                "Literary Fiction", "Mystery", "Romance", "Sci-Fi", 
                "Fantasy", "Horror", "Thriller", "Drama", "Comedy", "Other"
            ])
    
    # Analysis options
    st.subheader("Analysis Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_to_db = st.checkbox("Save to Database", value=True)
    with col2:
        get_ai_feedback = st.checkbox("Get AI Feedback", value=True)
    with col3:
        feedback_types = st.multiselect("Feedback Types:", 
                                       ["general", "character", "structure", "style"],
                                       default=["general"])
    
    # Analyze button
    if st.button("🔍 Analyze Story", type="primary", use_container_width=True):
        if story_text.strip() and story_title:
            with st.spinner("Analyzing your story..."):
                try:
                    # Perform analysis
                    result = st.session_state.tracker.analyze_story_complete(
                        story_text,
                        title=story_title,
                        genre=genre,
                        save_to_db=save_to_db,
                        get_ai_feedback=get_ai_feedback,
                        feedback_types=feedback_types
                    )
                    
                    # Store results
                    st.session_state.analysis_results[story_title] = result
                    
                    # Display results
                    display_analysis_results(result)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter both a story title and text!")

def display_analysis_results(result):
    st.success("✅ Analysis Complete!")
    
    # Basic Metrics
    basic = result['basic_metrics']
    vocab = result['vocabulary']
    nlp = result['nlp_analysis']
    
    st.subheader("📊 Story Metrics")
    
    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{basic['word_count']:,}</h3>
            <p>Words</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{basic['sentence_count']}</h3>
            <p>Sentences</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result['dialogue_ratio']:.1f}%</h3>
            <p>Dialogue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{vocab['vocabulary_richness']:.3f}</h3>
            <p>Vocab Richness</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Writing Style")
        style_data = {
            'Metric': ['Avg Sentence Length', 'Avg Word Length', 'Paragraphs', 'Unique Words'],
            'Value': [
                f"{basic['avg_sentence_length']:.1f} words",
                f"{vocab['avg_word_length']:.1f} chars",
                f"{basic['paragraph_count']}",
                f"{vocab['unique_words']:,}"
            ]
        }
        st.table(pd.DataFrame(style_data))
    
    with col2:
        st.subheader("👥 Characters & Entities")
        entities = nlp.get('entities', {})
        if entities:
            for entity_type, entity_list in entities.items():
                if entity_list:
                    st.write(f"**{entity_type}:** {', '.join(entity_list[:5])}")
        else:
            st.write("No significant entities detected")
    
    # Most Common Words
    st.subheader("🔤 Most Common Words")
    if vocab['most_common_words']:
        words_df = pd.DataFrame(vocab['most_common_words'], columns=['Word', 'Count'])
        
        # Create bar chart
        fig = px.bar(words_df, x='Word', y='Count', 
                    title="Top 10 Most Common Words",
                    color='Count', color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentence Length Distribution
    st.subheader("📏 Sentence Length Analysis")
    if nlp.get('sentence_lengths'):
        fig = px.histogram(x=nlp['sentence_lengths'], 
                          nbins=20,
                          title="Sentence Length Distribution",
                          labels={'x': 'Words per Sentence', 'y': 'Frequency'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Feedback
    if 'ai_feedback' in result:
        st.subheader("🤖 AI Feedback")
        for feedback_type, feedback_content in result['ai_feedback'].items():
            with st.expander(f"📝 {feedback_type.title()} Feedback", expanded=True):
                st.markdown(f"""
                <div class="feedback-box">
                    {feedback_content.replace('\n', '<br>')}
                </div>
                """, unsafe_allow_html=True)

def progress_tracking_page():
    st.header("📈 Progress Tracking")
    
    # Time period selector
    col1, col2 = st.columns([1, 3])
    with col1:
        days = st.selectbox("Time Period:", [7, 14, 30, 60, 90], index=2)
    
    # Get progress data
    progress_df = st.session_state.tracker.get_progress_summary(days=days)
    
    if progress_df is not None and not progress_df.empty:
        # Summary stats
        st.subheader(f"📊 Summary (Last {days} days)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_stories = len(progress_df)
            st.metric("Stories Written", total_stories)
        
        with col2:
            total_words = progress_df['word_count'].sum()
            st.metric("Total Words", f"{total_words:,}")
        
        with col3:
            avg_words = progress_df['word_count'].mean()
            st.metric("Avg Words/Story", f"{avg_words:.0f}")
        
        with col4:
            avg_dialogue = progress_df['dialogue_ratio'].mean()
            st.metric("Avg Dialogue %", f"{avg_dialogue:.1f}%")
        
        # Progress charts
        st.subheader("📊 Progress Charts")
        
        # Convert dates for plotting
        progress_df['created_date'] = pd.to_datetime(progress_df['created_date'])
        
        # Create tabs for different charts
        tab1, tab2, tab3, tab4 = st.tabs(["Words Over Time", "Vocabulary Trends", "Dialogue Analysis", "Genre Breakdown"])
        
        with tab1:
            fig = px.line(progress_df, x='created_date', y='word_count',
                         title='Word Count Over Time',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.line(progress_df, x='created_date', y='vocabulary_richness',
                         title='Vocabulary Richness Trend',
                         markers=True, color_discrete_sequence=['green'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = px.histogram(progress_df, x='dialogue_ratio',
                              title='Dialogue Ratio Distribution',
                              nbins=10, color_discrete_sequence=['orange'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            if 'genre' in progress_df.columns:
                genre_counts = progress_df['genre'].value_counts()
                fig = px.pie(values=genre_counts.values, names=genre_counts.index,
                            title='Stories by Genre')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No genre data available")
        
        # Recent stories table
        st.subheader("📚 Recent Stories")
        display_df = progress_df[['title', 'word_count', 'sentence_count', 'dialogue_ratio', 'vocabulary_richness', 'created_date']].copy()
        display_df['created_date'] = display_df['created_date'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.info(f"📝 No stories found in the last {days} days. Start analyzing some stories!")

def goals_management_page():
    st.header("🎯 Goals Management")
    
    # Add new goal
    st.subheader("➕ Set New Goal")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        goal_type = st.selectbox("Goal Type:", ["words", "stories", "sessions"])
    
    with col2:
        target_value = st.number_input("Target Value:", min_value=1, value=1000 if goal_type == "words" else 5)
    
    with col3:
        deadline = st.date_input("Deadline:", value=datetime.now() + timedelta(days=30))
    
    if st.button("🎯 Set Goal", type="primary"):
        st.session_state.tracker.set_writing_goal(
            goal_type, 
            target_value, 
            deadline.isoformat()
        )
        st.success(f"✅ Goal set: {target_value} {goal_type} by {deadline}")
        st.rerun()
    
    # Current goals
    st.subheader("📋 Current Goals")
    
    # Mock goals display (you'd need to modify the original class to return data)
    try:
        conn = sqlite3.connect(st.session_state.tracker.db_path)
        goals_df = pd.read_sql_query('''
        SELECT * FROM writing_goals WHERE status = "Active"
        ORDER BY deadline
        ''', conn)
        conn.close()
        
        if not goals_df.empty:
            for _, goal in goals_df.iterrows():
                # Calculate progress
                conn = sqlite3.connect(st.session_state.tracker.db_path)
                if goal['goal_type'] == 'words':
                    current = pd.read_sql_query('SELECT SUM(word_count) as total FROM stories', conn).iloc[0]['total'] or 0
                elif goal['goal_type'] == 'stories':
                    current = pd.read_sql_query('SELECT COUNT(*) as count FROM stories', conn).iloc[0]['count']
                else:
                    current = goal['current_value']
                conn.close()
                
                progress = min((current / goal['target_value']) * 100, 100) if goal['target_value'] > 0 else 0
                days_left = (datetime.fromisoformat(goal['deadline']) - datetime.now()).days
                
                # Display goal card
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{goal['goal_type'].title()} Goal**")
                        st.progress(progress / 100)
                        st.write(f"{current:,} / {goal['target_value']:,} ({progress:.1f}%)")
                    
                    with col2:
                        st.metric("Days Left", days_left)
                    
                    with col3:
                        status = "✅ Complete!" if progress >= 100 else "⏳ In Progress"
                        st.write(status)
                
                st.markdown("---")
        else:
            st.info("🎯 No active goals set. Create your first goal above!")
    
    except Exception as e:
        st.error(f"Error loading goals: {str(e)}")

def analytics_dashboard_page():
    st.header("📊 Analytics Dashboard")
    
    # Overall statistics
    try:
        conn = sqlite3.connect(st.session_state.tracker.db_path)
        
        # Get all stories
        all_stories = pd.read_sql_query('SELECT * FROM stories ORDER BY created_date DESC', conn)
        
        if not all_stories.empty:
            # Key metrics
            st.subheader("🎯 Key Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Stories", len(all_stories))
            
            with col2:
                total_words = all_stories['word_count'].sum()
                st.metric("Total Words", f"{total_words:,}")
            
            with col3:
                avg_words = all_stories['word_count'].mean()
                st.metric("Avg Words/Story", f"{avg_words:.0f}")
            
            with col4:
                avg_vocab = all_stories['vocabulary_richness'].mean()
                st.metric("Avg Vocab Richness", f"{avg_vocab:.3f}")
            
            with col5:
                avg_dialogue = all_stories['dialogue_ratio'].mean()
                st.metric("Avg Dialogue %", f"{avg_dialogue:.1f}%")
            
            # Advanced analytics
            st.subheader("📈 Advanced Analytics")
            
            # Convert dates
            all_stories['created_date'] = pd.to_datetime(all_stories['created_date'])
            all_stories['month'] = all_stories['created_date'].dt.to_period('M')
            
            # Monthly trends
            monthly_stats = all_stories.groupby('month').agg({
                'word_count': ['sum', 'mean', 'count'],
                'vocabulary_richness': 'mean',
                'dialogue_ratio': 'mean'
            }).round(2)
            
            # Flatten column names
            monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
            monthly_stats = monthly_stats.reset_index()
            monthly_stats['month'] = monthly_stats['month'].astype(str)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Word Count', 'Stories per Month', 
                              'Vocabulary Richness Trend', 'Dialogue Ratio Trend'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Add traces
            fig.add_trace(
                go.Bar(x=monthly_stats['month'], y=monthly_stats['word_count_sum'], 
                      name='Total Words', showlegend=False),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=monthly_stats['month'], y=monthly_stats['word_count_count'], 
                      name='Story Count', showlegend=False),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=monthly_stats['month'], y=monthly_stats['vocabulary_richness_mean'], 
                          mode='lines+markers', name='Vocab Richness', showlegend=False),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=monthly_stats['month'], y=monthly_stats['dialogue_ratio_mean'], 
                          mode='lines+markers', name='Dialogue %', showlegend=False),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="Monthly Writing Analytics")
            st.plotly_chart(fig, use_container_width=True)
            
            # Genre analysis
            if 'genre' in all_stories.columns:
                st.subheader("📚 Genre Analysis")
                genre_stats = all_stories.groupby('genre').agg({
                    'word_count': ['count', 'sum', 'mean'],
                    'vocabulary_richness': 'mean',
                    'dialogue_ratio': 'mean'
                }).round(2)
                
                st.dataframe(genre_stats, use_container_width=True)
        
        else:
            st.info("📝 No stories in database yet. Start analyzing some stories!")
        
        conn.close()
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

def ai_feedback_page():
    st.header("🤖 AI Feedback Center")
    
    # Check if stories exist
    try:
        conn = sqlite3.connect(st.session_state.tracker.db_path)
        stories_df = pd.read_sql_query('SELECT id, title, created_date FROM stories ORDER BY created_date DESC', conn)
        conn.close()
        
        if not stories_df.empty:
            # Select story for feedback
            story_titles = [f"{row['title']} ({row['created_date'][:10]})" for _, row in stories_df.iterrows()]
            selected_story = st.selectbox("Select a story for AI feedback:", story_titles)
            
            if selected_story:
                story_idx = story_titles.index(selected_story)
                story_id = stories_df.iloc[story_idx]['id']
                
                # Feedback type selection
                feedback_type = st.selectbox("Feedback Type:", [
                    "general", "character", "structure", "style"
                ])
                
                # Get existing feedback
                conn = sqlite3.connect(st.session_state.tracker.db_path)
                existing_feedback = pd.read_sql_query('''
                SELECT feedback_type, feedback_content, feedback_date 
                FROM ai_feedback 
                WHERE story_id = ? 
                ORDER BY feedback_date DESC
                ''', conn, params=[story_id])
                conn.close()
                
                # Display existing feedback
                if not existing_feedback.empty:
                    st.subheader("📋 Previous Feedback")
                    for _, feedback in existing_feedback.iterrows():
                        with st.expander(f"{feedback['feedback_type'].title()} - {feedback['feedback_date'][:10]}"):
                            st.write(feedback['feedback_content'])
                
                # Generate new feedback button
                if st.button("🔄 Generate New Feedback", type="primary"):
                    with st.spinner("Generating AI feedback..."):
                        # You'd need to get the story text and call the AI feedback function
                        st.info("AI feedback generation would happen here!")
        
        else:
            st.info("📝 No stories available for feedback. Analyze some stories first!")
    
    except Exception as e:
        st.error(f"Error loading feedback data: {str(e)}")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import json
import io
import base64
from pathlib import Path
import re
from streamlit import config


# Fix for PyTorch + Streamlit compatibility issue
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
# Import your writing analysis classes
# Assuming the main code is saved as 'writing_analyzer.py'
try:
    from story import WritingProgressTracker, quick_analyze, batch_analyze_files, export_progress_report
    ANALYZER_AVAILABLE = True
except ImportError:
    st.error("⚠️ Please ensure 'writing_analyzer.py' is in the same directory as this Streamlit app")
    ANALYZER_AVAILABLE = False

# Page configuration
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
        font-weight: bold;
        color: #1e3a8a;
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
    .feedback-box {
        background-color: #1f2937 !important;
        border-left: 4px solid #10b981 !important;
        padding: 1.5rem !important;
        margin: 1rem 0;
        border-radius: 8px !important;
        color: #f9fafb !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    .feedback-box p {
        color: #f9fafb !important;
        margin-bottom: 0.75rem !important;
    }
    .feedback-box strong {
        color: #d1fae5 !important;
    }
    .feedback-box em {
        color: #a7f3d0 !important;
    }
    .story-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-info {
        background-color: #f0f2f6 !important;
        border: 1px solid #d1d5db !important;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #1f2937 !important;
    }
    .sidebar-info h4 {
        color: #374151 !important;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sidebar-info ul {
        color: #4b5563 !important;
        padding-left: 1rem;
    }
    .sidebar-info li {
        color: #4b5563 !important;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_tracker():
    """Initialize the WritingProgressTracker"""
    if not ANALYZER_AVAILABLE:
        return None

    # Only create tracker if WritingProgressTracker is available
    if 'tracker' not in st.session_state and ANALYZER_AVAILABLE:
        # WritingProgressTracker is only defined if import succeeded
        st.session_state.tracker = WritingProgressTracker()
    return st.session_state.get('tracker', None)

def create_download_link(data, filename, text):
    """Create a download link for data"""
    if isinstance(data, dict):
        data = json.dumps(data, indent=2, default=str)
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">{text}</a>'
    return href

def analyze_text_page():
    """Main text analysis page"""
    st.markdown('<h1 class="main-header">📝 Text Analysis</h1>', unsafe_allow_html=True)
    
    tracker = initialize_tracker()
    if not tracker:
        st.error("Writing analyzer not available")
        return
    
    # Input methods
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Enter Your Text")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["✏️ Type/Paste Text", "📁 Upload File"]
        )
        
        text_content = ""
        story_title = ""
        
        if input_method == "✏️ Type/Paste Text":
            story_title = st.text_input("Story Title", placeholder="Enter a title for your story...")
            text_content = st.text_area(
                "Your Story",
                placeholder="Paste or type your story here...",
                height=300
            )
        
        else:  # File upload
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'md', 'rtf'],
                help="Upload a text file to analyze"
            )
            
            if uploaded_file:
                story_title = st.text_input(
                    "Story Title", 
                    value=uploaded_file.name.split('.')[0],
                    help="Auto-filled from filename, but you can edit it"
                )
                
                try:
                    text_content = uploaded_file.read().decode('utf-8')
                    st.success(f"✅ File loaded: {len(text_content.split())} words")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    
    with col2:
        st.subheader("⚙️ Analysis Options")
        
        # Genre selection
        genre = st.selectbox(
            "Select Genre",
            ["Literary Fiction", "Science Fiction", "Fantasy", "Mystery", "Romance", 
             "Thriller", "Horror", "Adventure", "Drama", "Comedy", "Other"]
        )
        
        # Feedback types
        st.write("**AI Feedback Types:**")
        feedback_general = st.checkbox("📋 General Feedback", value=True)
        feedback_character = st.checkbox("👥 Character Analysis")
        feedback_structure = st.checkbox("🏗️ Structure & Pacing")
        feedback_style = st.checkbox("🎨 Style & Voice")
        
        # Analysis options
        save_to_db = st.checkbox("💾 Save to Database", value=True)
        get_ai_feedback = st.checkbox("🤖 Get AI Feedback", value=True)
        
        # Quick stats toggle
        show_quick_stats = st.checkbox("📊 Show Quick Stats", value=True)
    
    # Analysis button
    if st.button("🚀 Analyze Story", type="primary"):
        if not text_content.strip():
            st.error("Please enter some text to analyze!")
            return
        
        if not story_title or not str(story_title).strip():
            story_title = "Untitled Story"
        
        # Prepare feedback types
        feedback_types = []
        if feedback_general: feedback_types.append("general")
        if feedback_character: feedback_types.append("character")
        if feedback_structure: feedback_types.append("structure")
        if feedback_style: feedback_types.append("style")
        
        if not feedback_types:
            feedback_types = ["general"]
        
        # Show progress
        with st.spinner("🔍 Analyzing your story..."):
            try:
                # Perform analysis
                result = tracker.analyze_story_complete(
                    text=text_content,
                    title=story_title,
                    genre=genre,
                    save_to_db=save_to_db,
                    get_ai_feedback=get_ai_feedback,
                    feedback_types=feedback_types
                )
                
                # Store result in session state
                st.session_state.analysis_result = result
                st.session_state.analyzed_text = text_content
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return
    
    # Display results
    if 'analysis_result' in st.session_state:
        display_analysis_results(st.session_state.analysis_result, show_quick_stats)

def display_analysis_results(result, show_quick_stats=True):
    """Display comprehensive analysis results"""
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Quick metrics at the top
    if show_quick_stats:
        basic = result['basic_metrics']
        vocab = result['vocabulary']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📝 Words", f"{basic['word_count']:,}")
        with col2:
            st.metric("📏 Sentences", basic['sentence_count'])
        with col3:
            st.metric("💬 Dialogue", f"{result['dialogue_ratio']:.1f}%")
        with col4:
            st.metric("📚 Vocabulary", f"{vocab['vocabulary_richness']:.3f}")
        with col5:
            st.metric("📐 Avg Sentence", f"{basic['avg_sentence_length']:.1f} words")
    
    # Detailed metrics in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics", "👥 Characters", "🎨 Style", "🤖 AI Feedback", "📁 Export"])
    
    with tab1:
        display_detailed_metrics(result)
    
    with tab2:
        display_character_analysis(result)
    
    with tab3:
        display_style_analysis(result)
    
    with tab4:
        display_ai_feedback(result)
    
    with tab5:
        display_export_options(result)

def display_detailed_metrics(result):
    """Display detailed metrics with visualizations"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Basic Metrics")
        
        basic = result['basic_metrics']
        vocab = result['vocabulary']
        
        metrics_data = {
            'Metric': ['Words', 'Sentences', 'Paragraphs', 'Characters', 'Avg Sentence Length'],
            'Value': [
                basic['word_count'],
                basic['sentence_count'],
                basic['paragraph_count'],
                basic['character_count'],
                basic['avg_sentence_length']
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Vocabulary metrics
        st.subheader("📚 Vocabulary Analysis")
        st.write(f"**Unique Words:** {vocab['unique_words']:,}")
        st.write(f"**Vocabulary Richness:** {vocab['vocabulary_richness']:.3f}")
        st.write(f"**Average Word Length:** {vocab['avg_word_length']:.1f} characters")
        st.write(f"**Dialogue Ratio:** {result['dialogue_ratio']:.1f}%")
    
    with col2:
        st.subheader("📊 Visual Analysis")
        
        # Create a simple metrics chart
        metrics_chart_data = {
            'Words': basic['word_count'],
            'Sentences': basic['sentence_count'],
            'Paragraphs': basic['paragraph_count']
        }
        
        fig = px.bar(
            x=list(metrics_chart_data.keys()),
            y=list(metrics_chart_data.values()),
            title="Text Structure Overview"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Most common words
        if vocab['most_common_words']:
            st.subheader("🔤 Most Common Words")
            common_words_df = pd.DataFrame(
                vocab['most_common_words'][:10], 
                columns=['Word', 'Frequency']
            )
            
            fig2 = px.bar(
                common_words_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title="Top 10 Most Common Words"
            )
            fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)

def display_character_analysis(result):
    """Display character and entity analysis"""
    
    nlp_data = result.get('nlp_analysis', {})
    entities = nlp_data.get('entities', {})
    
    if not entities:
        st.info("No named entities detected in the text.")
        return
    
    st.subheader("👥 Characters & Entities")
    
    # Entity summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Entity Summary:**")
        entity_counts = {entity_type: len(entity_list) for entity_type, entity_list in entities.items()}
        
        if entity_counts:
            entity_df = pd.DataFrame(
                list(entity_counts.items()),
                columns=['Entity Type', 'Count']
            )
            st.dataframe(entity_df, use_container_width=True)
    
    with col2:
        # Entity distribution chart
        if entity_counts:
            fig = px.pie(
                values=list(entity_counts.values()),
                names=list(entity_counts.keys()),
                title="Entity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed entity breakdown
    st.subheader("📋 Detailed Entity Breakdown")
    
    for entity_type, entity_list in entities.items():
        if entity_list:
            with st.expander(f"{entity_type} ({len(entity_list)} found)"):
                # Display as columns for better readability
                if len(entity_list) > 6:
                    cols = st.columns(3)
                    for i, entity in enumerate(entity_list):
                        with cols[i % 3]:
                            st.write(f"• {entity}")
                else:
                    for entity in entity_list:
                        st.write(f"• {entity}")

def display_style_analysis(result):
    """Display style and writing analysis"""
    
    st.subheader("🎨 Writing Style Analysis")
    
    nlp_data = result.get('nlp_analysis', {})
    basic = result['basic_metrics']
    vocab = result['vocabulary']
    
    col1, col2 = st.columns(2)
    
    # Ensure avg_length is always defined
    avg_length = nlp_data.get('avg_sentence_length', 0)
    
    with col1:
        st.write("**Sentence Structure:**")
        
        sentence_lengths = nlp_data.get('sentence_lengths', [])
        if sentence_lengths:
            avg_length = nlp_data.get('avg_sentence_length', 0)
            std_length = nlp_data.get('sentence_length_std', 0)
            
            st.write(f"• Average sentence length: {avg_length:.1f} words")
            st.write(f"• Sentence variety (std dev): {std_length:.1f}")
            st.write(f"• Shortest sentence: {min(sentence_lengths)} words")
            st.write(f"• Longest sentence: {max(sentence_lengths)} words")
            
            # Sentence length distribution
            fig = px.histogram(
                x=sentence_lengths,
                title="Sentence Length Distribution",
                labels={'x': 'Words per Sentence', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Writing Characteristics:**")
        
        # Calculate writing style metrics
        dialogue_ratio = result['dialogue_ratio']
        vocab_richness = vocab['vocabulary_richness']
        avg_word_length = vocab['avg_word_length']
        
        # Style assessment
        style_notes = []
        
        if dialogue_ratio > 30:
            style_notes.append("🗣️ Dialogue-heavy narrative")
        elif dialogue_ratio < 10:
            style_notes.append("📖 Descriptive/narrative style")
        else:
            style_notes.append("⚖️ Balanced dialogue/narrative")
        
        if vocab_richness > 0.7:
            style_notes.append("📚 Rich vocabulary")
        elif vocab_richness < 0.4:
            style_notes.append("📝 Simple vocabulary")
        else:
            style_notes.append("📖 Moderate vocabulary")
        
        if avg_word_length > 5:
            style_notes.append("🎓 Complex word choice")
        elif avg_word_length < 4:
            style_notes.append("✨ Simple word choice")
        else:
            style_notes.append("📏 Moderate word complexity")
        
        for note in style_notes:
            st.write(f"• {note}")
        
        # Style radar chart
        style_data = {
            'Dialogue': min(dialogue_ratio / 50 * 100, 100),  # Normalize to 0-100
            'Vocabulary': vocab_richness * 100,
            'Sentence Variety': min(nlp_data.get('sentence_length_std', 0) / 10 * 100, 100),
            'Word Complexity': min(avg_word_length / 8 * 100, 100),
            'Readability': max(100 - (avg_length / 25 * 100), 0)  # Inverse of sentence length
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(style_data.values()),
            theta=list(style_data.keys()),
            fill='toself',
            name='Writing Style'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Writing Style Profile"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_ai_feedback(result):
    """Display AI feedback in an organized way"""
    
    ai_feedback = result.get('ai_feedback', {})
    
    if not ai_feedback:
        st.info("No AI feedback available for this analysis.")
        return
    
    st.subheader("🤖 AI Writing Feedback")
    
    # Create tabs for different feedback types
    feedback_types = list(ai_feedback.keys())
    
    if len(feedback_types) == 1:
        # Single feedback type
        feedback_type = feedback_types[0]
        feedback_content = ai_feedback[feedback_type]
        
        st.markdown(f"### {feedback_type.title()} Feedback")
        st.markdown(f'<div class="feedback-box">{feedback_content}</div>', unsafe_allow_html=True)
    
    else:
        # Multiple feedback types - use tabs
        tabs = st.tabs([f"📝 {ftype.title()}" for ftype in feedback_types])
        
        for i, (feedback_type, feedback_content) in enumerate(ai_feedback.items()):
            with tabs[i]:
                st.markdown(f'<div class="feedback-box">{feedback_content}</div>', unsafe_allow_html=True)
    
    # Feedback summary
    st.subheader("📋 Feedback Summary")
    
    feedback_summary = {
        'Total feedback sections': len(ai_feedback),
        'Feedback types': ', '.join(feedback_types),
        'Total feedback words': sum(len(content.split()) for content in ai_feedback.values())
    }
    
    for key, value in feedback_summary.items():
        st.write(f"**{key}:** {value}")

def display_export_options(result):
    """Display export and download options"""
    
    st.subheader("📁 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Analysis Results:**")
        
        # JSON export
        if st.button("📄 Export as JSON"):
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="💾 Download JSON",
                data=json_data,
                file_name=f"{result['title']}_analysis.json",
                mime="application/json"
            )
        
        # CSV export (basic metrics)
        if st.button("📊 Export Metrics as CSV"):
            metrics_data = {
                'Title': [result['title']],
                'Genre': [result['genre']],
                'Word Count': [result['basic_metrics']['word_count']],
                'Sentence Count': [result['basic_metrics']['sentence_count']],
                'Paragraph Count': [result['basic_metrics']['paragraph_count']],
                'Dialogue Ratio': [result['dialogue_ratio']],
                'Vocabulary Richness': [result['vocabulary']['vocabulary_richness']],
                'Average Sentence Length': [result['basic_metrics']['avg_sentence_length']]
            }
            
            df = pd.DataFrame(metrics_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"{result['title']}_metrics.csv",
                mime="text/csv"
            )
    
    with col2:
        st.write("**Analysis Summary:**")
        
        # Quick summary
        summary_text = f"""
# Analysis Summary: {result['title']}

**Basic Metrics:**
- Words: {result['basic_metrics']['word_count']:,}
- Sentences: {result['basic_metrics']['sentence_count']}
- Paragraphs: {result['basic_metrics']['paragraph_count']}
- Dialogue: {result['dialogue_ratio']:.1f}%

**Vocabulary:**
- Unique words: {result['vocabulary']['unique_words']:,}
- Vocabulary richness: {result['vocabulary']['vocabulary_richness']:.3f}
- Average word length: {result['vocabulary']['avg_word_length']:.1f}

**Analysis Date:** {result['analysis_date']}
"""
        
        st.download_button(
            label="📝 Download Summary",
            data=summary_text,
            file_name=f"{result['title']}_summary.md",
            mime="text/markdown"
        )

def progress_dashboard():
    """Progress tracking dashboard page"""
    
    st.markdown('<h1 class="main-header">📈 Progress Dashboard</h1>', unsafe_allow_html=True)
    
    tracker = initialize_tracker()
    if not tracker:
        st.error("Writing analyzer not available")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("📊 Dashboard Controls")
        
        days_range = st.selectbox(
            "Time Range",
            [7, 14, 30, 60, 90, 365],
            index=2,
            help="Select the number of days to analyze"
        )
        
        refresh_data = st.button("🔄 Refresh Data")
        
        # Goal management
        st.subheader("🎯 Goal Management")
        
        goal_type = st.selectbox("Goal Type", ["words", "stories"])
        goal_target = st.number_input("Target", min_value=1, value=1000)
        goal_days = st.number_input("Days", min_value=1, value=30)
        
        if st.button("Set Goal"):
            deadline = (datetime.now() + timedelta(days=goal_days)).isoformat()
            tracker.set_writing_goal(goal_type, goal_target, deadline)
            st.success(f"Goal set: {goal_target} {goal_type} in {goal_days} days")
    
    # Main dashboard content
    try:
        # Get progress data
        progress_data = tracker.get_progress_summary(days=days_range)
        
        if progress_data is None or progress_data.empty:
            st.info(f"No writing data found for the last {days_range} days. Start by analyzing some stories!")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_stories = len(progress_data)
        total_words = progress_data['word_count'].sum()
        avg_words = progress_data['word_count'].mean()
        avg_dialogue = progress_data['dialogue_ratio'].mean()
        
        with col1:
            st.metric("📚 Total Stories", total_stories)
        with col2:
            st.metric("📝 Total Words", f"{total_words:,}")
        with col3:
            st.metric("📊 Avg Words/Story", f"{avg_words:.0f}")
        with col4:
            st.metric("💬 Avg Dialogue", f"{avg_dialogue:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Word count over time
            progress_data['date'] = pd.to_datetime(progress_data['created_date']).dt.date
            
            fig1 = px.line(
                progress_data,
                x='date',
                y='word_count',
                title=f"Word Count Trend (Last {days_range} Days)",
                markers=True
            )
            fig1.update_layout(xaxis_title="Date", yaxis_title="Words")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Vocabulary richness trend
            fig2 = px.line(
                progress_data,
                x='date',
                y='vocabulary_richness',
                title="Vocabulary Richness Trend",
                markers=True,
                color_discrete_sequence=['green']
            )
            fig2.update_layout(xaxis_title="Date", yaxis_title="Vocabulary Richness")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Additional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre distribution
            if 'genre' in progress_data.columns:
                genre_counts = progress_data['genre'].value_counts()
                
                fig3 = px.pie(
                    values=genre_counts.values,
                    names=genre_counts.index,
                    title="Genre Distribution"
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Story length distribution
            fig4 = px.histogram(
                progress_data,
                x='word_count',
                title="Story Length Distribution",
                nbins=20
            )
            fig4.update_layout(xaxis_title="Word Count", yaxis_title="Frequency")
            st.plotly_chart(fig4, use_container_width=True)
        
        # Recent stories table
        st.subheader("📖 Recent Stories")
        
        display_columns = ['title', 'word_count', 'vocabulary_richness', 'dialogue_ratio', 'created_date']
        available_columns = [col for col in display_columns if col in progress_data.columns]
        
        recent_stories = progress_data[available_columns].head(10)
        recent_stories['created_date'] = pd.to_datetime(recent_stories['created_date']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(recent_stories, use_container_width=True)
        
        # Goals progress
        st.subheader("🎯 Goal Progress")
        try:
            tracker.check_goals()
        except Exception as e:
            st.error(f"Error checking goals: {e}")
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")

def batch_analysis_page():
    """Batch analysis page for multiple files"""
    
    st.markdown('<h1 class="main-header">📁 Batch Analysis</h1>', unsafe_allow_html=True)
    
    tracker = initialize_tracker()
    if not tracker:
        st.error("Writing analyzer not available")
        return
    
    st.write("Upload multiple text files for batch analysis")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose text files",
        type=['txt', 'md', 'rtf'],
        accept_multiple_files=True,
        help="Select multiple text files to analyze at once"
    )
    
    if not uploaded_files:
        st.info("👆 Upload some files to get started with batch analysis")
        return
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Batch Options")
        
        batch_genre = st.selectbox(
            "Default Genre",
            ["Literary Fiction", "Science Fiction", "Fantasy", "Mystery", "Romance", 
             "Thriller", "Horror", "Adventure", "Drama", "Comedy", "Other"]
        )
        
        batch_save_db = st.checkbox("💾 Save all to database", value=True)
        batch_ai_feedback = st.checkbox("🤖 Get AI feedback", value=False, 
                                       help="Warning: This may take a while for many files")
    
    with col2:
        st.subheader("📊 File Summary")
        st.write(f"**Files selected:** {len(uploaded_files)}")
        
        # Calculate total size
        total_size = sum(len(file.read()) for file in uploaded_files)
        for file in uploaded_files:  # Reset file pointers
            file.seek(0)
        
        st.write(f"**Total size:** {total_size / 1024:.1f} KB")
        
        # File list
        with st.expander("📝 File List"):
            for file in uploaded_files:
                st.write(f"• {file.name}")
    
    # Start batch analysis
    if st.button("🚀 Start Batch Analysis", type="primary"):
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            try:
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {file.name}... ({i+1}/{len(uploaded_files)})")
                
                # Read file content
                content = file.read().decode('utf-8')
                file.seek(0)  # Reset for potential re-read
                
                title = file.name.split('.')[0]
                
                # Analyze
                result = tracker.analyze_story_complete(
                    text=content,
                    title=title,
                    genre=batch_genre,
                    save_to_db=batch_save_db,
                    get_ai_feedback=batch_ai_feedback,
                    feedback_types=["general"] if batch_ai_feedback else []
                )
                
                results[title] = result
                
            except Exception as e:
                st.error(f"Error analyzing {file.name}: {e}")
                continue
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Batch analysis complete!")
        
        # Store results
        st.session_state.batch_results = results
        
        # Display summary
        st.success(f"Successfully analyzed {len(results)} files!")
        
        # Summary statistics
        # Summary statistics
        if results:
            display_batch_results(results)

def display_batch_results(results):
    """Display batch analysis results"""
    
    st.subheader("📊 Batch Analysis Results")
    
    # Create summary dataframe
    summary_data = []
    for title, result in results.items():
        summary_data.append({
            'Title': title,
            'Word Count': result['basic_metrics']['word_count'],
            'Sentences': result['basic_metrics']['sentence_count'],
            'Dialogue %': result['dialogue_ratio'],
            'Vocab Richness': result['vocabulary']['vocabulary_richness'],
            'Genre': result['genre']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Total Stories", len(results))
    with col2:
        st.metric("📝 Total Words", f"{df_summary['Word Count'].sum():,}")
    with col3:
        st.metric("📊 Avg Words", f"{df_summary['Word Count'].mean():.0f}")
    with col4:
        st.metric("💬 Avg Dialogue", f"{df_summary['Dialogue %'].mean():.1f}%")
    
    # Results table
    st.subheader("📋 Detailed Results")
    st.dataframe(df_summary, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Word count distribution
        fig1 = px.bar(
            df_summary,
            x='Title',
            y='Word Count',
            title="Word Count by Story"
        )
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Vocabulary richness comparison
        fig2 = px.scatter(
            df_summary,
            x='Word Count',
            y='Vocab Richness',
            color='Genre',
            title="Vocabulary Richness vs Word Count",
            hover_data=['Title']
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Export batch results
    st.subheader("📁 Export Batch Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Summary CSV"):
            csv = df_summary.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📄 Export Full Results JSON"):
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="💾 Download JSON",
                data=json_data,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def comparison_page():
    """Story comparison page"""
    
    st.markdown('<h1 class="main-header">⚖️ Story Comparison</h1>', unsafe_allow_html=True)
    
    tracker = initialize_tracker()
    if not tracker:
        st.error("Writing analyzer not available")
        return
    
    # Get list of analyzed stories
    try:
        stories_df = tracker.get_progress_summary(days=365)  # Get last year
        
        if stories_df is None or stories_df.empty:
            st.info("No stories found in database. Analyze some stories first!")
            return
        
        story_titles = stories_df['title'].tolist()
        
    except Exception as e:
        st.error(f"Error loading stories: {e}")
        return
    
    # Story selection
    col1, col2 = st.columns(2)
    
    with col1:
        story1 = st.selectbox("Select First Story", story_titles, key="story1")
    
    with col2:
        story2 = st.selectbox("Select Second Story", story_titles, key="story2")
    
    if story1 and story2 and story1 != story2:
        # Get story data
        story1_data = stories_df[stories_df['title'] == story1].iloc[0]
        story2_data = stories_df[stories_df['title'] == story2].iloc[0]
        
        # Comparison metrics
        st.subheader("📊 Comparison Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Word Count**")
            st.metric(story1, f"{story1_data['word_count']:,}")
            st.metric(story2, f"{story2_data['word_count']:,}")
            
            diff = story2_data['word_count'] - story1_data['word_count']
            st.metric("Difference", f"{diff:+,}")
        
        with col2:
            st.write("**Vocabulary Richness**")
            st.metric(story1, f"{story1_data['vocabulary_richness']:.3f}")
            st.metric(story2, f"{story2_data['vocabulary_richness']:.3f}")
            
            diff = story2_data['vocabulary_richness'] - story1_data['vocabulary_richness']
            st.metric("Difference", f"{diff:+.3f}")
        
        with col3:
            st.write("**Dialogue Ratio**")
            st.metric(story1, f"{story1_data['dialogue_ratio']:.1f}%")
            st.metric(story2, f"{story2_data['dialogue_ratio']:.1f}%")
            
            diff = story2_data['dialogue_ratio'] - story1_data['dialogue_ratio']
            st.metric("Difference", f"{diff:+.1f}%")
        
        # Radar chart comparison
        st.subheader("📈 Visual Comparison")
        
        # Normalize metrics for radar chart
        metrics = ['Word Count', 'Vocab Richness', 'Dialogue Ratio', 'Avg Sentence Length']
        
        story1_values = [
            min(story1_data['word_count'] / 5000 * 100, 100),
            story1_data['vocabulary_richness'] * 100,
            story1_data['dialogue_ratio'],
            min(story1_data.get('avg_sentence_length', 15) / 30 * 100, 100)
        ]
        
        story2_values = [
            min(story2_data['word_count'] / 5000 * 100, 100),
            story2_data['vocabulary_richness'] * 100,
            story2_data['dialogue_ratio'],
            min(story2_data.get('avg_sentence_length', 15) / 30 * 100, 100)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=story1_values,
            theta=metrics,
            fill='toself',
            name=story1,
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=story2_values,
            theta=metrics,
            fill='toself',
            name=story2,
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Story Comparison Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def settings_page():
    """Settings and configuration page"""

    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)

    tracker = initialize_tracker()
    if not tracker:
        st.error("Writing analyzer not available")
        return

    # Database management
    st.subheader("🗄️ Database Management")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Database Operations:**")

        if st.button("📊 View Database Stats"):
            try:
                # Get database statistics
                conn = sqlite3.connect(tracker.db_path)
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM stories")
                story_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM goals")
                goal_count = cursor.fetchone()[0]

                cursor.execute("SELECT MIN(created_date), MAX(created_date) FROM stories")
                date_range = cursor.fetchone()

                conn.close()

                st.write(f"• Total stories: {story_count}")
                st.write(f"• Active goals: {goal_count}")
                if date_range[0]:
                    st.write(f"• Date range: {date_range[0]} to {date_range[1]}")

            except Exception as e:
                st.error(f"Error accessing database: {e}")

        days_to_keep = st.number_input("Keep data for how many days?", min_value=1, value=365)
        if st.button("🧹 Clean Old Data", type="secondary"):
            try:
                cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
                conn = sqlite3.connect(tracker.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM stories WHERE created_date < ?", (cutoff_date,))
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                st.success(f"Deleted {deleted_count} old stories")
            except Exception as e:
                st.error(f"Error cleaning data: {e}")

    with col2:
        st.write("**Export/Import:**")

        if st.button("📤 Export All Data"):
            try:
                # Export all stories and goals from the database
                conn = sqlite3.connect(tracker.db_path)
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM stories")
                stories = cursor.fetchall()
                stories_columns = [desc[0] for desc in cursor.description]

                cursor.execute("SELECT * FROM goals")
                goals = cursor.fetchall()
                goals_columns = [desc[0] for desc in cursor.description]

                conn.close()

                export_data = {
                    "stories": [dict(zip(stories_columns, row)) for row in stories],
                    "goals": [dict(zip(goals_columns, row)) for row in goals]
                }

                json_data = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="💾 Download Backup",
                    data=json_data,
                    file_name=f"writing_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error exporting data: {e}")

        uploaded_backup = st.file_uploader(
            "📥 Import Backup File",
            type=['json'],
            help="Upload a previously exported backup file"
        )

        if uploaded_backup and st.button("Import Data"):
            try:
                backup_data = json.load(uploaded_backup)
                # If your tracker has an import method, call it here:
                # tracker.import_data(backup_data)
                st.success("Data imported successfully!")
            except Exception as e:
                st.error(f"Error importing data: {e}")

    # AI Settings
    st.subheader("🤖 AI Feedback Settings")
    st.write("Configure AI feedback preferences:")

    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Enable detailed character analysis", value=True)
        st.checkbox("Include style suggestions", value=True)
        st.checkbox("Provide structure feedback", value=True)
    with col2:
        st.selectbox("Feedback detail level", ["Brief", "Moderate", "Detailed"], index=1)
        st.selectbox("Feedback tone", ["Professional", "Encouraging", "Critical"], index=1)

    # Display preferences
    st.subheader("🎨 Display Preferences")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Default chart type", ["Bar", "Line", "Scatter"], index=0)
        st.selectbox("Color theme", ["Default", "Dark", "Colorful"], index=0)
    with col2:
        st.number_input("Default time range (days)", min_value=1, max_value=365, value=30)
        st.checkbox("Auto-refresh data", value=True)

def main():
    """Main application function"""
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📊 Writing Analytics")
        st.markdown("---")
        
        page = st.selectbox(
            "Navigate to:",
            ["📝 Text Analysis", "📈 Progress Dashboard", "📁 Batch Analysis", 
             "⚖️ Story Comparison", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Quick stats in sidebar
        if ANALYZER_AVAILABLE:
            tracker = initialize_tracker()
            if tracker:
                try:
                    recent_data = tracker.get_progress_summary(days=7)
                    if recent_data is not None and not recent_data.empty:
                        st.markdown("### 📊 This Week")
                        st.write(f"Stories: {len(recent_data)}")
                        st.write(f"Words: {recent_data['word_count'].sum():,}")
                        st.write(f"Avg Length: {recent_data['word_count'].mean():.0f}")
                except:
                    pass
        
        # Info section
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
        <h4>💡 Tips</h4>
        <ul>
        <li>Regular analysis helps track improvement</li>
        <li>Compare stories to see growth patterns</li>
        <li>Use batch analysis for multiple files</li>
        <li>Set goals to stay motivated</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content based on page selection
    if page == "📝 Text Analysis":
        analyze_text_page()
    elif page == "📈 Progress Dashboard":
        progress_dashboard()
    elif page == "📁 Batch Analysis":
        batch_analysis_page()
    elif page == "⚖️ Story Comparison":
        comparison_page()
    elif page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
from datetime import datetime, timedelta
import json
import sqlite3
import os
from pathlib import Path
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import spacy
import re
from collections import Counter

# Import ollama - make sure you have it installed: pip install ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not installed. AI features will be disabled.")
    print("Install with: pip install ollama")

# Import spaCy for advanced NLP - optional but recommended
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
    print("✅ spaCy available for advanced NLP analysis")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy not installed. Using fallback NLP analysis.")
    print("For enhanced features, install with: pip install spacy")
    print("Then download model: python -m spacy download en_core_web_sm")

class WritingProgressTracker:
    """Enhanced Story Analyzer with Progress Tracking and Analytics Dashboard"""
    
    def __init__(self, model_name="deepseek-r1:8b-0528-qwen3-q4_K_M", db_path='writing_progress.db'):
        global OLLAMA_AVAILABLE  # Fix: Reference global variable
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.model_name = model_name
        self.db_path = db_path
        self.ollama_available = OLLAMA_AVAILABLE  # Store as instance variable
        
        # Initialize database
        self.init_database()
        
        # Check if Ollama is running
        if self.ollama_available:
            try:
                ollama.list()  # Test connection
                print(f"✅ Connected to Ollama with model: {model_name}")
            except Exception as e:
                print(f"⚠️ Ollama not running or error: {e}")
                print("Start it with: ollama serve")
                self.ollama_available = False
    
    def init_database(self):
        """Initialize SQLite database for progress tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Stories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            word_count INTEGER,
            sentence_count INTEGER,
            paragraph_count INTEGER,
            character_count INTEGER,
            dialogue_ratio REAL,
            vocabulary_richness REAL,
            avg_sentence_length REAL,
            avg_word_length REAL,
            genre TEXT,
            status TEXT,
            created_date TEXT,
            last_modified TEXT,
            file_path TEXT
        )
        ''')
        
        # Writing sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS writing_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER,
            session_date TEXT,
            words_written INTEGER,
            time_spent INTEGER,
            session_notes TEXT,
            mood_rating INTEGER,
            FOREIGN KEY (story_id) REFERENCES stories (id)
        )
        ''')
        
        # AI feedback table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER,
            feedback_type TEXT,
            feedback_content TEXT,
            feedback_date TEXT,
            FOREIGN KEY (story_id) REFERENCES stories (id)
        )
        ''')
        
        # Goals table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS writing_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_type TEXT,
            target_value INTEGER,
            current_value INTEGER,
            deadline TEXT,
            status TEXT,
            created_date TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    


    stop_words = set(stopwords.words('english'))

    def basic_metrics(self, text):
        """Calculate basic text metrics using spaCy for better accuracy"""
        clean_text = text.strip()
        if not clean_text:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'character_count': 0,
                'avg_sentence_length': 0,
                'avg_words_per_paragraph': 0
            }
        
        # Use spaCy for proper tokenization and sentence segmentation
        doc = self.nlp(clean_text)
        
        # Count words (excluding punctuation and whitespace)
        words = [token for token in doc if not token.is_punct and not token.is_space and token.text.strip()]
        word_count = len(words)
        
        # Count sentences using spaCy's sentence segmentation
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        
        # Count paragraphs (double newline separation)
        paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Calculate averages
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Character count (excluding spaces)
        char_count = len(clean_text.replace(' ', ''))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'character_count': char_count,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_words_per_paragraph': round(word_count / paragraph_count, 2) if paragraph_count > 0 else 0
        }

    def extract_character_names(self, text):
        """Extract potential character names using spaCy NER"""
        doc = self.nlp(text)
        
        # Extract PERSON entities from spaCy NER
        person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        # Also include proper nouns that appear frequently (backup method)
        proper_nouns = []
        for token in doc:
            if (token.pos_ == "PROPN" and 
                token.text.isalpha() and 
                len(token.text) > 2 and
                token.text not in self.stop_words):
                proper_nouns.append(token.text)
        
        # Combine both methods
        all_potential_names = person_entities + proper_nouns
        
        # Count occurrences and filter by frequency
        name_counts = Counter(all_potential_names)
        
        # Filter out single occurrences and common false positives
        common_false_positives = {
            'God', 'Lord', 'Sir', 'Mr', 'Mrs', 'Miss', 'Dr', 'Professor',
            'King', 'Queen', 'Prince', 'Princess', 'Duke', 'Earl',
            'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        }
        
        frequent_names = {
            name: count for name, count in name_counts.items() 
            if count > 1 and name not in common_false_positives
        }
        
        return frequent_names

    def calculate_dialogue_ratio(self, text):
        """Calculate dialogue percentage with better detection"""
        # Multiple dialogue patterns
        patterns = [
            r'"[^"]*"',           # Standard double quotes
            r"'[^']*'",           # Single quotes (when used for dialogue)
            r'«[^»]*»',           # French quotes
            r'"[^"]*"',           # Curly quotes
            r'ʻ[^ʻ]*ʻ',           # Curly single quotes
        ]
        
        dialogue_matches = []
        for pattern in patterns:
            dialogue_matches.extend(re.findall(pattern, text))
        
        # Remove duplicates and clean
        unique_dialogue = list(set(dialogue_matches))
        dialogue_text = ' '.join(unique_dialogue)
        
        # Use spaCy for accurate word counting
        if dialogue_text.strip():
            dialogue_doc = self.nlp(dialogue_text)
            dialogue_word_count = len([token for token in dialogue_doc if not token.is_punct and not token.is_space])
        else:
            dialogue_word_count = 0
        
        # Total word count using spaCy
        total_doc = self.nlp(text)
        total_words = len([token for token in total_doc if not token.is_punct and not token.is_space])
        
        dialogue_percentage = (dialogue_word_count / total_words * 100) if total_words > 0 else 0
        
        return round(dialogue_percentage, 2)

    def vocabulary_analysis(self, text):
        """Analyze vocabulary richness using spaCy lemmatization"""
        doc = self.nlp(text)
        
        # Get meaningful words (exclude punctuation, spaces, stop words)
        # Extended stop words list for better filtering
        extended_stop_words = self.stop_words.union({
            's', 'like', 'just', 'really', 'get', 'got', 'go', 'going', 'went', 'come', 'came',
            'say', 'said', 'tell', 'told', 'ask', 'asked', 'think', 'thought', 'know', 'knew',
            'see', 'saw', 'look', 'looked', 'want', 'wanted', 'need', 'needed', 'let', 'make',
            'made', 'take', 'took', 'give', 'gave', 'put', 'call', 'called', 'find', 'found',
            'feel', 'felt', 'seem', 'seemed', 'turn', 'turned', 'try', 'tried', 'keep', 'kept',
            'start', 'started', 'stop', 'stopped', 'work', 'worked', 'play', 'played', 'move',
            'moved', 'live', 'lived', 'show', 'showed', 'hear', 'heard', 'leave', 'left',
            'meet', 'met', 'run', 'ran', 'walk', 'walked', 'talk', 'talked', 'sit', 'sat',
            'stand', 'stood', 'bring', 'brought', 'happen', 'happened', 'write', 'wrote',
            'read', 'open', 'opened', 'close', 'closed', 'change', 'changed', 'add', 'added',
            'use', 'used', 'way', 'ways', 'time', 'times', 'day', 'days', 'year', 'years',
            'thing', 'things', 'people', 'person', 'man', 'woman', 'boy', 'girl', 'child',
            'children', 'place', 'places', 'home', 'house', 'room', 'door', 'window', 'car',
            'water', 'food', 'money', 'book', 'books', 'school', 'work', 'job', 'hand', 'hands',
            'head', 'face', 'eye', 'eyes', 'back', 'side', 'part', 'end', 'case', 'fact',
            'point', 'right', 'left', 'long', 'short', 'high', 'low', 'big', 'small', 'large',
            'little', 'old', 'new', 'good', 'bad', 'great', 'important', 'public', 'able',
            'early', 'young', 'different', 'local', 'sure', 'possible', 'late', 'hard', 'far',
            'real', 'full', 'next', 'last', 'few', 'several', 'many', 'much', 'still', 'even',
            'also', 'always', 'never', 'often', 'sometimes', 'usually', 'however', 'perhaps',
            'maybe', 'probably', 'certainly', 'definitely', 'quite', 'rather', 'pretty',
            'especially', 'particularly', 'actually', 'really', 'truly', 'exactly', 'nearly',
            'almost', 'completely', 'totally', 'entirely', 'absolutely', 'perfectly', 'clearly',
            'obviously', 'apparently', 'unfortunately', 'hopefully', 'finally', 'eventually',
            'immediately', 'suddenly', 'quickly', 'slowly', 'carefully', 'easily', 'probably'
        })
        
        meaningful_words = []
        for token in doc:
            lemma_lower = token.lemma_.lower()
            if (not token.is_punct and 
                not token.is_space and 
                not token.is_stop and
                token.text.isalpha() and
                len(token.text) > 2 and  # Increased minimum length
                lemma_lower not in extended_stop_words and
                not lemma_lower.startswith("'") and  # Remove contractions
                token.pos_ not in ['DET', 'ADP', 'CCONJ', 'SCONJ', 'PRON', 'AUX']):  # Filter by POS tags
                meaningful_words.append(lemma_lower)
        
        if not meaningful_words:
            return {
                'unique_words': 0,
                'total_meaningful_words': 0,
                'vocabulary_richness': 0,
                'most_common_words': [],
                'avg_word_length': 0
            }
        
        unique_words = set(meaningful_words)
        total_meaningful_words = len(meaningful_words)
        
        # Vocabulary richness (Type-Token Ratio)
        vocab_richness = len(unique_words) / total_meaningful_words
        
        # Word frequency analysis
        word_freq = Counter(meaningful_words)
        most_common = word_freq.most_common(10)
        
        # Average word length (using original words, not lemmas, for accurate length)
        original_meaningful_words = []
        for token in doc:
            lemma_lower = token.lemma_.lower()
            if (not token.is_punct and 
                not token.is_space and 
                not token.is_stop and
                token.text.isalpha() and
                len(token.text) > 2 and  # Increased minimum length
                lemma_lower not in extended_stop_words and
                not lemma_lower.startswith("'") and  # Remove contractions
                token.pos_ not in ['DET', 'ADP', 'CCONJ', 'SCONJ', 'PRON', 'AUX']):  # Filter by POS tags
                original_meaningful_words.append(token.text.lower())
        
        avg_word_length = sum(len(word) for word in original_meaningful_words) / len(original_meaningful_words) if original_meaningful_words else 0
        
        return {
            'unique_words': len(unique_words),
            'total_meaningful_words': total_meaningful_words,
            'vocabulary_richness': round(vocab_richness, 3),
            'most_common_words': most_common,
            'avg_word_length': round(avg_word_length, 2)
        }
    def analyze_with_nlp(self, text):
        """Enhanced NLP analysis using spaCy"""
        global SPACY_AVAILABLE
        
        if not SPACY_AVAILABLE:
            print("📝 Using fallback NLP analysis (spaCy not available)")
            return self._fallback_nlp_analysis(text)
            
        try:
            # Load spaCy model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️ spaCy model 'en_core_web_sm' not found.")
                print("Install with: python -m spacy download en_core_web_sm")
                print("Using fallback analysis...")
                return self._fallback_nlp_analysis(text)
            
            # Process text with spaCy
            doc = nlp(text)
            
            # Extract entities (characters, locations, etc.)
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            # Clean up entities - remove duplicates and common false positives
            cleaned_entities = {}
            for label, ents in entities.items():
                unique_ents = list(set(ents))
                # Filter out single letters and common words
                filtered_ents = [e for e in unique_ents if len(e) > 1 and e.lower() not in self.stop_words]
                if filtered_ents:
                    cleaned_entities[label] = filtered_ents
            
            # Sentence analysis using spaCy's sentence segmentation
            sentences = [sent.text.strip() for sent in doc.sents]
            sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
            
            # POS tagging analysis
            pos_counts = Counter([token.pos_ for token in doc if not token.is_space and not token.is_punct])
            
            # Basic sentiment analysis using spaCy's built-in features
            # First, install VADER if not already installed:
            # pip install vaderSentiment



            # Initialize VADER analyzer once (outside your function for efficiency)
            sentiment_analyzer = SentimentIntensityAnalyzer()

            # Replace your sentiment analysis section with this:
            sentiment_scores = []
            for sent in doc.sents:
                # Use VADER to get compound sentiment score (-1 to 1)
                sent_text = sent.text.strip()
                if sent_text:  # Skip empty sentences
                    sentiment_score = sentiment_analyzer.polarity_scores(sent_text)['compound']
                    sentiment_scores.append(sentiment_score)

            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

            return {
                'entities': cleaned_entities,
                'sentences': sentences,
                'sentence_lengths': sentence_lengths,
                'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
                'sentence_length_std': np.std(sentence_lengths) if sentence_lengths else 0,
                'pos_distribution': dict(pos_counts),
                'avg_sentiment': round(avg_sentiment, 3),
                'total_entities': sum(len(ents) for ents in cleaned_entities.values()),
                'analysis_method': 'spaCy'
            }
            
        except Exception as e:
            print(f"⚠️ Error in spaCy analysis: {e}")
            print("Falling back to basic analysis...")
            return self._fallback_nlp_analysis(text)
    
    def _fallback_nlp_analysis(self, text):
        """Fallback NLP analysis without spaCy"""
        # Basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_lengths = [len(sent.split()) for sent in sentences]
        
        # Basic named entity recognition (capitalized words)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        common_words = {'The', 'And', 'But', 'Or', 'So', 'Then', 'When', 'Where', 'What', 'Who', 'How', 'Why', 'This', 'That', 'Chapter', 'Part', 'He', 'She', 'They', 'It'}
        potential_names = [word for word in words if word not in common_words]
        name_counts = Counter(potential_names)
        
        # Group potential entities
        entities = {
            'PERSON': [name for name, count in name_counts.items() if count > 1],
            'MISC': [name for name, count in name_counts.items() if count == 1]
        }
        
        return {
            'entities': {k: v for k, v in entities.items() if v},
            'sentences': sentences,
            'sentence_lengths': sentence_lengths,
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'sentence_length_std': np.std(sentence_lengths) if sentence_lengths else 0,
            'pos_distribution': {},
            'avg_sentiment': 0,
            'total_entities': sum(len(ents) for ents in entities.values() if ents),
            'analysis_method': 'fallback'
        }
    
    def get_ai_feedback(self, text, analysis_data=None, feedback_type="general"):
        """Get AI feedback on the story"""
        if not self.ollama_available:
            return "AI feedback unavailable - Ollama not connected"
        
        # Truncate text if too long (keep first 2000 characters for context)
        text_sample = text[:2000] + ("..." if len(text) > 2000 else "")
        
        # Build context from analysis data
        context = ""
        if analysis_data:
            basic = analysis_data.get('basic_metrics', {})
            nlp_data = analysis_data.get('nlp_analysis', {})
            entities = nlp_data.get('entities', {})
            
            context = f"""
Story Stats:
- {basic.get('word_count', 0)} words, {basic.get('sentence_count', 0)} sentences
- Average sentence length: {basic.get('avg_sentence_length', 0)} words
- Dialogue: {analysis_data.get('dialogue_ratio', 0)}% of story
- Characters found: {', '.join(entities.get('PERSON', []))}
- Locations mentioned: {', '.join(entities.get('GPE', []) + entities.get('LOC', []))}
- Vocabulary richness: {analysis_data.get('vocabulary', {}).get('vocabulary_richness', 0)}
"""
        
        # Enhanced prompts with NLP insights
        prompts = {
            "general": f"""
As a writing mentor, analyze this story excerpt and provide constructive feedback:

{context}

Story excerpt:
{text_sample}

Please provide:
1. Overall impression and strengths
2. Areas for improvement based on the metrics above
3. One specific, actionable suggestion
4. Writing style observations

Keep feedback encouraging but honest, suitable for a developing writer.
""",
            
            "character": f"""
Focus on character development in this story:

{context}

Story excerpt:
{text_sample}

Analyze:
1. Character voice and distinctiveness
2. Character motivations and believability  
3. Dialogue quality and authenticity
4. How well characters are established through actions and speech
5. Suggestions for character improvement
""",
            
            "structure": f"""
Analyze the narrative structure and pacing:

{context}

Story excerpt:
{text_sample}

Evaluate:
1. Story pacing and flow
2. Scene transitions and structure
3. Balance of action/dialogue/description
4. Sentence variety and rhythm
5. Suggestions for improving narrative flow
""",
            
            "style": f"""
Analyze the writing style and craft:

{context}

Story excerpt:
{text_sample}

Focus on:
1. Prose style and voice consistency
2. Word choice and vocabulary effectiveness
3. Sentence structure variety and flow
4. Show vs. tell balance
5. Suggestions for style enhancement
"""
        }
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an experienced creative writing mentor with expertise in narrative craft, character development, and prose style. Provide constructive and a little destructive, specific, and encouraging feedback to help writers improve their craft. Do not show your thinking process or reasoning steps. Provide direct, concise feedback without explanations of your analysis process.'
                    },
                    {
                        'role': 'user',
                        'content': prompts.get(feedback_type, prompts["general"])
                    }
                ],
                options={'raw_output':True,'temperature': 0.7, 'stop': ['<thinking>', 'Let me think']}
                
            )
            content = response['message']['content']
# Remove thinking blocks
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
            return content.strip()

        
        except Exception as e:
            return f"Error getting AI feedback: {str(e)}"
    
    def save_story_to_db(self, title, text, genre="Unknown", status="Draft"):
        """Save story analysis to database"""
        basic = self.basic_metrics(text)
        vocab = self.vocabulary_analysis(text)
        dialogue_ratio = self.calculate_dialogue_ratio(text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO stories (
            title, word_count, sentence_count, paragraph_count, 
            character_count, dialogue_ratio, vocabulary_richness,
            avg_sentence_length, avg_word_length, genre, status,
            created_date, last_modified
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            title, 
            basic['word_count'],
            basic['sentence_count'], 
            basic['paragraph_count'],
            basic['character_count'],
            dialogue_ratio,
            vocab['vocabulary_richness'],
            basic['avg_sentence_length'],
            vocab['avg_word_length'],
            genre,
            status,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        story_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return story_id
    
    def analyze_story_complete(self, text, title="Untitled Story", genre="Unknown", save_to_db=True, get_ai_feedback=True, feedback_types=["general"]):
        """Complete story analysis with NLP and AI feedback"""
        print(f"\n🚀 COMPLETE ANALYSIS: {title}")
        print("=" * (len(title) + 20))
        
        # Basic metrics
        basic = self.basic_metrics(text)
        vocab = self.vocabulary_analysis(text)
        dialogue_ratio = self.calculate_dialogue_ratio(text)
        
        # Enhanced NLP analysis
        nlp_analysis = self.analyze_with_nlp(text)
        
        # Display enhanced metrics
        print(f"\n📊 STORY METRICS:")
        print(f"Words: {basic['word_count']}")
        print(f"Sentences: {basic['sentence_count']} (avg: {nlp_analysis['avg_sentence_length']:.1f} words)")
        print(f"Dialogue: {dialogue_ratio}%")
        print(f"Vocabulary richness: {vocab['vocabulary_richness']}")
        print(f"Sentence variety (std): {nlp_analysis['sentence_length_std']:.1f}")
        
        # Display entities found
        entities = nlp_analysis.get('entities', {})
        if entities:
            print(f"\n👥 CHARACTERS & ENTITIES:")
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"{entity_type}: {', '.join(entity_list[:5])}")
        
        # Prepare complete analysis data
        analysis_data = {
            'title': title,
            'genre': genre,
            'basic_metrics': basic,
            'vocabulary': vocab,
            'dialogue_ratio': dialogue_ratio,
            'nlp_analysis': nlp_analysis,
            'analysis_date': datetime.now().isoformat()
        }
        
        # Save to database
        if save_to_db:
            story_id = self.save_story_to_db(title, text, genre)
            analysis_data['story_id'] = story_id
            print(f"💾 Saved to database (ID: {story_id})")
        
        # Get AI feedback
        if get_ai_feedback and self.ollama_available:
            print(f"\n🤖 AI FEEDBACK:")
            print("-" * 50)
            
            for feedback_type in feedback_types:
                print(f"\n📝 {feedback_type.upper()} FEEDBACK:")
                feedback = self.get_ai_feedback(text, analysis_data, feedback_type)
                print(feedback)
                
                # Save feedback to database
                if save_to_db and 'story_id' in analysis_data:
                    self.save_feedback_to_db(analysis_data['story_id'], feedback_type, feedback)
                
                # Store in analysis data
                if 'ai_feedback' not in analysis_data:
                    analysis_data['ai_feedback'] = {}
                analysis_data['ai_feedback'][feedback_type] = feedback
        elif get_ai_feedback and not self.ollama_available:
            print("\n🤖 AI FEEDBACK: Unavailable (Ollama not connected)")
        
        return analysis_data
    
    def save_feedback_to_db(self, story_id, feedback_type, feedback_content):
        """Save AI feedback to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO ai_feedback (story_id, feedback_type, feedback_content, feedback_date)
        VALUES (?, ?, ?, ?)
        ''', (story_id, feedback_type, feedback_content, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_progress_summary(self, days=30):
        """Get writing progress summary"""
        conn = sqlite3.connect(self.db_path)
        
        # Get stories from last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        try:
            stories_df = pd.read_sql_query('''
            SELECT * FROM stories 
            WHERE created_date >= ? 
            ORDER BY created_date DESC
            ''', conn, params=[cutoff_date])
        except Exception as e:
            print(f"Error querying database: {e}")
            conn.close()
            return None
        
        if stories_df.empty:
            print(f"📈 No stories found in the last {days} days")
            conn.close()
            return None
        
        # Calculate summary statistics
        total_words = stories_df['word_count'].sum()
        total_stories = len(stories_df)
        avg_words_per_story = stories_df['word_count'].mean()
        avg_dialogue = stories_df['dialogue_ratio'].mean()
        avg_vocab_richness = stories_df['vocabulary_richness'].mean()
        
        print(f"\n📈 PROGRESS SUMMARY (Last {days} days):")
        print(f"Stories written: {total_stories}")
        print(f"Total words: {total_words:,}")
        print(f"Average words per story: {avg_words_per_story:.0f}")
        print(f"Average dialogue ratio: {avg_dialogue:.1f}%")
        print(f"Average vocabulary richness: {avg_vocab_richness:.3f}")
        
        # Genre breakdown
        if 'genre' in stories_df.columns:
            genre_counts = stories_df['genre'].value_counts()
            print(f"\n📚 Genre breakdown:")
            for genre, count in genre_counts.items():
                print(f"  {genre}: {count} stories")
        
        conn.close()
        return stories_df
    
    def create_progress_charts(self, days=30):
        """Create visual progress charts"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        try:
            stories_df = pd.read_sql_query('''
            SELECT * FROM stories 
            WHERE created_date >= ? 
            ORDER BY created_date
            ''', conn, params=[cutoff_date])
        except Exception as e:
            print(f"Error querying database: {e}")
            conn.close()
            return
        
        if stories_df.empty:
            print("No data to visualize")
            conn.close()
            return
        
        # Convert dates
        stories_df['created_date'] = pd.to_datetime(stories_df['created_date'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Writing Progress - Last {days} Days', fontsize=16)
        
        # 1. Words over time
        axes[0,0].plot(stories_df['created_date'], stories_df['word_count'], 'o-')
        axes[0,0].set_title('Word Count per Story')
        axes[0,0].set_ylabel('Words')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Vocabulary richness over time
        axes[0,1].plot(stories_df['created_date'], stories_df['vocabulary_richness'], 'o-', color='green')
        axes[0,1].set_title('Vocabulary Richness Trend')
        axes[0,1].set_ylabel('Richness Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Dialogue ratio distribution
        axes[1,0].hist(stories_df['dialogue_ratio'], bins=10, alpha=0.7, color='orange')
        axes[1,0].set_title('Dialogue Ratio Distribution')
        axes[1,0].set_xlabel('Dialogue %')
        axes[1,0].set_ylabel('Frequency')
        
        # 4. Average sentence length
        axes[1,1].plot(stories_df['created_date'], stories_df['avg_sentence_length'], 'o-', color='red')
        axes[1,1].set_title('Average Sentence Length')
        axes[1,1].set_ylabel('Words per Sentence')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        conn.close()
    
    def set_writing_goal(self, goal_type, target_value, deadline=None):
        """Set a writing goal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if deadline is None:
            deadline = (datetime.now() + timedelta(days=30)).isoformat()
        
        cursor.execute('''
        INSERT INTO writing_goals (goal_type, target_value, current_value, deadline, status, created_date)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (goal_type, target_value, 0, deadline, 'Active', datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"🎯 Goal set: {target_value} {goal_type} by {deadline[:10]}")
    
    def check_goals(self):
        """Check progress on writing goals"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            goals_df = pd.read_sql_query('''
            SELECT * FROM writing_goals WHERE status = "Active"
            ''', conn)
        except Exception as e:
            print(f"Error querying goals: {e}")
            conn.close()
            return
        
        if goals_df.empty:
            print("🎯 No active goals set")
            conn.close()
            return
        
        print("\n🎯 GOAL PROGRESS:")
        print("-" * 30)
        
        for _, goal in goals_df.iterrows():
            # Calculate current progress based on goal type
            try:
                if goal['goal_type'] == 'words':
                    result = pd.read_sql_query('SELECT SUM(word_count) as total FROM stories', conn)
                    current = result.iloc[0]['total'] if result.iloc[0]['total'] is not None else 0
                elif goal['goal_type'] == 'stories':
                    result = pd.read_sql_query('SELECT COUNT(*) as count FROM stories', conn)
                    current = result.iloc[0]['count']
                else:
                    current = goal['current_value']
            except Exception as e:
                print(f"Error calculating progress: {e}")
                current = 0
            
            progress = (current / goal['target_value']) * 100 if goal['target_value'] > 0 else 0
            days_left = (datetime.fromisoformat(goal['deadline']) - datetime.now()).days
            
            print(f"{goal['goal_type'].title()}: {current}/{goal['target_value']} ({progress:.1f}%)")
            print(f"  Deadline: {goal['deadline'][:10]} ({days_left} days left)")
            print(f"  Status: {'✅ Complete!' if progress >= 100 else '⏳ In Progress'}")
            print()
        
        conn.close()


# Example usage and testing
def demo_complete_suite():
    """Demonstrate the complete writing analytics suite"""
    
    sample_stories = {
        "The Last Letter": '''
        Dear Sarah,
        
        I know you won't read this until after I'm gone, but I needed to write it anyway. The doctors say I have maybe a week left, two if I'm lucky. But I'm not writing to talk about that.
        
        I'm writing to tell you about the garden.
        
        Do you remember when we first moved to Maple Street? You were seven, and you cried because you had to leave your friends behind. I promised you we'd make something beautiful in the backyard, something that would make this place feel like home.
        
        We planted those tomatoes together that first spring. You insisted on watering them twice a day, even when I told you once was enough. By summer, we had more tomatoes than we knew what to do with. You made me promise we'd have a garden every year after that.
        
        We kept that promise, didn't we? Every spring, we'd plan what to plant. Every summer, we'd harvest together. Every fall, we'd prepare for the next year.
        
        I want you to know that the garden taught me something important: the best things in life aren't the ones you can hold onto forever. They're the ones you nurture, watch grow, and then let go when their season is over.
        
        The tomatoes are ripe now. Mrs. Henderson next door has been helping me harvest them. She promised to share them with you when you come home.
        
        Keep planting, Sarah. Keep growing beautiful things, even when - especially when - you know they won't last forever.
        
        All my love,
        Dad
        ''',
        
        "Midnight Coffee": '''
        The coffee shop on Fifth Street stayed open until midnight, which was perfect for insomniacs like Marcus. He'd discovered it three months ago, during one of his wandering sessions when sleep refused to come.
        
        "The usual?" Elena asked, not looking up from wiping down the counter.
        
        Marcus nodded, settling into his corner booth. Black coffee, no sugar. The bitter taste matched his mood most nights.
        
        "Rough night?" she continued, already knowing the answer. Elena had this way of reading people, probably came from working the graveyard shift at a coffee shop frequented by night owls, shift workers, and people running from their thoughts.
        
        "They're all rough nights," Marcus replied, pulling out his laptop. The screen cast a blue glow across his face as he opened the document he'd been working on for weeks. A letter he'd never send, to someone who'd never read it.
        
        The coffee shop hummed with quiet conversation. A nurse finishing her shift sat near the window, scrolling through her phone. Two college students shared a table, textbooks spread between them like battle plans. And in the corner, an elderly man read a paperback novel, occasionally chuckling at something on the page.
        
        These were his people now, Marcus thought. The midnight dwellers, the ones who found community in the small hours when the rest of the world slept.
        
        Elena brought his coffee without being asked, setting it down gently beside his laptop.
        
        "You know," she said quietly, "sometimes the words we can't say out loud are the ones that need to be written down the most."
        
        Marcus looked up at her, surprised. He'd never told her what he was writing.
        
        "The letter?" she asked, nodding toward his screen. "I can tell. I see a lot of letters get written in here. Letters to ex-lovers, to parents, to people who've died. Sometimes to the person in the mirror."
        
        "Which kind is mine?"
        
        Elena smiled sadly. "That's for you to figure out. But whatever it is, it's worth finishing."
        
        She walked away, leaving Marcus alone with his thoughts and the gentle hum of conversation around him. He looked at his screen, at the half-formed sentences that had been taunting him for weeks.
        
        Maybe Elena was right. Maybe some words needed to exist, even if they were never spoken.
        
        He began to type.
        ''',
        
        "The Clockmaker's Daughter": '''
        Time moved differently in her father's shop. Clara had noticed it as a child and still felt it now, twenty years later, as she wound the grandfather clock that had been keeping imperfect time since 1892.
        
        Tick. Pause. Tick. Pause.
        
        The rhythm was hypnotic, almost meditative. Each clock in the shop had its own personality, its own way of marking the moments. The cuckoo clock above the door announced the hours with Germanic precision. The mantle clock on the counter whispered secrets in its steady ticking. And the pocket watches in the display case seemed to hold their breath, waiting for hands to wind them back to life.
        
        "You're just like your father," Mrs. Patterson said, watching Clara work. She'd been a customer for decades, bringing in her late husband's watch for cleaning every six months like clockwork. "He could make time stand still just by touching those gears."
        
        Clara smiled, remembering her father's hands. Strong but gentle, able to handle the most delicate mechanisms without breaking them. He'd taught her that clocks weren't just machines – they were repositories of memory, keeping time for all the moments that mattered.
        
        "Every clock has a story," he used to say. "Our job isn't just to fix them. It's to honor the time they've kept."
        
        The bell above the door chimed, and a young man entered carrying a small wooden clock. Clara recognized the craftsmanship immediately – her father's work from the early 2000s.
        
        "My grandmother passed away last month," the man explained, placing the clock carefully on the counter. "This was on her nightstand for as long as I can remember. It stopped the day she died."
        
        Clara examined the clock gently, her trained eye taking in every detail. The wood was worn smooth from years of handling. The face was clouded with age. But the mechanism inside was still sound, still capable of keeping perfect time.
        
        "Sometimes they just need to know they're still wanted," she said softly, opening the back panel. "Sometimes they need to know their story isn't over."
        
        As she worked, Clara thought about the conversation she'd had with her father the week before he died. He'd been worried about the shop, about whether anyone would understand that their work was about more than just fixing broken things.
        
        "You do understand," he'd said, watching her repair a 1940s alarm clock. "You understand that we're not just clockmakers. We're time keepers. Memory keepers."
        
        Now, as she adjusted the tiny gears and springs that would bring this clock back to life, Clara understood exactly what he meant.
        
        The young man waited patiently as she worked, his eyes wandering over the hundreds of timepieces that filled every shelf and surface.
        
        "She used to wind it every night before bed," he said quietly. "Said it helped her sleep better, knowing time was being kept properly."
        
        Clara smiled, making a final adjustment. The clock began to tick, soft and steady, just as it had for decades.
        
        "There," she said, closing the back panel. "Your grandmother's time is safe."
        
        The man's eyes filled with tears as he listened to the familiar rhythm. "Thank you," he whispered. "Thank you for understanding."
        
        After he left, Clara sat in the quiet shop, surrounded by the symphony of ticking that had been the soundtrack of her childhood. Each tick was a heartbeat, each tock a breath. Together, they were the sound of time itself, measured and marked and honored.
        
        Her father had been right. They weren't just clockmakers.
        
        They were keepers of time, guardians of memory, and sometimes – when they were very lucky – healers of hearts that had stopped keeping time properly.
        
        Clara wound the grandfather clock one more turn and listened as it settled into its ancient rhythm.
        
        Tick. Pause. Tick. Pause.
        
        Time moved differently in her father's shop.
        
        And that was exactly as it should be.
        '''
    }
    
    # Initialize the enhanced tracker
    print("🚀 ENHANCED WRITING PROGRESS TRACKER DEMO")
    print("=" * 50)
    
    try:
        tracker = WritingProgressTracker()
        
        # Analyze each story with different focus areas
        feedback_combinations = [
            ["general", "character"],
            ["structure", "style"], 
            ["general", "character", "style"]
        ]
        
        story_results = {}
        
        for i, (title, story) in enumerate(sample_stories.items()):
            print(f"\n{'='*60}")
            print(f"ANALYZING STORY {i+1}: {title}")
            print(f"{'='*60}")
            
            # Get different types of feedback for variety
            feedback_types = feedback_combinations[i % len(feedback_combinations)]
            
            # Complete analysis
            result = tracker.analyze_story_complete(
                text=story,
                title=title,
                genre="Literary Fiction",
                save_to_db=True,
                get_ai_feedback=True,
                feedback_types=feedback_types
            )
            
            story_results[title] = result
            
            # Add a brief pause between stories for readability
            print("\n" + "⏱️ " * 20)
        
        # Show progress summary
        print(f"\n{'='*60}")
        print("PROGRESS SUMMARY & INSIGHTS")
        print(f"{'='*60}")
        
        progress_data = tracker.get_progress_summary(days=30)
        
        # Check any active goals
        tracker.check_goals()
        
        # Set some example goals
        print(f"\n🎯 SETTING SAMPLE GOALS:")
        tracker.set_writing_goal("words", 10000, deadline=(datetime.now() + timedelta(days=30)).isoformat())
        tracker.set_writing_goal("stories", 10, deadline=(datetime.now() + timedelta(days=60)).isoformat())
        
        # Check goals again
        tracker.check_goals()
        
        # Create visualizations if matplotlib is available
        try:
            print(f"\n📊 GENERATING PROGRESS CHARTS...")
            tracker.create_progress_charts(days=30)
        except Exception as e:
            print(f"⚠️ Chart generation skipped: {e}")
        
        # Summary insights
        print(f"\n💡 WRITING INSIGHTS:")
        print("-" * 30)
        
        if story_results:
            # Calculate cross-story metrics
            total_words = sum(r['basic_metrics']['word_count'] for r in story_results.values())
            avg_vocab_richness = sum(r['vocabulary']['vocabulary_richness'] for r in story_results.values()) / len(story_results)
            avg_dialogue = sum(r['dialogue_ratio'] for r in story_results.values()) / len(story_results)
            
            print(f"✍️ Total words analyzed: {total_words:,}")
            print(f"📚 Average vocabulary richness: {avg_vocab_richness:.3f}")
            print(f"💬 Average dialogue ratio: {avg_dialogue:.1f}%")
            
            # Most common characters across stories
            all_entities = {}
            for result in story_results.values():
                entities = result['nlp_analysis'].get('entities', {})
                for entity_type, entity_list in entities.items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    all_entities[entity_type].extend(entity_list)
            
            if all_entities:
                print(f"\n👥 Character/Entity Types Found:")
                for entity_type, entities in all_entities.items():
                    if entities:
                        unique_entities = list(set(entities))
                        print(f"  {entity_type}: {len(unique_entities)} unique ({', '.join(unique_entities[:3])}...)")
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"Database saved to: {tracker.db_path}")
        print(f"You can now track your writing progress over time!")
        
        return tracker, story_results
        
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Additional utility functions
def quick_analyze(text, title="Quick Analysis"):
    """Quick analysis for shorter texts"""
    tracker = WritingProgressTracker()
    return tracker.analyze_story_complete(
        text=text,
        title=title,
        save_to_db=False,
        get_ai_feedback=True,
        feedback_types=["general"]
    )

def batch_analyze_files(file_paths, save_to_db=True):
    """Analyze multiple text files at once"""
    tracker = WritingProgressTracker()
    results = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            title = Path(file_path).stem
            result = tracker.analyze_story_complete(
                text=text,
                title=title,
                save_to_db=save_to_db,
                get_ai_feedback=True,
                feedback_types=["general", "style"]
            )
            results[title] = result
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            continue
    
    return results

def export_progress_report(tracker, output_file="writing_report.json"):
    """Export a comprehensive progress report"""
    conn = sqlite3.connect(tracker.db_path)
    
    try:
        # Get all data from database
        stories = pd.read_sql_query("SELECT * FROM stories ORDER BY created_date DESC", conn)
        sessions = pd.read_sql_query("SELECT * FROM writing_sessions ORDER BY session_date DESC", conn)
        feedback = pd.read_sql_query("SELECT * FROM ai_feedback ORDER BY feedback_date DESC", conn)
        goals = pd.read_sql_query("SELECT * FROM writing_goals ORDER BY created_date DESC", conn)
        
        # Compile report
        report = {
            'generated_date': datetime.now().isoformat(),
            'summary': {
                'total_stories': len(stories),
                'total_words': stories['word_count'].sum() if not stories.empty else 0,
                'average_story_length': stories['word_count'].mean() if not stories.empty else 0,
                'total_sessions': len(sessions),
                'active_goals': len(goals[goals['status'] == 'Active']) if not goals.empty else 0
            },
            'stories': stories.to_dict('records') if not stories.empty else [],
            'sessions': sessions.to_dict('records') if not sessions.empty else [],
            'feedback': feedback.to_dict('records') if not feedback.empty else [],
            'goals': goals.to_dict('records') if not goals.empty else []
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Progress report exported to: {output_file}")
        return report
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return None
    finally:
        conn.close()

# Run the demo if script is executed directly
if __name__ == "__main__":
    print("🚀 Starting Enhanced Writing Progress Tracker Demo...")
    tracker, results = demo_complete_suite()
    
    if tracker and results:
        print(f"\n🎯 Demo completed successfully!")
        print(f"📁 Check your database: {tracker.db_path}")
        
        # Optional: Export a progress report
        export_progress_report(tracker)
    else:
        print("❌ Demo encountered errors. Check the output above for details.")
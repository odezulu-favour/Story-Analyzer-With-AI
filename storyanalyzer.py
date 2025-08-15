import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from datetime import datetime
import spacy
# from transformers import


class StoryDataAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))



    def basic_metrics(self,text :str) -> dict:
        """
        Calculate basic text metrics"""

        clean_text = text.strip()


        # Word Count
        words = word_tokenize(clean_text)
        word_count = len(words)

        # Sentence Count
        sentences = sent_tokenize(clean_text)
        sentence_count = len(sentences)

        # Paragraph Count
        paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)

        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Character Count (Without spaces)
        char_count = len(clean_text.replace(" ", ""))

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_sentence_length": round(avg_sentence_length),
            "char_count": char_count,
            "avg_words_per_paragraph": round(word_count / paragraph_count,2) if paragraph_count > 0 else 0
        }


    def extract_characters(self, text: str) -> list:
        """
        Extract characters from the text using NLTK and Spacy"""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        characters = set()

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                characters.add(ent.text)

        

        return list(characters)
    

    def extract_locations(self, text: str) -> list:
        """
        Extract locations from the text using NLTK and Spacy"""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        locations = set()

        for ent in doc.ents:
            if ent.label_ == "GPE":
                locations.add(ent.text)

        return list(locations)
    
    def calculate_dialogue_ratio(self, text: str) -> float:
        """
        Calculate the dialogue ratio in the text"""
        sentences = sent_tokenize(text)
        dialogue_count = sum(1 for sentence in sentences if '"' in sentence or "'" in sentence)
        total_sentences = len(sentences)
        
        dialogue_percentage = (dialogue_count / total_sentences) * 100 if total_sentences > 0 else 0.0
        return round(dialogue_percentage, 2)
    
    
    def vocabulary_diversity(self, text: str) -> float:
        """
        Calculate vocabulary diversity in the text"""
        words = word_tokenize(text.lower())
        meaningful_words = [word for word in words if word.isalpha() and word not in self.stop_words and word not in string.punctuation]
        unique_words = set(meaningful_words)
        total_words = len(meaningful_words)
        
        diversity = len(unique_words) / total_words if total_words > 0 else 0.0
        return round(diversity, 2)
    
    def vocabulary_analysis(self, text: str) -> pd.DataFrame:
        """
        Analyze vocabulary in the text and return a DataFrame"""
        words = word_tokenize(text.lower())
        meaningful_words = [word for word in words if word.isalpha() and word not in self.stop_words and word not in string.punctuation]
        
        word_counts = Counter(meaningful_words)
        df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])
        df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
        
        return df
    
    def analyze_story(self, text: str, title="Untitled Story") -> dict:
        """
        Main analysis function that combines all metrics"""
        print(f"\n=== ANALYZING: {title} ===")
        print('=' * (len(title) + 20))

        basic = self.basic_metrics(text)
        print("Basic Metrics:")
        print(f"Word Count: {basic['word_count']}")
        print(f"Sentence Count: {basic['sentence_count']}")
        print(f"Paragraph Count: {basic['paragraph_count']}")
        print(f"Characters Count: {basic['char_count']}")
        print(f"Average Sentence Length: {basic['avg_sentence_length']}")
        print(f"Average Words per Paragraph: {basic['avg_words_per_paragraph']}")
    
        #Characters Analysis
        characters = self.extract_characters(text)
        print(f"\nCharacters Found:")
        if characters:
            for char in characters:
                print(f"- {char}")
        else:
            print("No characters found.")


    # Dialogue Ratio
        dialogue_ratio = self.calculate_dialogue_ratio(text)
        print(f"\nDialogue Percentage: {dialogue_ratio}%")

        if dialogue_ratio > 50:
            print("This story has a high dialogue ratio, indicating a conversational style.")
        elif dialogue_ratio > 20:
            print("This story has a moderate dialogue ratio, balancing narrative and dialogue.")
        else:
            print("This story has a low dialogue ratio, focusing more on narrative than conversation.")

        # Vocabulary Diversity
        vocab_diversity = self.vocabulary_diversity(text)
        print(f"\nVocabulary Diversity: {vocab_diversity}")

        # Vocabulary Analysis
        vocab_df = self.vocabulary_analysis(text)
        print("\nVocabulary Analysis (Top 10 Words):")
        print(vocab_df.head(10))

        return {
            "basic_metrics": basic,
            "characters": characters,
            "dialogue_ratio": dialogue_ratio,
            "vocabulary_diversity": vocab_diversity,
            "vocabulary_analysis": vocab_df
        }
    
    
    def compare_stories(self, story_data_list):
        """Compare multiple story analyses"""
        if len(story_data_list) < 2:
            print("Need at least 2 stories to compare!")
            return
        
        print(f"\n📈 COMPARISON OF {len(story_data_list)} STORIES:")
        print("=" * 50)
        
        # Create comparison table
        comparison_data = []
        for story in story_data_list:
            comparison_data.append({
                'Title': story['title'][:20] + '...' if len(story['title']) > 20 else story['title'],
                'Words': story['basic_metrics']['word_count'],
                'Sentences': story['basic_metrics']['sentence_count'],
                'Avg Sentence': story['basic_metrics']['avg_sentence_length'],
                'Dialogue %': story['dialogue_ratio'],
                'Vocab Richness': story['vocabulary']['vocabulary_richness']
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))


# Example usage and test function
def test_analyzer():
    """Test the analyzer with sample stories"""
    
    # Sample story 1 - dialogue heavy
    sample_story1 = '''
    "I can't believe you did that," Sarah said, shaking her head.
    
    "What choice did I have?" Mike replied, his voice barely above a whisper. "They were going to hurt you."
    
    Sarah looked at him with tears in her eyes. "There's always a choice, Mike. Always."
    
    The rain began to fall harder, creating a curtain between them and the rest of the world. Mike reached out to touch her face, but she stepped back.
    
    "Don't," she whispered. "Just don't."
    
    "Sarah, please. Let me explain."
    
    "There's nothing to explain. I saw what I saw." She turned away from him, her shoulders shaking. "I need time to think."
    
    Mike watched her walk away, knowing that this might be the last time he'd see her. The rain soaked through his jacket, but he didn't move. He couldn't move.
    '''
    
    # Sample story 2 - narrative heavy
    sample_story2 = '''
    The ancient library stood silent in the moonlight, its towering shelves casting long shadows across the marble floor. Elena had been searching for three days, turning page after page of forgotten manuscripts, looking for the answer that would save her kingdom.
    
    The prophecy spoke of a warrior who would rise when darkness threatened to consume the land. But Elena was no warrior – she was just a scholar, a keeper of books and ancient knowledge. Yet here she was, the only one left who could decipher the old texts.
    
    Her fingers traced the faded ink of a particularly old tome. The language was archaic, nearly impossible to read, but she persevered. Each symbol told a story, each word held power that had been dormant for centuries.
    
    As dawn approached, Elena finally found what she was looking for. The ritual was complex, requiring precise timing and unwavering focus. She gathered the necessary materials: silver dust from the highest mountain, water from the deepest well, and most importantly, a sacrifice of something precious.
    
    Elena looked at her grandmother's ring, the only thing she had left of her family. It would have to do.
    '''
    
    # Initialize analyzer
    analyzer = StoryDataAnalyzer()
    
    # Analyze both stories
    print("STORY ANALYSIS DEMO")
    print("=" * 50)
    
    analysis1 = analyzer.analyze_story(sample_story1, "The Confrontation")
    analysis2 = analyzer.analyze_story(sample_story2, "The Ancient Library")
    
    # Compare them
    analyzer.compare_stories([analysis1, analysis2])
    
    print(f"\n✅ Analysis complete! Try it with your own stories.")
    print("💡 To use: analyzer.analyze_story(your_story_text, 'Your Story Title')")


if __name__ == "__main__":
    test_analyzer()

    

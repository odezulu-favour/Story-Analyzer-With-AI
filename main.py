from storyanalyzer import StoryDataAnalyzer


analyzer = StoryDataAnalyzer()

text = """
Once upon a time in a small village, there lived a young whose name was Alice. She loved exploring the nearby forest and often spent her afternoons wandering through the trees. One day, she stumbled upon a hidden glade where she met a talking rabbit named Benny. Benny told her about a magical spring that granted wishes, but it was guarded by a fierce dragon named Draco."""

# analysis = analyzer.basic_metrics(text)
# print("Basic Metrics:", analysis)


charater = analyzer.extract_characters(text)
print("Characters:", charater)

print(analyzer.calculate_dialogue_ratio(text))
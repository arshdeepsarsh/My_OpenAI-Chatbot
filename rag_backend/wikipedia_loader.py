import wikipedia
def get_wikipedia_summary(query):
    try:
        summary = wikipedia.summary(query, sentences=5)
    except wikipedia.exceptions.DisambiguationError as e:
        summary = wikipedia.summary(e.options[0], sentences=5)
    except Exception as e:
        summary = "Wikipedia article not found."
    return summary


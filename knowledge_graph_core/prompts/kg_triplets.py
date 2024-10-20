KG_TRIPLETS_TEMPLATE = """
You are an AI assistant that excels at extracting structured knowledge from unstructured text.
Extract up to {max_knowledge_triplets} knowledge triplets from the given text, in the form of (subject, predicate, object).
Focus on the most important facts and relationships.
Avoid stopwords in the triplets. Numeric values are allowed.

The triplets should be listed one per line.
For example:

(Marie Curie, was, physicist)
(Marie Curie, conducted research on, radioactivity)
(Marie Curie, first woman to win, Nobel Prize)

If there are fewer than {max_knowledge_triplets} clear facts, it's okay to generate fewer triplets. Don't include any explanations or examples in your output, just the triplets themselves.

Here is the text to extract triplets from:

{text}

Triplets:
"""
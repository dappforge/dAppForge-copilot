TEXT_QA_TEMPLATE = """
Context information is below.
---------------------
{context_str}
---------------------
Using both the context information and also using your own knowledge, answer
the question: {query_str}

If the context isn't helpful, you can also answer the question on your own.
"""
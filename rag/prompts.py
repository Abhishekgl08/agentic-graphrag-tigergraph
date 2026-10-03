SYSTEM_INSTRUCTIONS = """Answer only from the supplied evidence. Do not use outside knowledge or invent facts. If the evidence is insufficient, say so plainly. Respect dates, comparisons, counts, and aggregation constraints in the question. Cite supporting evidence using [Source N]."""


def answer_input(question: str, evidence: str) -> str:
    return f"Question:\n{question}\n\nEvidence:\n{evidence}\n\nAnswer:" 

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import RAG_TOP_K, get_judge_llm, get_llm
from app.rag.retrieval import (_get_index, format_source, keyword_search,
                               reciprocal_rank_fusion, retrieve, rewrite_query, vector_search, hybrid_search)

EVAL_DIR = Path("data") / "eval"
TESTSET_PATH = EVAL_DIR / "testset.json"

# Questions your documents should NOT answer. Edit these to suit your library.
OUT_OF_SCOPE_QUESTIONS = [
    "Who won the 2022 FIFA World Cup?",
    "What is the capital of Australia?",
    "How do I make a margherita pizza?",
]


# ===================== Test set generation =====================

class GeneratedQA(BaseModel):
    usable: bool = Field(description="False if the passage has no substantive, self-contained "
                                     "information (only formulas, fragments, a table of contents...)")
    question: str = Field(
        description="A natural question a student might ask, which this passage answers")
    answer: str = Field(
        description="A short, correct answer based only on the passage")


GENERATE_PROMPT = """You create test questions for evaluating a document search system.
Given one passage, write ONE question that this passage answers well, plus its answer.
- Ask about the passage's main idea, not a minor detail.
- Paraphrase: a real user would not know the passage's exact wording.
- The question must make sense on its own. Never say "this passage", "the author" or "the figure".
If the passage has no substantive content, set usable to false."""


def generate_testset(n: int = 20, seed: int = 42) -> list[dict]:
    """Create n questions from random passages, spread across all documents."""
    by_id = _get_index()["by_id"]
    # fixed seed: the same passages are chosen every time
    rng = random.Random(seed)

    # Group substantial passages by document, so every document gets tested
    per_doc: dict[str, list[str]] = {}
    for chunk_id, (text, meta) in by_id.items():
        if len(text) > 400:
            per_doc.setdefault(meta["doc_id"], []).append(chunk_id)
    for ids in per_doc.values():
        rng.shuffle(ids)

    generator = get_llm().with_structured_output(GeneratedQA)
    testset = []
    while len(testset) < n and any(per_doc.values()):
        for ids in per_doc.values():  # take turns between documents
            if not ids or len(testset) >= n:
                continue
            chunk_id = ids.pop()
            text, meta = by_id[chunk_id]
            qa: GeneratedQA = generator.invoke(
                [("system", GENERATE_PROMPT), ("human", text)])  # type: ignore
            if not qa.usable:
                continue
            testset.append({
                "question": qa.question,
                "reference_answer": qa.answer,
                "chunk_id": chunk_id,
                "doc_id": meta["doc_id"],
                "page": meta["page"],
                "source": format_source({"filename": meta["filename"], "page": meta["page"]}),
            })
            print(
                f"   {len(testset):>2}. [{testset[-1]['source']}] {qa.question}")

    for question in OUT_OF_SCOPE_QUESTIONS:
        testset.append({"question": question, "reference_answer": None, "chunk_id": None,
                        "doc_id": None, "page": None, "source": None})

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    TESTSET_PATH.write_text(json.dumps(testset, indent=2), encoding="utf-8")
    return testset


def load_testset() -> list[dict]:
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


# ===================== Part 1: Retrieval =====================

METHODS = ["vector", "hybrid", "hybrid + rerank",
           "hybrid + rewrite", "full pipeline", "final setup"]


def _is_relevant(chunk_id: str, item: dict, by_id: dict) -> bool:
    """Correct if it's the source passage, or another passage from the same PDF page."""
    if chunk_id == item["chunk_id"]:
        return True
    meta = by_id[chunk_id][1]
    return item["page"] > 0 and meta["doc_id"] == item["doc_id"] and meta["page"] == item["page"]


def _ranked_ids(method: str, question: str, queries: list[str], k: int) -> list[str]:
    def ids(chunks):
        return [c["id"] for c in chunks]

    if method == "vector":
        return vector_search(question, k)
    if method == "hybrid":
        return hybrid_search([question])[:k]
    if method == "hybrid + rerank":
        return ids(retrieve(question, k, queries=[question], min_relevance=0.0, rerank_mode="rank"))
    if method == "hybrid + rewrite":
        return hybrid_search(queries)[:k]
    if method == "full pipeline":
        return ids(retrieve(question, k, queries=queries, min_relevance=0.0, rerank_mode="rank"))
    # Final setup: corrected question only, hybrid order, reranker as filter (the config defaults)
    return ids(retrieve(question, k, queries=queries[:1], min_relevance=0.0))


def evaluate_retrieval(testset: list[dict], k: int = RAG_TOP_K) -> dict:
    by_id = _get_index()["by_id"]
    # only questions with a known answer
    questions = [t for t in testset if t["chunk_id"]]
    totals = {m: {"hit@1": 0.0, f"hit@{k}": 0.0, "mrr": 0.0} for m in METHODS}

    for i, item in enumerate(questions, start=1):
        # once per question, shared by methods 3 and 4
        queries = rewrite_query(item["question"])
        for method in METHODS:
            ids = _ranked_ids(method, item["question"], queries, k)
            rank = next((r for r, cid in enumerate(ids, start=1)
                         if cid in by_id and _is_relevant(cid, item, by_id)), None)
            if rank:
                totals[method][f"hit@{k}"] += 1
                totals[method]["mrr"] += 1 / rank
                if rank == 1:
                    totals[method]["hit@1"] += 1
        print(f"   {i}/{len(questions)} questions done", end="\r")

    n = len(questions)
    return {m: {metric: round(value / n, 3) for metric, value in scores.items()}
            for m, scores in totals.items()}


# ===================== Part 2: Answers =====================

ANSWER_PROMPT = """Answer the question using ONLY the passages below.
- Base your answer on the specific sentences that address the question.
- If the passages contain relevant information, answer with it, even if it only partly
  answers the question, and say what is missing.
- For comparisons ("A vs B", "better / worse", "more / less"), state the direction exactly
  as the passages do. Double-check you have not reversed it.
- Cite pages like (file, p. N).
- Reply exactly "Not found in the documents." ONLY if no passage is relevant at all.

=== PASSAGES ===
{passages}"""


class Judgement(BaseModel):
    faithfulness: int = Field(description="1-5: is every claim in the answer supported by the "
                                          "retrieved passages? 5 = fully supported, 1 = mostly not")
    relevance: int = Field(
        description="1-5: does the answer directly address the question?")
    correctness: int = Field(
        description="1-5: does the answer agree with the reference answer?")
    explanation: str = Field(
        description="One or two sentences justifying the scores")


JUDGE_PROMPT = """You are a strict evaluator of a question-answering system that answers from documents.
Score the answer on faithfulness, relevance and correctness (1-5 each).
- Faithfulness: penalise ANY claim that is not supported by the retrieved passages,
  even if it is true in general.
- If the reference answer is NONE, the documents do not contain the answer. The correct
  behaviour is to say it was not found: score that 5 on all three; a made-up answer scores 1."""


def answer_question(question: str) -> tuple[str, list[dict]]:
    chunks = retrieve(question)
    if not chunks:
        return "Not found in the documents.", []
    passages = "\n\n".join(
        f"[{format_source(c)}]\n{c['text']}" for c in chunks)
    response = get_llm().invoke([("system", ANSWER_PROMPT.format(passages=passages)),
                                 ("human", question)])
    return response.content, chunks  # type: ignore


def evaluate_answers(testset: list[dict]) -> dict:
    judge = get_judge_llm().with_structured_output(Judgement)
    rows = []

    for i, item in enumerate(testset, start=1):
        answer, chunks = answer_question(item["question"])
        passages = "\n\n".join(
            f"[{format_source(c)}]\n{c['text']}" for c in chunks) or "(none)"
        reference = item["reference_answer"] or "NONE (the documents do not contain this)"

        verdict: Judgement = judge.invoke([
            ("system", JUDGE_PROMPT),
            ("human", f"Question: {item['question']}\n\nReference answer: {reference}\n\n"
             # type: ignore
             f"Retrieved passages:\n{passages}\n\nAnswer to evaluate:\n{answer}"),
        ])
        rows.append({
            "question": item["question"],
            "in_scope": item["chunk_id"] is not None,
            # Abstained if retrieval found nothing OR the LLM said it couldn't find the answer
            "abstained": not chunks or "not found in the documents" in answer.lower(),
            "answer": answer,
            **verdict.model_dump(),
        })
        print(f"   {i}/{len(testset)} answers graded", end="\r")

    in_scope = [r for r in rows if r["in_scope"]]
    out_scope = [r for r in rows if not r["in_scope"]]

    def avg(key):
        return round(sum(r[key] for r in in_scope) / len(in_scope), 2) if in_scope else 0.0

    summary = {
        "faithfulness": avg("faithfulness"),
        "relevance": avg("relevance"),
        "correctness": avg("correctness"),
        # Out-of-scope questions where it correctly found nothing
        "correct_abstentions": f"{sum(r['abstained'] for r in out_scope)}/{len(out_scope)}",
        # In-scope questions where it wrongly found nothing (threshold too strict?)
        "false_abstentions": f"{sum(r['abstained'] for r in in_scope)}/{len(in_scope)}",
    }
    return {"summary": summary, "rows": rows}

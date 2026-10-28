import json
import sys
from datetime import datetime

from app.config import RAG_TOP_K
from app.rag.evaluation import (EVAL_DIR, TESTSET_PATH, evaluate_answers,
                                evaluate_retrieval, generate_testset, load_testset)

# Reuse the saved test set so results are comparable between runs; use --new to regenerate
if "--new" in sys.argv or not TESTSET_PATH.exists():
    print("🧪 Generating a test set from your documents...")
    testset = generate_testset(n=20)
else:
    testset = load_testset()
    print(f"🧪 Using the saved test set ({len(testset)} questions). Use --new to regenerate.")

print("\n📊 PART 1: Retrieval quality")
retrieval = evaluate_retrieval(testset)
k = RAG_TOP_K
print(f"\n   {'Method':<20}{'Hit@1':>8}{f'Hit@{k}':>8}{'MRR':>8}")
for method, s in retrieval.items():
    print(f"   {method:<20}{s['hit@1']:>8.2f}{s[f'hit@{k}']:>8.2f}{s['mrr']:>8.2f}")

print("\n🧑‍⚖️ PART 2: Answer quality (scores out of 5)")
answers = evaluate_answers(testset)
for metric, value in answers["summary"].items():
    print(f"   {metric:<22}{value}")

# The weakest answers are where to look for improvements
worst = sorted((r for r in answers["rows"] if r["in_scope"]),
               key=lambda r: r["faithfulness"] + r["correctness"])[:3]
print("\n🔍 Weakest answers:")
for r in worst:
    print(f"   • {r['question']}\n     F={r['faithfulness']} C={r['correctness']}: {r['explanation']}")

EVAL_DIR.mkdir(parents=True, exist_ok=True)
path = EVAL_DIR / f"results-{datetime.now():%Y%m%d-%H%M}.json"
path.write_text(json.dumps({"retrieval": retrieval, "answers": answers}, indent=2),
                encoding="utf-8")
print(f"\n💾 Full results saved to {path}")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy_retriever import PolicyRetriever


def main():
    retriever = PolicyRetriever()

    queries = [
        "What is the waiting period for cataract treatment?",
        "What is the coverage limit for cataract surgery?",
        "Which clauses discuss pre-existing diseases and waiting periods?",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUESTION: {query}")
        print("=" * 80)

        results = retriever.search(query, top_k=5)

        for result in results:
            print(
                f"\nPage {result['page_number']} | "
                f"Score: {result['score']}"
            )
            print(f"Chunk: {result['chunk_id']}")
            print(result["text"])


if __name__ == "__main__":
    main()

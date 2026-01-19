from src.index.search.searcher import search
if __name__ == "__main__":
    query = "선박의 연료 소비 최소화 관련 내용"

    results = search(query, top_k=5, oversample=3)

    for r in results:
        print("=" * 80)
        print(f"Score: {r['score']:.4f}")
        print(f"File: {r['file_path']}")
        print("Text snippet:")
        print(r["text"][:300])
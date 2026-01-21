import argparse
import sys
from database import FactorizationDB
from search_engine import FunctionSearchEngine

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Automatic Factorization Function Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Run quick search on 20 functions
  python main.py --search --functions 20

  # Show top 10 functions
  python main.py --best --limit 10

  # Analyze specific function
  python main.py --analyze heuristic_pollard_v2
        """
    )

    parser.add_argument("--search", action="store_true", help="Run new search")
    parser.add_argument("--functions", type=int, default=50, help="Number of functions to test")
    parser.add_argument("--best", action="store_true", help="Show best functions")
    parser.add_argument("--limit", type=int, default=10, help="Number of functions to display")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum score threshold")
    parser.add_argument("--analyze", type=str, help="Analyze specific function")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--export", type=str, help="Export results to CSV file")
    parser.add_argument("--genetic", action="store_true", help="Run genetic evolution")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations in genetic evolution")
    parser.add_argument("--time-limit", type=int, default=30, help="Time limit for testing")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--large-numbers", action="store_true", help="Include larger test numbers")

    args = parser.parse_args()

    if not any([args.search, args.best, args.analyze, args.stats, args.export, args.genetic]):
        parser.print_help()
        sys.exit(1)

    db = FactorizationDB()
    engine = FunctionSearchEngine(db)

    if args.search:
        print("=" * 60)
        print("Starting Advanced Factorization Function Search")
        print("=" * 60)
        engine.run_batch_search(args.functions, time_limit=args.time_limit, use_parallel=args.parallel)
        print("\nSearch completed!")

    if args.genetic:
        print("=== Starting Genetic Evolution ===")
        best_individual = engine.genetic_evolution(generations=args.generations)
        print(f"Best genetic function: {best_individual['name']} with score {best_individual['fitness']:.3f}")
        print("=== Genetic Evolution Completed ===")

    if args.best:
        print("\n" + "=" * 60)
        print(f"Top {args.limit} Functions (filtered by F1 >= {args.min_score})")
        print("=" * 60)

        results = db.get_best_functions(min_score=args.min_score, limit=args.limit)

        if not results:
            print("No functions meet the criteria")
            return

        for idx, (name, code, avg_score, avg_precision, avg_recall, count, total_tp, total_fp) in enumerate(results, 1):
            print(f"\n{idx}. {name}")
            print(f"   Average F1: {avg_score:.3f} | Precision: {avg_precision:.1%} | Recall: {avg_recall:.1%}")
            print(f"   Tests: {count} | TP: {total_tp} | FP: {total_fp}")

            # Show code preview
            lines = code.split('\n')[:5]
            preview = '\n'.join(f"      {line.encode('utf-8', errors='ignore').decode('utf-8')}" for line in lines)
            try:
                print(f"   Code:\n{preview}")
            except UnicodeEncodeError:
                print(f"   Code: [Could not display code preview]")

    if args.analyze:
        print(f"\nAnalyzing function: {args.analyze}")
        stats = db.get_function_stats(args.analyze)

        if not stats or stats[0] is None:
            print(f"Function '{args.analyze}' does not exist or was not tested")
            return

        avg_score, min_score, max_score, avg_precision, avg_recall, tests = stats
        print(f"Average F1: {avg_score:.3f} (Min: {min_score:.3f}, Max: {max_score:.3f})")
        print(f"Precision: {avg_precision:.1%} | Recall: {avg_recall:.1%}")
        print(f"Number of tests: {tests}")

    if args.stats:
        print("\n=== Database Statistics ===")
        stats = db.get_statistics()
        print(f"Number of functions: {stats['total_functions']}")
        print(f"Number of results: {stats['total_results']}")
        print(f"Average score: {stats['average_score']:.3f}")
        if stats['best_function']:
            print(f"Best function: {stats['best_function'][0]} with score {stats['best_function'][1]:.3f}")
        print("\nScore distribution:")
        for category, count in stats['score_distribution']:
            print(f"  {category}: {count} results")

    if args.export:
        print(f"\n=== Exporting results to {args.export} ===")
        result = db.export_to_csv(args.export)
        print(result)

if __name__ == "__main__":
    main()
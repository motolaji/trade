from trademark_analyzer import TrademarkAnalyzer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Trademark Logo Analysis System')
    parser.add_argument('--action', choices=['build', 'search', 'analyze'],
                        required=True, help='Action to perform')
    parser.add_argument('--dataset', help='Dataset directory for building index')
    parser.add_argument('--query', help='Query image path for search')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of similar images to find')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = TrademarkAnalyzer()

    try:
        if args.action == 'build':
            if not args.dataset:
                raise ValueError("--dataset required for build action")
            analyzer.build_dataset_index(args.dataset)

        elif args.action == 'search':
            if not args.query:
                raise ValueError("--query required for search action")
            analyzer.load_index()
            results, viz_path = analyzer.find_similar(args.query, args.k)

            print("\nSearch Results:")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Path: {result['path']}")
                print(f"Similarity: {result['similarity']:.3f}")
            print(f"\nVisualization saved to: {viz_path}")

        elif args.action == 'analyze':
            if not args.query:
                raise ValueError("--query required for analyze action")
            features, viz_path = analyzer.analyze_logo(args.query)
            print(f"Analysis visualization saved to: {viz_path}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
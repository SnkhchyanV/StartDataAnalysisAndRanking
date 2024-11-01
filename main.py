import argparse
from preprocessing import Preprocessor
from rating_script import StartupRatingTool

# Define the available parameters and their default weights
PARAMETERS = {
    'Industry': 0,
    'Latest Funding Round': 0,
    'Latest Funding Date': 0,
    'Total Funding': 0,
    'All Investors': 0,
    'Latest Valuation': 0,
    'Latest Funding Amount': 0,
    'Exit Round': 0,
    'Exit Date': 0,
    'Acquirers': 0,
    'Mosaic (Overall)': 0,
    'Date Added': 0,
    'Applications': 0,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to specify data path and parameter weights")

    # Argument for the data path
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data file"
    )

    # Arguments for parameter weights
    for param in PARAMETERS:
        parser.add_argument(
            f"--{param.replace(' ', '_')}",
            type=float,
            default=PARAMETERS[param],
            help=f"Weight for {param} (default: {PARAMETERS[param]})"
        )

    args = parser.parse_args()

    # Convert arguments back to dictionary format with their respective weights
    weights = {param: getattr(args, param.replace(' ', '_')) for param in PARAMETERS}
    
    return args.data_path, weights

if __name__ == "__main__":
    # Parse arguments
    data_path, parameter_weights = parse_arguments()
    print(f"Data path: {data_path}")

    # Initialize and apply preprocessing
    preprocessor = Preprocessor(data_path)
    preprocessor.apply_strategy()  # Apply the required strategy based on your preprocessing needs

    # Retrieve processed data with scores
    data = preprocessor.get_data_with_scores()

    # Initialize the rating tool with the data
    rater = StartupRatingTool(data)


    print("Set parameters and their weights:")
    for param, weight in parameter_weights.items():
        print(f"{param}: {weight}")

    print('\nParemeters are Set\n\n')
    # Add parameters and their weight so the rating tool
    rater.set_weights(**parameter_weights)
    
    # Perform the rating computation
    rater.rerank()

    # Display 
    rater.display_ranking()


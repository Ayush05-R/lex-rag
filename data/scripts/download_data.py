from datasets import load_dataset  # To load the dataset
from pathlib import Path  # To handle file paths

RAW_DIR = (
    Path(__file__).resolve().parents[1] / "raw"
)  # Directory to store the raw dataset
RAW_DIR.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist


def main():
    # Load the dataset from Hugging Face
    dataset = load_dataset(
        "Exploration-Lab/IL-TUR", "cjpe"
    )  # Load the dataset with the specified configuration

    # This dataset has multiple splits (e.g., train, validation, test), so we will save each split separately
    for (
        split_name,
        split_data,
    ) in dataset.items():  # Iterate over each split in the dataset
        out_path = (
            RAW_DIR / f"cjpe_dataset_{split_name}.csv"
        )  # Define the output path for the split
        split_data.to_csv(out_path)  # Save the split data to a CSV file
        print(
            f"Saved {split_name}: {len(split_data)} rows -> {out_path}"
        )  # Print a message indicating the number of rows saved


if __name__ == "__main__":
    main()  # Call the main function to execute the script

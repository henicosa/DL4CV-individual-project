import argparse
import os
import csv

from dlcv.utils import plot_multiple_losses_and_accuracies

def plot_notebook(args):

    argv = []
    # iterate over the dictionary and add the key and value to the argv list
    for key, value in args.items():
        argv.append("--" + key.lower())
        argv.append(str(value))

    # call the main function with the argv list
    try: 
        main(parse_args(argv))
    except SystemExit as e:
        print(argv)
        # print the error message
        print("Error in the arguments")
        print(e)



def read_csv_data(filepath):
    """
    Reads CSV data from a file and extracts epoch-wise train losses, test losses, and test accuracies.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        tuple: Returns train_losses, test_losses, test_accuracies lists.
    """
    train_losses, test_losses, test_accuracies = [], [], []
    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            train_losses.append(float(row['Train Loss']))
            test_losses.append(float(row['Test Loss']))
            test_accuracies.append(float(row['Test Accuracy']))
    return train_losses, test_losses, test_accuracies


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Plot training and testing results from CSV files.')
    parser.add_argument('--folder', type=str, default="./results", help='Folder containing the CSV files.')
    parser.add_argument('--exclude', type=str,
                        help='Comma-separated string patterns to exclude files from being plotted.')
    return parser.parse_args(argv)


def main(args):
    """
    Main function to handle the plotting of training and testing results from multiple CSV files.
    It filters out files based on exclusion patterns and aggregates data for plotting.

    Args:
        args (Namespace): Parsed command line arguments with 'folder' and 'exclude' options.
    """
    model_data_list = []
    exclusion_patterns = args.exclude.split(',') if args.exclude else []

    for filename in os.listdir(args.folder):
        if filename.endswith(".csv") and not any(excl in filename for excl in exclusion_patterns):
            full_path = os.path.join(args.folder, filename)
            train_losses, test_losses, test_accuracies = read_csv_data(full_path)
            model_data = {
                'name': filename.replace('.csv', ''),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'test_accuracies': test_accuracies
            }
            model_data_list.append(model_data)

    plot_multiple_losses_and_accuracies(model_data_list) # ToDo <- add this function in utils.py


if __name__ == '__main__':
    args = parse_args()
    main(args)
import pandas as pd
import features_generation
import csv
import os

DIRECTORY = './GTZAN/genres_original'
ORIGIN_FILE = './GTZAN/features_30_sec.csv'
CSV_FILE = 'generated_features.csv'


def calculate_features(input_dir, output_file):

    csv_file = open(output_file, 'w')
    writer = csv.writer(csv_file)

    for d in os.listdir(input_dir):
        print(d)
        for file in os.listdir('/'.join([DIRECTORY, d])):
            try:
                row = features_generation.get_all_features_from_path('/'.join([DIRECTORY, d, file]))
            except Exception:
                print('Exception: ' + file)
                continue
            else:
                if row:
                    writer.writerow(row)
                else:
                    print(file)

    csv_file.close()


def main():
    # calculate_features(input_dir=DIRECTORY, output_file=CSV_FILE)

    df_given = pd.read_csv(ORIGIN_FILE)
    df_calculated = pd.read_csv(CSV_FILE, header=False)


if __name__ == '__main__':
    main()

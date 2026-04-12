import os
import csv
from PIL import Image # Keeping this import as it's part of your context

class ImageCSVFilter:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        if not os.path.isdir(self.image_dir):
            print(f"Warning: Image directory '{self.image_dir}' does not exist.")

    def filter_and_process_csv(self, input_csv_path, output_csv_path=None):
        valid_rows = []
        skipped_count = 0
        total_rows_read = 0

        print(f"Starting filter process for '{input_csv_path}'...")

        try:
            with open(input_csv_path, mode='r', newline='') as infile:
                reader = csv.reader(infile)

                # If your CSV has a header, you might want to uncomment these lines
                # header = next(reader, None)
                # if header:
                #     valid_rows.append(header)

                for row_num, row in enumerate(reader, 1):
                    total_rows_read += 1
                    if not row:
                        # print(f"  Skipping empty row at line {row_num}.") # Uncomment for verbose
                        continue

                    image_relative_path_in_csv = row[0].strip()
                    img_filename = os.path.basename(image_relative_path_in_csv)
                    image_full_path = os.path.join(self.image_dir, img_filename)

                    if os.path.exists(image_full_path) and    Image.open(image_full_path).convert("RGB"):
                        valid_rows.append(row)
                        # print(f"  Found image: {image_full_path}") # Uncomment for verbose
                    else:
                        print(f"  Warning: Image '{img_filename}' (from '{image_relative_path_in_csv}') not found at '{image_full_path}' (CSV line {row_num}). Skipping entry.")
                        skipped_count += 1
        except FileNotFoundError:
            print(f"Error: Input CSV file '{input_csv_path}' not found.")
            return []
        except Exception as e:
            print(f"An error occurred while reading the CSV: {e}")
            return []

        print(f"\nFiltering complete. Total entries read: {total_rows_read}")
        print(f"Valid entries (image exists): {len(valid_rows)}")
        print(f"Skipped entries (image missing): {skipped_count}")

        # This is the core part for saving to a new CSV:
        if output_csv_path:
            try:
                with open(output_csv_path, mode='w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(valid_rows) # Writes all collected valid_rows
                print(f"Filtered data successfully written to: {output_csv_path}")
            except Exception as e:
                print(f"Error writing filtered CSV to '{output_csv_path}': {e}")

        return valid_rows


def main():

    my_image_directory = '../synthetic_all/images/'
    my_input_csv = '../train_synth.csv'
    my_output_csv = '../train_synth_fil.csv' # This is your NEW CSV file
    filter_instance = ImageCSVFilter(my_image_directory)

    # Call the filter method, providing the input CSV and the desired output CSV path
    # The filtered data will be WRITTEN to 'temp_filtered_output.csv'
    filtered_results = filter_instance.filter_and_process_csv(
        input_csv_path=my_input_csv,
        output_csv_path=my_output_csv # <--- This is where you specify the new CSV file!
    )



main()

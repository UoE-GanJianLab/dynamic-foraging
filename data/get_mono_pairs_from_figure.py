# from the figure name, produce a csv file called mono_pairs.csv
# containing the following columns:
#   - session
#   - str
#   - pfc


import os
import re
import csv
import sys

# Get the directory name from the command line argument
if len(sys.argv) < 2:
    print('Usage: python get_mono_pairs_from_figure.py <directory>')
    sys.exit(1)
figure_dir = sys.argv[1]

# Define the regular expression pattern to extract the session, str, and pfc names
pattern = r'(\w+)_str_(\d+)_pfc_(\d+)\.png'

# Create a list to store the mono pairs
mono_pairs = []

# Iterate over the files in the directory
for filename in os.listdir(figure_dir):
    # Check if the file is a PNG image
    if filename.endswith('.png'):
        # Extract the session, str, and pfc names from the filename
        match = re.match(pattern, filename)
        if match:
            session = match.group(1)
            str_name = 'str_' + match.group(2)
            pfc_name = 'pfc_' + match.group(3)
            # Add the mono pair to the list
            mono_pairs.append((session, str_name, pfc_name))

# Write the mono pairs to a CSV file
with open('mono_pairs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['session', 'str', 'pfc'])
    writer.writerows(mono_pairs)
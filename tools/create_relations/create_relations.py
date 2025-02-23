import argparse
import json


def create_relations(file1_path, file2_path, output_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    min_length = min(len(lines1), len(lines2))
    relations = {lines1[i].strip(): lines2[i].strip() for i in range(min_length)}

    with open(output_path, 'w') as output_file:
        json.dump(relations, output_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create JSON relations from two text files')
    parser.add_argument('file1', type=str, help='Path to the first input text file')
    parser.add_argument('file2', type=str, help='Path to the second input text file')
    parser.add_argument('output', type=str, help='Path to the output JSON file')

    args = parser.parse_args()
    create_relations(args.file1, args.file2, args.output)

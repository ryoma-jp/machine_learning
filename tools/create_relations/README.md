# create_relations Tool

## Description
The `create_relations` tool is designed to establish relationships between different entities in a dataset. This tool can be used to create and manage relationships in a structured manner, making it easier to analyze and visualize data.

## Installation
To use the `create_relations` tool, ensure that you have the necessary dependencies installed. You can install the required packages using the following command:

```sh
pip install -r requirements.txt
```

## Usage
To run the `create_relations` tool, navigate to the `tools/create_relations` directory and execute the main script with the required arguments. Below is an example command to run the tool:

```sh
python create_relations.py file1 file2 output
```

## Configuration
The tool can be configured using a configuration file located in the `config` directory. Ensure that the configuration file is properly set up before running the tool.

## Example
Here is an example of how to use the `create_relations` tool:

1. Prepare your dataset and place it in the `input` directory.
2. Configure the tool using the configuration file in the `config` directory.
3. Run the tool using the command:

    ```sh
    python create_relations.py input/file1.txt input/file2.txt output/result.json
    ```

4. The output will be generated in the `output` directory.

## Docker

### Build Docker Image
To build the Docker image for the `create_relations` tool, navigate to the root directory of the project and run the following command:

```sh
docker build -t create_relations:latest -f Dockerfile .
```

### Run Docker Container
To run the `create_relations` tool inside a Docker container, use the following command:

```sh
docker run --rm -v $(pwd)/tools/create_relations/input:/app/input -v $(pwd)/tools/create_relations/output:/app/output create_relations:latest input/file1.txt input/file2.txt output/result.json
```

This command mounts the `input` and `output` directories to the container, allowing the tool to access the dataset and save the results.

## License
This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

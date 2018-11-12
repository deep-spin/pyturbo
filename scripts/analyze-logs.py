import argparse
import re
import json

"""
Script to analyze logs produced by the Turbo Parser
"""


def extract_data(filename):
    """
    Extract the best validation loss from the log file
    """
    expecting_best_value = False
    loss = None

    with open(filename, 'r') as f:
        config_line = next(f)
        config_line = config_line.replace("'", '"').replace('None', 'null') \
            .replace('False', 'false').replace('True', 'true')
        config_data = json.loads(config_line)

        for line in f:
            if 'Saved model' in line:
                expecting_best_value = True

            elif 'Validation Loss' in line and expecting_best_value:
                match = re.search('Validation Loss: ([\d.]+)\s', line)
                loss = float(match.group(1))
                expecting_best_value = False

    return config_data, loss


def create_output_line(config_data, loss):
    """
    Write a TSV line for the output with the model configuration and loss
    """
    fields = [config_data['embedding_size'],
              config_data['tag_embedding_size'],
              config_data['learning_rate'],
              config_data['rnn_size'],
              config_data['mlp_size'],
              config_data['dropout'],
              config_data['num_layers'],
              loss]
    line = '\t'.join(str(field) for field in fields)
    return line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', help='Log files', nargs='+')
    args = parser.parse_args()

    output_lines = []
    for name in args.logs:
        config, loss = extract_data(name)
        print(create_output_line(config, loss))

"""
forward propagator

Usage:
    forward_propagator.py -i INPUTS -w WEIGHTS

Options:
    -h, --help  show this help message
    -i INPUTS   Comma separated list of inputs
    -w WEIGHTS  Comma separated list of weights

"""
import docopt
from operator import mul

def forward_propagator(inputs, weights):
    if ((len(weights) - len(inputs)) % pow(len(inputs), 2)) != 0:
        raise Exception("Incorrect number of weights provided.")

    dot = lambda X, Y: sum(map(lambda x, y: x * y, X, Y))

    weights_per_epoch = pow(len(inputs), 2)

    num_epochs = (len(weights) - len(inputs)) / weights_per_epoch
    node_output_weights = weights[-len(inputs):]

    print("weights per epoch={}, number of epochs={}".format(weights_per_epoch, num_epochs))
    node_inputs = inputs

    for epoch in range(0, num_epochs):
        epoch_weights = weights[epoch * weights_per_epoch:(epoch + 1) * weights_per_epoch]
        node_input_weights = [epoch_weights[i:i + len(inputs)] for i in range(0, len(epoch_weights), len(inputs))]
        print("epoch: {}".format(epoch))
        print("node inputs: {}".format(node_inputs))
        print("epoch weights: {}".format(epoch_weights))
        print("node input weights: {}".format(node_input_weights))

        sums = [dot(node_weights, node_inputs) for node_weights in node_input_weights]
        print("sums: {}".format(sums))

        print("")

        node_inputs = sums
    return dot(node_output_weights, node_inputs)

if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    args['-i'] = [int(x) for x in args['-i'].split(',')]
    args['-w'] = [int(x) for x in args['-w'].split(',')]
    print(forward_propagator(inputs=args['-i'], weights=args['-w']))

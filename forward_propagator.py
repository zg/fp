"""
forward propagator

Usage:
    forward_propagator.py -i INPUTS -w WEIGHTS [--a=<ACTUAL>] [--act=<ACTIVATOR>]

Options:
    -h, --help        Show this help message
    -i INPUTS         Comma separated list of inputs
    -w WEIGHTS        Comma separated list of weights
    --a=<ACTUAL>      The actual value
    --act=<ACTIVATOR> The activator function
"""
import docopt

activators = {
    'identity': lambda x: x,
    'relu': lambda x: max(0, x),
    'binstep': lambda x: 0 if x < 0 else 1,
    'logistic': lambda x: 1 / (1 + math.exp(-x)),
    'softsign': lambda x: x / (1 + abs(x)),
    'leakyrelu': lambda x: 0.01 * x if x < 0 else x
}

def forward_propagator(inputs, weights, actual=0, activator='relu'):
    if ((len(weights) - len(inputs)) % pow(len(inputs), 2)) != 0:
        raise Exception("Incorrect number of weights provided.")

    try:
        activator = activators[activator]
    except KeyError:
        activator = activators['identity']

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

        sums = [activator(dot(node_weights, node_inputs)) for node_weights in node_input_weights]
        print("sums: {}".format(sums))

        print("")

        node_inputs = sums
    ret = dot(node_output_weights, node_inputs)

    return ret - actual

if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    args['-i'] = [float(x) for x in args['-i'].split(',')]
    args['-w'] = [float(x) for x in args['-w'].split(',')]
    actual = float(args['--a']) if args['--a'] is not None else 0
    activator = args['--act'] if args['--act'] is not None else 'identity'
    print(forward_propagator(inputs=args['-i'], weights=args['-w'], actual=actual, activator=activator))

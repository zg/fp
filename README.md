# forward propagator

This script will calculate the forward propagation output for a given set of
inputs and weights.

```
$ virtualenv venv
$ . venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ python forward_propagator.py -i 1,1 -w 2,4,4,-5,0,1,1,1,5,1
weights per epoch=4, number of epochs=2
epoch: 0
node inputs: [1, 1]
epoch weights: [2, 4, 4, -5]
node input weights: [[2, 4], [4, -5]]
sums: [6, -1]

epoch: 1
node inputs: [6, -1]
epoch weights: [0, 1, 1, 1]
node input weights: [[0, 1], [1, 1]]
sums: [-1, 5]

0
(venv) $ 
```

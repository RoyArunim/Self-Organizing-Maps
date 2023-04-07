# Self-Organizing-Maps
Self Organizing maps is a kind of ANN inspired by neural networks of biological models of 1970s. It is analogical to KNN algorithms in many ways. Self Organizing Maps are different
from Neural Networks where they do not have any activation function. The weights assigned to the nodes themselves are characteristic of the node. Weights are like coordinates in
3D space. Euclidean distance is calculated and the node drags itself (like an amoeba) to the datapoint that is closest to the best matching unit, and the weights get updated.

In each subsequent iterations that radii decreases, adjusting the nodes specifically over the data.
Steps to train a Self Organizing Map:
1. We start with a dataset composed of n independent variables.
2. We create a grid composed of nodes, each one having a weight vector of n feature elements.
3. Randomly initialize the values of weight vectors to small numbers close to 0.
4. Select one random observation point.
5. Compute Euclidean distance from this point to the different neurons.
6. Select the neuron that has minimum distance to this point; this node is called winning node.
7. Update the weights of the winning node to move it closer to the point
8. Using a gaussian neighbourhood function of mean, the winning node, also update winning node neighbours ot move them closer to the point. This neighbourhood is called the Sigma of the Gaussian function.


﻿using NeuralNetwork.Generic.Connections;
using NeuralNetwork.Generic.Layers;
using Framework.Generic.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Generic.Neurons;
using NeuralNetwork.Generic.Datasets;

namespace NeuralNetwork.Generic.Networks
{    
    /// <summary>
    /// This interface represents a Deep Feed Forward (DFF) Neural Network.
    /// </summary>
    public interface IDFFNeuralNetwork : INeuralNetwork
    {
        /// <summary>
        /// Randomizes all neuron activation levels, biases, and connection weights.
        /// </summary>
        void RandomizeNetwork();

        /// <summary>
        /// Trains the network with the supplied <paramref name="trainingIterations"/>.
        /// </summary>
        /// <param name="trainingIterations">Contains the data entries that will be used to train this network.</param>
        /// <returns>Returns the training iterations with updated training costs.</returns>
        IEnumerable<INetworkTrainingIteration> Train(IEnumerable<INetworkTrainingIteration> trainingIterations);

        /// <summary>
        /// Applies the set of <paramref name="neuronInputs"/> to the input layer of the network and performs all calculations to generate an output.
        /// </summary>
        /// <param name="neuronInputs">A dictionary of inputs that will be applied to the network. The INeuron key values must contain ALL input neurons for the network.</param>
        IEnumerable<INetworkOutput> ApplyInputs(IEnumerable<INetworkInput> networkInputs);

        /// <summary>
        /// Returns the current neuron activation levels.
        /// </summary>
        /// <returns>Returns the current neuron activation levels as an enumerable.</returns>
        IEnumerable<INetworkOutput> GetNeuronOutputs();
    }

    /// <summary>
    /// This class represents a Deep Feed Forward (DFF) Neural Network.
    /// </summary>
    public class DFFNeuralNetwork : NeuralNetworkBase, IDFFNeuralNetwork
    {
        /// <summary>
        /// The learning rate of the network
        /// </summary>
        private const double _dampingRate = 0.1;
        
        public DFFNeuralNetwork()
        {
            Layers = new List<INetworkLayer>();
        }
        
        public DFFNeuralNetwork(IEnumerable<INetworkLayer> layers)
            : base(layers)
        {
            Layers = layers;
        }
        
        /// <summary>
        /// Initializes this network with input, output, and hidden layers where the layers contain the specified number of neurons.
        /// </summary>
        /// <param name="inputLayerNeuronCount">The number of neurons put in the input layer.</param>
        /// <param name="hiddenLayersCount">The number of hidden layers in the network.</param>
        /// <param name="hiddenLayerNeuronCount">The number of neurons put in each hidden layer.</param>
        /// <param name="outputLayerNeuronCount">The number of neurons put in the output layer.</param>
        public DFFNeuralNetwork(int inputLayerNeuronCount, int hiddenLayersCount, int hiddenLayerNeuronCount, int outputLayerNeuronCount)
        {
            InitializeNetwork(inputLayerNeuronCount, hiddenLayersCount, hiddenLayerNeuronCount, outputLayerNeuronCount);
            ValidateNetwork();
        }

        /// <summary>
        /// Initializes this network with input, output, and hidden layers where the layers contain the specified number of neurons.
        /// </summary>
        /// <param name="inputLayerNeuronCount">The number of neurons put in the input layer.</param>
        /// <param name="hiddenLayersCount">The number of hidden layers in the network.</param>
        /// <param name="hiddenLayerNeuronCount">The number of neurons put in each hidden layer.</param>
        /// <param name="outputLayerNeuronCount">The number of neurons put in the output layer.</param>
        private void InitializeNetwork(int inputLayerNeuronCount, int hiddenLayersCount, int hiddenLayerNeuronCount, int outputLayerNeuronCount)
        {
            var inputNeurons = new List<IInputNeuron>();
            var outputNeurons = new List<IOutputNeuron>();
            var hiddenLayers = new List<IHiddenLayer>();

            // Setup input layer
            for (int i = 0; i < inputLayerNeuronCount; i++)
                inputNeurons.Add(new InputNeuron() { Id = i });
            
            // Setup output layer
            for (int i = 0; i < outputLayerNeuronCount; i++)
                outputNeurons.Add(new OutputNeuron() { Id = inputLayerNeuronCount + (hiddenLayersCount * hiddenLayerNeuronCount) + i });

            // Setup hidden layer(s) with the appropriate number of neurons.
            for (int hlayer = 0; hlayer < hiddenLayersCount; hlayer++)
            {
                var hiddenLayerNeurons = new List<IHiddenNeuron>();
                for (int hlayerNeuron = 0; hlayerNeuron < hiddenLayerNeuronCount; hlayerNeuron++)
                {
                    var hiddenNeuronId = inputLayerNeuronCount + (hlayer * hiddenLayerNeuronCount) + hlayerNeuron;
                    var hiddenNeuron = new HiddenNeuron() { Id = hiddenNeuronId };
                    hiddenLayerNeurons.Add(hiddenNeuron);
                    
                    if (hlayer == 0)
                        // Generate connections to input layer
                        foreach (var inputNeuron in inputNeurons)
                            hiddenNeuron.GenerateConnectionsWith(inputNeuron);
                    else
                        // Generate connections to the previous hidden layer.
                        foreach (IHiddenNeuron prevLayerNeuron in hiddenLayers[hlayer - 1].Neurons)
                            hiddenNeuron.GenerateConnectionsWith(prevLayerNeuron);
                    
                    if (hlayer == hiddenLayersCount - 1)
                        // Generate connections to output layer
                        foreach (var outputNeuron in outputNeurons)
                            hiddenNeuron.GenerateConnectionsWith(outputNeuron);
                }
                
                hiddenLayers.Add(new HiddenLayer(hlayer + 1, hiddenLayerNeurons));
            }

            var layers = new List<INetworkLayer>() { new InputLayer(0, inputNeurons), new OutputLayer(hiddenLayersCount + 1, outputNeurons) };

            if (!hiddenLayers.Any())
            {
                // Generate connections from input layer to output layer if there are no hidden layers
                foreach (var inputNeuron in inputNeurons)
                    foreach (var outputNeuron in outputNeurons)
                        inputNeuron.GenerateConnectionsWith(outputNeuron);
            }
            else
                layers.AddRange(hiddenLayers);

            Layers = layers;
        }

        /// <summary>
        /// Randomizes all neuron activation levels, biases, and connection weights.
        /// </summary>
        public void RandomizeNetwork()
        {
            ValidateNetwork();

            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    // Randomize neuron activation levels & biases
                    neuron.ActivationLevel = NumberUtils.GenerateRandomNumber(0, 100) / 100.0;
                    neuron.Bias = NumberUtils.GenerateRandomNumber(0, 100) / 100.0;

                    // Randomize all incoming and outgoing connection weights
                    foreach (var outgoingConnection in neuron.Connections.OfType<IOutgoingConnection>())
                    {
                        outgoingConnection.Weight = NumberUtils.GenerateRandomNumber(0, 100) / 100.0;

                        foreach (var incomingConnection in outgoingConnection.ToNeuron.Connections.OfType<IIncomingConnection>())
                        {
                            if (incomingConnection.FromNeuron.Id == neuron.Id)
                                incomingConnection.Weight = outgoingConnection.Weight;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Trains the network with the supplied <paramref name="trainingIterations"/>.
        /// </summary>
        /// <param name="trainingIterations">Contains the data entries that will be used to train this network.</param>
        /// <returns>Returns the training iterations with updated training costs.</returns>
        public IEnumerable<INetworkTrainingIteration> Train(IEnumerable<INetworkTrainingIteration> trainingIterations)
        {
            ValidateNetwork();
            ValidateTrainingIterations(trainingIterations);
            
            foreach (var entry in trainingIterations)
            {
                // Push the inputs through the network.
                ForwardPropogateInputs(entry.Inputs);

                // Store the calculated cost 
                entry.TrainingCost = CalculateCost(entry);

                // Pull the changes back through the network
                BackPropogateOutputs(entry.Outputs);
            }

            return trainingIterations;
        }

        /// <summary>
        /// Returns the quadratic cost of the test case for all inputs using:
        /// Cost = ((Expected - Actual)^2) ==> Take the average of all costs.
        /// </summary>
        /// <param name="testCase">The training iteration to calculate the cost for.</param>
        /// <returns>Returns the cost of the training iteration.</returns>
        private double CalculateCost(INetworkTrainingIteration testCase)
        {
            var neuronOutputs = GetNeuronOutputs();
            testCase.Outputs.ForEach(t => 
            {
                // Update the actual activation level of each test case.
                t.ActivationLevel = neuronOutputs.First(n => n.NeuronId == t.NeuronId).ActivationLevel;
            });

            return testCase.Outputs.Sum(t => Math.Pow(t.ActivationLevel - t.ExpectedActivationLevel, 2));
        }

        /// <summary>
        /// Applies the set of <paramref name="networkInputs"/> to the input layer of the network and performs all calculations to generate an output.
        /// </summary>
        /// <param name="networkInputs">The inputs to be applied to the network.</param>
        /// <returns>Returns the current neuron activation levels as an enumerable after the inputs have been applied.</returns>
        public IEnumerable<INetworkOutput> ApplyInputs(IEnumerable<INetworkInput> networkInputs)
        {
            ValidateNetwork();
            ValidateNetworkInputs(networkInputs);

            ForwardPropogateInputs(networkInputs);

            return GetNeuronOutputs();
        }

        /// <summary>
        /// Returns the current neuron activation levels.
        /// </summary>
        /// <returns>Returns the current neuron activation levels as an enumerable.</returns>
        public IEnumerable<INetworkOutput> GetNeuronOutputs()
        {
            var outputLayers = Layers?.OfType<IOutputLayer>();
            return outputLayers == null || !outputLayers.Any() ? 
                        null : 
                        outputLayers.Where(l => l.Neurons != null)
                                    .SelectMany(l => l.Neurons.Where(outputNeuron => outputNeuron != null)
                                                              .Select(outputNeuron => new NetworkOutput()
                                                              {
                                                                  NeuronId = outputNeuron.Id,
                                                                  ActivationLevel = outputNeuron.ActivationLevel
                                                              }));
        }

        /// <summary>
        /// Applies the set of <paramref name="inputs"/> to the input layer of the network and performs all calculations to generate an output.
        /// </summary>
        /// <param name="inputs">The inputs to be applied to the network.</param>
        /// <param name="validateNetwork">Bool to indicate whether the network should be verified before propogating. This is used to help performance since this method will be called many times.</param>
        private void ForwardPropogateInputs(IEnumerable<INetworkInput> networkInputs)
        {
            var inputLayer = Layers.OfType<IInputLayer>().First();

            // Apply activation levels to input layer neurons.
            foreach (var input in networkInputs)
            {
                GetNeuronById(input.NeuronId, inputLayer).ActivationLevel = input.ActivationLevel;
            }

            // Step through each layer's neurons and update the activation levels
            foreach (var layer in Layers.Where(l => l != inputLayer).OrderBy(l => l.SortOrder))
            {
                foreach (var neuron in layer.Neurons)
                {
                    // The intermediate summation of all incoming neuron connection activation levels times 
                    // the weight of the incoming connection, added to the neuron bias. This value represents
                    // the neuron activation level before the activation function (sigmoid, reLu, tanh) has been applied.
                    var zL = neuron.Connections.OfType<IIncomingConnection>().Sum(i => i.FromNeuron.ActivationLevel * i.Weight) + neuron.Bias;

                    // The final neuron activation level with an applied activation function.
                    neuron.ActivationLevel = ApplyActivationFunction(zL);
                }
            }
        }
        
        /// <summary>
        /// Calculates and applies weight and bias changes to each of the neuron connection and biases based on the
        /// difference between the <paramref name="expectedOutputs"/> and the actual network outputs.
        /// </summary>
        /// <param name="expectedOutputs">A list of the expected activation levels from the associated training iteration.</param>
        private void BackPropogateOutputs(IList<INetworkTrainingOutput> expectedOutputs)
        {
            // Dictionary holding the derivative of the cost with respect to the activation of the current neuron.
            // This calculation is computed by summing the cost of all outgoing connections from this neuron.
            var dC0_daLTotalDict = new Dictionary<INeuron, double>();

            foreach (var layer in Layers.OrderByDescending(l => l.SortOrder))
            {
                foreach (var neuron in layer.Neurons)
                {
                    // The activation level for this neuron   
                    var aL = neuron.ActivationLevel;

                    // The derivative of the cost with respect to the activation of this neuron. 
                    // This calculation changes if the neuron is in the output layer.
                    var dC0_daL = layer is OutputLayer ? (2.0 * (aL - expectedOutputs.First(o => o.NeuronId == neuron.Id).ExpectedActivationLevel)) : dC0_daLTotalDict[neuron];

                    // The derivative of the activation level of this neuron with respect to Z.
                    var daL_dzL = ApplyActivationFunctionDerivative(aL);

                    // The derivative of the cost with respect to the bias of this neuron.
                    // Update the bias of the neuron using this calculation.
                    //var dC0_dbL = 1.0 * daL_dzL * dC0_daL;
                    //neuron.Bias -= dC0_dbL * _dampingRate;

                    // Foreach incoming connection, compute the derivative of the cost with respect to the 
                    // activation of the neuron on the left.
                    foreach (var incomingConnection in neuron.Connections.OfType<IIncomingConnection>())
                    {
                        // The derivative of Z with respect to the incoming connection weight.
                        var dzL_dwL = incomingConnection.FromNeuron.ActivationLevel;

                        // The derivative of the cost with respect to the incoming connection weight.
                        var dC0_dwL = dzL_dwL * daL_dzL * dC0_daL;

                        // The derivative of Z with respect to the activation of the incoming connection neuron.  
                        // Update the weight of the incoming connection with this calculation.
                        var dzL_daLMinus1 = incomingConnection.Weight;
                        incomingConnection.Weight -= dC0_dwL * _dampingRate;
                        incomingConnection.FromNeuron.Connections.OfType<IOutgoingConnection>().Where(c => c.ToNeuron == neuron).First().Weight = incomingConnection.Weight;
                        
                        // The derivative of the cost with respect to the activation of the incoming connection neuron. 
                        // Add is value to the total derivative calculation for this neuron.
                        var dC0_daLMinus1 = dzL_daLMinus1 * daL_dzL * dC0_daL;

                        // Update the cost derivative for the incoming neuron.
                        if (!dC0_daLTotalDict.ContainsKey(incomingConnection.FromNeuron))
                            dC0_daLTotalDict.Add(incomingConnection.FromNeuron, dC0_daLMinus1);
                        else
                            dC0_daLTotalDict[incomingConnection.FromNeuron] += dC0_daLMinus1;
                    }
                }
            }
        }

        /// <summary>
        /// Calculates the output of the sigmoid function for the supplied <paramref name="value"/>.
        /// </summary>
        /// <param name="value">The value that will be inserted into the sigmoid function.</param>
        /// <returns>Returns the output of the sigmoid function using the supplied value.</returns>
        private double ApplyActivationFunction(double value)
        {
            // Sigmoid function
            return (1.0 / (1 + Math.Exp(-1.0 * value)));
        }

        /// <summary>
        /// Calculates the sigmoid function derivative output for a specified <paramref name="value"/>.
        /// </summary>
        /// <param name="value">The value to plug into the derivative of the sigmoid function.</param>
        /// <returns>Returns the sigmoid derivative output for the specified <paramref name="value"/>.</returns>
        private double ApplyActivationFunctionDerivative(double value)
        {
            // Sigmoid function
            return value * (1.0 - value);
        }

        /// <summary>
        /// Validates the network by checking all layers, neurons, and connections. Exception will be thrown if an invalid configuration is present.
        /// </summary>
        protected override void ValidateNetwork()
        {
            if (Layers == null)
                throw new InvalidOperationException("Network has not been initialized with layers.");

            if (Layers.Count(l => l is IInputLayer) != 1)
                throw new InvalidOperationException("Network should contain exactly one input layer.");

            if (Layers.Count(l => l is IOutputLayer) != 1)
                throw new InvalidOperationException("Network should contain exactly one output layer.");

            foreach (var layer in Layers)
            {
                if (layer == null)
                    throw new InvalidOperationException("Network contains null layer(s).");

                foreach (var neuron in layer.Neurons)
                {
                    if (neuron == null)
                        throw new InvalidOperationException("Network contains null neuron(s).");

                    if (neuron.Connections == null || !neuron.Connections.Any())
                        throw new InvalidOperationException("Neuron contains null or invalid connections.");

                    if (layer is IInputLayer && !(neuron is IInputNeuron))
                        throw new InvalidOperationException("Input layer contains neuron(s) that aren't input neurons.");

                    else if (layer is IHiddenLayer && !(neuron is IHiddenNeuron))
                        throw new InvalidOperationException("Hidden layer contains neuron(s) that aren't hidden neurons.");

                    else if (layer is IOutputLayer && !(neuron is IOutputNeuron))
                        throw new InvalidOperationException("Output layer contains neuron(s) that aren't output neurons.");

                    if (neuron is IHiddenNeuron && (!neuron.Connections.Any(c => c is IIncomingConnection) || !neuron.Connections.Any(c => c is IOutgoingConnection)))
                        throw new InvalidOperationException("Hidden neuron doesn't contain both incoming and outgoing connection(s).");

                    foreach (var connection in neuron.Connections)
                    {
                        if (connection == null)
                            throw new InvalidOperationException("Network contains null connection(s).");

                        if (neuron is IInputNeuron && !(connection is IOutgoingConnection))
                            throw new InvalidOperationException("Input neuron contains connection(s) that aren't outgoing or are invalid.");

                        else if (neuron is IOutputNeuron && !(connection is IIncomingConnection))
                            throw new InvalidOperationException("Input neuron contains connection(s) that aren't outgoing or are invalid.");

                        else if ((connection is IIncomingConnection && ((IIncomingConnection)connection).FromNeuron == null) ||
                                (connection is IOutgoingConnection && ((IOutgoingConnection)connection).ToNeuron == null))
                            throw new InvalidOperationException("Hidden neuron contains connection(s) that are invalid.");
                    }
                }
            }
        }
        
        /// <summary>
        /// Validates that the supplied <paramref name="networkInputs"/> can be applied to this network.
        /// </summary>
        /// <param name="networkInputs">The network inputs that should be applied to the input layer of this network.</param>
        protected override void ValidateNetworkInputs(IEnumerable<INetworkInput> networkInputs)
        {
            if (networkInputs == null || !networkInputs.Any())
                throw new ArgumentNullException("inputs");

            var inputLayer = Layers.OfType<IInputLayer>().First();
            var inputLayerNeurons = inputLayer.Neurons;
            var inputNeuronIds = networkInputs.Select(n => n.NeuronId).Distinct();

            // Verify there is one input for each input neuron.
            if (inputLayerNeurons.Count() != inputNeuronIds.Count())
                throw new ArgumentException("There should be one input value for each input neuron of the network.");

            // Verify neuron ids are valid.
            foreach (var neuronId in inputNeuronIds)
            {
                if (GetNeuronById(neuronId, inputLayer) == null)
                    throw new ArgumentException($"NetworkInputs contains an invalid neuron id: {neuronId}.");
            }
        }

        /// <summary>
        /// Validates that the supplied <paramref name="networkOutputs"/> can be applied to this network.
        /// </summary>
        /// <param name="networkOutputs">The network outputs for output layer of this network.</param>
        protected override void ValidateNetworkOutputs(IEnumerable<INetworkOutput> networkOutputs)
        {
            if (networkOutputs == null || !networkOutputs.Any())
                throw new ArgumentNullException("networkOutputs");

            var outputLayer = Layers.OfType<IOutputLayer>().First();
            var outputLayerNeurons = outputLayer.Neurons;
            var outputNeuronIds = networkOutputs.Select(n => n.NeuronId).Distinct();

            // Verify there is one output for each output neuron.
            if (outputLayerNeurons.Count() != outputNeuronIds.Count())
                throw new ArgumentException("There should be one output value for each output neuron of the network.");

            // Verify neuron ids are valid.
            foreach (var neuronId in outputNeuronIds)
            {
                if (GetNeuronById(neuronId, outputLayer) == null)
                    throw new ArgumentException($"NetworkOutputs contains an invalid neuron id: {neuronId}.");
            }
        }

        /// <summary>
        /// Validates each of the training entries has valid input and outputs for this network.
        /// </summary>
        /// <param name="trainingIterations">The training iterations that will be applied to this network.</param>
        protected void ValidateTrainingIterations(IEnumerable<INetworkTrainingIteration> trainingIterations)
        {
            if (trainingIterations == null || !trainingIterations.Any())
                throw new ArgumentNullException("trainingIterations");

            foreach (var entry in trainingIterations)
            {
                ValidateNetworkInputs(entry.Inputs);
                ValidateNetworkOutputs(entry.Outputs);
            }
        }

        /// <summary>
        /// Returns the neuron in the <paramref name="layer"/> with the matching <paramref name="neuronId"/> if one exists; otherwise null.
        /// </summary>
        /// <param name="neuronId">The id of the neuron to return.</param>
        /// <param name="layer">The network layer that the neuron exists in.</param>
        /// <returns>Returns the neuron in the <paramref name="layer"/> with the matching <paramref name="neuronId"/> if one exists; otherwise null.</returns>
        private INeuron GetNeuronById(int neuronId, INetworkLayer layer)
        {
            return layer.Neurons.FirstOrDefault(n => n.Id == neuronId);
        }
    }
}
using NeuralNetwork.Generic.Connections;
using NeuralNetwork.Generic.Layers;
using Framework.Generic.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Networks
{
    /// <summary>
    /// This interface represents a Deep Feed Forward (DFF) Neural Network.
    /// </summary>
    public interface IDFFNeuralNetwork : INeuralNetwork
    {
        IInputLayer GetInputLayer();
        IEnumerable<IHiddenLayer> GetHiddenLayers();
        IOutputLayer GetOutputLayer();

        void RandomizeNetwork();
        void ApplyInputs(IList<double> inputActivations);
    }

    /// <summary>
    /// This class represents a Deep Feed Forward (DFF) Neural Network.
    /// </summary>
    public class DFFNeuralNetwork : NeuralNetworkBase, IDFFNeuralNetwork
    {
        public DFFNeuralNetwork(INetworkLayer[] layers)
            : base(layers)
        {

        }

        public IEnumerable<IHiddenLayer> GetHiddenLayers()
        {
            return Layers?.OfType<IHiddenLayer>();
        }

        public IInputLayer GetInputLayer()
        {
            return Layers?.OfType<IInputLayer>().FirstOrDefault();
        }

        public IOutputLayer GetOutputLayer()
        {
            return Layers?.OfType<IOutputLayer>().FirstOrDefault();
        }

        /// <summary>
        /// Randomizes all neuron activation levels, biases, and connection weights.
        /// </summary>
        public void RandomizeNetwork()
        {
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer?.Neurons)
                {
                    // Randomize neuron activation levels & biases
                    neuron.ActivationLevel = NumberUtils.GenerateRandomNumber(-100, 100) / 100.0;
                    neuron.Bias = NumberUtils.GenerateRandomNumber(-100, 100) / 100.0;

                    foreach (var outConnection in neuron?.Connections?.OfType<IOutgoingConnection>())
                    {
                        // Randomize all output connection weights
                        outConnection.Weight = NumberUtils.GenerateRandomNumber(-100, 100) / 100.0;
                    }
                }
            }
        }

        public void ApplyInputs(IList<double> inputActivations)
        {
            if (inputActivations == null)
                throw new ArgumentNullException("inputActivations");

            AssignActivationsToNeurons(inputActivations);
            ForwardPropogate();
        }

        private void ForwardPropogate()
        {
            // Step through each layer's neurons and update the activation levels
            foreach (var layer in Layers.Where(l => !(l is IInputLayer)) ?? new List<INetworkLayer>())
            {
                foreach (var neuron in layer?.Neurons ?? new List<INeuron>())
                {
                    var incomingConnections = neuron?.Connections?.OfType<IIncomingConnection>();
                    if (incomingConnections == null || !incomingConnections.Any())
                        continue;

                    // Update neuron activation level by calculating the summation of all incoming neuron connection
                    // activation levels times the weight of the incoming connection. Next, add the neuron bias.
                    // Finally, insert this answer into the activation function (ReLu, sigmoid, tanh) to limit the output between -1 and 1.
                    neuron.ActivationLevel = ApplySigmoidFunction(incomingConnections.Sum(i => i.FromNeuron.ActivationLevel * i.Weight) +neuron.Bias);
                }
            }
        }

        private double ApplySigmoidFunction(double value)
        {
            // Possibly send this to parent abstract class..
            return (1.0 / (1 + Math.Exp(-1.0 * value)));
        }

        private void AssignActivationsToNeurons(IList<double> inputActivations)
        {
            var inputLayerNeurons = GetInputLayer()?.Neurons;

            if (inputLayerNeurons?.Count() != inputActivations.Count)
                throw new ArgumentException("The number of input activations should be equal to the number of input neurons of the network.");

            // Assign each of the input activations to the input layer neurons.
            int position = 0;
            foreach (var inputNeuron in inputLayerNeurons)
            {
                inputNeuron.ActivationLevel = inputActivations[position];
            }
        }

        private void BackwardPropogateCosts()
        {
            // Will have to determine if we should be calculating costs here OR if they should be supplied as arguments.
            // Also, consider how I will be storing/retrieving cost / change info for each neuron.
        }
    }
}
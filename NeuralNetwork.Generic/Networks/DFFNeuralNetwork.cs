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
        /// <summary>
        /// The learning rate of the network
        /// </summary>
        private const double _dampingRate = 0.01;

        public DFFNeuralNetwork(INetworkLayer[] layers)
            : base(layers)
        {

        }
        
        public IEnumerable<IHiddenLayer> GetHiddenLayers() => Layers?.OfType<IHiddenLayer>();

        public IInputLayer GetInputLayer() => Layers?.OfType<IInputLayer>().FirstOrDefault();

        public IOutputLayer GetOutputLayer() => Layers?.OfType<IOutputLayer>().FirstOrDefault();

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

        private void ForwardPropogateInputs(IDictionary<double, INeuron> neuronInputs)
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

        private double ApplySigmoidFunctionInverse(double value)
        {
            // check this calculation.
            return Math.Log(value / (1 - value));
        }

        private double ApplySigmoidFunctionDerivative(double value)
        {
            return value * (1.0 - value);
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

        public void BackPropogateCosts(IList<double> costs)
        {
            // Dictionary holding the derivative of the cost with respect to the activation of the previous neuron.
            // This is used so that as we go right-to-left, we are able to retrieve the cost derivate of the previous 
            // neuron on the right.
            var dC0_daLPlus1Dict = new Dictionary<INeuron, double>();

            foreach (var networkLayer in SortedNetworkLayers.Reverse())
            {
                foreach (var neuron in networkLayer.RegisteredNeurons)
                {
                    // The activation level for this neuron   
                    var aL = neuron.ActivationLevel;

                    // The Intermediate value 'Z' --> the neuron's activation level without applying the activation function.
                    var zL = ApplySigmoidFunctionInverse(neuron.ActivationLevel);

                    // The expected output of the neuron (This value is ignored if the neuron doesn't belong to the output layer).
                    var yL = networkLayer is OutputLayer ?
                                    expectedNeuronOutputsDict[neuron] :
                                    0.1234;

                    // The derivative of the cost with respect to the activation of this neuron. 
                    var dC0_daL = networkLayer is OutputLayer ?
                                                2.0 * (aL - yL) :
                                                dC0_daLPlus1Dict[neuron];

                    // The derivative of the activation level of this neuron with respect to Z.
                    var daL_dzL = ApplySigmoidFunctionDerivative(zL);

                    // The derivative of the cost with respect to the bias of this neuron.
                    // Update the bias of the neuron using this calculation.
                    var dC0_dbL = 1.0 * daL_dzL * dC0_daL;
                    neuron.Bias -= dC0_dbL * _dampingRate;

                    // The derivative of the cost with respect to the activation of the neuron on the left.
                    // To calculate this, compute the derivative of EACH incoming connection and add it 
                    // to this total. The derivative of the sum is the sum of the derivatives.
                    var dC0_daLMinus1Total = 0.0;

                    foreach (var incomingConnection in neuron.IncomingConnections)
                    {
                        // The derivative of Z with respect to the incoming connection weight.
                        var dzL_dwL = incomingConnection.FromNeuron.ActivationLevel;

                        // The derivative of the cost with respect to the incoming connection weight.
                        var dC0_dwL = dzL_dwL * daL_dzL * dC0_daL;

                        // The derivative of Z with respect to the activation of the incoming connection neuron.  
                        // Update the weight of the incoming connection with this calculation.
                        var dzL_daLMinus1 = incomingConnection.Weight;
                        incomingConnection.Weight -= dC0_dwL * _dampingRate;

                        // The derivative of the cost with respect to the activation of the incoming connection neuron. 
                        // Add is value to the total derivative calculation for this neuron.
                        var dC0_daLMinus1 = dzL_daLMinus1 * daL_dzL * dC0_daL;
                        dC0_daLMinus1Total += dC0_daLMinus1;
                    }

                    dC0_daLPlus1Dict.Add(neuron, dC0_daLMinus1Total);
                }
            }
        } 
    }
}
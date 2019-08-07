using NeuralNetwork.Generic.Connections;
using NeuralNetwork.Generic.Layers;
using Framework.Generic.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Networks
{
    public interface IDFFNeuralNetwork : INeuralNetwork
    {
        void RandomizeNetwork();

        IInputLayer GetInputLayer();
        IHiddenLayer[] GetHiddenLayers();
        IOutputLayer GetOutputLayer();
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

        public IHiddenLayer[] GetHiddenLayers()
        {
            return Layers?.OfType<IHiddenLayer>().ToArray();
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

                    foreach (var outConnection in neuron?.Connections.OfType<IOutgoingConnection>())
                    {
                        // Randomize all output connection weights
                        outConnection.Weight = NumberUtils.GenerateRandomNumber(-100, 100) / 100.0;
                    }
                }
            }
        }
    }
}

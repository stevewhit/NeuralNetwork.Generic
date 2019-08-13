using NeuralNetwork.Generic.Datasets;
using NeuralNetwork.Generic.Layers;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Networks
{
    public interface INeuralNetwork
    {
        /// <summary>
        /// The layers in this neural network.
        /// </summary>
        IEnumerable<INetworkLayer> Layers { get; set; }
    }

    public abstract class NeuralNetworkBase : INeuralNetwork
    {
        /// <summary>
        /// The layers in this neural network.
        /// </summary>
        public IEnumerable<INetworkLayer> Layers { get; set; }

        public NeuralNetworkBase()
        {
            Layers = new List<INetworkLayer>();
        }

        public NeuralNetworkBase(IEnumerable<INetworkLayer> layers)
        {
            Layers = layers;
        }

        protected abstract void ValidateNetwork();
        protected abstract void ValidateNetworkInputs(IEnumerable<INeuronInput> networkInputs);
        protected abstract void ValidateNetworkOutputs(IEnumerable<INeuronOutput> networkOutputs);
    }
}

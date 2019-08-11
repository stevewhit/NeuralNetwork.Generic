using NeuralNetwork.Generic.Neurons;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Layers
{
    public interface INetworkLayer
    {
        /// <summary>
        /// The position order of this layer.
        /// </summary>
        int SortOrder { get; set; }

        /// <summary>
        /// The neurons that exist in this layer.
        /// </summary>
        IEnumerable<INeuron> Neurons { get; set; }
    }

    public abstract class NetworkLayerBase : INetworkLayer
    {
        /// <summary>
        /// The position order of this layer.
        /// </summary>
        public int SortOrder { get; set; }

        /// <summary>
        /// The neurons that exist in this layer.
        /// </summary>
        public IEnumerable<INeuron> Neurons { get; set; }
        
        public NetworkLayerBase(int sortOrder)
        {
            Neurons = new List<INeuron>();
            SortOrder = sortOrder;
        }

        public NetworkLayerBase(int sortOrder, IEnumerable<INeuron> neurons)
        {
            Neurons = neurons;
            SortOrder = sortOrder;
        }
    }
}

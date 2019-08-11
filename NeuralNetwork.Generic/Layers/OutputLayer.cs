using NeuralNetwork.Generic.Neurons;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Layers
{
    public interface IOutputLayer : INetworkLayer
    {

    }

    public class OutputLayer : NetworkLayerBase, IOutputLayer
    {
        public OutputLayer(int sortOrder)
           : base(sortOrder)
        {

        }

        public OutputLayer(int sortOrder, IEnumerable<IOutputNeuron> outputNeurons)
           : base(sortOrder, outputNeurons)
        {

        }
    }
}

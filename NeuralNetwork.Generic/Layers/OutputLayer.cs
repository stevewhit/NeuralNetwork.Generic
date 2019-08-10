using NeuralNetwork.Generic.Neurons;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Layers
{
    public interface IOutputLayer : INetworkLayer
    {

    }

    public class OutputLayer : NetworkLayerBase, IOutputLayer
    {
        public OutputLayer(IEnumerable<IOutputNeuron> outputNeurons, int sortOrder)
           : base(outputNeurons, sortOrder)
        {

        }
    }
}

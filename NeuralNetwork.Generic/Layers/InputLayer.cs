using NeuralNetwork.Generic.Neurons;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Layers
{
    public interface IInputLayer : INetworkLayer
    {

    }

    public class InputLayer : NetworkLayerBase, IInputLayer
    {
        public InputLayer(IEnumerable<IInputNeuron> inputNeurons, int sortOrder)
           : base(inputNeurons, sortOrder)
        {

        }
    }
}

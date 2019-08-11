using NeuralNetwork.Generic.Neurons;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Layers
{
    public interface IInputLayer : INetworkLayer
    {

    }

    public class InputLayer : NetworkLayerBase, IInputLayer
    {
        public InputLayer(int sortOrder)
            : base(sortOrder)
        {

        }

        public InputLayer(int sortOrder, IEnumerable<IInputNeuron> inputNeurons)
           : base(sortOrder, inputNeurons)
        {

        }
    }
}

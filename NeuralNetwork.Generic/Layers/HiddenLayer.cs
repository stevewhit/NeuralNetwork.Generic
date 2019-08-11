using NeuralNetwork.Generic.Neurons;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Layers
{
    public interface IHiddenLayer : INetworkLayer
    {

    }

    public class HiddenLayer : NetworkLayerBase, IHiddenLayer
    {
        public HiddenLayer(int sortOrder)
           : base(sortOrder)
        {

        }

        public HiddenLayer(int sortOrder, IEnumerable<IHiddenNeuron> hiddenNeurons)
           : base(sortOrder, hiddenNeurons)
        {

        }
    }
}

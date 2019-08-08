using NeuralNetwork.Generic.Neurons;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Layers
{
    public interface IHiddenLayer : INetworkLayer
    {

    }

    public class HiddenLayer : NetworkLayerBase, IHiddenLayer
    {
        public HiddenLayer(IEnumerable<IHiddenNeuron> hiddenNeurons)
           : base(hiddenNeurons)
        {

        }
    }
}

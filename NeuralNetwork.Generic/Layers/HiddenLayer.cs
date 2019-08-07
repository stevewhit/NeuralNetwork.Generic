using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Layers
{
    public interface IHiddenLayer : INetworkLayer
    {

    }

    public class HiddenLayer : NetworkLayerBase, IHiddenLayer
    {
        public HiddenLayer(IHiddenNeuron[] hiddenNeurons)
           : base(hiddenNeurons)
        {

        }
    }
}

using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Layers
{
    public interface IInputLayer : INetworkLayer
    {

    }

    public class InputLayer : NetworkLayerBase, IInputLayer
    {
        public InputLayer(IInputNeuron[] inputNeurons)
           : base(inputNeurons)
        {

        }
    }
}

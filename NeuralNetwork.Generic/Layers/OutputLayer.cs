using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Layers
{
    public interface IOutputLayer : INetworkLayer
    {

    }

    public class OutputLayer : NetworkLayerBase, IOutputLayer
    {
        public OutputLayer(IOutputNeuron[] outputNeurons)
           : base(outputNeurons)
        {

        }
    }
}

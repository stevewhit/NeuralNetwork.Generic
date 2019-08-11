using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Networks;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Test.Builders
{
    public class FakeNeuralNetwork : NeuralNetworkBase
    {
        public FakeNeuralNetwork()
        {

        }

        public FakeNeuralNetwork(IEnumerable<INetworkLayer> layers)
            : base (layers)
        {

        }
    }
}

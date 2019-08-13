using NeuralNetwork.Generic.Datasets;
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

        protected override void ValidateNetwork()
        {
            throw new System.NotImplementedException();
        }

        protected override void ValidateNetworkInputs(IEnumerable<INeuronInput> networkInputs)
        {
            throw new System.NotImplementedException();
        }

        protected override void ValidateNetworkOutputs(IEnumerable<INeuronOutput> networkOutputs)
        {
            throw new System.NotImplementedException();
        }
    }
}

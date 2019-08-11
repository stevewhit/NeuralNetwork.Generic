using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Neurons;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Test.Builders
{
    public class FakeNeuronLayer : NetworkLayerBase
    { 
        public FakeNeuronLayer(int sortOrder)
            : base (sortOrder)
        {

        }

        public FakeNeuronLayer(int sortOrder, IEnumerable<INeuron> neurons)
            : base(sortOrder, neurons)
        {

        }
    }
}

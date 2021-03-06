﻿using NeuralNetwork.Generic.Datasets;
using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Networks;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace NeuralNetwork.Generic.Test.Builders
{
    [ExcludeFromCodeCoverage]
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

        protected override void ValidateNetworkInputs(IEnumerable<INetworkInput> networkInputs)
        {
            throw new System.NotImplementedException();
        }

        protected override void ValidateNetworkOutputs(IEnumerable<INetworkOutput> networkOutputs)
        {
            throw new System.NotImplementedException();
        }
    }
}

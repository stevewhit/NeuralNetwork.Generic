using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Neurons;
using NeuralNetwork.Generic.Test.Builders;

namespace NeuralNetwork.Generic.Test.Neurons
{
    [TestClass]
    public class NeuronBaseTest
    {
        private INeuron _neuron;

        [TestInitialize]
        public void Initialize()
        {
            _neuron = new FakeNeuron();
        }

        [TestMethod]
        public void Fail()
        {
            // Stopped here.. Test this class for total code coverage reasons.
            Assert.Fail();
        }
    }
}

using System;
using System.Collections.Generic;
using Framework.Generic.Utility;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Networks;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Networks
{
    [TestClass]
    public class DFFNeuralNetworkTest
    {
        private INeuralNetwork _network;
        private IInputLayer _inputLayer;
        private IHiddenLayer _hiddenLayer;
        private IOutputLayer _outputLayer;

        [TestInitialize]
        public void Initialize()
        {
            var inputNeurons = new List<IInputNeuron>();
            var hiddenNeurons = new List<IHiddenNeuron>();
            var outputNeurons = new List<IOutputNeuron>();

            for (int i = 0; i < 5; i++)
            {
                inputNeurons.Add(new InputNeuron() { Id = i });
            }

            for (int i = 5; i < 15; i++)
            {
                var hiddenNeuron = new HiddenNeuron() { Id = i };
                hiddenNeurons.Add(hiddenNeuron);

                foreach (var inputNeuron in inputNeurons)
                    hiddenNeuron.GenerateConnectionsWith(inputNeuron);
            }

            for (int i = 15; i < 17; i++)
            {
                var outputNeuron = new OutputNeuron() { Id = i };
                outputNeurons.Add(outputNeuron);

                foreach (var hiddenNeuron in hiddenNeurons)
                    outputNeuron.GenerateConnectionsWith(hiddenNeuron);
            }

            _inputLayer = new InputLayer(0, inputNeurons);
            _hiddenLayer = new HiddenLayer(1, hiddenNeurons);
            _outputLayer = new OutputLayer(2, outputNeurons);

            _network = new DFFNeuralNetwork(new List<INetworkLayer>()
            {
                _inputLayer, _hiddenLayer, _outputLayer
            });
        }

        [TestMethod]
        public void TestMethod1()
        {
        }
    }
}

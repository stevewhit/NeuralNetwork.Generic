using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Framework.Generic.Utility;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Connections;
using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Networks;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Networks
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public class DFFNeuralNetworkTest
    {
        private IDFFNeuralNetwork _network;
        private IInputLayer _inputLayer;
        private IHiddenLayer _hiddenLayer;
        private IOutputLayer _outputLayer;

        [TestInitialize]
        public void Initialize()
        {
            _network = new DFFNeuralNetwork(3, 1, 5, 2);
            _inputLayer = _network.Layers.OfType<IInputLayer>().First();
            _outputLayer = _network.Layers.OfType<IOutputLayer>().First();
            _hiddenLayer = _network.Layers.OfType<IHiddenLayer>().First();
        }

        #region Testing DFFNeuralNetwork()...

        [TestMethod]
        public void DFFNeuralNetwork_InitializesLayers()
        {
            // Arrange
            var network = new DFFNeuralNetwork();

            // Assert
            Assert.IsNotNull(network.Layers);
        }

        #endregion
        #region Testing DFFNeuralNetwork(IEnumerable<INetworkLayer> layers)...

        [TestMethod]
        public void DFFNeuralNetwork_WithLayers_InitializesLayers()
        {
            // Arrange
            var layers = new List<INetworkLayer>() { _inputLayer, _hiddenLayer, _outputLayer };

            // Act
            var network = new DFFNeuralNetwork(layers);

            // Assert
            Assert.IsNotNull(network.Layers);
            Assert.IsTrue(network.Layers.Count() == 3);
            Assert.IsTrue(network.Layers.Contains(_inputLayer));
            Assert.IsTrue(network.Layers.Contains(_hiddenLayer));
            Assert.IsTrue(network.Layers.Contains(_outputLayer));
        }

        #endregion
        #region Testing DFFNeuralNetwork(int inputLayerNeuronCount, int hiddenLayersCount, int hiddenLayerNeuronCount, int outputLayerNeuronCount)...

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCounts_InitializesLayers()
        {
            // Assert
            Assert.IsNotNull(_network.Layers);
            Assert.IsTrue(_network.Layers.Count() == 3);
            Assert.IsTrue(_network.Layers.Contains(_inputLayer));
            Assert.IsTrue(_network.Layers.Contains(_hiddenLayer));
            Assert.IsTrue(_network.Layers.Contains(_outputLayer));
        }

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCounts_InitializesInputLayer()
        {
            // Assert
            Assert.IsTrue(_inputLayer.SortOrder == 0);
            Assert.IsTrue(_inputLayer.Neurons.Count() == 3);
        }

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCounts_InitializesHiddenLayer()
        {
            // Assert
            Assert.IsTrue(_hiddenLayer.SortOrder == 1);
            Assert.IsTrue(_hiddenLayer.Neurons.Count() == 5);
        }

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCounts_InitializesMultipleHiddenLayers()
        {
            // Act
            _network = new DFFNeuralNetwork(3, 5, 5, 2);
            var hiddenLayers = _network.Layers.OfType<IHiddenLayer>();

            // Assert
            Assert.IsTrue(hiddenLayers.Count() == 5);
            Assert.IsTrue(hiddenLayers.All(l => l.Neurons.Count() == 5));
        }

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCounts_InitializesOutputLayer()
        {
            // Assert
            Assert.IsTrue(_outputLayer.SortOrder == 2);
            Assert.IsTrue(_outputLayer.Neurons.Count() == 2);
        }

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCounts_InitializesInputLayerConnections()
        {
            // Assert
            Assert.IsTrue(_inputLayer.Neurons.All(n => n.Connections.Count() == 5));
            Assert.IsTrue(_inputLayer.Neurons.All(n => n.Connections.OfType<IOutgoingConnection>().Count() == 5));
            Assert.IsTrue(_inputLayer.Neurons.All(n => n.Connections.OfType<IOutgoingConnection>().All(c => c.ToNeuron is IHiddenNeuron)));
        }

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCounts_InitializesHiddenLayerConnections()
        {
            // Assert
            Assert.IsTrue(_hiddenLayer.Neurons.All(n => n.Connections.Count() == 5));
            Assert.IsTrue(_hiddenLayer.Neurons.All(n => n.Connections.OfType<IIncomingConnection>().Count() == 3));
            Assert.IsTrue(_hiddenLayer.Neurons.All(n => n.Connections.OfType<IOutgoingConnection>().Count() == 2));
            Assert.IsTrue(_hiddenLayer.Neurons.All(n => n.Connections.OfType<IIncomingConnection>().All(c => c.FromNeuron is IInputNeuron)));
            Assert.IsTrue(_hiddenLayer.Neurons.All(n => n.Connections.OfType<IOutgoingConnection>().All(c => c.ToNeuron is IOutputNeuron)));
        }

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCounts_InitializesOutputLayerConnections()
        {
            // Assert
            Assert.IsTrue(_outputLayer.Neurons.All(n => n.Connections.Count() == 5));
            Assert.IsTrue(_outputLayer.Neurons.All(n => n.Connections.OfType<IIncomingConnection>().Count() == 5));
            Assert.IsTrue(_outputLayer.Neurons.All(n => n.Connections.OfType<IIncomingConnection>().All(c => c.FromNeuron is IHiddenNeuron)));
        }

        [TestMethod]
        public void DFFNeuralNetwork_WithLayerNeuronCountsAndNoHiddenLayers_InitializesConnectionsBetweenInputAndOutputLayers()
        {
            // Arrange
            _network = new DFFNeuralNetwork(3, 0, 0, 2);

            var inputLayer = _network.Layers.OfType<IInputLayer>().First();
            var outputLayer = _network.Layers.OfType<IOutputLayer>().First();

            // Assert
            Assert.IsTrue(inputLayer.Neurons.All(n => n.Connections.Count() == 2));
            Assert.IsTrue(inputLayer.Neurons.All(n => n.Connections.OfType<IOutgoingConnection>().Count() == 2));
            Assert.IsTrue(inputLayer.Neurons.All(n => n.Connections.OfType<IOutgoingConnection>().All(c => c.ToNeuron is IOutputNeuron)));

            Assert.IsTrue(outputLayer.Neurons.All(n => n.Connections.Count() == 3));
            Assert.IsTrue(outputLayer.Neurons.All(n => n.Connections.OfType<IIncomingConnection>().Count() == 3));
            Assert.IsTrue(outputLayer.Neurons.All(n => n.Connections.OfType<IIncomingConnection>().All(c => c.FromNeuron is IInputNeuron)));
        }

        #endregion
        #region Testing void RandomizeNetwork()...

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNullLayers_ThrowsException()
        {
            // Arrange 
            _network.Layers = null;

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithoutInputLayer_ThrowsException()
        {
            // Arrange 
            _network.Layers = new List<INetworkLayer>() { _outputLayer };

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithMoreThan1InputLayer_ThrowsException()
        {
            // Arrange 
            _network.Layers = new List<INetworkLayer>() { _inputLayer, _inputLayer, _outputLayer };

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithoutOutputLayer_ThrowsException()
        {
            // Arrange 
            _network.Layers = new List<INetworkLayer>() { _inputLayer };

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithMoreThan1OutputLayer_ThrowsException()
        {
            // Arrange 
            _network.Layers = new List<INetworkLayer>() { _inputLayer, _outputLayer, _outputLayer };

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNullLayer_ThrowsException()
        {
            // Arrange 
            _network.Layers = new List<INetworkLayer>() { _inputLayer, null, _outputLayer };

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNullLayerNeuron_ThrowsException()
        {
            // Arrange 
            var layerNeurons = new List<IHiddenNeuron>();
            layerNeurons = _hiddenLayer.Neurons.Cast<IHiddenNeuron>().ToList();
            layerNeurons.Add(null);

            var layerWithNullNeuron = new HiddenLayer(99, layerNeurons);
            var networkLayers = _network.Layers.ToList();
            networkLayers.Add(layerWithNullNeuron);

            _network.Layers = networkLayers;

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNullConnections_ThrowsException()
        {
            // Arrange 
            _hiddenLayer.Neurons.ForEach(n => n.Connections = null).ToList();

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithEmptyConnections_ThrowsException()
        {
            // Arrange 
            _hiddenLayer.Neurons.ForEach(n => n.Connections = new List<INeuronConnection>()).ToList();

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNonInputNeuronsInInputLayer_ThrowsException()
        {
            // Arrange 
            _inputLayer.Neurons = _hiddenLayer.Neurons;

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNonHiddenNeuronsInHiddenLayer_ThrowsException()
        {
            // Arrange 
            _hiddenLayer.Neurons = _inputLayer.Neurons;

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNonOutputNeuronsInOutputLayer_ThrowsException()
        {
            // Arrange 
            _outputLayer.Neurons = _inputLayer.Neurons;

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNullConnectionsInLayer_ThrowsException()
        {
            // Arrange 
            var outputNeuron = _outputLayer.Neurons.First();
            var invalidConnections = outputNeuron.Connections;
            invalidConnections.Add(null);

            outputNeuron.Connections = invalidConnections;
                
            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithInvalidConnectionsInInputLayer_ThrowsException()
        {
            // Arrange 
            _inputLayer.Neurons.ForEach(n => n.Connections = _outputLayer.Neurons.First().Connections);

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithInvalidConnectionsInOutputLayer_ThrowsException()
        {
            // Arrange 
            _outputLayer.Neurons.ForEach(n => n.Connections = _inputLayer.Neurons.First().Connections);

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithMissingHiddenNeuronIncomingConnections_ThrowsException()
        {
            // Arrange 
            _hiddenLayer.Neurons.ForEach(n => n.Connections = n.Connections.Where(c => c is IOutgoingConnection).ToList());

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithMissingHiddenNeuronOutgoingConnections_ThrowsException()
        {
            // Arrange 
            _hiddenLayer.Neurons.ForEach(n => n.Connections = n.Connections.Where(c => c is IIncomingConnection).ToList());

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithMissingInputNeuronOutgoingConnections_ThrowsException()
        {
            // Arrange 
            _inputLayer.Neurons.ForEach(n => n.Connections = n.Connections.Where(c => c is IIncomingConnection).ToList());

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithMissingOutputNeuronIncomingConnections_ThrowsException()
        {
            // Arrange 
            _outputLayer.Neurons.ForEach(n => n.Connections = n.Connections.Where(c => c is IOutgoingConnection).ToList());

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNullIncomingConnectionNeuron_ThrowsException()
        {
            // Arrange 
            _hiddenLayer.Neurons.First().Connections.OfType<IIncomingConnection>().ForEach(c => c.FromNeuron = null);

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void RandomizeNetwork_WithNullOutgoingConnectionNeuron_ThrowsException()
        {
            // Arrange 
            _hiddenLayer.Neurons.First().Connections.OfType<IOutgoingConnection>().ForEach(c => c.ToNeuron = null);

            // Act
            _network.RandomizeNetwork();
        }

        [TestMethod]
        public void Fail()
        {
            Assert.Fail();
        }

        #endregion
    }
}

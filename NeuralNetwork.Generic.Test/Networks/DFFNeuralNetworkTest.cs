using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Framework.Generic.Utility;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Connections;
using NeuralNetwork.Generic.Datasets;
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

        private IList<INetworkTrainingIteration> _trainingIterations;

        private const int _inputLayerNeuronCount = 3;
        private const int _hiddenLayersCount = 1;
        private const int _hiddenLayerNeuronCount = 5;
        private const int _outputLayerNeuronCount = 2;

        [TestInitialize]
        public void Initialize()
        {
            _network = new DFFNeuralNetwork(_inputLayerNeuronCount, _hiddenLayersCount, _hiddenLayerNeuronCount, _outputLayerNeuronCount);
            _inputLayer = _network.Layers.OfType<IInputLayer>().First();
            _outputLayer = _network.Layers.OfType<IOutputLayer>().First();
            _hiddenLayer = _network.Layers.OfType<IHiddenLayer>().First();

            _trainingIterations = new List<INetworkTrainingIteration>();
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

        #region ValidateNetwork()..
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

        #endregion

        [TestMethod]
        public void RandomizeNetwork_WithValidNetwork_RandomizesActivationLevels()
        {
            // Arrange 
            var hiddenNeuron = _hiddenLayer.Neurons.First();
            hiddenNeuron.ActivationLevel = 0.0;

            // Act
            _network.RandomizeNetwork();

            // Assert
            Assert.IsTrue(hiddenNeuron.ActivationLevel != 0.0);
        }

        [TestMethod]
        public void RandomizeNetwork_WithValidNetwork_RandomizesBiases()
        {
            // Arrange 
            var hiddenNeuron = _hiddenLayer.Neurons.First();
            hiddenNeuron.Bias = 0.0;

            // Act
            _network.RandomizeNetwork();

            // Assert
            Assert.IsTrue(hiddenNeuron.Bias != 0.0);
        }

        [TestMethod]
        public void RandomizeNetwork_WithValidNetwork_DoesNothingToIncomingConnectionWeights()
        {
            // Arrange 
            var hiddenNeuron = _hiddenLayer.Neurons.First();
            var incomingConnection = hiddenNeuron.Connections.OfType<IIncomingConnection>().First();
            incomingConnection.Weight = -0.111;

            // Act
            _network.RandomizeNetwork();

            // Assert
            Assert.IsTrue(incomingConnection.Weight != -0.111);
        }

        [TestMethod]
        public void RandomizeNetwork_WithValidNetwork_RandomizesOutgoingConnectionWeights()
        {
            // Arrange 
            var hiddenNeuron = _hiddenLayer.Neurons.First();
            var outgoingConnection = hiddenNeuron.Connections.OfType<IOutgoingConnection>().First();
            outgoingConnection.Weight = 0.0;

            // Act
            _network.RandomizeNetwork();

            // Assert
            Assert.IsTrue(outgoingConnection.Weight != 0.0);
        }

        [TestMethod]
        public void RandomizeNetwork_WithValidNetwork_SetsBothIncomingAndOutgoingConnectionWeights()
        {
            // Arrange 
            var connectedHiddenNeuron = _hiddenLayer.Neurons.First();
            var connectedOutputNeuron = connectedHiddenNeuron.Connections.OfType<IOutgoingConnection>().First().ToNeuron;
            
            // Act
            _network.RandomizeNetwork();

            var outgoingHiddenToOutputConn = connectedHiddenNeuron.Connections.OfType<IOutgoingConnection>().First(c => c.ToNeuron.Id == connectedOutputNeuron.Id);
            var incomingOutputFromHiddenConn = connectedOutputNeuron.Connections.OfType<IIncomingConnection>().First(c => c.FromNeuron.Id == connectedHiddenNeuron.Id);
                       
            // Assert
            Assert.IsTrue(outgoingHiddenToOutputConn.Weight == incomingOutputFromHiddenConn.Weight);
        }

        #endregion
        #region Testing IEnumerable<double> Train(INetworkTrainingDataset dataset)...

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Train_WithNullDataset_ThrowsException()
        {
            // Act
            _network.Train(null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Train_WithEmptyDataset_ThrowsException()
        {
            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Train_WithNullDatasetInputs_ThrowsException()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration
            {
                Inputs = null
            };

            _trainingIterations.Add(trainingIteration);

            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Train_WithEmptyDatasetInputs_ThrowsException()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration();
            _trainingIterations.Add(trainingIteration);

            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Train_WithoutInputsForEachInputNeuron_ThrowsException()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration();
            trainingIteration.Inputs.Add(new NetworkTrainingInput());

            _trainingIterations.Add(trainingIteration);

            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Train_WithoutInvalidInputNeuronIds_ThrowsException()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration();

            // 3 input neurons.
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 0 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 1 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = -1 });

            _trainingIterations.Add(trainingIteration);

            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Train_WithNullDatasetOutputs_ThrowsException()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration
            {
                Outputs = null
            };

            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 0 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 1 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 2 });

            _trainingIterations.Add(trainingIteration);

            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Train_WithEmptyDatasetOutputs_ThrowsException()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration();
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 0 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 1 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 2 });

            _trainingIterations.Add(trainingIteration);

            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Train_WithoutOutputsForEachOutputNeuron_ThrowsException()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration();
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 0 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 1 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 2 });

            trainingIteration.Outputs.Add(new NetworkTrainingOutput());

            _trainingIterations.Add(trainingIteration);

            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Train_WithoutInvalidOutputNeuronIds_ThrowsException()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration();
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 0 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 1 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 2 });

            // 2 output neurons
            trainingIteration.Outputs.Add(new NetworkTrainingOutput() { NeuronId = 8 });
            trainingIteration.Outputs.Add(new NetworkTrainingOutput() { NeuronId = -1 });

            _trainingIterations.Add(trainingIteration);
            _trainingIterations.Add(trainingIteration);

            // Act
            _network.Train(_trainingIterations);
        }

        [TestMethod]
        public void Train_WithValidDataset_UpdatesWeightsAndBiases()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration();
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 0, ActivationLevel = .75 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 1, ActivationLevel = .75 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 2, ActivationLevel = .75 });

            // 2 output neurons
            trainingIteration.Outputs.Add(new NetworkTrainingOutput() { NeuronId = 8, ExpectedActivationLevel = .25 });
            trainingIteration.Outputs.Add(new NetworkTrainingOutput() { NeuronId = 9, ExpectedActivationLevel = .25 });

            _trainingIterations.Add(trainingIteration);
            _trainingIterations.Add(trainingIteration);

            var inputNeuron = _network.Layers.First().Neurons.First();
            var inputNeuronOutgoingConnection = inputNeuron.Connections.OfType<IOutgoingConnection>().First();
            var neuronBiasBefore = inputNeuron.Bias;
            var connectionWeightBefore = inputNeuronOutgoingConnection.Weight;

            // Act
            _network.RandomizeNetwork();
            _network.Train(_trainingIterations);

            // Assert
            Assert.IsTrue(neuronBiasBefore != inputNeuron.Bias);
            Assert.IsTrue(connectionWeightBefore != inputNeuronOutgoingConnection.Weight);
        }

        [TestMethod]
        public void Train_WithValidDataset_ReturnsIterationCosts()
        {
            // Arrange
            var trainingIteration = new NetworkTrainingIteration();
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 0, ActivationLevel = .75 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 1, ActivationLevel = .75 });
            trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 2, ActivationLevel = .75 });

            // 2 output neurons
            trainingIteration.Outputs.Add(new NetworkTrainingOutput() { NeuronId = 8, ExpectedActivationLevel = .25 });
            trainingIteration.Outputs.Add(new NetworkTrainingOutput() { NeuronId = 9, ExpectedActivationLevel = .25 });

            _trainingIterations.Add(trainingIteration);
            _trainingIterations.Add(trainingIteration);
            _trainingIterations.Add(trainingIteration);
            _trainingIterations.Add(trainingIteration);
            _trainingIterations.Add(trainingIteration);
            
            // Act
            _network.RandomizeNetwork();
            var costs = _network.Train(_trainingIterations);
            
            // Assert
            Assert.IsNotNull(costs);
            Assert.IsTrue(costs.Count() == 5);
        }

        [TestMethod]
        public void Train_WithValidDataset_ReturnsImprovingCosts()
        {
            //for (int i = 0; i < 1000; i++)
            //{
                // Arrange
                var trainingIteration = new NetworkTrainingIteration();
                trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 0, ActivationLevel = .75 });
                trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 1, ActivationLevel = .75 });
                trainingIteration.Inputs.Add(new NetworkTrainingInput() { NeuronId = 2, ActivationLevel = .75 });

                // 2 output neurons
                trainingIteration.Outputs.Add(new NetworkTrainingOutput() { NeuronId = 8, ExpectedActivationLevel = .25 });
                trainingIteration.Outputs.Add(new NetworkTrainingOutput() { NeuronId = 9, ExpectedActivationLevel = .25 });

                _trainingIterations.Add(trainingIteration);
                _trainingIterations.Add(trainingIteration);
                _trainingIterations.Add(trainingIteration);
                _trainingIterations.Add(trainingIteration);
                _trainingIterations.Add(trainingIteration);

                // Act
                _network.RandomizeNetwork();
                var trainingIterations = _network.Train(_trainingIterations).ToList();

                // Assert
                var lastTrainingSet = trainingIterations.First();
                foreach (var trainingSet in trainingIterations)
                {
                    Assert.IsTrue(trainingSet.TrainingCost <= lastTrainingSet.TrainingCost);
                    lastTrainingSet = trainingSet;
                }
            //}
        }

        #endregion
        #region Testing IEnumerable<INetworkOutput> ApplyInputs(IEnumerable<INetworkInput> networkInputs)...

        [TestMethod]
        public void ApplyInputs_WithValidNetwork_AssignsActivationLevelsToInputNeurons()
        {
            // Arrange
            _network = new DFFNeuralNetwork(1, 1, 2, 2);

            var inputs = new List<INetworkInput>()
            {
                new NetworkInput() { ActivationLevel = .75 }
            };

            var inputNeuron = _network.Layers.OfType<IInputLayer>().First().Neurons.First();

            // Act
            _network.ApplyInputs(inputs);

            // Assert
            Assert.IsTrue(inputNeuron.ActivationLevel == .75);
        }

        [TestMethod]
        public void ApplyInputs_WithValidNetwork_PropogatesNetworkInputs()
        {
            // Arrange
            var network = new DFFNeuralNetwork(1, 1, 2, 2);

            var inputs = new List<INetworkInput>()
            {
                new NetworkInput() { ActivationLevel = .75 }
            };

            network.RandomizeNetwork();

            IInputNeuron input1 = network.Layers.OfType<IInputLayer>().First().Neurons.First() as IInputNeuron;
            IOutgoingConnection input1Hidden1Conn = input1.Connections.OfType<IOutgoingConnection>().First();
            IOutgoingConnection input1Hidden2Conn = input1.Connections.OfType<IOutgoingConnection>().ToList()[1];

            IHiddenNeuron hidden1 = input1Hidden1Conn.ToNeuron as IHiddenNeuron;
            IOutgoingConnection hidden1Output1Conn = hidden1.Connections.OfType<IOutgoingConnection>().First();
            IOutgoingConnection hidden1Output2Conn = hidden1.Connections.OfType<IOutgoingConnection>().ToList()[1];

            IHiddenNeuron hidden2 = input1Hidden2Conn.ToNeuron as IHiddenNeuron;
            IOutgoingConnection hidden2Output1Conn = hidden2.Connections.OfType<IOutgoingConnection>().First();
            IOutgoingConnection hidden2Output2Conn = hidden2.Connections.OfType<IOutgoingConnection>().ToList()[1];

            IOutputNeuron output1 = hidden1Output1Conn.ToNeuron as IOutputNeuron;
            IOutputNeuron output2 = hidden1Output2Conn.ToNeuron as IOutputNeuron;
            
            // Act
            network.ApplyInputs(inputs);

            var outputLayer = network.Layers.OfType<IOutputLayer>().First();
            IOutputNeuron output1After = outputLayer.Neurons.First() as IOutputNeuron;
            IOutputNeuron output2After = outputLayer.Neurons.ToList()[1] as IOutputNeuron;

            var h1Activation = ((.75 * input1Hidden1Conn.Weight) + hidden1.Bias);
            var h2Activation = ((.75 * input1Hidden2Conn.Weight) + hidden2.Bias);
            var o1Activation = (h1Activation * hidden1Output1Conn.Weight) + (h2Activation * hidden2Output1Conn.Weight) + output1.Bias;
            var o2Activation = (h1Activation * hidden1Output2Conn.Weight) + (h2Activation * hidden2Output2Conn.Weight) + output2.Bias;

            // Assert
            Assert.IsTrue(o1Activation == output1After.ActivationLevel);
            Assert.IsTrue(o2Activation == output2After.ActivationLevel);
        }

        #endregion
        #region Testing IEnumerable<INetworkOutput> GetNeuronOutputs()...

        [TestMethod]
        public void GetNeuronOutputs_WithNullLayers_ReturnsNull()
        {
            // Arrange 
            _network.Layers = null;

            // Act
            var outputs = _network.GetNeuronOutputs();

            // Assert
            Assert.IsNull(outputs);
        }

        [TestMethod]
        public void GetNeuronOutputs_WithoutOutputLayers_ReturnsNull()
        {
            // Arrange 
            _network.Layers = _network.Layers.OfType<IInputLayer>();

            // Act
            var outputs = _network.GetNeuronOutputs();

            // Assert
            Assert.IsNull(outputs);
        }

        [TestMethod]
        public void GetNeuronOutputs_WithNullLayerNeurons_ReturnsEmpty()
        {
            // Arrange 
            _outputLayer.Neurons = null;

            // Act
            var outputs = _network.GetNeuronOutputs();

            // Assert
            Assert.IsNotNull(outputs);
            Assert.IsTrue(outputs.Count() == 0);
        }

        [TestMethod]
        public void GetNeuronOutputs_WithNullOutputLayerNeurons_ReturnsAllButNullNeuron()
        {
            // Arrange 
            var outputNeurons = _outputLayer.Neurons.ToList();
            outputNeurons.Add(null);

            _outputLayer.Neurons = outputNeurons;

            // Act
            var outputs = _network.GetNeuronOutputs();

            // Assert
            Assert.IsTrue(outputs.Count() == outputNeurons.Count - 1);
        }

        [TestMethod]
        public void GetNeuronOutputs_WithOutputLayerNeurons_ReturnsNeuronOutputs()
        {
            // Arrange 
            var outputNeurons = _outputLayer.Neurons.ToList();

            var neuron = outputNeurons[0];

            // Act
            var outputs = _network.GetNeuronOutputs();
            var neuronOutput = outputs.First(o => o.NeuronId == neuron.Id);

            // Assert
            Assert.IsTrue(outputs.Count() == outputNeurons.Count);
            Assert.IsTrue(neuron.ActivationLevel == neuronOutput.ActivationLevel);
        }

        #endregion
        #region Testing PRIVATE METHODS... (Sanity check)

        //[TestMethod]
        //public void CalculateCost_CalculatesCorrectly()
        //{
        //    // Arrange
        //    var network = new DFFNeuralNetwork(1, 1, 2, 2);    // 1 input, 2 hidden, 2 output
        //    _network.Layers.OfType<IOutputLayer>().First().Neurons.ForEach(n => 
        //    {
        //        n.ActivationLevel = .5;
        //    });

        //    var testCase = new NetworkTrainingIteration()
        //    {
        //        Outputs = new List<INetworkTrainingOutput>()
        //        {
        //            new NetworkTrainingOutput() { ExpectedActivationLevel = .25, NeuronId = 3 },
        //            new NetworkTrainingOutput() { ExpectedActivationLevel = .25, NeuronId = 4 }
        //        }
        //    };

        //    // Act
        //    var cost = network.CalculateCost(testCase);
        //    var calculatedCost = 2.0 * Math.Pow(.5 - .25, 2) / 2.0;

        //    // Assert
        //    Assert.IsTrue(cost == calculatedCost);
        //}

        #endregion
    }
}

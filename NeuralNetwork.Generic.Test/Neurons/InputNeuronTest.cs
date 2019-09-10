using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Neurons
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public class InputNeuronTest
    {
        private IInputNeuron _inputNeuron;
        private IHiddenNeuron _hiddenNeuron;
        private IOutputNeuron _outputNeuron;

        [TestInitialize]
        public void Initialize()
        {
            _inputNeuron = new InputNeuron();
            _hiddenNeuron = new HiddenNeuron();
            _outputNeuron = new OutputNeuron();
        }

        #region Testing Properties

        [TestMethod]
        public void GetSetProperties()
        {
            // Arrange
            _inputNeuron.Id = 3;
            _inputNeuron.ActivationLevel = .75;
            _inputNeuron.Bias = .33;
            _inputNeuron.Description = "Fake Neuron";

            // Assert
            Assert.IsTrue(_inputNeuron.Id == 3);
            Assert.IsTrue(_inputNeuron.ActivationLevel == .75);
            Assert.IsTrue(_inputNeuron.Bias == .33);
            Assert.IsTrue(_inputNeuron.Description == "Fake Neuron");
        }

        #endregion
        #region Testing InputNeuron()...

        [TestMethod]
        public void InputNeuron_DoesNothing()
        {
            // Arrange
            var inputNeuron = new InputNeuron();

            // Assert
            Assert.IsNotNull(inputNeuron.Connections);
            Assert.IsTrue(inputNeuron.Connections.Count == 0);
        }

        #endregion
        #region Testing void GenerateConnectionsWith(IHiddenNeuron neuron)...

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GenerateConnectionsWith_WithNullNeuronHidden_ThrowsException()
        {
            // Act
            _inputNeuron.GenerateConnectionsWith((IHiddenNeuron)null);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionHidden_DoesNotAddNewOutgoingConnection()
        {
            // Act
            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);

            // Assert
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionHidden_CreatesOutgoingConnection()
        {
            // Act
            var connectionsBefore = _inputNeuron.GetOutgoingConnections().Count();

            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().Count() == 1);
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().First().ToNeuron == _hiddenNeuron);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionHidden_DoesNotAddNewIncomingConnection()
        {
            // Act
            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);

            // Assert
            Assert.IsTrue(_hiddenNeuron.GetIncomingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionHidden_CreatesIncomingConnection()
        {
            // Act
            var connectionsBefore = _hiddenNeuron.GetIncomingConnections().Count();

            _inputNeuron.GenerateConnectionsWith(_hiddenNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_hiddenNeuron.GetIncomingConnections().Count() == 1);
            Assert.IsTrue(_hiddenNeuron.GetIncomingConnections().First().FromNeuron == _inputNeuron);
        }

        #endregion
        #region Testing void GenerateConnectionsWith(IOutputNeuron neuron)...

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GenerateConnectionsWith_WithNullNeuronOutput_ThrowsException()
        {
            // Act
            _inputNeuron.GenerateConnectionsWith((IOutputNeuron)null);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionOutput_DoesNotAddNewOutgoingConnection()
        {
            // Act
            _inputNeuron.GenerateConnectionsWith(_outputNeuron);
            _inputNeuron.GenerateConnectionsWith(_outputNeuron);
            _inputNeuron.GenerateConnectionsWith(_outputNeuron);
            _inputNeuron.GenerateConnectionsWith(_outputNeuron);

            // Assert
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionOutput_CreatesOutgoingConnection()
        {
            // Act
            var connectionsBefore = _inputNeuron.GetOutgoingConnections().Count();

            _inputNeuron.GenerateConnectionsWith(_outputNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().Count() == 1);
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().First().ToNeuron == _outputNeuron);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionOutput_DoesNotAddNewIncomingConnection()
        {
            // Act
            _inputNeuron.GenerateConnectionsWith(_outputNeuron);
            _inputNeuron.GenerateConnectionsWith(_outputNeuron);
            _inputNeuron.GenerateConnectionsWith(_outputNeuron);
            _inputNeuron.GenerateConnectionsWith(_outputNeuron);

            // Assert
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionOutput_CreatesIncomingConnection()
        {
            // Act
            var connectionsBefore = _outputNeuron.GetIncomingConnections().Count();

            _inputNeuron.GenerateConnectionsWith(_outputNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().Count() == 1);
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().First().FromNeuron == _inputNeuron);
        }

        #endregion
    }
}

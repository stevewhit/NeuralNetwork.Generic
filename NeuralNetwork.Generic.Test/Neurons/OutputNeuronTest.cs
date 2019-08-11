using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Neurons
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public class OutputNeuronTest
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

        #region Testing OutputNeuron()...

        [TestMethod]
        public void OutputNeuron_DoesNothing()
        {
            // Arrange
            var outputNeuron = new OutputNeuron();

            // Assert
            Assert.IsNotNull(outputNeuron.Connections);
            Assert.IsTrue(outputNeuron.Connections.Count == 0);
        }

        #endregion
        #region Testing void GenerateConnectionsWith(IInputNeuron neuron)...

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GenerateConnectionsWith_WithNullNeuronInput_ThrowsException()
        {
            // Act
            _outputNeuron.GenerateConnectionsWith((IInputNeuron)null);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionInput_DoesNotAddNewIncomingConnection()
        {
            // Act
            _outputNeuron.GenerateConnectionsWith(_inputNeuron);
            _outputNeuron.GenerateConnectionsWith(_inputNeuron);
            _outputNeuron.GenerateConnectionsWith(_inputNeuron);
            _outputNeuron.GenerateConnectionsWith(_inputNeuron);

            // Assert
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionInput_CreatesIncomingConnection()
        {
            // Act
            var connectionsBefore = _outputNeuron.GetIncomingConnections().Count();

            _outputNeuron.GenerateConnectionsWith(_inputNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().Count() == 1);
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().First().FromNeuron == _inputNeuron);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionInput_DoesNotAddNewOutgoingConnection()
        {
            // Act
            _outputNeuron.GenerateConnectionsWith(_inputNeuron);
            _outputNeuron.GenerateConnectionsWith(_inputNeuron);
            _outputNeuron.GenerateConnectionsWith(_inputNeuron);
            _outputNeuron.GenerateConnectionsWith(_inputNeuron);

            // Assert
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionInput_CreatesOutgoingConnection()
        {
            // Act
            var connectionsBefore = _inputNeuron.GetOutgoingConnections().Count();

            _outputNeuron.GenerateConnectionsWith(_inputNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().Count() == 1);
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().First().ToNeuron == _outputNeuron);
        }

        #endregion
        #region Testing void GenerateConnectionsWith(IHiddenNeuron neuron)...

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GenerateConnectionsWith_WithNullNeuronHidden_ThrowsException()
        {
            // Act
            _outputNeuron.GenerateConnectionsWith((IHiddenNeuron)null);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionHidden_DoesNotAddNewIncomingConnection()
        {
            // Act
            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);

            // Assert
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionHidden_CreatesIncomingConnection()
        {
            // Act
            var connectionsBefore = _outputNeuron.GetIncomingConnections().Count();

            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().Count() == 1);
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().First().FromNeuron == _hiddenNeuron);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionHidden_DoesNotAddNewOutgoingConnection()
        {
            // Act
            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);
            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);

            // Assert
            Assert.IsTrue(_hiddenNeuron.GetOutgoingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionHidden_CreatesOutgoingConnection()
        {
            // Act
            var connectionsBefore = _hiddenNeuron.GetOutgoingConnections().Count();

            _outputNeuron.GenerateConnectionsWith(_hiddenNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_hiddenNeuron.GetOutgoingConnections().Count() == 1);
            Assert.IsTrue(_hiddenNeuron.GetOutgoingConnections().First().ToNeuron == _outputNeuron);
        }

        #endregion
    }
}

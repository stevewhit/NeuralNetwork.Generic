using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Neurons
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public class HiddenNeuronTest
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

        #region Testing HiddenNeuron()...

        [TestMethod]
        public void HiddenNeuron_DoesNothing()
        {
            // Arrange
            var hiddenNeuron = new HiddenNeuron();

            // Assert
            Assert.IsNotNull(hiddenNeuron.Connections);
            Assert.IsTrue(hiddenNeuron.Connections.Count == 0);
        }

        #endregion
        #region Testing void GenerateConnectionsWith(IInputNeuron neuron)...

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GenerateConnectionsWith_WithNullNeuronInput_ThrowsException()
        {
            // Act
            _hiddenNeuron.GenerateConnectionsWith((IInputNeuron)null);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionInput_DoesNotAddNewIncomingConnection()
        {
            // Act
            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);

            // Assert
            Assert.IsTrue(_hiddenNeuron.GetIncomingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionInput_CreatesIncomingConnection()
        {
            // Act
            var connectionsBefore = _hiddenNeuron.GetIncomingConnections().Count();

            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_hiddenNeuron.GetIncomingConnections().Count() == 1);
            Assert.IsTrue(_hiddenNeuron.GetIncomingConnections().First().FromNeuron == _inputNeuron);
        }
        
        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionInput_DoesNotAddNewOutgoingConnection()
        {
            // Act
            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);

            // Assert
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionInput_CreatesOutgoingConnection()
        {
            // Act
            var connectionsBefore = _inputNeuron.GetOutgoingConnections().Count();

            _hiddenNeuron.GenerateConnectionsWith(_inputNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().Count() == 1);
            Assert.IsTrue(_inputNeuron.GetOutgoingConnections().First().ToNeuron == _hiddenNeuron);
        }

        #endregion
        #region Testing void GenerateConnectionsWith(IOutputNeuron neuron)...

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GenerateConnectionsWith_WithNullNeuronOutput_ThrowsException()
        {
            // Act
            _hiddenNeuron.GenerateConnectionsWith((IOutputNeuron)null);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionOutput_DoesNotAddNewOutgoingConnection()
        {
            // Act
            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);

            // Assert
            Assert.IsTrue(_hiddenNeuron.GetOutgoingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionOutput_CreatesOutgoingConnection()
        {
            // Act
            var connectionsBefore = _hiddenNeuron.GetOutgoingConnections().Count();

            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_hiddenNeuron.GetOutgoingConnections().Count() == 1);
            Assert.IsTrue(_hiddenNeuron.GetOutgoingConnections().First().ToNeuron == _outputNeuron);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithExistingConnectionOutput_DoesNotAddNewIncomingConnection()
        {
            // Act
            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);
            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);

            // Assert
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().Count() == 1);
        }

        [TestMethod]
        public void GenerateConnectionsWith_WithoutConnectionOutput_CreatesIncomingConnection()
        {
            // Act
            var connectionsBefore = _outputNeuron.GetIncomingConnections().Count();

            _hiddenNeuron.GenerateConnectionsWith(_outputNeuron);

            // Assert
            Assert.IsTrue(connectionsBefore == 0);
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().Count() == 1);
            Assert.IsTrue(_outputNeuron.GetIncomingConnections().First().FromNeuron == _hiddenNeuron);
        }

        #endregion
    }
}

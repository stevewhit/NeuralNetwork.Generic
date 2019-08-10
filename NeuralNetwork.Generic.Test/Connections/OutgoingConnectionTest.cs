using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Connections;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Connections
{
    [TestClass]
    public class OutgoingConnectionTest
    {
        private IHiddenNeuron _neuron;
        private IOutgoingConnection _connection;
        
        [TestInitialize]
        public void Initialize()
        {
            _neuron = new HiddenNeuron();
            _connection = new OutgoingConnection(_neuron);
        }

        #region Testing Constructor OutgoingConnection()...
        [TestMethod]
        public void OutgoingConnection_DoesNothing()
        {
            // Arrange
            var connection = new OutgoingConnection();

            // Assert
            Assert.IsNull(connection.ToNeuron);
        }
        #endregion
        #region Testing Constructor OutgoingConnection(INeuron toNeuron)...
        [TestMethod]
        public void OutgoingConnection_WithNeuron_InitializesNeuron()
        {
            // Assert
            Assert.IsNotNull(_connection.ToNeuron);
            Assert.IsTrue(_connection.ToNeuron == _neuron);
        }
        #endregion
    }
}

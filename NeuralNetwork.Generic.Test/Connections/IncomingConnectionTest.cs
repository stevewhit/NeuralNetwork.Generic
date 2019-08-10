using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Connections;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Connections
{
    [TestClass]
    public class IncomingConnectionTest
    {
        private IHiddenNeuron _neuron;
        private IIncomingConnection _connection;

        [TestInitialize]
        public void Initialize()
        {
            _neuron = new HiddenNeuron();
            _connection = new IncomingConnection(_neuron);
        }

        #region Testing Constructor IncomingConnection()...
        [TestMethod]
        public void IncomingConnection_DoesNothing()
        {
            // Arrange
            var connection = new IncomingConnection();

            // Assert
            Assert.IsNull(connection.FromNeuron);
        }
        #endregion
        #region Testing Constructor IncomingConnection(INeuron fromNeuron)...
        [TestMethod]
        public void IncomingConnection_WithNeuron_InitializesNeuron()
        {
            // Assert
            Assert.IsNotNull(_connection.FromNeuron);
            Assert.IsTrue(_connection.FromNeuron == _neuron);
        }
        #endregion
    }
}

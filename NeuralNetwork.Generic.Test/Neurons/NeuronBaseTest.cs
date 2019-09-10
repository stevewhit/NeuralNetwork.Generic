using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Connections;
using NeuralNetwork.Generic.Neurons;
using NeuralNetwork.Generic.Test.Builders;

namespace NeuralNetwork.Generic.Test.Neurons
{
    [TestClass]
    public class NeuronBaseTest
    {
        private FakeNeuron _neuron;

        [TestInitialize]
        public void Initialize()
        {
            _neuron = new FakeNeuron();
        }

        #region Testing Properties...

        [TestMethod]
        public void GetSetProperties()
        {
            // Arrange
            _neuron.Id = 3;
            _neuron.ActivationLevel = .75;
            _neuron.Bias = .33;
            _neuron.Description = "Fake Neuron";

            // Assert
            Assert.IsTrue(_neuron.Id == 3);
            Assert.IsTrue(_neuron.ActivationLevel == .75);
            Assert.IsTrue(_neuron.Bias == .33);
            Assert.IsTrue(_neuron.Description == "Fake Neuron");
        }


        #endregion
        #region Testing NeuronBase()...

        [TestMethod]
        public void NeuronBase_InitializesConnections()
        {
            // Act
            _neuron = new FakeNeuron();

            // Assert
            Assert.IsNotNull(_neuron.Connections);
            Assert.IsTrue(_neuron.Connections.Count == 0);
        }

        #endregion
        #region Testing IEnumerable<IOutgoingConnection> GetOutgoingConnections()...

        [TestMethod]
        public void GetOutgoingConnections_WithNoConnections_ReturnsEmpty()
        {
            // Assert
            Assert.IsTrue(_neuron.GetOutgoingConnections().Count() == 0);
        }

        [TestMethod]
        public void GetOutgoingConnections_WithNullConnections_ReturnsNull()
        {
            // Arrange
            _neuron.Connections = null;

            // Act
            var outConnections = _neuron.GetOutgoingConnections();

            // Assert
            Assert.IsNull(outConnections);
        }

        [TestMethod]
        public void GetOutgoingConnections_WithoutOutgoingConnections_ReturnsEmpty()
        {
            // Arrange
            IHiddenNeuron hiddenNeuron = new HiddenNeuron();
            IIncomingConnection inConnection = new IncomingConnection(hiddenNeuron);

            _neuron.Connections.Add(inConnection);

            // Act
            var outConnections = _neuron.GetOutgoingConnections();

            // Assert
            Assert.IsTrue(outConnections.Count() == 0);
        }

        [TestMethod]
        public void GetOutgoingConnections_WithOutgoingConnections_ReturnsConnections()
        {
            // Arrange
            IHiddenNeuron hiddenNeuron = new HiddenNeuron();
            IOutgoingConnection outConnection = new OutgoingConnection(hiddenNeuron);

            _neuron.Connections.Add(outConnection);

            // Act
            var outConnections = _neuron.GetOutgoingConnections();

            // Assert
            Assert.IsTrue(outConnections.Count() == 1);
            Assert.IsTrue(outConnections.First() == outConnection);
        }

        #endregion
        #region Testing IEnumerable<IIncomingConnections> GetIncomingConnections()...

        [TestMethod]
        public void GetIncomingConnections_WithNoConnections_ReturnsEmpty()
        {
            // Assert
            Assert.IsTrue(_neuron.GetIncomingConnections().Count() == 0);
        }

        [TestMethod]
        public void GetIncomingConnections_WithNullConnections_ReturnsNull()
        {
            // Arrange
            _neuron.Connections = null;

            // Act
            var inConnections = _neuron.GetIncomingConnections();

            // Assert
            Assert.IsNull(inConnections);
        }

        [TestMethod]
        public void GetIncomingConnections_WithoutIncomingConnections_ReturnsEmpty()
        {
            // Arrange
            IHiddenNeuron hiddenNeuron = new HiddenNeuron();
            IOutgoingConnection outConnection = new OutgoingConnection(hiddenNeuron);

            _neuron.Connections.Add(outConnection);

            // Act
            var inConnections = _neuron.GetIncomingConnections();

            // Assert
            Assert.IsTrue(inConnections.Count() == 0);
        }

        [TestMethod]
        public void GetIncomingConnections_WithIncomingConnections_ReturnsConnections()
        {
            // Arrange
            IHiddenNeuron hiddenNeuron = new HiddenNeuron();
            IIncomingConnection inConnection = new IncomingConnection(hiddenNeuron);

            _neuron.Connections.Add(inConnection);

            // Act
            var inConnections = _neuron.GetIncomingConnections();

            // Assert
            Assert.IsTrue(inConnections.Count() == 1);
            Assert.IsTrue(inConnections.First() == inConnection);
        }

        #endregion
    }
}

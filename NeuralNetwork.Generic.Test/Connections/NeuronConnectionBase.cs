using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Test.Builders;

namespace NeuralNetwork.Generic.Test.Connections
{
    [TestClass]
    public class NeuronConnectionBase
    {
        private FakeNeuronConnection _connection;

        [TestInitialize]
        public void Initialize()
        {
            _connection = new FakeNeuronConnection();
        }

        #region Testing Properties

        [TestMethod]
        public void Weight_GetSet()
        {
            // Act
            _connection.Weight = .04;

            // Assert
            Assert.IsTrue(_connection.Weight == .04);
        }

        #endregion
        #region Testing NeuronConnectionBase()...

        [TestMethod]
        public void Fail()
        {
            // Act
            var connection = new FakeNeuronConnection();

            // Assert
            Assert.IsNotNull(connection);
        }

        #endregion
    }
}

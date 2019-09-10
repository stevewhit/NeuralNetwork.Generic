using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Datasets;

namespace NeuralNetwork.Generic.Test.Datasets
{
    [TestClass]
    public class NetworkInputTest
    {
        private INetworkInput _input;

        [TestInitialize]
        public void Initialize()
        {
            _input = new NetworkInput();
        }

        #region Testing Properties...

        [TestMethod]
        public void GetSetProperties()
        {
            // Arrange
            _input.NeuronId = 3;
            _input.Description = "Fake network input";
            _input.ActivationLevel = .75;

            // Assert
            Assert.IsTrue(_input.NeuronId == 3);
            Assert.IsTrue(_input.ActivationLevel == .75);
            Assert.IsTrue(_input.Description == "Fake network input");
        }
        
        #endregion
    }
}

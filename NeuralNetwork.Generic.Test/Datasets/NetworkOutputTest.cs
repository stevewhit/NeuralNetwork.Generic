using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Datasets;

namespace NeuralNetwork.Generic.Test.Datasets
{
    [TestClass]
    public class NetworkOutputTest
    {
        private INetworkOutput _output;

        [TestInitialize]
        public void Initialize()
        {
            _output = new NetworkOutput();
        }

        #region Testing Properties...

        [TestMethod]
        public void GetSetProperties()
        {
            // Arrange
            _output.NeuronId = 3;
            _output.Description = "Fake network output";
            _output.ActivationLevel = .75;

            // Assert
            Assert.IsTrue(_output.NeuronId == 3);
            Assert.IsTrue(_output.ActivationLevel == .75);
            Assert.IsTrue(_output.Description == "Fake network output");
        }

        #endregion
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Test.Builders;

namespace NeuralNetwork.Generic.Test.Networks
{
    [TestClass]
    public class NeuralNetworkBaseTest
    {
        #region Testing NeuralNetworkBase()...

        [TestMethod]
        public void NeuralNetworkBase_InitializesLayers()
        {
            // Arrange
            var network = new FakeNeuralNetwork();

            // Assert
            Assert.IsNotNull(network.Layers);
        }

        #endregion
        #region Testing NeuralNetworkBase(IEnumerable<INetworkLayer> layers)...

        [TestMethod]
        public void NeuralNetworkBase_WithLayers_InitializesLayers()
        {
            // Arrange
            var layer = new InputLayer(1);
            var network = new FakeNeuralNetwork(new List<INetworkLayer>() { layer });

            // Assert
            Assert.IsNotNull(network.Layers);
            Assert.IsTrue(network.Layers.Count() == 1);
            Assert.IsTrue(network.Layers.First() == layer);
        }

        #endregion
    }
}

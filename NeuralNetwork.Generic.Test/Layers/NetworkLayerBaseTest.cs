using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Neurons;
using NeuralNetwork.Generic.Test.Builders;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Generic.Test.Layers
{
    [TestClass]
    public class NetworkLayerBaseTest
    {
        private FakeNeuronLayer _layer;

        [TestInitialize]
        public void Initialize()
        {
            _layer = new FakeNeuronLayer(1);
        }

        #region Testing Properties...

        [TestMethod]
        public void Properties_GetSet()
        {
            // Arrange
            var neuronToAdd = new InputNeuron();

            _layer.SortOrder = 11;
            _layer.Neurons = new List<INeuron>()
            {
                neuronToAdd
            };

            // Assert
            Assert.IsTrue(_layer.SortOrder == 11);
            Assert.IsTrue(_layer.Neurons.First() == neuronToAdd);
        }

        #endregion
        #region Testing NetworkLayerBase(int sortOrder)...

        [TestMethod]
        public void NetworkLayerBase_InitializesProperties()
        {
            // Assert
            Assert.IsTrue(_layer.SortOrder == 1);
            Assert.IsNotNull(_layer.Neurons);
            Assert.IsTrue(_layer.Neurons.Count() == 0);
        }

        #endregion
        #region Testing NetworkLayerBase(IEnumerable<INeuron> neurons, int sortOrder)...

        [TestMethod]
        public void NetworkLayerBase_WithArguments_InitializesProperties()
        {
            // Arrange
            var neurons = new List<INeuron>()
            {
               new InputNeuron()
            };

            _layer = new FakeNeuronLayer(1, neurons);

            // Assert
            Assert.IsTrue(_layer.SortOrder == 1);
            Assert.IsNotNull(_layer.Neurons);
            Assert.IsTrue(_layer.Neurons.Count() == 1);
        }

        #endregion
    }
}

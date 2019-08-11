using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Layers
{
    [TestClass]
    public class HiddenLayerTest
    {
        private IHiddenLayer _layer;

        [TestInitialize]
        public void Initialize()
        {
            _layer = new HiddenLayer(1);
        }

        #region Testing HiddenLayer(int sortOrder)...

        [TestMethod]
        public void HiddenLayer_WithSortOrder_InitializesProperty()
        {
            // Arrange
            _layer = new HiddenLayer(5);

            // Assert
            Assert.IsTrue(_layer.SortOrder == 5);
        }

        #endregion
        #region Testing HiddenLayer(int sortOrder, IEnumerable<IHiddenNeuron> hiddenNeurons)...

        [TestMethod]
        public void HiddenLayer_WithSortOrderAndNeurons_InitializesProperty()
        {
            // Arrange
            var hiddenNeuron = new HiddenNeuron();

            _layer = new HiddenLayer(5, new List<IHiddenNeuron>() { hiddenNeuron });

            // Assert
            Assert.IsTrue(_layer.Neurons.Count() == 1);
            Assert.IsTrue(_layer.Neurons.First() == hiddenNeuron);
        }

        #endregion
    }
}

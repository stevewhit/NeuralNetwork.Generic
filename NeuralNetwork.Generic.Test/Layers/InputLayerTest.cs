using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Layers
{
    [TestClass]
    public class InputLayerTest
    {
        private IInputLayer _layer;

        [TestInitialize]
        public void Initialize()
        {
            _layer = new InputLayer(1);
        }

        #region Testing InputLayer(int sortOrder)...

        [TestMethod]
        public void InputLayer_WithSortOrder_InitializesProperty()
        {
            // Arrange
            _layer = new InputLayer(5);

            // Assert
            Assert.IsTrue(_layer.SortOrder == 5);
        }

        #endregion
        #region Testing InputLayer(int sortOrder, IEnumerable<IInputNeuron> inputNeurons)...

        [TestMethod]
        public void InputLayer_WithSortOrderAndNeurons_InitializesProperty()
        {
            // Arrange
            var inputNeuron = new InputNeuron();

            _layer = new InputLayer(5, new List<IInputNeuron>() { inputNeuron });

            // Assert
            Assert.IsTrue(_layer.Neurons.Count() == 1);
            Assert.IsTrue(_layer.Neurons.First() == inputNeuron);
        }

        #endregion
    }
}

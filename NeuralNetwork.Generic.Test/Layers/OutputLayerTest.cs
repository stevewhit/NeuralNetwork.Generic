using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Generic.Layers;
using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Test.Layers
{
    [TestClass]
    public class OutputLayerTest
    {
        private IOutputLayer _layer;

        [TestInitialize]
        public void Initialize()
        {
            _layer = new OutputLayer(1);
        }

        #region Testing OutputLayer(int sortOrder)...

        [TestMethod]
        public void OutputLayer_WithSortOrder_InitializesProperty()
        {
            // Arrange
            _layer = new OutputLayer(5);

            // Assert
            Assert.IsTrue(_layer.SortOrder == 5);
        }

        #endregion
        #region Testing OutputLayer(int sortOrder, IEnumerable<IOutputNeuron> outputNeurons)...

        [TestMethod]
        public void OutputLayer_WithSortOrderAndNeurons_InitializesProperty()
        {
            // Arrange
            var outputNeuron = new OutputNeuron();

            _layer = new OutputLayer(5, new List<IOutputNeuron>() { outputNeuron });

            // Assert
            Assert.IsTrue(_layer.Neurons.Count() == 1);
            Assert.IsTrue(_layer.Neurons.First() == outputNeuron);
        }

        #endregion
    }
}

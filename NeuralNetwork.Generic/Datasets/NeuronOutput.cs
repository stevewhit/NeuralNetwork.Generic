using NeuralNetwork.Generic.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INeuronOutput
    {
        double ExpectedActivationLevel { get; set; }
        IOutputNeuron Neuron { get; set; }
    }

    public class NeuronOutput : INeuronOutput
    {
        public double ExpectedActivationLevel { get; set; }
        public IOutputNeuron Neuron { get; set; }
    }
}

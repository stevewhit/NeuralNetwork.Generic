using NeuralNetwork.Generic.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INeuronInput
    {
        double ActivationLevel { get; set; }
        IInputNeuron Neuron { get; set; }
    }

    public class NeuronInput : INeuronInput
    {
        public double ActivationLevel { get; set; }
        public IInputNeuron Neuron { get; set; }
    }
}

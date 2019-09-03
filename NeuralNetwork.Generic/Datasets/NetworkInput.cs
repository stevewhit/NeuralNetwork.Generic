using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkInput
    {
        /// <summary>
        /// The id of this neuron.
        /// </summary>
        int NeuronId { get; set; }
        
        /// <summary>
        /// The activation level of this neuron.
        /// </summary>
        double ActivationLevel { get; set; }
    }

    public class NetworkInput : INetworkInput
    {
        /// <summary>
        /// The id of this neuron.
        /// </summary>
        public int NeuronId { get; set; }
        
        /// <summary>
        /// The activation level of this neuron.
        /// </summary>
        public double ActivationLevel { get; set; }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkOutput
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

    public class NetworkOutput : INetworkOutput
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

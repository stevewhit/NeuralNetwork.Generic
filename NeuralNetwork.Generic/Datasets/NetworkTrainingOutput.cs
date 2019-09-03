using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkTrainingOutput : INetworkOutput
    {
        /// <summary>
        /// The expected activation level of this neuron after the inputs have propogated through the network.
        /// </summary>
        double ExpectedActivationLevel { get; set; }
    }

    public class NetworkTrainingOutput : NetworkOutput, INetworkTrainingOutput
    {
        /// <summary>
        /// The expected activation level of this neuron after the inputs have propogated through the network.
        /// </summary>
        public double ExpectedActivationLevel { get; set; }
    }
}

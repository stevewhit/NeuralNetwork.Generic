using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkInput
    {
        int NeuronId { get; set; }
        double ActivationLevel { get; set; }
    }

    public class NetworkInput
    {
        public int NeuronId { get; set; }
        public double ActivationLevel { get; set; }
    }
}

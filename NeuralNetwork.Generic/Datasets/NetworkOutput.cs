using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkOutput
    {
        int NeuronId { get; set; }
        double ActivationLevel { get; set; }
    }

    public class NetworkOutput : INetworkOutput
    {
        public int NeuronId { get; set; }
        public double ActivationLevel { get; set; }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkTrainingOutput : INetworkOutput
    {
        double ExpectedActivationLevel { get; set; }
    }

    public class NetworkTrainingOutput : NetworkOutput, INetworkTrainingOutput
    {
        public double ExpectedActivationLevel { get; set; }
    }
}

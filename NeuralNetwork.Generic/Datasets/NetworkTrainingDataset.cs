using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkTrainingDataset
    {
        IList<INetworkTrainingIteration> TestCases { get; set; }
    }

    public class NetworkTrainingDataset : INetworkTrainingDataset
    {
        public IList<INetworkTrainingIteration> TestCases { get; set; }
    }
}

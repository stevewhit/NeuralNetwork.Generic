using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkTrainingIteration
    {
        IList<INetworkTrainingInput> Inputs { get; set; }
        IList<INetworkTrainingOutput> Outputs { get; set; }
        double TrainingCost { get; set; }
    }

    public class NetworkTrainingIteration : INetworkTrainingIteration
    {
        public IList<INetworkTrainingInput> Inputs { get; set; }
        public IList<INetworkTrainingOutput> Outputs { get; set; }
        public double TrainingCost { get; set; }
    }
}

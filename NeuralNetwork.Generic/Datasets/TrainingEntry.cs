using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface ITrainingEntry
    {
        IList<ITrainingNeuronInput> Inputs { get; set; }
        IList<ITrainingNeuronOutput> Outputs { get; set; }
    }

    public class TrainingEntry : ITrainingEntry
    {
        public IList<ITrainingNeuronInput> Inputs { get; set; }
        public IList<ITrainingNeuronOutput> Outputs { get; set; }
    }
}

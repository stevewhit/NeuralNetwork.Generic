using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface ITrainingDataset
    {
        IList<ITrainingEntry> Entries { get; set; }
    }

    public class TrainingDataset : ITrainingDataset
    {
        public IList<ITrainingEntry> Entries { get; set; }
    }
}

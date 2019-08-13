using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface ITrainingNeuronInput : INeuronInput
    {

    }
    public class TrainingNeuronInput : NeuronInput, ITrainingNeuronInput
    {

    }
}

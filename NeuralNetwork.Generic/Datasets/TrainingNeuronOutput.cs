using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Datasets
{
    public interface ITrainingNeuronOutput : INeuronOutput
    {
        double ActualActivationLevel { get; set; }
    }
    public class TrainingNeuronOutput : NeuronOutput, ITrainingNeuronOutput
    {
        public double ActualActivationLevel { get; set; }
    }
}

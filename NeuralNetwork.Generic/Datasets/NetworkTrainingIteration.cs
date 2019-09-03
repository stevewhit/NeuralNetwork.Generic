using System.Collections.Generic;

namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkTrainingIteration
    {
        /// <summary>
        /// The training inputs for this iteration.
        /// </summary>
        IList<INetworkTrainingInput> Inputs { get; set; }

        /// <summary>
        /// The training outputs for this iteration.
        /// </summary>
        IList<INetworkTrainingOutput> Outputs { get; set; }

        /// <summary>
        /// The training cost of this iteration.
        /// </summary>
        double TrainingCost { get; set; }
    }

    public class NetworkTrainingIteration : INetworkTrainingIteration
    {
        /// <summary>
        /// The training inputs for this iteration.
        /// </summary>
        public IList<INetworkTrainingInput> Inputs { get; set; }

        /// <summary>
        /// The training outputs for this iteration.
        /// </summary>
        public IList<INetworkTrainingOutput> Outputs { get; set; }

        /// <summary>
        /// The training cost of this iteration.
        /// </summary>
        public double TrainingCost { get; set; }

        public NetworkTrainingIteration()
        {
            Inputs = new List<INetworkTrainingInput>();
            Outputs = new List<INetworkTrainingOutput>();
        }
    }
}

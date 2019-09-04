
namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkOutput
    {
        /// <summary>
        /// The id of this neuron.
        /// </summary>
        int NeuronId { get; set; }
        
        /// <summary>
        /// The activation level of this neuron.
        /// </summary>
        double ActivationLevel { get; set; }

        /// <summary>
        /// The description of the output.
        /// </summary>
        string Description { get; set; }
    }

    public class NetworkOutput : INetworkOutput
    {
        /// <summary>
        /// The id of this neuron.
        /// </summary>
        public int NeuronId { get; set; }
        
        /// <summary>
        /// The activation level of this neuron.
        /// </summary>
        public double ActivationLevel { get; set; }

        /// <summary>
        /// The description of the output.
        /// </summary>
        public string Description { get; set; }
    }
}


namespace NeuralNetwork.Generic.Datasets
{
    public interface INetworkInput
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
        /// The description of the input.
        /// </summary>
        string Description { get; set; }
    }

    public class NetworkInput : INetworkInput
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
        /// The description of the input.
        /// </summary>
        public string Description { get; set; }
    }
}

using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Connections
{
    public interface IIncomingConnection : INeuronConnection
    {
        /// <summary>
        /// The neuron that this connection is coming from.
        /// </summary>
        INeuron FromNeuron { get; set; }
    }

    public class IncomingConnection : NeuronConnectionBase, IIncomingConnection
    {
        /// <summary>
        /// The neuron that this connection is coming from.
        /// </summary>
        public INeuron FromNeuron { get; set; }

        public IncomingConnection()
        {

        }

        public IncomingConnection(INeuron fromNeuron)
        {
            FromNeuron = fromNeuron;
        }
    }
}

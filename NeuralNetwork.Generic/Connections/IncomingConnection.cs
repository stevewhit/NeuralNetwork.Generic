using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Connections
{
    public interface IIncomingConnection : INeuronConnection
    {
        /// <summary>
        /// The neuron that is initiating the connection.
        /// </summary>
        INeuron FromNeuron { get; set; }
    }

    public class IncomingConnection : NeuronConnectionBase, IIncomingConnection
    {
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

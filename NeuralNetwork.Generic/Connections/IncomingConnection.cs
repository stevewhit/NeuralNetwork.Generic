using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Connections
{
    public interface IIncomingConnection : INeuronConnection
    {
        INeuron FromNeuron { get; set; }
    }

    public class IncomingConnection : NeuronConnectionBase, IIncomingConnection
    {
        public INeuron FromNeuron { get; set; }

        public IncomingConnection(INeuron fromNeuron)
        {
            FromNeuron = fromNeuron;
        }
    }
}

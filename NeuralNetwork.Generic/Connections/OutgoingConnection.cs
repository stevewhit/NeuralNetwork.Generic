using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Connections
{
    public interface IOutgoingConnection : INeuronConnection
    {
        INeuron ToNeuron { get; set; }
    }

    public class OutgoingConnection : NeuronConnectionBase, IOutgoingConnection
    {
        public INeuron ToNeuron { get; set; }

        public OutgoingConnection(INeuron toNeuron)
        {
            ToNeuron = toNeuron;
        }
    }
}

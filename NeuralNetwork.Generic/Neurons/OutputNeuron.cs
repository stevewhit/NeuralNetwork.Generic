using System.Linq;
using NeuralNetwork.Generic.Connections;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IOutputNeuron : INeuron
    {
        IIncomingConnection[] GetIncomingConnections();
    }

    public class OutputNeuron : NeuronBase, IOutputNeuron
    {
        public OutputNeuron()
        {

        }

        public OutputNeuron(IIncomingConnection[] connections)
            : base(connections)
        {

        }

        public IIncomingConnection[] GetIncomingConnections() => Connections?.OfType<IIncomingConnection>().ToArray();
    }
}

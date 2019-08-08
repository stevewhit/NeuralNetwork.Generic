using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Generic.Connections;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IOutputNeuron : INeuron
    {
        IEnumerable<IIncomingConnection> GetIncomingConnections();
    }

    public class OutputNeuron : NeuronBase, IOutputNeuron
    {
        public OutputNeuron()
        {

        }

        public OutputNeuron(IEnumerable<IIncomingConnection> connections)
            : base(connections)
        {

        }

        public IEnumerable<IIncomingConnection> GetIncomingConnections() => Connections?.OfType<IIncomingConnection>();
    }
}

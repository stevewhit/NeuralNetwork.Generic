using NeuralNetwork.Generic.Connections;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IHiddenNeuron : INeuron
    {
        IEnumerable<IOutgoingConnection> GetOutgoingConnections();
        IEnumerable<IIncomingConnection> GetIncomingConnections();
    }

    public class HiddenNeuron : NeuronBase, IHiddenNeuron
    {
        public HiddenNeuron()
        {

        }

        public HiddenNeuron(IEnumerable<IOutgoingConnection> connections)
            : base(connections)
        {

        }

        public IEnumerable<IOutgoingConnection> GetOutgoingConnections() => Connections?.OfType<IOutgoingConnection>();
        public IEnumerable<IIncomingConnection> GetIncomingConnections() => Connections?.OfType<IIncomingConnection>();
    }
}

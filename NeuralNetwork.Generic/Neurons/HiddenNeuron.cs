using NeuralNetwork.Generic.Connections;
using System.Linq;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IHiddenNeuron : INeuron
    {
        IOutgoingConnection[] GetOutgoingConnections();
        IIncomingConnection[] GetIncomingConnections();
    }

    public class HiddenNeuron : NeuronBase, IHiddenNeuron
    {
        public HiddenNeuron()
        {

        }

        public HiddenNeuron(IOutgoingConnection[] connections)
            : base(connections)
        {

        }

        public IOutgoingConnection[] GetOutgoingConnections() => Connections?.OfType<IOutgoingConnection>().ToArray();
        public IIncomingConnection[] GetIncomingConnections() => Connections?.OfType<IIncomingConnection>().ToArray();
    }
}

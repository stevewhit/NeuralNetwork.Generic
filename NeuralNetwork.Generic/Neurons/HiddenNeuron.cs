using System.Linq;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IHiddenNeuron : INeuron
    {
        IOutgoingNeuronConnection[] GetOutgoingConnections();
        IIncomingNeuronConnection[] GetIncomingConnections();
    }

    public class HiddenNeuron : NeuronBase, IHiddenNeuron
    {
        public HiddenNeuron()
            : base(null)
        {

        }

        public HiddenNeuron(IOutgoingNeuronConnection[] connections)
            : base(connections)
        {

        }

        public IOutgoingNeuronConnection[] GetOutgoingConnections() => Connections?.OfType<IOutgoingNeuronConnection>().ToArray();
        public IIncomingNeuronConnection[] GetIncomingConnections() => Connections?.OfType<IIncomingNeuronConnection>().ToArray();
    }
}

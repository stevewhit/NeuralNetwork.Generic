using System.Linq;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IInputNeuron : INeuron
    {
        IOutgoingNeuronConnection[] GetOutgoingConnections();
    }

    public class InputNeuron : NeuronBase, IInputNeuron
    {
        public InputNeuron()
            : base(null)
        {

        }

        public InputNeuron(IOutgoingNeuronConnection[] connections)
            : base(connections)
        {

        }
        
        public IOutgoingNeuronConnection[] GetOutgoingConnections() => Connections?.OfType<IOutgoingNeuronConnection>().ToArray();
    }
}

using NeuralNetwork.Generic.Connections;
using System.Linq;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IInputNeuron : INeuron
    {
        IOutgoingConnection[] GetOutgoingConnections();
    }

    public class InputNeuron : NeuronBase, IInputNeuron
    {
        public InputNeuron()
        {

        }

        public InputNeuron(IOutgoingConnection[] connections)
            : base(connections)
        {

        }
        
        public IOutgoingConnection[] GetOutgoingConnections() => Connections?.OfType<IOutgoingConnection>().ToArray();
    }
}

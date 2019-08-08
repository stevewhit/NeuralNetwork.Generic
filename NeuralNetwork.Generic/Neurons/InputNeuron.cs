using NeuralNetwork.Generic.Connections;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IInputNeuron : INeuron
    {
        IEnumerable<IOutgoingConnection> GetOutgoingConnections();
    }

    public class InputNeuron : NeuronBase, IInputNeuron
    {
        public InputNeuron()
        {

        }

        public InputNeuron(IEnumerable<IOutgoingConnection> connections)
            : base(connections)
        {

        }
        
        public IEnumerable<IOutgoingConnection> GetOutgoingConnections() => Connections?.OfType<IOutgoingConnection>();
    }
}

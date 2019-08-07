using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IOutputNeuron : INeuron
    {
        IIncomingNeuronConnection[] GetIncomingConnections();
    }

    public class OutputNeuron : NeuronBase, IOutputNeuron
    {
        public OutputNeuron()
            : base(null)
        {

        }

        public OutputNeuron(IOutgoingNeuronConnection[] connections)
            : base(connections)
        {

        }

        public IIncomingNeuronConnection[] GetIncomingConnections() => Connections?.OfType<IIncomingNeuronConnection>().ToArray();
    }
}

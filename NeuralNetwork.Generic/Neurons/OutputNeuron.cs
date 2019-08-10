using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Generic.Connections;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IOutputNeuron : INeuron
    {
        /// <summary>
        /// Returns all incoming connections to this neuron.
        /// </summary>
        /// <returns>Returns all outgoing connections for this neuron.</returns>
        IEnumerable<IIncomingConnection> GetIncomingConnections();

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a hidden neuron.
        /// </summary>
        /// <param name="neuron">The hidden neuron to connect this neuron to.</param>
        void GenerateConnectionsWith(IHiddenNeuron neuron);
    }

    public class OutputNeuron : NeuronBase, IOutputNeuron
    {
        public OutputNeuron()
        {

        }

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a hidden neuron.
        /// </summary>
        /// <param name="neuron">The hidden neuron to connect this neuron to.</param>
        public void GenerateConnectionsWith(IHiddenNeuron neuron)
        {
            if (neuron == null)
                throw new ArgumentNullException("neuron");

            // Create connection only if this neuron doesn't already have an incoming connection from the hidden neuron.
            if (!GetIncomingConnections().Any(c => c.FromNeuron == neuron))
                Connections.Add(new IncomingConnection(neuron));

            // Create connection only if the hidden neuron doesn't already have an outgoing connection to this neuron.
            if (!neuron.GetOutgoingConnections().Any(c => c.ToNeuron == this))
                neuron.Connections.Add(new OutgoingConnection(this));
        }
    }
}

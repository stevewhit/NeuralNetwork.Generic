using NeuralNetwork.Generic.Connections;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IInputNeuron : INeuron
    {
        /// <summary>
        /// Returns all outgoing connections for this neuron.
        /// </summary>
        /// <returns>Returns all outgoing connections for this neuron.</returns>
        IEnumerable<IOutgoingConnection> GetOutgoingConnections();

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a hidden neuron.
        /// </summary>
        /// <param name="neuron">The hidden neuron to connect this neuron to.</param>
        void GenerateConnectionsWith(IHiddenNeuron neuron);

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a output neuron.
        /// </summary>
        /// <param name="neuron">The output neuron to connect this neuron to.</param>
        void GenerateConnectionsWith(IOutputNeuron neuron);
    }

    public class InputNeuron : NeuronBase, IInputNeuron
    {
        public InputNeuron()
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

            // Create connection only if this neuron doesn't already have an outgoing connection to the hidden neuron.
            if (!GetOutgoingConnections().Any(c => c.ToNeuron == neuron))
                Connections.Add(new OutgoingConnection(neuron));

            // Create connection only if the hidden neuron doesn't already have an incoming connection from this neuron.
            if (!neuron.GetIncomingConnections().Any(c => c.FromNeuron == this))
                neuron.Connections.Add(new IncomingConnection(this));
        }

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a output neuron.
        /// </summary>
        /// <param name="neuron">The output neuron to connect this neuron to.</param>
        public void GenerateConnectionsWith(IOutputNeuron neuron)
        {
            if (neuron == null)
                throw new ArgumentNullException("neuron");

            // Create connection only if this neuron doesn't already have an outgoing connection to the output neuron.
            if (!GetOutgoingConnections().Any(c => c.ToNeuron == neuron))
                Connections.Add(new OutgoingConnection(neuron));

            // Create connection only if the output neuron doesn't already have an incoming connection from this neuron.
            if (!neuron.GetIncomingConnections().Any(c => c.FromNeuron == this))
                neuron.Connections.Add(new IncomingConnection(this));
        }
    }
}

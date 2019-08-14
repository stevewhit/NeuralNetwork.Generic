using NeuralNetwork.Generic.Connections;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Generic.Neurons
{
    public interface IHiddenNeuron : INeuron
    {
        /// <summary>
        /// Returns all outgoing connections for this neuron.
        /// </summary>
        /// <returns>Returns all outgoing connections for this neuron.</returns>
        IEnumerable<IOutgoingConnection> GetOutgoingConnections();

        /// <summary>
        /// Returns all incoming connections to this neuron.
        /// </summary>
        /// <returns>Returns all outgoing connections for this neuron.</returns>
        IEnumerable<IIncomingConnection> GetIncomingConnections();

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a input neuron.
        /// </summary>
        /// <param name="neuron">The input neuron to connect this neuron to.</param>
        void GenerateConnectionsWith(IInputNeuron neuron);

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a previous hidden neuron.
        /// </summary>
        /// <param name="incomingNeuron">The incoming neuron to connect this neuron to.</param>
        void GenerateConnectionsWith(IHiddenNeuron incomingNeuron);

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a output neuron.
        /// </summary>
        /// <param name="neuron">The output neuron to connect this neuron to.</param>
        void GenerateConnectionsWith(IOutputNeuron neuron);
    }

    public class HiddenNeuron : NeuronBase, IHiddenNeuron
    {
        public HiddenNeuron()
        {

        }

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a input neuron.
        /// </summary>
        /// <param name="neuron">The input neuron to connect this neuron to.</param>
        public void GenerateConnectionsWith(IInputNeuron neuron)
        {
            if (neuron == null)
                throw new ArgumentNullException("neuron");

            // Create connection only if this neuron doesn't already have an incoming connection from the input neuron.
            if (!GetIncomingConnections().Any(c => c.FromNeuron == neuron))
                Connections.Add(new IncomingConnection(neuron));

            // Create connection only if the input neuron doesn't already have an outgoing connection to this neuron.
            if (!neuron.GetOutgoingConnections().Any(c => c.ToNeuron == this))
                neuron.Connections.Add(new OutgoingConnection(this));
        }

        /// <summary>
        /// Creates the incoming and outgoing connections between this neuron and a previous hidden neuron.
        /// </summary>
        /// <param name="incomingNeuron">The incoming neuron to connect this neuron to.</param>
        public void GenerateConnectionsWith(IHiddenNeuron incomingNeuron)
        {
            if (incomingNeuron == null)
                throw new ArgumentNullException("incomingNeuron");

            // Create connection only if this neuron doesn't already have an incoming connection from the incoming hidden neuron.
            if (!GetIncomingConnections().Any(c => c.FromNeuron == incomingNeuron))
                Connections.Add(new IncomingConnection(incomingNeuron));

            // Create connection only if the incoming hidden neuron doesn't already have an outgoing connection to this neuron.
            if (!incomingNeuron.GetOutgoingConnections().Any(c => c.ToNeuron == this))
                incomingNeuron.Connections.Add(new OutgoingConnection(this));
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

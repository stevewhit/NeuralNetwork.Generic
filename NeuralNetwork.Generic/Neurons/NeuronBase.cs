using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Generic.Connections;

namespace NeuralNetwork.Generic.Neurons
{
    public interface INeuron
    {
        /// <summary>
        /// The activation level of this neuron.
        /// </summary>
        double ActivationLevel { get; set; }

        /// <summary>
        /// The bias of this neuron.
        /// </summary>
        double Bias { get; set; }

        /// <summary>
        /// The description of this neuron.
        /// </summary>
        string Description { get; set; }

        /// <summary>
        /// The connections to this neuron.
        /// </summary>
        IList<INeuronConnection> Connections { get; set; }
    }

    public abstract class NeuronBase : INeuron
    {
        /// <summary>
        /// The activation level of this neuron.
        /// </summary>
        public double ActivationLevel { get; set; }

        /// <summary>
        /// The bias of this neuron.
        /// </summary>
        public double Bias { get; set; }

        /// <summary>
        /// The description of this neuron.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// The connections to this neuron.
        /// </summary>
        public IList<INeuronConnection> Connections { get; set; }

        public NeuronBase()
        {
            Connections = new List<INeuronConnection>();
        }
                
        /// <summary>
        /// Returns all outgoing connections to this neuron.
        /// </summary>
        /// <returns>Returns all outgoing connections for this neuron.</returns>
        public IEnumerable<IOutgoingConnection> GetOutgoingConnections() => Connections?.OfType<IOutgoingConnection>();

        /// <summary>
        /// Returns all incoming connections to this neuron.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<IIncomingConnection> GetIncomingConnections() => Connections?.OfType<IIncomingConnection>();
    }
}

﻿using NeuralNetwork.Generic.Neurons;

namespace NeuralNetwork.Generic.Connections
{
    public interface IOutgoingConnection : INeuronConnection
    {
        /// <summary>
        /// The neuron that this connection is going to.
        /// </summary>
        INeuron ToNeuron { get; set; }
    }

    public class OutgoingConnection : NeuronConnectionBase, IOutgoingConnection
    {
        /// <summary>
        /// The neuron that this connection is going to.
        /// </summary>
        public INeuron ToNeuron { get; set; }

        public OutgoingConnection()
        {

        }

        public OutgoingConnection(INeuron toNeuron)
        {
            ToNeuron = toNeuron;
        }
    }
}

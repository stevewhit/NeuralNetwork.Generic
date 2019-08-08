using System;
using System.Collections.Generic;
using Framework.Generic.Utility;
using NeuralNetwork.Generic.Connections;

namespace NeuralNetwork.Generic.Neurons
{
    public interface INeuron : IDisposable
    {
        double ActivationLevel { get; set; }
        double Bias { get; set; }
        IEnumerable<INeuronConnection> Connections { get; set; }
    }

    public abstract class NeuronBase : INeuron
    {
        public double ActivationLevel { get; set; }
        public double Bias { get; set; }

        public IEnumerable<INeuronConnection> Connections { get; set; }

        public NeuronBase()
        {

        }

        public NeuronBase(IEnumerable<INeuronConnection> connections)
        {
            Connections = connections; 
        }

        #region IDisposable
        private bool disposed = false;
        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    // Free managed objects
                    Connections?.Dispose();
                    Connections = null;
                }

                // Free unmanaged objects
            }

            disposed = true;
        }

        public virtual void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}

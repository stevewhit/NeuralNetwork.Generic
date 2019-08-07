using Framework.Generic.Utility;
using NeuralNetwork.Generic.Layers;
using System;

namespace NeuralNetwork.Generic.Networks
{
    public interface INeuralNetwork : IDisposable
    {
        INetworkLayer[] Layers { get; set; }
    }

    public abstract class NeuralNetworkBase : INeuralNetwork
    {
        public INetworkLayer[] Layers { get; set; }

        public NeuralNetworkBase(INetworkLayer[] layers)
        {
            Layers = layers;
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
                    Layers.Dispose();
                    Layers = null;
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

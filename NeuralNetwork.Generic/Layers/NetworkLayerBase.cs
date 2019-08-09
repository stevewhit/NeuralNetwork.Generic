using System;
using NeuralNetwork.Generic.Neurons;
using Framework.Generic.Utility;
using System.Collections.Generic;

namespace NeuralNetwork.Generic.Layers
{
    public interface INetworkLayer : IDisposable
    {
        IEnumerable<INeuron> Neurons { get; set; }
        int SortOrder { get; set; }
    }

    public abstract class NetworkLayerBase : INetworkLayer
    {
        public IEnumerable<INeuron> Neurons { get; set; }

        public int SortOrder { get; set; }

        public NetworkLayerBase(IEnumerable<INeuron> neurons, int sortOrder)
        {
            Neurons = neurons;
            SortOrder = sortOrder;
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
                    Neurons.Dispose();
                    Neurons = null;
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

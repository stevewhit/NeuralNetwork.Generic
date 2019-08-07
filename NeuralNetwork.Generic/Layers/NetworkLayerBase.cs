using System;
using NeuralNetwork.Generic.Neurons;
using Framework.Generic.Utility;

namespace NeuralNetwork.Generic.Layers
{
    public interface INetworkLayer : IDisposable
    {
        INeuron[] Neurons { get; set; }
    }

    public abstract class NetworkLayerBase : INetworkLayer
    {
        public INeuron[] Neurons { get; set; }

        public NetworkLayerBase(INeuron[] neurons)
        {
            Neurons = neurons;
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

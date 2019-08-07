using System;

namespace NeuralNetwork.Generic.Connections
{
    public interface INeuronConnection : IDisposable
    {
        double Weight { get; set; }
    }

    public abstract class NeuronConnectionBase
    {
        public double Weight { get; set; }

        public NeuronConnectionBase()
        {

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

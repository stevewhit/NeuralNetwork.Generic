
namespace NeuralNetwork.Generic.Connections
{
    public interface INeuronConnection
    {
        /// <summary>
        /// The weight of the connection.
        /// </summary>
        double Weight { get; set; }
    }

    public abstract class NeuronConnectionBase
    {
        /// <summary>
        /// The weight of the connection.
        /// </summary>
        public double Weight { get; set; }

        public NeuronConnectionBase()
        {

        }
    }
}

using System.Text;

namespace NnEngine
{
    public class NeuralLayer
    {
        public NeuralLayer(IActivationFunction p_ActivationFunction = null, bool p_IsInputLayer = false)
        {
            ActivationFunction = p_ActivationFunction;
            IsInputLayer = p_IsInputLayer;
            if (!IsInputLayer && ActivationFunction == null)
                throw new ArgumentException("IsInputLayer==false with ActivationFunction==null");
        }

        public List<Neuron> Neurons { get; set; } = new();
        public NeuralLayer NextLayer { get; set; }
        public NeuralLayer LastLayer => NextLayer == null ? this : NextLayer.LastLayer;

        public IActivationFunction ActivationFunction { get; set; }
        public bool IsInputLayer { get; set; }


        /// <summary>
        /// Join all Neurons from one layer to all Neurons of other layer - by Synapse class (referenced by both neurons from to)
        /// </summary>
        /// <param name="p_NextLayer"></param>
        public void AddNextLayer(NeuralLayer p_NextLayer)
        {
            NextLayer = p_NextLayer;
            p_NextLayer.Neurons.ForEach(nextLayerNeuron =>
            {
                Neurons.ForEach(thisLayerNeuron =>
                {
                    var synapse = new Synapse(thisLayerNeuron, nextLayerNeuron);
                    thisLayerNeuron.SynapsesToNextLayer.Add(synapse);
                    nextLayerNeuron.SynapsesToPreviousLayer.Add(synapse);
                });
            });
        }

        public Neuron FindNeuronById(string p_Id)
        {
            var result = Neurons.Find(i => i.Id == p_Id);
            return result ?? NextLayer?.FindNeuronById(p_Id);
        }

        public void FeedForward(List<float> p_Inputs)
        {
            Neurons.ForEach(n =>
            {
                if (!n.SynapsesToPreviousLayer.Any())
                    n.SynapsesToPreviousLayer.Add(new Synapse(null, n, 1));
                n.Calculate(p_Inputs, ActivationFunction);
            });
            NextLayer?.FeedForwardToNextLayer(ActivationFunction);
        }

        private void FeedForwardToNextLayer(IActivationFunction p_ActivationFunction)
        {
            Neurons.ForEach(n =>
            {
                var data = n.SynapsesToPreviousLayer.Select(i => i.From.LastCalculatedOutputActivated).ToList();
                n.Calculate(data, p_ActivationFunction);
            });
            NextLayer?.FeedForwardToNextLayer(ActivationFunction);
        }

        public void GetDebugInfo(StringBuilder sb)
        {
            sb.Append($"NeuralLayer: {Environment.NewLine}");
            foreach (Neuron neuron in Neurons)
            {
                sb.Append($"Neuron {neuron.Id} ");
                foreach (var synapse in neuron.SynapsesToPreviousLayer)
                {
                    sb.Append($"input weight {synapse.Weight:f3} ");
                } 
                sb.Append($"bias {neuron.Bias:f3}{Environment.NewLine}");
            }

            NextLayer?.GetDebugInfo(sb);
        }
    }
}

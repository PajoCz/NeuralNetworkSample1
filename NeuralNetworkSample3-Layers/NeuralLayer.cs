using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkSample3_Layers
{
    public class NeuralLayer
    {
        public List<Neuron> Neurons { get; set; } = new List<Neuron>();
        public NeuralLayer NextLayer { get; set; }
        public NeuralLayer LastLayer => NextLayer == null ? this : NextLayer.LastLayer;


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

        public void CalculateInputs(List<double> p_Inputs)
        {
            Neurons.ForEach(n =>
            {
                if (!n.SynapsesToPreviousLayer.Any())
                    n.SynapsesToPreviousLayer.Add(new Synapse(null, n, 1));
                n.CalcBySigmoid(p_Inputs);
            });
            NextLayer.CalculateByPreviousLayer();
        }

        private void CalculateByPreviousLayer()
        {
            Neurons.ForEach(n =>
            {
                var data = n.SynapsesToPreviousLayer.Select(i => i.From.LastCalculatedOutputSigmoid).ToList();
                n.CalcBySigmoid(data);
            });
        }

        public void GetDebugInfo(StringBuilder sb)
        {
            sb.Append($"NeuralLayer: {Environment.NewLine}");
            foreach (Neuron neuron in Neurons)
            {
                sb.Append($"Neuron {neuron.Id} ");
                foreach (var synapse in neuron.SynapsesToPreviousLayer)
                {
                    sb.Append($"input weight {synapse.Weight} ");
                } 
                sb.Append($"bias {neuron.Bias}{Environment.NewLine}");
            }

            NextLayer?.GetDebugInfo(sb);
        }
    }
}

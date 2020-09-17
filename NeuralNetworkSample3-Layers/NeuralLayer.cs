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

        public void CalculateInputs(List<double> inputs)
        {
            Neurons.ForEach(n =>
            {
                if (!n.SynapsesToPreviousLayer.Any())
                    n.SynapsesToPreviousLayer.Add(new Synapse(null, n, 1));
                n.CalcBySigmoid(inputs);
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

        public void WriteToConsole()
        {
            Console.WriteLine("NeuralLayer: ");
            foreach (Neuron neuron in Neurons)
            {
                Console.Write($"Neuron {neuron.Id} ");
                foreach (var synapse in neuron.SynapsesToPreviousLayer)
                {
                    Console.Write($"input weight {synapse.Weight} ");
                } 
                Console.WriteLine($"bias {neuron.Bias} ");
            }
            NextLayer?.WriteToConsole();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkSample3_Layers
{
    public class Synapse
    {
        public Synapse(Neuron p_From, Neuron p_To, double? p_Weight = null)
        {
            From = p_From;
            To = p_To;
            Weight = p_Weight ?? (new Random().NextDouble()-0.5) * 2;
        }

        public Neuron From { get; set; }
        public Neuron To { get; set; }
        public double Weight { get; set; }
    }
}

using System;
using System.Collections.Generic;

namespace NeuralNetworkSample3_Layers
{
    //https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9
    //https://github.com/vzhou842/neural-network-from-scratch/blob/master/network.py
    //https://gist.githubusercontent.com/vzhou842/986459249c2510da9b5c92bd3f3ca7fb/raw/3f99c7caaff534d09590842ef3c33a6a33effba8/fullnetwork.py
    class Program
    {
        static void Main(string[] args)
        {
            List<List<double>> data = new List<List<double>>()
            {
                new List<double> {-2, -1},
                new List<double> {25, 6},
                new List<double> {17, 4},
                new List<double> {-15, -6},
            };
            List<double> expectedOutputs = new List<double> {1, 0, 0, 1};

            NeuralLayer layerInputs = new NeuralLayer();
            layerInputs.Neurons.Add(new Neuron("i1", 0));
            layerInputs.Neurons.Add(new Neuron("i2", 0));

            NeuralLayer layerh1h2 = new NeuralLayer();
            layerh1h2.Neurons.Add(new Neuron("h1") { Bias = 0.9888773586480377});
            layerh1h2.Neurons.Add(new Neuron("h2") { Bias = 0.32188634502570845});
            layerInputs.AddNextLayer(layerh1h2);
            layerInputs.Neurons[0].SynapsesToNextLayer[0].Weight = 0.8815793758627867;
            layerInputs.Neurons[0].SynapsesToNextLayer[1].Weight = -0.0037441705087075737;
            layerInputs.Neurons[1].SynapsesToNextLayer[0].Weight = -0.5202642691344876;
            layerInputs.Neurons[1].SynapsesToNextLayer[1].Weight = 0.2667151772486819;

            NeuralLayer layero1 = new NeuralLayer();
            layero1.Neurons.Add(new Neuron("o1") { Bias = -1.1927510125913223});
            layerh1h2.AddNextLayer(layero1);
            layerh1h2.Neurons[0].SynapsesToNextLayer[0].Weight = -0.038516025100668934;
            layerh1h2.Neurons[1].SynapsesToNextLayer[0].Weight = 1.0484903515494195;
            
            var network = new NeuralNetworkEngine(layerh1h2);
            //var network = new NeuralNetworkEngine(layerInputs);
            network.Train(data, expectedOutputs);
        }
    }
}

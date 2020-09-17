using System.Collections.Generic;
using System.Security.Cryptography;

namespace NeuralNetworkSample2_Refactor
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
            var network = new NeuralNetworkEngine();
            network.Train(data, expectedOutputs);
        }
    }
}

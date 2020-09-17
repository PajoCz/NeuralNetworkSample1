using System;
using System.Collections.Generic;

namespace NeuralNetworkSample3_Layers
{
    public class NeuralNetworkEngine
    {
        private readonly NeuralLayer _Layer;

        public NeuralNetworkEngine(NeuralLayer p_Layer)
        {
            _Layer = p_Layer;
        }

        public void Train(List<List<double>> data, List<double> expectedResults, int epochs = 1000)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int x = 0; x < expectedResults.Count; x++)
                {
                    _Layer.CalculateInputs(data[x]);

                    //Calculate partial derivatives.
                    //Naming: d_L_d_w1 represents "partial L / partial w1"
                    var o1 = _Layer.NextLayer.Neurons[0].LastCalculatedOutputSigmoid;
                    var partialDerivates = -2 * (expectedResults[x] - o1);
                    _Layer.NextLayer.Neurons[0].BackPropagate(data[x], partialDerivates);
                }

                //Calculate total loss at the end of each epoch
                if (epoch % 10 == 0)
                {
                    var percentMiss = PercentMiss(data, expectedResults);
                    Console.WriteLine($"Epoch {epoch} percent missed: {percentMiss}");
                }
            }

            _Layer.WriteToConsole();
        }

        private double PercentMiss(List<List<double>> data, List<double> expectedResults)
        {
            double percentSum = 0;
            int percentCount = 0;
            for (int x = 0; x < expectedResults.Count; x++)
            {
                _Layer.CalculateInputs(data[x]);
                var o1 = _Layer.NextLayer.Neurons[0].LastCalculatedOutputSigmoid;

                var actual = o1;
                var expected = expectedResults[x];
                var percent = expected != 0
                    ? (expected - actual) / expected * 100
                    : actual * 100;
                percentSum += percent;
                percentCount++;
            }

            double percentMiss = percentSum / percentCount;
            return percentMiss;
        }
    }
}
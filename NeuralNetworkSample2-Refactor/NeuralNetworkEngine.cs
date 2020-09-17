using System;
using System.Collections.Generic;

namespace NeuralNetworkSample2_Refactor
{
    public class NeuralNetworkEngine
    {
        public void Train(List<List<double>> data, List<double> expectedResults, int epochs = 1000)
        {
            //Neuron neuronh1 = new Neuron("h1",2);
            //Neuron neuronh2 = new Neuron("h2",2);
            //Neuron neurono1 = new Neuron("o1",2);
            Neuron neuronh1 = new Neuron("h1", new List<double>() {0.8815793758627867, -0.5202642691344876}, 0.9888773586480377);
            Neuron neuronh2 = new Neuron("h2", new List<double>() {-0.0037441705087075737, 0.2667151772486819}, 0.32188634502570845);
            Neuron neurono1 = new Neuron("o1", new List<double>() {-0.038516025100668934, 1.0484903515494195}, -1.1927510125913223);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int x = 0; x < expectedResults.Count; x++)
                {
                    var h1 = neuronh1.CalcBySigmoid(data[x]);
                    var h2 = neuronh2.CalcBySigmoid(data[x]);
                    var listh1h2 = new List<double>() {h1, h2};
                    var o1 = neurono1.CalcBySigmoid(listh1h2);
                    //Calculate partial derivatives.
                    //Naming: d_L_d_w1 represents "partial L / partial w1"
                    var partialDerivates = -2 * (expectedResults[x] - o1);
                    neurono1.BackPropagate(listh1h2, partialDerivates);
                    var weightsMultiplyByDerivatesSigmoidLastOutput = neurono1.WeightsMultiplyByDerivatesSigmoidLastOutput();
                    neuronh1.BackPropagate(data[x], partialDerivates * weightsMultiplyByDerivatesSigmoidLastOutput[0]);
                    neuronh2.BackPropagate(data[x], partialDerivates * weightsMultiplyByDerivatesSigmoidLastOutput[1]);
                }

                //Calculate total loss at the end of each epoch
                if (epoch % 10 == 0)
                {
                    var percentMiss = PercentMiss(data, expectedResults, neuronh1, neuronh2, neurono1);
                    //var totalError = TotalError(data, expectedResults, neuronh1, neuronh2, neurono1);
                    Console.WriteLine($"Epoch {epoch} percent missed: {percentMiss}");
                }
            }

            Console.WriteLine($"w1 {neuronh1.Weights[0]}");
            Console.WriteLine($"w2 {neuronh1.Weights[1]}");
            Console.WriteLine($"w3 {neuronh2.Weights[0]}");
            Console.WriteLine($"w4 {neuronh2.Weights[1]}");
            Console.WriteLine($"w5 {neurono1.Weights[0]}");
            Console.WriteLine($"w6 {neurono1.Weights[1]}");
            Console.WriteLine($"b1 {neuronh1.Bias}");
            Console.WriteLine($"b2 {neuronh2.Bias}");
            Console.WriteLine($"b3 {neurono1.Bias}");
        }

        private static double PercentMiss(List<List<double>> data, List<double> expectedResults, Neuron neuronh1, Neuron neuronh2, Neuron neurono1)
        {
            double percentSum = 0;
            int percentCount = 0;
            for (int x = 0; x < expectedResults.Count; x++)
            {
                var h1 = neuronh1.CalcBySigmoid(data[x]);
                var h2 = neuronh2.CalcBySigmoid(data[x]);
                var o1 = neurono1.CalcBySigmoid(new List<double>() {h1, h2});

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

        private static double TotalError(List<List<double>> data, List<double> expectedResults, Neuron neuronh1, Neuron neuronh2, Neuron neurono1)
        {
            double result = 0;
            for (int x = 0; x < expectedResults.Count; x++)
            {
                var h1 = neuronh1.CalcBySigmoid(data[x]);
                var h2 = neuronh2.CalcBySigmoid(data[x]);
                var o1 = neurono1.CalcBySigmoid(new List<double>() {h1, h2});

                var error = Math.Pow(o1 - expectedResults[x], 2);
                result += error;
            }

            return result;
        }
    }
}
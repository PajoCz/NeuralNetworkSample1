using System;
using System.Collections.Generic;

namespace NeuralNetworkSample1
{
    class Program
    {
        //https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9
        //https://github.com/vzhou842/neural-network-from-scratch/blob/master/network.py
        //https://gist.githubusercontent.com/vzhou842/986459249c2510da9b5c92bd3f3ca7fb/raw/3f99c7caaff534d09590842ef3c33a6a33effba8/fullnetwork.py
        static void Main(string[] args)
        {
            //Define dataset
            List<double[]> data = new List<double[]>()
            {
                new double[] {-2, -1},
                new double[] {25, 6},
                new double[] {17, 4},
                new double[] {-15, -6},
            };
            double[] all_y_trues = {1, 0, 0, 1};

            //Train our neural network!
            var network = new OurNeuralNetwork();
            network.train(data, all_y_trues);
        }

        static double sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        static double deriv_sigmoid(double x)
        {
            var sig = sigmoid(x);
            return sig * (1 - sig);
        }

        static double mse_loss(double[] y_true, double[] y_pred)
        {
            return double.MinValue;
        }

        class OurNeuralNetwork
        {
            static Random _Rnd = new Random();
            private double w1 = 0.8815793758627867; //_Rnd.NextDouble();
            private double w2 = -0.5202642691344876; //_Rnd.NextDouble();
            private double w3 = -0.0037441705087075737; //_Rnd.NextDouble();
            private double w4 = 0.2667151772486819; //_Rnd.NextDouble();
            private double w5 = -0.038516025100668934; //_Rnd.NextDouble();
            private double w6 = 1.0484903515494195; //_Rnd.NextDouble();
            private double b1 = 0.9888773586480377; //_Rnd.NextDouble();
            private double b2 = 0.32188634502570845; //_Rnd.NextDouble();
            private double b3 = -1.1927510125913223; //_Rnd.NextDouble();

            public double feedforward(double[] x)
            {
                var h1 = sigmoid(w1 * x[0] + w2 * x[1] + b1);
                var h2 = sigmoid(w3 * x[0] + w4 * x[1] + b2);
                var o1 = sigmoid(w5 * h1 + w6 * h2 + b3);
                return o1;
            }

            public void train(List<double[]> data, double[] all_y_trues)
            {
                double learn_rate = 0.1;
                int epochs = 1000;
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    for (int x = 0; x < all_y_trues.Length; x++)
                    {
                        //Do a feedforward (we'll need these values later)
                        var sum_h1 = w1 * data[x][0] + w2 * data[x][1] + b1;
                        var h1 = sigmoid(sum_h1);

                        var sum_h2 = w3 * data[x][0] + w4 * data[x][1] + b2;
                        var h2 = sigmoid(sum_h2);

                        var sum_o1 = w5 * h1 + w6 * h2 + b3;
                        var o1 = sigmoid(sum_o1);

                        var y_pred = o1;

                        //Calculate partial derivatives.
                        //Naming: d_L_d_w1 represents "partial L / partial w1"
                        var d_L_d_ypred = -2 * (all_y_trues[x] - y_pred);

                        //Neuron o1
                        var d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1);
                        var d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1);
                        var d_ypred_d_b3 = deriv_sigmoid(sum_o1);

                        var d_ypred_d_h1 = w5 * deriv_sigmoid(sum_o1);
                        var d_ypred_d_h2 = w6 * deriv_sigmoid(sum_o1);

                        //Neuron h1
                        var d_h1_d_w1 = data[x][0] * deriv_sigmoid(sum_h1);
                        var d_h1_d_w2 = data[x][1] * deriv_sigmoid(sum_h1);
                        var d_h1_d_b1 = deriv_sigmoid(sum_h1);

                        //Neuron h2
                        var d_h2_d_w3 = data[x][0] * deriv_sigmoid(sum_h2);
                        var d_h2_d_w4 = data[x][1] * deriv_sigmoid(sum_h2);
                        var d_h2_d_b2 = deriv_sigmoid(sum_h2);

                        //--- Update weights and biases
                        //Neuron h1
                        w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
                        w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
                        b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;

                        //Neuron h2
                        w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
                        w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
                        b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

                        //Neuron o1
                        w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5;
                        w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6;
                        b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3;
                    }

                    //Calculate total loss at the end of each epoch
                    if (epoch % 10 == 0)
                    {
                        double percentSum = 0;
                        int percentCount = 0;
                        for (int x = 0; x < all_y_trues.Length; x++)
                        {
                            var actual = feedforward(data[x]);
                            var expected = all_y_trues[x];
                            var percent = expected != 0 
                                ? (expected - actual) / expected * 100
                                : actual * 100;
                            percentSum += percent;
                            percentCount++;
                        }

                        double percentMiss = percentSum / percentCount;
                        Console.WriteLine($"Epoch {epoch} percent missed: {percentMiss}");
                    }
                }

                Console.WriteLine($"w1 {w1}");
                Console.WriteLine($"w2 {w2}");
                Console.WriteLine($"w3 {w3}");
                Console.WriteLine($"w4 {w4}");
                Console.WriteLine($"w5 {w5}");
                Console.WriteLine($"w6 {w6}");
                Console.WriteLine($"b1 {b1}");
                Console.WriteLine($"b2 {b2}");
                Console.WriteLine($"b3 {b3}");
            }
        }
    }
}

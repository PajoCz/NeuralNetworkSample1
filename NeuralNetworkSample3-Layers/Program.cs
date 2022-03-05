using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text;

namespace NeuralNetworkSample3_Layers
{
    //https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9
    //https://github.com/vzhou842/neural-network-from-scratch/blob/master/network.py
    //https://gist.githubusercontent.com/vzhou842/986459249c2510da9b5c92bd3f3ca7fb/raw/3f99c7caaff534d09590842ef3c33a6a33effba8/fullnetwork.py
    class Program
    {
        //SETTINGS
        static bool setMyInitValues = false;    //set non random neurons biases and weights
        static string logFolder = "D:\\NeuralNetworkProgress";
        static bool logStepsToImages = true;   //enable/disable log images to log folder
        static bool logTextFile = true;   //enable/disable text log start/end neurons to log folder

        //DATA
        const int epochs = 100;

        const double calibWeight = -135;  //Weight [lb]
        const double calibHeight = -66;   //Height [in]
        static List<List<double>> data = new List<List<double>>()
        {
            //new List<double> {-2, -1},
            //new List<double> {25, 6},
            //new List<double> {17, 4},
            //new List<double> {-15, -6},

            new List<double> {133+calibWeight, 65+calibHeight},
            new List<double> {160+calibWeight, 72+calibHeight},
            new List<double> {152+calibWeight, 70+calibHeight},
            new List<double> {120+calibWeight, 60+calibHeight},

            //new List<double> {133, 65},
            //new List<double> {160, 72},
            //new List<double> {152, 70},
            //new List<double> {120, 60},
        };
        static List<double> expectedOutputs = new List<double> {1, 0, 0, 1};   //Gender 1 female, 0 male

        static void Main(string[] args)
        {
            NeuralLayer layerInputs = new NeuralLayer();
            layerInputs.Neurons.Add(new Neuron("i1", 0));
            layerInputs.Neurons.Add(new Neuron("i2", 0));

            NeuralLayer layerh1h2 = new NeuralLayer();
            layerh1h2.Neurons.Add(new Neuron("h1"));
            layerh1h2.Neurons.Add(new Neuron("h2"));
            //layerh1h2.Neurons.Add(new Neuron("h3"));
            //layerh1h2.Neurons.Add(new Neuron("h4"));
            layerInputs.AddNextLayer(layerh1h2);
            NeuralLayer layero1 = new NeuralLayer();
            layero1.Neurons.Add(new Neuron("o1"));
            layerh1h2.AddNextLayer(layero1);

            //var network = new NeuralNetworkEngine(layerInputs);
            var network = new NeuralNetworkEngine(layerh1h2);

            if (setMyInitValues)
            {
                network.FindNeuronById("h1").InitValues(new List<double>() {0.8815793758627867, -0.5202642691344876}, 0.9888773586480377);
                network.FindNeuronById("h2").InitValues(new List<double>() {-0.0037441705087075737, 0.2667151772486819}, 0.32188634502570845);
                network.FindNeuronById("o1").InitValues(new List<double>() {-0.038516025100668934, 1.0484903515494195}, -1.1927510125913223);
            }
            
            network.OnAfterTrainOneItem += NetworkOnOnAfterTrainOneItem;
            
            network.Train(data, expectedOutputs, epochs);

            //var result = network.Calculate(new List<double> {-2, -1})[0];
            //result = network.Calculate(new List<double> {-20, -5})[0];
            //result = network.Calculate(new List<double> {10, 3})[0];
            //result = network.Calculate(new List<double> {0, 0})[0];
        }

        private static void NetworkOnOnAfterTrainOneItem(NeuralNetworkEngine.OnTrainProgressTime p_OnTrainProgressTime, NeuralLayer p_NeuralInputLayer, int p_Epoch, int p_DataIndex, List<double> p_Data, double p_ExpectedResult, double? p_PercentMissAll, double? p_PercentMiss)
        {
            if (logTextFile)
            {
                if (p_Epoch == 0 && p_DataIndex == 0 &&
                    p_OnTrainProgressTime == NeuralNetworkEngine.OnTrainProgressTime.BeforeCalculate)
                {
                    StringBuilder sb =
                        new StringBuilder(
                            $"{Environment.NewLine}------{Environment.NewLine}{DateTime.Now} INIT. PercentMiss: {p_PercentMissAll}{Environment.NewLine}");
                    p_NeuralInputLayer.GetDebugInfo(sb);
                    File.AppendAllText($"{logFolder}\\_log.txt", sb.ToString());
                }

                if (p_Epoch == epochs - 1 && p_DataIndex == data.Count - 1 &&
                    p_OnTrainProgressTime == NeuralNetworkEngine.OnTrainProgressTime.AfterBackPropagation)
                {
                    StringBuilder sb =
                        new StringBuilder($"{DateTime.Now} END. PercentMiss: {p_PercentMissAll}{Environment.NewLine}");
                    p_NeuralInputLayer.GetDebugInfo(sb);
                    File.AppendAllText($"{logFolder}\\_log.txt", sb.ToString());
                }
            }

            if (logStepsToImages)
            {
                var im = new NeuralNetworkToImage();
                using Bitmap bmp = im.Draw(p_NeuralInputLayer, 1024, 768, p_Epoch, p_DataIndex, p_Data,
                    p_ExpectedResult, p_PercentMissAll, p_PercentMiss, p_OnTrainProgressTime);
                bmp.Save(
                    $"{logFolder}\\Epoch{p_Epoch}-DataIndex{p_DataIndex}-{(int)p_OnTrainProgressTime}{p_OnTrainProgressTime}.png",
                    ImageFormat.Png);
            }
        }
    }
}


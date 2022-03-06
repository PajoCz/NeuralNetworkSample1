using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text;

namespace NeuralNetworkSample3_Layers
{
    //https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9 - documentation of this sample written in python
    //https://github.com/vzhou842/neural-network-from-scratch/blob/master/network.py - python sample source code
    //https://github.com/PajoCz/NeuralNetworkSample1 - my source code of this sample
    //https://www.analyticsvidhya.com/blog/2021/04/is-gradient-descent-sufficient-for-neural-network/ - article info about "Forward propagation" and "Backward propagation" and "Learning rate". Why use Adam optimizer
    class Program
    {
        //SETTINGS - neural network
        static int neuronsInHiddenLayer = 2;    //0 means no hidden layer
        //static int neuronsInSecondHiddenLayer = 2;    //0 means no hidden layer
        const int epochs = 100;
        const double learnRate = 2.5;
        const double trainEndWithLossPercent = 0.8;

        //setMyInitValues use only for start sample with 2 hidden neurons
        static bool setMyInitValues = false;    //set non random neurons biases and weights

        //SETTINGS - logging progress
        static string logFolder = "D:\\NeuralNetworkProgress";
        static bool logStepsToImages = true;   //enable/disable log images to log folder
        static int logStepsToImageWidth = 1024;
        static int logStepsToImageHeight = 768;
        static bool logTextFile = true;   //enable/disable text log start/end neurons to log folder

        //TRAIN DATA
        const double calibWeight = -135;  //Weight [lb]
        const double calibHeight = -66;   //Height [in]
        static List<List<double>> data = new()
        {
            //new List<double> {-2, -1},
            //new List<double> {25, 6},
            //new List<double> {17, 4},
            //new List<double> {-15, -6},

            new() {133+calibWeight, 65+calibHeight},
            new() {160+calibWeight, 72+calibHeight},
            new() {152+calibWeight, 70+calibHeight},
            new() {120+calibWeight, 60+calibHeight},
        };
        static List<double> expectedOutputs = new() {1, 0, 0, 1};   //Gender 1 female, 0 male

        static void Main(string[] args)
        {
            NeuralLayer layerInput = new NeuralLayer();
            layerInput.Neurons.Add(new Neuron("i1", 0));
            layerInput.Neurons.Add(new Neuron("i2", 0));

            NeuralLayer layerOutput = new NeuralLayer();
            layerOutput.Neurons.Add(new Neuron("o1"));

            NeuralNetworkEngine neuralNetwork;
            if (neuronsInHiddenLayer == 0)
            {
                layerInput.AddNextLayer(layerOutput);
            }
            else
            {
                NeuralLayer layerHidden = new NeuralLayer();
                for(int i = 0; i < neuronsInHiddenLayer; i++)
                    layerHidden.Neurons.Add(new Neuron($"h{i}"));
                layerInput.AddNextLayer(layerHidden);
                //if (neuronsInSecondHiddenLayer > 0)
                //{
                //    NeuralLayer layerSecondHidden = new NeuralLayer();
                //    for(int i = 0; i < neuronsInSecondHiddenLayer; i++)
                //        layerSecondHidden.Neurons.Add(new Neuron($"H{i}"));
                //    layerHidden.AddNextLayer(layerSecondHidden);
                //    layerSecondHidden.AddNextLayer(layerOutput);
                //}
                //else
                {
                    layerHidden.AddNextLayer(layerOutput);
                }
            }
            neuralNetwork = new NeuralNetworkEngine(layerInput);

            if (setMyInitValues)
            {
                //it works only for start sample with 2 hidden neurons
                neuralNetwork.FindNeuronById("h1").InitValues(new List<double>() {0.8815793758627867, -0.5202642691344876}, 0.9888773586480377);
                neuralNetwork.FindNeuronById("h2").InitValues(new List<double>() {-0.0037441705087075737, 0.2667151772486819}, 0.32188634502570845);
                neuralNetwork.FindNeuronById("o1").InitValues(new List<double>() {-0.038516025100668934, 1.0484903515494195}, -1.1927510125913223);
            }
            
            neuralNetwork.OnAfterTrainOneItem += NetworkOnOnAfterTrainOneItem;
            
            neuralNetwork.Train(data, expectedOutputs, epochs, learnRate, trainEndWithLossPercent);

            //PREDICT - check non trained data
            var result = neuralNetwork.Calculate(new List<double> { -2, -1 })[0];
            result = neuralNetwork.Calculate(new List<double> { -20, -5 })[0];
            result = neuralNetwork.Calculate(new List<double> { 10, 3 })[0];
            result = neuralNetwork.Calculate(new List<double> { 0, 0 })[0];
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
                            $"{Environment.NewLine}------{Environment.NewLine}{DateTime.Now} INIT. PercentMissAll: {p_PercentMissAll:f3}%{Environment.NewLine}");
                    p_NeuralInputLayer.NextLayer.GetDebugInfo(sb);
                    File.AppendAllText($"{logFolder}\\_log.txt", sb.ToString());
                }

                if (p_Epoch == epochs - 1 && p_DataIndex == data.Count - 1 &&
                    p_OnTrainProgressTime == NeuralNetworkEngine.OnTrainProgressTime.AfterBackPropagation)
                {
                    StringBuilder sb =
                        new StringBuilder($"{DateTime.Now} END. PercentMissAll: {p_PercentMissAll:f3}%{Environment.NewLine}");
                    p_NeuralInputLayer.NextLayer.GetDebugInfo(sb);
                    File.AppendAllText($"{logFolder}\\_log.txt", sb.ToString());
                }

                if (p_PercentMissAll <= trainEndWithLossPercent)
                {
                    StringBuilder sb =
                        new StringBuilder($"{DateTime.Now} TRAINED Epoch={p_Epoch} , DataIndex={p_DataIndex} {p_OnTrainProgressTime}. PercentMissAll: {p_PercentMissAll:f3}%{Environment.NewLine} <= trainEndWithLossPercent {trainEndWithLossPercent}%{Environment.NewLine}");
                    p_NeuralInputLayer.NextLayer.GetDebugInfo(sb);
                    File.AppendAllText($"{logFolder}\\_log.txt", sb.ToString());
                }
            }

            if (logStepsToImages)
            {
                var im = new NeuralNetworkToImage();
                using Bitmap bmp = im.Draw(p_NeuralInputLayer, logStepsToImageWidth, logStepsToImageHeight, p_Epoch, p_DataIndex, p_Data,
                    p_ExpectedResult, p_PercentMissAll, p_PercentMiss, p_OnTrainProgressTime);
                bmp.Save(
                    $"{logFolder}\\Epoch{p_Epoch}-DataIndex{p_DataIndex}-{(int)p_OnTrainProgressTime}{p_OnTrainProgressTime}.png",
                    ImageFormat.Png);
            }
        }
    }
}


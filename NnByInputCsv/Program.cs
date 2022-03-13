﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text;
using NnEngine;
using NnEngineStateToImage;

namespace NnByInputCsv
{
    class Program
    {
        //SETTINGS - neural network
        const float neuronsInHiddenMultiplyByInput = 2; //0 means no hidden layer
        const float neuronsInSecondHiddenLayerMultiplyByInput = 0;    //0 means no second hidden layer
        //static int neuronsInSecondHiddenLayer = 2;    //0 means no hidden layer
        const int epochs = 10000;
        const float learnRate = 2f;
        const float trainEndWithLossPercent = 0.8f;
        const bool onlyCalculateOnLoadedTrainedModel = false;

        //setMyInitValues use only for start sample with 2 hidden neurons
        static bool setMyInitValues = false;    //set non random neurons biases and weights

        //SETTINGS - logging progress
        static string logFolder = "D:\\NeuralNetworkProgress2";
        static bool logStepsToImages = false;   //enable/disable log images to log folder
        static bool logLearnProgressToImage = false;   //enable/disable log images to log folder
        static int logStepsToImageWidth = 1024;
        static int logStepsToImageHeight = 768;
        static bool logTextFile = false;   //enable/disable text log start/end neurons to log folder

        static void Main(string[] args)
        {

            //var fn = "Dataset1-xor.csv";
            //var fn = "Dataset1-sample-posun.csv";
            var fn = "Dataset1-sample.csv";
            //var fn = "Dataset1-sample-outputVelky.csv";
            //var fn = "Dataset1-ukol.csv";
            var fnWithPath = @"c:\Users\pajo\source\repos\NeuralNetworkSample1\NnByInputCsv\" + fn;
            using FileStream fs =
                new FileStream(fnWithPath, FileMode.Open);
            
            var reader = new DatasetReader(fs);
            var data = reader.ReadLinesNumbers();

            if (!onlyCalculateOnLoadedTrainedModel)
            {
                //MinMaxScaler
                MinMaxScaler minMaxScalerInput = new MinMaxScaler();
                minMaxScalerInput.Fit(data.Item1);
                var columnsWithConstValues = minMaxScalerInput.ColumnsWithConstValues();
                MinMaxScaler minMaxScalerOutput = new MinMaxScaler();
                minMaxScalerOutput.Fit(data.Item2.ConvertAll(i => new List<float>() { i }));

                var afSigmoid = new ActivationFunctionSigmoid();

                //NN
                NeuralLayer layerInput = new NeuralLayer(p_IsInputLayer: true);
                for (int i = 0; i < reader._Header.Count; i++)
                {
                    var rh = reader._Header[i];
                    if (rh != reader.OutputColumn && !columnsWithConstValues.Contains(i))
                        layerInput.Neurons.Add(new Neuron(rh, 0));
                }

                NeuralLayer layerOutput = new NeuralLayer(afSigmoid);
                layerOutput.Neurons.Add(new Neuron(reader.OutputColumn));

                var neuronsInHiddenLayer = (int)(layerInput.Neurons.Count * neuronsInHiddenMultiplyByInput);

                if (neuronsInHiddenLayer == 0)
                {
                    layerInput.AddNextLayer(layerOutput);
                }
                else
                {
                    NeuralLayer layerHidden = new NeuralLayer(afSigmoid);
                    for (int i = 0; i < neuronsInHiddenLayer; i++)
                        layerHidden.Neurons.Add(new Neuron($"h{i + 1}"));
                    layerInput.AddNextLayer(layerHidden);
                    var neuronsInSecondHiddenLayer =
                        (int)(layerInput.Neurons.Count * neuronsInSecondHiddenLayerMultiplyByInput);
                    if (neuronsInSecondHiddenLayer > 0)
                    {
                        NeuralLayer layerSecondHidden = new NeuralLayer(afSigmoid);
                        for (int i = 0; i < neuronsInSecondHiddenLayer; i++)
                            layerSecondHidden.Neurons.Add(new Neuron($"H{i}"));
                        layerHidden.AddNextLayer(layerSecondHidden);
                        layerSecondHidden.AddNextLayer(layerOutput);
                    }
                    else
                    {
                        layerHidden.AddNextLayer(layerOutput);
                    }
                }

                NeuralNetworkEngine nne = new NeuralNetworkEngine(layerInput);

                if (setMyInitValues)
                {
                    //it works only for start sample with 2 hidden neurons
                    nne.FindNeuronById("h1").InitValues(new List<float>() { 0.8815793758627867f, -0.5202642691344876f }, 0.9888773586480377f);
                    nne.FindNeuronById("h2").InitValues(new List<float>() { -0.0037441705087075737f, 0.2667151772486819f }, 0.32188634502570845f);
                    nne.FindNeuronById("Out").InitValues(new List<float>() { -0.038516025100668934f, 1.0484903515494195f }, -1.1927510125913223f);
                }

                nne.OnAfterTrainOneItem += NetworkOnOnAfterTrainOneItem;

                //nne.Train(data.Item1, data.Item2, epochs, learnRate, trainEndWithLossPercent);
                nne.Train(data.Item1, data.Item2, epochs, learnRate, trainEndWithLossPercent, minMaxScalerInput, minMaxScalerOutput);

                using (var nneFile = new FileStream(fnWithPath + ".trained.txt", FileMode.Create))
                {
                    new NnEngineStorage().SaveToStream(nne, nneFile);
                }

                if (logLearnProgressToImage)
                {
                    using (Bitmap imgLearn =
                           learnProgressToImage.CreateImage(logStepsToImageWidth, logStepsToImageHeight))
                    {
                        imgLearn.Save(
                            $"{logFolder}\\_LearnProgress.png", ImageFormat.Png);
                    }
                }

                //Test Nn calculation
                Calculate(reader, data, nne);
            }

            //Test Nn calculation
            Console.WriteLine("NeuralNetwork Save/Load from file and then Calculate:");
            using (var nneFile = new FileStream(fnWithPath + ".trained.txt", FileMode.Open))
            {
                var nneLoaded = new NnEngineStorage().LoadFromStream(nneFile);
                Calculate(reader, data, nneLoaded);
            }
        }

        private static void Calculate(DatasetReader reader, Tuple<List<List<float>>, List<float>> data, NeuralNetworkEngine nne)
        {
            Console.WriteLine(string.Join("; ", reader._Header));
            for (int i = 0; i < data.Item1.Count; i++)
            {
                var actual = nne.Calculate(data.Item1[i])[0]; //only one output neuron at index [0]
                var expected = data.Item2[i];
                Console.Write(string.Join("; ", data.Item1[i]));
                var missed = expected != 0
                    ? Math.Abs((actual / expected - 1f) * 100f)
                    : actual * 100f;
                Console.WriteLine($"; Expected={expected}; Calculated={actual}; {missed:f3}% missed");
            }
        }

        static LearnProgressToImage learnProgressToImage = new();
        private static void NetworkOnOnAfterTrainOneItem(NeuralNetworkEngine.OnTrainProgressTime p_OnTrainProgressTime, NeuralLayer p_NeuralInputLayer, int p_Epoch, int p_DataIndex, List<float> p_Data, float p_ExpectedResult, float? p_PercentMissAll, float? p_PercentMiss)
        {
            if (p_Epoch == 0 && p_DataIndex == 0 &&
                p_OnTrainProgressTime == NeuralNetworkEngine.OnTrainProgressTime.BeforeCalculate)
            {
                learnProgressToImage.PercentMissAllStart = p_PercentMissAll ?? 100;
            }

            if (p_OnTrainProgressTime == NeuralNetworkEngine.OnTrainProgressTime.AfterBackPropagation)
            {
                if (p_DataIndex == p_Data.Count-1)
                 learnProgressToImage.PercentMissAll.Add(p_PercentMissAll ?? 100);
                if (learnProgressToImage.PercentMissDataIndex.Count == p_DataIndex)
                    learnProgressToImage.PercentMissDataIndex.Add(new List<float>());
                learnProgressToImage.PercentMissDataIndex[p_DataIndex].Add(p_PercentMissAll ?? 100);
            }

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

                if (p_Epoch == epochs - 1 && p_DataIndex == p_Data.Count - 1 &&
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


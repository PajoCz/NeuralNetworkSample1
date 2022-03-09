using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text;
using NnEngine;

namespace NnByInputCsv
{
    class Program
    {
        static void Main(string[] args)
        {
            const int epochs = 1000;
            const float learnRate = 2.5f;
            const float trainEndWithLossPercent = 0.8f;
            const float neuronsInHiddenMultiplyByInput = 2;
            const int neuronsInSecondHiddenLayerMultiplyByInput = 0;

            //var fn = "Dataset1-xor.csv";
            //var fn = "Dataset1-sample-posun.csv";
            var fn = "Dataset1-sample.csv";
            //var fn = "Dataset1-sample-outputVelky.csv";
            //var fn = "Dataset1-ukol-posun.csv";
            //var fn = "Dataset1-ukol.csv";
            using FileStream fs =
                new FileStream(@"c:\Users\pajo\source\repos\NeuralNetworkSample1\NnByInputCsv\" + fn,
                    FileMode.Open);
            
            var reader = new DatasetReader(fs);
            var data = reader.ReadLinesNumbers();

            //MinMaxScaler
            MinMaxScaler minMaxScalerInput = new MinMaxScaler();
            minMaxScalerInput.Fit(data.Item1);
            var columnsWithConstValues = minMaxScalerInput.ColumnsWithConstValues();
            var dataNormalizedInput = minMaxScalerInput.Transform(data.Item1, -0.5f);   //vystup -0.5 .. 0.5
            MinMaxScaler minMaxScalerOutput = new MinMaxScaler();
            minMaxScalerOutput.Fit(data.Item2.ConvertAll(i => new List<float>() {i}));
            var dataNormalizedOutput = minMaxScalerOutput.Transform(data.Item2.ConvertAll(i => new List<float>() {i})); //vystup 0 .. 1

            //NN
            NeuralLayer layerInput = new NeuralLayer();
            for(int i=0; i<reader._Header.Count; i++)
            {
                var rh = reader._Header[i];
                if (rh != reader.OutputColumn && !columnsWithConstValues.Contains(i))
                    layerInput.Neurons.Add(new Neuron(rh, 0));
            }

            NeuralLayer layerOutput = new NeuralLayer();
            layerOutput.Neurons.Add(new Neuron(reader.OutputColumn));

            var neuronsInHiddenLayer = (int)(layerInput.Neurons.Count * neuronsInHiddenMultiplyByInput);

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
                var neuronsInSecondHiddenLayer = (int)(layerInput.Neurons.Count * neuronsInSecondHiddenLayerMultiplyByInput);
                if (neuronsInSecondHiddenLayer > 0)
                {
                    NeuralLayer layerSecondHidden = new NeuralLayer();
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

            //nne.Train(data.Item1, data.Item2, epochs, learnRate, trainEndWithLossPercent);
            nne.Train(dataNormalizedInput, dataNormalizedOutput.Select(i => i.First()).ToList(), epochs, learnRate, trainEndWithLossPercent, minMaxScalerOutput);
        }
    }
}


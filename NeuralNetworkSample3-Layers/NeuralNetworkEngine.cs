using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkSample3_Layers
{
    public class NeuralNetworkEngine
    {
        private readonly NeuralLayer _LayerInput;

        public enum OnTrainProgressTime
        {
            BeforeCalculate,
            AfterBackPropagation
        }
        public delegate void OnTrainProgressDelegate(OnTrainProgressTime p_OnTrainProgressTime, NeuralLayer p_NeuralInputLayer, int p_Epoch, int p_DataIndex, List<double> p_Data, double p_ExpectedResult, double? p_PercentMissAll, double? p_PercentMiss);
        public event OnTrainProgressDelegate OnAfterTrainOneItem;

        public NeuralNetworkEngine(NeuralLayer p_Layer)
        {
            _LayerInput = p_Layer;
        }

        public Neuron FindNeuronById(string p_Id)
        {
            return _LayerInput.FindNeuronById(p_Id);

        }

        public List<double> Calculate(List<double> p_Data)
        {
            _LayerInput.NextLayer.CalculateInputs(p_Data);
            return _LayerInput.LastLayer.Neurons.Select(n => n.LastCalculatedOutputSigmoid).ToList();
        }

        public void Train(List<List<double>> p_Data, List<double> p_ExpectedResults, int p_Epochs = 1000, double p_LearnRate = 2.5, double p_TrainEndWithLossPercent = 0)
        {
            StringBuilder sb = new StringBuilder();
            _LayerInput.NextLayer.GetDebugInfo(sb);
            Console.WriteLine(sb.ToString());

            bool trained = false;
            for (int epoch = 0; epoch < p_Epochs; epoch++)
            {
                if (epoch == 0)
                    Console.WriteLine($"Epoch 0 START PercentMissAll: {PercentMiss(p_Data, p_ExpectedResults):f3}%");
                for (int x = 0; x < p_ExpectedResults.Count; x++)
                {
                    var percentMissAll = PercentMiss(p_Data, p_ExpectedResults);
                    var percentMiss = PercentMiss(p_Data[x], p_ExpectedResults[x]);
                    _LayerInput.NextLayer.CalculateInputs(p_Data[x]);
                    
                    OnAfterTrainOneItem?.Invoke(OnTrainProgressTime.BeforeCalculate, _LayerInput, epoch, x, p_Data[x], p_ExpectedResults[x], percentMissAll, percentMiss);
                    if (percentMissAll <= p_TrainEndWithLossPercent)
                    {
                        Console.WriteLine($"Epoch {epoch} DataIndex {x} TRAINED: PercentMissAll {percentMissAll:f3}% <= TrainEndWithLossPercent {p_TrainEndWithLossPercent}%");
                        trained = true;
                        break;
                    }

                    //Calculate partial derivatives.
                    //Naming: d_L_d_w1 represents "partial L / partial w1"
                    var o1 = _LayerInput.LastLayer.Neurons[0].LastCalculatedOutputSigmoid;
                    var partialDerivates = -2 * (p_ExpectedResults[x] - o1);
                    _LayerInput.LastLayer.Neurons[0].BackPropagate(p_Data[x], partialDerivates, p_LearnRate);
                    
                    percentMissAll = PercentMiss(p_Data, p_ExpectedResults);
                    percentMiss = PercentMiss(p_Data[x], p_ExpectedResults[x]);
                    OnAfterTrainOneItem?.Invoke(OnTrainProgressTime.AfterBackPropagation, _LayerInput, epoch, x, p_Data[x], p_ExpectedResults[x], percentMissAll, percentMiss);
                    if (percentMissAll <= p_TrainEndWithLossPercent)
                    {
                        Console.WriteLine($"Epoch {epoch} DataIndex {x} TRAINED: PercentMissAll {percentMissAll:f3}% <= TrainEndWithLossPercent {p_TrainEndWithLossPercent}%");
                        trained = true;
                        break;
                    }
                }

                if (trained)
                    break;

                //Calculate total loss at the end of each epoch
                if (epoch % 10 == 0 || epoch == p_Epochs - 1)
                {
                    var percentMissAll = PercentMiss(p_Data, p_ExpectedResults);
                    Console.WriteLine($"Epoch {epoch} END PercentMissAll: {percentMissAll:f3}%");
                }
            }

            sb = new StringBuilder();
            _LayerInput.NextLayer.GetDebugInfo(sb);
            Console.WriteLine(sb.ToString());
        }

        private double PercentMiss(List<double> p_Data, double p_ExpectedResults)
        {
            _LayerInput.NextLayer.CalculateInputs(p_Data);
            var o1 = _LayerInput.LastLayer.Neurons[0].LastCalculatedOutputSigmoid;

            var actual = o1;
            var expected = p_ExpectedResults;
            var percent = expected != 0
                ? (expected - actual) / expected * 100
                : actual * 100;
            return percent;
        }

        private double PercentMiss(List<List<double>> p_Data, List<double> p_ExpectedResults)
        {
            double percentSum = 0;
            int percentCount = 0;
            for (int x = 0; x < p_ExpectedResults.Count; x++)
            {
                _LayerInput.NextLayer.CalculateInputs(p_Data[x]);
                var o1 = _LayerInput.LastLayer.Neurons[0].LastCalculatedOutputSigmoid;

                var actual = o1;
                var expected = p_ExpectedResults[x];
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
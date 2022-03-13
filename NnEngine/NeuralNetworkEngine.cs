using System.Text;

namespace NnEngine
{
    public class NeuralNetworkEngine
    {
        public readonly NeuralLayer LayerInput;
        public MinMaxScaler MinMaxScalerInput, MinMaxScalerOutput;

        public List<NeuralLayer> LayersList()
        {
            List<NeuralLayer> result = new List<NeuralLayer>();
            var la = LayerInput;
            result.Add(la);
            while (la.NextLayer != null)
            {
                result.Add(la.NextLayer);
                la = la.NextLayer;
            }

            return result;
        }

        public enum OnTrainProgressTime
        {
            BeforeCalculate,
            AfterBackPropagation
        }
        public delegate void OnTrainProgressDelegate(OnTrainProgressTime p_OnTrainProgressTime, NeuralLayer p_NeuralInputLayer, int p_Epoch, int p_DataIndex, List<float> p_Data, float p_ExpectedResult, float? p_PercentMissAll, float? p_PercentMiss);
        public event OnTrainProgressDelegate OnAfterTrainOneItem;

        public NeuralNetworkEngine(NeuralLayer p_Layer)
        {
            LayerInput = p_Layer;
        }

        public Neuron FindNeuronById(string p_Id)
        {
            return LayerInput.FindNeuronById(p_Id);

        }

        public List<float> Calculate(List<float> p_Data)
        {
            var data = MinMaxScalerInput?.Transform(new List<List<float>>() {p_Data}, -0.5f).First().ToList() ?? p_Data;   //vystup -0.5 .. 0.5
            LayerInput.NextLayer.FeedForward(data);
            var calculated = LayerInput.LastLayer.Neurons.Select(n => n.LastCalculatedOutputActivated).ToList();
            var calculatedScaled = MinMaxScalerOutput?.InverseTransform(new List<List<float>>() {calculated}).First().ToList() ?? calculated; //vystup 0 .. 1
            return calculatedScaled;
        }

        public void Train(List<List<float>> p_Data, List<float> p_ExpectedResults, int p_Epochs = 1000,
            float p_LearnRate = 2.5f, float p_TrainEndWithLossPercent = 0, MinMaxScaler p_MinMaxScalerInput = null,
            MinMaxScaler p_MinMaxScalerOutput = null)
        {
            MinMaxScalerInput = p_MinMaxScalerInput;
            MinMaxScalerOutput = p_MinMaxScalerOutput;

            var data = p_MinMaxScalerInput?.Transform(p_Data, -0.5f) ?? p_Data;   //vystup -0.5 .. 0.5
            var dataOutput = p_MinMaxScalerOutput?.Transform(p_ExpectedResults.ConvertAll(i => new List<float>() {i})).Select(i => i.First()).ToList() ?? p_ExpectedResults; //vystup 0 .. 1

            StringBuilder sb = new StringBuilder();
            LayerInput.NextLayer.GetDebugInfo(sb);
            Console.WriteLine(sb.ToString());

            bool trained = false;
            for (int epoch = 0; epoch < p_Epochs; epoch++)
            {
                if (epoch == 0)
                    Console.WriteLine($"Epoch 0 START PercentMissAll: {PercentMiss(data, dataOutput):f3}%");
                for (int x = 0; x < dataOutput.Count; x++)
                {
                    var percentMissAll = PercentMiss(data, dataOutput);
                    var percentMiss = PercentMiss(data[x], dataOutput[x]);
                    LayerInput.NextLayer.FeedForward(data[x]);
                    
                    OnAfterTrainOneItem?.Invoke(OnTrainProgressTime.BeforeCalculate, LayerInput, epoch, x, data[x], dataOutput[x], percentMissAll, percentMiss);
                    if (percentMissAll <= p_TrainEndWithLossPercent)
                    {
                        Console.WriteLine($"Epoch {epoch} DataIndex {x} TRAINED: PercentMissAll {percentMissAll:f3}% <= TrainEndWithLossPercent {p_TrainEndWithLossPercent}%");
                        trained = true;
                        break;
                    }

                    BackPropagate(data[x], dataOutput[x], p_LearnRate);

                    percentMissAll = PercentMiss(data, dataOutput);
                    percentMiss = PercentMiss(data[x], dataOutput[x]);
                    OnAfterTrainOneItem?.Invoke(OnTrainProgressTime.AfterBackPropagation, LayerInput, epoch, x, data[x], dataOutput[x], percentMissAll, percentMiss);
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
                    var percentMissAll = PercentMiss(data, dataOutput);
                    Console.WriteLine($"Epoch {epoch} END PercentMissAll: {percentMissAll:f3}%");
                }
            }

            sb = new StringBuilder();
            LayerInput.NextLayer.GetDebugInfo(sb);
            Console.WriteLine(sb.ToString());
        }

        private void BackPropagate(List<float> p_Data, float p_ExpectedResult, float learningRate)
        {
            var layers = LayersList();

            List<float[]> gamma = new List<float[]>();
            for (int i = 0; i < layers.Count; i++)
                gamma.Add(new float[layers[i].Neurons.Count]);

            //Last layer
            var lastLayer = LayerInput.LastLayer;
            for (int iNeuron = 0; iNeuron < lastLayer.Neurons.Count; iNeuron++)
            {
                gamma[layers.Count - 1][iNeuron] = (lastLayer.Neurons[iNeuron].LastCalculatedOutputActivated - p_ExpectedResult) * lastLayer.ActivationFunction.CalculateDerivation(lastLayer.Neurons[iNeuron].LastCalculatedOutputActivated);
            }

            for (int iNeuron = 0; iNeuron < lastLayer.Neurons.Count; iNeuron++)
            {
                lastLayer.Neurons[iNeuron].Bias -= gamma[layers.Count - 1][iNeuron] * learningRate;
                foreach (var synapse in lastLayer.Neurons[iNeuron].SynapsesToPreviousLayer)
                {
                    synapse.Weight -= gamma[layers.Count - 1][iNeuron] * synapse.From.LastCalculatedOutputActivated * learningRate;
                }
            }

            //Hidden layers
            for (int iLayer = layers.Count - 2; iLayer > 0; iLayer--)
            {
                for (int iNeuron = 0; iNeuron < layers[iLayer].Neurons.Count; iNeuron++)
                {
                    gamma[iLayer][iNeuron] = 0;
                    for (int iNeuronNextLayer = 0; iNeuronNextLayer < layers[iLayer + 1].Neurons.Count; iNeuronNextLayer++)
                    {
                        gamma[iLayer][iNeuron] += gamma[iLayer + 1][iNeuronNextLayer] * layers[iLayer].Neurons[iNeuron].SynapsesToNextLayer[iNeuronNextLayer].Weight;
                    }

                    gamma[iLayer][iNeuron] *= layers[iLayer].ActivationFunction.CalculateDerivation(layers[iLayer].Neurons[iNeuron].LastCalculatedOutputActivated);
                }

                for (int iNeuron = 0; iNeuron < layers[iLayer].Neurons.Count; iNeuron++)
                {
                    layers[iLayer].Neurons[iNeuron].Bias -= gamma[iLayer][iNeuron] * learningRate;
                    for (int iNeuronPrevLayer = 0; iNeuronPrevLayer < layers[iLayer - 1].Neurons.Count; iNeuronPrevLayer++)
                    {
                        var neuronValue = layers[iLayer - 1].IsInputLayer ? p_Data[iNeuronPrevLayer] : layers[iLayer - 1].Neurons[iNeuronPrevLayer].LastCalculatedOutputActivated;
                        layers[iLayer].Neurons[iNeuron].SynapsesToPreviousLayer[iNeuronPrevLayer].Weight -= gamma[iLayer][iNeuron] * neuronValue * learningRate;
                    }
                }
            }
        }

        private float PercentMiss(List<float> p_Data, float p_ExpectedResults)
        {
            LayerInput.NextLayer.FeedForward(p_Data);
            var o1 = LayerInput.LastLayer.Neurons[0].LastCalculatedOutputActivated;

            var actual = o1;
            var expected = p_ExpectedResults;
            var percent = expected != 0
                ? (expected - actual) / expected * 100
                : actual * 100;
            return Math.Abs(percent);
        }

        private float PercentMiss(List<List<float>> p_Data, List<float> p_ExpectedResults)
        {
            float percentSum = 0;
            int percentCount = 0;
            for (int x = 0; x < p_ExpectedResults.Count; x++)
            {
                LayerInput.NextLayer.FeedForward(p_Data[x]);
                var o1 = LayerInput.LastLayer.Neurons[0].LastCalculatedOutputActivated;

                var actual = o1;
                var expected = p_ExpectedResults[x];
                var percent = expected != 0
                    ? (expected - actual) / expected * 100
                    : actual * 100;
                percentSum += Math.Abs(percent);
                percentCount++;
            }

            double percentMiss = percentSum / percentCount;
            return (float)percentMiss;
        }
    }
}
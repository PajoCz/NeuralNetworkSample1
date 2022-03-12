using System.Text;

namespace NnEngine
{
    public class NeuralNetworkEngine
    {
        private readonly NeuralLayer _LayerInput;

        private List<NeuralLayer> LayersList()
        {
            List<NeuralLayer> result = new List<NeuralLayer>();
            var la = _LayerInput;
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
            _LayerInput = p_Layer;
        }

        public Neuron FindNeuronById(string p_Id)
        {
            return _LayerInput.FindNeuronById(p_Id);

        }

        public List<float> Calculate(List<float> p_Data)
        {
            _LayerInput.NextLayer.FeedForward(p_Data);
            return _LayerInput.LastLayer.Neurons.Select(n => n.LastCalculatedOutputActivated).ToList();
        }

        public void Train(List<List<float>> p_Data, List<float> p_ExpectedResults, int p_Epochs = 1000,
            float p_LearnRate = 2.5f, float p_TrainEndWithLossPercent = 0, MinMaxScaler p_MinMaxScalerOutput = null)
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
                    _LayerInput.NextLayer.FeedForward(p_Data[x]);
                    
                    OnAfterTrainOneItem?.Invoke(OnTrainProgressTime.BeforeCalculate, _LayerInput, epoch, x, p_Data[x], p_ExpectedResults[x], percentMissAll, percentMiss);
                    if (percentMissAll <= p_TrainEndWithLossPercent)
                    {
                        Console.WriteLine($"Epoch {epoch} DataIndex {x} TRAINED: PercentMissAll {percentMissAll:f3}% <= TrainEndWithLossPercent {p_TrainEndWithLossPercent}%");
                        trained = true;
                        break;
                    }


                    ////Calculate partial derivatives.
                    ////Naming: d_L_d_w1 represents "partial L / partial w1"
                    //var o1 = _LayerInput.LastLayer.Neurons[0].LastCalculatedOutputActivated;
                    ////if (p_MinMaxScalerOutput != null)
                    ////    o1 = p_MinMaxScalerOutput.InverseTransform(new List<List<float>>() { new List<float>() { o1 } }).First().First();
                    //var partialDerivates = -2 * (p_ExpectedResults[x] - o1);
                    //_LayerInput.LastLayer.Neurons[0].BackPropagate(p_Data[x], partialDerivates, p_LearnRate);

                    BackPropagate(p_Data[x], p_ExpectedResults[x], p_LearnRate);


                    //var layers = LayersList();

                    //float[][] gamma;


                    //List<float[]> gammaList = new List<float[]>();
                    //for (int i = 0; i < layers.Count; i++)
                    //{
                    //    gammaList.Add(new float[layers[i].Neurons.Count]);
                    //}
                    //gamma = gammaList.ToArray();//gamma initialization

                    //for (int i = 0; i < _LayerInput.LastLayer.Neurons.Count; i++) gamma[layers.Count - 1][i] = (p_ExpectedResults[x] - _LayerInput.LastLayer.Neurons[i].LastCalculatedOutputSigmoid) * -(layers.Count-1);//Gamma calculation
                    //for (int i = 0; i < _LayerInput.LastLayer.Neurons.Count; i++)//calculates the w' and b' for the last layer in the network
                    //{
                    //    _LayerInput.LastLayer.Neurons[i].Bias -= gamma[layers.Count - 1][i] * Neuron.DerivSigmoid(_LayerInput.LastLayer.Neurons[i].LastCalculatedOutput) * p_LearnRate;
                    //    for (int j = 0; j < _LayerInput.LastLayer.Neurons[i].SynapsesToPreviousLayer.Count; j++)
                    //    {
                    //        _LayerInput.LastLayer.Neurons[i].SynapsesToPreviousLayer[j].Weight -= gamma[layers.Count - 1][i] * Neuron.DerivSigmoid(_LayerInput.LastLayer.Neurons[i].LastCalculatedOutput) * _LayerInput.LastLayer.Neurons[i].SynapsesToPreviousLayer[j].From.LastCalculatedOutputSigmoid * p_LearnRate;//*learning 
                    //    }
                    //}

                    //for (int i = layers.Count - 2; i > 0; i--)//runs on all hidden layers
                    //{
                    //    for (int j = 0; j < layers[i].Neurons.Count; j++)//outputs
                    //    {
                    //        gamma[i][j] = 0;
                    //        for (int k = 0; k < gamma[i + 1].Length; k++)
                    //        {
                    //            gamma[i][j] += gamma[i + 1][k] * Neuron.DerivSigmoid(layers[i].Neurons[j].SynapsesToNextLayer[k].To.LastCalculatedOutput) * layers[i].Neurons[j].SynapsesToNextLayer[k].Weight;
                    //        }

                    //        //gamma[i][j] *= layers[i].Neurons[j].LastCalculatedOutputSigmoid;
                    //    }
                    //    for (int j = 0; j < layers[i].Neurons.Count; j++)//itterate over outputs of layer
                    //    {
                    //        layers[i].Neurons[j].Bias -= gamma[i][j] * Neuron.DerivSigmoid(layers[i].Neurons[j].LastCalculatedOutput) * p_LearnRate;
                    //        for (int k = 0; k < layers[i - 1].Neurons[j].SynapsesToPreviousLayer.Count; k++)//itterate over inputs to layer
                    //        {
                    //            layers[i].Neurons[j].SynapsesToPreviousLayer[k].Weight -= gamma[i][j] * layers[i - 1].Neurons[j].LastCalculatedOutputSigmoid * p_LearnRate;
                    //        }

                    //        if (!layers[i - 1].Neurons[0].SynapsesToPreviousLayer.Any())
                    //            for (int k = 0; k < p_Data[x].Count; k++)
                    //                layers[i].Neurons[j].SynapsesToPreviousLayer[k].Weight -= gamma[i][j] * p_Data[x][k] * Neuron.DerivSigmoid(layers[i].Neurons[j].LastCalculatedOutput) * p_LearnRate;
                    //    }
                    //}

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

        private void BackPropagate(List<float> p_Data, float p_ExpectedResult, float learningRate)
        {
            var layers = LayersList();

            List<float[]> gamma = new List<float[]>();
            for (int i = 0; i < layers.Count; i++)
                gamma.Add(new float[layers[i].Neurons.Count]);

            //Last layer
            var lastLayer = _LayerInput.LastLayer;
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
            _LayerInput.NextLayer.FeedForward(p_Data);
            var o1 = _LayerInput.LastLayer.Neurons[0].LastCalculatedOutputActivated;

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
                _LayerInput.NextLayer.FeedForward(p_Data[x]);
                var o1 = _LayerInput.LastLayer.Neurons[0].LastCalculatedOutputActivated;

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
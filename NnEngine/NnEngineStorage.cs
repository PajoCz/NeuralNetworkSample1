using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnEngine
{
    public class NnEngineStorage
    {
        private const string FILE_FIRST_LINE = "NnEngineStorageV1";
        private const string Key_MinMaxScalerInput = "MinMaxScalerInput";
        private const string Key_MinMaxScalerOutput = "MinMaxScalerOutput";
        private const string Key_Layer = "Layer";
        private const string Key_Neurons = "Neurons";
        private const string Key_SynapsesToPreviousLayer = "SynapsesToPreviousLayer";

        public void SaveToStream(NeuralNetworkEngine p_NnEngine, Stream p_Stream)
        {
            using (StreamWriter sw = new StreamWriter(p_Stream))
            {
                sw.WriteLine(FILE_FIRST_LINE);

                sw.WriteLine($"{Key_MinMaxScalerInput}={string.Join("|", p_NnEngine.MinMaxScalerInput.ColumnMinMaxValues.Select(i => $"{i.Min};{i.Max}"))}");
                sw.WriteLine($"{Key_MinMaxScalerOutput}={string.Join("|", p_NnEngine.MinMaxScalerOutput.ColumnMinMaxValues.Select(i => $"{i.Min};{i.Max}"))}");

                var layers = p_NnEngine.LayersList();
                for (var iLayer = 0; iLayer < layers.Count; iLayer++)
                {
                    var layer = layers[iLayer];
                    sw.WriteLine($"{Key_Layer}={layer.ActivationFunction}");
                    StringBuilder sbSynapse = new StringBuilder();
                    sw.Write($"{Key_Neurons}=");
                    for (int iNeuron = 0; iNeuron < layers[iLayer].Neurons.Count; iNeuron++)
                    {
                        var neuron = layer.Neurons[iNeuron];
                        sw.Write($"{neuron.Id};{neuron.Bias};");
                        for (int iSynapse = 0; iSynapse < layers[iLayer].Neurons[iNeuron].SynapsesToPreviousLayer.Count; iSynapse++)
                        {
                            var synapse = neuron.SynapsesToPreviousLayer[iSynapse];
                            sbSynapse.Append($"{synapse.Weight};");
                        }
                    }
                    sw.WriteLine();
                    sw.WriteLine($"{Key_SynapsesToPreviousLayer}=" + sbSynapse);
                }
            }
        }

        public NeuralNetworkEngine LoadFromStream(Stream p_Stream)
        {
            using (StreamReader sw = new StreamReader(p_Stream))
            {
                var version = sw.ReadLine();
                if (version != FILE_FIRST_LINE)
                    throw new Exception($"File first line must be '{FILE_FIRST_LINE}' but value is '{version}'");

                string lineValue;
                if (!ReadLine(sw, Key_MinMaxScalerInput, out lineValue)) 
                    return null;
                var minMaxScalerInput = lineValue.Split('|');

                if (!ReadLine(sw, Key_MinMaxScalerOutput, out lineValue)) 
                    return null;
                var minMaxScalerOutput = lineValue.Split('|');

                var layer = LoadLayer(sw);
                var result = new NeuralNetworkEngine(layer.Item1);
                while (layer != null)
                {
                    var prevLayer = layer.Item1;
                    layer = LoadLayer(sw);
                    if (layer != null)
                        prevLayer.AddNextLayer(layer.Item1, layer.Item2);
                }

                result.MinMaxScalerInput = new MinMaxScaler();
                foreach (var mms in minMaxScalerInput)
                {
                    var mmsSplitted = mms.Split(';');
                    result.MinMaxScalerInput.ColumnMinMaxValues.Add(new MinMaxScaler.MinMaxValue(float.Parse(mmsSplitted[0]), float.Parse(mmsSplitted[1])));
                }

                result.MinMaxScalerOutput = new MinMaxScaler();
                foreach (var mms in minMaxScalerOutput)
                {
                    var mmsSplitted = mms.Split(';');
                    result.MinMaxScalerOutput.ColumnMinMaxValues.Add(new MinMaxScaler.MinMaxValue(float.Parse(mmsSplitted[0]), float.Parse(mmsSplitted[1])));
                }

                return result;
            }
        }

        private static bool ReadLine(StreamReader sw, string expectedLine, out string? lineValue)
        {
            lineValue = null;
            var line = sw.ReadLine();
            if (line == null)
                return false;
            var lineParsed = line.Split('=');
            if (lineParsed[0] != expectedLine)
                throw new Exception($"{expectedLine} data expected");
            lineValue = lineParsed[1];
            return true;
        }

        private static Tuple<NeuralLayer, List<float>>? LoadLayer(StreamReader sw)
        {
            string? lineValue;
            if (!ReadLine(sw, Key_Layer, out lineValue)) 
                return null;
            var lineValues = lineValue.Split(';');

            var activationFunction = string.IsNullOrEmpty(lineValues[0]) ? null : Activator.CreateInstance("NnEngine", lineValues[0])?.Unwrap() as IActivationFunction;
            NeuralLayer layer = new NeuralLayer(activationFunction, true);

            ReadLine(sw, Key_Neurons, out lineValue);
            lineValues = lineValue.Trim(';').Split(';');
            for (int iNeuron = 0; iNeuron < lineValues.Length / 2; iNeuron++)
                layer.Neurons.Add(new Neuron(lineValues[iNeuron * 2], float.Parse(lineValues[iNeuron * 2 + 1])));

            ReadLine(sw, Key_SynapsesToPreviousLayer, out lineValue);
            var synapseWeights = string.IsNullOrEmpty(lineValue) ? null : lineValue.Trim(';').Split(';').Select(i => float.Parse(i)).ToList();

            return new Tuple<NeuralLayer, List<float>>(layer, synapseWeights);
        }
    }
}

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
        public void SaveToStream(NeuralNetworkEngine p_NnEngine, Stream p_Stream)
        {
            using (StreamWriter sw = new StreamWriter(p_Stream))
            {
                sw.WriteLine(FILE_FIRST_LINE);

                sw.WriteLine($"MinMaxScalerInput={string.Join("|", p_NnEngine.MinMaxScalerInput.ColumnMinMaxValues.Select(i => $"{i.Min};{i.Max}"))}");
                sw.WriteLine($"MinMaxScalerOutput={string.Join("|", p_NnEngine.MinMaxScalerOutput.ColumnMinMaxValues.Select(i => $"{i.Min};{i.Max}"))}");

                var layers = p_NnEngine.LayersList();
                for (var iLayer = 0; iLayer < layers.Count; iLayer++)
                {
                    var layer = layers[iLayer];
                    sw.WriteLine($"Layer={layer.ActivationFunction}");
                    StringBuilder sbSynapse = new StringBuilder();
                    sw.Write("Neurons=");
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
                    sw.WriteLine("SynapsesToPreviousLayer=" + sbSynapse);
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

                var line = sw.ReadLine();
                if (line == null)
                    return null;
                var lineParsed = line.Split('=');
                if (lineParsed[0] != "MinMaxScalerInput")
                    throw new Exception("MinMaxScalerInput data expected");
                var minMaxScalerInput = lineParsed[1].Split('|');

                line = sw.ReadLine();
                if (line == null)
                    return null;
                lineParsed = line.Split('=');
                if (lineParsed[0] != "MinMaxScalerOutput")
                    throw new Exception("MinMaxScalerOutput data expected");
                var minMaxScalerOutput = lineParsed[1].Split('|');

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

        private static Tuple<NeuralLayer, List<float>> LoadLayer(StreamReader sw)
        {
            var line = sw.ReadLine();
            if (line == null)
                return null;
            var lineParsed = line.Split('=');
            if (lineParsed[0] != "Layer")
                throw new Exception("Layer data expected");
            var lineValues = lineParsed[1].Split(';');

            var activationFunction = string.IsNullOrEmpty(lineValues[0])
                ? null
                : Activator.CreateInstance("NnEngine", lineValues[0])?.Unwrap() as IActivationFunction;
            NeuralLayer layer = new NeuralLayer(activationFunction, true);

            line = sw.ReadLine();
            lineParsed = line.Split('=');
            if (lineParsed[0] != "Neurons")
                throw new Exception("Neurons data expected");
            lineValues = lineParsed[1].Trim(';').Split(';');
            for (int iNeuron = 0; iNeuron < lineValues.Length / 2; iNeuron++)
                layer.Neurons.Add(new Neuron(lineValues[iNeuron * 2], float.Parse(lineValues[iNeuron * 2 + 1])));

            line = sw.ReadLine();
            lineParsed = line.Split('=');
            if (lineParsed[0] != "SynapsesToPreviousLayer")
                throw new Exception("SynapsesToPreviousLayer data expected");
            var synapseWeights = string.IsNullOrEmpty(lineParsed[1]) ? null : lineParsed[1].Trim(';').Split(';').Select(i => float.Parse(i)).ToList();

            return new Tuple<NeuralLayer, List<float>>(layer, synapseWeights);
        }
    }
}

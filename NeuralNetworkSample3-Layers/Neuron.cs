using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkSample3_Layers
{
    public class Neuron
    {
        public readonly string Id;
        private static Random _Rnd = new Random();
        public Neuron(string id)
        {
            Id = id;
            Bias = (_Rnd.NextDouble()-0.5) * 2;
        }

        public List<Synapse> SynapsesToNextLayer { get; set; } = new List<Synapse>();
        public List<Synapse> SynapsesToPreviousLayer { get; set; } = new List<Synapse>();

        public Neuron(string id, double bias)
        {
            Id = id;
            Bias = bias;
        }

        public void InitValues(List<double> p_Weights, double p_Bias)
        {
            if (p_Weights.Count != SynapsesToPreviousLayer.Count)
                throw new ArgumentOutOfRangeException(nameof(p_Weights));

            for (int i = 0; i < p_Weights.Count; i++)
                SynapsesToPreviousLayer[i].Weight = p_Weights[i];
            Bias = p_Bias;
        }

        public double Bias { get; set; }
        public double LastCalculatedOutput { get; set; }
        public double LastCalculatedOutputSigmoid { get; set; }

        public virtual double CalcBySigmoid(List<double> p_Inputs)
        {
            if (p_Inputs.Count != SynapsesToPreviousLayer.Count)
                throw new Exception($"Expected {SynapsesToPreviousLayer.Count}x inputs but input has {p_Inputs.Count} numbers");
            LastCalculatedOutput = 0;
            for (int i = 0; i < p_Inputs.Count; i++)
                LastCalculatedOutput += SynapsesToPreviousLayer[i].Weight * p_Inputs[i];
            LastCalculatedOutput += Bias;
            LastCalculatedOutputSigmoid = Sigmoid(LastCalculatedOutput);
            return LastCalculatedOutputSigmoid;
        }

        public void BackPropagate(List<double> p_Data, double p_PartialDerivates, double p_LearnRate)
        {
            var inputs = SynapsesToPreviousLayer.Exists(i => i.From.SynapsesToPreviousLayer.Any())
                ? SynapsesToPreviousLayer.Select(i => i.From.LastCalculatedOutputSigmoid).ToList()
                : p_Data;
            var bias = DerivSigmoid(LastCalculatedOutput);
            for (int i = 0; i < inputs.Count; i++)
                SynapsesToPreviousLayer[i].Weight -= p_LearnRate * p_PartialDerivates * inputs[i] * bias;

            Bias -= p_LearnRate * p_PartialDerivates * bias;

            if (SynapsesToPreviousLayer.Exists(i => i.From.SynapsesToPreviousLayer.Any()))
            {
                var weightsMultiplyByDerivatesSigmoidLastOutput = WeightsMultiplyByDerivatesSigmoidLastOutput();
                for (int i = 0; i < SynapsesToPreviousLayer.Count; i++)
                    SynapsesToPreviousLayer[i].From.BackPropagate(p_Data, p_PartialDerivates * weightsMultiplyByDerivatesSigmoidLastOutput[i], p_LearnRate);
            }
        }

        private List<double> WeightsMultiplyByDerivatesSigmoidLastOutput()
        {
            List<double> result = new List<double>(SynapsesToPreviousLayer.Count);
            var deriv = DerivSigmoid(LastCalculatedOutput);
            foreach (var syn in SynapsesToPreviousLayer)
            {
                result.Add(syn.Weight * deriv);
            }

            return result;
        }


        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private double DerivSigmoid(double x)
        {
            var sig = Sigmoid(x);
            return sig * (1 - sig);
        }
    }
}
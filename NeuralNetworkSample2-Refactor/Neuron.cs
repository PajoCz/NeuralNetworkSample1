using System;
using System.Collections.Generic;

namespace NeuralNetworkSample2_Refactor
{
    public class Neuron
    {
        private readonly int _InputsCount;
        private readonly string _Id;
        private static Random _Rnd = new Random();
        public Neuron(string id, int inputsCount)
        {
            _Id = id;
            _InputsCount = inputsCount;
            Weights = new List<double>(inputsCount);
            for(int i=0; i<inputsCount;i++)
                Weights.Add(_Rnd.NextDouble());
            Bias = _Rnd.NextDouble();
        }

        public Neuron(string id, List<double> weights, double bias)
        {
            _Id = id;
            _InputsCount = weights.Count;
            Weights = weights;
            Bias = bias;
        }

        public List<double> Weights { get; set; }
        public double Bias { get; set; }
        public double LastCalculatedOutput { get; set; }

        public double CalcBySigmoid(List<double> inputs)
        {
            if (inputs.Count != _InputsCount)
                throw new Exception($"Expected {_InputsCount}x inputs but input has {inputs.Count} numbers");
            LastCalculatedOutput = 0;
            for (int i = 0; i < inputs.Count; i++)
                LastCalculatedOutput += Weights[i] * inputs[i];
            LastCalculatedOutput += Bias;
            return Sigmoid(LastCalculatedOutput);
        }

        public void BackPropagate(List<double> inputs, double partialDerivates, double learn_rate = 2.5)
        {
            var bias = DerivSigmoid(LastCalculatedOutput);
            for (int i = 0; i < inputs.Count; i++)
                Weights[i] -= learn_rate * partialDerivates * inputs[i] * bias;

            Bias -= learn_rate * partialDerivates * bias;
        }

        public List<double> WeightsMultiplyByDerivatesSigmoidLastOutput()
        {
            List<double> result = new List<double>(Weights.Count);
            var deriv = DerivSigmoid(LastCalculatedOutput);
            foreach (var weight in Weights)
            {
                result.Add(weight * deriv);
            }

            return result;
        }


        double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        double DerivSigmoid(double x)
        {
            var sig = Sigmoid(x);
            return sig * (1 - sig);
        }
    }
}
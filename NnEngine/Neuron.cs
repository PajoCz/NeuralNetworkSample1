namespace NnEngine
{
    public class Neuron
    {
        public readonly string Id;
        private static Random _Rnd = new Random();
        public Neuron(string id)
        {
            Id = id;
            Bias = (_Rnd.NextSingle()-0.5f) * 2f;
        }

        public List<Synapse> SynapsesToNextLayer { get; set; } = new List<Synapse>();
        public List<Synapse> SynapsesToPreviousLayer { get; set; } = new List<Synapse>();

        public Neuron(string id, float bias)
        {
            Id = id;
            Bias = bias;
        }

        public void InitValues(List<float> p_Weights, float p_Bias)
        {
            if (p_Weights.Count != SynapsesToPreviousLayer.Count)
                throw new ArgumentOutOfRangeException(nameof(p_Weights));

            for (int i = 0; i < p_Weights.Count; i++)
                SynapsesToPreviousLayer[i].Weight = p_Weights[i];
            Bias = p_Bias;
        }

        public float Bias { get; set; }
        public float LastCalculatedOutput { get; set; }
        public float LastCalculatedOutputSigmoid { get; set; }

        public virtual float CalcBySigmoid(List<float> p_Inputs)
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

        public void BackPropagate(List<float> p_Data, float p_PartialDerivates, float p_LearnRate)
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

        private List<float> WeightsMultiplyByDerivatesSigmoidLastOutput()
        {
            List<float> result = new List<float>(SynapsesToPreviousLayer.Count);
            var deriv = DerivSigmoid(LastCalculatedOutput);
            foreach (var syn in SynapsesToPreviousLayer)
            {
                result.Add(syn.Weight * deriv);
            }

            return result;
        }


        private float Sigmoid(float x)
        {
            return (float)(1 / (1 + Math.Exp(-x)));
        }

        private float DerivSigmoid(float x)
        {
            var sig = Sigmoid(x);
            return sig * (1 - sig);
        }
    }
}
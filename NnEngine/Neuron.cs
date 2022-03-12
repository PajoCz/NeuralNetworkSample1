namespace NnEngine
{
    public class Neuron
    {
        public readonly string Id;
        private static readonly Random _Rnd = new();

        public Neuron(string id)
        {
            Id = id;
            Bias = (_Rnd.NextSingle() - 0.5f) * 2f;
        }

        public List<Synapse> SynapsesToNextLayer { get; set; } = new();
        public List<Synapse> SynapsesToPreviousLayer { get; set; } = new();

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
        public float LastCalculatedOutputActivated { get; set; }

        public virtual float Calculate(List<float> p_Inputs, IActivationFunction p_ActivationFunction)
        {
            if (p_Inputs.Count != SynapsesToPreviousLayer.Count)
                throw new Exception(
                    $"Expected {SynapsesToPreviousLayer.Count}x inputs but input has {p_Inputs.Count} numbers");
            LastCalculatedOutput = 0;
            for (int i = 0; i < p_Inputs.Count; i++)
                LastCalculatedOutput += SynapsesToPreviousLayer[i].Weight * p_Inputs[i];
            LastCalculatedOutput += Bias;
            LastCalculatedOutputActivated = p_ActivationFunction.Calculate(LastCalculatedOutput);
            return LastCalculatedOutputActivated;
        }
    }
}
namespace NnEngine
{
    public class Synapse
    {
        public Synapse(Neuron p_From, Neuron p_To, float? p_Weight = null)
        {
            From = p_From;
            To = p_To;
            Weight = p_Weight ?? (new Random().NextSingle()-0.5f) * 2;
        }

        public Neuron From { get; set; }
        public Neuron To { get; set; }
        public float Weight { get; set; }
    }
}

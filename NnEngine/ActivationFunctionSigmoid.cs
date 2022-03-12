using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnEngine
{
    public class ActivationFunctionSigmoid : IActivationFunction
    {
        public float Calculate(float input)
        {
            float k = (float)Math.Exp(input);
            return k / (1.0f + k);
        }

        public float CalculateDerivation(float input)
        {
            return input * (1 - input);
        }
    }
}

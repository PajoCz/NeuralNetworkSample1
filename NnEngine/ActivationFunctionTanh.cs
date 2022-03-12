using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnEngine
{
    public class ActivationFunctionTanh : IActivationFunction
    {
        public float Calculate(float input)
        {
            return (float)Math.Tanh(input);
        }

        public float CalculateDerivation(float input)
        {
            return 1 - input * input;
        }
    }
}

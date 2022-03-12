using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnEngine
{
    public class ActivationFunctionRelu : IActivationFunction
    {
        public float Calculate(float input)
        {
            return input <= 0 ? 0 : input;
        }

        public float CalculateDerivation(float input)
        {
            return input <= 0 ? 0 : 1;
        }
    }
}

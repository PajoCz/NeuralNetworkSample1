using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnEngine
{
    public class ActivationFunctionLinear : IActivationFunction
    {
        private readonly float _M;

        public ActivationFunctionLinear(float m = 1)
        {
            _M = m;
        }

        public float Calculate(float input)
        {
            return input * _M;
        }

        public float CalculateDerivation(float input)
        {
            return _M;
        }
    }
}

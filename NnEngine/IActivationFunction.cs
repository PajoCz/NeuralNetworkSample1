using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnEngine
{
    public interface IActivationFunction
    {
        float Calculate(float input);
        float CalculateDerivation(float input);
    }
}

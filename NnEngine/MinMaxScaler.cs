using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnEngine
{
    public class MinMaxScaler
    {
        public class MinMaxValue
        {
            public MinMaxValue()
            {
            }

            public MinMaxValue(float p_Min, float p_Max)
            {
                Min = p_Min;
                Max = p_Max;
            }

            public float Min { get; set; } = float.MaxValue;
            public float Max { get; set; } = float.MinValue;

            public float MaxMinusMin => Max - Min;
        }
        public List<MinMaxValue> ColumnMinMaxValues = new List<MinMaxValue>();

        public List<int> ColumnsWithConstValues()
        {
            List<int> result = new List<int>();
            for(int i= 0; i < ColumnMinMaxValues.Count; i++)
                if (ColumnMinMaxValues[i].MaxMinusMin == 0)
                    result.Add(i);
            return result;
        }

        public void Fit(List<List<float>> input)
        {
            foreach (var inputItem in input)
                for (var iColumn = 0; iColumn < inputItem.Count; iColumn++)
                {
                    if (ColumnMinMaxValues.Count < inputItem.Count)
                        ColumnMinMaxValues.Add(new MinMaxValue());

                    var val = inputItem[iColumn];
                    var colMinMax = ColumnMinMaxValues[iColumn];
                    if (val < colMinMax.Min)
                        colMinMax.Min = val;
                    if (val > colMinMax.Max)
                        colMinMax.Max = val;
                }
        }

        public List<List<float>> Transform(List<List<float>> input, float resultInc = 0f)
        {
            List<List<float>> result = new List<List<float>>(input.Count);
            for (var iInput = 0; iInput < input.Count; iInput++)
            {
                var inputItem = input[iInput];
                List<float> resultItem = new List<float>(inputItem.Count);
                for (var iColumn = 0; iColumn < inputItem.Count; iColumn++)
                {
                    var colMinMax = ColumnMinMaxValues[iColumn];
                    if (colMinMax.MaxMinusMin != 0)
                        resultItem.Add((inputItem[iColumn] - colMinMax.Min)/colMinMax.MaxMinusMin + resultInc);
                }
                result.Add(resultItem);
            }

            return result;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public List<List<float>> InverseTransform(List<List<float>> input, float resultInc = 0f)
        {
            List<List<float>> result = new List<List<float>>(input.Count);
            for (var iInput = 0; iInput < input.Count; iInput++)
            {
                var inputItem = input[iInput];
                List<float> resultItem = new List<float>(inputItem.Count);
                for (var iColumn = 0; iColumn < inputItem.Count; iColumn++)
                {
                    var colMinMax = ColumnMinMaxValues[iColumn];
                    resultItem.Add(inputItem[iColumn] * colMinMax.MaxMinusMin + colMinMax.Min + resultInc);
                }
                result.Add(resultItem);
            }

            return result;
        }
    }
}

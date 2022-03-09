using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NnByInputCsv
{
    internal class DatasetReader
    {
        private StreamReader reader;
        public readonly string OutputColumn;
        private readonly int _OutputColumnIndex;
        private char separator;
        public readonly List<string> _Header;

        public DatasetReader(Stream p_Stream, string p_OutputColumn = null, char p_Separator = ';')
        {
            OutputColumn = p_OutputColumn;
            separator = p_Separator;
            reader = new StreamReader(p_Stream);
            var line = reader.ReadLine();
            if (!string.IsNullOrEmpty(line))
                _Header = line.Split(p_Separator).ToList();
            if (!string.IsNullOrEmpty(p_OutputColumn))
            {
                _OutputColumnIndex = _Header.IndexOf(OutputColumn);
                if (_OutputColumnIndex < 0)
                    throw new ArgumentException($"Output column '{p_OutputColumn}' not found in input data stream");
            }
            else
            {
                _OutputColumnIndex = _Header.Count - 1; //last
                OutputColumn = _Header[_OutputColumnIndex];
            }
        }

        public IEnumerable<List<string>> ReadLineData()
        {
            var line = reader.ReadLine();
            if (!string.IsNullOrEmpty(line))
                yield return line.Split(separator).ToList();
        }

        public Tuple<List<List<float>>, List<float>> ReadLinesNumbers()
        {
            string line;
            List<List<float>> inputs = new List<List<float>>();
            List<float> outputs = new List<float>();
            int iLine = 1;  //header
            while (!string.IsNullOrEmpty(line = reader.ReadLine()))
            {
                iLine++;
                var splitted = line.Split(separator).ToList();
                if (splitted.Count != _Header.Count)
                    throw new Exception(
                        $"Line {iLine} contains {splitted.Count} values, but header contains {_Header.Count} values");
                var outputValue = float.Parse(splitted[_OutputColumnIndex].Replace('.',','));
                splitted.RemoveAt(_OutputColumnIndex);
                var inputValues = splitted.ConvertAll(i => float.Parse(i.Replace('.',',')));

                inputs.Add(inputValues);
                outputs.Add(outputValue);
            }

            return new Tuple<List<List<float>>, List<float>>(inputs, outputs);
        }
    }
}

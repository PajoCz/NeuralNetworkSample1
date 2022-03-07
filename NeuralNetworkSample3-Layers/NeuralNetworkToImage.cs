using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkSample3_Layers
{
    public class NeuralNetworkToImage
    {
        public Bitmap Draw(NeuralLayer p_NeuralNetwork, int p_Width, int p_Height, int p_Epoch, int p_DataIndex,
            List<double> p_Data,
            double p_ExpectedResult,
            double? p_PercentMissAll, double? p_PercentMiss,
            NeuralNetworkEngine.OnTrainProgressTime p_OnTrainProgressTime)
        {
            Bitmap bitmap = new Bitmap(p_Width, p_Height);
            using Graphics graphics = Graphics.FromImage(bitmap);
            graphics.TextRenderingHint = TextRenderingHint.AntiAlias;
            graphics.SmoothingMode = SmoothingMode.AntiAlias;

            int maxNeuronsInLayer = 0;
            var l = p_NeuralNetwork;
            List<NeuralLayer> layers = new List<NeuralLayer>();
            while (l != null)
            {
                layers.Add(l);
                maxNeuronsInLayer = l.Neurons.Count > maxNeuronsInLayer ? l.Neurons.Count : maxNeuronsInLayer;
                l = l.NextLayer;
            }

            if (maxNeuronsInLayer == 0)
                return null;

            const int widthBorder = 150;
            const int heightBorder = 80;
            const int neuronSize = 50;

            var neuronXInc = (p_Width - 2 * widthBorder) / (layers.Count - 1);

            List<Brush> brushNeurons = new List<Brush>()
            {
                new SolidBrush(Color.Red), new SolidBrush(Color.Green), new SolidBrush(Color.Blue),
                new SolidBrush(Color.BlueViolet), new SolidBrush(Color.Coral)
            };
            using Brush brushInputNeurons = new SolidBrush(Color.Black);
            List<Pen> penNeurons = brushNeurons.Select(i => new Pen(i)).ToList();
            using Pen penInputNeurons = new Pen(brushInputNeurons);
            
            using Font fontNeuron = new Font(FontFamily.GenericMonospace, 32, FontStyle.Bold, GraphicsUnit.Pixel);
            using Font fontNeuronBias = new Font(FontFamily.GenericMonospace, 12, FontStyle.Regular, GraphicsUnit.Pixel);

            graphics.DrawRectangle(new Pen(Color.Red), 0, 0, p_Width-1, p_Height-1);
            graphics.DrawString($"Epoch: {p_Epoch} ; DataIndex: {p_DataIndex} ; PercentMissAll: {p_PercentMissAll:f3} ; PercentMissItem: {p_PercentMiss:f3} ; {p_OnTrainProgressTime}", fontNeuronBias, new SolidBrush(Color.Black), 1, 13);

            int neuronPenBrushIndex = 0;
            for (int iLayer = 0; iLayer < layers.Count; iLayer++)
            {
                var layer = layers[iLayer];
                var neuronYInc = layer.Neurons.Count > 1 ? (p_Height - 2 * heightBorder) / (layer.Neurons.Count - 1) : 0;
                for (int iNeuron = 0; iNeuron < layer.Neurons.Count; iNeuron++)
                {
                    var penNeuron = iLayer == 0 ? penInputNeurons : penNeurons[neuronPenBrushIndex];
                    var brushNeuron = iLayer == 0 ? brushInputNeurons : brushNeurons[neuronPenBrushIndex];
                    var neuron = layer.Neurons[iNeuron];
                    var x = widthBorder + iLayer * neuronXInc;
                    var y = heightBorder + iNeuron * neuronYInc;
                    if (layer.Neurons.Count == 1)
                        y = p_Height / 2;
                    graphics.DrawEllipse(penNeuron, x - neuronSize / 2, y - neuronSize / 2, neuronSize, neuronSize);
                    graphics.DrawString(neuron.Id, fontNeuron, brushNeuron, x - neuronSize / 2, y - neuronSize / 3);
                    if (iLayer == 0)
                    {
                        graphics.DrawString($"In: {p_Data[iNeuron]:f3}", fontNeuronBias, brushNeuron, x - neuronSize / 2,
                            y + neuronSize / 2);
                    }
                    else
                    {
                        graphics.DrawString($"Bias {neuron.Bias:f3}", fontNeuronBias, brushNeuron, x - neuronSize / 2,
                            y + neuronSize / 2);
                        graphics.DrawString(
                            $"Out:{neuron.LastCalculatedOutput:f3}{Environment.NewLine}OutSig: {neuron.LastCalculatedOutputSigmoid:f3}",
                            fontNeuronBias, brushNeuron, x - neuronSize / 2, y + neuronSize);
                        if (iLayer == layers.Count - 1 && layer.Neurons.Count == 1)
                        {
                            graphics.DrawString($"Expected: {p_ExpectedResult:f3}", fontNeuronBias, brushNeuron,
                                x - neuronSize / 2, y - neuronSize / 2);
                            if (p_PercentMiss.HasValue)
                            {
                                graphics.DrawString(
                                    $"{Environment.NewLine}Miss {p_PercentMissAll:f3}% ({p_PercentMiss:f3}%)",
                                    fontNeuronBias, brushNeuron,
                                    x - neuronSize / 2, y + neuronSize / 2);
                            }
                        }
                    }

                    for (int iPrevNeuron = 0; iPrevNeuron < neuron.SynapsesToPreviousLayer.Count; iPrevNeuron++)
                    {
                        var prevX = widthBorder + (iLayer-1) * neuronXInc;
                        var prevNeuronYInc = neuron.SynapsesToPreviousLayer.Count > 1 ? (p_Height - 2 * heightBorder) / (neuron.SynapsesToPreviousLayer.Count - 1) : 0;
                        var prevY = heightBorder + iPrevNeuron * prevNeuronYInc;
                        if (iLayer > 0)
                        {
                            graphics.DrawLine(penNeuron, prevX, prevY, x, y);
                            graphics.DrawString($"Weight {neuron.SynapsesToPreviousLayer[iPrevNeuron].Weight:f3}",
                                fontNeuronBias, brushNeuron, prevX + (x - prevX) / 3 * 2, prevY + (y - prevY) / 3 * 2);
                        }
                    }

                    if (iLayer > 0)
                    {
                        neuronPenBrushIndex++;
                        if (neuronPenBrushIndex >= brushNeurons.Count)
                            neuronPenBrushIndex = 0;
                    }
                }
            }

            return bitmap;
        }
    }
}


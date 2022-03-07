using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkSample3_Layers
{
    internal class LearnProgressToImage
    {
        public double PercentMissAllStart { get; set; }
        public List<double> PercentMissAll { get; set; } = new();
        public List<List<double>> PercentMissDataIndex { get; set; } = new();

        public Bitmap CreateImage(int p_Width, int p_Height)
        {
            const int widthBorder = 30;
            const int heightBorder = 30;

            //var width = PercentMissAll.Count + 1 + 2*widthBorder;
            var width = p_Width;

            Bitmap bitmap = new Bitmap(width, p_Height);
            using Graphics graphics = Graphics.FromImage(bitmap);
            graphics.TextRenderingHint = TextRenderingHint.AntiAlias;
            graphics.SmoothingMode = SmoothingMode.AntiAlias;

            using Pen penBlack = new Pen(Color.Black);
            using Brush brushRed = new SolidBrush(Color.Red);
            using Pen penRed = new Pen(brushRed);
            using Font font = new Font(FontFamily.GenericMonospace, 16, FontStyle.Regular, GraphicsUnit.Pixel);

            List<Brush> brushDataIndex = new List<Brush>()
            {
                new SolidBrush(Color.DarkGoldenrod), new SolidBrush(Color.Green), new SolidBrush(Color.Blue), new SolidBrush(Color.Coral)
            };
            List<Pen> penDataIndex = brushDataIndex.Select(i => new Pen(i)).ToList();

            graphics.DrawRectangle(penBlack, 0, 0, width-1, p_Height-1);
            graphics.DrawLine(penBlack, widthBorder, p_Height-heightBorder, width - widthBorder, p_Height-heightBorder);
            graphics.DrawLine(penBlack, widthBorder, p_Height-heightBorder, widthBorder, heightBorder);

            List<Point> points;
            double x, y;
            for (var iData = 0; iData < PercentMissDataIndex.Count; iData++)
            {
                var data = PercentMissDataIndex[iData];
                points = new List<Point>(data.Count);
                for (int i = 0; i < data.Count; i++)
                {
                    x = (width - 2 * widthBorder) / (double)data.Count * i + widthBorder;
                    y = (100 - data[i]) / 100 * (p_Height - 2 * heightBorder) + heightBorder;
                    points.Add(new Point((int)x, (int)y));
                }
                graphics.DrawLines(penDataIndex[iData], points.ToArray());
                string s = "";
                for (int i = 0; i < iData; i++)
                    s += Environment.NewLine;
                s += $"DataIndex{iData}";
                graphics.DrawString(s, font, brushDataIndex[iData], widthBorder, heightBorder);
            }

            //last paint PercentMissAll
            points = new List<Point>(PercentMissAll.Count + 1);
            x = widthBorder;
            y = (100 - PercentMissAllStart) / 100 * (p_Height - 2 * heightBorder) + heightBorder;
            points.Add(new Point((int)x, (int)y));
            for (var i = 0; i < PercentMissAll.Count; i++)
            {
                x = (width - 2 * widthBorder) / (double)PercentMissAll.Count * (i + 1) + widthBorder;
                y = (100 - PercentMissAll[i]) / 100 * (p_Height - 2 * heightBorder) + heightBorder;
                points.Add(new Point((int)x, (int)y));
            }
            graphics.DrawLines(penRed, points.ToArray());
            string str = string.Empty;
            for (int i = 0; i < PercentMissDataIndex.Count-1; i++)
                str += Environment.NewLine;
            str += "All";
            graphics.DrawString(str, font, brushRed, widthBorder, heightBorder * 2);

            return bitmap;
        }
    }
}

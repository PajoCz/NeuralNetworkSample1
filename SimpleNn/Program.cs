namespace SimpleNn
{
    class Program
    {
        /// <summary>
        /// engine by https://github.com/kipgparker/BackPropNetwork/blob/master/BackpropNeuralNetwork/Assets/Manager.cs
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            NeuralNetwork net;
            int[] layers = { 2, 2, 1 };
            string[] activation = { "sigmoid", "sigmoid" };

            //int[] layers = { 2, 4, 8, 1 };
            //string[] activation = { "sigmoid", "sigmoid", "sigmoid"};


            net = new NeuralNetwork(layers, activation
                //, 
                //new float[] {0.9888773586480377f, 0.32188634502570845f, -1.1927510125913223f}, 
                //new float[] {0.8815793758627867f, -0.5202642691344876f, -0.0037441705087075737f, 0.2667151772486819f,-0.038516025100668934f, 1.0484903515494195f}
                );
            for (int i = 0; i < 10000; i++)
            {
                //Regression test
                //net.BackPropagate(new float[] { -2, -1 }, new float[] { 1 });
                //net.BackPropagate(new float[] { 25, 6 }, new float[] { 0 });
                //net.BackPropagate(new float[] { 17, 4 }, new float[] { 0 });
                //net.BackPropagate(new float[] { -15, -6 }, new float[] { 1 });
                
                //XOR test
                net.BackPropagate(new float[] { 0, 0 }, new float[] { 1 });
                net.BackPropagate(new float[] { 0, 1 }, new float[] { 0 });
                net.BackPropagate(new float[] { 1, 0 }, new float[] { 0 });
                net.BackPropagate(new float[] { 1, 1 }, new float[] { 1 });
            }
            Console.WriteLine("cost: "+net.cost);
        
            //Regression test
            //Console.WriteLine(net.FeedForward(new float[] { -2, -1 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { 25, 6 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { 17, 4 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { -15, -6 })[0]);

            //XOR test
            Console.WriteLine(net.FeedForward(new float[] { 0, 0 })[0]);
            Console.WriteLine(net.FeedForward(new float[] { 0, 1 })[0]);
            Console.WriteLine(net.FeedForward(new float[] { 1, 0 })[0]);
            Console.WriteLine(net.FeedForward(new float[] { 1, 1 })[0]);
        }
    }
}


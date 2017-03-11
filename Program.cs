using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN_translated
{
    class Program
    {
        static void Main(string[] args)
        {

            //3 Input - 1 Output
            double[,] training_set_inputs = new double[,] { { 0, 0, 1},
                                                            { 1, 1, 1},
                                                            { 1, 0, 1},
                                                            { 0, 1, 1}};

            double[,] training_set_outputs = new double[,] { {0},
                                                             {1},
                                                             {1},
                                                             {0}};

            NeuralNetwork neuralNetwork = new NeuralNetwork();


            Console.WriteLine("Start training ...");

            // Train the neural network using a training set.
            // Do it 10,000 times
            neuralNetwork.Train(training_set_inputs, training_set_outputs, 10000);
            Console.WriteLine("End training ...\n\n");

            // Predict
            Console.WriteLine("Considering new situation [1, 0, 0] -> ?\n");
            var result = neuralNetwork.Think(new double[,] { { 1, 0, 0 } });

            Console.WriteLine(result);
            Console.ReadKey();
        }
    }
}

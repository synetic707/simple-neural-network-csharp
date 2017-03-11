using System;
using MathNet.Numerics.LinearAlgebra;


namespace SimpleNN_translated
{
    public class NeuralNetwork
    {
         Matrix<double> synapticWeights;

        public NeuralNetwork()
        {
            synapticWeights = createMatrixWithRandomNumbers(3, 1);
        }

        private Matrix<double> sigmoid(Matrix<double> x, bool derivative)
        {
            if (derivative)
            {
                return x.PointwiseMultiply(1 - x);
            }
            else
            {
                return 1 / (1 + (1 / x.PointwiseExp()));
            }
        }

        public void Train(double[,] trainingInput, double[,] trainingOutput, int trainingIterations)
        {
            var mtrainingInput = buildMatrixFromArray(trainingInput);
            var mtrainingOutput = buildMatrixFromArray(trainingOutput);
             
            for (int i = 0; i < trainingIterations; i++)
            {
                var output = Think(trainingInput);
                var error = mtrainingOutput - output;
                var adjustment = dotProduct(mtrainingInput.Transpose(), error.PointwiseMultiply(sigmoid(output, true)));
                synapticWeights += adjustment;
            }
        }

        public Matrix<double> Think(double[,] inputs)
        {
            var minputs = buildMatrixFromArray(inputs);

            return sigmoid(dotProduct(minputs, synapticWeights), false);
        }

        private Matrix<double> dotProduct(Matrix<double> matrixOne, Matrix<double> matrixTwo)
        {
            return matrixOne * matrixTwo;
        }

        private Matrix<double> createMatrixWithRandomNumbers(int rowCount, int columnCount)
        {
            Random rand = new Random();
            double[,] matrix = new double[rowCount, columnCount];

            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    double maxValue = 1.0;
                    double minValue = -1.0;

                    double betweenMinusOneToOne = rand.NextDouble() * (maxValue - minValue) + minValue;
                    matrix[i, j] = betweenMinusOneToOne;
                }
            }

            return Matrix<double>.Build.DenseOfArray(matrix);
        }

        private Matrix<double> buildMatrixFromArray(double[,] array)
        {
            return Matrix<double>.Build.DenseOfArray(array);
        }
    }
}

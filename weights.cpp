#include <iostream>
#include "get_window.h"
#include <Eigen/Dense>
#include "File.h"
#include<string>

using namespace std;
extern string filepath;// = "weights/3/";

#include <unsupported/Eigen/FFT>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::Matrix;

MatrixXf get_input2layer1()
{
    float input_to_layer1[1020 * 256];
	
    Read_File(input_to_layer1, 1020 * 256, filepath+"input_to_layer1.txt");

//    Matrix<float,Dynamic,Dynamic,RowMajor> M2(mfcc);
//    Map<MatrixXf,ColMajor> model_input(M2.data(), 1,1020);

    Map<MatrixXf> input2layer1(input_to_layer1, 256, 1020);
    return input2layer1.transpose();
}
MatrixXf get_layer1bias()
{
    float bias_layer1[256 * 1];

    Read_File(bias_layer1, 256 * 1, filepath+"bias_layer1.txt");

    Map<MatrixXf> BiasLayer1(bias_layer1, 1, 256);
    return BiasLayer1.transpose();
}
MatrixXf get_layer12layer2()
{
    float layer1_to_layer2[256 * 256];
    Read_File(layer1_to_layer2, 256 * 256, filepath + "layer1_to_layer2.txt");

    Map<MatrixXf> Layer12Layer2(layer1_to_layer2, 256, 256);
    return Layer12Layer2.transpose();
}
MatrixXf get_layer2bias()
{
    float bias_layer2[256 * 1];
    Read_File(bias_layer2, 256 * 1, filepath + "bias_layer2.txt");

    Map<MatrixXf> BiasLayer2(bias_layer2, 1, 256);
    return BiasLayer2.transpose();
}
MatrixXf get_layer22layer3()
{
    float layer2_to_layer3[256 * 256];
    Read_File(layer2_to_layer3, 256 * 256, filepath + "layer2_to_layer3.txt");

    Map<MatrixXf> Layer22Layer3(layer2_to_layer3, 256, 256);
    return Layer22Layer3.transpose();
}
MatrixXf get_layer3bias()
{
    float bias_layer3[256 * 1];
    Read_File(bias_layer3, 256 * 1, filepath + "bias_layer3.txt");

    Map<MatrixXf> BiasLayer3(bias_layer3, 1, 256);
    return BiasLayer3.transpose();
}
MatrixXf get_layer32output()
{
    float layer3_to_output[256 * 12];
    Read_File(layer3_to_output, 256 * 12, filepath + "layer3_to_output.txt");

    Map<MatrixXf> layer32output(layer3_to_output, 12, 256);
    return layer32output.transpose();
}
MatrixXf get_outputbias()
{
    MatrixXf BiasOutput(1,12);
	Read_File(BiasOutput, filepath + "bias_output.txt");
	//BiasOutput <<
	//	-6.707094609737396240e-02,
	//	-5.290204659104347229e-02;

	return BiasOutput;

}

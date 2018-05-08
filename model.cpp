#include "model.h"
#include "weights.h"

#include <iostream>
#include <Eigen/Dense>


using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::Matrix;
using namespace std;



model::model()
{
	weights.input_to_layer1 = get_input2layer1();
	weights.bias_layer1 = get_layer1bias();
	weights.layer1_to_layer2 = get_layer12layer2();
	weights.bias_layer2 = get_layer2bias();
    weights.layer2_to_layer3 = get_layer22layer3();
    weights.bias_layer3 = get_layer3bias();

    weights.layer3_to_output = get_layer32output();
	weights.bias_output = get_outputbias();
}


model::~model()
{
}

MatrixXf model::predict(MatrixXf feature_data)
{
	int n_frames = feature_data.rows();

	//replicate bias to do batch procession,
	MatrixXf bias = weights.bias_layer1.replicate(n_frames, 1);
    MatrixXf layer1 = weights.input_to_layer1;
    MatrixXf input1 = feature_data*weights.input_to_layer1;
    MatrixXf layer1input = feature_data*weights.input_to_layer1 + bias.transpose();

	MatrixXf layer1output = ReLU(layer1input);

	bias = weights.bias_layer2.replicate(n_frames, 1);
    MatrixXf layer2input = layer1output*weights.layer1_to_layer2 + bias.transpose();

	MatrixXf layer2output = ReLU(layer2input);

    bias = weights.bias_layer3.replicate(n_frames, 1);
    cout<<weights.layer2_to_layer3.rows()<<"*"<<weights.layer2_to_layer3.cols()<<endl;
    MatrixXf layer3input = layer2output*weights.layer2_to_layer3 + bias.transpose();

    MatrixXf layer3output = ReLU(layer3input);

    bias = weights.bias_output.replicate(n_frames, 1);
    cout<<weights.layer3_to_output.rows()<<"*"<<weights.layer3_to_output.cols()<<endl;
    MatrixXf denseinput = layer3output*weights.layer3_to_output + bias;

    MatrixXf output = softmax(denseinput);

	return output.transpose();
}

MatrixXf model::ReLU(MatrixXf input)
{
	MatrixXf output(input.rows(), input.cols());
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			output(i, j) = std::max((float)0, input(i, j));
		}
	}
	return output;
}
/*
  MLP sigmoid output
              1
  f(x) = -----------
         1 + exp(-x)
*/
MatrixXf model::sigmoid(MatrixXf input)
{
	MatrixXf output(2, 1);
	output = (1 / ((input*-1).array().exp() + 1)).matrix();
    return output;
}

MatrixXf model::softmax(MatrixXf a)
{
//    int label_count = input.cols();
//    MatrixXf output = (input.array().exp()/input.array().exp().sum()).matrix();
    MatrixXf c = (a.array().exp()/a.array().exp().sum()).matrix();
    return c;

}

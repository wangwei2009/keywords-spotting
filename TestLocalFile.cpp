#include <iostream>
#include <Eigen/Dense>
#include "wavread.h"
#include "get_window.h"
#include "get_mel_basis.h"
#include "get_dct_basis.h"
#include "feature_normalizer.h"
#include "feature_aggregator.h"
#include "parameters.h"
#include "weights.h"
#include "model.h"
#include "preprocessing.h"
#include "postprocessing.h"
#include <unsupported/Eigen/FFT>

#include<time.h>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::Array;
using Eigen::Matrix;

using namespace std;

int TestLocalFile()
{
	clock_t start_time = clock();

    int frame_length = 16000*0.04;
    int hop_length = 16000 * 0.02;

	wav_struct wav;
	cout << sizeof(wav.channel) << endl;
	cout << sizeof(wav.frequency) << endl;

    char *filename = "stop.wav";

	int pos = GetWavArgu(filename, &wav);
	wav.wavdata = new double[wav.num_per_channel];
	wavread(filename, &wav, pos);

    MatrixXf y(wav.num_per_channel + n_fft, 1);
	// Pad the time series so that frames are centered
    preprocessing preprocessor;
    y = preprocessor.ParseWavdata(wav.data,wav.num_per_channel,wav.channel);


    y = preprocessor.pad(y);
    //MatrixXf y0 = preprocessor.pad(wav);
    //y = preprocessor.pad(wav);


	// Compute the number of frames that will fit.The end may get truncated.
    int	n_frames = 1 + int((y.size() - n_fft) / hop_length);
	cout << "n_frames = " << n_frames << endl;

    MatrixXf y_frames(n_fft, n_frames);
	for (int i = 0; i < n_frames; i++)
	{
        y_frames.col(i) = y.block(i*hop_length, 0, n_fft, 1);
	}
    //cout << "y_frames.size()=" << y_frames.block(296, 243, 1, 6) << endl;

	// Pre - allocate the STFT matrix
    MatrixXf stft_matrix(n_fft, n_frames);
    Matrix < complex<float>, n_fft, 1> stft_matrix_complex;
	MatrixXf fft_window = get_window();
	FFT<float> fft;

    Eigen::Matrix<float, n_fft, 1> frame_windowed;
	for (int i = 0; i < n_frames; i++)
	{
		frame_windowed = (y_frames.col(i).array()*fft_window.array()).matrix();

		fft.fwd(stft_matrix_complex, frame_windowed);
		//cout << stft_matrix_complex(0, 0) << endl;

		//stft_matrix.col(i) = stft_matrix_complex.array().abs().matrix();

		stft_matrix.col(i) = (stft_matrix_complex.real().array()*stft_matrix_complex.real().array() +
			stft_matrix_complex.imag().array()*stft_matrix_complex.imag().array()).sqrt();
	}
	cout << "stft_matrix =" << stft_matrix.block(158, 10, 1, 5) << endl;
    MatrixXf mel_basis(40, n_fft/2+1);
	mel_basis = get_mel_basis();

    //MatrixXf mel_spectrum(n_frames, 40);
    MatrixXf mel_spectrum = (mel_basis*stft_matrix.block(0, 0, n_fft / 2 + 1, stft_matrix.cols()));
    //cout << mel_spectrum.rows() << "X" << mel_spectrum.cols() << endl;

    MatrixXf logmel_spectrum = 10*mel_spectrum.array().log10().matrix();

    MatrixXf dct_basis = get_dct_basis();


    //MatrixXf feature_data = Normalize(mel_spectrum);

    //cout << feature_data.block(210, 23, 1, 10) << endl;

    MatrixXf mfcc = dct_basis*logmel_spectrum;

    Matrix<float,Dynamic,Dynamic,RowMajor> M2(mfcc);
    Map<MatrixXf,ColMajor> model_input(M2.data(), 1,1020);

	////cout<< mel_spectrum.block(25,240,1,10)<<endl;
	//cout << "aggregated_frame.block(250, 180, 1, 10) = "<<aggregated_frame.block(250, 180, 1, 10) << endl;

	model AED;
    MatrixXf output_prob = AED.predict(model_input);

    cout<<output_prob<<endl;

    MatrixXf::Index max_index;
    output_prob.col(0).maxCoeff(&max_index);
    cout<<max_index<<endl;
//    string dnn_labe[] = {"_silence_","_unknown_","yes","no","stop"};
    string dnn_labe[] = {"_silence_","_unknown_","yes","no","up","down","left","right","on","off","stop","go"};
    cout<<"you speak:"<<dnn_labe[max_index]<<endl;



	cout << "Hello World!" << endl;
	delete[] wav.wavdata;

	clock_t end_time = clock();
	cout << "time cost:" << 1.0*(end_time - start_time) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
	return 0;
}
int Test_LongLocalFile()
{
    clock_t start_time = clock();

    int frame_length = 16000*0.04;
    int hop_length = 16000 * 0.02;

    wav_struct wav;
    cout << sizeof(wav.channel) << endl;
    cout << sizeof(wav.frequency) << endl;

    char *filename = "stop_ipc2.wav";

    int pos = GetWavArgu(filename, &wav);
    wav.wavdata = new double[wav.num_per_channel];
    wavread(filename, &wav, pos);

    MatrixXf y(16000 + n_fft, 1);
    // Pad the time series so that frames are centered
    preprocessing preprocessor;
    MatrixXf audio_data = preprocessor.ParseWavdata(wav.data,wav.num_per_channel,wav.channel);

    // Compute the number of frames that will fit.The end may get truncated.
    int	n_frames = 1 + int((y.size()- n_fft) / hop_length);
    cout << "n_frames = " << n_frames << endl;
    MatrixXf y_frames(n_fft, n_frames);

    // Pre - allocate the STFT matrix
    MatrixXf stft_matrix(n_fft, n_frames);
    Matrix < complex<float>, n_fft, 1> stft_matrix_complex;
    MatrixXf fft_window = get_window();

    FFT<float> fft;

    Eigen::Matrix<float, n_fft, 1> frame_windowed;

    MatrixXf mel_basis(40, n_fft/2+1);
    mel_basis = get_mel_basis();

    MatrixXf mel_spectrum;

    MatrixXf logmel_spectrum;

    MatrixXf dct_basis;

    MatrixXf mfcc;

    model AED;

    MatrixXf::Index max_index;
    MatrixXf output_prob;

    string dnn_labe[] = {"_silence_","_unknown_","yes","no","up","down","left","right","on","off","stop","go"};

    for(int t = 0;t<wav.num_per_channel-16000;t=t+8000)
    {
        y = audio_data.block(0,t,1,16000);
        y = preprocessor.pad(y);
        //MatrixXf y0 = preprocessor.pad(wav);
        //y = preprocessor.pad(wav);

        for (int i = 0; i < n_frames; i++)
        {
            y_frames.col(i) = y.block(i*hop_length, 0, n_fft, 1);
        }
        //cout << "y_frames.size()=" << y_frames.block(296, 243, 1, 6) << endl;


        for (int i = 0; i < n_frames; i++)
        {
            frame_windowed = (y_frames.col(i).array()*fft_window.array()).matrix();

            fft.fwd(stft_matrix_complex, frame_windowed);
            //cout << stft_matrix_complex(0, 0) << endl;

            //stft_matrix.col(i) = stft_matrix_complex.array().abs().matrix();

            stft_matrix.col(i) = (stft_matrix_complex.real().array()*stft_matrix_complex.real().array() +
                stft_matrix_complex.imag().array()*stft_matrix_complex.imag().array()).sqrt();
        }
        //cout << "stft_matrix =" << stft_matrix.block(158, 10, 1, 5) << endl;


        //MatrixXf mel_spectrum(n_frames, 40);
        mel_spectrum = (mel_basis*stft_matrix.block(0, 0, n_fft / 2 + 1, stft_matrix.cols()));
        //cout << mel_spectrum.rows() << "X" << mel_spectrum.cols() << endl;

        logmel_spectrum = 10*mel_spectrum.array().log10().matrix();

        dct_basis = get_dct_basis();


        //MatrixXf feature_data = Normalize(mel_spectrum);

        //cout << feature_data.block(210, 23, 1, 10) << endl;

        mfcc = dct_basis*logmel_spectrum;

        Matrix<float,Dynamic,Dynamic,RowMajor> M2(mfcc);
        Map<MatrixXf,ColMajor> model_input(M2.data(), 1,1020);

        ////cout<< mel_spectrum.block(25,240,1,10)<<endl;
        //cout << "aggregated_frame.block(250, 180, 1, 10) = "<<aggregated_frame.block(250, 180, 1, 10) << endl;


        output_prob = AED.predict(model_input);

        cout<<output_prob<<endl;


        output_prob.col(0).maxCoeff(&max_index);
        cout<<max_index<<endl;
        cout<<"you speak:"<<dnn_labe[max_index]<<endl;
    }






    cout << "Hello World!" << endl;
    delete[] wav.wavdata;

    clock_t end_time = clock();
    cout << "time cost:" << 1.0*(end_time - start_time) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    return 0;
}

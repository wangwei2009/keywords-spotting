#include "EventDetector.h"
#include "get_mel_basis.h"
#include "get_dct_basis.h"
#include "feature_normalizer.h"
#include "feature_aggregator.h"
#include "parameters.h"

EventDetector::EventDetector()
{
	mel_basis = get_mel_basis();
	fft_window = get_window();
}


EventDetector::~EventDetector()
{
}

MatrixXf EventDetector::detect(char * data, unsigned int frames, unsigned int channel)
{
    int frame_length = n_fft;
	int hop_length = 44100 * 0.02;

	MatrixXf y(frames + N_FFT, 1);
	// Pad the time series so that frames are centered
	y = preprocessor.ParseWavdata(data, frames, channel);
	y = preprocessor.pad(y);

	// Compute the number of frames that will fit.The end may get truncated.
	int	n_frames = 1 + int((y.size() - frame_length) / hop_length);
	// enframe
	MatrixXf y_frames(frame_length, n_frames);
	for (int i = 0; i < n_frames; i++)
	{
        y_frames.col(i) = y.block(i*hop_length, 0, n_fft, 1);
	}

	// Pre - allocate the STFT matrix
	//Matrix<float, N_FFT, 1>stft_matrix;
	MatrixXf stft_matrix(N_FFT, n_frames);
	Matrix < complex<float>, N_FFT, 1> stft_matrix_complex;
	//MatrixXf fft_window = get_window();
	FFT<float> fft;

	for (int i = 0; i < n_frames; i++)
	{
		frame_windowed = (y_frames.col(i).array()*fft_window.array()).matrix();

		fft.fwd(stft_matrix_complex, frame_windowed);
		//cout << stft_matrix_complex(0, 0) << endl;

		//stft_matrix.col(i) = stft_matrix_complex.array().abs().matrix();

		stft_matrix.col(i) = (stft_matrix_complex.real().array()*stft_matrix_complex.real().array() +
			stft_matrix_complex.imag().array()*stft_matrix_complex.imag().array()).sqrt();
	}
	MatrixXf mel_spectrum(n_frames, 40);
    mel_spectrum = (mel_basis*stft_matrix.block(0, 0, n_fft / 2 + 1, stft_matrix.cols())).array().log().matrix().transpose();

	MatrixXf feature_data = Normalize(mel_spectrum);
	MatrixXf aggregated_frame = Aggregate(feature_data);

	MatrixXf output_prob = AED.predict(aggregated_frame);
	output_prob = postprocessor.frame_binarization(output_prob);
	output_prob = postprocessor.frame_medfilt(output_prob);
	MatrixXf event_segments = postprocessor.find_contiguous_regions(output_prob);
	cout << event_segments << endl;
    return event_segments;
}
MatrixXf EventDetector::detect_enh(char * data, unsigned int frames, unsigned int channel)
{
    clock_t start_time = clock();

    int frame_length = 16000*0.04;
    int hop_length = 16000 * 0.02;

    MatrixXf y(16000 + n_fft, 1);

    y = preprocessor.ParseWavdata(data, frames, channel);

    cout<<"audio:y.size() = "<<y.size()<<endl;


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
//    cout << "stft_matrix =" << stft_matrix.block(158, 10, 1, 5) << endl;
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

    //model AED;
    MatrixXf output_prob = AED.predict(model_input);

    cout<<"output_prob="<<endl<<output_prob<<endl;


    clock_t end_time = clock();
    cout << "time cost:" << 1.0*(end_time - start_time) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    return output_prob;
}

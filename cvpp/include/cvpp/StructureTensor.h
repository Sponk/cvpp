#ifndef __STRUCTURE_TENSOR_H__
#define __STRUCTURE_TENSOR_H__

#include "Image.h"
#include "CommonFilters.h"
#include "Convolution.h"

namespace cvpp
{

template<typename T>
CPUImage<Eigen::Matrix2f> StructureTensor(const CPUImage<T>& in)
{
	auto gray = MakeGrayscale(ConvertType<T, float>(in));
	auto sampler = ClampView(gray);

	auto Dx = Convolute2D(sampler, ScharrFilterH());
	auto Dy = Convolute2D(sampler, ScharrFilterV());

	// ConvertType<float, unsigned char>(Dx).save("DX.png");
	// ConvertType<float, unsigned char>(Dy).save("DY.png");

	auto Sx = Dx*Dx;
	auto Sy = Dy*Dy;
	auto Sxy = Dx*Dy;

	Sx = ConvoluteSeparable(ClampView(Sx), GaussFilter<3>(1.0f));
	Sy = ConvoluteSeparable(ClampView(Sy), GaussFilter<3>(1.0f));
	Sxy = ConvoluteSeparable(ClampView(Sxy), GaussFilter<3>(1.0f));

	CPUImage<Eigen::Matrix2f> output(in.getWidth(), in.getHeight(), 1);
	#pragma omp parallel for
	for(int y = 0; y < in.getHeight(); y++)
	{
		const size_t yoff = y*in.getWidth();
		for(int x = 0; x < in.getWidth(); x++)
		{
			const size_t off = yoff+x;
			output[off] <<	ColorToFloat(Sx[off]), ColorToFloat(Sxy[off]),
							ColorToFloat(Sxy[off]), ColorToFloat(Sy[off]);
		}
	}

	return output;
}


template<typename T>
CPUImage<Eigen::Matrix2f> HessianTensor(const CPUImage<T>& in)
{
	auto gray = MakeGrayscale(ConvertType<T, float>(in));
	auto sampler = ClampView(gray);

	gray = ConvoluteSeparable(sampler, GaussFilter<3>(0.25f));

	auto Dx = Convolute1D<HORIZONTAL>(sampler, LaplaceFilterX);
	auto Dy = Convolute1D<VERTICAL>(sampler, LaplaceFilterX);
	auto Dxy = Convolute2D(sampler, LaplaceFilterXY());
	auto Dyx = Convolute2D(sampler, LaplaceFilterXY().transpose());

	CPUImage<Eigen::Matrix2f> output(in.getWidth(), in.getHeight(), 1);
	#pragma omp parallel for
	for(int y = 0; y < in.getHeight(); y++)
	{
		const size_t yoff = y*in.getWidth();
		for(int x = 0; x < in.getWidth(); x++)
		{
			const size_t off = yoff+x;
			output[off] <<	ColorToFloat(Dx[off]), ColorToFloat(Dxy[off]),
							ColorToFloat(Dyx[off]), ColorToFloat(Dy[off]);
		}
	}

	return output;
}

}

#endif

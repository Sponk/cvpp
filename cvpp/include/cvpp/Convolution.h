#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include "Image.h"
#include "Sampler.h"
#include <eigen3/Eigen/StdVector>

namespace cvpp
{

template<typename T, typename K>
CPUImage<T> Convolute2D(const SamplerView<T>& sampler, const K& kernel, unsigned int size)
{
	const CPUImage<T>& in = *sampler.getImage();
	CPUImage<T> out(in.getWidth(), in.getHeight(), in.getComponents());
	const int halfSize = size/2;

#pragma omp parallel for
	for(int y = 0; y < in.getHeight(); y++)
	{
		const size_t yoff = y*in.getWidth();
		for(int x = 0; x < in.getWidth(); x++)
		{
			const size_t xoff = (yoff+x)*in.getComponents();
			Eigen::Vector4f sum(0, 0, 0, 0);
			for(int kx = -halfSize; kx <= halfSize; kx++)
			{
				for(int ky = -halfSize; ky <= halfSize; ky++)
				{
					sum += kernel(ky + halfSize, kx + halfSize) * sampler.sample(x + kx, y + ky);
				}
			}

			auto* outPtr = &out.getData()[xoff];
			for(int c = 0; c < in.getComponents(); c++)
			{
				outPtr[c] = FloatToColor<T>(sum[c]);
			}
		}
	}

	return out;
}

template<int Stride = 1, typename T, typename Fn, typename Finisher>
CPUImage<T> NonLinearConv2D(const SamplerView<T>& sampler, unsigned int size, Fn fn, Finisher fin)
{
	const CPUImage<T>& in = *sampler.getImage();
	CPUImage<T> out(in.getWidth(), in.getHeight(), in.getComponents());
	const int halfSize = size/2;
	const int stride = (Stride == -1 ? size : Stride);

#pragma omp parallel for
	for(int y = 0; y < in.getHeight(); y += stride)
	{
		const size_t yoff = y*in.getWidth();
		for(int x = 0; x < in.getWidth(); x += stride)
		{
			const size_t xoff = (yoff+x)*in.getComponents();
			Eigen::Vector4f sum(0, 0, 0, 0);
			for(int kx = -halfSize; kx <= halfSize; kx++)
			{
				for(int ky = -halfSize; ky <= halfSize; ky++)
				{
					fn(kx, ky, sampler.sample(x + kx, y + ky), sum);
				}
			}

			fin(uint32_t(x), uint32_t(y), sum);

			auto* outPtr = &out.getData()[xoff];
			for(int c = 0; c < in.getComponents(); c++)
			{
				outPtr[c] = FloatToColor<T>(sum[c]);
			}
		}
	}

	return out;
}

template<int Stride = 1, typename T, typename Fn>
CPUImage<T> NonLinearConv2D(const SamplerView<T>& sampler, unsigned int size, Fn fn)
{
	return NonLinearConv2D<Stride>(sampler, size, fn, [](auto, auto, auto){});
}

template<typename T, int Size>
CPUImage<T> Convolute2D(const SamplerView<T>& sampler, const Eigen::Matrix<float, Size, Size>& kernel)
{
	static_assert(Size % 2 != 0, "A kernel needs an odd size!");
	return Convolute2D(sampler, kernel, Size);
}

template<typename T>
CPUImage<T> Convolute2D(const SamplerView<T>& sampler, const Eigen::MatrixXf& kernel)
{
	assert(kernel.rows() == kernel.cols() && "Wrong size of kernel!");
	return Convolute2D(sampler, kernel, kernel.rows());
}

enum CONVOLUTION_TYPE
{
	HORIZONTAL,
	VERTICAL
};

template<CONVOLUTION_TYPE Dir = HORIZONTAL, typename T, typename K>
CPUImage<T> Convolute1D(const SamplerView<T>& sampler, const K& kernel, unsigned int size)
{
	const CPUImage<T>& in = *sampler.getImage();
	CPUImage<T> out(in.getWidth(), in.getHeight(), in.getComponents());
	const int halfSize = size/2;

#pragma omp parallel for
	for(int y = 0; y < in.getHeight(); y++)
	{
		const size_t yoff = y*in.getWidth();
		for(int x = 0; x < in.getWidth(); x++)
		{
			const size_t xoff = (yoff+x)*in.getComponents();
			Eigen::Vector4f sum(0, 0, 0, 0);
			for(int k = -halfSize; k <= halfSize; k++)
			{
				if constexpr(Dir == HORIZONTAL)
				{
					sum += kernel[k + halfSize] * sampler.sample(x + k, y);
				}
				else
				{
					sum += kernel[k + halfSize] * sampler.sample(x, y + k);
				}
			}

			auto* outPtr = &out.getData()[xoff];
			for(int c = 0; c < in.getComponents(); c++)
			{
				outPtr[c] = FloatToColor<T>(sum[c]);
			}
		}
	}

	return out;
}

template<CONVOLUTION_TYPE Dir = HORIZONTAL, typename T, int Rows, int Cols>
CPUImage<T> Convolute1D(const SamplerView<T>& sampler, const Eigen::Matrix<float, Rows, Cols>& kernel)
{
	return Convolute1D<Dir>(sampler, kernel, Rows);
}

template<CONVOLUTION_TYPE Dir = HORIZONTAL, typename T>
CPUImage<T> Convolute1D(const SamplerView<T>& sampler, const Eigen::VectorXf& kernel)
{
	return Convolute1D<Dir>(sampler, kernel, kernel.rows());
}

template<typename T, typename K>
auto ConvoluteSeparable(const T& sampler, const K& kernel, unsigned int size)
{
	auto r1 = Convolute1D<HORIZONTAL>(sampler, kernel, size);
	return Convolute1D<VERTICAL>(T(r1), kernel, size);
}

template<typename T, int Rows, int Cols>
auto ConvoluteSeparable(const T& sampler, const Eigen::Matrix<float, Rows, Cols>& kernel)
{
	auto r1 = Convolute1D<HORIZONTAL>(sampler, kernel, Rows);
	return Convolute1D<VERTICAL>(T(r1), kernel, Rows);
}

template<typename T>
auto ConvoluteSeparable(const T& sampler, const Eigen::VectorXf& kernel)
{
	auto r1 = Convolute1D<HORIZONTAL>(sampler, kernel, kernel.rows());
	return Convolute1D<VERTICAL>(T(r1), kernel, kernel.rows());
}

}

#endif

#ifndef __SYCL_CONVOLUTION_H__
#define __SYCL_CONVOLUTION_H__

#include <cvpp/Image.h>
#include "Sampler.h"
#include "Image.h"
#include <eigen3/Eigen/StdVector>

namespace cvsycl
{

template<typename A, typename B, typename C> class convolute2d_kernel;

template<typename Sampler, typename T, typename K>
Image<T> Convolute2D(Image<T>& in, const K& kernel, unsigned int size, cl::sycl::queue& queue)
{
	Image<T> out(in.getWidth(), in.getHeight(), in.getComponents());
	const int halfSize = size/2;

	using namespace cl::sycl;

	queue.submit([&](cl::sycl::handler& cgh) {
		Sampler sampler(cgh, in);
		auto outAcc = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);
	
		cgh.parallel_for<convolute2d_kernel<Sampler, T, K>>(cl::sycl::range<2>(in.getWidth(), in.getHeight()), [=](cl::sycl::id<2> p)
		{
			const size_t xoff = (sampler.getWidth()*p[1] + p[0]) * sampler.getComponents();
			cl::sycl::float4 sum(0, 0, 0, 0);
			for(int kx = -halfSize; kx <= halfSize; kx++)
			{
				for(int ky = -halfSize; ky <= halfSize; ky++)
				{
					const auto s = sampler.sample((int) p[0] + kx, (int) p[1] + ky);
					sum += kernel(ky + halfSize, kx + halfSize) * s;
				}
			}

			auto* outPtr = &outAcc[xoff];
			float* sumPtr = (float*) &sum;
			for(int c = 0; c < sampler.getComponents(); c++)
			{
				outPtr[c] = cvpp::FloatToColor<T>(sumPtr[c]);
			}
		});
	});

	return out;
}

template<typename Sampler, typename T, int W, int H>
Image<T> Convolute2D(Image<T>& sampler, const Eigen::Matrix<float, W, H>& kernel, cl::sycl::queue& queue)
{
	static_assert(W == H, "A kernel needs to be square!");
	static_assert(W % 2 != 0, "A kernel needs an odd size!");
	return Convolute2D<Sampler>(sampler, kernel, W, queue);
}

#if 0
template<typename Sampler, typename T>
Image<T> Convolute2D(Image<T>& sampler, const Eigen::MatrixXf& kernel, cl::sycl::queue& queue)
{
	//static_assert(false, "Variable size matrices are not supported yet!");
	assert(kernel.rows() == kernel.cols() && "Wrong size of kernel!");
	return Convolute2D<Sampler>(sampler, kernel, kernel.rows(), queue);
}
#endif

template<typename A, typename B, typename C, typename D, int S> class non_linear_convolute2d_kernel;

template<typename Sampler, int Stride = 1, typename T, typename Fn, typename Finisher>
Image<T> NonLinearConv2D(Image<T>& in, unsigned int size, Fn fn, Finisher fin, cl::sycl::queue& queue)
{
	Image<T> out(in.getWidth(), in.getHeight(), in.getComponents());
	const int halfSize = size/2;
	const int stride = (Stride == -1 ? size : Stride);

	using namespace cl::sycl;

	queue.submit([&](cl::sycl::handler& cgh) {
		Sampler sampler(cgh, in);
		auto outAcc = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);
	
		cgh.parallel_for<non_linear_convolute2d_kernel<Sampler, T, Fn, Finisher, Stride>>(cl::sycl::range<2>(in.getWidth(), in.getHeight()), [=](cl::sycl::id<2> p)
		{
			const size_t xoff = (sampler.getWidth()*p[1] + p[0]) * sampler.getComponents();
			cl::sycl::float4 sum(0, 0, 0, 0);
			for(int kx = -halfSize; kx <= halfSize; kx++)
			{
				for(int ky = -halfSize; ky <= halfSize; ky++)
				{
					const auto s = sampler.sample((int) p[0] + kx, (int) p[1] + ky);
					sum = fn(kx, ky, s, sum);
				}
			}

			sum = fin(uint32_t(p[0]), uint32_t(p[1]), sum);

			auto* outPtr = &outAcc[xoff];
			float* sumPtr = (float*) &sum;
			for(int c = 0; c < sampler.getComponents(); c++)
			{
				outPtr[c] = cvpp::FloatToColor<T>(sumPtr[c]);
			}
		});
	});

	return out;
}

template<typename Sampler, int Stride = 1, typename T, typename Fn>
Image<T> NonLinearConv2D(Image<T>& in, unsigned int size, Fn fn)
{
	return NonLinearConv2D<Sampler, Stride>(in, size, fn, [](auto, auto, auto sum){ return sum; });
}

class HORIZONTAL;
class VERTICAL;

template<typename Sampler, typename Dir, typename T, typename K> class convolute1d_kernel;

template<typename Sampler, typename Dir, typename T, typename K>
Image<T> Convolute1D(Image<T>& in, const K& kernel, unsigned int size, cl::sycl::queue& queue)
{
	Image<T> out(in.getWidth(), in.getHeight(), in.getComponents());
	const int halfSize = size/2;

	using namespace cl::sycl;

	queue.submit([&](cl::sycl::handler& cgh) {
		Sampler sampler(cgh, in);
		auto outAcc = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);

		cgh.parallel_for<convolute1d_kernel<Sampler, Dir, T, K>>(cl::sycl::range<2>(in.getWidth(), in.getHeight()), [halfSize, sampler, outAcc, kernel](cl::sycl::id<2> p)
		{
			const size_t xoff = (sampler.getWidth()*p[1] + p[0]) * sampler.getComponents();
			cl::sycl::float4 sum(0, 0, 0, 0);
			for(int k = -halfSize; k <= halfSize; k++)
			{
				if constexpr(std::is_same_v<Dir, HORIZONTAL>)
				{
					sum += kernel[k + halfSize] * sampler.sample((int) p[0] + k, (int) p[1]);
				}
				else
				{
					sum += kernel[k + halfSize] * sampler.sample((int) p[0], (int) p[1] + k);
				}
			}

			auto* outPtr = &outAcc[xoff];
			float* sumPtr = (float*) &sum;
			for(int c = 0; c < sampler.getComponents(); c++)
			{
				outPtr[c] = cvpp::FloatToColor<T>(sumPtr[c]);
			}
		});
	});

	return out;
}

template<typename Sampler, typename Dir, typename T, int Rows, int Cols>
Image<T> Convolute1D(Image<T>& sampler, const Eigen::Matrix<float, Rows, Cols>& kernel, cl::sycl::queue& queue)
{
	return Convolute1D<Sampler, Dir>(sampler, kernel, Rows, queue);
}

template<typename Sampler, typename Dir, typename T>
Image<T> Convolute1D(Image<T>& sampler, const Eigen::VectorXf& kernel, cl::sycl::queue& queue)
{
	return Convolute1D<Sampler, Dir>(sampler, kernel, kernel.rows(), queue);
}

template<typename Sampler, typename T, typename K>
auto ConvoluteSeparable(Image<T>& sampler, const K& kernel, unsigned int size, cl::sycl::queue& queue)
{
	auto r1 = Convolute1D<Sampler, HORIZONTAL>(sampler, kernel, size, queue);
	return Convolute1D<Sampler, VERTICAL>(r1, kernel, size);
}

template<typename Sampler, typename T, int Rows, int Cols>
auto ConvoluteSeparable(Image<T>& sampler, const Eigen::Matrix<float, Rows, Cols>& kernel, cl::sycl::queue& queue)
{
	auto r1 = Convolute1D<Sampler, HORIZONTAL>(sampler, kernel, Rows, queue);
	return Convolute1D<Sampler, VERTICAL>(r1, kernel, Rows, queue);
}

template<typename Sampler, typename T>
auto ConvoluteSeparable(Image<T>& sampler, const Eigen::VectorXf& kernel, cl::sycl::queue& queue)
{
	auto r1 = Convolute1D<Sampler, HORIZONTAL>(sampler, kernel, kernel.rows(), queue);
	return Convolute1D<Sampler, VERTICAL>(r1, kernel, kernel.rows(), queue);
}

}

#endif

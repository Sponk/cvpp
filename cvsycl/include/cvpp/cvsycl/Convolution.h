#ifndef __SYCL_CONVOLUTION_H__
#define __SYCL_CONVOLUTION_H__

#include <cvpp/Image.h>
#include "Sampler.h"
#include "Image.h"
#include <eigen3/Eigen/StdVector>

namespace cvsycl
{

template<typename A, typename B> class convolute2d_kernel;

template<typename Sampler, typename T>
Image<T> Convolute2D(Image<T>& in, const float* kernel, int size, cl::sycl::queue& queue)
{
	Image<T> out(in.getWidth(), in.getHeight(), in.getComponents());
	const int halfSize = size/2;

	using namespace cl::sycl;

	auto kernelBuf = buffer<float>(kernel, cl::sycl::range<1>(size*size));

	queue.submit([&](cl::sycl::handler& cgh) {
		Sampler sampler(cgh, in);
		auto outAcc = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);
		auto kernelAcc = kernelBuf.template get_access<access::mode::read, access::target::constant_buffer>(cgh);

		cgh.parallel_for<convolute2d_kernel<Sampler, T>>(cl::sycl::range<2>(in.getWidth(), in.getHeight()), [=](cl::sycl::id<2> p)
		{
			const size_t xoff = (sampler.getWidth()*p[1] + p[0]) * sampler.getComponents();
			cl::sycl::float4 sum(0, 0, 0, 0);
			for(int kx = -halfSize; kx <= halfSize; kx++)
			{
				for(int ky = -halfSize; ky <= halfSize; ky++)
				{
					const auto s = sampler.sample((int) p[0] + kx, (int) p[1] + ky);
					sum += kernelAcc[(ky + halfSize)*size + kx + halfSize] * s;
				}
			}

			auto* outPtr = &outAcc[xoff];
			float* sumPtr = (float*) &sum;
			for(int c = 0; c < sampler.getComponents(); c++)
			{
				outPtr[c] = cvpp::FloatToColor<T>(sumPtr[c]);
			}
		});
	}).wait();

	return out;
}

template<typename Sampler, typename T, int W, int H>
Image<T> Convolute2D(Image<T>& sampler, const Eigen::Matrix<float, W, H>& kernel, cl::sycl::queue& queue)
{
	static_assert(W == H, "A kernel needs to be square!");
	static_assert(W % 2 != 0, "A kernel needs an odd size!");
	return Convolute2D<Sampler>(sampler, kernel.data(), W, queue);
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
	}).wait();

	return out;
}

template<typename Sampler, int Stride = 1, typename T, typename Fn>
Image<T> NonLinearConv2D(Image<T>& in, unsigned int size, Fn fn)
{
	return NonLinearConv2D<Sampler, Stride>(in, size, fn, [](auto, auto, auto sum){ return sum; });
}

class HORIZONTAL;
class VERTICAL;

template<typename Sampler, typename Dir, typename T> class convolute1d_kernel;

template<typename Sampler, typename Dir, typename T>
Image<T> Convolute1D(Image<T>& in, const float* kernel, int size, cl::sycl::queue& queue)
{
	Image<T> out(in.getWidth(), in.getHeight(), in.getComponents());
	const int halfSize = size/2;

	using namespace cl::sycl;

	auto kernelBuf = buffer<float>(kernel, cl::sycl::range<1>(size));

	queue.submit([&](cl::sycl::handler& cgh) {
		Sampler sampler(cgh, in);
		auto outAcc = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);
		auto kernelAcc = kernelBuf.template get_access<access::mode::read, access::target::constant_buffer>(cgh);

		cgh.parallel_for<convolute1d_kernel<Sampler, Dir, T>>(cl::sycl::range<2>(in.getWidth(), in.getHeight()), [halfSize, sampler, outAcc, kernelAcc](cl::sycl::id<2> p)
		{
			const size_t xoff = (sampler.getWidth()*p[1] + p[0]) * sampler.getComponents();
			cl::sycl::float4 sum(0, 0, 0, 0);
			for(int k = -halfSize; k <= halfSize; k++)
			{
				if constexpr(std::is_same_v<Dir, HORIZONTAL>)
				{
					sum += kernelAcc[k + halfSize] * sampler.sample((int) p[0] + k, (int) p[1]);
				}
				else
				{
					sum += kernelAcc[k + halfSize] * sampler.sample((int) p[0], (int) p[1] + k);
				}
			}

#if 0
			auto* outPtr = &outAcc[xoff];
			switch(sampler.getComponents())
			{
			case 1:
				outPtr[0] = cvpp::FloatToColor<T>(sum.x());
			break;
			
			case 2:
				outPtr[0] = cvpp::FloatToColor<T>(sum.x());
				outPtr[1] = cvpp::FloatToColor<T>(sum.y());
			break;
			
			case 3:
				outPtr[0] = cvpp::FloatToColor<T>(sum.x());
				outPtr[1] = cvpp::FloatToColor<T>(sum.y());
				outPtr[2] = cvpp::FloatToColor<T>(sum.z());
			break;
		
			case 4:
				outPtr[0] = cvpp::FloatToColor<T>(sum.x());
				outPtr[1] = cvpp::FloatToColor<T>(sum.y());
				outPtr[2] = cvpp::FloatToColor<T>(sum.z());
				outPtr[3] = cvpp::FloatToColor<T>(sum.w());
			break;
			}
#endif

			#if 1
			auto* outPtr = &outAcc[xoff];
			float* sumPtr = (float*) &sum;
			for(int c = 0; c < sampler.getComponents(); c++)
			{
				outPtr[c] = cvpp::FloatToColor<T>(sumPtr[c]);
			}
			#endif
		});
	}).wait();

	return out;
}

template<typename Sampler, typename Dir, typename T, int Rows, int Cols>
Image<T> Convolute1D(Image<T>& sampler, const Eigen::Matrix<float, Rows, Cols>& kernel, cl::sycl::queue& queue)
{
	return Convolute1D<Sampler, Dir>(sampler, kernel.data(), Rows, queue);
}

template<typename Sampler, typename Dir, typename T>
Image<T> Convolute1D(Image<T>& sampler, const Eigen::VectorXf& kernel, cl::sycl::queue& queue)
{
	return Convolute1D<Sampler, Dir>(sampler, kernel.data(), kernel.rows(), queue);
}

template<typename Sampler, typename T, typename K>
auto ConvoluteSeparable(Image<T>& sampler, const K& kernel, unsigned int size, cl::sycl::queue& queue)
{
	auto r1 = Convolute1D<Sampler, HORIZONTAL>(sampler, kernel.data(), size, queue);
	return Convolute1D<Sampler, VERTICAL>(r1, kernel.data(), size);
}

template<typename Sampler, typename T, int Rows, int Cols>
auto ConvoluteSeparable(Image<T>& sampler, const Eigen::Matrix<float, Rows, Cols>& kernel, cl::sycl::queue& queue)
{
	auto r1 = Convolute1D<Sampler, HORIZONTAL>(sampler, kernel.data(), Rows, queue);
	return Convolute1D<Sampler, VERTICAL>(r1, kernel.data(), Rows, queue);
}

template<typename Sampler, typename T>
auto ConvoluteSeparable(Image<T>& sampler, const Eigen::VectorXf& kernel, cl::sycl::queue& queue)
{
	auto r1 = Convolute1D<Sampler, HORIZONTAL>(sampler, kernel.data(), kernel.rows(), queue);
	return Convolute1D<Sampler, VERTICAL>(r1, kernel.data(), kernel.rows(), queue);
}

}

#endif

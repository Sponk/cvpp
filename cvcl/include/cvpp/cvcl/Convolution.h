#ifndef __CVCL_CONVOLUTION_H__
#define __CVCL_CONVOLUTION_H__

#include "Queue.h"
#include "Image.h"
#include <eigen3/Eigen/StdVector>

namespace cvcl
{

template<typename T, typename K>
Image<T> Convolute2D(Image<T>& in, const K& kernel, unsigned int size, Queue& queue)
{
	Image<T> out(in.getWidth(), in.getHeight(), in.getComponents(), queue);

	constexpr auto __krnl_types = MakeTypeDefine<T>(cs("T"));
	constexpr auto __krnl = __krnl_types + cs(R"CLC(
		// TODO: Sampler as template argument!
		__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
		__kernel void Convolute2D(__read_only image2d_t in, __write_only image2d_t out, __constant float* k, int size)
		{
			const int2 pos = {get_global_id(0), get_global_id(1)};
			const int halfSize = size/2;
			float4 sum = (float4)(0, 0, 0, 1.0f);
			for(int kx = -halfSize; kx <= halfSize; kx++)
			{
				for(int ky = -halfSize; ky <= halfSize; ky++)
				{
					const float4 s = read_imagef(in, sampler, pos + (int2)(kx, ky));
					sum += k[(ky + halfSize)*size + kx + halfSize] * s;
				}
			}

			write_imagef(out, pos, sum);
		}
	)CLC");

	queue.addProgram(uuid(__krnl), __krnl.c_str());

	auto kbuf = cl::Buffer(queue.getCtx(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(kernel), (void*) &kernel);
	queue(in.getWidth(), in.getHeight(), "Convolute2D", in.getImage(), out.getImage(), kbuf, size);
	queue.flush();

	return out;
}

template<typename T, int W, int H>
Image<T> Convolute2D(Image<T>& sampler, const Eigen::Matrix<float, W, H>& kernel, Queue& queue)
{
	static_assert(W == H, "A kernel needs to be square!");
	static_assert(W % 2 != 0, "A kernel needs an odd size!");
	return Convolute2D(sampler, kernel, W, queue);
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

enum CONVOLUTION_TYPE
{
	HORIZONTAL = 0,
	VERTICAL
};

template<CONVOLUTION_TYPE Dir = HORIZONTAL, typename T, typename K>
Image<T> Convolute1D(Image<T>& in, const K& kernel, unsigned int size, Queue& queue)
{
	Image<T> out(in.getWidth(), in.getHeight(), in.getComponents(), queue);

	constexpr auto __krnl = cs(R"CLC(
		// TODO: Sampler as template argument!
		__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
		__kernel void Convolute1D(__read_only image2d_t in, __write_only image2d_t out, __constant float* krnl, const int size, const int dir)
		{
			const int2 pos = {get_global_id(0), get_global_id(1)};
			const int halfSize = size/2;
			float4 sum = (float4)(0, 0, 0, 1.0f);
			
			for(int k = -halfSize; k <= halfSize; k++)
			{
				if(dir == 0)
				{
					sum += krnl[k + halfSize] * read_imagef(in, sampler, pos + (int2)(k, 0));
				}
				else
				{
					sum += krnl[k + halfSize] * read_imagef(in, sampler, pos + (int2)(0, k));
				}
			}

			write_imagef(out, pos, sum);
		}
	)CLC");

	queue.addProgram(uuid(__krnl), __krnl.c_str());

	auto kbuf = cl::Buffer(queue.getCtx(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(kernel), (void*) &kernel);
	queue(in.getWidth(), in.getHeight(), "Convolute1D", in.getImage(), out.getImage(), kbuf, size, Dir);

	return out;
}

template<typename Dir, typename T, int Rows, int Cols>
Image<T> Convolute1D(Image<T>& sampler, const Eigen::Matrix<float, Rows, Cols>& kernel, Queue& queue)
{
	return Convolute1D<Dir>(sampler, kernel, Rows, queue);
}

template<typename Dir, typename T>
Image<T> Convolute1D(Image<T>& sampler, const Eigen::VectorXf& kernel, Queue& queue)
{
	return Convolute1D<Dir>(sampler, kernel, kernel.rows(), queue);
}

template<typename T, typename K>
auto ConvoluteSeparable(Image<T>& sampler, const K& kernel, unsigned int size, Queue& queue)
{
	auto r1 = Convolute1D<HORIZONTAL>(sampler, kernel, size, queue);
	return Convolute1D<VERTICAL>(r1, kernel, size);
}

template<typename T, int Rows, int Cols>
auto ConvoluteSeparable(Image<T>& sampler, const Eigen::Matrix<float, Rows, Cols>& kernel, Queue& queue)
{
	auto r1 = Convolute1D<HORIZONTAL>(sampler, kernel, Rows, queue);
	return Convolute1D<VERTICAL>(r1, kernel, Rows, queue);
}

template<typename T>
auto ConvoluteSeparable(Image<T>& sampler, const Eigen::VectorXf& kernel, Queue& queue)
{
	auto r1 = Convolute1D<HORIZONTAL>(sampler, kernel, kernel.rows(), queue);
	return Convolute1D<VERTICAL>(r1, kernel, kernel.rows(), queue);
}

}
#endif

#ifndef __CL_STRUCTURE_TENSOR_H__
#define __CL_STRUCTURE_TENSOR_H__

#include "Image.h"
#include "Convolution.h"

#include <cvpp/CommonFilters.h>

namespace cvcl
{

template<typename T>
Image<T> StructureTensor(Image<T>& in, Queue& q)
{
	auto gray = MakeGrayscale(in, q);

	auto Dx = Convolute2D(gray, cvpp::ScharrFilterH(), q);
	auto Dy = Convolute2D(gray, cvpp::ScharrFilterV(), q);

	auto Sx = Dx.mul(Dx, q);
	auto Sy = Dy.mul(Dy, q);
	auto Sxy = Dx.mul(Dy, q);

	Sx = ConvoluteSeparable(Sx, cvpp::GaussFilter<3>(1.0f), q);
	Sy = ConvoluteSeparable(Sy, cvpp::GaussFilter<3>(1.0f), q);
	Sxy = ConvoluteSeparable(Sxy, cvpp::GaussFilter<3>(1.0f), q);

	Image<T> output(in.getWidth(), in.getHeight(), 4, q);

	constexpr auto __krnl = cs(R"CLC(
		// TODO: Sampler as template argument!
		__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
		__kernel void StructureTensor(__read_only image2d_t Sx, __read_only image2d_t Sxy, __read_only image2d_t Sy, __write_only image2d_t out)
		{
			const int2 pos = {get_global_id(0), get_global_id(1)};
			
			float4 px;
			px.x = read_imagef(Sx, sampler, pos).x;
			px.y = read_imagef(Sxy, sampler, pos).x;
			px.w = read_imagef(Sy, sampler, pos).x;
			px.z = px.y;

			write_imagef(out, pos, px);
		}
	)CLC");

	q.addProgram(uuid(__krnl), __krnl.c_str());
	q(in.getWidth(), in.getHeight(), "StructureTensor", Sx.getImage(), Sxy.getImage(), Sy.getImage(), output.getImage());
	return output;
}

#if 0
template<typename T>
Image<Eigen::Matrix2f> HessianTensor(Image<T>& in, cl::sycl::queue& q)
{
	auto gray = ConvertType<T, float>(in, q);
	gray = MakeGrayscale(gray, q);
	gray = ConvoluteSeparable<ClampView<float>>(gray, cvpp::GaussFilter<3>(0.25f), q);

	auto Dx = Convolute1D<ClampView<float>, HORIZONTAL>(gray, cvpp::LaplaceFilterX, q);
	auto Dy = Convolute1D<ClampView<float>, VERTICAL>(gray, cvpp::LaplaceFilterX, q);
	auto Dxy = Convolute2D<ClampView<float>>(gray, cvpp::LaplaceFilterXY(), q);
	auto Dyx = Convolute2D<ClampView<float>>(gray, (Eigen::Matrix3f) cvpp::LaplaceFilterXY().transpose(), q);

	Image<Eigen::Matrix2f> output(in.getWidth(), in.getHeight(), 1);

	using namespace cl::sycl;
	q.submit([&](cl::sycl::handler& cgh) {
		auto outAcc = output.getBuffer()->template get_access<access::mode::discard_write>(cgh);
		auto DxAcc = Dx.getBuffer()->template get_access<access::mode::read>(cgh);
		auto DyAcc = Dy.getBuffer()->template get_access<access::mode::read>(cgh);
		auto DxyAcc = Dxy.getBuffer()->template get_access<access::mode::read>(cgh);
		auto DyxAcc = Dyx.getBuffer()->template get_access<access::mode::read>(cgh);

		cgh.parallel_for<hessian_tensor_kernel<T>>(cl::sycl::range<1>(in.getWidth()*in.getHeight()), [=](cl::sycl::id<1> off)
		{
			outAcc[off] <<	cvpp::ColorToFloat(DxAcc[off]), cvpp::ColorToFloat(DxyAcc[off]),
							cvpp::ColorToFloat(DyxAcc[off]), cvpp::ColorToFloat(DyAcc[off]);
		});
	});

	return output;
}
#endif

}

#endif

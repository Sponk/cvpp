#ifndef __SYCL_STRUCTURE_TENSOR_H__
#define __SYCL_STRUCTURE_TENSOR_H__

#include "Image.h"
#include "Convolution.h"

#include <cvpp/CommonFilters.h>

namespace cvsycl
{

template<typename T> class structure_tensor_kernel;
template<typename T>
Image<Eigen::Matrix2f> StructureTensor(Image<T>& in, cl::sycl::queue& q)
{
	auto gray = cvsycl::ConvertType<T, float>(in, q);
	gray = MakeGrayscale<float>(gray, q);

	auto Dx = Convolute2D<ClampView<float>>(gray, cvpp::ScharrFilterH(), q);
	auto Dy = Convolute2D<ClampView<float>>(gray, cvpp::ScharrFilterV(), q);

	auto Sx = Dx.mul(Dx, q);
	auto Sy = Dy.mul(Dy, q);
	auto Sxy = Dx.mul(Dy, q);

	#if 0
	ConvertType<float, uint8_t>(Dx, q).save("StructureTensorDebug_Dx.png");
	ConvertType<float, uint8_t>(Dy, q).save("StructureTensorDebug_Dy.png");

	ConvertType<float, uint8_t>(Sx, q).save("StructureTensorDebug_Sx.png");
	ConvertType<float, uint8_t>(Sy, q).save("StructureTensorDebug_Sy.png");
	ConvertType<float, uint8_t>(Sxy, q).save("StructureTensorDebug_Sxy.png");
	#endif

	Sx = ConvoluteSeparable<ClampView<float>>(Sx, cvpp::GaussFilter<3>(1.0f), q);
	Sy = ConvoluteSeparable<ClampView<float>>(Sy, cvpp::GaussFilter<3>(1.0f), q);
	Sxy = ConvoluteSeparable<ClampView<float>>(Sxy, cvpp::GaussFilter<3>(1.0f), q);

	Image<Eigen::Matrix2f> output(in.getWidth(), in.getHeight(), 1);

	using namespace cl::sycl;
	q.submit([&](cl::sycl::handler& cgh) {
		auto outAcc = output.getBuffer()->template get_access<access::mode::discard_write>(cgh);
		auto SxAcc = Sx.getBuffer()->template get_access<access::mode::read>(cgh);
		auto SyAcc = Sy.getBuffer()->template get_access<access::mode::read>(cgh);
		auto SxyAcc = Sxy.getBuffer()->template get_access<access::mode::read>(cgh);

		cgh.parallel_for<structure_tensor_kernel<T>>(cl::sycl::range<1>(in.getWidth()*in.getHeight()), [=](cl::sycl::id<1> off)
		{
			outAcc[off] <<	cvpp::ColorToFloat(SxAcc[off]), cvpp::ColorToFloat(SxyAcc[off]),
							cvpp::ColorToFloat(SxyAcc[off]), cvpp::ColorToFloat(SyAcc[off]);
		});
	}).wait();

	return output;
}

template<typename T> class hessian_tensor_kernel;
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
	}).wait();

	return output;
}

}

#endif

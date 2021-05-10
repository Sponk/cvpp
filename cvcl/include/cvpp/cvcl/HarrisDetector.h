#ifndef __CL_HARRIS_DETECTOR_H__
#define __CL_HARRIS_DETECTOR_H__

#include <cvpp/Convolution.h>
#include <cvpp/Feature.h>
#include "StructureTensor.h"

#include <mutex>

namespace cvcl
{
template<typename T>
void HarrisDetector(Image<T>& in, unsigned int patchSize, float threshold, std::vector<cvpp::Feature>& features, Queue& q)
{
	Image<float> determinant(in.getWidth(), in.getHeight(), 1, q);

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

		constexpr auto __krnl = cs(R"CLC(
			// TODO: Sampler as template argument!
			__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
			__kernel void StructureTensorHarris(__read_only image2d_t Sx, __read_only image2d_t Sxy, __read_only image2d_t Sy, __write_only image2d_t out)
			{
				const int2 pos = {get_global_id(0), get_global_id(1)};
				
				float4 px;
				px.x = read_imagef(Sx, sampler, pos).x;
				px.y = read_imagef(Sxy, sampler, pos).x;
				px.w = read_imagef(Sy, sampler, pos).x;
				px.z = px.y;

				float d = (px.x*px.w - px.y*px.z) / (px.x+px.w);
				write_imagef(out, pos, d);
			}
		)CLC");

		q.addProgram(uuid(__krnl), __krnl.c_str());
		q(in.getWidth(), in.getHeight(), "StructureTensorHarris", Sx.getImage(), Sxy.getImage(), Sy.getImage(), determinant.getImage()).wait();
	}

	std::mutex mtx;

	cvpp::Image<float> cpuImg = determinant.toImage(q);
	cvpp::NonLinearConv2D<-1>(cvpp::ClampView(cpuImg), patchSize,
	[](int x, int y, auto v, auto& result) {
		auto vabs = std::abs(v[0]);
		if(vabs >= result[0])
		{
			result[0] = vabs;
			result[1] = x;
			result[2] = y;
		}
	},
	[&features, threshold, &mtx](auto x, auto y, auto v) {
		if(v[0] >= threshold)
		{
			std::lock_guard<std::mutex> g(mtx);
			features.emplace_back(cvpp::Feature{static_cast<unsigned int>(x + v[1]), static_cast<unsigned int>(y + v[2]), 0.0f});
		}
	});
}

#if 0

template<typename T>
void HessianDetector(Image<T>& in, unsigned int patchSize, float threshold, std::vector<cvpp::Feature>& features, cl::sycl::queue& q)
{
	auto gray = MakeGrayscale(ConvertType<T, float>(in, q), q);

	auto tensor = HessianTensor(in, q);
	auto determinant = tensor.transform(q, [](const Eigen::Matrix2f& mtx) -> float {
		return mtx.determinant();
	});
	
	std::mutex mtx;
	cvpp::Image<float> cpuImg = determinant;
	cvpp::NonLinearConv2D<-1>(cvpp::ClampView(cpuImg), patchSize,
	[](int x, int y, auto v, auto& result) {
		auto vabs = std::abs(v[0]);
		if(vabs >= result[0])
		{
			result[0] = vabs;
			result[1] = x;
			result[2] = y;
		}
	},
	[&features, threshold, &mtx](auto x, auto y, auto v) {
		if(v[0] >= threshold)
		{
			std::lock_guard<std::mutex> g(mtx);
			features.emplace_back(cvpp::Feature{static_cast<unsigned int>(x + v[1]), static_cast<unsigned int>(y + v[2]), 0.0f});
		}
	});
}
#endif

}

#endif

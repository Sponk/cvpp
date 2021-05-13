#ifndef __SYCL_HARRIS_DETECTOR_H__
#define __SYCL_HARRIS_DETECTOR_H__

#include <cvpp/Convolution.h>
#include <cvpp/Feature.h>
#include "StructureTensor.h"

#include <mutex>

namespace cvsycl
{
template<typename T>
void HarrisDetector(Image<T>& in, unsigned int patchSize, float threshold, std::vector<cvpp::Feature>& features, cl::sycl::queue& q)
{
	Image<float> determinant(in.getWidth(), in.getHeight(), 1);
	auto tensor = StructureTensor(in, q);
	{
		auto gray = cvsycl::ConvertType<T, float>(in, q);
		gray = MakeGrayscale<float>(gray, q);

		auto Dx = Convolute2D<ClampView<float>>(gray, cvpp::ScharrFilterH(), q);
		auto Dy = Convolute2D<ClampView<float>>(gray, cvpp::ScharrFilterV(), q);

		auto Sx = Dx.mul(Dx, q);
		auto Sy = Dy.mul(Dy, q);
		auto Sxy = Dx.mul(Dy, q);

		Sx = ConvoluteSeparable<ClampView<float>>(Sx, cvpp::GaussFilter<3>(1.0f), q);
		Sy = ConvoluteSeparable<ClampView<float>>(Sy, cvpp::GaussFilter<3>(1.0f), q);
		Sxy = ConvoluteSeparable<ClampView<float>>(Sxy, cvpp::GaussFilter<3>(1.0f), q);

		//Image<Eigen::Matrix2f> output(in.getWidth(), in.getHeight(), 1);

		using namespace cl::sycl;
		q.submit([&](cl::sycl::handler& cgh) {
			auto outAcc = determinant.getBuffer()->template get_access<access::mode::discard_write>(cgh);
			auto SxAcc = Sx.getBuffer()->template get_access<access::mode::read>(cgh);
			auto SyAcc = Sy.getBuffer()->template get_access<access::mode::read>(cgh);
			auto SxyAcc = Sxy.getBuffer()->template get_access<access::mode::read>(cgh);

			cgh.parallel_for(cl::sycl::range<1>(in.getWidth()*in.getHeight()), [=](cl::sycl::id<1> off)
			{
				const float a = cvpp::ColorToFloat(SxAcc[off]);
				const float b = cvpp::ColorToFloat(SxyAcc[off]);
				const float c = b;
				const float d = cvpp::ColorToFloat(SyAcc[off]);

				outAcc[off] = (a*d - b*c)/(a+d);
			});
		});
	}
	
	//auto determinant = tensor.template transform<class HarrisDetectorKernel>(q, [](const Eigen::Matrix2f& mtx) -> float {
	//	return (mtx.determinant() / mtx.trace());
	//});

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

}

#endif

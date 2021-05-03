#ifndef __HARRIS_DETECTOR_H__
#define __HARRIS_DETECTOR_H__

#include "Feature.h"
#include "StructureTensor.h"

#include <mutex>

namespace cvpp
{
template<typename T>
void HarrisDetector(const Image<T>& in, unsigned int patchSize, float threshold, std::vector<Feature>& features)
{
	auto tensor = cvpp::StructureTensor(in);
	auto determinant = tensor.transform([](const Eigen::Matrix2f& mtx) -> float {
		return (mtx.determinant() / mtx.trace());
	});
	
	std::mutex mtx;

	determinant = cvpp::NonLinearConv2D<-1>(cvpp::ClampView(determinant), patchSize,
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
			features.emplace_back(Feature{static_cast<unsigned int>(x + v[1]), static_cast<unsigned int>(y + v[2]), 0.0f});
		}
	});
}

template<typename T>
void HessianDetector(const Image<T>& in, unsigned int patchSize, float threshold, std::vector<Feature>& features)
{
	auto gray = MakeGrayscale(ConvertType<T, float>(in));
	auto sampler = ClampView(gray);

	auto tensor = cvpp::HessianTensor(in);
	auto determinant = tensor.transform([](const Eigen::Matrix2f& mtx) -> float {
		return mtx.determinant();
	});
	
	std::mutex mtx;
	determinant = cvpp::NonLinearConv2D<-1>(cvpp::ClampView(determinant), patchSize,
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
			features.emplace_back(Feature{static_cast<unsigned int>(x + v[1]), static_cast<unsigned int>(y + v[2]), 0.0f});
		}
	});
}

}

#endif

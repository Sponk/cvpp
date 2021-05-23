#ifndef __HARRIS_DETECTOR_H__
#define __HARRIS_DETECTOR_H__

#include "Feature.h"
#include "StructureTensor.h"

#include <mutex>

namespace cvpp
{
template<typename T>
void HarrisDetector(const CPUImage<T>& in, unsigned int patchSize, float threshold, std::vector<Feature>& features)
{
//	auto tensor = cvpp::StructureTensor(in);
//	auto determinant = tensor.transform([](const Eigen::Matrix2f& mtx) -> float {
//		return (mtx.determinant() / mtx.trace());
//	});

	CPUImage<float> scaledHarris(in.getWidth(), in.getHeight(), 2);

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

		#pragma omp parallel for
		for(int y = 0; y < in.getHeight(); y++)
		{
			GaussView SxView(Sx);
			GaussView SyView(Sy);
			GaussView SxyView(Sxy);

			const size_t yoff = y*in.getWidth();
			for(int x = 0; x < in.getWidth(); x++)
			{
				const size_t off = (yoff+x)*2;

				float sigma = 1;
				float harris = 0;
				float h0 = 0;

				float u = float(x)/in.getWidth();
				float v = float(y)/in.getHeight();

				// First iteration
				{
					const float a = SxView.sample(u, v).x();
					const float b = SxyView.sample(u, v).x();
					const float c = b;
					const float d = SyView.sample(u, v).x();

					h0 = harris = std::abs((a*d - b*c)/(a+d));
				}

				for(int i = 0; i < 1000; i++)
				{
					SxView.setSigma(sigma);
					SyView.setSigma(sigma);
					SxyView.setSigma(sigma);

					const float a = SxView.sample(u, v).x();
					const float b = SxyView.sample(u, v).x();
					const float c = b;
					const float d = SyView.sample(u, v).x();

					const float nharris = std::abs((a*d - b*c)/(a+d));
					const float dh = nharris - harris;

					if(dh <= 0) break;
					
					harris = nharris;
					sigma += 0.1;
				}

#if 0
				if(sigma != 1)
				{
					std::cout << "Harris: " << harris << std::endl;
					std::cout << "SIGMA: " << sigma << std::endl;
				}
#endif

				scaledHarris[off] = h0;
				scaledHarris[off + 1] = sigma;

				//output[off] <<	ColorToFloat(Sx[off]), ColorToFloat(Sxy[off]),
				//				ColorToFloat(Sxy[off]), ColorToFloat(Sy[off]);
			}
		}
	}
	
	std::mutex mtx;

	auto determinant = cvpp::NonLinearConv2D<-1>(cvpp::ClampView(scaledHarris), patchSize,
	[](int x, int y, auto v, auto& result) {
		auto vabs = std::abs(v[0]);
		auto sigma = v[1];
		if(vabs >= result[0])
		{
			result[0] = vabs;
			result[1] = x;
			result[2] = y;
			result[3] = sigma;
		}
	},
	[&features, threshold, &mtx](auto x, auto y, auto v) {
		if(v[0] >= threshold)
		{
			std::lock_guard<std::mutex> g(mtx);
			features.emplace_back(Feature{static_cast<unsigned int>(x + v[1]), static_cast<unsigned int>(y + v[2]), v[3]});
		}
	});
}

template<typename T>
void HessianDetector(const CPUImage<T>& in, unsigned int patchSize, float threshold, std::vector<Feature>& features)
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

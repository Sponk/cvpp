#ifndef __COMMON_FILTERS_H__
#define __COMMON_FILTERS_H__

#include <eigen3/Eigen/StdVector>

namespace cvpp
{

const Eigen::Vector3f SimpleEdgeDetector(0.5f, 0, -0.5f);

Eigen::Matrix3f SobelFilterH()
{
	Eigen::Matrix3f mtx;
	mtx <<	-1, 0, 1,
			-2, 0, 2,
			-1, 0, 1;

	return mtx;
}

Eigen::Matrix3f SobelFilterV()
{
	return SobelFilterH().transpose();
}

Eigen::Matrix3f ScharrFilterH()
{
	Eigen::Matrix3f mtx;
	mtx <<	3,  0, -3,
			10, 0, -10,
			3,  0, -3;

	return mtx;
}

Eigen::Matrix3f ScharrFilterV()
{
	return ScharrFilterH().transpose();
}

const Eigen::Vector3f LaplaceFilterX(1, -2, 1);
auto LaplaceFilter()
{
	Eigen::Matrix3f mtx;
	mtx <<	1,  1, 1,
			1, -8, 1,
			1,  1, 1;

	return mtx;
}

auto LaplaceFilterXY()
{
	Eigen::Matrix3f mtx;
	mtx <<	1,  0, 0,
			0, -2, 0,
			0,  0, 1;

	return mtx;
}

template<unsigned int SZ>
constexpr auto BoxFilter()
{
	constexpr unsigned int S = (SZ % 2 != 0) ? SZ : SZ + 1;
	constexpr float Norm = S;//(S*S);
	Eigen::Matrix<float, S, 1> krnl;

	for(int i = 0; i < S; i++)
		krnl[i] = 1.0f/Norm;

	return krnl;
}

// https://en.wikipedia.org/wiki/Box_blur
// https://observablehq.com/@jobleonard/mario-klingemans-stackblur
template<unsigned int SZ>
constexpr auto StackBlurFilter()
{
	constexpr int S = (SZ % 2 != 0) ? SZ : SZ + 1;
	Eigen::Matrix<float, S, 1> krnl;
	float norm = 0.0f;

	for(int i = 0; i <= S/2; i++)
	{
		krnl[i] = (i + 1.0f);
		norm += krnl[i];
	}

	for(int i = S/2 + 1; i < S; i++)
	{
		krnl[i] = (S-i);
		norm += krnl[i];
	}

	for(int i = 0; i < S; i++)
	{
		krnl[i] /= norm;
	}

	return krnl;
}

template<unsigned int SZ>
constexpr auto GaussFilter(float sigma)
{
	constexpr int S = (SZ % 2 != 0) ? SZ : SZ + 1;
	Eigen::Matrix<float, S, 1> krnl;

	const float sigmaSq = sigma*sigma;
	const float twoSigmaSq = sigmaSq + sigmaSq;

	float sum = 0.0f;
	for(int i = 0; i < S; i++)
	{
		const float s = (i - S) + 1;
		const float x = -(s/2.0f) + 0.5f;
		krnl[i] = (1.0f/std::sqrt(2.0f*M_PI*sigmaSq)) * std::exp(-(x*x)/twoSigmaSq);
		sum += krnl[i];
	}

	krnl /= sum;
	return krnl;
}

}

#endif

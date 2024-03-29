#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include "Image.h"
#include <eigen3/Eigen/StdVector>

namespace cvpp
{

template<typename T>
class SamplerView
{
public:
	SamplerView(CPUImage<T>* img):
		m_image(img) {}

	SamplerView(const CPUImage<T>& img):
		m_image(&img) {}

	virtual Eigen::Vector4f sample(float u, float v) const
	{
		assert(u >= 0.0f && u <= 1.0f);
		assert(v >= 0.0f && v <= 1.0f);

		Eigen::Vector4f px(0, 0, 0, 1.0f);
		auto xy = getXY(u, v);
		auto* s = m_image->get(xy.x(), xy.y());

		for(int i = 0; i < m_image->getComponents(); i++)
			px[i] = ColorToFloat<T>(s[i]);

		return px;
	}

	virtual Eigen::Vector4f sample(int x, int y) const
	{
		const float u = static_cast<float>(x)/(m_image->getWidth() - 1);
		const float v = static_cast<float>(y)/(m_image->getHeight() - 1);
		return sample(u, v);
	}

	Eigen::Vector4f texel(int x, int y) const
	{
		assert(x >= 0 && x < m_image->getWidth());
		assert(y >= 0 && y < m_image->getHeight());

		Eigen::Vector4f px(0, 0, 0, 1.0f);
		auto* s = m_image->get(x, y);

		for(int i = 0; i < m_image->getComponents(); i++)
			px[i] = ColorToFloat<T>(s[i]);

		return px;
	}

	Eigen::Vector2i getXY(float u, float v) const
	{
		const float x = std::trunc(u*(m_image->getWidth() - 1));
		const float y = std::trunc(v*(m_image->getHeight() - 1));
		return Eigen::Vector2i(x, y);
	}

	const CPUImage<T>* getImage() const { return m_image; }

protected:
	const CPUImage<T>* m_image = nullptr;
};

template<typename T>
class ClampView : public SamplerView<T>
{
public:
	ClampView(const CPUImage<T>* img):
		SamplerView<T>(img) {}

	ClampView(const CPUImage<T>& img):
		SamplerView<T>(img) {}

	Eigen::Vector4f sample(float u, float v) const
	{
		u = std::clamp(u, 0.0f, 1.0f);
		v = std::clamp(v, 0.0f, 1.0f);

		return SamplerView<T>::sample(u, v);
	}
};


template<typename T>
class GaussView : public SamplerView<T>
{
public:
	GaussView(const CPUImage<T>* img):
		SamplerView<T>(img) {}

	GaussView(const CPUImage<T>& img):
		SamplerView<T>(img) {}

	using SamplerView<T>::texel;
	Eigen::Vector4f sample(float u, float v) const
	{
		const int w = SamplerView<T>::m_image->getWidth() - 1;
		const int h = SamplerView<T>::m_image->getHeight() - 1;

		const int x = u*w;
		const int y = v*h;

		const int sz = 3*m_sigma;

		const float sigmaSq = m_sigma*m_sigma;
		const float twoSigmaSq = sigmaSq + sigmaSq;

		Eigen::Vector4f sum;
		for(int dx = -sz; dx <= sz; dx++)
			for(int dy = -sz; dy <= sz; dy++)
			{
				const float weight = (1.0f/std::sqrt(2.0f*M_PI*sigmaSq)) * std::exp(-((dx*dx) + (dy*dy))/twoSigmaSq);
				sum += weight * texel(std::clamp(x + dx, 0, w), std::clamp(y + dy, 0, h));
			}

		sum /= sz;
		return sum;
	}

	void setSigma(float s) { m_sigma = s; }

private:
	float m_sigma = 1.0f;
};

template<typename T>
inline T mod(T a, T b)
{
	assert(b != 0);
	if(b < T(0))
		return -mod(-a, -b);

	if constexpr(std::is_unsigned_v<T> && std::is_integral_v<T>)
		return a%b;
	
	T r;
	if constexpr(std::is_integral_v<T>)
	{
		r = a%b;
	}
	else
	{
		r = std::fmod(a, b);
	}

	return (r < T(0) ? (r + b) % b : r);
}

template<typename T>
class RepeatView : public SamplerView<T>
{
public:
	RepeatView(const CPUImage<T>* img):
		SamplerView<T>(img) {}

	RepeatView(const CPUImage<T>& img):
		SamplerView<T>(img) {}

	Eigen::Vector4f sample(float u, float v) const final
	{
		auto p = SamplerView<T>::getXY(u, v);
		auto* img = SamplerView<T>::getImage();
		return SamplerView<T>::texel(mod(p.x(), (int) img->getWidth()), mod(p.y(), (int) img->getHeight()));
	}
};

template<typename T>
class BlackEdgeView : public SamplerView<T>
{
public:
	BlackEdgeView(const CPUImage<T>* img):
		SamplerView<T>(img) {}

	BlackEdgeView(const CPUImage<T>& img):
		SamplerView<T>(img) {}

	Eigen::Vector4f sample(float u, float v) const final
	{
		if(u < 0.0f || u >= 1.0f || v < 0.0f || v >= 1.0f)
			return Eigen::Vector4f(0, 0, 0, 0);

		return SamplerView<T>::sample(u, v);
	}
};

}

#endif

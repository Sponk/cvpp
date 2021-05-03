#ifndef __SYCL_SAMPLER_H__
#define __SYCL_SAMPLER_H__

#include "Image.h"
#include <eigen3/Eigen/StdVector>

namespace cvsycl
{

template<typename T>
class SamplerView
{
public:
	SamplerView(cl::sycl::handler& cgh, Image<T>* img):
		m_width(img->getWidth()),
		m_height(img->getHeight()),
		m_components(img->getComponents()),
		m_accessor(img->getBuffer()->template get_access<cl::sycl::access::mode::read_write>(cgh)) {}

	SamplerView(cl::sycl::handler& cgh, Image<T>& img):
		m_width(img.getWidth()),
		m_height(img.getHeight()),
		m_components(img.getComponents()),
		m_accessor(img.getBuffer()->template get_access<cl::sycl::access::mode::read_write>(cgh)) {}
	
	T const* get(unsigned int x, unsigned int y) const
	{
		return &m_accessor[(x+y*m_width)*m_components];
	}

	T* get(unsigned int x, unsigned int y)
	{
		return &m_accessor[(x+y*m_width)*m_components];
	}

	cl::sycl::float4 sample(float u, float v) const
	{
		assert(u >= 0.0f && u <= 1.0f);
		assert(v >= 0.0f && v <= 1.0f);

		cl::sycl::float4 px(0, 0, 0, 1.0f);
		auto xy = getXY(u, v);
		auto* s = get(xy.x(), xy.y());

		switch(m_components)
		{
		case 1: px = {cvpp::ColorToFloat(s[0]), 0, 0, 1}; break;
		case 2: px = {cvpp::ColorToFloat(s[0]), cvpp::ColorToFloat(s[1]), 0, 1}; break;
		case 3: px = {cvpp::ColorToFloat(s[0]), cvpp::ColorToFloat(s[1]), cvpp::ColorToFloat(s[2]), 1}; break;
		case 4: px = {cvpp::ColorToFloat(s[0]), cvpp::ColorToFloat(s[1]), cvpp::ColorToFloat(s[2]), cvpp::ColorToFloat(s[3])}; break;
		}

		return px;
	}

	cl::sycl::int2 getXY(float u, float v) const
	{
		const float x = std::trunc(u*(m_width - 1));
		const float y = std::trunc(v*(m_height - 1));
		return cl::sycl::int2(x, y);
	}

	cl::sycl::float4 sample(int x, int y) const
	{
		const float u = static_cast<float>(x)/(m_width - 1);
		const float v = static_cast<float>(y)/(m_height - 1);
		return sample(u, v);
	}

	cl::sycl::float4 texel(int x, int y) const
	{
		assert(x >= 0 && x < m_width);
		assert(y >= 0 && y < m_height);

		cl::sycl::float4 px(0, 0, 0, 1.0f);
		auto* s = get(x, y);

		switch(m_components)
		{
		case 1: px = {cvpp::ColorToFloat<T>(s[0]), 0, 0, 1}; break;
		
		case 2: px = { cvpp::ColorToFloat<T>(s[0]),
						cvpp::ColorToFloat<T>(s[1]), 0, 1};
		break;

		case 3: px = { cvpp::ColorToFloat<T>(s[0]),
						cvpp::ColorToFloat<T>(s[1]),
						cvpp::ColorToFloat<T>(s[2]), 1};
		break;

		case 4: px = { cvpp::ColorToFloat<T>(s[0]),
						cvpp::ColorToFloat<T>(s[1]),
						cvpp::ColorToFloat<T>(s[2]),
						cvpp::ColorToFloat<T>(s[3]) };
		break;
		}

		return px;
	}

	unsigned int getWidth() const { return m_width; }
	unsigned int getHeight() const { return m_height; }
	unsigned int getComponents() const { return m_components; }

protected:

	// TODO Only requires 'read'!
	cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write> m_accessor;
	unsigned int m_width = 0, m_height = 0, m_components = 0;
};

template<typename T>
class ClampView : public SamplerView<T>
{
public:
	ClampView(cl::sycl::handler& cgh, Image<T>* img):
		SamplerView<T>(cgh, img) {}

	ClampView(cl::sycl::handler& cgh, Image<T>& img):
		SamplerView<T>(cgh, img) {}

	cl::sycl::float4 sample(float u, float v) const
	{
		u = std::clamp(u, 0.0f, 1.0f);
		v = std::clamp(v, 0.0f, 1.0f);

		return SamplerView<T>::sample(u, v);
	}

	cl::sycl::float4 sample(int x, int y) const
	{
		const float u = static_cast<float>(x)/(SamplerView<T>::m_width - 1);
		const float v = static_cast<float>(y)/(SamplerView<T>::m_height - 1);
		return sample(u, v);
	}
};

}

#endif

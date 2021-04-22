#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <vector>
#include <string>
#include <limits>

#include <type_traits>

namespace cvpp
{

namespace ImageLoader
{
void loadUChar(const std::string& file, unsigned int& w, unsigned int& h, unsigned int& c, std::vector<unsigned char>& data);
void loadUShort(const std::string& file, unsigned int& w, unsigned int& h, unsigned int& c, std::vector<unsigned short>& data);
void loadFloat(const std::string& file, unsigned int& w, unsigned int& h, unsigned int& c, std::vector<float>& data);

void saveUChar(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const std::vector<unsigned char>& data);
void saveUShort(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const std::vector<unsigned short>& data);
void saveFloat(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const std::vector<float>& data);
}

template<typename T>
float ColorToFloat(T v)
{
	if constexpr(std::is_same<T, float>::value)
		return v;

	return ((float) v) / std::numeric_limits<T>::max();
}

template<typename T>
T FloatToColor(float v)
{
	if constexpr(std::is_same<T, float>::value)
		return v;

	return T(v * std::numeric_limits<T>::max());
}

template<typename From, typename To>
To ColorToColor(const From& v)
{
	if constexpr(std::is_same<From, To>::value)
		return v;

	return FloatToColor<To>(ColorToFloat<From>(v));
}

template<typename T>
class Image
{
public:
	Image() = default;
	~Image() = default;
	Image(Image<T>&&) = default;
	Image(const Image<T>&) = default;

	Image<T>& operator=(const Image<T>&) = default;
	Image<T>& operator=(Image<T>&&) = default;

	Image(const std::string& path)
	{
		load(path);
	}

	Image(unsigned int w, unsigned int h, unsigned int c):
		m_width(w),
		m_height(h),
		m_components(c)
	{
		m_data.resize(w*h*c);
	}

	void load(const std::string& path)
	{
		if constexpr(std::is_same<T, float>::value)
		{
			ImageLoader::loadFloat(path, m_width, m_height, m_components, m_data);
		}
		else if constexpr(std::is_same<T, unsigned char>::value)
		{
			ImageLoader::loadUChar(path, m_width, m_height, m_components, m_data);
		}
		else if constexpr(std::is_same<T, unsigned short>::value)
		{
			ImageLoader::loadUShort(path, m_width, m_height, m_components, m_data);
		}
	}
	
	void save(const std::string& path)
	{
		if constexpr(std::is_same<T, float>::value)
		{
			ImageLoader::saveFloat(path, m_width, m_height, m_components, m_data);
		}
		else if constexpr(std::is_same<T, unsigned char>::value)
		{
			ImageLoader::saveUChar(path, m_width, m_height, m_components, m_data);
		}
		else if constexpr(std::is_same<T, unsigned short>::value)
		{
			ImageLoader::saveUShort(path, m_width, m_height, m_components, m_data);
		}
	}

	const std::vector<T>& getData() const { return m_data; }
	std::vector<T>& getData() { return m_data; }

	unsigned int getWidth() const { return m_width; }
	unsigned int getHeight() const { return m_height; }
	unsigned int getComponents() const { return m_components; }

	T const* get(unsigned int x, unsigned int y) const
	{
		return &m_data[(x+y*m_width)*m_components];
	}

	T* get(unsigned int x, unsigned int y)
	{
		return &m_data[(x+y*m_width)*m_components];
	}

	T& operator[](size_t idx) { return m_data[idx]; }
	const T& operator[](size_t idx) const { return m_data[idx]; }

	template<typename S>
	Image<T> operator+(const Image<S>& b) const
	{
		assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());
		Image<T> out(getWidth(), getHeight(), getComponents());

		#pragma omp parallel for
		for(int i = 0; i < m_data.size(); i++)
		{
//			out[i] = (*this)[i] + ColorToColor<S, T>(b[i]);
			out[i] = FloatToColor<T>(ColorToFloat((*this)[i]) + ColorToFloat(b[i]));
		}

		return out;
	}

	template<typename S>
	Image<T> operator-(const Image<S>& b) const
	{
		assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());
		Image<S> out(getWidth(), getHeight(), getComponents());

		#pragma omp parallel for
		for(int i = 0; i < m_data.size(); i++)
		{
			out[i] = FloatToColor<T>(ColorToFloat((*this)[i]) - ColorToFloat(b[i]));
		}

		return out;
	}

	template<typename S>
	Image<T> operator*(const Image<S>& b) const
	{
		assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());
		Image<S> out(getWidth(), getHeight(), getComponents());

		#pragma omp parallel for
		for(int i = 0; i < m_data.size(); i++)
		{
//			out[i] = (*this)[i] * ColorToColor<S, T>(b[i]);
			out[i] = FloatToColor<T>(ColorToFloat((*this)[i]) * ColorToFloat(b[i]));
		}

		return out;
	}

	template<typename S>
	Image<T> operator/(const Image<S>& b) const
	{
		assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());
		Image<S> out(getWidth(), getHeight(), getComponents());

		#pragma omp parallel for
		for(int i = 0; i < m_data.size(); i++)
		{
			out[i] = FloatToColor<T>(ColorToFloat((*this)[i]) / ColorToFloat(b[i]));
		}

		return out;
	}

	Image<T> operator-() const
	{
		static_assert(std::is_signed_v<T>, "Negative values are undefined for this image!");
		Image<T> out(getWidth(), getHeight(), getComponents());

		#pragma omp parallel for
		for(int i = 0; i < m_data.size(); i++)
		{
			out[i] = -(*this)[i];
		}

		return out;
	}

	template<typename Fn>
	auto transform(Fn&& fn) const
	{
		using P = std::invoke_result_t<Fn, T>;

		Image<P> result(m_width, m_height, m_components);
		#pragma omp parallel for
		for(int i = 0; i < m_data.size(); i++)
			result[i] = fn(m_data[i]);

		return result;
	}

private:
	unsigned int m_width = 0, m_height = 0;
	unsigned int m_components = 0;

	std::vector<T> m_data;
};

template<typename In, typename Out>
Image<Out> ConvertType(const Image<In>& img)
{
	const unsigned int comps = img.getComponents();
	Image<Out> out(img.getWidth(), img.getHeight(), comps);

	const auto& inData = img.getData();
	auto& outData = out.getData();

#pragma omp parallel for
	for(unsigned int idxIn = 0; idxIn < inData.size(); idxIn += comps)
	{
		for(int i = 0; i < comps; i++)
		{
			float val = ColorToFloat<In>(inData[idxIn + i]);

			if constexpr(std::is_same<Out, float>::value)
				outData[idxIn + i] = val;
			else
				outData[idxIn + i] = static_cast<Out>(val*std::numeric_limits<Out>::max());
		}
	}

	return out;
}

template<typename T>
Image<T> MakeGrayscale(const Image<T>& img, const float weights[4])
{
	const unsigned int comps = img.getComponents();
	Image<T> out(img.getWidth(), img.getHeight(), 1);

	const auto& inData = img.getData();
	auto& outData = out.getData();

#pragma omp parallel for
	for(unsigned int idx = 0; idx < outData.size(); idx++)
	{
		float sum = 0.0f;
		for(int i = 0; i < comps; i++)
		{
			sum += weights[i] * ColorToFloat<T>(inData[idx*comps + i]);
		}

		outData[idx] = FloatToColor<T>(sum / comps);
	}

	return out;
}

template<typename T>
Image<T> MakeGrayscale(const Image<T>& img)
{
	const float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
	return MakeGrayscale(img, w);
}

template<typename T>
Image<T> MakeRGB(const Image<T>& img)
{
	Image<T> out(img.getWidth(), img.getHeight(), 3);

#pragma omp parallel for
	for(int y = 0; y < img.getHeight(); y++)
	{
		const size_t yoff = y*img.getWidth();
		for(int x = 0; x < img.getWidth(); x++)
		{
			const size_t off = (yoff+x);
			const size_t rgbOff = off*3;
			out[rgbOff] = img[off];
			out[rgbOff + 1] = img[off];
			out[rgbOff + 2] = img[off];
		}
	}

	return out;
}

template<typename T>
Image<T> Sum(const Image<T>& r1, const Image<T>& r2)
{
	Image<T> out(r1.getWidth(), r1.getHeight(), r1.getComponents());
	for(unsigned int idx = 0; idx < out.getData().size(); idx++)
	{
		out.getData()[idx] = r1.getData()[idx] + r2.getData()[idx];
	}
	return out;
}

}
#endif

#ifndef __SYCL_IMAGE_H__
#define __SYCL_IMAGE_H__

#include <CL/sycl.hpp>
#include <cvpp/Image.h>

#include <type_traits>
namespace cvsycl
{

template<typename T>
class Image
{
public:
	Image(unsigned int w, unsigned int h, unsigned int c):
		m_data(cl::sycl::buffer<T>(cl::sycl::range<1>(w*h*c))),
		m_hostData(w*h*c),
		m_width(w),
		m_height(h),
		m_components(c)
		{}

	Image(const std::string& file):
		Image(cvpp::CPUImage<T>(file)) {}

	Image(const cvpp::CPUImage<T>& src):
		m_data(cl::sycl::buffer<T>(cl::sycl::range<1>(0))),
		m_hostData(src.getData()),
		m_width(src.getWidth()),
		m_height(src.getHeight()),
		m_components(src.getComponents())
	{
		m_data = cl::sycl::buffer<T>(m_hostData.data(), cl::sycl::range<1>(m_hostData.size()));
	}

	Image(Image<T>&&) = default;

	Image(const Image<T>& src):
		Image(src.getWidth(), src.getHeight(), src.getComponents())
	{

	}

	~Image() = default;
	Image<T>& operator=(const Image<T>& src) = default;

	Image<T>& operator=(const cvpp::CPUImage<T>& src)
	{
		m_data = cl::sycl::buffer<T>(src.getData().data(), cl::sycl::range<1>(src.getData().size()));
		m_width = src.getWidth();
		m_height = src.getHeight();
		m_components = src.getComponents();

		return *this;
	}

	operator cvpp::CPUImage<T>()
	{
		cvpp::CPUImage<T> out(m_width, m_height, m_components);

		auto acc = m_data.template get_access<cl::sycl::access::mode::read>();
		std::vector<T>& buf = out.getData();

		for(size_t i = 0; i < buf.size(); i++)
			buf[i] = acc[i];

		return out;
	}

	void save(const std::string& path)
	{
		auto acc = m_data.template get_access<cl::sycl::access::mode::read>();
		std::vector<T> buf(acc.get_count());

		for(size_t i = 0; i < buf.size(); i++)
			buf[i] = acc[i];

		if constexpr(std::is_same<T, float>::value)
		{
			cvpp::ImageLoader::saveFloat(path, m_width, m_height, m_components, buf);
		}
		else if constexpr(std::is_same<T, unsigned char>::value)
		{
			cvpp::ImageLoader::saveUChar(path, m_width, m_height, m_components, buf);
		}
		else if constexpr(std::is_same<T, unsigned short>::value)
		{
			cvpp::ImageLoader::saveUShort(path, m_width, m_height, m_components, buf);
		}
	}

	const cl::sycl::buffer<T>* getBuffer() const { return &m_data; }
	cl::sycl::buffer<T>* getBuffer() { return &m_data; }

	unsigned int getWidth() const { return m_width; }
	unsigned int getHeight() const { return m_height; }
	unsigned int getComponents() const { return m_components; }

	#define MAKE_OPERATOR(name, op)                                                                                                            \
		template <typename S, typename Q>                                                                                                      \
		class name##_kernel;                                                                                                                   \
		template <typename S>                                                                                                                  \
		Image<T> name(Image<S> &b, cl::sycl::queue &q)                                                                                         \
		{                                                                                                                                      \
			assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());                        \
			Image<T> out(getWidth(), getHeight(), getComponents());                                                                            \
																																			\
			using namespace cl::sycl;                                                                                                          \
																																			\
			q.submit([&](cl::sycl::handler &cgh) {                                                                                             \
				auto outAcc = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);                                          \
				auto aAcc = m_data.template get_access<access::mode::read>(cgh);                                                               \
				auto bAcc = b.getBuffer()->template get_access<access::mode::read>(cgh);                                                     \
																																			\
				cgh.parallel_for<name##_kernel<S, T>>(cl::sycl::range<1>(getWidth() * getHeight() * getComponents()), [=](cl::sycl::id<1> p) { \
					outAcc[p] = cvpp::FloatToColor<T>(cvpp::ColorToFloat(aAcc[p]) op cvpp::ColorToFloat(bAcc[p]));                             \
				});                                                                                                                            \
			}).wait();                                                                                                                         \
			return out;                                                                                                                        \
		}

		MAKE_OPERATOR(add, +)
		MAKE_OPERATOR(sub, -)
		MAKE_OPERATOR(mul, *)
		MAKE_OPERATOR(div, /)
	#undef MAKE_OPERATOR

	template<typename S> class neg_kernel;
	Image<T> neg(cl::sycl::queue& q)
	{
		static_assert(std::is_signed_v<T>, "Negative values are undefined for this image!");
		Image<T> out(getWidth(), getHeight(), getComponents());

		using namespace cl::sycl;

		q.submit([&](cl::sycl::handler& cgh) {
			auto outAcc = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);
			auto aAcc = m_data->get_access<access::mode::read>(cgh);

			cgh.parallel_for<neg_kernel<T>>(cl::sycl::range<1>(getWidth()*getHeight()*getComponents()), [=](cl::sycl::id<1> p)
			{
				outAcc[p] = cvpp::FloatToColor<T>(-cvpp::ColorToFloat(aAcc[p]));
			});
		}).wait();

		return out;
	}

	template<typename KernelName, typename Fn>
	auto transform(cl::sycl::queue& q, Fn&& fn)
	{
		using P = std::invoke_result_t<Fn, T>;

		Image<P> result(m_width, m_height, m_components);
		q.submit([&, this](cl::sycl::handler& cgh) {
			auto outAcc = result.getBuffer()->template get_access<cl::sycl::access::mode::discard_write>(cgh);
			auto inAcc = m_data.template get_access<cl::sycl::access::mode::read>(cgh);
			cgh.parallel_for<KernelName>(cl::sycl::range<1>(getWidth()*getHeight()*getComponents()), [=](cl::sycl::id<1> p)
			{
				outAcc[p] = fn(inAcc[p]);
			});
		}).wait();

		return result;
	}

private:
	cl::sycl::buffer<T> m_data;
	std::vector<T> m_hostData;
	unsigned int m_width = 0, m_height = 0, m_components = 0;
};

template<typename A, typename B> class convert_type_kernel;

template<typename In, typename Out>
Image<Out> ConvertType(Image<In>& img, cl::sycl::queue& queue)
{
	const unsigned int comps = img.getComponents();
	Image<Out> out(img.getWidth(), img.getHeight(), comps);
	using namespace cl::sycl;

	queue.submit([&](cl::sycl::handler& cgh) {
		auto outAcc = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);
		auto inAcc = img.getBuffer()->template get_access<access::mode::read>(cgh);

		cgh.parallel_for<convert_type_kernel<In, Out>>(cl::sycl::range<1>(img.getWidth()*img.getHeight()), [=](cl::sycl::id<1> p)
		{
			auto idxIn = p * comps;
			for(int i = 0; i < comps; i++)
			{
				float val = cvpp::ColorToFloat<In>(inAcc[idxIn + i]);

				if constexpr(std::is_same<Out, float>::value)
					outAcc[idxIn + i] = val;
				else
					outAcc[idxIn + i] = static_cast<Out>(val*std::numeric_limits<Out>::max());
			}
		});
	}).wait();

	return out;
}

template<typename T> class make_grayscale_kernel;
template<typename T>
Image<T> MakeGrayscale(Image<T>& img, const cl::sycl::float4 weights, cl::sycl::queue& queue)
{
	const unsigned int comps = img.getComponents();
	Image<T> out(img.getWidth(), img.getHeight(), 1);
	
	using namespace cl::sycl;

	queue.submit([&](cl::sycl::handler& cgh) {
		auto outData = out.getBuffer()->template get_access<access::mode::discard_write>(cgh);
		auto inData = img.getBuffer()->template get_access<access::mode::read>(cgh);
		//auto weightAcc = weightBuf.get_access<access::mode::read, access::target::constant_buffer>(cgh);

		cgh.parallel_for<make_grayscale_kernel<T>>(cl::sycl::range<1>(img.getWidth()*img.getHeight()), [=](cl::sycl::id<1> idx)
		{
			float sum = 0.0f;
			for(int i = 0; i < comps; i++)
			{
				sum += weights[i] * cvpp::ColorToFloat<T>(inData[idx*comps + i]);
			}

			outData[idx] = cvpp::FloatToColor<T>(sum / (weights[0] + weights[1] + weights[2] + weights[3]));
		});
	}).wait();

	return out;
}

template<typename T>
Image<T> MakeGrayscale(Image<T>& img, cl::sycl::queue& queue)
{
	const cl::sycl::float4 w {1.0f, 1.0f, 1.0f, img.getComponents() >= 4 ? 1.0f : 0.0f};
	return MakeGrayscale(img, w, queue);
}

}

#endif

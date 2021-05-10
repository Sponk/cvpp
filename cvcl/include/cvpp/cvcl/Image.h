#ifndef __CLIMAGE_H__
#define __CLIMAGE_H__

#include <cvpp/Image.h>
#include "Queue.h"

#include <iostream>

namespace cvcl
{

template<typename T>
class Image
{
public:
	Image(unsigned int w, unsigned int h, unsigned int c, Queue& q)
	{
		auto ctx = q.getCtx();
		m_image = cl::Image2D(ctx, CL_MEM_READ_ONLY, getFormat<T>(c), w, h);
	}

	Image(const std::string& file, Queue& q):
		Image(cvpp::Image<T>(file), q) {}

	Image(const cvpp::Image<T>& src, Queue& q)
	{
		set(src, q);
	}

	operator cl::Image2D()
	{
		return m_image;
	}

	template<typename Q>
	void set(const cvpp::Image<Q>& src, Queue& q)
	{
		auto w = src.getWidth();
		auto h = src.getHeight();
		auto c = src.getComponents();

		auto& ctx = q.getCtx();

		if(c != 3)
		{
			m_image = cl::Image2D(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, getFormat<T>(src.getComponents()), src.getWidth(), src.getHeight(), 0, (void*) src.getData().data());
			return;
		}

		cl::Buffer buf(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, src.getData().size()*sizeof(Q), (void*) src.getData().data());

		m_image = cl::Image2D(ctx, CL_MEM_READ_ONLY, getFormat<T>(c), w, h);

		constexpr auto __krnl_name = cs("CopyImage_") + GetTypeStr<Q>();
		constexpr auto __krnl_types = MakeTypeDefine<Q>(cs("Q"))
									+ cs("#define KernelName ") + __krnl_name + cs("\n");
									
		constexpr auto __krnl = __krnl_types + cs(R"CLC(
			// TODO: Sampler as template argument!
			__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
			__kernel void KernelName(__global Q* in, unsigned int w, unsigned int h, unsigned int c, __write_only image2d_t out, const Q maxValue)
			{
				const int2 pos = {get_global_id(0), get_global_id(1)};
				__global Q* ptr = in + (((pos.y*w) + pos.x) * c);
				
				float4 px = (float4)(
							(float) *ptr / maxValue,
							c >= 2 ? (float) ptr[1]/maxValue : 0.0f,
							c >= 3 ? (float) ptr[2]/maxValue : 0.0f,
							c >= 4 ? (float) ptr[4]/maxValue : 1.0f);

				write_imagef(out, pos, px);
			}
		)CLC");

		q.addProgram(uuid(__krnl), __krnl.c_str());
		q(w, h, __krnl_name.c_str(), buf, w, h, c, m_image, std::numeric_limits<Q>::max());
	}

	cl::Image2D& getImage() { return m_image; }

	unsigned int getWidth() const { return m_image.getImageInfo<CL_IMAGE_WIDTH>(); }
	unsigned int getHeight() const { return m_image.getImageInfo<CL_IMAGE_HEIGHT>(); }
	unsigned int getComponents() const
	{
		switch(m_image.getImageInfo<CL_IMAGE_FORMAT>().image_channel_order)
		{
			case CL_R: return 1;
			case CL_RG: return 2;

			default:
			case CL_RGBA: return 4;
		}
	}

	void save(const std::string& path, Queue& q)
	{
		auto w = getWidth();
		auto h = getHeight();
		auto c = getComponents();
		
		std::vector<T> buf(w*h*c);

		auto& cq = q.getQueue();
		cq.enqueueReadImage(m_image, true, {0, 0, 0}, {w, h, 1}, 0, 0, buf.data(), nullptr, nullptr);

		if constexpr(std::is_same<T, float>::value)
		{
			cvpp::ImageLoader::saveFloat(path, w, h, c, buf);
		}
		else if constexpr(std::is_same<T, unsigned char>::value)
		{
			cvpp::ImageLoader::saveUChar(path, w, h, c, buf);
		}
		else if constexpr(std::is_same<T, unsigned short>::value)
		{
			cvpp::ImageLoader::saveUShort(path, w, h, c, buf);
		}
	}

	template <typename S>
	Image<T> add(Image<S> &b, Queue &q)
	{
		assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());
		Image<T> out(getWidth(), getHeight(), getComponents(), q);

		constexpr auto __krnl = cs(R"CLC(
			// TODO: Sampler as template argument!
			__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
			__kernel void AddImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t out)
			{
				const int2 pos = {get_global_id(0), get_global_id(1)};
				float4 sum = read_imagef(a, sampler, pos) + read_imagef(b, sampler, pos);
				write_imagef(out, pos, sum);
			}
		)CLC");
		
		q.addProgram(uuid(__krnl), __krnl.c_str());
		q(getWidth(), getHeight(), "AddImage", m_image, b.getImage(), out.getImage());
		return out;
	}

	template <typename S>
	Image<T> sub(Image<S> &b, Queue &q)
	{
		assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());
		Image<T> out(getWidth(), getHeight(), getComponents(), q);

		constexpr auto __krnl = cs(R"CLC(
			// TODO: Sampler as template argument!
			__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
			__kernel void SubImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t out)
			{
				const int2 pos = {get_global_id(0), get_global_id(1)};
				float4 sum = read_imagef(a, sampler, pos) - read_imagef(b, sampler, pos);
				write_imagef(out, pos, sum);
			}
		)CLC");

		q.addProgram(uuid(__krnl), __krnl.c_str());
		q(getWidth(), getHeight(), "SubImage", m_image, b.getImage(), out.getImage());
		return out;
	}

	template <typename S>
	Image<T> mul(Image<S> &b, Queue &q)
	{
		assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());
		Image<T> out(getWidth(), getHeight(), getComponents(), q);

		constexpr auto __krnl = cs(R"CLC(
			// TODO: Sampler as template argument!
			__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
			__kernel void MulImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t out)
			{
				const int2 pos = {get_global_id(0), get_global_id(1)};
				float4 sum = read_imagef(a, sampler, pos) * read_imagef(b, sampler, pos);
				write_imagef(out, pos, sum);
			}
		)CLC");

		q.addProgram(uuid(__krnl), __krnl.c_str());
		q(getWidth(), getHeight(), "MulImage", m_image, b.getImage(), out.getImage());
		return out;
	}

	template <typename S>
	Image<T> div(Image<S> &b, Queue &q)
	{
		assert(getWidth() == b.getWidth() && getHeight() == b.getHeight() && getComponents() == b.getComponents());
		Image<T> out(getWidth(), getHeight(), getComponents(), q);

		constexpr auto __krnl = cs(R"CLC(
			// TODO: Sampler as template argument!
			__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
			__kernel void DivImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t out)
			{
				const int2 pos = {get_global_id(0), get_global_id(1)};
				float4 sum = read_imagef(a, sampler, pos) / read_imagef(b, sampler, pos);
				write_imagef(out, pos, sum);
			}
		)CLC");

		q.addProgram(uuid(__krnl), __krnl.c_str());
		q(getWidth(), getHeight(), "DivImage", m_image, b.getImage(), out.getImage());
		return out;
	}

	template <typename S>
	Image<T> neg(Queue& q)
	{
		Image<T> out(getWidth(), getHeight(), getComponents(), q);

		constexpr auto __krnl = cs(R"CLC(
			// TODO: Sampler as template argument!
			__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
			__kernel void NegImage(__read_only image2d_t a, __write_only image2d_t out)
			{
				const int2 pos = {get_global_id(0), get_global_id(1)};
				float4 sum = -read_imagef(a, sampler, pos);
				write_imagef(out, pos, sum);
			}
		)CLC");

		q.addProgram(uuid(__krnl), __krnl.c_str());
		q(getWidth(), getHeight(), "NegImage", m_image, out.getImage());
		return out;
	}

	cvpp::Image<T> toImage(Queue& q) const
	{
		auto w = getWidth();
		auto h = getHeight();
		auto c = getComponents();

		cvpp::Image<T> out(w, h, c);

		auto& cq = q.getQueue();
		cq.enqueueReadImage(m_image, true, {0, 0, 0}, {w, h, 1}, 0, 0, out.getData().data(), nullptr, nullptr);

		return out;
	}

private:
	cl::Image2D m_image;

	template<typename Q>
	auto getFormat(unsigned int channels)
	{
		cl::ImageFormat format;
		
		switch(channels)
		{
			case 1:
				format.image_channel_order = CL_R;
			break;

			case 2:
				format.image_channel_order = CL_RG;
			break;

			case 3:
				format.image_channel_order = CL_RGBA;
			break;

			case 4:
				format.image_channel_order = CL_RGBA;
			break;
			
		}

		if constexpr(std::is_same_v<Q, uint8_t>)
		{
			format.image_channel_data_type = CL_UNORM_INT8;
		}
		else if constexpr(std::is_same_v<Q, uint16_t>)
		{
			format.image_channel_data_type = CL_UNORM_INT16;
		}
		else if constexpr(std::is_same_v<Q, float>)
		{
			format.image_channel_data_type = CL_FLOAT;
		}

		return format;
	}
};

template<typename T>
Image<T> MakeGrayscale(Image<T>& img, const cl_float4& weights, Queue& queue)
{
	const unsigned int comps = img.getComponents();
	Image<T> out(img.getWidth(), img.getHeight(), 1, queue);

	constexpr auto __krnl_types = MakeTypeDefine<T>(cs("T"));
	constexpr auto __krnl = __krnl_types + cs(R"CLC(
		// TODO: Sampler as template argument!
		__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
		__kernel void MakeGrayscale(__read_only image2d_t in, __write_only image2d_t out, float4 w)
		{
			const int2 pos = {get_global_id(0), get_global_id(1)};
			const float4 s = read_imagef(in, sampler, pos);
			write_imagef(out, pos, (s.x*w.x + s.y*w.y + s.z*w.z) / (w.x + w.y + w.z + w.w));
		}
	)CLC");

	queue.addProgram(uuid(__krnl), __krnl.c_str());

	queue(img.getWidth(), img.getHeight(), "MakeGrayscale", img.getImage(), out.getImage(), weights);

	return out;
}

template<typename T>
Image<T> MakeGrayscale(Image<T>& img, Queue& queue)
{
	const cl_float4 w = {1.0f, 1.0f, 1.0f, (img.getComponents() == 4 ? 1.0f : 0.0f)};
	return MakeGrayscale(img, w, queue);
}

template<typename T>
Image<T> MakeRGBA(const Image<T>& img, Queue& q)
{
	Image<T> out(img.getWidth(), img.getHeight(), 4, q);

	cl::Event event;
	auto& cq = q.getQueue();
	cq.enqueueCopyImage(img.getImage(), out.getImage(),
		{0, 0, 0}, {0, 0, 0},
		{img.getWidth(), img.getHeight(), 0}, nullptr, &event);

	return out;
}

}

#endif

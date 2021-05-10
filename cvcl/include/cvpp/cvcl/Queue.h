#ifndef __QUEUE_H__
#define __QUEUE_H__

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <vector>
#include <string>

#include "ConstexprString.h"

namespace cvcl
{

template<typename Fn>
uintptr_t uuid(Fn& f)
{
	return reinterpret_cast<uintptr_t>(&f);
}

template <size_type Size>
constexpr ConstexprString<Size-1> cs(char const (&src) [Size])
{
	return ConstexprString<Size-1>(src);
}

template<typename T>
constexpr auto GetTypeStr()
{
	if constexpr(std::is_same_v<T, float>)
	{
		return MakeConstexprString("float");
	}
	else if constexpr(std::is_same_v<T, int>)
	{
		return MakeConstexprString("int");
	}
	else if constexpr(std::is_same_v<T, unsigned int>)
	{
		return MakeConstexprString("uint");
	}
	else if constexpr(std::is_same_v<T, short>)
	{
		return MakeConstexprString("short");
	}
	else if constexpr(std::is_same_v<T, unsigned short>)
	{
		return MakeConstexprString("ushort");
	}
	else if constexpr(std::is_same_v<T, char>)
	{
		return MakeConstexprString("char");
	}
	else if constexpr(std::is_same_v<T, unsigned char>)
	{
		return MakeConstexprString("uchar");
	}
	else
	{
		// Trick from here
		// https://stackoverflow.com/questions/38304847/constexpr-if-and-static-assert
		[]<bool flag = false>() {static_assert(flag, "Unknown type given!");}();
	}
}

template<typename T, size_type Size>
constexpr auto MakeTypeDefine(ConstexprString<Size> Type)
{
	return cs("#define ") + Type + cs(" ") + GetTypeStr<T>() + cs("\n");
}

class Queue
{
public:
	static Queue QueueGPU();
	static Queue QueueCPU();
	static Queue QueueDefault();

	Queue(cl::Platform platform, cl::Device dev);

	void printInfo();
	void addProgram(uintptr_t uuid, const char* src, const char* prefix = nullptr);

	template<typename... Args>
	cl::Event operator()(const cl::EnqueueArgs& eqargs, const std::string& kernel, Args... args)
	{
		auto krnl = m_kernels.find(kernel);
		if(krnl == m_kernels.end())
			throw std::runtime_error("Unknown kernel: " + kernel);

		cl_int err;
		auto fn = cl::KernelFunctor<Args...>(krnl->second);
		return fn(eqargs, std::forward<Args>(args)... , err);
	}

	template<typename... Args>
	cl::Event operator()(unsigned int sizeX, const std::string& kernel, Args... args)
	{
		return operator()(cl::EnqueueArgs(m_queue, cl::NDRange(sizeX)), kernel, std::forward<Args>(args)...);
	}

	template<typename... Args>
	cl::Event operator()(unsigned int sizeX, unsigned int sizeY, const std::string& kernel, Args&&... args)
	{
		return operator()(cl::EnqueueArgs(m_queue, cl::NDRange(sizeX, sizeY)), kernel, std::forward<Args>(args)...);
	}

	template<typename... Args>
	cl::Event operator()(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ, const std::string& kernel, Args&&... args)
	{
		return operator()(cl::EnqueueArgs(m_queue, cl::NDRange(sizeX, sizeY, sizeZ)), kernel, std::forward<Args>(args)...);
	}

	cl::Context& getCtx() { return m_ctx; }
	cl::CommandQueue& getQueue() { return m_queue; }

	void flush() { m_queue.flush(); }

private:
	cl::Platform m_platform;
	cl::Device m_device;
	cl::Context m_ctx;
	cl::CommandQueue m_queue;

	std::unordered_map<uintptr_t, cl::Program> m_programs;
	std::unordered_map<std::string, cl::Kernel> m_kernels;
};
}
#endif

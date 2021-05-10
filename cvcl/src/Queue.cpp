#include <cvpp/cvcl/Queue.h>
#include <iostream>

using namespace cvcl;

Queue Queue::QueueGPU()
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	
	cl::Platform::get(&platforms);

	for(const auto& p : platforms)
	{
		p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		if(!devices.empty())
		{
			return Queue(p, devices.front());
		}
	}

	throw std::runtime_error("No GPU device found!");
}

Queue Queue::QueueCPU()
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	
	cl::Platform::get(&platforms);

	for(const auto& p : platforms)
	{
		p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
		if(!devices.empty())
		{
			return Queue(p, devices.front());
		}
	}

	throw std::runtime_error("No CPU device found!");
}

Queue Queue::QueueDefault()
{
	std::vector<cl::Device> devices;
	auto platform = cl::Platform::getDefault();
	platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

	if(devices.empty())
		throw std::runtime_error("No default device found!");

	return Queue(platform, devices.front());
}

Queue::Queue(cl::Platform platform, cl::Device dev):
	m_device(dev),
	m_platform(platform)
{
	cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform(), 0};
	m_ctx = cl::Context(dev, properties);
	m_queue = cl::CommandQueue(m_ctx, m_device, cl::QueueProperties::None);
}

void Queue::addProgram(uintptr_t uuid, const char* src, const char* prefix)
{
	if(m_programs.contains(uuid))
		return;
		
	cl::Program prog(m_ctx, prefix ? std::string(prefix) + src : src);

	try
	{
		prog.build();
	}
	catch (...)
	{
		cl_int buildErr = CL_SUCCESS;
		auto buildInfo = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
		for (auto &pair : buildInfo)
		{
			std::cerr << pair.second << std::endl << std::endl;
		}

		throw std::runtime_error("Could not build program!");
	}
	
	std::vector<cl::Kernel> kernels;
	prog.createKernels(&kernels);

	for(const auto& k : kernels)
	{
		std::cout << "Loaded kernel " << k.getInfo<CL_KERNEL_FUNCTION_NAME>() << std::endl;
		m_kernels[k.getInfo<CL_KERNEL_FUNCTION_NAME>()] = k;
	}

	m_programs[uuid] = prog;
}

void Queue::printInfo()
{
	std::cout << "Vendor: " << m_device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
	std::cout << "Device: " << m_device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Memory: " << m_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024.0f / 1024.0f / 1024.0f << " GB" << std::endl;
}

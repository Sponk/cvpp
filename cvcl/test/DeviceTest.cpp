#include <gtest/gtest.h>
#include <cvpp/cvcl/Queue.h>
#include <cvpp/cvcl/Image.h>
#include <cvpp/cvcl/Convolution.h>
#include <cvpp/cvcl/StructureTensor.h>
#include <cvpp/cvcl/HarrisDetector.h>

#include <cvpp/CommonFilters.h>

using namespace cvcl;

constexpr const char* __kernel = { R"CLC(
	__kernel void kernelA(int a)
	{
		printf("a + 1 = %d\n", a + 1);
	}

	__kernel void kernelB()
	{
		printf("Hello World!");
	}
)CLC"};

TEST(OpenCL, Queue)
{
	Queue q = Queue::QueueGPU();
	q.printInfo();

	q.addProgram(cvcl::uuid(__kernel), __kernel);

	q(1, "kernelA", 32).wait();
	q(1, "kernelB").wait();
}

constexpr const char* __image_kernel = { R"CLC(
	__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	__kernel void kernelA(__read_only image2d_t in, __write_only image2d_t out)
	{
		const int2 pos = {get_global_id(0), get_global_id(1)};
		float4 px = read_imagef(in, sampler, pos);
		write_imagef(out, pos, px);
	}
)CLC"};

TEST(OpenCL, Image)
{
	Queue q = Queue::QueueGPU();
	q.printInfo();
	q.addProgram(cvcl::uuid(__image_kernel), __image_kernel);

	cvcl::Image<uint8_t> img("test1.png", q);
	cvcl::Image<uint8_t> img2(img.getWidth(), img.getHeight(), img.getComponents(), q);

	q(img.getWidth(), img.getHeight(), "kernelA", img.getImage(), img2.getImage()).wait();
	img2.save("cvcltest.png", q);
}

TEST(OpenCL, Sobel)
{
	Queue q = Queue::QueueGPU();
	q.printInfo();

	cvcl::Image<uint8_t> img("test1.png", q);

	auto gr = cvcl::MakeGrayscale(img, q);

	gr.save("grayscale_cl.png", q);

	gr = Convolute2D(gr, cvpp::SobelFilterH(), q);
	gr.save("test1_sobel_x_cl.png", q);
}

TEST(OpenCL, StackBlur)
{
	Queue q = Queue::QueueGPU();
	q.printInfo();

	cvcl::Image<uint8_t> img("test1.png", q);

	constexpr int sz = 15;
	auto out = cvcl::ConvoluteSeparable(img, cvpp::StackBlurFilter<sz>(), q);

	out.save("test1_stackblur_cl.png", q);
}

TEST(OpenCL, StructureTensor)
{
	Queue q = Queue::QueueGPU();
	q.printInfo();

	cvcl::Image<unsigned char> img("test1.png", q);
	auto tensor = cvcl::StructureTensor(img, q);
	//auto tensor1 = cvcl::HessianTensor(img, q);

	/*auto out = tensor1.transform<class MatrixDeterminantKernel>(q, [](const auto& mtx) {
		return mtx.determinant();
	});*/

	//auto saveImg = cvsycl::ConvertType<float, uint8_t>(out, q);
	tensor.save("cl_determinant.png", q);
}

TEST(OpenCL, HarrisFeatures)
{
	Queue q = Queue::QueueGPU();
	q.printInfo();

	cvpp::Image<uint8_t> img("test1.png");
	Image<uint8_t> gpuImg(img, q);

	//gpuImg = ConvoluteSeparable(gpuImg, cvpp::StackBlurFilter<3>(), q);

	std::vector<cvpp::Feature> features;
	HarrisDetector(gpuImg, 11, 1.0f, features, q);

	cvpp::MarkFeatures(img, Eigen::Vector4f(1, 0, 1, 1), features);
	img.save("OpenCLHarrisFeatures.png");
	std::cout << "FEATURES: " << features.size() << std::endl;
	
}

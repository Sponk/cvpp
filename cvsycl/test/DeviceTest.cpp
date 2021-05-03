#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <CL/sycl.hpp>

#include <cvpp/cvsycl/Image.h>
#include <cvpp/cvsycl/Sampler.h>
#include <cvpp/cvsycl/Convolution.h>
#include <cvpp/CommonFilters.h>

#include <cvpp/cvsycl/StructureTensor.h>

class kernel_name;

// https://techdecoded.intel.io/resources/programming-data-parallel-c/#gs.mflkxv
#if 0
TEST(SYCL, Image)
{
	//cl::sycl::queue q(cl::sycl::gpu_selector{});
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvsycl::Image<unsigned char> img("test1.png");
	auto img1 = cvsycl::MakeGrayscale(img, q);
	std::cout << "Done Grayscale" << std::endl;
	auto img2 = cvsycl::ConvoluteSeparable<cvsycl::ClampView<unsigned char>>(img1, cvpp::GaussFilter<31>(5.0f), q);

	std::cout << "Done convolution" << std::endl;

	img2.save("testsycl.png");
}
#endif

TEST(SYCL, StructureTensor)
{
	//cl::sycl::queue q(cl::sycl::gpu_selector{});
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvsycl::Image<unsigned char> img("test1.png");
	auto tensor = cvsycl::StructureTensor(img, q);
	auto tensor1 = cvsycl::HessianTensor(img, q);

	auto out = tensor1.transform<class MatrixDeterminantKernel>(q, [](const auto& mtx) {
		return mtx.determinant();
	});

	auto saveImg = cvsycl::ConvertType<float, uint8_t>(out, q);
	saveImg.save("sycl_determinant.png");
}


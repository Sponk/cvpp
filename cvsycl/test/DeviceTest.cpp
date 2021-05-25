#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <CL/sycl.hpp>

#include <cvpp/cvsycl/Image.h>
#include <cvpp/cvsycl/Sampler.h>
#include <cvpp/cvsycl/Convolution.h>
#include <cvpp/CommonFilters.h>

#include <cvpp/cvsycl/StructureTensor.h>
#include <cvpp/cvsycl/HarrisDetector.h>

#define TESTIMG "../test1.png"

using namespace cvsycl;

class kernel_name;

#if 0
// https://techdecoded.intel.io/resources/programming-data-parallel-c/#gs.mflkxv
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

TEST(SYCL, Sobel)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img("test1.png");
	auto gr = cvsycl::ConvertType<uint8_t, float>(img, q);

	gr = cvsycl::MakeGrayscale(gr, q);
	gr = cvsycl::Convolute2D<cvsycl::ClampView<float>>(gr, cvpp::SobelFilterH(), q);
	
	cvsycl::ConvertType<float, uint8_t>(gr, q).save("test1_sobel_x_sycl.png");
}
#endif

TEST(Image, Add)
{
	//cl::sycl::queue q(cl::sycl::gpu_selector{});
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvsycl::Image<unsigned char> img(TESTIMG);
	auto fimage = cvsycl::ConvertType<uint8_t, float>(img, q);

	fimage = fimage.add(fimage, q);

	cvsycl::ConvertType<float, uint8_t>(fimage, q).save("ImageAdd.png");
}

#if 1
TEST(Detector, StructureTensor)
{
	//cl::sycl::queue q(cl::sycl::gpu_selector{});
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvsycl::Image<unsigned char> img(TESTIMG);
	auto tensor = cvsycl::StructureTensor(img, q);
	auto tensor1 = cvsycl::HessianTensor(img, q);

	auto out = tensor1.transform<class MatrixDeterminantKernel>(q, [](const auto& mtx) {
		return mtx.determinant() > 0.0f ? 1.0f : 0.0f;
	});

	auto saveImg = cvsycl::ConvertType<float, uint8_t>(out, q);
	saveImg.save("DetectorStructureTensor.png");
}
#endif

TEST(Detector, Harris)
{
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvpp::CPUImage<uint8_t> img(TESTIMG);
	cvsycl::Image<uint8_t> gpuImg(img);

	std::vector<cvpp::Feature> features;
	cvsycl::HarrisDetector(gpuImg, 11, 1.0f, features, q);

	cvpp::MarkFeatures(img, Eigen::Vector4f(1, 0, 1, 1), features);
	img.save("DetectorHarris.png");
	std::cout << "FEATURES: " << features.size() << std::endl;
}

TEST(Image, LoadSave)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img(TESTIMG);

	img.save("ImageLoadSave.png");
	img.save("ImageLoadSave.jpg");
	img.save("ImageLoadSave.tga");
	
	EXPECT_THROW(img.save("ImageLoadSave.hdr"), std::runtime_error);
}

TEST(Image, ConvertType)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img(TESTIMG);

	auto shortImg = ConvertType<uint8_t, float>(img, q);
	shortImg.save("ImageConvertType.hdr");
}

TEST(Convolution, Conv1D)
{
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvsycl::Image<uint8_t> img(TESTIMG);
	auto grey = MakeGrayscale(img, q);

	Eigen::Vector3f krnl;
	krnl << -1, 0, 1;

	auto out = Convolute1D<cvsycl::ClampView<uint8_t>, cvsycl::HORIZONTAL>(img, krnl, q);
	auto out2 = Convolute1D<cvsycl::ClampView<uint8_t>, cvsycl::VERTICAL>(img, krnl, q);

	out.save("ConvolutionConv1DHorizontal.png");
	out2.save("ConvolutionConv1DVertical.png");
}

TEST(Convolution, Conv2D)
{
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvsycl::Image<uint8_t> img(TESTIMG);

	Eigen::Matrix3f mtx;
	mtx.fill(1.0f/9.0f);

	auto out = Convolute2D<cvsycl::ClampView<uint8_t>>(img, mtx, q);
	out.save("ConvolutionConv2D.png");
}

TEST(Convolution, Conv2DMat3)
{
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvsycl::Image<uint8_t> img(TESTIMG);

	Eigen::Matrix3f mtx;
	mtx.fill(1.0f/9.0f);

	auto out = Convolute2D<cvsycl::ClampView<uint8_t>>(img, mtx, q);

	out.save("ConvolutionConv2DMat3.png");
}

TEST(Convolution, ConvSep)
{
	cl::sycl::queue q(cl::sycl::default_selector{});

	cvsycl::Image<uint8_t> img(TESTIMG);
	auto out = ConvoluteSeparable<cvsycl::ClampView<uint8_t>>(img, cvpp::BoxFilter<5>(), q);

	out.save("ConvolutionConvSep.png");
}

TEST(Convolution, GaussFilter)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img(TESTIMG);
	auto out = ConvoluteSeparable<cvsycl::ClampView<uint8_t>>(img, cvpp::GaussFilter<11>(2.0f), q);

	out.save("ConvolutionGaussFilter.png");
}

TEST(Convolution, StackBlur)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img(TESTIMG);

	constexpr int sz = 15;
	auto out = ConvoluteSeparable<cvsycl::ClampView<uint8_t>>(img, cvpp::StackBlurFilter<sz>(), q);

	out.save("ConvolutionStackBlur.png");
}

TEST(Convolution, BlurAndEdge)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img(TESTIMG);

	img = MakeGrayscale(img, q);
	img = ConvoluteSeparable<cvsycl::ClampView<uint8_t>>(img, cvpp::StackBlurFilter<9>(), q);
	img = Convolute1D<cvsycl::ClampView<uint8_t>, cvsycl::HORIZONTAL>(img, cvpp::SimpleEdgeDetector, q);

	img.save("ConvolutionBlurAndEdge.png");
}

TEST(Convert, Grayscale)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img(TESTIMG);
	auto gr = MakeGrayscale(img, q);

	gr.save("ConvertGrayscale.png");
}

TEST(Convolution, Sobel)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img(TESTIMG);
	auto gr = MakeGrayscale(img, q);
	gr = cvsycl::Convolute2D<cvsycl::ClampView<uint8_t>>(gr, cvpp::SobelFilterH(), q);
	gr.save("ConvolutionSobelH.png");

	gr = cvsycl::Convolute2D<cvsycl::ClampView<uint8_t>>(gr, cvpp::SobelFilterV(), q);
	gr.save("ConvolutionSobelV.png");
}

#if 0
TEST(Detector, Hessian)
{
	cvsycl::Image<uint8_t> img(TESTIMG);

	std::vector<cvpp::Feature> features;
	cvpp::HessianDetector(img, 21, 0.5f, features);

	cvpp::MarkFeatures(img, Eigen::Vector4f(1, 0, 1, 1), features);
	img.save("DetectorHessian.png");
	std::cout << "FEATURES: " << features.size() << std::endl;
}
#endif

TEST(Convolution, Laplace)
{
	cl::sycl::queue q(cl::sycl::default_selector{});
	cvsycl::Image<uint8_t> img(TESTIMG);
	auto gr = cvsycl::MakeGrayscale(img, q);

	auto Dx = cvsycl::Convolute1D<cvsycl::ClampView<uint8_t>, cvsycl::HORIZONTAL>(gr, cvpp::LaplaceFilterX, q);
	auto Dy = cvsycl::Convolute1D<cvsycl::ClampView<uint8_t>, cvsycl::VERTICAL>(gr, cvpp::LaplaceFilterX, q);
	auto Dxy = cvsycl::Convolute2D<cvsycl::ClampView<uint8_t>>(gr, cvpp::LaplaceFilterXY(), q);

	Dx.save("ConvolutionLaplaceX.png");
	Dy.save("ConvolutionLaplaceY.png");
	Dxy.save("ConvolutionLaplaceXY.png");
}

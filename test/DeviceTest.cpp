#include <gtest/gtest.h>
#include <cvpp/Image.h>
#include <cvpp/Sampler.h>
#include <cvpp/Convolution.h>
#include <cvpp/CommonFilters.h>
#include <cvpp/StructureTensor.h>
#include <cvpp/HarrisDetector.h>

#include <Eigen/Dense>

TEST(Image, Load)
{
	cvpp::Image<uint8_t> img("test1.png");

	img.save("testsave.png");
	img.save("testsave.jpg");
	img.save("testsave.tga");
	
	EXPECT_THROW(img.save("testsave.hdr"), std::runtime_error);
}

TEST(Image, ConvertType)
{
	cvpp::Image<uint8_t> img("test1.png");

	auto shortImg = ConvertType<uint8_t, float>(img);
	shortImg.save("testsave.hdr");
}

TEST(Sampler, ClampSampler)
{
	cvpp::Image<uint8_t> img("test1.png");
	auto sampler = cvpp::ClampView(img);

	auto c1 = sampler.sample(0.0, 0.0);
	auto c2 = sampler.sample(1.0, 1.0);
	
	auto c3 = sampler.sample(-0.5, -0.5);
	auto c4 = sampler.sample(1.1, 1.1);
	
	EXPECT_LT((c1-c3).norm(), 0.0001f);
	EXPECT_LT((c2-c4).norm(), 0.0001f);
}

TEST(Utils, Mod)
{
	EXPECT_EQ(cvpp::mod(-5, 2), cvpp::mod(5, 2));
	EXPECT_EQ(cvpp::mod(-15, 20), cvpp::mod(5, 20));
}

TEST(Convolution, Conv1D)
{
	cvpp::Image<uint8_t> img("test1.png");
	auto grey = cvpp::MakeGrayscale(img);
	cvpp::ClampView sampler(grey);

	Eigen::VectorXf krnl(3);
	krnl << -0.5, 0, 0.5;

	auto out = cvpp::Convolute1D(sampler, krnl);
	auto out2 = cvpp::Convolute1D<cvpp::VERTICAL>(sampler, krnl);

	out.save("test1_convolved_1d_hor.png");
	out2.save("test1_convolved_1d_vert.png");
}

TEST(Convolution, Conv2D)
{
	cvpp::Image<uint8_t> img("test1.png");
	cvpp::ClampView sampler(img);

	Eigen::MatrixXf mtx(7, 7);
	mtx.fill(1.0f/(7.0f*7.0f));

	auto out = cvpp::Convolute2D(sampler, mtx);

	out.save("test1_conv2d.png");
}

TEST(Convolution, Conv2DMat3)
{
	cvpp::Image<uint8_t> img("test1.png");
	cvpp::ClampView sampler(img);

	Eigen::Matrix3f mtx;
	mtx.fill(1.0f/9.0f);

	auto out = cvpp::Convolute2D(sampler, mtx);

	out.save("test1_conv2d_mat3.png");
}

TEST(Convolution, ConvSep)
{
	cvpp::Image<uint8_t> img("test1.png");
	cvpp::ClampView sampler(img);

	auto out = cvpp::ConvoluteSeparable(sampler, cvpp::BoxFilter<5>());

	out.save("test1_convsep.png");
}

TEST(Convolution, GaussFilter)
{
	cvpp::Image<uint8_t> img("test1.png");
	cvpp::ClampView sampler(img);

	auto out = cvpp::ConvoluteSeparable(sampler, cvpp::GaussFilter<11>(2.0f));

	out.save("test1_gauss.png");
}

TEST(Convolution, StackBlur)
{
	cvpp::Image<uint8_t> img("test1.png");
	cvpp::ClampView sampler(img);

	constexpr int sz = 15;
	auto out = cvpp::ConvoluteSeparable(sampler, cvpp::StackBlurFilter<sz>());

	out.save("test1_stackblur.png");
}

TEST(Convolution, BlurAndEdge)
{
	cvpp::Image<uint8_t> img("test1.png");
	cvpp::ClampView sampler(img);

	img = cvpp::MakeGrayscale(img);
	img = cvpp::ConvoluteSeparable(sampler, cvpp::StackBlurFilter<9>());
	img = cvpp::Convolute1D(sampler, cvpp::SimpleEdgeDetector, 3);

	img.save("test1_edges_hor.png");
}

TEST(Convert, Grayscale)
{
	cvpp::Image<uint8_t> img("test1.png");
	auto gr = cvpp::MakeGrayscale(img);

	gr.save("test1_greyscale.png");
}

TEST(Convolution, Sobel)
{
	cvpp::Image<uint8_t> img("test1.png");
	auto gr = cvpp::MakeGrayscale(img);
	gr = cvpp::Convolute2D(cvpp::ClampView(gr), cvpp::SobelFilterH());
	gr.save("test1_sobel_x.png");

	gr = cvpp::Convolute2D(cvpp::ClampView(gr), cvpp::SobelFilterV());
	gr.save("test1_sobel_y.png");
}

TEST(Structure, StructureTensor)
{
	cvpp::Image<uint8_t> img("test1.png");
	auto tensor = cvpp::StructureTensor(img);

	float max = 0.0f;
	auto determinant = tensor.transform([&max](const Eigen::Matrix2f& mtx) -> float {
		max = std::max(max, mtx.determinant());
		return (mtx.determinant() / mtx.trace());
	});
	std::cout << "Max: " << max << std::endl;
	max = 0.0f;

	determinant = cvpp::NonLinearConv2D(cvpp::ClampView(determinant), 7, [&max](int x, int y, auto v, auto& result) {
		max = std::max(max, std::abs(v[0]));
		result[0] = std::max(result[0], std::abs(v[0]));
	});
	std::cout << "Max: " << max << std::endl;
	max = 0.0f;

	determinant = determinant.transform([&max](auto v) -> float {
		max = std::max(max, std::abs(v));
		return (std::abs(v) >= 0.8 ? 0.0f : 1.0f);
	});

	std::cout << "Max: " << max << std::endl;
	(img * cvpp::MakeRGB(cvpp::ConvertType<float, unsigned char>(determinant))).save("structure_determinant.png");
}

TEST(Detector, Harris)
{
	cvpp::Image<uint8_t> img("test1.png");

	std::vector<cvpp::Feature> features;
	cvpp::HarrisDetector(img, 11, 1.1f, features);

	cvpp::MarkFeatures(img, Eigen::Vector4f(1, 0, 1, 1), features);
	img.save("HarrisFeatures.png");
	std::cout << "FEATURES: " << features.size() << std::endl;
}

TEST(Detector, Hessian)
{
	cvpp::Image<uint8_t> img("test1.png");

	std::vector<cvpp::Feature> features;
	cvpp::HessianDetector(img, 21, 0.5f, features);

	cvpp::MarkFeatures(img, Eigen::Vector4f(1, 0, 1, 1), features);
	img.save("HessianFeatures.png");
	std::cout << "FEATURES: " << features.size() << std::endl;
}

TEST(Convolution, Laplace)
{
	cvpp::Image<uint8_t> img("test1.png");
	auto gr = cvpp::MakeGrayscale(img);

	cvpp::ClampView sampler(gr);

	auto Dx = Convolute1D<cvpp::HORIZONTAL>(sampler, cvpp::LaplaceFilterX);
	auto Dy = Convolute1D<cvpp::VERTICAL>(sampler, cvpp::LaplaceFilterX);
	auto Dxy = Convolute2D(sampler, cvpp::LaplaceFilterXY());

	Dx.save("test1_laplace_x.png");
	Dy.save("test1_laplace_y.png");
	Dxy.save("test1_laplace_xy.png");
}
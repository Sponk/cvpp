#include <benchmark/benchmark.h>
#include <cvpp/cvsycl/Image.h>
#include <cvpp/cvsycl/Convolution.h>
#include <cvpp/cvsycl/StructureTensor.h>
#include <cvpp/CommonFilters.h>

using namespace cvpp;

class SYCLFixture : public benchmark::Fixture
{
public:
	void SetUp(const ::benchmark::State& state)
	{
		q = cl::sycl::queue(cl::sycl::default_selector());
		img = std::make_unique<cvsycl::Image<uint8_t>>("test1.png");

		// std::cout << "Using device: " << q.get_device().get_info<cl::sycl::info::device::name>() << std::endl;
	}

	void TearDown(const ::benchmark::State& state)
	{
	}

	cl::sycl::queue q;
	std::unique_ptr<cvsycl::Image<uint8_t>> img;
};


BENCHMARK_DEFINE_F(SYCLFixture, MakeGrayscale)(benchmark::State& state)
{
	auto& imgRef = *img;
	for(auto _ : state)
	{
		cvsycl::MakeGrayscale(imgRef, q);
		q.wait();
	}
}
BENCHMARK_REGISTER_F(SYCLFixture, MakeGrayscale)->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(SYCLFixture, Convolution2D)(benchmark::State& state)
{
	const auto k = cvpp::SobelFilterH();
	auto& imgRef = *img;
	for(auto _ : state)
	{
		cvsycl::Convolute2D<cvsycl::ClampView<uint8_t>>(imgRef, k, q);
		q.wait();
	}
}
BENCHMARK_REGISTER_F(SYCLFixture, Convolution2D)->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(SYCLFixture, ConvolutionSeparable)(benchmark::State& state)
{
	const auto k = cvpp::BoxFilter<3>();
	auto& imgRef = *img;
	for(auto _ : state)
	{
		cvsycl::ConvoluteSeparable<cvsycl::ClampView<uint8_t>>(imgRef, k, q);
		q.wait();
	}
}
BENCHMARK_REGISTER_F(SYCLFixture, ConvolutionSeparable)->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(SYCLFixture, ConvolutionVertical)(benchmark::State& state)
{
	const auto k = cvpp::BoxFilter<3>();
	auto& imgRef = *img;
	for(auto _ : state)
	{
		cvsycl::Convolute1D<cvsycl::ClampView<uint8_t>, cvsycl::VERTICAL>(imgRef, k, q);
		q.wait();
	}
}
BENCHMARK_REGISTER_F(SYCLFixture, ConvolutionVertical)->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(SYCLFixture, ConvolutionHorizontal)(benchmark::State& state)
{
	const auto k = cvpp::BoxFilter<3>();
	auto& imgRef = *img;
	for(auto _ : state)
	{
		cvsycl::Convolute1D<cvsycl::ClampView<uint8_t>, cvsycl::HORIZONTAL>(imgRef, k, q);
		q.wait();
	}
}
BENCHMARK_REGISTER_F(SYCLFixture, ConvolutionHorizontal)->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(SYCLFixture, StructureTensor)(benchmark::State& state)
{
	auto& imgRef = *img;
	for(auto _ : state)
	{
		cvsycl::StructureTensor(imgRef, q);
		q.wait();
	}
}
BENCHMARK_REGISTER_F(SYCLFixture, StructureTensor)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

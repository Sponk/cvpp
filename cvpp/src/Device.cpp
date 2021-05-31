#include <cvpp/Device.h>
#include <cvpp/Image.h>

using namespace cvpp;

template<typename T>
struct CPUOperator : public ImageOperator
{
	CPUImage<T> image;

	CPUOperator() = default;
	CPUOperator(const std::string& file):
		image(file)
	{}

	Image* operator->() override { return &image; }

	ImageOperator& makeGrayscale() override
	{
		image = cvpp::MakeGrayscale(image);
		return *this;
	}
};

ImageOperator* CPUDevice::loadImagePtr(const std::string& file, cvpp::IMAGE_TYPE type)
{
	switch(type)
	{
		case UCHAR:
			return new CPUOperator<uint8_t>(file);

		case USHORT:
			return new CPUOperator<uint16_t>(file);

		case FLOAT:
			return new CPUOperator<float>(file);

		case OTHER:
			throw std::runtime_error("Unknown type!");
	}
}

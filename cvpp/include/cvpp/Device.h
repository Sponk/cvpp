#ifndef __DEVICE_H__
#define __DEVICE_H__

#include <memory>
#include <string>
#include "Image.h"

namespace cvpp
{

class DeviceBase {};

struct ImageOperator
{
	virtual ~ImageOperator() = default;
	DeviceBase* device = nullptr;

	virtual Image* operator->() = 0;
	virtual ImageOperator& makeGrayscale() = 0;
};

class Device : public DeviceBase
{
public:
	virtual ~Device() = default;
	virtual const std::string getName() const = 0;

#ifdef SWIG
	%newobject loadImagePtr(const std::string&, IMAGE_TYPE);
	%rename (loadImage) loadImagePtr(const std::string&, IMAGE_TYPE = UCHAR);
#endif
	virtual ImageOperator* loadImagePtr(const std::string& file, IMAGE_TYPE type = UCHAR) = 0;

#ifndef SWIG
	std::shared_ptr<ImageOperator> loadImage(const std::string& file, IMAGE_TYPE type = UCHAR)
	{
		return std::shared_ptr<ImageOperator>(loadImage(file, type));
	}
#endif
};

class CPUDevice : public Device
{
public:
	const std::string getName() const override { return "CPU OpenMP"; }

#ifdef SWIG
	%newobject loadImagePtr(const std::string&, IMAGE_TYPE);
	%rename (loadImage) loadImagePtr(const std::string&, IMAGE_TYPE = UCHAR);
#endif
	ImageOperator* loadImagePtr(const std::string& file, IMAGE_TYPE type = UCHAR) override;

private:
};

}

#endif

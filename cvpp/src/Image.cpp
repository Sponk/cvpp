#include <cvpp/Image.h>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace cvpp;

void cvpp::ImageLoader::loadUChar(const std::string& file, unsigned int& w, unsigned int& h, unsigned int& c, std::vector<unsigned char>& data)
{
	auto* ptr = stbi_load(file.c_str(), (int*) &w, (int*) &h, (int*) &c, 0);
	
	if(!ptr)
		throw std::runtime_error(std::string("Could not load image: ") + stbi_failure_reason());

	data.resize(w*h*c);
	memcpy(data.data(), ptr, data.size()*sizeof(unsigned char));

	free(ptr);
}

void cvpp::ImageLoader::loadUShort(const std::string& file, unsigned int& w, unsigned int& h, unsigned int& c, std::vector<unsigned short>& data)
{
	auto* ptr = stbi_load_16(file.c_str(), (int*) &w, (int*) &h, (int*) &c, 0);

	if(!ptr)
		throw std::runtime_error(std::string("Could not load image: ") + stbi_failure_reason());

	data.resize(w*h*c);
	memcpy(data.data(), ptr, data.size()*sizeof(unsigned char));

	free(ptr);
}

void cvpp::ImageLoader::loadFloat(const std::string& file, unsigned int& w, unsigned int& h, unsigned int& c, std::vector<float>& data)
{
	auto* ptr = stbi_loadf(file.c_str(), (int*) &w, (int*) &h, (int*) &c, 0);

	if(!ptr)
		throw std::runtime_error(std::string("Could not load image: ") + stbi_failure_reason());

	data.resize(w*h*c);
	memcpy(data.data(), ptr, data.size()*sizeof(unsigned char));

	free(ptr);
}

#include <filesystem>
#include <algorithm>

#include <iostream>

template<typename T>
void save(const std::string& file, const T* data, unsigned int w, unsigned int h, unsigned int c)
{
	std::filesystem::path path(file);
	auto ext = path.extension().string().substr(1);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	int err = 0;
	using PT = typename std::remove_pointer<T>::type;
	if constexpr(std::is_same<T, unsigned char>::value)
	{
		if(ext == "png")
		{
			err = stbi_write_png(file.c_str(), w, h, c, data, 0);
		}
		else if(ext == "jpg" || ext == "jpeg")
		{
			err = stbi_write_jpg(file.c_str(), w, h, c, data, 90);
		}
		else if(ext == "bmp")
		{
			err = stbi_write_bmp(file.c_str(), w, h, c, data);
		}
		else if(ext == "tga")
		{
			err = stbi_write_tga(file.c_str(), w, h, c, data);
		}
	}
	else if constexpr(std::is_same<T, unsigned short>::value)
	{
		throw std::runtime_error("Writing ushort images is currently not supported!");
	}
	else if constexpr(std::is_same<PT, float>::value)
	{	
		if(ext == "hdr")
		{
			err = stbi_write_hdr(file.c_str(), w, h, c, data);
		}
	}

	if(!err)
		throw std::runtime_error("Could not write image as " + ext);
}

void cvpp::ImageLoader::saveUChar(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const std::vector<unsigned char>& data)
{
	save<unsigned char>(file, data.data(), w, h, c);
}

void cvpp::ImageLoader::saveUShort(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const std::vector<unsigned short>& data)
{
	save<unsigned short>(file, data.data(), w, h, c);
}

void cvpp::ImageLoader::saveFloat(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const std::vector<float>& data)
{
	save<float>(file, data.data(), w, h, c);
}

void cvpp::ImageLoader::saveUChar(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const unsigned char* data)
{
	save<unsigned char>(file, data, w, h, c);
}

void cvpp::ImageLoader::saveUShort(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const unsigned short* data)
{
	save<unsigned short>(file, data, w, h, c);
}

void cvpp::ImageLoader::saveFloat(const std::string& file, unsigned int w, unsigned int h, unsigned int c, const float* data)
{
	save<float>(file, data, w, h, c);
}


#ifndef __FEATURE_H__
#define __FEATURE_H__

#include <Eigen/StdVector>

namespace cvpp
{
struct Feature
{
	unsigned int x = -1, y = -1;
	float scale = 0.0f;
};

template<typename T>
void MarkFeatures(Image<T>& img, const Eigen::Vector4f& color, const std::vector<Feature>& features)
{
	#pragma omp parallel for
	for(int i = 0; i < features.size(); i++)
	{
		const auto& f = features[i];
		auto* px = img.get(f.x, f.y);
		for(int c = 0; c < img.getComponents(); c++)
		{
			px[c] = FloatToColor<T>(color[c]);
		}
	}
}

}

#endif

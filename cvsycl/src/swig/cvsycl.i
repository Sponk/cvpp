%{
#include <cvpp/cvsycl/Image.h>
using namespace cvsycl;
%}

%import <cvpp.i>

%nspace cvsycl;
%include <cvpp/cvsycl/Image.h>

%template(ImageUChar) cvsycl::Image<unsigned char>;
%template(ImageUShort) cvsycl::Image<unsigned short>;
%template(ImageFloat) cvsycl::Image<float>;

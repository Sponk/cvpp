%{
#include <cvpp/Device.h>
using namespace cvpp;
%}

%nspace cvpp;
%include <std_string.i>
%include <std_vector.i>
%include <std_except.i>

%include <cvpp/Image.h>
%include <cvpp/Device.h>

%template(CPUImageUChar) cvpp::CPUImage<unsigned char>;
%template(CPUImageUShort) cvpp::CPUImage<unsigned short>;
%template(CPUImageFloat) cvpp::CPUImage<float>;

%template(VectorUChar) std::vector<unsigned char>;
%template(VectorUShort) std::vector<unsigned short>;
%template(VectorFloat) std::vector<float>;
%template(VectorUInt) std::vector<unsigned int>;

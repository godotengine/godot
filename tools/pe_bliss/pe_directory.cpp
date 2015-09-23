#include "pe_directory.h"

namespace pe_bliss
{
//Default constructor
image_directory::image_directory()
	:rva_(0), size_(0)
{}

//Constructor from data
image_directory::image_directory(uint32_t rva, uint32_t size)
	:rva_(rva), size_(size)
{}

//Returns RVA
uint32_t image_directory::get_rva() const
{
	return rva_;
}

//Returns size
uint32_t image_directory::get_size() const
{
	return size_;
}

//Sets RVA
void image_directory::set_rva(uint32_t rva)
{
	rva_ = rva;
}

//Sets size
void image_directory::set_size(uint32_t size)
{
	size_ = size;
}
}

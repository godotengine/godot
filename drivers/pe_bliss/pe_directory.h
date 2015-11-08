#pragma once
#include "stdint_defs.h"

namespace pe_bliss
{
//Class representing image directory data
class image_directory
{
public:
	//Default constructor
	image_directory();
	//Constructor from data
	image_directory(uint32_t rva, uint32_t size);

	//Returns RVA
	uint32_t get_rva() const;
	//Returns size
	uint32_t get_size() const;

	//Sets RVA
	void set_rva(uint32_t rva);
	//Sets size
	void set_size(uint32_t size);

private:
	uint32_t rva_;
	uint32_t size_;
};
}

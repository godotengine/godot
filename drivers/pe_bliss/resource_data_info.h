#pragma once
#include <string>
#include "stdint_defs.h"

namespace pe_bliss
{
class resource_data_entry;

//Class representing resource data
class resource_data_info
{
public:
	//Constructor from data
	resource_data_info(const std::string& data, uint32_t codepage);
	//Constructor from data
	explicit resource_data_info(const resource_data_entry& data);

	//Returns resource data
	const std::string& get_data() const;
	//Returns resource codepage
	uint32_t get_codepage() const;

private:
	std::string data_;
	uint32_t codepage_;
};
}

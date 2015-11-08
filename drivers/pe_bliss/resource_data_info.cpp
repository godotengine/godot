#include "resource_data_info.h"
#include "pe_resource_viewer.h"

namespace pe_bliss
{
//Default constructor
resource_data_info::resource_data_info(const std::string& data, uint32_t codepage)
	:data_(data), codepage_(codepage)
{}

//Constructor from data
resource_data_info::resource_data_info(const resource_data_entry& data)
	:data_(data.get_data()), codepage_(data.get_codepage())
{}

//Returns resource data
const std::string& resource_data_info::get_data() const
{
	return data_;
}

//Returns resource codepage
uint32_t resource_data_info::get_codepage() const
{
	return codepage_;
}
}

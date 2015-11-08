#pragma once
#include <memory>
#include <istream>
#include <fstream>
#include "pe_base.h"

namespace pe_bliss
{
class pe_factory
{
public:
	//Creates pe_base class instance from PE or PE+ istream
	//If read_bound_import_raw_data, raw bound import data will be read (used to get bound import info)
	//If read_debug_raw_data, raw debug data will be read (used to get image debug info)
	static pe_base create_pe(std::istream& file, bool read_debug_raw_data = true);
	static pe_base create_pe(const char* file_path, bool read_debug_raw_data = true);
};
}

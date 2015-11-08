#include "pe_factory.h"
#include "pe_properties_generic.h"

namespace pe_bliss
{
pe_base pe_factory::create_pe(std::istream& file, bool read_debug_raw_data)
{
	return pe_base::get_pe_type(file) == pe_type_32
		? pe_base(file, pe_properties_32(), read_debug_raw_data)
		: pe_base(file, pe_properties_64(), read_debug_raw_data);
}

pe_base pe_factory::create_pe(const char* file_path, bool read_debug_raw_data)
{
	std::ifstream pe_file(file_path, std::ios::in | std::ios::binary);
	if(!pe_file)
	{
		throw pe_exception("Error in open file.", pe_exception::stream_is_bad);
	}
	return pe_factory::create_pe(pe_file,read_debug_raw_data);
}
}

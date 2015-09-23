#pragma once
#include "version_info_types.h"
#include "file_version_info.h"

namespace pe_bliss
{
class pe_resource_manager;

class resource_version_info_writer
{
public:
	resource_version_info_writer(pe_resource_manager& res);
	
	//Sets/replaces full version information:
	//file_version_info: versions and file info
	//lang_string_values_map: map of version info strings with encodings
	//translation_values_map: map of translations
	void set_version_info(const file_version_info& file_info,
		const lang_string_values_map& string_values,
		const translation_values_map& translations,
		uint32_t language,
		uint32_t codepage = 0,
		uint32_t timestamp = 0);
	
	//Removes version info by language (ID = 1)
	bool remove_version_info(uint32_t language);

private:
	pe_resource_manager& res_;
};
}

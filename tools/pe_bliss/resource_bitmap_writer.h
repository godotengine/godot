#pragma once
#include <string>
#include "stdint_defs.h"

namespace pe_bliss
{
class pe_resource_manager;

class resource_bitmap_writer
{
public:
	resource_bitmap_writer(pe_resource_manager& res);

	//Adds bitmap from bitmap file data. If bitmap already exists, replaces it
	//timestamp will be used for directories that will be added
	void add_bitmap(const std::string& bitmap_file, uint32_t id, uint32_t language, uint32_t codepage = 0, uint32_t timestamp = 0);
	void add_bitmap(const std::string& bitmap_file, const std::wstring& name, uint32_t language, uint32_t codepage = 0, uint32_t timestamp = 0);

	//Removes bitmap by name/ID and language
	bool remove_bitmap(const std::wstring& name, uint32_t language);
	bool remove_bitmap(uint32_t id, uint32_t language);

private:
	pe_resource_manager& res_;
};
}

#pragma once
#include <string>
#include "stdint_defs.h"

namespace pe_bliss
{
class pe_resource_viewer;

class resource_bitmap_reader
{
public:
	resource_bitmap_reader(const pe_resource_viewer& res);

	//Returns bitmap data by name and language (minimum checks of format correctness)
	const std::string get_bitmap_by_name(uint32_t language, const std::wstring& name) const;
	//Returns bitmap data by name and index in language directory (instead of language) (minimum checks of format correctness)
	const std::string get_bitmap_by_name(const std::wstring& name, uint32_t index = 0) const;
	//Returns bitmap data by ID and language (minimum checks of format correctness)
	const std::string get_bitmap_by_id_lang(uint32_t language, uint32_t id) const;
	//Returns bitmap data by ID and index in language directory (instead of language) (minimum checks of format correctness)
	const std::string get_bitmap_by_id(uint32_t id, uint32_t index = 0) const;

private:
	//Helper function of creating bitmap header
	static const std::string create_bitmap(const std::string& resource_data);

	const pe_resource_viewer& res_;
};
}

#pragma once
#include <string>
#include <map>
#include "stdint_defs.h"

namespace pe_bliss
{
class pe_resource_viewer;

//ID; string
typedef std::map<uint16_t, std::wstring> resource_string_list;

class resource_string_table_reader
{
public:
	resource_string_table_reader(const pe_resource_viewer& res);

public:
	//Returns string table data by ID and language
	const resource_string_list get_string_table_by_id_lang(uint32_t language, uint32_t id) const;
	//Returns string table data by ID and index in language directory (instead of language)
	const resource_string_list get_string_table_by_id(uint32_t id, uint32_t index = 0) const;
	//Returns string from string table by ID and language
	const std::wstring get_string_by_id_lang(uint32_t language, uint16_t id) const;
	//Returns string from string table by ID and index in language directory (instead of language)
	const std::wstring get_string_by_id(uint16_t id, uint32_t index = 0) const;

private:
	const pe_resource_viewer& res_;

	//Helper function of parsing string list table
	//Id of resource is needed to calculate string IDs correctly
	//resource_data is raw string table resource data
	static const resource_string_list parse_string_list(uint32_t id, const std::string& resource_data);
};
}

#pragma once
#include "message_table.h"

namespace pe_bliss
{
class pe_resource_viewer;

//ID; message_table_item
typedef std::map<uint32_t, message_table_item> resource_message_list;

class resource_message_list_reader
{
public:
	resource_message_list_reader(const pe_resource_viewer& res);

	//Returns message table data by ID and language
	const resource_message_list get_message_table_by_id_lang(uint32_t language, uint32_t id) const;
	//Returns message table data by ID and index in language directory (instead of language)
	const resource_message_list get_message_table_by_id(uint32_t id, uint32_t index = 0) const;

	//Helper function of parsing message list table
	//resource_data - raw message table resource data
	static const resource_message_list parse_message_list(const std::string& resource_data);

private:
	const pe_resource_viewer& res_;
};
}

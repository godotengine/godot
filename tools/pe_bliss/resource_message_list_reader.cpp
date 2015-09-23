#include "resource_message_list_reader.h"
#include "pe_resource_viewer.h"

namespace pe_bliss
{
using namespace pe_win;

resource_message_list_reader::resource_message_list_reader(const pe_resource_viewer& res)
	:res_(res)
{}

//Helper function of parsing message list table
const resource_message_list resource_message_list_reader::parse_message_list(const std::string& resource_data)
{
	resource_message_list ret;

	//Check resource data length
	if(resource_data.length() < sizeof(message_resource_data))
		throw pe_exception("Incorrect resource message table", pe_exception::resource_incorrect_message_table);

	const message_resource_data* message_data = reinterpret_cast<const message_resource_data*>(resource_data.data());

	//Check resource data length more carefully and some possible overflows
	if(message_data->NumberOfBlocks >= pe_utils::max_dword / sizeof(message_resource_block)
		|| !pe_utils::is_sum_safe(message_data->NumberOfBlocks * sizeof(message_resource_block), sizeof(message_resource_data))
		|| resource_data.length() < message_data->NumberOfBlocks * sizeof(message_resource_block) + sizeof(message_resource_data))
		throw pe_exception("Incorrect resource message table", pe_exception::resource_incorrect_message_table);

	//Iterate over all message resource blocks
	for(unsigned long i = 0; i != message_data->NumberOfBlocks; ++i)
	{
		//Get block
		const message_resource_block* block =
			reinterpret_cast<const message_resource_block*>(resource_data.data() + sizeof(message_resource_data) - sizeof(message_resource_block) + sizeof(message_resource_block) * i);

		//Check resource data length and IDs
		if(resource_data.length() < block->OffsetToEntries || block->LowId > block->HighId)
			throw pe_exception("Incorrect resource message table", pe_exception::resource_incorrect_message_table);

		unsigned long current_pos = 0;
		static const unsigned long size_of_entry_headers = 4;
		//List all message resource entries in block
		for(uint32_t curr_id = block->LowId; curr_id <= block->HighId; curr_id++)
		{
			//Check resource data length and some possible overflows
			if(!pe_utils::is_sum_safe(block->OffsetToEntries, current_pos)
				|| !pe_utils::is_sum_safe(block->OffsetToEntries + current_pos, size_of_entry_headers)
				|| resource_data.length() < block->OffsetToEntries + current_pos + size_of_entry_headers)
				throw pe_exception("Incorrect resource message table", pe_exception::resource_incorrect_message_table);

			//Get entry
			const message_resource_entry* entry = reinterpret_cast<const message_resource_entry*>(resource_data.data() + block->OffsetToEntries + current_pos);

			//Check resource data length and entry length and some possible overflows
			if(entry->Length < size_of_entry_headers
				|| !pe_utils::is_sum_safe(block->OffsetToEntries + current_pos, entry->Length)
				|| resource_data.length() < block->OffsetToEntries + current_pos + entry->Length
				|| entry->Length < size_of_entry_headers)
				throw pe_exception("Incorrect resource message table", pe_exception::resource_incorrect_message_table);

			if(entry->Flags & message_resource_unicode)
			{
				//If string is UNICODE
				//Check its length
				if(entry->Length % 2)
					throw pe_exception("Incorrect resource message table", pe_exception::resource_incorrect_message_table);

				//Add ID and string to message table
#ifdef PE_BLISS_WINDOWS
				ret.insert(std::make_pair(curr_id, message_table_item(
					std::wstring(reinterpret_cast<const wchar_t*>(resource_data.data() + block->OffsetToEntries + current_pos + size_of_entry_headers),
					(entry->Length - size_of_entry_headers) / 2)
					)));
#else
				ret.insert(std::make_pair(curr_id, message_table_item(
					pe_utils::from_ucs2(u16string(reinterpret_cast<const unicode16_t*>(resource_data.data() + block->OffsetToEntries + current_pos + size_of_entry_headers),
					(entry->Length - size_of_entry_headers) / 2))
					)));
#endif
			}
			else
			{
				//If string is ANSI
				//Add ID and string to message table
				ret.insert(std::make_pair(curr_id, message_table_item(
					std::string(resource_data.data() + block->OffsetToEntries + current_pos + size_of_entry_headers,
					entry->Length - size_of_entry_headers)
					)));
			}

			//Go to next entry
			current_pos += entry->Length;
		}
	}

	return ret;
}

//Returns message table data by ID and index in language directory (instead of language)
const resource_message_list resource_message_list_reader::get_message_table_by_id(uint32_t id, uint32_t index) const
{
	return parse_message_list(res_.get_resource_data_by_id(pe_resource_viewer::resource_message_table, id, index).get_data());
}

//Returns message table data by ID and language
const resource_message_list resource_message_list_reader::get_message_table_by_id_lang(uint32_t language, uint32_t id) const
{
	return parse_message_list(res_.get_resource_data_by_id(language, pe_resource_viewer::resource_message_table, id).get_data());
}
}

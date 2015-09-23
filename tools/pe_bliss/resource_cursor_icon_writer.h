#pragma once
#include <string>
#include <vector>
#include "stdint_defs.h"
#include "pe_resource_manager.h"

namespace pe_bliss
{
class pe_resource_manager;

class resource_cursor_icon_writer
{
public:
	//Determines, how new icon(s) or cursor(s) will be placed
	enum icon_place_mode
	{
		icon_place_after_max_icon_id, //Icon(s) will be placed after all existing
		icon_place_free_ids //New icon(s) will take all free IDs between existing icons
	};
	
public:
	resource_cursor_icon_writer(pe_resource_manager& res);

	//Removes icon group and all its icons by name/ID and language
	bool remove_icon_group(const std::wstring& icon_group_name, uint32_t language);
	bool remove_icon_group(uint32_t icon_group_id, uint32_t language);

	//Adds icon(s) from icon file data
	//timestamp will be used for directories that will be added
	//If icon group with name "icon_group_name" or ID "icon_group_id" already exists, it will be appended with new icon(s)
	//(Codepage of icon group and icons will not be changed in this case)
	//icon_place_mode determines, how new icon(s) will be placed
	void add_icon(const std::string& icon_file,
		const std::wstring& icon_group_name,
		uint32_t language, icon_place_mode mode = icon_place_after_max_icon_id,
		uint32_t codepage = 0, uint32_t timestamp = 0);

	void add_icon(const std::string& icon_file,
		uint32_t icon_group_id,
		uint32_t language, icon_place_mode mode = icon_place_after_max_icon_id,
		uint32_t codepage = 0, uint32_t timestamp = 0);
	
	//Removes cursor group and all its cursors by name/ID and language
	bool remove_cursor_group(const std::wstring& cursor_group_name, uint32_t language);
	bool remove_cursor_group(uint32_t cursor_group_id, uint32_t language);

	//Adds cursor(s) from cursor file data
	//timestamp will be used for directories that will be added
	//If cursor group with name "cursor_group_name" or ID "cursor_group_id" already exists, it will be appended with new cursor(s)
	//(Codepage of cursor group and cursors will not be changed in this case)
	//icon_place_mode determines, how new cursor(s) will be placed
	void add_cursor(const std::string& cursor_file, const std::wstring& cursor_group_name, uint32_t language, icon_place_mode mode = icon_place_after_max_icon_id, uint32_t codepage = 0, uint32_t timestamp = 0);
	void add_cursor(const std::string& cursor_file, uint32_t cursor_group_id, uint32_t language, icon_place_mode mode = icon_place_after_max_icon_id, uint32_t codepage = 0, uint32_t timestamp = 0);

private:
	pe_resource_manager& res_;

	//Add icon helper
	void add_icon(const std::string& icon_file, const resource_data_info* group_icon_info /* or zero */, resource_directory_entry& new_icon_group_entry, const resource_directory::entry_finder& finder, uint32_t language, icon_place_mode mode, uint32_t codepage, uint32_t timestamp);
	
	//Remove icon group helper
	void remove_icons_from_icon_group(const std::string& icon_group_data, uint32_t language);

	//Add cursor helper
	void add_cursor(const std::string& cursor_file, const resource_data_info* group_cursor_info /* or zero */, resource_directory_entry& new_cursor_group_entry, const resource_directory::entry_finder& finder, uint32_t language, icon_place_mode mode, uint32_t codepage, uint32_t timestamp);

	//Remove cursor group helper
	void remove_cursors_from_cursor_group(const std::string& cursor_group_data, uint32_t language);

	//Returns free icon or cursor ID list depending on icon_place_mode
	const std::vector<uint16_t> get_icon_or_cursor_free_id_list(pe_resource_manager::resource_type type, icon_place_mode mode, uint32_t count);
};
}

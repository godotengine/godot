/*************************************************************************/
/* Copyright (c) 2015 dx, http://kaimi.ru                                */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person           */
/* obtaining a copy of this software and associated documentation        */
/* files (the "Software"), to deal in the Software without               */
/* restriction, including without limitation the rights to use,          */
/* copy, modify, merge, publish, distribute, sublicense, and/or          */
/* sell copies of the Software, and to permit persons to whom the        */
/* Software is furnished to do so, subject to the following conditions:  */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include <algorithm>
#include <string.h>
#include "resource_cursor_icon_writer.h"

namespace pe_bliss
{
using namespace pe_win;

resource_cursor_icon_writer::resource_cursor_icon_writer(pe_resource_manager& res)
	:res_(res)
{}

//Add icon helper
void resource_cursor_icon_writer::add_icon(const std::string& icon_file, const resource_data_info* group_icon_info /* or zero */, resource_directory_entry& new_icon_group_entry, const resource_directory::entry_finder& finder, uint32_t language, icon_place_mode mode, uint32_t codepage, uint32_t timestamp)
{
	//Check icon for correctness
	if(icon_file.length() < sizeof(ico_header))
		throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

	const ico_header* icon_header = reinterpret_cast<const ico_header*>(&icon_file[0]);

	unsigned long size_of_headers = sizeof(ico_header) + icon_header->Count * sizeof(icondirentry);
	if(icon_file.length() < size_of_headers || icon_header->Count == 0)
		throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

	//Enumerate all icons in file
	for(uint16_t i = 0; i != icon_header->Count; ++i)
	{
		//Check icon entries
		const icondirentry* icon_entry = reinterpret_cast<const icondirentry*>(&icon_file[sizeof(ico_header) + i * sizeof(icondirentry)]);
		if(icon_entry->SizeInBytes == 0
			|| icon_entry->ImageOffset < size_of_headers
			|| !pe_utils::is_sum_safe(icon_entry->ImageOffset, icon_entry->SizeInBytes)
			|| icon_entry->ImageOffset + icon_entry->SizeInBytes > icon_file.length())
			throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);
	}

	std::string icon_group_data;
	ico_header* info = 0;

	if(group_icon_info)
	{
		//If icon group already exists
		{
			icon_group_data = group_icon_info->get_data();
			codepage = group_icon_info->get_codepage(); //Don't change codepage of icon group entry
		}

		//Check resource data size
		if(icon_group_data.length() < sizeof(ico_header))
			throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

		//Get icon header
		info = reinterpret_cast<ico_header*>(&icon_group_data[0]);

		//Check resource data size
		if(icon_group_data.length() < sizeof(ico_header) + info->Count * sizeof(icon_group))
			throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

		icon_group_data.resize(sizeof(ico_header) + (info->Count + icon_header->Count) * sizeof(icon_group));
		info = reinterpret_cast<ico_header*>(&icon_group_data[0]); //In case if memory was reallocated
	}
	else //Entry not found - icon group doesn't exist
	{
		icon_group_data.resize(sizeof(ico_header) + icon_header->Count * sizeof(icon_group));
		memcpy(&icon_group_data[0], icon_header, sizeof(ico_header));
	}

	//Search for available icon IDs
	std::vector<uint16_t> icon_id_list(get_icon_or_cursor_free_id_list(pe_resource_viewer::resource_icon, mode, icon_header->Count));

	//Enumerate all icons in file
	for(uint16_t i = 0; i != icon_header->Count; ++i)
	{
		const icondirentry* icon_entry = reinterpret_cast<const icondirentry*>(&icon_file[sizeof(ico_header) + i * sizeof(icondirentry)]);
		icon_group group = {0};

		//Fill icon resource header
		group.BitCount = icon_entry->BitCount;
		group.ColorCount = icon_entry->ColorCount;
		group.Height = icon_entry->Height;
		group.Planes = icon_entry->Planes;
		group.Reserved = icon_entry->Reserved;
		group.SizeInBytes = icon_entry->SizeInBytes;
		group.Width = icon_entry->Width;
		group.Number = icon_id_list.at(i);

		memcpy(&icon_group_data[sizeof(ico_header) + ((info ? info->Count : 0) + i) * sizeof(icon_group)], &group, sizeof(group));

		//Add icon to resources
		resource_directory_entry new_entry;
		new_entry.set_id(group.Number);
		res_.add_resource(icon_file.substr(icon_entry->ImageOffset, icon_entry->SizeInBytes), pe_resource_viewer::resource_icon, new_entry, resource_directory::entry_finder(group.Number), language, codepage, timestamp);
	}

	if(info)
		info->Count += icon_header->Count; //Increase icon count, if we're adding icon to existing group

	{
		//Add or replace icon group data entry
		res_.add_resource(icon_group_data, pe_resource_viewer::resource_icon_group, new_icon_group_entry, finder, language, codepage, timestamp);
	}
}

//Returns free icon or cursor ID list depending on icon_place_mode
const std::vector<uint16_t> resource_cursor_icon_writer::get_icon_or_cursor_free_id_list(pe_resource_viewer::resource_type type, icon_place_mode mode, uint32_t count)
{
	//Search for available icon/cursor IDs
	std::vector<uint16_t> icon_cursor_id_list;

	try
	{
		//If any icon exists
		//List icon IDs
		std::vector<uint32_t> id_list(res_.list_resource_ids(type));
		std::sort(id_list.begin(), id_list.end());

		//If we are placing icon on free spaces
		//I.e., icon IDs 1, 3, 4, 7, 8 already exist
		//We'll place five icons on IDs 2, 5, 6, 9, 10
		if(mode != icon_place_after_max_icon_id)
		{
			if(!id_list.empty())
			{
				//Determine and list free icon IDs
				for(std::vector<uint32_t>::const_iterator it = id_list.begin(); it != id_list.end(); ++it)
				{
					if(it == id_list.begin())
					{
						if(*it > 1)
						{
							for(uint16_t i = 1; i != *it; ++i)
							{
								icon_cursor_id_list.push_back(i);
								if(icon_cursor_id_list.size() == count)
									break;
							}
						}
					}
					else if(*(it - 1) - *it > 1)
					{
						for(uint16_t i = static_cast<uint16_t>(*(it - 1) + 1); i != static_cast<uint16_t>(*it); ++i)
						{
							icon_cursor_id_list.push_back(i);
							if(icon_cursor_id_list.size() == count)
								break;
						}
					}

					if(icon_cursor_id_list.size() == count)
						break;
				}
			}
		}

		uint32_t max_id = id_list.empty() ? 0 : *std::max_element(id_list.begin(), id_list.end());
		for(uint32_t i = static_cast<uint32_t>(icon_cursor_id_list.size()); i != count; ++i)
			icon_cursor_id_list.push_back(static_cast<uint16_t>(++max_id));
	}
	catch(const pe_exception&) //Entry not found
	{
		for(uint16_t i = 1; i != count + 1; ++i)
			icon_cursor_id_list.push_back(i);
	}

	return icon_cursor_id_list;
}

//Add cursor helper
void resource_cursor_icon_writer::add_cursor(const std::string& cursor_file, const resource_data_info* group_cursor_info /* or zero */, resource_directory_entry& new_cursor_group_entry, const resource_directory::entry_finder& finder, uint32_t language, icon_place_mode mode, uint32_t codepage, uint32_t timestamp)
{
	//Check cursor for correctness
	if(cursor_file.length() < sizeof(cursor_header))
		throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

	const cursor_header* cur_header = reinterpret_cast<const cursor_header*>(&cursor_file[0]);

	unsigned long size_of_headers = sizeof(cursor_header) + cur_header->Count * sizeof(cursordirentry);
	if(cursor_file.length() < size_of_headers || cur_header->Count == 0)
		throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

	//Enumerate all cursors in file
	for(uint16_t i = 0; i != cur_header->Count; ++i)
	{
		//Check cursor entries
		const cursordirentry* cursor_entry = reinterpret_cast<const cursordirentry*>(&cursor_file[sizeof(cursor_header) + i * sizeof(cursordirentry)]);
		if(cursor_entry->SizeInBytes == 0
			|| cursor_entry->ImageOffset < size_of_headers
			|| !pe_utils::is_sum_safe(cursor_entry->ImageOffset, cursor_entry->SizeInBytes)
			|| cursor_entry->ImageOffset + cursor_entry->SizeInBytes > cursor_file.length())
			throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);
	}

	std::string cursor_group_data;
	cursor_header* info = 0;

	if(group_cursor_info)
	{
		//If cursor group already exists
		{
			cursor_group_data = group_cursor_info->get_data();
			codepage = group_cursor_info->get_codepage(); //Don't change codepage of cursor group entry
		}

		//Check resource data size
		if(cursor_group_data.length() < sizeof(cursor_header))
			throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

		//Get cursor header
		info = reinterpret_cast<cursor_header*>(&cursor_group_data[0]);

		//Check resource data size
		if(cursor_group_data.length() < sizeof(cursor_header) + info->Count * sizeof(cursor_group))
			throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

		cursor_group_data.resize(sizeof(cursor_header) + (info->Count + cur_header->Count) * sizeof(cursor_group));
		info = reinterpret_cast<cursor_header*>(&cursor_group_data[0]); //In case if memory was reallocated
	}
	else //Entry not found - cursor group doesn't exist
	{
		cursor_group_data.resize(sizeof(cursor_header) + cur_header->Count * sizeof(cursor_group));
		memcpy(&cursor_group_data[0], cur_header, sizeof(cursor_header));
	}

	//Search for available cursor IDs
	std::vector<uint16_t> cursor_id_list(get_icon_or_cursor_free_id_list(pe_resource_viewer::resource_cursor, mode, cur_header->Count));

	//Enumerate all cursors in file
	for(uint16_t i = 0; i != cur_header->Count; ++i)
	{
		const cursordirentry* cursor_entry = reinterpret_cast<const cursordirentry*>(&cursor_file[sizeof(cursor_header) + i * sizeof(cursordirentry)]);
		cursor_group group = {0};

		//Fill cursor resource header
		group.Height = cursor_entry->Height * 2;
		group.SizeInBytes = cursor_entry->SizeInBytes + 2 * sizeof(uint16_t) /* hotspot coordinates */;
		group.Width = cursor_entry->Width;
		group.Number = cursor_id_list.at(i);

		memcpy(&cursor_group_data[sizeof(cursor_header) + ((info ? info->Count : 0) + i) * sizeof(cursor_group)], &group, sizeof(group));

		//Add cursor to resources
		resource_directory_entry new_entry;
		new_entry.set_id(group.Number);

		//Fill resource data (two WORDs for hotspot of cursor, and cursor bitmap data)
		std::string cur_data;
		cur_data.resize(sizeof(uint16_t) * 2);
		memcpy(&cur_data[0], &cursor_entry->HotspotX, sizeof(uint16_t));
		memcpy(&cur_data[sizeof(uint16_t)], &cursor_entry->HotspotY, sizeof(uint16_t));
		cur_data.append(cursor_file.substr(cursor_entry->ImageOffset, cursor_entry->SizeInBytes));

		res_.add_resource(cur_data, pe_resource_viewer::resource_cursor, new_entry, resource_directory::entry_finder(group.Number), language, codepage, timestamp);
	}

	if(info)
		info->Count += cur_header->Count; //Increase cursor count, if we're adding cursor to existing group

	{
		//Add or replace cursor group data entry
		res_.add_resource(cursor_group_data, pe_resource_viewer::resource_cursor_group, new_cursor_group_entry, finder, language, codepage, timestamp);
	}
}

//Adds icon(s) from icon file data
//timestamp will be used for directories that will be added
//If icon group with name "icon_group_name" or ID "icon_group_id" already exists, it will be appended with new icon(s)
//(Codepage of icon group and icons will not be changed in this case)
//icon_place_mode determines, how new icon(s) will be placed
void resource_cursor_icon_writer::add_icon(const std::string& icon_file, const std::wstring& icon_group_name, uint32_t language, icon_place_mode mode, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_icon_group_entry;
	new_icon_group_entry.set_name(icon_group_name);
	std::auto_ptr<resource_data_info> data_info;

	try
	{
		data_info.reset(new resource_data_info(res_.get_resource_data_by_name(language, pe_resource_viewer::resource_icon_group, icon_group_name)));
	}
	catch(const pe_exception&) //Entry not found
	{
	}

	add_icon(icon_file, data_info.get(), new_icon_group_entry, resource_directory::entry_finder(icon_group_name), language, mode, codepage, timestamp);
}

void resource_cursor_icon_writer::add_icon(const std::string& icon_file, uint32_t icon_group_id, uint32_t language, icon_place_mode mode, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_icon_group_entry;
	new_icon_group_entry.set_id(icon_group_id);
	std::auto_ptr<resource_data_info> data_info;

	try
	{
		data_info.reset(new resource_data_info(res_.get_resource_data_by_id(language, pe_resource_viewer::resource_icon_group, icon_group_id)));
	}
	catch(const pe_exception&) //Entry not found
	{
	}

	add_icon(icon_file, data_info.get(), new_icon_group_entry, resource_directory::entry_finder(icon_group_id), language, mode, codepage, timestamp);
}

//Adds cursor(s) from cursor file data
//timestamp will be used for directories that will be added
//If cursor group with name "cursor_group_name" or ID "cursor_group_id" already exists, it will be appended with new cursor(s)
//(Codepage of cursor group and cursors will not be changed in this case)
//icon_place_mode determines, how new cursor(s) will be placed
void resource_cursor_icon_writer::add_cursor(const std::string& cursor_file, const std::wstring& cursor_group_name, uint32_t language, icon_place_mode mode, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_cursor_group_entry;
	new_cursor_group_entry.set_name(cursor_group_name);
	std::auto_ptr<resource_data_info> data_info;

	try
	{
		data_info.reset(new resource_data_info(res_.get_resource_data_by_name(language, pe_resource_viewer::resource_cursor_group, cursor_group_name)));
	}
	catch(const pe_exception&) //Entry not found
	{
	}

	add_cursor(cursor_file, data_info.get(), new_cursor_group_entry, resource_directory::entry_finder(cursor_group_name), language, mode, codepage, timestamp);
}

void resource_cursor_icon_writer::add_cursor(const std::string& cursor_file, uint32_t cursor_group_id, uint32_t language, icon_place_mode mode, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_cursor_group_entry;
	new_cursor_group_entry.set_id(cursor_group_id);
	std::auto_ptr<resource_data_info> data_info;

	try
	{
		data_info.reset(new resource_data_info(res_.get_resource_data_by_id(language, pe_resource_viewer::resource_cursor_group, cursor_group_id)));
	}
	catch(const pe_exception&) //Entry not found
	{
	}

	add_cursor(cursor_file, data_info.get(), new_cursor_group_entry, resource_directory::entry_finder(cursor_group_id), language, mode, codepage, timestamp);
}

//Remove icon group helper
void resource_cursor_icon_writer::remove_icons_from_icon_group(const std::string& icon_group_data, uint32_t language)
{
	//Check resource data size
	if(icon_group_data.length() < sizeof(ico_header))
		throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

	//Get icon header
	const ico_header* info = reinterpret_cast<const ico_header*>(icon_group_data.data());

	uint16_t icon_count = info->Count;

	//Check resource data size
	if(icon_group_data.length() < sizeof(ico_header) + icon_count * sizeof(icon_group))
		throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

	//Remove icon data
	for(uint16_t i = 0; i != icon_count; ++i)
	{
		const icon_group* group = reinterpret_cast<const icon_group*>(icon_group_data.data() + sizeof(ico_header) + i * sizeof(icon_group));
		res_.remove_resource(pe_resource_viewer::resource_icon, group->Number, language);
	}
}

//Remove cursor group helper
void resource_cursor_icon_writer::remove_cursors_from_cursor_group(const std::string& cursor_group_data, uint32_t language)
{
	//Check resource data size
	if(cursor_group_data.length() < sizeof(cursor_header))
		throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

	//Get icon header
	const cursor_header* info = reinterpret_cast<const cursor_header*>(cursor_group_data.data());

	uint16_t cursor_count = info->Count;

	//Check resource data size
	if(cursor_group_data.length() < sizeof(cursor_header) + cursor_count * sizeof(cursor_group))
		throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

	//Remove icon data
	for(uint16_t i = 0; i != cursor_count; ++i)
	{
		const icon_group* group = reinterpret_cast<const icon_group*>(cursor_group_data.data() + sizeof(cursor_header) + i * sizeof(cursor_group));
		res_.remove_resource(pe_resource_viewer::resource_cursor, group->Number, language);
	}
}

//Removes cursor group and all its cursors by name/ID and language
bool resource_cursor_icon_writer::remove_cursor_group(const std::wstring& cursor_group_name, uint32_t language)
{
	//Get resource by name and language
	const std::string data = res_.get_resource_data_by_name(language, pe_resource_viewer::resource_cursor_group, cursor_group_name).get_data();
	remove_cursors_from_cursor_group(data, language);
	return res_.remove_resource(pe_resource_viewer::resource_cursor_group, cursor_group_name, language);
}

//Removes cursor group and all its cursors by name/ID and language
bool resource_cursor_icon_writer::remove_cursor_group(uint32_t cursor_group_id, uint32_t language)
{
	//Get resource by name and language
	const std::string data = res_.get_resource_data_by_id(language, pe_resource_viewer::resource_cursor_group, cursor_group_id).get_data();
	remove_cursors_from_cursor_group(data, language);
	return res_.remove_resource(pe_resource_viewer::resource_cursor_group, cursor_group_id, language);
}

//Removes icon group and all its icons by name/ID and language
bool resource_cursor_icon_writer::remove_icon_group(const std::wstring& icon_group_name, uint32_t language)
{
	//Get resource by name and language
	const std::string data = res_.get_resource_data_by_name(language, pe_resource_viewer::resource_icon_group, icon_group_name).get_data();
	remove_icons_from_icon_group(data, language);
	return res_.remove_resource(pe_resource_viewer::resource_icon_group, icon_group_name, language);
}

//Removes icon group and all its icons by name/ID and language
bool resource_cursor_icon_writer::remove_icon_group(uint32_t icon_group_id, uint32_t language)
{
	//Get resource by name and language
	const std::string data = res_.get_resource_data_by_id(language, pe_resource_viewer::resource_icon_group, icon_group_id).get_data();
	remove_icons_from_icon_group(data, language);
	return res_.remove_resource(pe_resource_viewer::resource_icon_group, icon_group_id, language);
}
}

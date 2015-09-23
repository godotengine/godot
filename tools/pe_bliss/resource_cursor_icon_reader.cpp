#include <algorithm>
#include "resource_cursor_icon_reader.h"
#include "pe_structures.h"
#include "pe_resource_viewer.h"

namespace pe_bliss
{
using namespace pe_win;

resource_cursor_icon_reader::resource_cursor_icon_reader(const pe_resource_viewer& res)
	:res_(res)
{}

//Helper function of creating icon headers from ICON_GROUP resource data
//Returns icon count
uint16_t resource_cursor_icon_reader::format_icon_headers(std::string& ico_data, const std::string& resource_data)
{
	//Check resource data size
	if(resource_data.length() < sizeof(ico_header))
		throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

	//Get icon header
	const ico_header* info = reinterpret_cast<const ico_header*>(resource_data.data());

	//Check resource data size
	if(resource_data.length() < sizeof(ico_header) + info->Count * sizeof(icon_group))
		throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

	//Reserve memory to speed up a little
	ico_data.reserve(sizeof(ico_header) + info->Count * sizeof(icondirentry));
	ico_data.append(reinterpret_cast<const char*>(info), sizeof(ico_header));

	//Iterate over all listed icons
	uint32_t offset = sizeof(ico_header) + sizeof(icondirentry) * info->Count;
	for(uint16_t i = 0; i != info->Count; ++i)
	{
		const icon_group* group = reinterpret_cast<const icon_group*>(resource_data.data() + sizeof(ico_header) + i * sizeof(icon_group));

		//Fill icon data
		icondirentry direntry;
		direntry.BitCount = group->BitCount;
		direntry.ColorCount = group->ColorCount;
		direntry.Height = group->Height;
		direntry.Planes = group->Planes;
		direntry.Reserved = group->Reserved;
		direntry.SizeInBytes = group->SizeInBytes;
		direntry.Width = group->Width;
		direntry.ImageOffset = offset;

		//Add icon header to returned value
		ico_data.append(reinterpret_cast<const char*>(&direntry), sizeof(icondirentry));

		offset += group->SizeInBytes;
	}

	//Return icon count
	return info->Count;
}

//Returns single icon data by ID and language (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_single_icon_by_id_lang(uint32_t language, uint32_t id) const
{
	//Get icon headers
	std::string icon_data(lookup_icon_group_data_by_icon(id, language));
	//Append icon data
	icon_data.append(res_.get_resource_data_by_id(language, pe_resource_viewer::resource_icon, id).get_data());
	return icon_data;
}

//Returns single icon data by ID and index in language directory (instead of language) (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_single_icon_by_id(uint32_t id, uint32_t index) const
{
	pe_resource_viewer::resource_language_list languages(res_.list_resource_languages(pe_resource_viewer::resource_icon, id));
	if(languages.size() <= index)
		throw pe_exception("Resource data entry not found", pe_exception::resource_data_entry_not_found);

	//Get icon headers
	std::string icon_data(lookup_icon_group_data_by_icon(id, languages.at(index)));
	//Append icon data
	icon_data.append(res_.get_resource_data_by_id(pe_resource_viewer::resource_icon, id, index).get_data());
	return icon_data;
}

//Returns icon data by name and index in language directory (instead of language) (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_icon_by_name(const std::wstring& name, uint32_t index) const
{
	std::string ret;

	//Get resource by name and index
	const std::string data = res_.get_resource_data_by_name(pe_resource_viewer::resource_icon_group, name, index).get_data();

	//Create icon headers
	uint16_t icon_count = format_icon_headers(ret, data);

	//Append icon data
	for(uint16_t i = 0; i != icon_count; ++i)
	{
		const icon_group* group = reinterpret_cast<const icon_group*>(data.data() + sizeof(ico_header) + i * sizeof(icon_group));
		ret += res_.get_resource_data_by_id(pe_resource_viewer::resource_icon, group->Number, index).get_data();
	}

	return ret;
}

//Returns icon data by name and language (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_icon_by_name(uint32_t language, const std::wstring& name) const
{
	std::string ret;

	//Get resource by name and language
	const std::string data = res_.get_resource_data_by_name(language, pe_resource_viewer::resource_icon_group, name).get_data();

	//Create icon headers
	uint16_t icon_count = format_icon_headers(ret, data);

	//Append icon data
	for(uint16_t i = 0; i != icon_count; ++i)
	{
		const icon_group* group = reinterpret_cast<const icon_group*>(data.data() + sizeof(ico_header) + i * sizeof(icon_group));
		ret += res_.get_resource_data_by_id(language, pe_resource_viewer::resource_icon, group->Number).get_data();
	}

	return ret;
}

//Returns icon data by ID and language (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_icon_by_id_lang(uint32_t language, uint32_t id) const
{
	std::string ret;

	//Get resource by language and id
	const std::string data = res_.get_resource_data_by_id(language, pe_resource_viewer::resource_icon_group, id).get_data();

	//Create icon headers
	uint16_t icon_count = format_icon_headers(ret, data);

	//Append icon data
	for(uint16_t i = 0; i != icon_count; ++i)
	{
		const icon_group* group = reinterpret_cast<const icon_group*>(data.data() + sizeof(ico_header) + i * sizeof(icon_group));
		ret += res_.get_resource_data_by_id(language, pe_resource_viewer::resource_icon, group->Number).get_data();
	}

	return ret;
}

//Returns icon data by ID and index in language directory (instead of language) (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_icon_by_id(uint32_t id, uint32_t index) const
{
	std::string ret;

	//Get resource by id and index
	const std::string data = res_.get_resource_data_by_id(pe_resource_viewer::resource_icon_group, id, index).get_data();

	//Create icon headers
	uint16_t icon_count = format_icon_headers(ret, data);

	//Append icon data
	for(uint16_t i = 0; i != icon_count; ++i)
	{
		const icon_group* group = reinterpret_cast<const icon_group*>(data.data() + sizeof(ico_header) + i * sizeof(icon_group));
		ret += res_.get_resource_data_by_id(pe_resource_viewer::resource_icon, group->Number, index).get_data();
	}

	return ret;
}

//Checks for icon presence inside icon group, fills icon headers if found
bool resource_cursor_icon_reader::check_icon_presence(const std::string& icon_group_resource_data, uint32_t icon_id, std::string& ico_data)
{
	//Check resource data size
	if(icon_group_resource_data.length() < sizeof(ico_header))
		throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

	//Get icon header
	const ico_header* info = reinterpret_cast<const ico_header*>(icon_group_resource_data.data());

	//Check resource data size
	if(icon_group_resource_data.length() < sizeof(ico_header) + info->Count * sizeof(icon_group))
		throw pe_exception("Incorrect resource icon", pe_exception::resource_incorrect_icon);

	for(uint16_t i = 0; i != info->Count; ++i)
	{
		const icon_group* group = reinterpret_cast<const icon_group*>(icon_group_resource_data.data() + sizeof(ico_header) + i * sizeof(icon_group));
		if(group->Number == icon_id)
		{
			//Reserve memory to speed up a little
			ico_data.reserve(sizeof(ico_header) + sizeof(icondirentry));
			//Write single-icon icon header
			ico_header new_header = *info;
			new_header.Count = 1;
			ico_data.append(reinterpret_cast<const char*>(&new_header), sizeof(ico_header));

			//Fill icon data
			icondirentry direntry;
			direntry.BitCount = group->BitCount;
			direntry.ColorCount = group->ColorCount;
			direntry.Height = group->Height;
			direntry.Planes = group->Planes;
			direntry.Reserved = group->Reserved;
			direntry.SizeInBytes = group->SizeInBytes;
			direntry.Width = group->Width;
			direntry.ImageOffset = sizeof(ico_header) + sizeof(icondirentry);
			ico_data.append(reinterpret_cast<const char*>(&direntry), sizeof(direntry));

			return true;
		}
	}

	return false;
}

//Looks up icon group by icon id and returns full icon headers if found
const std::string resource_cursor_icon_reader::lookup_icon_group_data_by_icon(uint32_t icon_id, uint32_t language) const
{
	std::string icon_header_data;

	{
		//List all ID-resources
		pe_resource_viewer::resource_id_list ids(res_.list_resource_ids(pe_resource_viewer::resource_icon_group));

		for(pe_resource_viewer::resource_id_list::const_iterator it = ids.begin(); it != ids.end(); ++it)
		{
			pe_resource_viewer::resource_language_list group_languages(res_.list_resource_languages(pe_resource_viewer::resource_icon_group, *it));
			if(std::find(group_languages.begin(), group_languages.end(), language) != group_languages.end()
				&& check_icon_presence(res_.get_resource_data_by_id(language, pe_resource_viewer::resource_icon_group, *it).get_data(), icon_id, icon_header_data))
				return icon_header_data;
		}
	}

	{
		//List all named resources
		pe_resource_viewer::resource_name_list names(res_.list_resource_names(pe_resource_viewer::resource_icon_group));
		for(pe_resource_viewer::resource_name_list::const_iterator it = names.begin(); it != names.end(); ++it)
		{
			pe_resource_viewer::resource_language_list group_languages(res_.list_resource_languages(pe_resource_viewer::resource_icon_group, *it));
			if(std::find(group_languages.begin(), group_languages.end(), language) != group_languages.end()
				&& check_icon_presence(res_.get_resource_data_by_name(language, pe_resource_viewer::resource_icon_group, *it).get_data(), icon_id, icon_header_data))
				return icon_header_data;
		}
	}

	throw pe_exception("No icon group find for requested icon", pe_exception::no_icon_group_found);
}

//Returns single cursor data by ID and language (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_single_cursor_by_id_lang(uint32_t language, uint32_t id) const
{
	std::string raw_cursor_data(res_.get_resource_data_by_id(language, pe_resource_viewer::resource_cursor, id).get_data());
	//Get cursor headers
	std::string cursor_data(lookup_cursor_group_data_by_cursor(id, language, raw_cursor_data));
	//Append cursor data
	cursor_data.append(raw_cursor_data.substr(sizeof(uint16_t) * 2 /* hotspot position */));
	return cursor_data;
}

//Returns single cursor data by ID and index in language directory (instead of language) (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_single_cursor_by_id(uint32_t id, uint32_t index) const
{
	pe_resource_viewer::resource_language_list languages(res_.list_resource_languages(pe_resource_viewer::resource_cursor, id));
	if(languages.size() <= index)
		throw pe_exception("Resource data entry not found", pe_exception::resource_data_entry_not_found);
	
	std::string raw_cursor_data(res_.get_resource_data_by_id(pe_resource_viewer::resource_cursor, id, index).get_data());
	//Get cursor headers
	std::string cursor_data(lookup_cursor_group_data_by_cursor(id, languages.at(index), raw_cursor_data));
	//Append cursor data
	cursor_data.append(raw_cursor_data.substr(sizeof(uint16_t) * 2 /* hotspot position */));
	return cursor_data;
}

//Helper function of creating cursor headers
//Returns cursor count
uint16_t resource_cursor_icon_reader::format_cursor_headers(std::string& cur_data, const std::string& resource_data, uint32_t language, uint32_t index) const
{
	//Check resource data length
	if(resource_data.length() < sizeof(cursor_header))
		throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

	const cursor_header* info = reinterpret_cast<const cursor_header*>(resource_data.data());

	//Check resource data length
	if(resource_data.length() < sizeof(cursor_header) + sizeof(cursor_group) * info->Count)
		throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

	//Reserve needed space to speed up a little
	cur_data.reserve(sizeof(cursor_header) + info->Count * sizeof(cursordirentry));
	//Add cursor header
	cur_data.append(reinterpret_cast<const char*>(info), sizeof(cursor_header));

	//Iterate over all cursors listed in cursor group
	uint32_t offset = sizeof(cursor_header) + sizeof(cursordirentry) * info->Count;
	for(uint16_t i = 0; i != info->Count; ++i)
	{
		const cursor_group* group = reinterpret_cast<const cursor_group*>(resource_data.data() + sizeof(cursor_header) + i * sizeof(cursor_group));

		//Fill cursor info
		cursordirentry direntry;
		direntry.ColorCount = 0; //OK
		direntry.Width = static_cast<uint8_t>(group->Width);
		direntry.Height = static_cast<uint8_t>(group->Height)  / 2;
		direntry.Reserved = 0;

		//Now read hotspot data from cursor data directory
		const std::string cursor = index == 0xFFFFFFFF
			? res_.get_resource_data_by_id(language, pe_resource_viewer::resource_cursor, group->Number).get_data()
			: res_.get_resource_data_by_id(pe_resource_viewer::resource_cursor, group->Number, index).get_data();
		if(cursor.length() < 2 * sizeof(uint16_t))
			throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

		//Here it is - two words in the very beginning of cursor data
		direntry.HotspotX = *reinterpret_cast<const uint16_t*>(cursor.data());
		direntry.HotspotY = *reinterpret_cast<const uint16_t*>(cursor.data() + sizeof(uint16_t));

		//Fill the rest data
		direntry.SizeInBytes = group->SizeInBytes - 2 * sizeof(uint16_t);
		direntry.ImageOffset = offset;

		//Add cursor header
		cur_data.append(reinterpret_cast<const char*>(&direntry), sizeof(cursordirentry));

		offset += direntry.SizeInBytes;
	}

	//Return cursor count
	return info->Count;
}

//Returns cursor data by name and language (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_cursor_by_name(uint32_t language, const std::wstring& name) const
{
	std::string ret;

	//Get resource by name and language
	const std::string resource_data = res_.get_resource_data_by_name(language, pe_resource_viewer::resource_cursor_group, name).get_data();

	//Create cursor headers
	uint16_t cursor_count = format_cursor_headers(ret, resource_data, language);

	//Add cursor data
	for(uint16_t i = 0; i != cursor_count; ++i)
	{
		const cursor_group* group = reinterpret_cast<const cursor_group*>(resource_data.data() + sizeof(cursor_header) + i * sizeof(cursor_group));
		ret += res_.get_resource_data_by_id(language, pe_resource_viewer::resource_cursor, group->Number).get_data().substr(2 * sizeof(uint16_t));
	}

	return ret;
}

//Returns cursor data by name and index in language directory (instead of language) (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_cursor_by_name(const std::wstring& name, uint32_t index) const
{
	std::string ret;

	//Get resource by name and index
	const std::string resource_data = res_.get_resource_data_by_name(pe_resource_viewer::resource_cursor_group, name, index).get_data();

	//Create cursor headers
	uint16_t cursor_count = format_cursor_headers(ret, resource_data, 0, index);

	//Add cursor data
	for(uint16_t i = 0; i != cursor_count; ++i)
	{
		const cursor_group* group = reinterpret_cast<const cursor_group*>(resource_data.data() + sizeof(cursor_header) + i * sizeof(cursor_group));
		ret += res_.get_resource_data_by_id(pe_resource_viewer::resource_cursor, group->Number, index).get_data().substr(2 * sizeof(uint16_t));
	}

	return ret;
}

//Returns cursor data by ID and language (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_cursor_by_id_lang(uint32_t language, uint32_t id) const
{
	std::string ret;

	//Get resource by ID and language
	const std::string resource_data = res_.get_resource_data_by_id(language, pe_resource_viewer::resource_cursor_group, id).get_data();

	//Create cursor headers
	uint16_t cursor_count = format_cursor_headers(ret, resource_data, language);

	//Add cursor data
	for(uint16_t i = 0; i != cursor_count; ++i)
	{
		const cursor_group* group = reinterpret_cast<const cursor_group*>(resource_data.data() + sizeof(cursor_header) + i * sizeof(cursor_group));
		ret += res_.get_resource_data_by_id(language, pe_resource_viewer::resource_cursor, group->Number).get_data().substr(2 * sizeof(uint16_t));
	}

	return ret;
}

//Returns cursor data by ID and index in language directory (instead of language) (minimum checks of format correctness)
const std::string resource_cursor_icon_reader::get_cursor_by_id(uint32_t id, uint32_t index) const
{
	std::string ret;

	//Get resource by ID and index
	const std::string resource_data = res_.get_resource_data_by_id(pe_resource_viewer::resource_cursor_group, id, index).get_data();

	//Create cursor headers
	uint16_t cursor_count = format_cursor_headers(ret, resource_data, 0, index);

	//Add cursor data
	for(uint16_t i = 0; i != cursor_count; ++i)
	{
		const cursor_group* group = reinterpret_cast<const cursor_group*>(resource_data.data() + sizeof(cursor_header) + i * sizeof(cursor_group));
		ret += res_.get_resource_data_by_id(pe_resource_viewer::resource_cursor, group->Number, index).get_data().substr(2 * sizeof(uint16_t));
	}

	return ret;
}

//Checks for cursor presence inside cursor group, fills cursor headers if found
bool resource_cursor_icon_reader::check_cursor_presence(const std::string& cursor_group_resource_data, uint32_t cursor_id, std::string& cur_header_data, const std::string& raw_cursor_data)
{
	//Check resource data length
	if(cursor_group_resource_data.length() < sizeof(cursor_header))
		throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

	const cursor_header* info = reinterpret_cast<const cursor_header*>(cursor_group_resource_data.data());

	//Check resource data length
	if(cursor_group_resource_data.length() < sizeof(cursor_header) + sizeof(cursor_group))
		throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

	//Iterate over all cursors listed in cursor group
	for(uint16_t i = 0; i != info->Count; ++i)
	{
		const cursor_group* group = reinterpret_cast<const cursor_group*>(cursor_group_resource_data.data() + sizeof(cursor_header) + i * sizeof(cursor_group));

		if(group->Number == cursor_id)
		{
			//Reserve needed space to speed up a little
			cur_header_data.reserve(sizeof(cursor_header) + sizeof(cursordirentry));
			//Write single-cursor cursor header
			cursor_header new_header = *info;
			new_header.Count = 1;
			cur_header_data.append(reinterpret_cast<const char*>(&new_header), sizeof(cursor_header));

			//Fill cursor info
			cursordirentry direntry;
			direntry.ColorCount = 0; //OK
			direntry.Width = static_cast<uint8_t>(group->Width);
			direntry.Height = static_cast<uint8_t>(group->Height)  / 2;
			direntry.Reserved = 0;

			if(raw_cursor_data.length() < 2 * sizeof(uint16_t))
				throw pe_exception("Incorrect resource cursor", pe_exception::resource_incorrect_cursor);

			//Here it is - two words in the very beginning of cursor data
			direntry.HotspotX = *reinterpret_cast<const uint16_t*>(raw_cursor_data.data());
			direntry.HotspotY = *reinterpret_cast<const uint16_t*>(raw_cursor_data.data() + sizeof(uint16_t));

			//Fill the rest data
			direntry.SizeInBytes = group->SizeInBytes - 2 * sizeof(uint16_t);
			direntry.ImageOffset = sizeof(cursor_header) + sizeof(cursordirentry);

			//Add cursor header
			cur_header_data.append(reinterpret_cast<const char*>(&direntry), sizeof(cursordirentry));

			return true;
		}
	}

	return false;
}

//Looks up cursor group by cursor id and returns full cursor headers if found
const std::string resource_cursor_icon_reader::lookup_cursor_group_data_by_cursor(uint32_t cursor_id, uint32_t language, const std::string& raw_cursor_data) const
{
	std::string cursor_header_data;

	{
		//List all ID-resources
		pe_resource_viewer::resource_id_list ids(res_.list_resource_ids(pe_resource_viewer::resource_cursor_group));

		for(pe_resource_viewer::resource_id_list::const_iterator it = ids.begin(); it != ids.end(); ++it)
		{
			pe_resource_viewer::resource_language_list group_languages(res_.list_resource_languages(pe_resource_viewer::resource_cursor_group, *it));
			if(std::find(group_languages.begin(), group_languages.end(), language) != group_languages.end()
				&& check_cursor_presence(res_.get_resource_data_by_id(language, pe_resource_viewer::resource_cursor_group, *it).get_data(), cursor_id, cursor_header_data, raw_cursor_data))
				return cursor_header_data;
		}
	}

	{
		//List all named resources
		pe_resource_viewer::resource_name_list names(res_.list_resource_names(pe_resource_viewer::resource_cursor_group));
		for(pe_resource_viewer::resource_name_list::const_iterator it = names.begin(); it != names.end(); ++it)
		{
			pe_resource_viewer::resource_language_list group_languages(res_.list_resource_languages(pe_resource_viewer::resource_cursor_group, *it));
			if(std::find(group_languages.begin(), group_languages.end(), language) != group_languages.end()
				&& check_cursor_presence(res_.get_resource_data_by_name(language, pe_resource_viewer::resource_cursor_group, *it).get_data(), cursor_id, cursor_header_data, raw_cursor_data))
				return cursor_header_data;
		}
	}

	throw pe_exception("No cursor group find for requested icon", pe_exception::no_cursor_group_found);
}
}

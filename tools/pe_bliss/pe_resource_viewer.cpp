#include <algorithm>
#include <cmath>
#include "pe_resource_viewer.h"
#include "pe_structures.h"

namespace pe_bliss
{
using namespace pe_win;

//Constructor from root resource_directory
pe_resource_viewer::pe_resource_viewer(const resource_directory& root_directory)
	:root_dir_(root_directory)
{}

const resource_directory& pe_resource_viewer::get_root_directory() const
{
	return root_dir_;
}

//Finder helpers
bool pe_resource_viewer::has_name::operator()(const resource_directory_entry& entry) const
{
	return entry.is_named();
}

bool pe_resource_viewer::has_id::operator()(const resource_directory_entry& entry) const
{
	return !entry.is_named();
}

//Lists resource types existing in PE file (non-named only)
const pe_resource_viewer::resource_type_list pe_resource_viewer::list_resource_types() const
{
	resource_type_list ret;

	//Get root directory entries list
	const resource_directory::entry_list& entries = root_dir_.get_entry_list();
	for(resource_directory::entry_list::const_iterator it = entries.begin(); it != entries.end(); ++it)
	{
		//List all non-named items
		if(!(*it).is_named())
			ret.push_back((*it).get_id());
	}

	return ret;
}

//Returns true if resource type exists
bool pe_resource_viewer::resource_exists(resource_type type) const
{
	const resource_directory::entry_list& entries = root_dir_.get_entry_list();
	return std::find_if(entries.begin(), entries.end(), resource_directory::id_entry_finder(type)) != entries.end();
}

//Returns true if resource name exists
bool pe_resource_viewer::resource_exists(const std::wstring& root_name) const
{
	const resource_directory::entry_list& entries = root_dir_.get_entry_list();
	return std::find_if(entries.begin(), entries.end(), resource_directory::name_entry_finder(root_name)) != entries.end();
}

//Helper function to get name list from entry list
const pe_resource_viewer::resource_name_list pe_resource_viewer::get_name_list(const resource_directory::entry_list& entries)
{
	resource_name_list ret;

	for(resource_directory::entry_list::const_iterator it = entries.begin(); it != entries.end(); ++it)
	{
		//List all named items
		if((*it).is_named())
			ret.push_back((*it).get_name());
	}

	return ret;
}

//Helper function to get ID list from entry list
const pe_resource_viewer::resource_id_list pe_resource_viewer::get_id_list(const resource_directory::entry_list& entries)
{
	resource_id_list ret;

	for(resource_directory::entry_list::const_iterator it = entries.begin(); it != entries.end(); ++it)
	{
		//List all non-named items
		if(!(*it).is_named())
			ret.push_back((*it).get_id());
	}

	return ret;
}

//Lists resource names existing in PE file by resource type
const pe_resource_viewer::resource_name_list pe_resource_viewer::list_resource_names(resource_type type) const
{
	return get_name_list(root_dir_.entry_by_id(type).get_resource_directory().get_entry_list());
}

//Lists resource names existing in PE file by resource name
const pe_resource_viewer::resource_name_list pe_resource_viewer::list_resource_names(const std::wstring& root_name) const
{
	return get_name_list(root_dir_.entry_by_name(root_name).get_resource_directory().get_entry_list());
}

//Lists resource IDs existing in PE file by resource type
const pe_resource_viewer::resource_id_list pe_resource_viewer::list_resource_ids(resource_type type) const
{
	return get_id_list(root_dir_.entry_by_id(type).get_resource_directory().get_entry_list());
}

//Lists resource IDs existing in PE file by resource name
const pe_resource_viewer::resource_id_list pe_resource_viewer::list_resource_ids(const std::wstring& root_name) const
{
	return get_id_list(root_dir_.entry_by_name(root_name).get_resource_directory().get_entry_list());
}

//Returns resource count by type
unsigned long pe_resource_viewer::get_resource_count(resource_type type) const
{
	return static_cast<unsigned long>(
		root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.get_entry_list()
		.size());
}

//Returns resource count by name
unsigned long pe_resource_viewer::get_resource_count(const std::wstring& root_name) const
{
	return static_cast<unsigned long>(
		root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.get_entry_list()
		.size());
}

//Returns language count of resource by resource type and name
unsigned long pe_resource_viewer::get_language_count(resource_type type, const std::wstring& name) const
{
	const resource_directory::entry_list& entries =
		root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.entry_by_name(name)
		.get_resource_directory() //Language directory
		.get_entry_list();

	return static_cast<unsigned long>(std::count_if(entries.begin(), entries.end(), has_id()));
}

//Returns language count of resource by resource names
unsigned long pe_resource_viewer::get_language_count(const std::wstring& root_name, const std::wstring& name) const
{
	const resource_directory::entry_list& entries =
		root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.entry_by_name(name)
		.get_resource_directory() //Language directory
		.get_entry_list();

	return static_cast<unsigned long>(std::count_if(entries.begin(), entries.end(), has_id()));
}

//Returns language count of resource by resource type and ID
unsigned long pe_resource_viewer::get_language_count(resource_type type, uint32_t id) const
{
	const resource_directory::entry_list& entries =
		root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(id)
		.get_resource_directory() //Language directory
		.get_entry_list();

	return static_cast<unsigned long>(std::count_if(entries.begin(), entries.end(), has_id()));
}

//Returns language count of resource by resource name and ID
unsigned long pe_resource_viewer::get_language_count(const std::wstring& root_name, uint32_t id) const
{
	const resource_directory::entry_list& entries =
		root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(id)
		.get_resource_directory() //Language directory
		.get_entry_list();

	return static_cast<unsigned long>(std::count_if(entries.begin(), entries.end(), has_id()));
}

//Lists resource languages by resource type and name
const pe_resource_viewer::resource_language_list pe_resource_viewer::list_resource_languages(resource_type type, const std::wstring& name) const
{
	const resource_directory::entry_list& entries =
		root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.entry_by_name(name)
		.get_resource_directory() //Language directory
		.get_entry_list();

	return get_id_list(entries);
}

//Lists resource languages by resource names
const pe_resource_viewer::resource_language_list pe_resource_viewer::list_resource_languages(const std::wstring& root_name, const std::wstring& name) const
{
	const resource_directory::entry_list& entries =
		root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.entry_by_name(name)
		.get_resource_directory() //Language directory
		.get_entry_list();

	return get_id_list(entries);
}

//Lists resource languages by resource type and ID
const pe_resource_viewer::resource_language_list pe_resource_viewer::list_resource_languages(resource_type type, uint32_t id) const
{
	const resource_directory::entry_list& entries =
		root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(id)
		.get_resource_directory() //Language directory
		.get_entry_list();

	return get_id_list(entries);
}

//Lists resource languages by resource name and ID
const pe_resource_viewer::resource_language_list pe_resource_viewer::list_resource_languages(const std::wstring& root_name, uint32_t id) const
{
	const resource_directory::entry_list& entries =
		root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(id)
		.get_resource_directory() //Language directory
		.get_entry_list();

	return get_id_list(entries);
}

//Returns raw resource data by type, name and language
const resource_data_info pe_resource_viewer::get_resource_data_by_name(uint32_t language, resource_type type, const std::wstring& name) const
{
	return resource_data_info(root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.entry_by_name(name)
		.get_resource_directory() //Language directory
		.entry_by_id(language)
		.get_data_entry()); //Data directory
}

//Returns raw resource data by root name, name and language
const resource_data_info pe_resource_viewer::get_resource_data_by_name(uint32_t language, const std::wstring& root_name, const std::wstring& name) const
{
	return resource_data_info(root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.entry_by_name(name)
		.get_resource_directory() //Language directory
		.entry_by_id(language)
		.get_data_entry()); //Data directory
}

//Returns raw resource data by type, ID and language
const resource_data_info pe_resource_viewer::get_resource_data_by_id(uint32_t language, resource_type type, uint32_t id) const
{
	return resource_data_info(root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(id)
		.get_resource_directory() //Language directory
		.entry_by_id(language)
		.get_data_entry()); //Data directory
}

//Returns raw resource data by root name, ID and language
const resource_data_info pe_resource_viewer::get_resource_data_by_id(uint32_t language, const std::wstring& root_name, uint32_t id) const
{
	return resource_data_info(root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(id)
		.get_resource_directory() //Language directory
		.entry_by_id(language)
		.get_data_entry()); //Data directory
}

//Returns raw resource data by type, name and index in language directory (instead of language)
const resource_data_info pe_resource_viewer::get_resource_data_by_name(resource_type type, const std::wstring& name, uint32_t index) const
{
	const resource_directory::entry_list& entries = root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.entry_by_name(name)
		.get_resource_directory() //Language directory
		.get_entry_list();

	if(entries.size() <= index)
		throw pe_exception("Resource data entry not found", pe_exception::resource_data_entry_not_found);

	return resource_data_info(entries.at(index).get_data_entry()); //Data directory
}

//Returns raw resource data by root name, name and index in language directory (instead of language)
const resource_data_info pe_resource_viewer::get_resource_data_by_name(const std::wstring& root_name, const std::wstring& name, uint32_t index) const
{
	const resource_directory::entry_list& entries = root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.entry_by_name(name)
		.get_resource_directory() //Language directory
		.get_entry_list();

	if(entries.size() <= index)
		throw pe_exception("Resource data entry not found", pe_exception::resource_data_entry_not_found);

	return resource_data_info(entries.at(index).get_data_entry()); //Data directory
}

//Returns raw resource data by type, ID and index in language directory (instead of language)
const resource_data_info pe_resource_viewer::get_resource_data_by_id(resource_type type, uint32_t id, uint32_t index) const
{
	const resource_directory::entry_list& entries = root_dir_ //Type directory
		.entry_by_id(type)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(id)
		.get_resource_directory() //Language directory
		.get_entry_list();

	if(entries.size() <= index)
		throw pe_exception("Resource data entry not found", pe_exception::resource_data_entry_not_found);

	return resource_data_info(entries.at(index).get_data_entry()); //Data directory
}

//Returns raw resource data by root name, ID and index in language directory (instead of language)
const resource_data_info pe_resource_viewer::get_resource_data_by_id(const std::wstring& root_name, uint32_t id, uint32_t index) const
{
	const resource_directory::entry_list& entries = root_dir_ //Type directory
		.entry_by_name(root_name)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(id)
		.get_resource_directory() //Language directory
		.get_entry_list();

	if(entries.size() <= index)
		throw pe_exception("Resource data entry not found", pe_exception::resource_data_entry_not_found);

	return resource_data_info(entries.at(index).get_data_entry()); //Data directory
}
}

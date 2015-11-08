#include <algorithm>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <string.h>
#include "pe_resource_manager.h"
#include "resource_internal.h"

namespace pe_bliss
{
using namespace pe_win;

//Constructor from root resource directory
pe_resource_manager::pe_resource_manager(resource_directory& root_directory)
	:pe_resource_viewer(root_directory), root_dir_edit_(root_directory)
{}

resource_directory& pe_resource_manager::get_root_directory()
{
	return root_dir_edit_;
}

//Removes all resources of given type or root name
//If there's more than one directory entry of a given type, only the
//first one will be deleted (that's an unusual situation)
//Returns true if resource was deleted
bool pe_resource_manager::remove_resource_type(resource_type type)
{
	//Search for resource type
	resource_directory::entry_list& entries = root_dir_edit_.get_entry_list();
	resource_directory::entry_list::iterator it = std::find_if(entries.begin(), entries.end(), resource_directory::id_entry_finder(type));
	if(it != entries.end())
	{
		//Remove it, if found
		entries.erase(it);
		return true;
	}

	return false;
}

bool pe_resource_manager::remove_resource(const std::wstring& root_name)
{
	//Search for resource type
	resource_directory::entry_list& entries = root_dir_edit_.get_entry_list();
	resource_directory::entry_list::iterator it = std::find_if(entries.begin(), entries.end(), resource_directory::name_entry_finder(root_name));
	if(it != entries.end())
	{
		//Remove it, if found
		entries.erase(it);
		return true;
	}

	return false;
}

//Helper to remove resource
bool pe_resource_manager::remove_resource(const resource_directory::entry_finder& root_finder, const resource_directory::entry_finder& finder)
{
	//Search for resource type
	resource_directory::entry_list& entries_type = root_dir_edit_.get_entry_list();
	resource_directory::entry_list::iterator it_type = std::find_if(entries_type.begin(), entries_type.end(), root_finder);
	if(it_type != entries_type.end())
	{
		//Search for resource name/ID with "finder"
		resource_directory::entry_list& entries_name = (*it_type).get_resource_directory().get_entry_list();
		resource_directory::entry_list::iterator it_name = std::find_if(entries_name.begin(), entries_name.end(), finder);
		if(it_name != entries_name.end())
		{
			//Erase resource, if found
			entries_name.erase(it_name);
			if(entries_name.empty())
				entries_type.erase(it_type);

			return true;
		}
	}

	return false;
}

//Removes all resource languages by resource type/root name and name
//Deletes only one entry of given type and name
//Returns true if resource was deleted
bool pe_resource_manager::remove_resource(resource_type type, const std::wstring& name)
{
	return remove_resource(resource_directory::entry_finder(type), resource_directory::entry_finder(name));
}

bool pe_resource_manager::remove_resource(const std::wstring& root_name, const std::wstring& name)
{
	return remove_resource(resource_directory::entry_finder(root_name), resource_directory::entry_finder(name));
}

//Removes all resource languages by resource type/root name and ID
//Deletes only one entry of given type and ID
//Returns true if resource was deleted
bool pe_resource_manager::remove_resource(resource_type type, uint32_t id)
{
	return remove_resource(resource_directory::entry_finder(type), resource_directory::entry_finder(id));
}

bool pe_resource_manager::remove_resource(const std::wstring& root_name, uint32_t id)
{
	return remove_resource(resource_directory::entry_finder(root_name), resource_directory::entry_finder(id));
}

//Helper to remove resource
bool pe_resource_manager::remove_resource(const resource_directory::entry_finder& root_finder, const resource_directory::entry_finder& finder, uint32_t language)
{
	//Search for resource type
	resource_directory::entry_list& entries_type = root_dir_edit_.get_entry_list();
	resource_directory::entry_list::iterator it_type = std::find_if(entries_type.begin(), entries_type.end(), root_finder);
	if(it_type != entries_type.end())
	{
		//Search for resource name/ID with "finder"
		resource_directory::entry_list& entries_name = (*it_type).get_resource_directory().get_entry_list();
		resource_directory::entry_list::iterator it_name = std::find_if(entries_name.begin(), entries_name.end(), finder);
		if(it_name != entries_name.end())
		{
			//Search for resource language
			resource_directory::entry_list& entries_lang = (*it_name).get_resource_directory().get_entry_list();
			resource_directory::entry_list::iterator it_lang = std::find_if(entries_lang.begin(), entries_lang.end(), resource_directory::id_entry_finder(language));
			if(it_lang != entries_lang.end())
			{
				//Erase resource, if found
				entries_lang.erase(it_lang);
				if(entries_lang.empty())
				{
					entries_name.erase(it_name);
					if(entries_name.empty())
						entries_type.erase(it_type);
				}

				return true;
			}
		}
	}

	return false;
}

//Removes resource language by resource type/root name and name
//Deletes only one entry of given type, name and language
//Returns true if resource was deleted
bool pe_resource_manager::remove_resource(resource_type type, const std::wstring& name, uint32_t language)
{
	return remove_resource(resource_directory::entry_finder(type), resource_directory::entry_finder(name), language);
}

bool pe_resource_manager::remove_resource(const std::wstring& root_name, const std::wstring& name, uint32_t language)
{
	return remove_resource(resource_directory::entry_finder(root_name), resource_directory::entry_finder(name), language);
}

//Removes recource language by resource type/root name and ID
//Deletes only one entry of given type, ID and language
//Returns true if resource was deleted
bool pe_resource_manager::remove_resource(resource_type type, uint32_t id, uint32_t language)
{
	return remove_resource(resource_directory::entry_finder(type), resource_directory::entry_finder(id), language);
}

bool pe_resource_manager::remove_resource(const std::wstring& root_name, uint32_t id, uint32_t language)
{
	return remove_resource(resource_directory::entry_finder(root_name), resource_directory::entry_finder(id), language);
}

//Helper to add/replace resource
void pe_resource_manager::add_resource(const std::string& data, resource_type type, resource_directory_entry& new_entry, const resource_directory::entry_finder& finder, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_type_entry;
	new_type_entry.set_id(type);

	add_resource(data, new_type_entry, resource_directory::entry_finder(type), new_entry, finder, language, codepage, timestamp);
}

//Helper to add/replace resource
void pe_resource_manager::add_resource(const std::string& data, const std::wstring& root_name, resource_directory_entry& new_entry, const resource_directory::entry_finder& finder, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_type_entry;
	new_type_entry.set_name(root_name);
	
	add_resource(data, new_type_entry, resource_directory::entry_finder(root_name), new_entry, finder, language, codepage, timestamp);
}

//Helper to add/replace resource
void pe_resource_manager::add_resource(const std::string& data, resource_directory_entry& new_root_entry, const resource_directory::entry_finder& root_finder, resource_directory_entry& new_entry, const resource_directory::entry_finder& finder, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	//Search for resource type
	resource_directory::entry_list* entries = &root_dir_edit_.get_entry_list();
	resource_directory::entry_list::iterator it = std::find_if(entries->begin(), entries->end(), root_finder);
	if(it == entries->end())
	{
		//Add resource type directory, if it was not found
		resource_directory dir;
		dir.set_timestamp(timestamp);
		new_root_entry.add_resource_directory(dir);
		entries->push_back(new_root_entry);
		it = entries->end() - 1;
	}

	//Search for resource name/ID directory with "finder"
	entries = &(*it).get_resource_directory().get_entry_list();
	it = std::find_if(entries->begin(), entries->end(), finder);
	if(it == entries->end())
	{
		//Add resource name/ID directory, if it was not found
		resource_directory dir;
		dir.set_timestamp(timestamp);
		new_entry.add_resource_directory(dir);
		entries->push_back(new_entry);
		it = entries->end() - 1;
	}

	//Search for data resource entry by language
	entries = &(*it).get_resource_directory().get_entry_list();
	it = std::find_if(entries->begin(), entries->end(), resource_directory::id_entry_finder(language));
	if(it != entries->end())
		entries->erase(it); //Erase it, if found

	//Add new data entry
	resource_directory_entry new_dir_data_entry;
	resource_data_entry data_dir(data, codepage);
	new_dir_data_entry.add_data_entry(data_dir);
	new_dir_data_entry.set_id(language);
	entries->push_back(new_dir_data_entry);
}

//Adds resource. If resource already exists, replaces it
void pe_resource_manager::add_resource(const std::string& data, resource_type type, const std::wstring& name, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_entry;
	new_entry.set_name(name);

	add_resource(data, type, new_entry, resource_directory::entry_finder(name), language, codepage, timestamp);
}

//Adds resource. If resource already exists, replaces it
void pe_resource_manager::add_resource(const std::string& data, const std::wstring& root_name, const std::wstring& name, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_entry;
	new_entry.set_name(name);

	add_resource(data, root_name, new_entry, resource_directory::entry_finder(name), language, codepage, timestamp);
}

//Adds resource. If resource already exists, replaces it
void pe_resource_manager::add_resource(const std::string& data, resource_type type, uint32_t id, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_entry;
	new_entry.set_id(id);

	add_resource(data, type, new_entry, resource_directory::entry_finder(id), language, codepage, timestamp);
}

//Adds resource. If resource already exists, replaces it
void pe_resource_manager::add_resource(const std::string& data, const std::wstring& root_name, uint32_t id, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	resource_directory_entry new_entry;
	new_entry.set_id(id);

	add_resource(data, root_name, new_entry, resource_directory::entry_finder(id), language, codepage, timestamp);
}
}

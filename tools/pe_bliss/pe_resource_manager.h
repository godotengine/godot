#pragma once
#include <map>
#include <sstream>
#include <string>
#include <memory>
#include "pe_base.h"
#include "pe_structures.h"
#include "pe_resources.h"
#include "message_table.h"
#include "file_version_info.h"
#include "pe_resource_viewer.h"
#include "resource_data_info.h"

namespace pe_bliss
{
//Derived class to edit PE resources
class pe_resource_manager : public pe_resource_viewer
{
public:
	//Constructor from root resource directory
	explicit pe_resource_manager(resource_directory& root_directory);
	
	resource_directory& get_root_directory();

public: //Resource editing
	//Removes all resources of given type or root name
	//If there's more than one directory entry of a given type, only the
	//first one will be deleted (that's an unusual situation)
	//Returns true if resource was deleted
	bool remove_resource_type(resource_type type);
	bool remove_resource(const std::wstring& root_name);
	
	//Removes all resource languages by resource type/root name and name
	//Deletes only one entry of given type and name
	//Returns true if resource was deleted
	bool remove_resource(resource_type type, const std::wstring& name);
	bool remove_resource(const std::wstring& root_name, const std::wstring& name);
	//Removes all resource languages by resource type/root name and ID
	//Deletes only one entry of given type and ID
	//Returns true if resource was deleted
	bool remove_resource(resource_type type, uint32_t id);
	bool remove_resource(const std::wstring& root_name, uint32_t id);

	//Removes resource language by resource type/root name and name
	//Deletes only one entry of given type, name and language
	//Returns true if resource was deleted
	bool remove_resource(resource_type type, const std::wstring& name, uint32_t language);
	bool remove_resource(const std::wstring& root_name, const std::wstring& name, uint32_t language);
	//Removes recource language by resource type/root name and ID
	//Deletes only one entry of given type, ID and language
	//Returns true if resource was deleted
	bool remove_resource(resource_type type, uint32_t id, uint32_t language);
	bool remove_resource(const std::wstring& root_name, uint32_t id, uint32_t language);
	
	//Adds resource. If resource already exists, replaces it
	//timestamp will be used for directories that will be added
	void add_resource(const std::string& data, resource_type type, const std::wstring& name, uint32_t language, uint32_t codepage = 0, uint32_t timestamp = 0);
	void add_resource(const std::string& data, const std::wstring& root_name, const std::wstring& name, uint32_t language, uint32_t codepage = 0, uint32_t timestamp = 0);
	//Adds resource. If resource already exists, replaces it
	//timestamp will be used for directories that will be added
	void add_resource(const std::string& data, resource_type type, uint32_t id, uint32_t language, uint32_t codepage = 0, uint32_t timestamp = 0);
	void add_resource(const std::string& data, const std::wstring& root_name, uint32_t id, uint32_t language, uint32_t codepage = 0, uint32_t timestamp = 0);

public:
	//Helpers to add/replace resource
	void add_resource(const std::string& data, resource_type type,
		resource_directory_entry& new_entry,
		const resource_directory::entry_finder& finder,
		uint32_t language, uint32_t codepage, uint32_t timestamp);

	void add_resource(const std::string& data, const std::wstring& root_name,
		resource_directory_entry& new_entry,
		const resource_directory::entry_finder& finder,
		uint32_t language, uint32_t codepage, uint32_t timestamp);

	void add_resource(const std::string& data, resource_directory_entry& new_root_entry,
		const resource_directory::entry_finder& root_finder,
		resource_directory_entry& new_entry,
		const resource_directory::entry_finder& finder,
		uint32_t language, uint32_t codepage, uint32_t timestamp);

private:
	//Root resource directory. We're not copying it, because it might be heavy
	resource_directory& root_dir_edit_;

	//Helper to remove resource
	bool remove_resource(const resource_directory::entry_finder& root_finder, const resource_directory::entry_finder& finder);

	//Helper to remove resource
	bool remove_resource(const resource_directory::entry_finder& root_finder, const resource_directory::entry_finder& finder, uint32_t language);
};
}

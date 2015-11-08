#pragma once
#include <map>
#include <string>
#include "pe_structures.h"
#include "pe_resources.h"
#include "message_table.h"
#include "resource_data_info.h"

namespace pe_bliss
{
	//PE resource manager allows to read resources from PE files
class pe_resource_viewer
{
public:
	//Resource type enumeration
	enum resource_type
	{
		resource_cursor = 1,
		resource_bitmap = 2,
		resource_icon = 3,
		resource_menu = 4,
		resource_dialog = 5,
		resource_string = 6,
		resource_fontdir = 7,
		resource_font = 8,
		resource_accelerator = 9,
		resource_rcdata = 10,
		resource_message_table = 11,
		resource_cursor_group = 12,
		resource_icon_group = 14,
		resource_version = 16,
		resource_dlginclude = 17,
		resource_plugplay = 19,
		resource_vxd = 20,
		resource_anicursor = 21,
		resource_aniicon = 22,
		resource_html = 23,
		resource_manifest = 24
	};

public:
	//Some useful typedefs
	typedef std::vector<uint32_t> resource_type_list;
	typedef std::vector<uint32_t> resource_id_list;
	typedef std::vector<std::wstring> resource_name_list;
	typedef std::vector<uint32_t> resource_language_list;
	
public:
	//Constructor from root resource_directory from PE file
	explicit pe_resource_viewer(const resource_directory& root_directory);

	const resource_directory& get_root_directory() const;

	//Lists resource types existing in PE file (non-named only)
	const resource_type_list list_resource_types() const;
	//Returns true if resource type exists
	bool resource_exists(resource_type type) const;
	//Returns true if resource name exists
	bool resource_exists(const std::wstring& root_name) const;

	//Lists resource names existing in PE file by resource type
	const resource_name_list list_resource_names(resource_type type) const;
	//Lists resource names existing in PE file by resource name
	const resource_name_list list_resource_names(const std::wstring& root_name) const;
	//Lists resource IDs existing in PE file by resource type
	const resource_id_list list_resource_ids(resource_type type) const;
	//Lists resource IDs existing in PE file by resource name
	const resource_id_list list_resource_ids(const std::wstring& root_name) const;
	//Returns resource count by type
	unsigned long get_resource_count(resource_type type) const;
	//Returns resource count by name
	unsigned long get_resource_count(const std::wstring& root_name) const;

	//Returns language count of resource by resource type and name
	unsigned long get_language_count(resource_type type, const std::wstring& name) const;
	//Returns language count of resource by resource names
	unsigned long get_language_count(const std::wstring& root_name, const std::wstring& name) const;
	//Returns language count of resource by resource type and ID
	unsigned long get_language_count(resource_type type, uint32_t id) const;
	//Returns language count of resource by resource name and ID
	unsigned long get_language_count(const std::wstring& root_name, uint32_t id) const;
	//Lists resource languages by resource type and name
	const resource_language_list list_resource_languages(resource_type type, const std::wstring& name) const;
	//Lists resource languages by resource names
	const resource_language_list list_resource_languages(const std::wstring& root_name, const std::wstring& name) const;
	//Lists resource languages by resource type and ID
	const resource_language_list list_resource_languages(resource_type type, uint32_t id) const;
	//Lists resource languages by resource name and ID
	const resource_language_list list_resource_languages(const std::wstring& root_name, uint32_t id) const;

	//Returns raw resource data by type, name and language
	const resource_data_info get_resource_data_by_name(uint32_t language, resource_type type, const std::wstring& name) const;
	//Returns raw resource data by root name, name and language
	const resource_data_info get_resource_data_by_name(uint32_t language, const std::wstring& root_name, const std::wstring& name) const;
	//Returns raw resource data by type, ID and language
	const resource_data_info get_resource_data_by_id(uint32_t language, resource_type type, uint32_t id) const;
	//Returns raw resource data by root name, ID and language
	const resource_data_info get_resource_data_by_id(uint32_t language, const std::wstring& root_name, uint32_t id) const;
	//Returns raw resource data by type, name and index in language directory (instead of language)
	const resource_data_info get_resource_data_by_name(resource_type type, const std::wstring& name, uint32_t index = 0) const;
	//Returns raw resource data by root name, name and index in language directory (instead of language)
	const resource_data_info get_resource_data_by_name(const std::wstring& root_name, const std::wstring& name, uint32_t index = 0) const;
	//Returns raw resource data by type, ID and index in language directory (instead of language)
	const resource_data_info get_resource_data_by_id(resource_type type, uint32_t id, uint32_t index = 0) const;
	//Returns raw resource data by root name, ID and index in language directory (instead of language)
	const resource_data_info get_resource_data_by_id(const std::wstring& root_name, uint32_t id, uint32_t index = 0) const;

protected:
	//Root resource directory. We're not copying it, because it might be heavy
	const resource_directory& root_dir_;

	//Helper function to get ID list from entry list
	static const resource_id_list get_id_list(const resource_directory::entry_list& entries);
	//Helper function to get name list from entry list
	static const resource_name_list get_name_list(const resource_directory::entry_list& entries);

protected:
	//Helper structure - finder of resource_directory_entry that is named
	struct has_name
	{
	public:
		bool operator()(const resource_directory_entry& entry) const;
	};

	//Helper structure - finder of resource_directory_entry that is not named (has id)
	struct has_id
	{
	public:
		bool operator()(const resource_directory_entry& entry) const;
	};
};
}

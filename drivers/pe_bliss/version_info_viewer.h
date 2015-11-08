#pragma once
#include <map>
#include <vector>
#include <string>
#include "pe_resource_viewer.h"
#include "pe_structures.h"
#include "version_info_types.h"

namespace pe_bliss
{
//Helper class to read version information
//lang_string_values_map: map of version info strings with encodings
//translation_values_map: map of translations
class version_info_viewer
{
public:
	//Useful typedefs
	typedef std::pair<uint16_t, uint16_t> translation_pair;
	typedef std::vector<std::wstring> translation_list;

public:
	//Default constructor
	//strings - version info strings with charsets
	//translations - version info translations map
	version_info_viewer(const lang_string_values_map& strings, const translation_values_map& translations);

	//Below functions have parameter translation
	//If it's empty, the default language translation will be taken
	//If there's no default language translation, the first one will be taken

	//Returns company name
	const std::wstring get_company_name(const std::wstring& translation = std::wstring()) const;
	//Returns file description
	const std::wstring get_file_description(const std::wstring& translation = std::wstring()) const;
	//Returns file version
	const std::wstring get_file_version(const std::wstring& translation = std::wstring()) const;
	//Returns internal file name
	const std::wstring get_internal_name(const std::wstring& translation = std::wstring()) const;
	//Returns legal copyright
	const std::wstring get_legal_copyright(const std::wstring& translation = std::wstring()) const;
	//Returns original file name
	const std::wstring get_original_filename(const std::wstring& translation = std::wstring()) const;
	//Returns product name
	const std::wstring get_product_name(const std::wstring& translation = std::wstring()) const;
	//Returns product version
	const std::wstring get_product_version(const std::wstring& translation = std::wstring()) const;

	//Returns list of translations in string representation
	const translation_list get_translation_list() const;

	//Returns version info property value
	//property_name - required property name
	//If throw_if_absent = true, will throw exception if property does not exist
	//If throw_if_absent = false, will return empty string if property does not exist
	const std::wstring get_property(const std::wstring& property_name, const std::wstring& translation = std::wstring(), bool throw_if_absent = false) const;

	//Converts translation HEX-string to pair of language ID and codepage ID
	static const translation_pair translation_from_string(const std::wstring& translation);

public:
	//Default process language, UNICODE
	static const std::wstring default_language_translation;

private:
	const lang_string_values_map& strings_;
	const translation_values_map& translations_;
};
}

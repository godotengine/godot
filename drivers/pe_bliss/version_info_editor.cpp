#include <sstream>
#include <iomanip>
#include "version_info_types.h"
#include "version_info_editor.h"
#include "version_info_viewer.h"

namespace pe_bliss
{
//Default constructor
//strings - version info strings with charsets
//translations - version info translations map
version_info_editor::version_info_editor(lang_string_values_map& strings, translation_values_map& translations)
	:version_info_viewer(strings, translations),
	strings_edit_(strings),
	translations_edit_(translations)
{}

//Below functions have parameter translation
//If it's empty, the default language translation will be taken
//If there's no default language translation, the first one will be taken

//Sets company name
void version_info_editor::set_company_name(const std::wstring& value, const std::wstring& translation)
{
	set_property(L"CompanyName", value, translation);
}

//Sets file description
void version_info_editor::set_file_description(const std::wstring& value, const std::wstring& translation)
{
	set_property(L"FileDescription", value, translation);
}

//Sets file version
void version_info_editor::set_file_version(const std::wstring& value, const std::wstring& translation)
{
	set_property(L"FileVersion", value, translation);
}

//Sets internal file name
void version_info_editor::set_internal_name(const std::wstring& value, const std::wstring& translation)
{
	set_property(L"InternalName", value, translation);
}

//Sets legal copyright
void version_info_editor::set_legal_copyright(const std::wstring& value, const std::wstring& translation)
{
	set_property(L"LegalCopyright", value, translation);
}

//Sets original file name
void version_info_editor::set_original_filename(const std::wstring& value, const std::wstring& translation)
{
	set_property(L"OriginalFilename", value, translation);
}

//Sets product name
void version_info_editor::set_product_name(const std::wstring& value, const std::wstring& translation)
{
	set_property(L"ProductName", value, translation);
}

//Sets product version
void version_info_editor::set_product_version(const std::wstring& value, const std::wstring& translation)
{
	set_property(L"ProductVersion", value, translation);
}

//Sets version info property value
//property_name - property name
//value - property value
//If translation does not exist, it will be added
//If property does not exist, it will be added
void version_info_editor::set_property(const std::wstring& property_name, const std::wstring& value, const std::wstring& translation)
{
	lang_string_values_map::iterator it = strings_edit_.begin();

	if(translation.empty())
	{
		//If no translation was specified
		it = strings_edit_.find(default_language_translation); //Find default translation table
		if(it == strings_edit_.end()) //If there's no default translation table, take the first one
		{
			it = strings_edit_.begin();
			if(it == strings_edit_.end()) //If there's no any translation table, add default one
			{
				it = strings_edit_.insert(std::make_pair(default_language_translation, string_values_map())).first;
				//Also add it to translations list
				add_translation(default_language_translation);
			}
		}
	}
	else
	{
		it = strings_edit_.find(translation); //Find specified translation table
		if(it == strings_edit_.end()) //If there's no translation, add it
		{
			it = strings_edit_.insert(std::make_pair(translation, string_values_map())).first;
			//Also add it to translations list
			add_translation(translation);
		}
	}

	//Change value of the required property
	((*it).second)[property_name] = value;
}

//Adds translation to translation list
void version_info_editor::add_translation(const std::wstring& translation)
{
	std::pair<uint16_t, uint16_t> translation_ids(translation_from_string(translation));
	add_translation(translation_ids.first, translation_ids.second);
}

void version_info_editor::add_translation(uint16_t language_id, uint16_t codepage_id)
{
	std::pair<translation_values_map::const_iterator, translation_values_map::const_iterator>
		range(translations_edit_.equal_range(language_id));

	//If translation already exists
	for(translation_values_map::const_iterator it = range.first; it != range.second; ++it)
	{
		if((*it).second == codepage_id)
			return;
	}

	translations_edit_.insert(std::make_pair(language_id, codepage_id));
}

//Removes translation from translations and strings lists
void version_info_editor::remove_translation(const std::wstring& translation)
{
	std::pair<uint16_t, uint16_t> translation_ids(translation_from_string(translation));
	remove_translation(translation_ids.first, translation_ids.second);
}

void version_info_editor::remove_translation(uint16_t language_id, uint16_t codepage_id)
{
	{
		//Erase string table (if exists)
		std::wstringstream ss;
		ss << std::hex
			<< std::setw(4) << std::setfill(L'0') << language_id
			<< std::setw(4) << std::setfill(L'0') << codepage_id;
		
		strings_edit_.erase(ss.str());
	}

	//Find and erase translation from translations table
	std::pair<translation_values_map::iterator, translation_values_map::iterator>
		it_pair = translations_edit_.equal_range(language_id);

	for(translation_values_map::iterator it = it_pair.first; it != it_pair.second; ++it)
	{
		if((*it).second == codepage_id)
		{
			translations_edit_.erase(it);
			break;
		}
	}
}
}

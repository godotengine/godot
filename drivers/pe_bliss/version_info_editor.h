#pragma once
#include "version_info_types.h"
#include "version_info_viewer.h"

namespace pe_bliss
{
	//Helper class to read and edit version information
	//lang_string_values_map: map of version info strings with encodings
	//translation_values_map: map of translations
	class version_info_editor : public version_info_viewer
	{
	public:
		//Default constructor
		//strings - version info strings with charsets
		//translations - version info translations map
		version_info_editor(lang_string_values_map& strings, translation_values_map& translations);

		//Below functions have parameter translation
		//If it's empty, the default language translation will be taken
		//If there's no default language translation, the first one will be taken

		//Sets company name
		void set_company_name(const std::wstring& value, const std::wstring& translation = std::wstring());
		//Sets file description
		void set_file_description(const std::wstring& value, const std::wstring& translation = std::wstring());
		//Sets file version
		void set_file_version(const std::wstring& value, const std::wstring& translation = std::wstring());
		//Sets internal file name
		void set_internal_name(const std::wstring& value, const std::wstring& translation = std::wstring());
		//Sets legal copyright
		void set_legal_copyright(const std::wstring& value, const std::wstring& translation = std::wstring());
		//Sets original file name
		void set_original_filename(const std::wstring& value, const std::wstring& translation = std::wstring());
		//Sets product name
		void set_product_name(const std::wstring& value, const std::wstring& translation = std::wstring());
		//Sets product version
		void set_product_version(const std::wstring& value, const std::wstring& translation = std::wstring());

		//Sets version info property value
		//property_name - property name
		//value - property value
		//If translation does not exist, it will be added to strings and translations lists
		//If property does not exist, it will be added
		void set_property(const std::wstring& property_name, const std::wstring& value, const std::wstring& translation = std::wstring());

		//Adds translation to translation list
		void add_translation(const std::wstring& translation);
		void add_translation(uint16_t language_id, uint16_t codepage_id);

		//Removes translation from translations and strings lists
		void remove_translation(const std::wstring& translation);
		void remove_translation(uint16_t language_id, uint16_t codepage_id);

	private:
		lang_string_values_map& strings_edit_;
		translation_values_map& translations_edit_;
	};
}

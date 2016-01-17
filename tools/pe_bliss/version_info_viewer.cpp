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
#include <iomanip>
#include <sstream>
#include "pe_exception.h"
#include "version_info_viewer.h"

namespace pe_bliss
{
//Default process language, UNICODE
const std::wstring version_info_viewer::default_language_translation(L"040904b0");

//Default constructor
//strings - version info strings with charsets
//translations - version info translations map
version_info_viewer::version_info_viewer(const lang_string_values_map& strings, const translation_values_map& translations)
	:strings_(strings), translations_(translations)
{}

//Below functions have parameter translation
//If it's empty, the default language translation will be taken
//If there's no default language translation, the first one will be taken

//Returns company name
const std::wstring version_info_viewer::get_company_name(const std::wstring& translation) const
{
	return get_property(L"CompanyName", translation);
}

//Returns file description
const std::wstring version_info_viewer::get_file_description(const std::wstring& translation) const
{
	return get_property(L"FileDescription", translation);
}

//Returns file version
const std::wstring version_info_viewer::get_file_version(const std::wstring& translation) const
{
	return get_property(L"FileVersion", translation);
}

//Returns internal file name
const std::wstring version_info_viewer::get_internal_name(const std::wstring& translation) const
{
	return get_property(L"InternalName", translation);
}

//Returns legal copyright
const std::wstring version_info_viewer::get_legal_copyright(const std::wstring& translation) const
{
	return get_property(L"LegalCopyright", translation);
}

//Returns original file name
const std::wstring version_info_viewer::get_original_filename(const std::wstring& translation) const
{
	return get_property(L"OriginalFilename", translation);
}

//Returns product name
const std::wstring version_info_viewer::get_product_name(const std::wstring& translation) const
{
	return get_property(L"ProductName", translation);
}

//Returns product version
const std::wstring version_info_viewer::get_product_version(const std::wstring& translation) const
{
	return get_property(L"ProductVersion", translation);
}

//Returns list of translations in string representation
const version_info_viewer::translation_list version_info_viewer::get_translation_list() const
{
	translation_list ret;

	//Enumerate all translations
	for(translation_values_map::const_iterator it = translations_.begin(); it != translations_.end(); ++it)
	{
		//Create string representation of translation value
		std::wstringstream ss;
		ss << std::hex
			<< std::setw(4) << std::setfill(L'0') << (*it).first
			<< std::setw(4) << std::setfill(L'0') <<  (*it).second;

		//Save it
		ret.push_back(ss.str());
	}

	return ret;
}

//Returns version info property value
//property_name - required property name
//If throw_if_absent = true, will throw exception if property does not exist
//If throw_if_absent = false, will return empty string if property does not exist
const std::wstring version_info_viewer::get_property(const std::wstring& property_name, const std::wstring& translation, bool throw_if_absent) const
{
	std::wstring ret;

	//If there're no strings
	if(strings_.empty())
	{
		if(throw_if_absent)
			throw pe_exception("Version info string does not exist", pe_exception::version_info_string_does_not_exist);

		return ret;
	}
	
	lang_string_values_map::const_iterator it = strings_.begin();

	if(translation.empty())
	{
		//If no translation was specified
		it = strings_.find(default_language_translation); //Find default translation table
		if(it == strings_.end()) //If there's no default translation table, take the first one
			it = strings_.begin();
	}
	else
	{
		it = strings_.find(translation); //Find specified translation table
		if(it == strings_.end())
		{
			if(throw_if_absent)
				throw pe_exception("Version info string does not exist", pe_exception::version_info_string_does_not_exist);

			return ret;
		}
	}
	
	//Find value of the required property
	string_values_map::const_iterator str_it = (*it).second.find(property_name);

	if(str_it == (*it).second.end())
	{
		if(throw_if_absent)
			throw pe_exception("Version info string does not exist", pe_exception::version_info_string_does_not_exist);

		return ret;
	}

	ret = (*str_it).second;

	return ret;
}

//Converts translation HEX-string to pair of language ID and codepage ID
const version_info_viewer::translation_pair version_info_viewer::translation_from_string(const std::wstring& translation)
{
	uint32_t translation_id = 0;

	{
		//Convert string to DWORD
		std::wstringstream ss;
		ss << std::hex << translation;
		ss >> translation_id;
	}

	return std::make_pair(static_cast<uint16_t>(translation_id >> 16), static_cast<uint16_t>(translation_id & 0xFFFF));
}
}

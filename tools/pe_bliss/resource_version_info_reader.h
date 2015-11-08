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
#pragma once
#include <map>
#include "file_version_info.h"
#include "pe_structures.h"
#include "version_info_types.h"

namespace pe_bliss
{
class pe_resource_viewer;

class resource_version_info_reader
{
public: //VERSION INFO
	resource_version_info_reader(const pe_resource_viewer& res);

	//Returns full version information:
	//file_version_info: versions and file info
	//lang_lang_string_values_map: map of version info strings with encodings with encodings
	//translation_values_map: map of translations
	const file_version_info get_version_info(lang_string_values_map& string_values, translation_values_map& translations, uint32_t index = 0) const;
	const file_version_info get_version_info_by_lang(lang_string_values_map& string_values, translation_values_map& translations, uint32_t language) const;

public:
	//L"VS_VERSION_INFO" key of root version info block
	static const u16string version_info_key;

private:
	const pe_resource_viewer& res_;
	
	//VERSION INFO helpers
	//Returns aligned version block value position
	static uint32_t get_version_block_value_pos(uint32_t base_pos, const unicode16_t* key);

	//Returns aligned version block first child position
	static uint32_t get_version_block_first_child_pos(uint32_t base_pos, uint32_t value_length, const unicode16_t* key);

	//Returns full version information:
	//file_version_info: versions and file info
	//lang_string_values_map: map of version info strings with encodings
	//translation_values_map: map of translations
	const file_version_info get_version_info(lang_string_values_map& string_values, translation_values_map& translations, const std::string& resource_data) const;

	//Throws an exception (id = resource_incorrect_version_info)
	static void throw_incorrect_version_info();
};
}

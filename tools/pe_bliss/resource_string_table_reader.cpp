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
#include "resource_string_table_reader.h"
#include "pe_resource_viewer.h"

namespace pe_bliss
{
resource_string_table_reader::resource_string_table_reader(const pe_resource_viewer& res)
	:res_(res)
{}

//Returns string table data by ID and index in language directory (instead of language)
const resource_string_list resource_string_table_reader::get_string_table_by_id(uint32_t id, uint32_t index) const
{
	return parse_string_list(id, res_.get_resource_data_by_id(pe_resource_viewer::resource_string, id, index).get_data());
}

//Returns string table data by ID and language
const resource_string_list resource_string_table_reader::get_string_table_by_id_lang(uint32_t language, uint32_t id) const
{
	return parse_string_list(id, res_.get_resource_data_by_id(language, pe_resource_viewer::resource_string, id).get_data());
}

//Helper function of parsing string list table
const resource_string_list resource_string_table_reader::parse_string_list(uint32_t id, const std::string& resource_data)
{
	resource_string_list ret;

	//16 is maximum count of strings in a string table
	static const unsigned long max_string_list_entries = 16;
	unsigned long passed_bytes = 0;
	for(unsigned long i = 0; i != max_string_list_entries; ++i)
	{
		//Check resource data length
		if(resource_data.length() < sizeof(uint16_t) + passed_bytes)
			throw pe_exception("Incorrect resource string table", pe_exception::resource_incorrect_string_table);

		//Get string length - the first WORD
		uint16_t string_length = *reinterpret_cast<const uint16_t*>(resource_data.data() + passed_bytes);
		passed_bytes += sizeof(uint16_t); //WORD containing string length

		//Check resource data length again
		if(resource_data.length() < string_length + passed_bytes)
			throw pe_exception("Incorrect resource string table", pe_exception::resource_incorrect_string_table);

		if(string_length)
		{
			//Create and save string (UNICODE)
#ifdef PE_BLISS_WINDOWS
			ret.insert(
				std::make_pair(static_cast<uint16_t>(((id - 1) << 4) + i), //ID of string is calculated such way
				std::wstring(reinterpret_cast<const wchar_t*>(resource_data.data() + passed_bytes), string_length)));
#else
			ret.insert(
				std::make_pair(static_cast<uint16_t>(((id - 1) << 4) + i), //ID of string is calculated such way
				pe_utils::from_ucs2(u16string(reinterpret_cast<const unicode16_t*>(resource_data.data() + passed_bytes), string_length))));
#endif
		}

		//Go to next string
		passed_bytes += string_length * 2;
	}

	return ret;
}

//Returns string from string table by ID and language
const std::wstring resource_string_table_reader::get_string_by_id_lang(uint32_t language, uint16_t id) const
{
	//List strings by string table id and language
	const resource_string_list strings(get_string_table_by_id_lang(language, (id >> 4) + 1));
	resource_string_list::const_iterator it = strings.find(id); //Find string by id
	if(it == strings.end())
		throw pe_exception("Resource string not found", pe_exception::resource_string_not_found);

	return (*it).second;
}

//Returns string from string table by ID and index in language directory (instead of language)
const std::wstring resource_string_table_reader::get_string_by_id(uint16_t id, uint32_t index) const
{
	//List strings by string table id and index
	const resource_string_list strings(get_string_table_by_id((id >> 4) + 1, index));
	resource_string_list::const_iterator it = strings.find(id); //Find string by id
	if(it == strings.end())
		throw pe_exception("Resource string not found", pe_exception::resource_string_not_found);

	return (*it).second;
}
}

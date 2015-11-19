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
#include <string>
#include <map>
#include "stdint_defs.h"

namespace pe_bliss
{
class pe_resource_viewer;

//ID; string
typedef std::map<uint16_t, std::wstring> resource_string_list;

class resource_string_table_reader
{
public:
	resource_string_table_reader(const pe_resource_viewer& res);

public:
	//Returns string table data by ID and language
	const resource_string_list get_string_table_by_id_lang(uint32_t language, uint32_t id) const;
	//Returns string table data by ID and index in language directory (instead of language)
	const resource_string_list get_string_table_by_id(uint32_t id, uint32_t index = 0) const;
	//Returns string from string table by ID and language
	const std::wstring get_string_by_id_lang(uint32_t language, uint16_t id) const;
	//Returns string from string table by ID and index in language directory (instead of language)
	const std::wstring get_string_by_id(uint16_t id, uint32_t index = 0) const;

private:
	const pe_resource_viewer& res_;

	//Helper function of parsing string list table
	//Id of resource is needed to calculate string IDs correctly
	//resource_data is raw string table resource data
	static const resource_string_list parse_string_list(uint32_t id, const std::string& resource_data);
};
}

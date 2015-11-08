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
#include "message_table.h"

namespace pe_bliss
{
class pe_resource_viewer;

//ID; message_table_item
typedef std::map<uint32_t, message_table_item> resource_message_list;

class resource_message_list_reader
{
public:
	resource_message_list_reader(const pe_resource_viewer& res);

	//Returns message table data by ID and language
	const resource_message_list get_message_table_by_id_lang(uint32_t language, uint32_t id) const;
	//Returns message table data by ID and index in language directory (instead of language)
	const resource_message_list get_message_table_by_id(uint32_t id, uint32_t index = 0) const;

	//Helper function of parsing message list table
	//resource_data - raw message table resource data
	static const resource_message_list parse_message_list(const std::string& resource_data);

private:
	const pe_resource_viewer& res_;
};
}

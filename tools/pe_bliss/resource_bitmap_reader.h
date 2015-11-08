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
#include "stdint_defs.h"

namespace pe_bliss
{
class pe_resource_viewer;

class resource_bitmap_reader
{
public:
	resource_bitmap_reader(const pe_resource_viewer& res);

	//Returns bitmap data by name and language (minimum checks of format correctness)
	const std::string get_bitmap_by_name(uint32_t language, const std::wstring& name) const;
	//Returns bitmap data by name and index in language directory (instead of language) (minimum checks of format correctness)
	const std::string get_bitmap_by_name(const std::wstring& name, uint32_t index = 0) const;
	//Returns bitmap data by ID and language (minimum checks of format correctness)
	const std::string get_bitmap_by_id_lang(uint32_t language, uint32_t id) const;
	//Returns bitmap data by ID and index in language directory (instead of language) (minimum checks of format correctness)
	const std::string get_bitmap_by_id(uint32_t id, uint32_t index = 0) const;

private:
	//Helper function of creating bitmap header
	static const std::string create_bitmap(const std::string& resource_data);

	const pe_resource_viewer& res_;
};
}

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
class pe_resource_manager;

class resource_bitmap_writer
{
public:
	resource_bitmap_writer(pe_resource_manager& res);

	//Adds bitmap from bitmap file data. If bitmap already exists, replaces it
	//timestamp will be used for directories that will be added
	void add_bitmap(const std::string& bitmap_file, uint32_t id, uint32_t language, uint32_t codepage = 0, uint32_t timestamp = 0);
	void add_bitmap(const std::string& bitmap_file, const std::wstring& name, uint32_t language, uint32_t codepage = 0, uint32_t timestamp = 0);

	//Removes bitmap by name/ID and language
	bool remove_bitmap(const std::wstring& name, uint32_t language);
	bool remove_bitmap(uint32_t id, uint32_t language);

private:
	pe_resource_manager& res_;
};
}

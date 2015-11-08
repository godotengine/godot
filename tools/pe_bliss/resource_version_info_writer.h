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
#include "version_info_types.h"
#include "file_version_info.h"

namespace pe_bliss
{
class pe_resource_manager;

class resource_version_info_writer
{
public:
	resource_version_info_writer(pe_resource_manager& res);
	
	//Sets/replaces full version information:
	//file_version_info: versions and file info
	//lang_string_values_map: map of version info strings with encodings
	//translation_values_map: map of translations
	void set_version_info(const file_version_info& file_info,
		const lang_string_values_map& string_values,
		const translation_values_map& translations,
		uint32_t language,
		uint32_t codepage = 0,
		uint32_t timestamp = 0);
	
	//Removes version info by language (ID = 1)
	bool remove_version_info(uint32_t language);

private:
	pe_resource_manager& res_;
};
}

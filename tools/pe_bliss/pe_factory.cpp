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
#include "pe_factory.h"
#include "pe_properties_generic.h"

namespace pe_bliss
{
pe_base pe_factory::create_pe(std::istream& file, bool read_debug_raw_data)
{
	return pe_base::get_pe_type(file) == pe_type_32
		? pe_base(file, pe_properties_32(), read_debug_raw_data)
		: pe_base(file, pe_properties_64(), read_debug_raw_data);
}

pe_base pe_factory::create_pe(const char* file_path, bool read_debug_raw_data)
{
	std::ifstream pe_file(file_path, std::ios::in | std::ios::binary);
	if(!pe_file)
	{
		throw pe_exception("Error in open file.", pe_exception::stream_is_bad);
	}
	return pe_factory::create_pe(pe_file,read_debug_raw_data);
}
}

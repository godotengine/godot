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
#include <string.h>
#include "utils.h"
#include "pe_exception.h"

#ifndef PE_BLISS_WINDOWS
#include <iconv.h>
#endif

namespace pe_bliss
{
const double pe_utils::log_2 = 1.44269504088896340736; //instead of using M_LOG2E

//Returns stream size
std::streamoff pe_utils::get_file_size(std::istream& file)
{
	//Get old istream offset
	std::streamoff old_offset = file.tellg();
	file.seekg(0, std::ios::end);
	std::streamoff filesize = file.tellg();
	//Set old istream offset
	file.seekg(old_offset);
	return filesize;
}

#ifndef PE_BLISS_WINDOWS
const u16string pe_utils::to_ucs2(const std::wstring& str)
{
	u16string ret;
	if(str.empty())
		return ret;

	ret.resize(str.length());

	iconv_t conv = iconv_open("UCS-2", "WCHAR_T");
	if(conv == reinterpret_cast<iconv_t>(-1))
		throw pe_exception("Error opening iconv", pe_exception::encoding_convertion_error);

	size_t inbytesleft = str.length() * sizeof(wchar_t);
	size_t outbytesleft = ret.length() * sizeof(unicode16_t);
	const wchar_t* in_pos = str.c_str();
	unicode16_t* out_pos = &ret[0];

	size_t result = iconv(conv, const_cast<char**>(reinterpret_cast<const char**>(&in_pos)), &inbytesleft, reinterpret_cast<char**>(&out_pos), &outbytesleft);
	iconv_close(conv);
	
	if(result == static_cast<size_t>(-1))
		throw pe_exception("Iconv error", pe_exception::encoding_convertion_error);

	return ret;
}

const std::wstring pe_utils::from_ucs2(const u16string& str)
{
	std::wstring ret;
	if(str.empty())
		return ret;

	ret.resize(str.length());

	iconv_t conv = iconv_open("WCHAR_T", "UCS-2");
	if(conv == reinterpret_cast<iconv_t>(-1))
		throw pe_exception("Error opening iconv", pe_exception::encoding_convertion_error);

	size_t inbytesleft = str.length() * sizeof(unicode16_t);
	size_t outbytesleft = ret.length() * sizeof(wchar_t);
	const unicode16_t* in_pos = str.c_str();
	wchar_t* out_pos = &ret[0];

	size_t result = iconv(conv, const_cast<char**>(reinterpret_cast<const char**>(&in_pos)), &inbytesleft, reinterpret_cast<char**>(&out_pos), &outbytesleft);
	iconv_close(conv);

	if(result == static_cast<size_t>(-1))
		throw pe_exception("Iconv error", pe_exception::encoding_convertion_error);

	return ret;
}
#endif

bool operator==(const pe_win::guid& guid1, const pe_win::guid& guid2)
{
	return guid1.Data1 == guid2.Data1
		&& guid1.Data2 == guid2.Data2
		&& guid1.Data3 == guid2.Data3
		&& !memcmp(guid1.Data4, guid2.Data4, sizeof(guid1.Data4));
}
}

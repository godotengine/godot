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

	int len = str.length();
	
	ret.resize(len);
	
	for(int i=0;i<len;i++) {
		ret[i]=str[i]&0xFFFF;
	}

	return ret;
}

const std::wstring pe_utils::from_ucs2(const u16string& str)
{
	std::wstring ret;
	if(str.empty())
		return ret;

	int len = str.length();
	ret.resize(str.length());
	
	for(int i=0;i<len;i++) {
		ret[i]=str[i];
	}

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

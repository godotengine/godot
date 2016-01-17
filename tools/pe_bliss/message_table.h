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
//Structure representing message table string
class message_table_item
{
public:
	//Default constructor
	message_table_item();
	//Constructors from ANSI and UNICODE strings
	explicit message_table_item(const std::string& str);
	explicit message_table_item(const std::wstring& str);

	//Returns true if string is UNICODE
	bool is_unicode() const;
	//Returns ANSI string
	const std::string& get_ansi_string() const;
	//Returns UNICODE string
	const std::wstring& get_unicode_string() const;

public:
	//Sets ANSI or UNICODE string
	void set_string(const std::string& str);
	void set_string(const std::wstring& str);

private:
	bool unicode_;
	std::string ansi_str_;
	std::wstring unicode_str_;
};
}

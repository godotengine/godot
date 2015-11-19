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
#include "message_table.h"
#include "utils.h"

namespace pe_bliss
{
//Default constructor
message_table_item::message_table_item()
	:unicode_(false)
{}

//Constructor from ANSI string
message_table_item::message_table_item(const std::string& str)
	:unicode_(false), ansi_str_(str)
{
	pe_utils::strip_nullbytes(ansi_str_);
}

//Constructor from UNICODE string
message_table_item::message_table_item(const std::wstring& str)
	:unicode_(true), unicode_str_(str)
{
	pe_utils::strip_nullbytes(unicode_str_);
}

//Returns true if contained string is unicode
bool message_table_item::is_unicode() const
{
	return unicode_;
}

//Returns ANSI string
const std::string& message_table_item::get_ansi_string() const
{
	return ansi_str_;
}

//Returns UNICODE string
const std::wstring& message_table_item::get_unicode_string() const
{
	return unicode_str_;
}

//Sets ANSI string (clears UNICODE one)
void message_table_item::set_string(const std::string& str)
{
	ansi_str_ = str;
	pe_utils::strip_nullbytes(ansi_str_);
	unicode_str_.clear();
	unicode_ = false;
}

//Sets UNICODE string (clears ANSI one)
void message_table_item::set_string(const std::wstring& str)
{
	unicode_str_ = str;
	pe_utils::strip_nullbytes(unicode_str_);
	ansi_str_.clear();
	unicode_ = true;
}
}

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

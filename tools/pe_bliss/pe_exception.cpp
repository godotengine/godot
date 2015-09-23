#include "pe_exception.h"

namespace pe_bliss
{
//PE exception class constructors
pe_exception::pe_exception(const char* text, exception_id id)
	:std::runtime_error(text), id_(id)
{}

pe_exception::pe_exception(const std::string& text, exception_id id)
	:std::runtime_error(text), id_(id)
{}

//Returns exception ID
pe_exception::exception_id pe_exception::get_id() const
{
	return id_;
}
}

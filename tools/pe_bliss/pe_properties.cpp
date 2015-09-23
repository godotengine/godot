#include "pe_properties.h"

namespace pe_bliss
{
//Destructor
pe_properties::~pe_properties()
{}

//Clears PE characteristics flag
void pe_properties::clear_characteristics_flags(uint16_t flags)
{
	set_characteristics(get_characteristics() & ~flags);
}

//Sets PE characteristics flag
void pe_properties::set_characteristics_flags(uint16_t flags)
{
	set_characteristics(get_characteristics() | flags);
}
}

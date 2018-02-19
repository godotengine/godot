/*
 * typedefs.cpp
 * ------------
 * Purpose: Basic data type definitions and assorted compiler-related helpers.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "typedefs.h"

#include "Endianness.h"


OPENMPT_NAMESPACE_BEGIN

#if MPT_PLATFORM_ENDIAN_KNOWN

MPT_MSVC_WORKAROUND_LNK4221(typedefs)

#else

int24::int24(int other)
{
	MPT_MAYBE_CONSTANT_IF(mpt::endian_is_big()) {
		bytes[0] = (static_cast<unsigned int>(other)>>16)&0xff;
		bytes[1] = (static_cast<unsigned int>(other)>> 8)&0xff;
		bytes[2] = (static_cast<unsigned int>(other)>> 0)&0xff;
	} else {
		bytes[0] = (static_cast<unsigned int>(other)>> 0)&0xff;
		bytes[1] = (static_cast<unsigned int>(other)>> 8)&0xff;
		bytes[2] = (static_cast<unsigned int>(other)>>16)&0xff;
	}
}

int24::operator int() const
{
	MPT_MAYBE_CONSTANT_IF(mpt::endian_is_big()) {
		return (static_cast<int8>(bytes[0]) * 65536) + (bytes[1] * 256) + bytes[2];
	} else {
		return (static_cast<int8>(bytes[2]) * 65536) + (bytes[1] * 256) + bytes[0];
	}
}

#endif

OPENMPT_NAMESPACE_END

/*
 * misc_util.cpp
 * -------------
 * Purpose: Various useful utility functions.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "misc_util.h"


OPENMPT_NAMESPACE_BEGIN



#ifdef MODPLUG_TRACKER

namespace Util
{


#if MPT_OS_WINDOWS


#endif // MPT_OS_WINDOWS


} // namespace Util

#endif // MODPLUG_TRACKER


namespace Util
{


static const MPT_UCHAR_TYPE EncodeNibble[16] = {
	MPT_UCHAR('0'), MPT_UCHAR('1'), MPT_UCHAR('2'), MPT_UCHAR('3'),
	MPT_UCHAR('4'), MPT_UCHAR('5'), MPT_UCHAR('6'), MPT_UCHAR('7'),
	MPT_UCHAR('8'), MPT_UCHAR('9'), MPT_UCHAR('A'), MPT_UCHAR('B'),
	MPT_UCHAR('C'), MPT_UCHAR('D'), MPT_UCHAR('E'), MPT_UCHAR('F') };

static inline bool DecodeByte(uint8 &byte, MPT_UCHAR_TYPE c1, MPT_UCHAR_TYPE c2)
{
	byte = 0;
	if(MPT_UCHAR('0') <= c1 && c1 <= MPT_UCHAR('9'))
	{
		byte += static_cast<uint8>((c1 - MPT_UCHAR('0')) << 4);
	} else if(MPT_UCHAR('A') <= c1 && c1 <= MPT_UCHAR('F'))
	{
		byte += static_cast<uint8>((c1 - MPT_UCHAR('A') + 10) << 4);
	} else if(MPT_UCHAR('a') <= c1 && c1 <= MPT_UCHAR('f'))
	{
		byte += static_cast<uint8>((c1 - MPT_UCHAR('a') + 10) << 4);
	} else
	{
		return false;
	}
	if(MPT_UCHAR('0') <= c2 && c2 <= MPT_UCHAR('9'))
	{
		byte += static_cast<uint8>(c2 - MPT_UCHAR('0'));
	} else if(MPT_UCHAR('A') <= c2 && c2 <= MPT_UCHAR('F'))
	{
		byte += static_cast<uint8>(c2 - MPT_UCHAR('A') + 10);
	} else if(MPT_UCHAR('a') <= c2 && c2 <= MPT_UCHAR('f'))
	{
		byte += static_cast<uint8>(c2 - MPT_UCHAR('a') + 10);
	} else
	{
		return false;
	}
	return true;
}

mpt::ustring BinToHex(mpt::const_byte_span src)
{
	mpt::ustring result;
	result.reserve(src.size() * 2);
	for(uint8 byte : src)
	{
		result.push_back(EncodeNibble[(byte & 0xf0) >> 4]);
		result.push_back(EncodeNibble[byte & 0x0f]);
	}
	return result;
}

std::vector<mpt::byte> HexToBin(const mpt::ustring &src)
{
	std::vector<mpt::byte> result;
	result.reserve(src.size() / 2);
	for(std::size_t i = 0; (i + 1) < src.size(); i += 2)
	{
		uint8 byte = 0;
		if(!DecodeByte(byte, src[i], src[i + 1]))
		{
			return result;
		}
		result.push_back(byte);
	}
	return result;
}


} // namespace Util


#if defined(MODPLUG_TRACKER) || (defined(LIBOPENMPT_BUILD) && defined(LIBOPENMPT_BUILD_TEST))

namespace mpt
{

std::string getenv(const std::string &env_var, const std::string &def)
{
#if MPT_OS_WINDOWS && MPT_OS_WINDOWS_WINRT
	MPT_UNREFERENCED_PARAMETER(env_var);
	return def;
#else
	const char *val = std::getenv(env_var.c_str());
	if(!val)
	{
		return def;
	}
	return val;
#endif
}

} // namespace mpt

#endif // MODPLUG_TRACKER || (LIBOPENMPT_BUILD && LIBOPENMPT_BUILD_TEST)


OPENMPT_NAMESPACE_END

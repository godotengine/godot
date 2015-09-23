#pragma once
#ifdef _MSC_VER
#if _MSC_VER < 1600
namespace pe_bliss
{
	//stdint.h definitions for MSVC 2008 and earlier, as
	//it doesn't have them
	typedef signed char int8_t;
	typedef short int16_t;
	typedef int int32_t;

	typedef unsigned char uint8_t;
	typedef unsigned short uint16_t;
	typedef unsigned int uint32_t;

	typedef long long int64_t;
	typedef unsigned long long uint64_t;
}
#else
#include <stdint.h>
#endif
#else
#include <stdint.h>
#endif

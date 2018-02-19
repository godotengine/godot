/*
 * Loaders.h
 * ---------
 * Purpose: Common functions for module loaders
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#include "../common/misc_util.h"
#include "../common/FileReader.h"
#include "Sndfile.h"
#include "SampleIO.h"

OPENMPT_NAMESPACE_BEGIN

// Macros to create magic bytes in little-endian format
#define MAGIC4LE(a, b, c, d)	static_cast<uint32>((static_cast<uint8>(d) << 24) | (static_cast<uint8>(c) << 16) | (static_cast<uint8>(b) << 8) | static_cast<uint8>(a))
#define MAGIC2LE(a, b)		static_cast<uint16>((static_cast<uint8>(b) << 8) | static_cast<uint8>(a))
// Macros to create magic bytes in big-endian format
#define MAGIC4BE(a, b, c, d)	static_cast<uint32>((static_cast<uint8>(a) << 24) | (static_cast<uint8>(b) << 16) | (static_cast<uint8>(c) << 8) | static_cast<uint8>(d))
#define MAGIC2BE(a, b)		static_cast<uint16>((static_cast<uint8>(a) << 8) | static_cast<uint8>(b))


// Read 'howMany' order items from an array.
// 'stopIndex' is treated as '---', 'ignoreIndex' is treated as '+++'. If the format doesn't support such indices, just pass ORDERINDEX_INVALID.
template<typename T, size_t arraySize>
bool ReadOrderFromArray(ModSequence &order, const T(&orders)[arraySize], size_t howMany = arraySize, uint16 stopIndex = ORDERINDEX_INVALID, uint16 ignoreIndex = ORDERINDEX_INVALID)
{
	STATIC_ASSERT(mpt::is_binary_safe<T>::value);
	LimitMax(howMany, arraySize);
	LimitMax(howMany, MAX_ORDERS);
	ORDERINDEX readEntries = static_cast<ORDERINDEX>(howMany);

	order.resize(readEntries);
	for(int i = 0; i < readEntries; i++)
	{
		PATTERNINDEX pat = static_cast<PATTERNINDEX>(orders[i]);
		if(pat == stopIndex) pat = order.GetInvalidPatIndex();
		else if(pat == ignoreIndex) pat = order.GetIgnoreIndex();
		order.at(i) = pat;
	}
	return true;
}


// Read 'howMany' order items as integers with defined endianness from a file.
// 'stopIndex' is treated as '---', 'ignoreIndex' is treated as '+++'. If the format doesn't support such indices, just pass ORDERINDEX_INVALID.
template<typename T>
bool ReadOrderFromFile(ModSequence &order, FileReader &file, size_t howMany, uint16 stopIndex = ORDERINDEX_INVALID, uint16 ignoreIndex = ORDERINDEX_INVALID)
{
	STATIC_ASSERT(mpt::is_binary_safe<T>::value);
	if(!file.CanRead(howMany * sizeof(T)))
		return false;
	LimitMax(howMany, MAX_ORDERS);
	ORDERINDEX readEntries = static_cast<ORDERINDEX>(howMany);

	order.resize(readEntries);
	T patF;
	for(auto &pat : order)
	{
		file.ReadStruct(patF);
		pat = static_cast<PATTERNINDEX>(patF);
		if(pat == stopIndex) pat = order.GetInvalidPatIndex();
		else if(pat == ignoreIndex) pat = order.GetIgnoreIndex();
	}
	return true;
}

OPENMPT_NAMESPACE_END

/*
bParse
Copyright (c) 2006-2009 Charlie C & Erwin Coumans  http://gamekit.googlecode.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "b3Chunk.h"
#include "b3Defines.h"
#include "b3File.h"

#if !defined(__CELLOS_LV2__) && !defined(__MWERKS__)
#include <memory.h>
#endif
#include <string.h>

using namespace bParse;

// ----------------------------------------------------- //
short ChunkUtils::swapShort(short sht)
{
	B3_SWITCH_SHORT(sht);
	return sht;
}

// ----------------------------------------------------- //
int ChunkUtils::swapInt(int inte)
{
	B3_SWITCH_INT(inte);
	return inte;
}

// ----------------------------------------------------- //
b3Long64 ChunkUtils::swapLong64(b3Long64 lng)
{
	B3_SWITCH_LONGINT(lng);
	return lng;
}

// ----------------------------------------------------- //
int ChunkUtils::getOffset(int flags)
{
	// if the file is saved in a
	// different format, get the
	// file's chunk size
	int res = CHUNK_HEADER_LEN;

	if (VOID_IS_8)
	{
		if (flags & FD_BITS_VARIES)
			res = sizeof(bChunkPtr4);
	}
	else
	{
		if (flags & FD_BITS_VARIES)
			res = sizeof(bChunkPtr8);
	}
	return res;
}

//eof

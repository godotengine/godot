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

#ifndef __BCHUNK_H__
#define __BCHUNK_H__

#if defined (_WIN32) && ! defined (__MINGW32__)
	#define b3Long64 __int64
#elif defined (__MINGW32__)	
	#include <stdint.h>
	#define b3Long64 int64_t
#else
	#define b3Long64 long long
#endif


namespace bParse {


	// ----------------------------------------------------- //
	class bChunkPtr4
	{
	public:
		bChunkPtr4(){}
		int code;
		int len;
		union
		{
			int m_uniqueInt;
		};
		int dna_nr;
		int nr;
	};

	// ----------------------------------------------------- //
	class bChunkPtr8
	{
	public:
		bChunkPtr8(){}
		int code,  len;
		union
		{
			b3Long64 oldPrev;
			int	m_uniqueInts[2];
		};
		int dna_nr, nr;
	};

	// ----------------------------------------------------- //
	class bChunkInd
	{
	public:
		bChunkInd(){}
		int code, len;
		void *oldPtr;
		int dna_nr, nr;
	};


	// ----------------------------------------------------- //
	class ChunkUtils
	{
	public:
		
		// file chunk offset
		static int getOffset(int flags);

		// endian utils
		static short swapShort(short sht);
		static int swapInt(int inte);
		static b3Long64 swapLong64(b3Long64 lng);

	};


	const int CHUNK_HEADER_LEN = ((sizeof(bChunkInd)));
	const bool VOID_IS_8 = ((sizeof(void*)==8));
}

#endif//__BCHUNK_H__

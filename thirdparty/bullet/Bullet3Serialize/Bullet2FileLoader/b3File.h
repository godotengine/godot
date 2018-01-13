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

#ifndef __BFILE_H__
#define __BFILE_H__

#include "b3Common.h"
#include "b3Chunk.h"
#include <stdio.h>

namespace bParse {

	// ----------------------------------------------------- //
	enum bFileFlags
	{
		FD_INVALID   =0,
		FD_OK        =1,
		FD_VOID_IS_8 =2,
		FD_ENDIAN_SWAP      =4,
		FD_FILE_64   =8,
		FD_BITS_VARIES    =16,
		FD_VERSION_VARIES = 32,
		FD_DOUBLE_PRECISION =64,
		FD_BROKEN_DNA = 128
	};

	enum bFileVerboseMode
	{
		FD_VERBOSE_EXPORT_XML = 1,
		FD_VERBOSE_DUMP_DNA_TYPE_DEFINITIONS = 2,
		FD_VERBOSE_DUMP_CHUNKS = 4,
		FD_VERBOSE_DUMP_FILE_INFO=8,
	};
	// ----------------------------------------------------- //
	class bFile
	{
	protected:
		
		char				m_headerString[7];

		bool				mOwnsBuffer;
		char*				mFileBuffer;
		int					mFileLen;
		int					mVersion;


		bPtrMap				mLibPointers;

		int					mDataStart;
		bDNA*				mFileDNA;
		bDNA*				mMemoryDNA;

		b3AlignedObjectArray<char*>	m_pointerFixupArray;
		b3AlignedObjectArray<char*>	m_pointerPtrFixupArray;
		
		b3AlignedObjectArray<bChunkInd>	m_chunks;
        b3HashMap<b3HashPtr, bChunkInd> m_chunkPtrPtrMap;

        // 
	
		bPtrMap				mDataPointers;

		
		int					mFlags;

		// ////////////////////////////////////////////////////////////////////////////

			// buffer offset util
		int getNextBlock(bChunkInd *dataChunk,  const char *dataPtr, const int flags);
		void safeSwapPtr(char *dst, const char *src);

		virtual	void parseHeader();
		
		virtual	void parseData() = 0;

		void resolvePointersMismatch();
		void resolvePointersChunk(const bChunkInd& dataChunk, int verboseMode);

		int resolvePointersStructRecursive(char *strcPtr, int old_dna, int verboseMode, int recursion);
		//void swapPtr(char *dst, char *src);

		void parseStruct(char *strcPtr, char *dtPtr, int old_dna, int new_dna, bool fixupPointers);
		void getMatchingFileDNA(short* old, const char* lookupName, const char* lookupType, char *strcData, char *data, bool fixupPointers);
		char* getFileElement(short *firstStruct, char *lookupName, char *lookupType, char *data, short **foundPos);


		void swap(char *head, class bChunkInd& ch, bool ignoreEndianFlag);
		void swapData(char *data, short type, int arraySize, bool ignoreEndianFlag);
		void swapStruct(int dna_nr, char *data, bool ignoreEndianFlag);
		void swapLen(char *dataPtr);
		void swapDNA(char* ptr);


		char* readStruct(char *head, class bChunkInd& chunk);
		char *getAsString(int code);

		void	parseInternal(int verboseMode, char* memDna,int memDnaLength);

	public:
		bFile(const char *filename, const char headerString[7]);
		
		//todo: make memoryBuffer const char
		//bFile( const char *memoryBuffer, int len);
		bFile( char *memoryBuffer, int len, const char headerString[7]);
		virtual ~bFile();

		bDNA*				getFileDNA()
		{
			return mFileDNA;
		}

		virtual	void	addDataBlock(char* dataBlock) = 0;

		int	getFlags() const
		{
			return mFlags;
		}

		bPtrMap&		getLibPointers()
		{
			return mLibPointers;
		}
		
		void* findLibPointer(void *ptr);

		bool ok();

		virtual	void parse(int verboseMode) = 0;

		virtual	int	write(const char* fileName, bool fixupPointers=false) = 0;

		virtual	void	writeChunks(FILE* fp, bool fixupPointers );

		virtual	void	writeDNA(FILE* fp) = 0;

		void	updateOldPointers();
		void	resolvePointers(int verboseMode);

		void	dumpChunks(bDNA* dna);
		
		int		getVersion() const
		{
			return mVersion;
		}
		//pre-swap the endianness, so that data loaded on a target with different endianness doesn't need to be swapped
		void preSwap();
		void writeFile(const char* fileName);

	};
}


#endif//__BFILE_H__

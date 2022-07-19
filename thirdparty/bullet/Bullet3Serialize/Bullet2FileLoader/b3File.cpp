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
#include "b3File.h"
#include "b3Common.h"
#include "b3Chunk.h"
#include "b3DNA.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "b3Defines.h"
#include "Bullet3Serialize/Bullet2FileLoader/b3Serializer.h"
#include "Bullet3Common/b3AlignedAllocator.h"
#include "Bullet3Common/b3MinMax.h"

#define B3_SIZEOFBLENDERHEADER 12
#define MAX_ARRAY_LENGTH 512
using namespace bParse;
#define MAX_STRLEN 1024

const char *getCleanName(const char *memName, char *buffer)
{
	int slen = strlen(memName);
	assert(slen < MAX_STRLEN);
	slen = b3Min(slen, MAX_STRLEN);
	for (int i = 0; i < slen; i++)
	{
		if (memName[i] == ']' || memName[i] == '[')
		{
			buffer[i] = 0;  //'_';
		}
		else
		{
			buffer[i] = memName[i];
		}
	}
	buffer[slen] = 0;
	return buffer;
}

// ----------------------------------------------------- //
bFile::bFile(const char *filename, const char headerString[7])
	: mOwnsBuffer(true),
	  mFileBuffer(0),
	  mFileLen(0),
	  mVersion(0),
	  mDataStart(0),
	  mFileDNA(0),
	  mMemoryDNA(0),
	  mFlags(FD_INVALID)
{
	for (int i = 0; i < 7; i++)
	{
		m_headerString[i] = headerString[i];
	}

	FILE *fp = fopen(filename, "rb");
	if (fp)
	{
		fseek(fp, 0L, SEEK_END);
		mFileLen = ftell(fp);
		fseek(fp, 0L, SEEK_SET);

		mFileBuffer = (char *)malloc(mFileLen + 1);
		int bytesRead;
		bytesRead = fread(mFileBuffer, mFileLen, 1, fp);

		fclose(fp);

		//
		parseHeader();
	}
}

// ----------------------------------------------------- //
bFile::bFile(char *memoryBuffer, int len, const char headerString[7])
	: mOwnsBuffer(false),
	  mFileBuffer(0),
	  mFileLen(0),
	  mVersion(0),
	  mDataStart(0),
	  mFileDNA(0),
	  mMemoryDNA(0),
	  mFlags(FD_INVALID)
{
	for (int i = 0; i < 7; i++)
	{
		m_headerString[i] = headerString[i];
	}
	mFileBuffer = memoryBuffer;
	mFileLen = len;

	parseHeader();
}

// ----------------------------------------------------- //
bFile::~bFile()
{
	if (mOwnsBuffer && mFileBuffer)
	{
		free(mFileBuffer);
		mFileBuffer = 0;
	}

	delete mMemoryDNA;
	delete mFileDNA;
}

// ----------------------------------------------------- //
void bFile::parseHeader()
{
	if (!mFileLen || !mFileBuffer)
		return;

	char *blenderBuf = mFileBuffer;
	char header[B3_SIZEOFBLENDERHEADER + 1];
	memcpy(header, blenderBuf, B3_SIZEOFBLENDERHEADER);
	header[B3_SIZEOFBLENDERHEADER] = '\0';

	if (strncmp(header, m_headerString, 6) != 0)
	{
		memcpy(header, m_headerString, B3_SIZEOFBLENDERHEADER);
		return;
	}

	if (header[6] == 'd')
	{
		mFlags |= FD_DOUBLE_PRECISION;
	}

	char *ver = header + 9;
	mVersion = atoi(ver);
	if (mVersion <= 241)
	{
		//printf("Warning, %d not fully tested : <= 242\n", mVersion);
	}

	int littleEndian = 1;
	littleEndian = ((char *)&littleEndian)[0];

	// swap ptr sizes...
	if (header[7] == '-')
	{
		mFlags |= FD_FILE_64;
		if (!VOID_IS_8)
			mFlags |= FD_BITS_VARIES;
	}
	else if (VOID_IS_8)
		mFlags |= FD_BITS_VARIES;

	// swap endian...
	if (header[8] == 'V')
	{
		if (littleEndian == 1)
			mFlags |= FD_ENDIAN_SWAP;
	}
	else if (littleEndian == 0)
		mFlags |= FD_ENDIAN_SWAP;

	mFlags |= FD_OK;
}

// ----------------------------------------------------- //
bool bFile::ok()
{
	return (mFlags & FD_OK) != 0;
}

// ----------------------------------------------------- //
void bFile::parseInternal(int verboseMode, char *memDna, int memDnaLength)
{
	if ((mFlags & FD_OK) == 0)
		return;

	char *blenderData = mFileBuffer;
	bChunkInd dna;
	dna.oldPtr = 0;

	char *tempBuffer = blenderData;
	for (int i = 0; i < mFileLen; i++)
	{
		// looking for the data's starting position
		// and the start of SDNA decls

		if (!mDataStart && strncmp(tempBuffer, "REND", 4) == 0)
			mDataStart = i;

		if (strncmp(tempBuffer, "DNA1", 4) == 0)
		{
			// read the DNA1 block and extract SDNA
			if (getNextBlock(&dna, tempBuffer, mFlags) > 0)
			{
				if (strncmp((tempBuffer + ChunkUtils::getOffset(mFlags)), "SDNANAME", 8) == 0)
					dna.oldPtr = (tempBuffer + ChunkUtils::getOffset(mFlags));
				else
					dna.oldPtr = 0;
			}
			else
				dna.oldPtr = 0;
		}
		// Some Bullet files are missing the DNA1 block
		// In Blender it's DNA1 + ChunkUtils::getOffset() + SDNA + NAME
		// In Bullet tests its SDNA + NAME
		else if (strncmp(tempBuffer, "SDNANAME", 8) == 0)
		{
			dna.oldPtr = blenderData + i;
			dna.len = mFileLen - i;

			// Also no REND block, so exit now.
			if (mVersion == 276) break;
		}

		if (mDataStart && dna.oldPtr) break;
		tempBuffer++;
	}
	if (!dna.oldPtr || !dna.len)
	{
		//printf("Failed to find DNA1+SDNA pair\n");
		mFlags &= ~FD_OK;
		return;
	}

	mFileDNA = new bDNA();

	///mFileDNA->init will convert part of DNA file endianness to current CPU endianness if necessary
	mFileDNA->init((char *)dna.oldPtr, dna.len, (mFlags & FD_ENDIAN_SWAP) != 0);

	if (mVersion == 276)
	{
		int i;
		for (i = 0; i < mFileDNA->getNumNames(); i++)
		{
			if (strcmp(mFileDNA->getName(i), "int") == 0)
			{
				mFlags |= FD_BROKEN_DNA;
			}
		}
		if ((mFlags & FD_BROKEN_DNA) != 0)
		{
			//printf("warning: fixing some broken DNA version\n");
		}
	}

	if (verboseMode & FD_VERBOSE_DUMP_DNA_TYPE_DEFINITIONS)
		mFileDNA->dumpTypeDefinitions();

	mMemoryDNA = new bDNA();
	int littleEndian = 1;
	littleEndian = ((char *)&littleEndian)[0];

	mMemoryDNA->init(memDna, memDnaLength, littleEndian == 0);

	///@todo we need a better version check, add version/sub version info from FileGlobal into memory DNA/header files
	if (mMemoryDNA->getNumNames() != mFileDNA->getNumNames())
	{
		mFlags |= FD_VERSION_VARIES;
		//printf ("Warning, file DNA is different than built in, performance is reduced. Best to re-export file with a matching version/platform");
	}

	// as long as it kept up to date it will be ok!!
	if (mMemoryDNA->lessThan(mFileDNA))
	{
		//printf ("Warning, file DNA is newer than built in.");
	}

	mFileDNA->initCmpFlags(mMemoryDNA);

	parseData();

	resolvePointers(verboseMode);

	updateOldPointers();
}

// ----------------------------------------------------- //
void bFile::swap(char *head, bChunkInd &dataChunk, bool ignoreEndianFlag)
{
	char *data = head;
	short *strc = mFileDNA->getStruct(dataChunk.dna_nr);

	const char s[] = "SoftBodyMaterialData";
	int szs = sizeof(s);
	if (strncmp((char *)&dataChunk.code, "ARAY", 4) == 0)
	{
		short *oldStruct = mFileDNA->getStruct(dataChunk.dna_nr);
		char *oldType = mFileDNA->getType(oldStruct[0]);
		if (strncmp(oldType, s, szs) == 0)
		{
			return;
		}
	}

	int len = mFileDNA->getLength(strc[0]);

	for (int i = 0; i < dataChunk.nr; i++)
	{
		swapStruct(dataChunk.dna_nr, data, ignoreEndianFlag);
		data += len;
	}
}

void bFile::swapLen(char *dataPtr)
{
	const bool VOID_IS_8 = ((sizeof(void *) == 8));
	if (VOID_IS_8)
	{
		if (mFlags & FD_BITS_VARIES)
		{
			bChunkPtr4 *c = (bChunkPtr4 *)dataPtr;
			if ((c->code & 0xFFFF) == 0)
				c->code >>= 16;
			B3_SWITCH_INT(c->len);
			B3_SWITCH_INT(c->dna_nr);
			B3_SWITCH_INT(c->nr);
		}
		else
		{
			bChunkPtr8 *c = (bChunkPtr8 *)dataPtr;
			if ((c->code & 0xFFFF) == 0)
				c->code >>= 16;
			B3_SWITCH_INT(c->len);
			B3_SWITCH_INT(c->dna_nr);
			B3_SWITCH_INT(c->nr);
		}
	}
	else
	{
		if (mFlags & FD_BITS_VARIES)
		{
			bChunkPtr8 *c = (bChunkPtr8 *)dataPtr;
			if ((c->code & 0xFFFF) == 0)
				c->code >>= 16;
			B3_SWITCH_INT(c->len);
			B3_SWITCH_INT(c->dna_nr);
			B3_SWITCH_INT(c->nr);
		}
		else
		{
			bChunkPtr4 *c = (bChunkPtr4 *)dataPtr;
			if ((c->code & 0xFFFF) == 0)
				c->code >>= 16;
			B3_SWITCH_INT(c->len);

			B3_SWITCH_INT(c->dna_nr);
			B3_SWITCH_INT(c->nr);
		}
	}
}

void bFile::swapDNA(char *ptr)
{
	bool swap = ((mFlags & FD_ENDIAN_SWAP) != 0);

	char *data = &ptr[20];
	//	void bDNA::init(char *data, int len, bool swap)
	int *intPtr = 0;
	short *shtPtr = 0;
	char *cp = 0;
	int dataLen = 0;
	//long nr=0;
	intPtr = (int *)data;

	/*
		SDNA (4 bytes) (magic number)
		NAME (4 bytes)
		<nr> (4 bytes) amount of names (int)
		<string>
		<string>
	*/

	if (strncmp(data, "SDNA", 4) == 0)
	{
		// skip ++ NAME
		intPtr++;
		intPtr++;
	}

	// Parse names
	if (swap)
		dataLen = ChunkUtils::swapInt(*intPtr);
	else
		dataLen = *intPtr;

	*intPtr = ChunkUtils::swapInt(*intPtr);
	intPtr++;

	cp = (char *)intPtr;
	int i;
	for (i = 0; i < dataLen; i++)
	{
		while (*cp) cp++;
		cp++;
	}

	cp = b3AlignPointer(cp, 4);

	/*
		TYPE (4 bytes)
		<nr> amount of types (int)
		<string>
		<string>
	*/

	intPtr = (int *)cp;
	assert(strncmp(cp, "TYPE", 4) == 0);
	intPtr++;

	if (swap)
		dataLen = ChunkUtils::swapInt(*intPtr);
	else
		dataLen = *intPtr;

	*intPtr = ChunkUtils::swapInt(*intPtr);

	intPtr++;

	cp = (char *)intPtr;
	for (i = 0; i < dataLen; i++)
	{
		while (*cp) cp++;
		cp++;
	}

	cp = b3AlignPointer(cp, 4);

	/*
		TLEN (4 bytes)
		<len> (short) the lengths of types
		<len>
	*/

	// Parse type lens
	intPtr = (int *)cp;
	assert(strncmp(cp, "TLEN", 4) == 0);
	intPtr++;

	shtPtr = (short *)intPtr;
	for (i = 0; i < dataLen; i++, shtPtr++)
	{
		//??????if (swap)
		shtPtr[0] = ChunkUtils::swapShort(shtPtr[0]);
	}

	if (dataLen & 1)
		shtPtr++;

	/*
		STRC (4 bytes)
		<nr> amount of structs (int)
		<typenr>
		<nr_of_elems>
		<typenr>
		<namenr>
		<typenr>
		<namenr>
	*/

	intPtr = (int *)shtPtr;
	cp = (char *)intPtr;
	assert(strncmp(cp, "STRC", 4) == 0);
	intPtr++;

	if (swap)
		dataLen = ChunkUtils::swapInt(*intPtr);
	else
		dataLen = *intPtr;

	*intPtr = ChunkUtils::swapInt(*intPtr);

	intPtr++;

	shtPtr = (short *)intPtr;
	for (i = 0; i < dataLen; i++)
	{
		//if (swap)
		{
			int len = shtPtr[1];

			shtPtr[0] = ChunkUtils::swapShort(shtPtr[0]);
			shtPtr[1] = ChunkUtils::swapShort(shtPtr[1]);

			shtPtr += 2;

			for (int a = 0; a < len; a++, shtPtr += 2)
			{
				shtPtr[0] = ChunkUtils::swapShort(shtPtr[0]);
				shtPtr[1] = ChunkUtils::swapShort(shtPtr[1]);
			}
		}
		//		else
		//			shtPtr+= (2*shtPtr[1])+2;
	}
}

void bFile::writeFile(const char *fileName)
{
	FILE *f = fopen(fileName, "wb");
	fwrite(mFileBuffer, 1, mFileLen, f);
	fclose(f);
}

void bFile::preSwap()
{
	//const bool brokenDNA = (mFlags&FD_BROKEN_DNA)!=0;
	//FD_ENDIAN_SWAP
	//byte 8 determines the endianness of the file, little (v) versus big (V)
	int littleEndian = 1;
	littleEndian = ((char *)&littleEndian)[0];

	if (mFileBuffer[8] == 'V')
	{
		mFileBuffer[8] = 'v';
	}
	else
	{
		mFileBuffer[8] = 'V';
	}

	mDataStart = 12;

	char *dataPtr = mFileBuffer + mDataStart;

	bChunkInd dataChunk;
	dataChunk.code = 0;
	bool ignoreEndianFlag = true;

	//we always want to swap here

	int seek = getNextBlock(&dataChunk, dataPtr, mFlags);
	//dataPtr += ChunkUtils::getOffset(mFlags);
	char *dataPtrHead = 0;

	while (1)
	{
		// one behind
		if (dataChunk.code == B3_SDNA || dataChunk.code == B3_DNA1 || dataChunk.code == B3_TYPE || dataChunk.code == B3_TLEN || dataChunk.code == B3_STRC)
		{
			swapDNA(dataPtr);
			break;
		}
		else
		{
			//if (dataChunk.code == DNA1) break;
			dataPtrHead = dataPtr + ChunkUtils::getOffset(mFlags);

			swapLen(dataPtr);
			if (dataChunk.dna_nr >= 0)
			{
				swap(dataPtrHead, dataChunk, ignoreEndianFlag);
			}
			else
			{
				//printf("unknown chunk\n");
			}
		}

		// next please!
		dataPtr += seek;

		seek = getNextBlock(&dataChunk, dataPtr, mFlags);
		if (seek < 0)
			break;
	}

	if (mFlags & FD_ENDIAN_SWAP)
	{
		mFlags &= ~FD_ENDIAN_SWAP;
	}
	else
	{
		mFlags |= FD_ENDIAN_SWAP;
	}
}

// ----------------------------------------------------- //
char *bFile::readStruct(char *head, bChunkInd &dataChunk)
{
	bool ignoreEndianFlag = false;

	if (mFlags & FD_ENDIAN_SWAP)
		swap(head, dataChunk, ignoreEndianFlag);

	if (!mFileDNA->flagEqual(dataChunk.dna_nr))
	{
		// Ouch! need to rebuild the struct
		short *oldStruct, *curStruct;
		char *oldType, *newType;
		int oldLen, curLen, reverseOld;

		oldStruct = mFileDNA->getStruct(dataChunk.dna_nr);
		oldType = mFileDNA->getType(oldStruct[0]);

		oldLen = mFileDNA->getLength(oldStruct[0]);

		if ((mFlags & FD_BROKEN_DNA) != 0)
		{
			if ((strcmp(oldType, "b3QuantizedBvhNodeData") == 0) && oldLen == 20)
			{
				return 0;
			}
			if ((strcmp(oldType, "b3ShortIntIndexData") == 0))
			{
				int allocLen = 2;
				char *dataAlloc = new char[(dataChunk.nr * allocLen) + 1];
				memset(dataAlloc, 0, (dataChunk.nr * allocLen) + 1);
				short *dest = (short *)dataAlloc;
				const short *src = (short *)head;
				for (int i = 0; i < dataChunk.nr; i++)
				{
					dest[i] = src[i];
					if (mFlags & FD_ENDIAN_SWAP)
					{
						B3_SWITCH_SHORT(dest[i]);
					}
				}
				addDataBlock(dataAlloc);
				return dataAlloc;
			}
		}

		///don't try to convert Link block data, just memcpy it. Other data can be converted.
		if (strcmp("Link", oldType) != 0)
		{
			reverseOld = mMemoryDNA->getReverseType(oldType);

			if ((reverseOld != -1))
			{
				// make sure it's here
				//assert(reverseOld!= -1 && "getReverseType() returned -1, struct required!");

				//
				curStruct = mMemoryDNA->getStruct(reverseOld);
				newType = mMemoryDNA->getType(curStruct[0]);
				curLen = mMemoryDNA->getLength(curStruct[0]);

				// make sure it's the same
				assert((strcmp(oldType, newType) == 0) && "internal error, struct mismatch!");

				// numBlocks * length

				int allocLen = (curLen);
				char *dataAlloc = new char[(dataChunk.nr * allocLen) + 1];
				memset(dataAlloc, 0, (dataChunk.nr * allocLen));

				// track allocated
				addDataBlock(dataAlloc);

				char *cur = dataAlloc;
				char *old = head;
				for (int block = 0; block < dataChunk.nr; block++)
				{
					bool fixupPointers = true;
					parseStruct(cur, old, dataChunk.dna_nr, reverseOld, fixupPointers);
					mLibPointers.insert(old, (bStructHandle *)cur);

					cur += curLen;
					old += oldLen;
				}
				return dataAlloc;
			}
		}
		else
		{
			//printf("Link found\n");
		}
	}
	else
	{
//#define DEBUG_EQUAL_STRUCTS
#ifdef DEBUG_EQUAL_STRUCTS
		short *oldStruct;
		char *oldType;
		oldStruct = mFileDNA->getStruct(dataChunk.dna_nr);
		oldType = mFileDNA->getType(oldStruct[0]);
		printf("%s equal structure, just memcpy\n", oldType);
#endif  //
	}

	char *dataAlloc = new char[(dataChunk.len) + 1];
	memset(dataAlloc, 0, dataChunk.len + 1);

	// track allocated
	addDataBlock(dataAlloc);

	memcpy(dataAlloc, head, dataChunk.len);
	return dataAlloc;
}

// ----------------------------------------------------- //
void bFile::parseStruct(char *strcPtr, char *dtPtr, int old_dna, int new_dna, bool fixupPointers)
{
	if (old_dna == -1) return;
	if (new_dna == -1) return;

	//disable this, because we need to fixup pointers/ListBase
	if (0)  //mFileDNA->flagEqual(old_dna))
	{
		short *strc = mFileDNA->getStruct(old_dna);
		int len = mFileDNA->getLength(strc[0]);

		memcpy(strcPtr, dtPtr, len);
		return;
	}

	// Ok, now build the struct
	char *memType, *memName, *cpc, *cpo;
	short *fileStruct, *filePtrOld, *memoryStruct, *firstStruct;
	int elementLength, size, revType, old_nr, new_nr, fpLen;
	short firstStructType;

	// File to memory lookup
	memoryStruct = mMemoryDNA->getStruct(new_dna);
	fileStruct = mFileDNA->getStruct(old_dna);
	firstStruct = fileStruct;

	filePtrOld = fileStruct;
	firstStructType = mMemoryDNA->getStruct(0)[0];

	// Get number of elements
	elementLength = memoryStruct[1];
	memoryStruct += 2;

	cpc = strcPtr;
	cpo = 0;
	for (int ele = 0; ele < elementLength; ele++, memoryStruct += 2)
	{
		memType = mMemoryDNA->getType(memoryStruct[0]);
		memName = mMemoryDNA->getName(memoryStruct[1]);

		size = mMemoryDNA->getElementSize(memoryStruct[0], memoryStruct[1]);
		revType = mMemoryDNA->getReverseType(memoryStruct[0]);

		if (revType != -1 && memoryStruct[0] >= firstStructType && memName[0] != '*')
		{
			cpo = getFileElement(firstStruct, memName, memType, dtPtr, &filePtrOld);
			if (cpo)
			{
				int arrayLen = mFileDNA->getArraySizeNew(filePtrOld[1]);
				old_nr = mFileDNA->getReverseType(memType);
				new_nr = revType;
				fpLen = mFileDNA->getElementSize(filePtrOld[0], filePtrOld[1]);
				if (arrayLen == 1)
				{
					parseStruct(cpc, cpo, old_nr, new_nr, fixupPointers);
				}
				else
				{
					char *tmpCpc = cpc;
					char *tmpCpo = cpo;

					for (int i = 0; i < arrayLen; i++)
					{
						parseStruct(tmpCpc, tmpCpo, old_nr, new_nr, fixupPointers);
						tmpCpc += size / arrayLen;
						tmpCpo += fpLen / arrayLen;
					}
				}
				cpc += size;
				cpo += fpLen;
			}
			else
				cpc += size;
		}
		else
		{
			getMatchingFileDNA(fileStruct, memName, memType, cpc, dtPtr, fixupPointers);
			cpc += size;
		}
	}
}

// ----------------------------------------------------- //
static void getElement(int arrayLen, const char *cur, const char *old, char *oldPtr, char *curData)
{
#define b3GetEle(value, current, type, cast, size, ptr) \
	if (strcmp(current, type) == 0)                     \
	{                                                   \
		value = (*(cast *)ptr);                         \
		ptr += size;                                    \
	}

#define b3SetEle(value, current, type, cast, size, ptr) \
	if (strcmp(current, type) == 0)                     \
	{                                                   \
		(*(cast *)ptr) = (cast)value;                   \
		ptr += size;                                    \
	}
	double value = 0.0;

	for (int i = 0; i < arrayLen; i++)
	{
		b3GetEle(value, old, "char", char, sizeof(char), oldPtr);
		b3SetEle(value, cur, "char", char, sizeof(char), curData);
		b3GetEle(value, old, "short", short, sizeof(short), oldPtr);
		b3SetEle(value, cur, "short", short, sizeof(short), curData);
		b3GetEle(value, old, "ushort", unsigned short, sizeof(unsigned short), oldPtr);
		b3SetEle(value, cur, "ushort", unsigned short, sizeof(unsigned short), curData);
		b3GetEle(value, old, "int", int, sizeof(int), oldPtr);
		b3SetEle(value, cur, "int", int, sizeof(int), curData);
		b3GetEle(value, old, "long", int, sizeof(int), oldPtr);
		b3SetEle(value, cur, "long", int, sizeof(int), curData);
		b3GetEle(value, old, "float", float, sizeof(float), oldPtr);
		b3SetEle(value, cur, "float", float, sizeof(float), curData);
		b3GetEle(value, old, "double", double, sizeof(double), oldPtr);
		b3SetEle(value, cur, "double", double, sizeof(double), curData);
	}
}

// ----------------------------------------------------- //
void bFile::swapData(char *data, short type, int arraySize, bool ignoreEndianFlag)
{
	if (ignoreEndianFlag || (mFlags & FD_ENDIAN_SWAP))
	{
		if (type == 2 || type == 3)
		{
			short *sp = (short *)data;
			for (int i = 0; i < arraySize; i++)
			{
				sp[0] = ChunkUtils::swapShort(sp[0]);
				sp++;
			}
		}
		if (type > 3 && type < 8)
		{
			char c;
			char *cp = data;
			for (int i = 0; i < arraySize; i++)
			{
				c = cp[0];
				cp[0] = cp[3];
				cp[3] = c;
				c = cp[1];
				cp[1] = cp[2];
				cp[2] = c;
				cp += 4;
			}
		}
	}
}

void bFile::safeSwapPtr(char *dst, const char *src)
{
	int ptrFile = mFileDNA->getPointerSize();
	int ptrMem = mMemoryDNA->getPointerSize();

	if (!src && !dst)
		return;

	if (ptrFile == ptrMem)
	{
		memcpy(dst, src, ptrMem);
	}
	else if (ptrMem == 4 && ptrFile == 8)
	{
		b3PointerUid *oldPtr = (b3PointerUid *)src;
		b3PointerUid *newPtr = (b3PointerUid *)dst;

		if (oldPtr->m_uniqueIds[0] == oldPtr->m_uniqueIds[1])
		{
			//Bullet stores the 32bit unique ID in both upper and lower part of 64bit pointers
			//so it can be used to distinguish between .blend and .bullet
			newPtr->m_uniqueIds[0] = oldPtr->m_uniqueIds[0];
		}
		else
		{
			//deal with pointers the Blender .blend style way, see
			//readfile.c in the Blender source tree
			b3Long64 longValue = *((b3Long64 *)src);
			//endian swap for 64bit pointer otherwise truncation will fail due to trailing zeros
			if (mFlags & FD_ENDIAN_SWAP)
				B3_SWITCH_LONGINT(longValue);
			*((int *)dst) = (int)(longValue >> 3);
		}
	}
	else if (ptrMem == 8 && ptrFile == 4)
	{
		b3PointerUid *oldPtr = (b3PointerUid *)src;
		b3PointerUid *newPtr = (b3PointerUid *)dst;
		if (oldPtr->m_uniqueIds[0] == oldPtr->m_uniqueIds[1])
		{
			newPtr->m_uniqueIds[0] = oldPtr->m_uniqueIds[0];
			newPtr->m_uniqueIds[1] = 0;
		}
		else
		{
			*((b3Long64 *)dst) = *((int *)src);
		}
	}
	else
	{
		printf("%d %d\n", ptrFile, ptrMem);
		assert(0 && "Invalid pointer len");
	}
}

// ----------------------------------------------------- //
void bFile::getMatchingFileDNA(short *dna_addr, const char *lookupName, const char *lookupType, char *strcData, char *data, bool fixupPointers)
{
	// find the matching memory dna data
	// to the file being loaded. Fill the
	// memory with the file data...

	int len = dna_addr[1];
	dna_addr += 2;

	for (int i = 0; i < len; i++, dna_addr += 2)
	{
		const char *type = mFileDNA->getType(dna_addr[0]);
		const char *name = mFileDNA->getName(dna_addr[1]);

		int eleLen = mFileDNA->getElementSize(dna_addr[0], dna_addr[1]);

		if ((mFlags & FD_BROKEN_DNA) != 0)
		{
			if ((strcmp(type, "short") == 0) && (strcmp(name, "int") == 0))
			{
				eleLen = 0;
			}
		}

		if (strcmp(lookupName, name) == 0)
		{
			//int arrayLenold = mFileDNA->getArraySize((char*)name.c_str());
			int arrayLen = mFileDNA->getArraySizeNew(dna_addr[1]);
			//assert(arrayLenold == arrayLen);

			if (name[0] == '*')
			{
				// cast pointers
				int ptrFile = mFileDNA->getPointerSize();
				int ptrMem = mMemoryDNA->getPointerSize();
				safeSwapPtr(strcData, data);

				if (fixupPointers)
				{
					if (arrayLen > 1)
					{
						//void **sarray = (void**)strcData;
						//void **darray = (void**)data;

						char *cpc, *cpo;
						cpc = (char *)strcData;
						cpo = (char *)data;

						for (int a = 0; a < arrayLen; a++)
						{
							safeSwapPtr(cpc, cpo);
							m_pointerFixupArray.push_back(cpc);
							cpc += ptrMem;
							cpo += ptrFile;
						}
					}
					else
					{
						if (name[1] == '*')
							m_pointerPtrFixupArray.push_back(strcData);
						else
							m_pointerFixupArray.push_back(strcData);
					}
				}
				else
				{
					//					printf("skipped %s %s : %x\n",type.c_str(),name.c_str(),strcData);
				}
			}

			else if (strcmp(type, lookupType) == 0)
				memcpy(strcData, data, eleLen);
			else
				getElement(arrayLen, lookupType, type, data, strcData);

			// --
			return;
		}
		data += eleLen;
	}
}

// ----------------------------------------------------- //
char *bFile::getFileElement(short *firstStruct, char *lookupName, char *lookupType, char *data, short **foundPos)
{
	short *old = firstStruct;  //mFileDNA->getStruct(old_nr);
	int elementLength = old[1];
	old += 2;

	for (int i = 0; i < elementLength; i++, old += 2)
	{
		char *type = mFileDNA->getType(old[0]);
		char *name = mFileDNA->getName(old[1]);
		int len = mFileDNA->getElementSize(old[0], old[1]);

		if (strcmp(lookupName, name) == 0)
		{
			if (strcmp(type, lookupType) == 0)
			{
				if (foundPos)
					*foundPos = old;
				return data;
			}
			return 0;
		}
		data += len;
	}
	return 0;
}

// ----------------------------------------------------- //
void bFile::swapStruct(int dna_nr, char *data, bool ignoreEndianFlag)
{
	if (dna_nr == -1) return;

	short *strc = mFileDNA->getStruct(dna_nr);
	//short *firstStrc = strc;

	int elementLen = strc[1];
	strc += 2;

	short first = mFileDNA->getStruct(0)[0];

	char *buf = data;
	for (int i = 0; i < elementLen; i++, strc += 2)
	{
		char *type = mFileDNA->getType(strc[0]);
		char *name = mFileDNA->getName(strc[1]);

		int size = mFileDNA->getElementSize(strc[0], strc[1]);
		if (strc[0] >= first && name[0] != '*')
		{
			int old_nr = mFileDNA->getReverseType(type);
			int arrayLen = mFileDNA->getArraySizeNew(strc[1]);
			if (arrayLen == 1)
			{
				swapStruct(old_nr, buf, ignoreEndianFlag);
			}
			else
			{
				char *tmpBuf = buf;
				for (int i = 0; i < arrayLen; i++)
				{
					swapStruct(old_nr, tmpBuf, ignoreEndianFlag);
					tmpBuf += size / arrayLen;
				}
			}
		}
		else
		{
			//int arrayLenOld = mFileDNA->getArraySize(name);
			int arrayLen = mFileDNA->getArraySizeNew(strc[1]);
			//assert(arrayLenOld == arrayLen);
			swapData(buf, strc[0], arrayLen, ignoreEndianFlag);
		}
		buf += size;
	}
}

void bFile::resolvePointersMismatch()
{
	//	printf("resolvePointersStructMismatch\n");

	int i;

	for (i = 0; i < m_pointerFixupArray.size(); i++)
	{
		char *cur = m_pointerFixupArray.at(i);
		void **ptrptr = (void **)cur;
		void *ptr = *ptrptr;
		ptr = findLibPointer(ptr);
		if (ptr)
		{
			//printf("Fixup pointer!\n");
			*(ptrptr) = ptr;
		}
		else
		{
			//			printf("pointer not found: %x\n",cur);
		}
	}

	for (i = 0; i < m_pointerPtrFixupArray.size(); i++)
	{
		char *cur = m_pointerPtrFixupArray.at(i);
		void **ptrptr = (void **)cur;

		bChunkInd *block = m_chunkPtrPtrMap.find(*ptrptr);
		if (block)
		{
			int ptrMem = mMemoryDNA->getPointerSize();
			int ptrFile = mFileDNA->getPointerSize();

			int blockLen = block->len / ptrFile;

			void *onptr = findLibPointer(*ptrptr);
			if (onptr)
			{
				char *newPtr = new char[blockLen * ptrMem];
				addDataBlock(newPtr);
				memset(newPtr, 0, blockLen * ptrMem);

				void **onarray = (void **)onptr;
				char *oldPtr = (char *)onarray;

				int p = 0;
				while (blockLen-- > 0)
				{
					b3PointerUid dp = {{0}};
					safeSwapPtr((char *)dp.m_uniqueIds, oldPtr);

					void **tptr = (void **)(newPtr + p * ptrMem);
					*tptr = findLibPointer(dp.m_ptr);

					oldPtr += ptrFile;
					++p;
				}

				*ptrptr = newPtr;
			}
		}
	}
}

///this loop only works fine if the Blender DNA structure of the file matches the headerfiles
void bFile::resolvePointersChunk(const bChunkInd &dataChunk, int verboseMode)
{
	bParse::bDNA *fileDna = mFileDNA ? mFileDNA : mMemoryDNA;

	short int *oldStruct = fileDna->getStruct(dataChunk.dna_nr);
	short oldLen = fileDna->getLength(oldStruct[0]);
	//char* structType = fileDna->getType(oldStruct[0]);

	char *cur = (char *)findLibPointer(dataChunk.oldPtr);
	for (int block = 0; block < dataChunk.nr; block++)
	{
		resolvePointersStructRecursive(cur, dataChunk.dna_nr, verboseMode, 1);
		cur += oldLen;
	}
}

int bFile::resolvePointersStructRecursive(char *strcPtr, int dna_nr, int verboseMode, int recursion)
{
	bParse::bDNA *fileDna = mFileDNA ? mFileDNA : mMemoryDNA;

	char *memType;
	char *memName;
	short firstStructType = fileDna->getStruct(0)[0];

	char *elemPtr = strcPtr;

	short int *oldStruct = fileDna->getStruct(dna_nr);

	int elementLength = oldStruct[1];
	oldStruct += 2;

	int totalSize = 0;

	for (int ele = 0; ele < elementLength; ele++, oldStruct += 2)
	{
		memType = fileDna->getType(oldStruct[0]);
		memName = fileDna->getName(oldStruct[1]);

		int arrayLen = fileDna->getArraySizeNew(oldStruct[1]);
		if (memName[0] == '*')
		{
			if (arrayLen > 1)
			{
				void **array = (void **)elemPtr;
				for (int a = 0; a < arrayLen; a++)
				{
					if (verboseMode & FD_VERBOSE_EXPORT_XML)
					{
						for (int i = 0; i < recursion; i++)
						{
							printf("  ");
						}
						//skip the *
						printf("<%s type=\"pointer\"> ", &memName[1]);
						printf("%p ", array[a]);
						printf("</%s>\n", &memName[1]);
					}

					array[a] = findLibPointer(array[a]);
				}
			}
			else
			{
				void **ptrptr = (void **)elemPtr;
				void *ptr = *ptrptr;
				if (verboseMode & FD_VERBOSE_EXPORT_XML)
				{
					for (int i = 0; i < recursion; i++)
					{
						printf("  ");
					}
					printf("<%s type=\"pointer\"> ", &memName[1]);
					printf("%p ", ptr);
					printf("</%s>\n", &memName[1]);
				}
				ptr = findLibPointer(ptr);

				if (ptr)
				{
					//				printf("Fixup pointer at 0x%x from 0x%x to 0x%x!\n",ptrptr,*ptrptr,ptr);
					*(ptrptr) = ptr;
					if (memName[1] == '*' && ptrptr && *ptrptr)
					{
						// This	will only work if the given	**array	is continuous
						void **array = (void **)*(ptrptr);
						void *np = array[0];
						int n = 0;
						while (np)
						{
							np = findLibPointer(array[n]);
							if (np) array[n] = np;
							n++;
						}
					}
				}
				else
				{
					//				printf("Cannot fixup pointer at 0x%x from 0x%x to 0x%x!\n",ptrptr,*ptrptr,ptr);
				}
			}
		}
		else
		{
			int revType = fileDna->getReverseType(oldStruct[0]);
			if (oldStruct[0] >= firstStructType)  //revType != -1 &&
			{
				char cleanName[MAX_STRLEN];
				getCleanName(memName, cleanName);

				int arrayLen = fileDna->getArraySizeNew(oldStruct[1]);
				int byteOffset = 0;

				if (verboseMode & FD_VERBOSE_EXPORT_XML)
				{
					for (int i = 0; i < recursion; i++)
					{
						printf("  ");
					}

					if (arrayLen > 1)
					{
						printf("<%s type=\"%s\" count=%d>\n", cleanName, memType, arrayLen);
					}
					else
					{
						printf("<%s type=\"%s\">\n", cleanName, memType);
					}
				}

				for (int i = 0; i < arrayLen; i++)
				{
					byteOffset += resolvePointersStructRecursive(elemPtr + byteOffset, revType, verboseMode, recursion + 1);
				}
				if (verboseMode & FD_VERBOSE_EXPORT_XML)
				{
					for (int i = 0; i < recursion; i++)
					{
						printf("  ");
					}
					printf("</%s>\n", cleanName);
				}
			}
			else
			{
				//export a simple type
				if (verboseMode & FD_VERBOSE_EXPORT_XML)
				{
					if (arrayLen > MAX_ARRAY_LENGTH)
					{
						printf("too long\n");
					}
					else
					{
						//printf("%s %s\n",memType,memName);

						bool isIntegerType = (strcmp(memType, "char") == 0) || (strcmp(memType, "int") == 0) || (strcmp(memType, "short") == 0);

						if (isIntegerType)
						{
							const char *newtype = "int";
							int dbarray[MAX_ARRAY_LENGTH];
							int *dbPtr = 0;
							char *tmp = elemPtr;
							dbPtr = &dbarray[0];
							if (dbPtr)
							{
								char cleanName[MAX_STRLEN];
								getCleanName(memName, cleanName);

								int i;
								getElement(arrayLen, newtype, memType, tmp, (char *)dbPtr);
								for (i = 0; i < recursion; i++)
									printf("  ");
								if (arrayLen == 1)
									printf("<%s type=\"%s\">", cleanName, memType);
								else
									printf("<%s type=\"%s\" count=%d>", cleanName, memType, arrayLen);
								for (i = 0; i < arrayLen; i++)
									printf(" %d ", dbPtr[i]);
								printf("</%s>\n", cleanName);
							}
						}
						else
						{
							const char *newtype = "double";
							double dbarray[MAX_ARRAY_LENGTH];
							double *dbPtr = 0;
							char *tmp = elemPtr;
							dbPtr = &dbarray[0];
							if (dbPtr)
							{
								int i;
								getElement(arrayLen, newtype, memType, tmp, (char *)dbPtr);
								for (i = 0; i < recursion; i++)
									printf("  ");
								char cleanName[MAX_STRLEN];
								getCleanName(memName, cleanName);

								if (arrayLen == 1)
								{
									printf("<%s type=\"%s\">", memName, memType);
								}
								else
								{
									printf("<%s type=\"%s\" count=%d>", cleanName, memType, arrayLen);
								}
								for (i = 0; i < arrayLen; i++)
									printf(" %f ", dbPtr[i]);
								printf("</%s>\n", cleanName);
							}
						}
					}
				}
			}
		}

		int size = fileDna->getElementSize(oldStruct[0], oldStruct[1]);
		totalSize += size;
		elemPtr += size;
	}

	return totalSize;
}

///Resolve pointers replaces the original pointers in structures, and linked lists by the new in-memory structures
void bFile::resolvePointers(int verboseMode)
{
	bParse::bDNA *fileDna = mFileDNA ? mFileDNA : mMemoryDNA;

	//char *dataPtr = mFileBuffer+mDataStart;

	if (1)  //mFlags & (FD_BITS_VARIES | FD_VERSION_VARIES))
	{
		resolvePointersMismatch();
	}

	{
		if (verboseMode & FD_VERBOSE_EXPORT_XML)
		{
			printf("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
			int numitems = m_chunks.size();
			printf("<bullet_physics version=%d itemcount = %d>\n", b3GetVersion(), numitems);
		}
		for (int i = 0; i < m_chunks.size(); i++)
		{
			const bChunkInd &dataChunk = m_chunks.at(i);

			if (!mFileDNA || fileDna->flagEqual(dataChunk.dna_nr))
			{
				//dataChunk.len
				short int *oldStruct = fileDna->getStruct(dataChunk.dna_nr);
				char *oldType = fileDna->getType(oldStruct[0]);

				if (verboseMode & FD_VERBOSE_EXPORT_XML)
					printf(" <%s pointer=%p>\n", oldType, dataChunk.oldPtr);

				resolvePointersChunk(dataChunk, verboseMode);

				if (verboseMode & FD_VERBOSE_EXPORT_XML)
					printf(" </%s>\n", oldType);
			}
			else
			{
				//printf("skipping mStruct\n");
			}
		}
		if (verboseMode & FD_VERBOSE_EXPORT_XML)
		{
			printf("</bullet_physics>\n");
		}
	}
}

// ----------------------------------------------------- //
void *bFile::findLibPointer(void *ptr)
{
	bStructHandle **ptrptr = getLibPointers().find(ptr);
	if (ptrptr)
		return *ptrptr;
	return 0;
}

void bFile::updateOldPointers()
{
	int i;

	for (i = 0; i < m_chunks.size(); i++)
	{
		bChunkInd &dataChunk = m_chunks[i];
		dataChunk.oldPtr = findLibPointer(dataChunk.oldPtr);
	}
}
void bFile::dumpChunks(bParse::bDNA *dna)
{
	int i;

	for (i = 0; i < m_chunks.size(); i++)
	{
		bChunkInd &dataChunk = m_chunks[i];
		char *codeptr = (char *)&dataChunk.code;
		char codestr[5] = {codeptr[0], codeptr[1], codeptr[2], codeptr[3], 0};

		short *newStruct = dna->getStruct(dataChunk.dna_nr);
		char *typeName = dna->getType(newStruct[0]);
		printf("%3d: %s  ", i, typeName);

		printf("code=%s  ", codestr);

		printf("ptr=%p  ", dataChunk.oldPtr);
		printf("len=%d  ", dataChunk.len);
		printf("nr=%d  ", dataChunk.nr);
		if (dataChunk.nr != 1)
		{
			printf("not 1\n");
		}
		printf("\n");
	}

#if 0
	IDFinderData ifd;
	ifd.success = 0;
	ifd.IDname = NULL;
	ifd.just_print_it = 1;
	for (i=0; i<bf->m_blocks.size(); ++i)
	{
		BlendBlock* bb = bf->m_blocks[i];
		printf("tag='%s'\tptr=%p\ttype=%s\t[%4d]",		bb->tag, bb,bf->types[bb->type_index].name,bb->m_array_entries_.size());
		block_ID_finder(bb, bf, &ifd);
		printf("\n");
	}
#endif
}

void bFile::writeChunks(FILE *fp, bool fixupPointers)
{
	bParse::bDNA *fileDna = mFileDNA ? mFileDNA : mMemoryDNA;

	for (int i = 0; i < m_chunks.size(); i++)
	{
		bChunkInd &dataChunk = m_chunks.at(i);

		// Ouch! need to rebuild the struct
		short *oldStruct, *curStruct;
		char *oldType, *newType;
		int oldLen, curLen, reverseOld;

		oldStruct = fileDna->getStruct(dataChunk.dna_nr);
		oldType = fileDna->getType(oldStruct[0]);
		oldLen = fileDna->getLength(oldStruct[0]);
		///don't try to convert Link block data, just memcpy it. Other data can be converted.
		reverseOld = mMemoryDNA->getReverseType(oldType);

		if ((reverseOld != -1))
		{
			// make sure it's here
			//assert(reverseOld!= -1 && "getReverseType() returned -1, struct required!");
			//
			curStruct = mMemoryDNA->getStruct(reverseOld);
			newType = mMemoryDNA->getType(curStruct[0]);
			// make sure it's the same
			assert((strcmp(oldType, newType) == 0) && "internal error, struct mismatch!");

			curLen = mMemoryDNA->getLength(curStruct[0]);
			dataChunk.dna_nr = reverseOld;
			if (strcmp("Link", oldType) != 0)
			{
				dataChunk.len = curLen * dataChunk.nr;
			}
			else
			{
				//				printf("keep length of link = %d\n",dataChunk.len);
			}

			//write the structure header
			fwrite(&dataChunk, sizeof(bChunkInd), 1, fp);

			short int *curStruct1;
			curStruct1 = mMemoryDNA->getStruct(dataChunk.dna_nr);
			assert(curStruct1 == curStruct);

			char *cur = fixupPointers ? (char *)findLibPointer(dataChunk.oldPtr) : (char *)dataChunk.oldPtr;

			//write the actual contents of the structure(s)
			fwrite(cur, dataChunk.len, 1, fp);
		}
		else
		{
			printf("serious error, struct mismatch: don't write\n");
		}
	}
}

// ----------------------------------------------------- //
int bFile::getNextBlock(bChunkInd *dataChunk, const char *dataPtr, const int flags)
{
	bool swap = false;
	bool varies = false;

	if (flags & FD_ENDIAN_SWAP)
		swap = true;
	if (flags & FD_BITS_VARIES)
		varies = true;

	if (VOID_IS_8)
	{
		if (varies)
		{
			bChunkPtr4 head;
			memcpy(&head, dataPtr, sizeof(bChunkPtr4));

			bChunkPtr8 chunk;

			chunk.code = head.code;
			chunk.len = head.len;
			chunk.m_uniqueInts[0] = head.m_uniqueInt;
			chunk.m_uniqueInts[1] = 0;
			chunk.dna_nr = head.dna_nr;
			chunk.nr = head.nr;

			if (swap)
			{
				if ((chunk.code & 0xFFFF) == 0)
					chunk.code >>= 16;

				B3_SWITCH_INT(chunk.len);
				B3_SWITCH_INT(chunk.dna_nr);
				B3_SWITCH_INT(chunk.nr);
			}

			memcpy(dataChunk, &chunk, sizeof(bChunkInd));
		}
		else
		{
			bChunkPtr8 c;
			memcpy(&c, dataPtr, sizeof(bChunkPtr8));

			if (swap)
			{
				if ((c.code & 0xFFFF) == 0)
					c.code >>= 16;

				B3_SWITCH_INT(c.len);
				B3_SWITCH_INT(c.dna_nr);
				B3_SWITCH_INT(c.nr);
			}

			memcpy(dataChunk, &c, sizeof(bChunkInd));
		}
	}
	else
	{
		if (varies)
		{
			bChunkPtr8 head;
			memcpy(&head, dataPtr, sizeof(bChunkPtr8));

			bChunkPtr4 chunk;
			chunk.code = head.code;
			chunk.len = head.len;

			if (head.m_uniqueInts[0] == head.m_uniqueInts[1])
			{
				chunk.m_uniqueInt = head.m_uniqueInts[0];
			}
			else
			{
				b3Long64 oldPtr = 0;
				memcpy(&oldPtr, &head.m_uniqueInts[0], 8);
				if (swap)
					B3_SWITCH_LONGINT(oldPtr);
				chunk.m_uniqueInt = (int)(oldPtr >> 3);
			}

			chunk.dna_nr = head.dna_nr;
			chunk.nr = head.nr;

			if (swap)
			{
				if ((chunk.code & 0xFFFF) == 0)
					chunk.code >>= 16;

				B3_SWITCH_INT(chunk.len);
				B3_SWITCH_INT(chunk.dna_nr);
				B3_SWITCH_INT(chunk.nr);
			}

			memcpy(dataChunk, &chunk, sizeof(bChunkInd));
		}
		else
		{
			bChunkPtr4 c;
			memcpy(&c, dataPtr, sizeof(bChunkPtr4));

			if (swap)
			{
				if ((c.code & 0xFFFF) == 0)
					c.code >>= 16;

				B3_SWITCH_INT(c.len);
				B3_SWITCH_INT(c.dna_nr);
				B3_SWITCH_INT(c.nr);
			}
			memcpy(dataChunk, &c, sizeof(bChunkInd));
		}
	}

	if (dataChunk->len < 0)
		return -1;

#if 0
	print ("----------");
	print (dataChunk->code);
	print (dataChunk->len);
	print (dataChunk->old);
	print (dataChunk->dna_nr);
	print (dataChunk->nr);
#endif
	return (dataChunk->len + ChunkUtils::getOffset(flags));
}

//eof

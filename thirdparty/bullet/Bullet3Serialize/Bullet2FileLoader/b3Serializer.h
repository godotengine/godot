/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_SERIALIZER_H
#define B3_SERIALIZER_H

#include "Bullet3Common/b3Scalar.h"  // has definitions like B3_FORCE_INLINE
#include "Bullet3Common/b3StackAlloc.h"
#include "Bullet3Common/b3HashMap.h"

#if !defined(__CELLOS_LV2__) && !defined(__MWERKS__)
#include <memory.h>
#endif
#include <string.h>

extern char b3s_bulletDNAstr[];
extern int b3s_bulletDNAlen;
extern char b3s_bulletDNAstr64[];
extern int b3s_bulletDNAlen64;

B3_FORCE_INLINE int b3StrLen(const char* str)
{
	if (!str)
		return (0);
	int len = 0;

	while (*str != 0)
	{
		str++;
		len++;
	}

	return len;
}

class b3Chunk
{
public:
	int m_chunkCode;
	int m_length;
	void* m_oldPtr;
	int m_dna_nr;
	int m_number;
};

enum b3SerializationFlags
{
	B3_SERIALIZE_NO_BVH = 1,
	B3_SERIALIZE_NO_TRIANGLEINFOMAP = 2,
	B3_SERIALIZE_NO_DUPLICATE_ASSERT = 4
};

class b3Serializer
{
public:
	virtual ~b3Serializer() {}

	virtual const unsigned char* getBufferPointer() const = 0;

	virtual int getCurrentBufferSize() const = 0;

	virtual b3Chunk* allocate(size_t size, int numElements) = 0;

	virtual void finalizeChunk(b3Chunk* chunk, const char* structType, int chunkCode, void* oldPtr) = 0;

	virtual void* findPointer(void* oldPtr) = 0;

	virtual void* getUniquePointer(void* oldPtr) = 0;

	virtual void startSerialization() = 0;

	virtual void finishSerialization() = 0;

	virtual const char* findNameForPointer(const void* ptr) const = 0;

	virtual void registerNameForPointer(const void* ptr, const char* name) = 0;

	virtual void serializeName(const char* ptr) = 0;

	virtual int getSerializationFlags() const = 0;

	virtual void setSerializationFlags(int flags) = 0;
};

#define B3_HEADER_LENGTH 12
#if defined(__sgi) || defined(__sparc) || defined(__sparc__) || defined(__PPC__) || defined(__ppc__) || defined(__BIG_ENDIAN__)
#define B3_MAKE_ID(a, b, c, d) ((int)(a) << 24 | (int)(b) << 16 | (c) << 8 | (d))
#else
#define B3_MAKE_ID(a, b, c, d) ((int)(d) << 24 | (int)(c) << 16 | (b) << 8 | (a))
#endif

#define B3_SOFTBODY_CODE B3_MAKE_ID('S', 'B', 'D', 'Y')
#define B3_COLLISIONOBJECT_CODE B3_MAKE_ID('C', 'O', 'B', 'J')
#define B3_RIGIDBODY_CODE B3_MAKE_ID('R', 'B', 'D', 'Y')
#define B3_CONSTRAINT_CODE B3_MAKE_ID('C', 'O', 'N', 'S')
#define B3_BOXSHAPE_CODE B3_MAKE_ID('B', 'O', 'X', 'S')
#define B3_QUANTIZED_BVH_CODE B3_MAKE_ID('Q', 'B', 'V', 'H')
#define B3_TRIANLGE_INFO_MAP B3_MAKE_ID('T', 'M', 'A', 'P')
#define B3_SHAPE_CODE B3_MAKE_ID('S', 'H', 'A', 'P')
#define B3_ARRAY_CODE B3_MAKE_ID('A', 'R', 'A', 'Y')
#define B3_SBMATERIAL_CODE B3_MAKE_ID('S', 'B', 'M', 'T')
#define B3_SBNODE_CODE B3_MAKE_ID('S', 'B', 'N', 'D')
#define B3_DYNAMICSWORLD_CODE B3_MAKE_ID('D', 'W', 'L', 'D')
#define B3_DNA_CODE B3_MAKE_ID('D', 'N', 'A', '1')

struct b3PointerUid
{
	union {
		void* m_ptr;
		int m_uniqueIds[2];
	};
};

///The b3DefaultSerializer is the main Bullet serialization class.
///The constructor takes an optional argument for backwards compatibility, it is recommended to leave this empty/zero.
class b3DefaultSerializer : public b3Serializer
{
	b3AlignedObjectArray<char*> mTypes;
	b3AlignedObjectArray<short*> mStructs;
	b3AlignedObjectArray<short> mTlens;
	b3HashMap<b3HashInt, int> mStructReverse;
	b3HashMap<b3HashString, int> mTypeLookup;

	b3HashMap<b3HashPtr, void*> m_chunkP;

	b3HashMap<b3HashPtr, const char*> m_nameMap;

	b3HashMap<b3HashPtr, b3PointerUid> m_uniquePointers;
	int m_uniqueIdGenerator;

	int m_totalSize;
	unsigned char* m_buffer;
	int m_currentSize;
	void* m_dna;
	int m_dnaLength;

	int m_serializationFlags;

	b3AlignedObjectArray<b3Chunk*> m_chunkPtrs;

protected:
	virtual void* findPointer(void* oldPtr)
	{
		void** ptr = m_chunkP.find(oldPtr);
		if (ptr && *ptr)
			return *ptr;
		return 0;
	}

	void writeDNA()
	{
		b3Chunk* dnaChunk = allocate(m_dnaLength, 1);
		memcpy(dnaChunk->m_oldPtr, m_dna, m_dnaLength);
		finalizeChunk(dnaChunk, "DNA1", B3_DNA_CODE, m_dna);
	}

	int getReverseType(const char* type) const
	{
		b3HashString key(type);
		const int* valuePtr = mTypeLookup.find(key);
		if (valuePtr)
			return *valuePtr;

		return -1;
	}

	void initDNA(const char* bdnaOrg, int dnalen)
	{
		///was already initialized
		if (m_dna)
			return;

		int littleEndian = 1;
		littleEndian = ((char*)&littleEndian)[0];

		m_dna = b3AlignedAlloc(dnalen, 16);
		memcpy(m_dna, bdnaOrg, dnalen);
		m_dnaLength = dnalen;

		int* intPtr = 0;
		short* shtPtr = 0;
		char* cp = 0;
		int dataLen = 0;
		intPtr = (int*)m_dna;

		/*
				SDNA (4 bytes) (magic number)
				NAME (4 bytes)
				<nr> (4 bytes) amount of names (int)
				<string>
				<string>
			*/

		if (strncmp((const char*)m_dna, "SDNA", 4) == 0)
		{
			// skip ++ NAME
			intPtr++;
			intPtr++;
		}

		// Parse names
		if (!littleEndian)
			*intPtr = b3SwapEndian(*intPtr);

		dataLen = *intPtr;

		intPtr++;

		cp = (char*)intPtr;
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

		intPtr = (int*)cp;
		b3Assert(strncmp(cp, "TYPE", 4) == 0);
		intPtr++;

		if (!littleEndian)
			*intPtr = b3SwapEndian(*intPtr);

		dataLen = *intPtr;
		intPtr++;

		cp = (char*)intPtr;
		for (i = 0; i < dataLen; i++)
		{
			mTypes.push_back(cp);
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
		intPtr = (int*)cp;
		b3Assert(strncmp(cp, "TLEN", 4) == 0);
		intPtr++;

		dataLen = (int)mTypes.size();

		shtPtr = (short*)intPtr;
		for (i = 0; i < dataLen; i++, shtPtr++)
		{
			if (!littleEndian)
				shtPtr[0] = b3SwapEndian(shtPtr[0]);
			mTlens.push_back(shtPtr[0]);
		}

		if (dataLen & 1) shtPtr++;

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

		intPtr = (int*)shtPtr;
		cp = (char*)intPtr;
		b3Assert(strncmp(cp, "STRC", 4) == 0);
		intPtr++;

		if (!littleEndian)
			*intPtr = b3SwapEndian(*intPtr);
		dataLen = *intPtr;
		intPtr++;

		shtPtr = (short*)intPtr;
		for (i = 0; i < dataLen; i++)
		{
			mStructs.push_back(shtPtr);

			if (!littleEndian)
			{
				shtPtr[0] = b3SwapEndian(shtPtr[0]);
				shtPtr[1] = b3SwapEndian(shtPtr[1]);

				int len = shtPtr[1];
				shtPtr += 2;

				for (int a = 0; a < len; a++, shtPtr += 2)
				{
					shtPtr[0] = b3SwapEndian(shtPtr[0]);
					shtPtr[1] = b3SwapEndian(shtPtr[1]);
				}
			}
			else
			{
				shtPtr += (2 * shtPtr[1]) + 2;
			}
		}

		// build reverse lookups
		for (i = 0; i < (int)mStructs.size(); i++)
		{
			short* strc = mStructs.at(i);
			mStructReverse.insert(strc[0], i);
			mTypeLookup.insert(b3HashString(mTypes[strc[0]]), i);
		}
	}

public:
	b3DefaultSerializer(int totalSize = 0)
		: m_totalSize(totalSize),
		  m_currentSize(0),
		  m_dna(0),
		  m_dnaLength(0),
		  m_serializationFlags(0)
	{
		m_buffer = m_totalSize ? (unsigned char*)b3AlignedAlloc(totalSize, 16) : 0;

		const bool VOID_IS_8 = ((sizeof(void*) == 8));

#ifdef B3_INTERNAL_UPDATE_SERIALIZATION_STRUCTURES
		if (VOID_IS_8)
		{
#if _WIN64
			initDNA((const char*)b3s_bulletDNAstr64, b3s_bulletDNAlen64);
#else
			b3Assert(0);
#endif
		}
		else
		{
#ifndef _WIN64
			initDNA((const char*)b3s_bulletDNAstr, b3s_bulletDNAlen);
#else
			b3Assert(0);
#endif
		}

#else   //B3_INTERNAL_UPDATE_SERIALIZATION_STRUCTURES
		if (VOID_IS_8)
		{
			initDNA((const char*)b3s_bulletDNAstr64, b3s_bulletDNAlen64);
		}
		else
		{
			initDNA((const char*)b3s_bulletDNAstr, b3s_bulletDNAlen);
		}
#endif  //B3_INTERNAL_UPDATE_SERIALIZATION_STRUCTURES
	}

	virtual ~b3DefaultSerializer()
	{
		if (m_buffer)
			b3AlignedFree(m_buffer);
		if (m_dna)
			b3AlignedFree(m_dna);
	}

	void writeHeader(unsigned char* buffer) const
	{
#ifdef B3_USE_DOUBLE_PRECISION
		memcpy(buffer, "BULLETd", 7);
#else
		memcpy(buffer, "BULLETf", 7);
#endif  //B3_USE_DOUBLE_PRECISION

		int littleEndian = 1;
		littleEndian = ((char*)&littleEndian)[0];

		if (sizeof(void*) == 8)
		{
			buffer[7] = '-';
		}
		else
		{
			buffer[7] = '_';
		}

		if (littleEndian)
		{
			buffer[8] = 'v';
		}
		else
		{
			buffer[8] = 'V';
		}

		buffer[9] = '2';
		buffer[10] = '8';
		buffer[11] = '1';
	}

	virtual void startSerialization()
	{
		m_uniqueIdGenerator = 1;
		if (m_totalSize)
		{
			unsigned char* buffer = internalAlloc(B3_HEADER_LENGTH);
			writeHeader(buffer);
		}
	}

	virtual void finishSerialization()
	{
		writeDNA();

		//if we didn't pre-allocate a buffer, we need to create a contiguous buffer now
		int mysize = 0;
		if (!m_totalSize)
		{
			if (m_buffer)
				b3AlignedFree(m_buffer);

			m_currentSize += B3_HEADER_LENGTH;
			m_buffer = (unsigned char*)b3AlignedAlloc(m_currentSize, 16);

			unsigned char* currentPtr = m_buffer;
			writeHeader(m_buffer);
			currentPtr += B3_HEADER_LENGTH;
			mysize += B3_HEADER_LENGTH;
			for (int i = 0; i < m_chunkPtrs.size(); i++)
			{
				int curLength = sizeof(b3Chunk) + m_chunkPtrs[i]->m_length;
				memcpy(currentPtr, m_chunkPtrs[i], curLength);
				b3AlignedFree(m_chunkPtrs[i]);
				currentPtr += curLength;
				mysize += curLength;
			}
		}

		mTypes.clear();
		mStructs.clear();
		mTlens.clear();
		mStructReverse.clear();
		mTypeLookup.clear();
		m_chunkP.clear();
		m_nameMap.clear();
		m_uniquePointers.clear();
		m_chunkPtrs.clear();
	}

	virtual void* getUniquePointer(void* oldPtr)
	{
		if (!oldPtr)
			return 0;

		b3PointerUid* uptr = (b3PointerUid*)m_uniquePointers.find(oldPtr);
		if (uptr)
		{
			return uptr->m_ptr;
		}
		m_uniqueIdGenerator++;

		b3PointerUid uid;
		uid.m_uniqueIds[0] = m_uniqueIdGenerator;
		uid.m_uniqueIds[1] = m_uniqueIdGenerator;
		m_uniquePointers.insert(oldPtr, uid);
		return uid.m_ptr;
	}

	virtual const unsigned char* getBufferPointer() const
	{
		return m_buffer;
	}

	virtual int getCurrentBufferSize() const
	{
		return m_currentSize;
	}

	virtual void finalizeChunk(b3Chunk* chunk, const char* structType, int chunkCode, void* oldPtr)
	{
		if (!(m_serializationFlags & B3_SERIALIZE_NO_DUPLICATE_ASSERT))
		{
			b3Assert(!findPointer(oldPtr));
		}

		chunk->m_dna_nr = getReverseType(structType);

		chunk->m_chunkCode = chunkCode;

		void* uniquePtr = getUniquePointer(oldPtr);

		m_chunkP.insert(oldPtr, uniquePtr);  //chunk->m_oldPtr);
		chunk->m_oldPtr = uniquePtr;         //oldPtr;
	}

	virtual unsigned char* internalAlloc(size_t size)
	{
		unsigned char* ptr = 0;

		if (m_totalSize)
		{
			ptr = m_buffer + m_currentSize;
			m_currentSize += int(size);
			b3Assert(m_currentSize < m_totalSize);
		}
		else
		{
			ptr = (unsigned char*)b3AlignedAlloc(size, 16);
			m_currentSize += int(size);
		}
		return ptr;
	}

	virtual b3Chunk* allocate(size_t size, int numElements)
	{
		unsigned char* ptr = internalAlloc(int(size) * numElements + sizeof(b3Chunk));

		unsigned char* data = ptr + sizeof(b3Chunk);

		b3Chunk* chunk = (b3Chunk*)ptr;
		chunk->m_chunkCode = 0;
		chunk->m_oldPtr = data;
		chunk->m_length = int(size) * numElements;
		chunk->m_number = numElements;

		m_chunkPtrs.push_back(chunk);

		return chunk;
	}

	virtual const char* findNameForPointer(const void* ptr) const
	{
		const char* const* namePtr = m_nameMap.find(ptr);
		if (namePtr && *namePtr)
			return *namePtr;
		return 0;
	}

	virtual void registerNameForPointer(const void* ptr, const char* name)
	{
		m_nameMap.insert(ptr, name);
	}

	virtual void serializeName(const char* name)
	{
		if (name)
		{
			//don't serialize name twice
			if (findPointer((void*)name))
				return;

			int len = b3StrLen(name);
			if (len)
			{
				int newLen = len + 1;
				int padding = ((newLen + 3) & ~3) - newLen;
				newLen += padding;

				//serialize name string now
				b3Chunk* chunk = allocate(sizeof(char), newLen);
				char* destinationName = (char*)chunk->m_oldPtr;
				for (int i = 0; i < len; i++)
				{
					destinationName[i] = name[i];
				}
				destinationName[len] = 0;
				finalizeChunk(chunk, "char", B3_ARRAY_CODE, (void*)name);
			}
		}
	}

	virtual int getSerializationFlags() const
	{
		return m_serializationFlags;
	}

	virtual void setSerializationFlags(int flags)
	{
		m_serializationFlags = flags;
	}
};

#endif  //B3_SERIALIZER_H

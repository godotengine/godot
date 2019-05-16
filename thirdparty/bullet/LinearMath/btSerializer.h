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

#ifndef BT_SERIALIZER_H
#define BT_SERIALIZER_H

#include "btScalar.h"  // has definitions like SIMD_FORCE_INLINE
#include "btHashMap.h"

#if !defined(__CELLOS_LV2__) && !defined(__MWERKS__)
#include <memory.h>
#endif
#include <string.h>

extern char sBulletDNAstr[];
extern int sBulletDNAlen;
extern char sBulletDNAstr64[];
extern int sBulletDNAlen64;

SIMD_FORCE_INLINE int btStrLen(const char* str)
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

class btChunk
{
public:
	int m_chunkCode;
	int m_length;
	void* m_oldPtr;
	int m_dna_nr;
	int m_number;
};

enum btSerializationFlags
{
	BT_SERIALIZE_NO_BVH = 1,
	BT_SERIALIZE_NO_TRIANGLEINFOMAP = 2,
	BT_SERIALIZE_NO_DUPLICATE_ASSERT = 4,
	BT_SERIALIZE_CONTACT_MANIFOLDS = 8,
};

class btSerializer
{
public:
	virtual ~btSerializer() {}

	virtual const unsigned char* getBufferPointer() const = 0;

	virtual int getCurrentBufferSize() const = 0;

	virtual btChunk* allocate(size_t size, int numElements) = 0;

	virtual void finalizeChunk(btChunk* chunk, const char* structType, int chunkCode, void* oldPtr) = 0;

	virtual void* findPointer(void* oldPtr) = 0;

	virtual void* getUniquePointer(void* oldPtr) = 0;

	virtual void startSerialization() = 0;

	virtual void finishSerialization() = 0;

	virtual const char* findNameForPointer(const void* ptr) const = 0;

	virtual void registerNameForPointer(const void* ptr, const char* name) = 0;

	virtual void serializeName(const char* ptr) = 0;

	virtual int getSerializationFlags() const = 0;

	virtual void setSerializationFlags(int flags) = 0;

	virtual int getNumChunks() const = 0;

	virtual const btChunk* getChunk(int chunkIndex) const = 0;
};

#define BT_HEADER_LENGTH 12
#if defined(__sgi) || defined(__sparc) || defined(__sparc__) || defined(__PPC__) || defined(__ppc__) || defined(__BIG_ENDIAN__)
#define BT_MAKE_ID(a, b, c, d) ((int)(a) << 24 | (int)(b) << 16 | (c) << 8 | (d))
#else
#define BT_MAKE_ID(a, b, c, d) ((int)(d) << 24 | (int)(c) << 16 | (b) << 8 | (a))
#endif

#define BT_MULTIBODY_CODE BT_MAKE_ID('M', 'B', 'D', 'Y')
#define BT_MB_LINKCOLLIDER_CODE BT_MAKE_ID('M', 'B', 'L', 'C')
#define BT_SOFTBODY_CODE BT_MAKE_ID('S', 'B', 'D', 'Y')
#define BT_COLLISIONOBJECT_CODE BT_MAKE_ID('C', 'O', 'B', 'J')
#define BT_RIGIDBODY_CODE BT_MAKE_ID('R', 'B', 'D', 'Y')
#define BT_CONSTRAINT_CODE BT_MAKE_ID('C', 'O', 'N', 'S')
#define BT_BOXSHAPE_CODE BT_MAKE_ID('B', 'O', 'X', 'S')
#define BT_QUANTIZED_BVH_CODE BT_MAKE_ID('Q', 'B', 'V', 'H')
#define BT_TRIANLGE_INFO_MAP BT_MAKE_ID('T', 'M', 'A', 'P')
#define BT_SHAPE_CODE BT_MAKE_ID('S', 'H', 'A', 'P')
#define BT_ARRAY_CODE BT_MAKE_ID('A', 'R', 'A', 'Y')
#define BT_SBMATERIAL_CODE BT_MAKE_ID('S', 'B', 'M', 'T')
#define BT_SBNODE_CODE BT_MAKE_ID('S', 'B', 'N', 'D')
#define BT_DYNAMICSWORLD_CODE BT_MAKE_ID('D', 'W', 'L', 'D')
#define BT_CONTACTMANIFOLD_CODE BT_MAKE_ID('C', 'O', 'N', 'T')
#define BT_DNA_CODE BT_MAKE_ID('D', 'N', 'A', '1')

struct btPointerUid
{
	union {
		void* m_ptr;
		int m_uniqueIds[2];
	};
};

struct btBulletSerializedArrays
{
	btBulletSerializedArrays()
	{
	}
	btAlignedObjectArray<struct btQuantizedBvhDoubleData*> m_bvhsDouble;
	btAlignedObjectArray<struct btQuantizedBvhFloatData*> m_bvhsFloat;
	btAlignedObjectArray<struct btCollisionShapeData*> m_colShapeData;
	btAlignedObjectArray<struct btDynamicsWorldDoubleData*> m_dynamicWorldInfoDataDouble;
	btAlignedObjectArray<struct btDynamicsWorldFloatData*> m_dynamicWorldInfoDataFloat;
	btAlignedObjectArray<struct btRigidBodyDoubleData*> m_rigidBodyDataDouble;
	btAlignedObjectArray<struct btRigidBodyFloatData*> m_rigidBodyDataFloat;
	btAlignedObjectArray<struct btCollisionObjectDoubleData*> m_collisionObjectDataDouble;
	btAlignedObjectArray<struct btCollisionObjectFloatData*> m_collisionObjectDataFloat;
	btAlignedObjectArray<struct btTypedConstraintFloatData*> m_constraintDataFloat;
	btAlignedObjectArray<struct btTypedConstraintDoubleData*> m_constraintDataDouble;
	btAlignedObjectArray<struct btTypedConstraintData*> m_constraintData;  //for backwards compatibility
	btAlignedObjectArray<struct btSoftBodyFloatData*> m_softBodyFloatData;
	btAlignedObjectArray<struct btSoftBodyDoubleData*> m_softBodyDoubleData;
};

///The btDefaultSerializer is the main Bullet serialization class.
///The constructor takes an optional argument for backwards compatibility, it is recommended to leave this empty/zero.
class btDefaultSerializer : public btSerializer
{
protected:
	btAlignedObjectArray<char*> mTypes;
	btAlignedObjectArray<short*> mStructs;
	btAlignedObjectArray<short> mTlens;
	btHashMap<btHashInt, int> mStructReverse;
	btHashMap<btHashString, int> mTypeLookup;

	btHashMap<btHashPtr, void*> m_chunkP;

	btHashMap<btHashPtr, const char*> m_nameMap;

	btHashMap<btHashPtr, btPointerUid> m_uniquePointers;
	int m_uniqueIdGenerator;

	int m_totalSize;
	unsigned char* m_buffer;
	bool m_ownsBuffer;
	int m_currentSize;
	void* m_dna;
	int m_dnaLength;

	int m_serializationFlags;

	btAlignedObjectArray<btChunk*> m_chunkPtrs;

protected:
	virtual void* findPointer(void* oldPtr)
	{
		void** ptr = m_chunkP.find(oldPtr);
		if (ptr && *ptr)
			return *ptr;
		return 0;
	}

	virtual void writeDNA()
	{
		btChunk* dnaChunk = allocate(m_dnaLength, 1);
		memcpy(dnaChunk->m_oldPtr, m_dna, m_dnaLength);
		finalizeChunk(dnaChunk, "DNA1", BT_DNA_CODE, m_dna);
	}

	int getReverseType(const char* type) const
	{
		btHashString key(type);
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

		m_dna = btAlignedAlloc(dnalen, 16);
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
			*intPtr = btSwapEndian(*intPtr);

		dataLen = *intPtr;

		intPtr++;

		cp = (char*)intPtr;
		int i;
		for (i = 0; i < dataLen; i++)
		{
			while (*cp) cp++;
			cp++;
		}
		cp = btAlignPointer(cp, 4);

		/*
				TYPE (4 bytes)
				<nr> amount of types (int)
				<string>
				<string>
			*/

		intPtr = (int*)cp;
		btAssert(strncmp(cp, "TYPE", 4) == 0);
		intPtr++;

		if (!littleEndian)
			*intPtr = btSwapEndian(*intPtr);

		dataLen = *intPtr;
		intPtr++;

		cp = (char*)intPtr;
		for (i = 0; i < dataLen; i++)
		{
			mTypes.push_back(cp);
			while (*cp) cp++;
			cp++;
		}

		cp = btAlignPointer(cp, 4);

		/*
				TLEN (4 bytes)
				<len> (short) the lengths of types
				<len>
			*/

		// Parse type lens
		intPtr = (int*)cp;
		btAssert(strncmp(cp, "TLEN", 4) == 0);
		intPtr++;

		dataLen = (int)mTypes.size();

		shtPtr = (short*)intPtr;
		for (i = 0; i < dataLen; i++, shtPtr++)
		{
			if (!littleEndian)
				shtPtr[0] = btSwapEndian(shtPtr[0]);
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
		btAssert(strncmp(cp, "STRC", 4) == 0);
		intPtr++;

		if (!littleEndian)
			*intPtr = btSwapEndian(*intPtr);
		dataLen = *intPtr;
		intPtr++;

		shtPtr = (short*)intPtr;
		for (i = 0; i < dataLen; i++)
		{
			mStructs.push_back(shtPtr);

			if (!littleEndian)
			{
				shtPtr[0] = btSwapEndian(shtPtr[0]);
				shtPtr[1] = btSwapEndian(shtPtr[1]);

				int len = shtPtr[1];
				shtPtr += 2;

				for (int a = 0; a < len; a++, shtPtr += 2)
				{
					shtPtr[0] = btSwapEndian(shtPtr[0]);
					shtPtr[1] = btSwapEndian(shtPtr[1]);
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
			mTypeLookup.insert(btHashString(mTypes[strc[0]]), i);
		}
	}

public:
	btHashMap<btHashPtr, void*> m_skipPointers;

	btDefaultSerializer(int totalSize = 0, unsigned char* buffer = 0)
		: m_uniqueIdGenerator(0),
		  m_totalSize(totalSize),
		  m_currentSize(0),
		  m_dna(0),
		  m_dnaLength(0),
		  m_serializationFlags(0)
	{
		if (buffer == 0)
		{
			m_buffer = m_totalSize ? (unsigned char*)btAlignedAlloc(totalSize, 16) : 0;
			m_ownsBuffer = true;
		}
		else
		{
			m_buffer = buffer;
			m_ownsBuffer = false;
		}

		const bool VOID_IS_8 = ((sizeof(void*) == 8));

#ifdef BT_INTERNAL_UPDATE_SERIALIZATION_STRUCTURES
		if (VOID_IS_8)
		{
#if _WIN64
			initDNA((const char*)sBulletDNAstr64, sBulletDNAlen64);
#else
			btAssert(0);
#endif
		}
		else
		{
#ifndef _WIN64
			initDNA((const char*)sBulletDNAstr, sBulletDNAlen);
#else
			btAssert(0);
#endif
		}

#else   //BT_INTERNAL_UPDATE_SERIALIZATION_STRUCTURES
		if (VOID_IS_8)
		{
			initDNA((const char*)sBulletDNAstr64, sBulletDNAlen64);
		}
		else
		{
			initDNA((const char*)sBulletDNAstr, sBulletDNAlen);
		}
#endif  //BT_INTERNAL_UPDATE_SERIALIZATION_STRUCTURES
	}

	virtual ~btDefaultSerializer()
	{
		if (m_buffer && m_ownsBuffer)
			btAlignedFree(m_buffer);
		if (m_dna)
			btAlignedFree(m_dna);
	}

	static int getMemoryDnaSizeInBytes()
	{
		const bool VOID_IS_8 = ((sizeof(void*) == 8));

		if (VOID_IS_8)
		{
			return sBulletDNAlen64;
		}
		return sBulletDNAlen;
	}
	static const char* getMemoryDna()
	{
		const bool VOID_IS_8 = ((sizeof(void*) == 8));
		if (VOID_IS_8)
		{
			return (const char*)sBulletDNAstr64;
		}
		return (const char*)sBulletDNAstr;
	}

	void insertHeader()
	{
		writeHeader(m_buffer);
		m_currentSize += BT_HEADER_LENGTH;
	}

	void writeHeader(unsigned char* buffer) const
	{
#ifdef BT_USE_DOUBLE_PRECISION
		memcpy(buffer, "BULLETd", 7);
#else
		memcpy(buffer, "BULLETf", 7);
#endif  //BT_USE_DOUBLE_PRECISION

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
		buffer[11] = '8';
	}

	virtual void startSerialization()
	{
		m_uniqueIdGenerator = 1;
		if (m_totalSize)
		{
			unsigned char* buffer = internalAlloc(BT_HEADER_LENGTH);
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
				btAlignedFree(m_buffer);

			m_currentSize += BT_HEADER_LENGTH;
			m_buffer = (unsigned char*)btAlignedAlloc(m_currentSize, 16);

			unsigned char* currentPtr = m_buffer;
			writeHeader(m_buffer);
			currentPtr += BT_HEADER_LENGTH;
			mysize += BT_HEADER_LENGTH;
			for (int i = 0; i < m_chunkPtrs.size(); i++)
			{
				int curLength = sizeof(btChunk) + m_chunkPtrs[i]->m_length;
				memcpy(currentPtr, m_chunkPtrs[i], curLength);
				btAlignedFree(m_chunkPtrs[i]);
				currentPtr += curLength;
				mysize += curLength;
			}
		}

		mTypes.clear();
		mStructs.clear();
		mTlens.clear();
		mStructReverse.clear();
		mTypeLookup.clear();
		m_skipPointers.clear();
		m_chunkP.clear();
		m_nameMap.clear();
		m_uniquePointers.clear();
		m_chunkPtrs.clear();
	}

	virtual void* getUniquePointer(void* oldPtr)
	{
		btAssert(m_uniqueIdGenerator >= 0);
		if (!oldPtr)
			return 0;

		btPointerUid* uptr = (btPointerUid*)m_uniquePointers.find(oldPtr);
		if (uptr)
		{
			return uptr->m_ptr;
		}

		void** ptr2 = m_skipPointers[oldPtr];
		if (ptr2)
		{
			return 0;
		}

		m_uniqueIdGenerator++;

		btPointerUid uid;
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

	virtual void finalizeChunk(btChunk* chunk, const char* structType, int chunkCode, void* oldPtr)
	{
		if (!(m_serializationFlags & BT_SERIALIZE_NO_DUPLICATE_ASSERT))
		{
			btAssert(!findPointer(oldPtr));
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
			btAssert(m_currentSize < m_totalSize);
		}
		else
		{
			ptr = (unsigned char*)btAlignedAlloc(size, 16);
			m_currentSize += int(size);
		}
		return ptr;
	}

	virtual btChunk* allocate(size_t size, int numElements)
	{
		unsigned char* ptr = internalAlloc(int(size) * numElements + sizeof(btChunk));

		unsigned char* data = ptr + sizeof(btChunk);

		btChunk* chunk = (btChunk*)ptr;
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

			int len = btStrLen(name);
			if (len)
			{
				int newLen = len + 1;
				int padding = ((newLen + 3) & ~3) - newLen;
				newLen += padding;

				//serialize name string now
				btChunk* chunk = allocate(sizeof(char), newLen);
				char* destinationName = (char*)chunk->m_oldPtr;
				for (int i = 0; i < len; i++)
				{
					destinationName[i] = name[i];
				}
				destinationName[len] = 0;
				finalizeChunk(chunk, "char", BT_ARRAY_CODE, (void*)name);
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
	int getNumChunks() const
	{
		return m_chunkPtrs.size();
	}

	const btChunk* getChunk(int chunkIndex) const
	{
		return m_chunkPtrs[chunkIndex];
	}
};

///In general it is best to use btDefaultSerializer,
///in particular when writing the data to disk or sending it over the network.
///The btInMemorySerializer is experimental and only suitable in a few cases.
///The btInMemorySerializer takes a shortcut and can be useful to create a deep-copy
///of objects. There will be a demo on how to use the btInMemorySerializer.
#ifdef ENABLE_INMEMORY_SERIALIZER

struct btInMemorySerializer : public btDefaultSerializer
{
	btHashMap<btHashPtr, btChunk*> m_uid2ChunkPtr;
	btHashMap<btHashPtr, void*> m_orgPtr2UniqueDataPtr;
	btHashMap<btHashString, const void*> m_names2Ptr;

	btBulletSerializedArrays m_arrays;

	btInMemorySerializer(int totalSize = 0, unsigned char* buffer = 0)
		: btDefaultSerializer(totalSize, buffer)
	{
	}

	virtual void startSerialization()
	{
		m_uid2ChunkPtr.clear();
		//todo: m_arrays.clear();
		btDefaultSerializer::startSerialization();
	}

	btChunk* findChunkFromUniquePointer(void* uniquePointer)
	{
		btChunk** chkPtr = m_uid2ChunkPtr[uniquePointer];
		if (chkPtr)
		{
			return *chkPtr;
		}
		return 0;
	}

	virtual void registerNameForPointer(const void* ptr, const char* name)
	{
		btDefaultSerializer::registerNameForPointer(ptr, name);
		m_names2Ptr.insert(name, ptr);
	}

	virtual void finishSerialization()
	{
	}

	virtual void* getUniquePointer(void* oldPtr)
	{
		if (oldPtr == 0)
			return 0;

		// void* uniquePtr = getUniquePointer(oldPtr);
		btChunk* chunk = findChunkFromUniquePointer(oldPtr);
		if (chunk)
		{
			return chunk->m_oldPtr;
		}
		else
		{
			const char* n = (const char*)oldPtr;
			const void** ptr = m_names2Ptr[n];
			if (ptr)
			{
				return oldPtr;
			}
			else
			{
				void** ptr2 = m_skipPointers[oldPtr];
				if (ptr2)
				{
					return 0;
				}
				else
				{
					//If this assert hit, serialization happened in the wrong order
					// 'getUniquePointer'
					btAssert(0);
				}
			}
			return 0;
		}
		return oldPtr;
	}

	virtual void finalizeChunk(btChunk* chunk, const char* structType, int chunkCode, void* oldPtr)
	{
		if (!(m_serializationFlags & BT_SERIALIZE_NO_DUPLICATE_ASSERT))
		{
			btAssert(!findPointer(oldPtr));
		}

		chunk->m_dna_nr = getReverseType(structType);
		chunk->m_chunkCode = chunkCode;
		//void* uniquePtr = getUniquePointer(oldPtr);
		m_chunkP.insert(oldPtr, oldPtr);  //chunk->m_oldPtr);
		// chunk->m_oldPtr = uniquePtr;//oldPtr;

		void* uid = findPointer(oldPtr);
		m_uid2ChunkPtr.insert(uid, chunk);

		switch (chunk->m_chunkCode)
		{
			case BT_SOFTBODY_CODE:
			{
#ifdef BT_USE_DOUBLE_PRECISION
				m_arrays.m_softBodyDoubleData.push_back((btSoftBodyDoubleData*)chunk->m_oldPtr);
#else
				m_arrays.m_softBodyFloatData.push_back((btSoftBodyFloatData*)chunk->m_oldPtr);
#endif
				break;
			}
			case BT_COLLISIONOBJECT_CODE:
			{
#ifdef BT_USE_DOUBLE_PRECISION
				m_arrays.m_collisionObjectDataDouble.push_back((btCollisionObjectDoubleData*)chunk->m_oldPtr);
#else   //BT_USE_DOUBLE_PRECISION
				m_arrays.m_collisionObjectDataFloat.push_back((btCollisionObjectFloatData*)chunk->m_oldPtr);
#endif  //BT_USE_DOUBLE_PRECISION
				break;
			}
			case BT_RIGIDBODY_CODE:
			{
#ifdef BT_USE_DOUBLE_PRECISION
				m_arrays.m_rigidBodyDataDouble.push_back((btRigidBodyDoubleData*)chunk->m_oldPtr);
#else
				m_arrays.m_rigidBodyDataFloat.push_back((btRigidBodyFloatData*)chunk->m_oldPtr);
#endif  //BT_USE_DOUBLE_PRECISION
				break;
			};
			case BT_CONSTRAINT_CODE:
			{
#ifdef BT_USE_DOUBLE_PRECISION
				m_arrays.m_constraintDataDouble.push_back((btTypedConstraintDoubleData*)chunk->m_oldPtr);
#else
				m_arrays.m_constraintDataFloat.push_back((btTypedConstraintFloatData*)chunk->m_oldPtr);
#endif
				break;
			}
			case BT_QUANTIZED_BVH_CODE:
			{
#ifdef BT_USE_DOUBLE_PRECISION
				m_arrays.m_bvhsDouble.push_back((btQuantizedBvhDoubleData*)chunk->m_oldPtr);
#else
				m_arrays.m_bvhsFloat.push_back((btQuantizedBvhFloatData*)chunk->m_oldPtr);
#endif
				break;
			}

			case BT_SHAPE_CODE:
			{
				btCollisionShapeData* shapeData = (btCollisionShapeData*)chunk->m_oldPtr;
				m_arrays.m_colShapeData.push_back(shapeData);
				break;
			}
			case BT_TRIANLGE_INFO_MAP:
			case BT_ARRAY_CODE:
			case BT_SBMATERIAL_CODE:
			case BT_SBNODE_CODE:
			case BT_DYNAMICSWORLD_CODE:
			case BT_DNA_CODE:
			{
				break;
			}
			default:
			{
			}
		};
	}

	int getNumChunks() const
	{
		return m_uid2ChunkPtr.size();
	}

	const btChunk* getChunk(int chunkIndex) const
	{
		return *m_uid2ChunkPtr.getAtIndex(chunkIndex);
	}
};
#endif  //ENABLE_INMEMORY_SERIALIZER

#endif  //BT_SERIALIZER_H

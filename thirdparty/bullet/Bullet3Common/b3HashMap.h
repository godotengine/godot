/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_HASH_MAP_H
#define B3_HASH_MAP_H

#include "b3AlignedObjectArray.h"

#include <string>

///very basic hashable string implementation, compatible with b3HashMap
struct b3HashString
{
	std::string m_string;
	unsigned int m_hash;

	B3_FORCE_INLINE unsigned int getHash() const
	{
		return m_hash;
	}

	b3HashString(const char* name)
		: m_string(name)
	{
		/* magic numbers from http://www.isthe.com/chongo/tech/comp/fnv/ */
		static const unsigned int InitialFNV = 2166136261u;
		static const unsigned int FNVMultiple = 16777619u;

		/* Fowler / Noll / Vo (FNV) Hash */
		unsigned int hash = InitialFNV;
		int len = m_string.length();
		for (int i = 0; i < len; i++)
		{
			hash = hash ^ (m_string[i]); /* xor  the low 8 bits */
			hash = hash * FNVMultiple;   /* multiply by the magic number */
		}
		m_hash = hash;
	}

	int portableStringCompare(const char* src, const char* dst) const
	{
		int ret = 0;

		while (!(ret = *(unsigned char*)src - *(unsigned char*)dst) && *dst)
			++src, ++dst;

		if (ret < 0)
			ret = -1;
		else if (ret > 0)
			ret = 1;

		return (ret);
	}

	bool equals(const b3HashString& other) const
	{
		return (m_string == other.m_string);
	}
};

const int B3_HASH_NULL = 0xffffffff;

class b3HashInt
{
	int m_uid;

public:
	b3HashInt(int uid) : m_uid(uid)
	{
	}

	int getUid1() const
	{
		return m_uid;
	}

	void setUid1(int uid)
	{
		m_uid = uid;
	}

	bool equals(const b3HashInt& other) const
	{
		return getUid1() == other.getUid1();
	}
	//to our success
	B3_FORCE_INLINE unsigned int getHash() const
	{
		int key = m_uid;
		// Thomas Wang's hash
		key += ~(key << 15);
		key ^= (key >> 10);
		key += (key << 3);
		key ^= (key >> 6);
		key += ~(key << 11);
		key ^= (key >> 16);
		return key;
	}
};

class b3HashPtr
{
	union {
		const void* m_pointer;
		int m_hashValues[2];
	};

public:
	b3HashPtr(const void* ptr)
		: m_pointer(ptr)
	{
	}

	const void* getPointer() const
	{
		return m_pointer;
	}

	bool equals(const b3HashPtr& other) const
	{
		return getPointer() == other.getPointer();
	}

	//to our success
	B3_FORCE_INLINE unsigned int getHash() const
	{
		const bool VOID_IS_8 = ((sizeof(void*) == 8));

		int key = VOID_IS_8 ? m_hashValues[0] + m_hashValues[1] : m_hashValues[0];

		// Thomas Wang's hash
		key += ~(key << 15);
		key ^= (key >> 10);
		key += (key << 3);
		key ^= (key >> 6);
		key += ~(key << 11);
		key ^= (key >> 16);
		return key;
	}
};

template <class Value>
class b3HashKeyPtr
{
	int m_uid;

public:
	b3HashKeyPtr(int uid) : m_uid(uid)
	{
	}

	int getUid1() const
	{
		return m_uid;
	}

	bool equals(const b3HashKeyPtr<Value>& other) const
	{
		return getUid1() == other.getUid1();
	}

	//to our success
	B3_FORCE_INLINE unsigned int getHash() const
	{
		int key = m_uid;
		// Thomas Wang's hash
		key += ~(key << 15);
		key ^= (key >> 10);
		key += (key << 3);
		key ^= (key >> 6);
		key += ~(key << 11);
		key ^= (key >> 16);
		return key;
	}
};

template <class Value>
class b3HashKey
{
	int m_uid;

public:
	b3HashKey(int uid) : m_uid(uid)
	{
	}

	int getUid1() const
	{
		return m_uid;
	}

	bool equals(const b3HashKey<Value>& other) const
	{
		return getUid1() == other.getUid1();
	}
	//to our success
	B3_FORCE_INLINE unsigned int getHash() const
	{
		int key = m_uid;
		// Thomas Wang's hash
		key += ~(key << 15);
		key ^= (key >> 10);
		key += (key << 3);
		key ^= (key >> 6);
		key += ~(key << 11);
		key ^= (key >> 16);
		return key;
	}
};

///The b3HashMap template class implements a generic and lightweight hashmap.
///A basic sample of how to use b3HashMap is located in Demos\BasicDemo\main.cpp
template <class Key, class Value>
class b3HashMap
{
protected:
	b3AlignedObjectArray<int> m_hashTable;
	b3AlignedObjectArray<int> m_next;

	b3AlignedObjectArray<Value> m_valueArray;
	b3AlignedObjectArray<Key> m_keyArray;

	void growTables(const Key& /*key*/)
	{
		int newCapacity = m_valueArray.capacity();

		if (m_hashTable.size() < newCapacity)
		{
			//grow hashtable and next table
			int curHashtableSize = m_hashTable.size();

			m_hashTable.resize(newCapacity);
			m_next.resize(newCapacity);

			int i;

			for (i = 0; i < newCapacity; ++i)
			{
				m_hashTable[i] = B3_HASH_NULL;
			}
			for (i = 0; i < newCapacity; ++i)
			{
				m_next[i] = B3_HASH_NULL;
			}

			for (i = 0; i < curHashtableSize; i++)
			{
				//const Value& value = m_valueArray[i];
				//const Key& key = m_keyArray[i];

				int hashValue = m_keyArray[i].getHash() & (m_valueArray.capacity() - 1);  // New hash value with new mask
				m_next[i] = m_hashTable[hashValue];
				m_hashTable[hashValue] = i;
			}
		}
	}

public:
	void insert(const Key& key, const Value& value)
	{
		int hash = key.getHash() & (m_valueArray.capacity() - 1);

		//replace value if the key is already there
		int index = findIndex(key);
		if (index != B3_HASH_NULL)
		{
			m_valueArray[index] = value;
			return;
		}

		int count = m_valueArray.size();
		int oldCapacity = m_valueArray.capacity();
		m_valueArray.push_back(value);
		m_keyArray.push_back(key);

		int newCapacity = m_valueArray.capacity();
		if (oldCapacity < newCapacity)
		{
			growTables(key);
			//hash with new capacity
			hash = key.getHash() & (m_valueArray.capacity() - 1);
		}
		m_next[count] = m_hashTable[hash];
		m_hashTable[hash] = count;
	}

	void remove(const Key& key)
	{
		int hash = key.getHash() & (m_valueArray.capacity() - 1);

		int pairIndex = findIndex(key);

		if (pairIndex == B3_HASH_NULL)
		{
			return;
		}

		// Remove the pair from the hash table.
		int index = m_hashTable[hash];
		b3Assert(index != B3_HASH_NULL);

		int previous = B3_HASH_NULL;
		while (index != pairIndex)
		{
			previous = index;
			index = m_next[index];
		}

		if (previous != B3_HASH_NULL)
		{
			b3Assert(m_next[previous] == pairIndex);
			m_next[previous] = m_next[pairIndex];
		}
		else
		{
			m_hashTable[hash] = m_next[pairIndex];
		}

		// We now move the last pair into spot of the
		// pair being removed. We need to fix the hash
		// table indices to support the move.

		int lastPairIndex = m_valueArray.size() - 1;

		// If the removed pair is the last pair, we are done.
		if (lastPairIndex == pairIndex)
		{
			m_valueArray.pop_back();
			m_keyArray.pop_back();
			return;
		}

		// Remove the last pair from the hash table.
		int lastHash = m_keyArray[lastPairIndex].getHash() & (m_valueArray.capacity() - 1);

		index = m_hashTable[lastHash];
		b3Assert(index != B3_HASH_NULL);

		previous = B3_HASH_NULL;
		while (index != lastPairIndex)
		{
			previous = index;
			index = m_next[index];
		}

		if (previous != B3_HASH_NULL)
		{
			b3Assert(m_next[previous] == lastPairIndex);
			m_next[previous] = m_next[lastPairIndex];
		}
		else
		{
			m_hashTable[lastHash] = m_next[lastPairIndex];
		}

		// Copy the last pair into the remove pair's spot.
		m_valueArray[pairIndex] = m_valueArray[lastPairIndex];
		m_keyArray[pairIndex] = m_keyArray[lastPairIndex];

		// Insert the last pair into the hash table
		m_next[pairIndex] = m_hashTable[lastHash];
		m_hashTable[lastHash] = pairIndex;

		m_valueArray.pop_back();
		m_keyArray.pop_back();
	}

	int size() const
	{
		return m_valueArray.size();
	}

	const Value* getAtIndex(int index) const
	{
		b3Assert(index < m_valueArray.size());

		return &m_valueArray[index];
	}

	Value* getAtIndex(int index)
	{
		b3Assert(index < m_valueArray.size());

		return &m_valueArray[index];
	}

	Key getKeyAtIndex(int index)
	{
		b3Assert(index < m_keyArray.size());
		return m_keyArray[index];
	}

	const Key getKeyAtIndex(int index) const
	{
		b3Assert(index < m_keyArray.size());
		return m_keyArray[index];
	}

	Value* operator[](const Key& key)
	{
		return find(key);
	}

	const Value* find(const Key& key) const
	{
		int index = findIndex(key);
		if (index == B3_HASH_NULL)
		{
			return NULL;
		}
		return &m_valueArray[index];
	}

	Value* find(const Key& key)
	{
		int index = findIndex(key);
		if (index == B3_HASH_NULL)
		{
			return NULL;
		}
		return &m_valueArray[index];
	}

	int findIndex(const Key& key) const
	{
		unsigned int hash = key.getHash() & (m_valueArray.capacity() - 1);

		if (hash >= (unsigned int)m_hashTable.size())
		{
			return B3_HASH_NULL;
		}

		int index = m_hashTable[hash];
		while ((index != B3_HASH_NULL) && key.equals(m_keyArray[index]) == false)
		{
			index = m_next[index];
		}
		return index;
	}

	void clear()
	{
		m_hashTable.clear();
		m_next.clear();
		m_valueArray.clear();
		m_keyArray.clear();
	}
};

#endif  //B3_HASH_MAP_H

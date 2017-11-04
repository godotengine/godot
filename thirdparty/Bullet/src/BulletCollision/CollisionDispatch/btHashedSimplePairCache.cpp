/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/



#include "btHashedSimplePairCache.h"


#include <stdio.h>

int	gOverlappingSimplePairs = 0;
int gRemoveSimplePairs =0;
int gAddedSimplePairs =0;
int gFindSimplePairs =0;




btHashedSimplePairCache::btHashedSimplePairCache() {
	int initialAllocatedSize= 2;
	m_overlappingPairArray.reserve(initialAllocatedSize);
	growTables();
}




btHashedSimplePairCache::~btHashedSimplePairCache()
{
}






void btHashedSimplePairCache::removeAllPairs()
{
	m_overlappingPairArray.clear();
	m_hashTable.clear();
	m_next.clear();

	int initialAllocatedSize= 2;
	m_overlappingPairArray.reserve(initialAllocatedSize);
	growTables();
}



btSimplePair* btHashedSimplePairCache::findPair(int indexA, int indexB)
{
	gFindSimplePairs++;
	
	
	/*if (indexA > indexB) 
		btSwap(indexA, indexB);*/

	int hash = static_cast<int>(getHash(static_cast<unsigned int>(indexA), static_cast<unsigned int>(indexB)) & (m_overlappingPairArray.capacity()-1));

	if (hash >= m_hashTable.size())
	{
		return NULL;
	}

	int index = m_hashTable[hash];
	while (index != BT_SIMPLE_NULL_PAIR && equalsPair(m_overlappingPairArray[index], indexA, indexB) == false)
	{
		index = m_next[index];
	}

	if (index == BT_SIMPLE_NULL_PAIR)
	{
		return NULL;
	}

	btAssert(index < m_overlappingPairArray.size());

	return &m_overlappingPairArray[index];
}

//#include <stdio.h>

void	btHashedSimplePairCache::growTables()
{

	int newCapacity = m_overlappingPairArray.capacity();

	if (m_hashTable.size() < newCapacity)
	{
		//grow hashtable and next table
		int curHashtableSize = m_hashTable.size();

		m_hashTable.resize(newCapacity);
		m_next.resize(newCapacity);


		int i;

		for (i= 0; i < newCapacity; ++i)
		{
			m_hashTable[i] = BT_SIMPLE_NULL_PAIR;
		}
		for (i = 0; i < newCapacity; ++i)
		{
			m_next[i] = BT_SIMPLE_NULL_PAIR;
		}

		for(i=0;i<curHashtableSize;i++)
		{
	
			const btSimplePair& pair = m_overlappingPairArray[i];
			int indexA = pair.m_indexA;
			int indexB = pair.m_indexB;
			
			int	hashValue = static_cast<int>(getHash(static_cast<unsigned int>(indexA),static_cast<unsigned int>(indexB)) & (m_overlappingPairArray.capacity()-1));	// New hash value with new mask
			m_next[i] = m_hashTable[hashValue];
			m_hashTable[hashValue] = i;
		}


	}
}

btSimplePair* btHashedSimplePairCache::internalAddPair(int indexA, int indexB)
{

	int	hash = static_cast<int>(getHash(static_cast<unsigned int>(indexA),static_cast<unsigned int>(indexB)) & (m_overlappingPairArray.capacity()-1));	// New hash value with new mask


	btSimplePair* pair = internalFindPair(indexA, indexB, hash);
	if (pair != NULL)
	{
		return pair;
	}

	int count = m_overlappingPairArray.size();
	int oldCapacity = m_overlappingPairArray.capacity();
	void* mem = &m_overlappingPairArray.expandNonInitializing();

	int newCapacity = m_overlappingPairArray.capacity();

	if (oldCapacity < newCapacity)
	{
		growTables();
		//hash with new capacity
		hash = static_cast<int>(getHash(static_cast<unsigned int>(indexA),static_cast<unsigned int>(indexB)) & (m_overlappingPairArray.capacity()-1));
	}
	
	pair = new (mem) btSimplePair(indexA,indexB);

	pair->m_userPointer = 0;
	
	m_next[count] = m_hashTable[hash];
	m_hashTable[hash] = count;

	return pair;
}



void* btHashedSimplePairCache::removeOverlappingPair(int indexA, int indexB)
{
	gRemoveSimplePairs++;
	

	/*if (indexA > indexB) 
		btSwap(indexA, indexB);*/

	int	hash = static_cast<int>(getHash(static_cast<unsigned int>(indexA),static_cast<unsigned int>(indexB)) & (m_overlappingPairArray.capacity()-1));

	btSimplePair* pair = internalFindPair(indexA, indexB, hash);
	if (pair == NULL)
	{
		return 0;
	}

	
	void* userData = pair->m_userPointer;


	int pairIndex = int(pair - &m_overlappingPairArray[0]);
	btAssert(pairIndex < m_overlappingPairArray.size());

	// Remove the pair from the hash table.
	int index = m_hashTable[hash];
	btAssert(index != BT_SIMPLE_NULL_PAIR);

	int previous = BT_SIMPLE_NULL_PAIR;
	while (index != pairIndex)
	{
		previous = index;
		index = m_next[index];
	}

	if (previous != BT_SIMPLE_NULL_PAIR)
	{
		btAssert(m_next[previous] == pairIndex);
		m_next[previous] = m_next[pairIndex];
	}
	else
	{
		m_hashTable[hash] = m_next[pairIndex];
	}

	// We now move the last pair into spot of the
	// pair being removed. We need to fix the hash
	// table indices to support the move.

	int lastPairIndex = m_overlappingPairArray.size() - 1;

	// If the removed pair is the last pair, we are done.
	if (lastPairIndex == pairIndex)
	{
		m_overlappingPairArray.pop_back();
		return userData;
	}

	// Remove the last pair from the hash table.
	const btSimplePair* last = &m_overlappingPairArray[lastPairIndex];
		/* missing swap here too, Nat. */ 
	int lastHash = static_cast<int>(getHash(static_cast<unsigned int>(last->m_indexA), static_cast<unsigned int>(last->m_indexB)) & (m_overlappingPairArray.capacity()-1));

	index = m_hashTable[lastHash];
	btAssert(index != BT_SIMPLE_NULL_PAIR);

	previous = BT_SIMPLE_NULL_PAIR;
	while (index != lastPairIndex)
	{
		previous = index;
		index = m_next[index];
	}

	if (previous != BT_SIMPLE_NULL_PAIR)
	{
		btAssert(m_next[previous] == lastPairIndex);
		m_next[previous] = m_next[lastPairIndex];
	}
	else
	{
		m_hashTable[lastHash] = m_next[lastPairIndex];
	}

	// Copy the last pair into the remove pair's spot.
	m_overlappingPairArray[pairIndex] = m_overlappingPairArray[lastPairIndex];

	// Insert the last pair into the hash table
	m_next[pairIndex] = m_hashTable[lastHash];
	m_hashTable[lastHash] = pairIndex;

	m_overlappingPairArray.pop_back();

	return userData;
}
//#include <stdio.h>











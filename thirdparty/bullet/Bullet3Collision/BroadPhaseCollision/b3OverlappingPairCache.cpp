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



#include "b3OverlappingPairCache.h"

//#include "b3Dispatcher.h"
//#include "b3CollisionAlgorithm.h"
#include "Bullet3Geometry/b3AabbUtil.h"

#include <stdio.h>

int	b3g_overlappingPairs = 0;
int b3g_removePairs =0;
int b3g_addedPairs =0;
int b3g_findPairs =0;




b3HashedOverlappingPairCache::b3HashedOverlappingPairCache():
	m_overlapFilterCallback(0)
//,	m_blockedForChanges(false)
{
	int initialAllocatedSize= 2;
	m_overlappingPairArray.reserve(initialAllocatedSize);
	growTables();
}




b3HashedOverlappingPairCache::~b3HashedOverlappingPairCache()
{
}



void	b3HashedOverlappingPairCache::cleanOverlappingPair(b3BroadphasePair& pair,b3Dispatcher* dispatcher)
{
/*	if (pair.m_algorithm)
	{
		{
			pair.m_algorithm->~b3CollisionAlgorithm();
			dispatcher->freeCollisionAlgorithm(pair.m_algorithm);
			pair.m_algorithm=0;
		}
	}
	*/

}




void	b3HashedOverlappingPairCache::cleanProxyFromPairs(int proxy,b3Dispatcher* dispatcher)
{

	class	CleanPairCallback : public b3OverlapCallback
	{
		int m_cleanProxy;
		b3OverlappingPairCache*	m_pairCache;
		b3Dispatcher* m_dispatcher;

	public:
		CleanPairCallback(int cleanProxy,b3OverlappingPairCache* pairCache,b3Dispatcher* dispatcher)
			:m_cleanProxy(cleanProxy),
			m_pairCache(pairCache),
			m_dispatcher(dispatcher)
		{
		}
		virtual	bool	processOverlap(b3BroadphasePair& pair)
		{
			if ((pair.x == m_cleanProxy) ||
				(pair.y == m_cleanProxy))
			{
				m_pairCache->cleanOverlappingPair(pair,m_dispatcher);
			}
			return false;
		}
		
	};

	CleanPairCallback cleanPairs(proxy,this,dispatcher);

	processAllOverlappingPairs(&cleanPairs,dispatcher);

}




void	b3HashedOverlappingPairCache::removeOverlappingPairsContainingProxy(int proxy,b3Dispatcher* dispatcher)
{

	class	RemovePairCallback : public b3OverlapCallback
	{
		int m_obsoleteProxy;

	public:
		RemovePairCallback(int obsoleteProxy)
			:m_obsoleteProxy(obsoleteProxy)
		{
		}
		virtual	bool	processOverlap(b3BroadphasePair& pair)
		{
			return ((pair.x == m_obsoleteProxy) ||
				(pair.y == m_obsoleteProxy));
		}
		
	};


	RemovePairCallback removeCallback(proxy);

	processAllOverlappingPairs(&removeCallback,dispatcher);
}





b3BroadphasePair* b3HashedOverlappingPairCache::findPair(int proxy0, int proxy1)
{
	b3g_findPairs++;
	if(proxy0 >proxy1) 
		b3Swap(proxy0,proxy1);
	int proxyId1 = proxy0;
	int proxyId2 = proxy1;

	/*if (proxyId1 > proxyId2) 
		b3Swap(proxyId1, proxyId2);*/

	int hash = static_cast<int>(getHash(static_cast<unsigned int>(proxyId1), static_cast<unsigned int>(proxyId2)) & (m_overlappingPairArray.capacity()-1));

	if (hash >= m_hashTable.size())
	{
		return NULL;
	}

	int index = m_hashTable[hash];
	while (index != B3_NULL_PAIR && equalsPair(m_overlappingPairArray[index], proxyId1, proxyId2) == false)
	{
		index = m_next[index];
	}

	if (index == B3_NULL_PAIR)
	{
		return NULL;
	}

	b3Assert(index < m_overlappingPairArray.size());

	return &m_overlappingPairArray[index];
}

//#include <stdio.h>

void	b3HashedOverlappingPairCache::growTables()
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
			m_hashTable[i] = B3_NULL_PAIR;
		}
		for (i = 0; i < newCapacity; ++i)
		{
			m_next[i] = B3_NULL_PAIR;
		}

		for(i=0;i<curHashtableSize;i++)
		{
	
			const b3BroadphasePair& pair = m_overlappingPairArray[i];
			int proxyId1 = pair.x;
			int proxyId2 = pair.y;
			/*if (proxyId1 > proxyId2) 
				b3Swap(proxyId1, proxyId2);*/
			int	hashValue = static_cast<int>(getHash(static_cast<unsigned int>(proxyId1),static_cast<unsigned int>(proxyId2)) & (m_overlappingPairArray.capacity()-1));	// New hash value with new mask
			m_next[i] = m_hashTable[hashValue];
			m_hashTable[hashValue] = i;
		}


	}
}

b3BroadphasePair* b3HashedOverlappingPairCache::internalAddPair(int proxy0, int proxy1)
{
	if(proxy0>proxy1) 
		b3Swap(proxy0,proxy1);
	int proxyId1 = proxy0;
	int proxyId2 = proxy1;

	/*if (proxyId1 > proxyId2) 
		b3Swap(proxyId1, proxyId2);*/

	int	hash = static_cast<int>(getHash(static_cast<unsigned int>(proxyId1),static_cast<unsigned int>(proxyId2)) & (m_overlappingPairArray.capacity()-1));	// New hash value with new mask


	b3BroadphasePair* pair = internalFindPair(proxy0, proxy1, hash);
	if (pair != NULL)
	{
		return pair;
	}
	/*for(int i=0;i<m_overlappingPairArray.size();++i)
		{
		if(	(m_overlappingPairArray[i].m_pProxy0==proxy0)&&
			(m_overlappingPairArray[i].m_pProxy1==proxy1))
			{
			printf("Adding duplicated %u<>%u\r\n",proxyId1,proxyId2);
			internalFindPair(proxy0, proxy1, hash);
			}
		}*/
	int count = m_overlappingPairArray.size();
	int oldCapacity = m_overlappingPairArray.capacity();
	pair = &m_overlappingPairArray.expandNonInitializing();

	//this is where we add an actual pair, so also call the 'ghost'
//	if (m_ghostPairCallback)
//		m_ghostPairCallback->addOverlappingPair(proxy0,proxy1);

	int newCapacity = m_overlappingPairArray.capacity();

	if (oldCapacity < newCapacity)
	{
		growTables();
		//hash with new capacity
		hash = static_cast<int>(getHash(static_cast<unsigned int>(proxyId1),static_cast<unsigned int>(proxyId2)) & (m_overlappingPairArray.capacity()-1));
	}
	
	*pair = b3MakeBroadphasePair(proxy0,proxy1);
	
//	pair->m_pProxy0 = proxy0;
//	pair->m_pProxy1 = proxy1;
	//pair->m_algorithm = 0;
	//pair->m_internalTmpValue = 0;
	

	m_next[count] = m_hashTable[hash];
	m_hashTable[hash] = count;

	return pair;
}



void* b3HashedOverlappingPairCache::removeOverlappingPair(int proxy0, int proxy1,b3Dispatcher* dispatcher)
{
	b3g_removePairs++;
	if(proxy0>proxy1) 
		b3Swap(proxy0,proxy1);
	int proxyId1 = proxy0;
	int proxyId2 = proxy1;

	/*if (proxyId1 > proxyId2) 
		b3Swap(proxyId1, proxyId2);*/

	int	hash = static_cast<int>(getHash(static_cast<unsigned int>(proxyId1),static_cast<unsigned int>(proxyId2)) & (m_overlappingPairArray.capacity()-1));

	b3BroadphasePair* pair = internalFindPair(proxy0, proxy1, hash);
	if (pair == NULL)
	{
		return 0;
	}

	cleanOverlappingPair(*pair,dispatcher);

	

	int pairIndex = int(pair - &m_overlappingPairArray[0]);
	b3Assert(pairIndex < m_overlappingPairArray.size());

	// Remove the pair from the hash table.
	int index = m_hashTable[hash];
	b3Assert(index != B3_NULL_PAIR);

	int previous = B3_NULL_PAIR;
	while (index != pairIndex)
	{
		previous = index;
		index = m_next[index];
	}

	if (previous != B3_NULL_PAIR)
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

	int lastPairIndex = m_overlappingPairArray.size() - 1;

	//if (m_ghostPairCallback)
	//	m_ghostPairCallback->removeOverlappingPair(proxy0, proxy1,dispatcher);

	// If the removed pair is the last pair, we are done.
	if (lastPairIndex == pairIndex)
	{
		m_overlappingPairArray.pop_back();
		return 0;
	}

	// Remove the last pair from the hash table.
	const b3BroadphasePair* last = &m_overlappingPairArray[lastPairIndex];
		/* missing swap here too, Nat. */ 
	int lastHash = static_cast<int>(getHash(static_cast<unsigned int>(last->x), static_cast<unsigned int>(last->y)) & (m_overlappingPairArray.capacity()-1));

	index = m_hashTable[lastHash];
	b3Assert(index != B3_NULL_PAIR);

	previous = B3_NULL_PAIR;
	while (index != lastPairIndex)
	{
		previous = index;
		index = m_next[index];
	}

	if (previous != B3_NULL_PAIR)
	{
		b3Assert(m_next[previous] == lastPairIndex);
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

	return 0;
}
//#include <stdio.h>

void	b3HashedOverlappingPairCache::processAllOverlappingPairs(b3OverlapCallback* callback,b3Dispatcher* dispatcher)
{

	int i;

//	printf("m_overlappingPairArray.size()=%d\n",m_overlappingPairArray.size());
	for (i=0;i<m_overlappingPairArray.size();)
	{
	
		b3BroadphasePair* pair = &m_overlappingPairArray[i];
		if (callback->processOverlap(*pair))
		{
			removeOverlappingPair(pair->x,pair->y,dispatcher);

			b3g_overlappingPairs--;
		} else
		{
			i++;
		}
	}
}





void	b3HashedOverlappingPairCache::sortOverlappingPairs(b3Dispatcher* dispatcher)
{
	///need to keep hashmap in sync with pair address, so rebuild all
	b3BroadphasePairArray tmpPairs;
	int i;
	for (i=0;i<m_overlappingPairArray.size();i++)
	{
		tmpPairs.push_back(m_overlappingPairArray[i]);
	}

	for (i=0;i<tmpPairs.size();i++)
	{
		removeOverlappingPair(tmpPairs[i].x,tmpPairs[i].y,dispatcher);
	}
	
	for (i = 0; i < m_next.size(); i++)
	{
		m_next[i] = B3_NULL_PAIR;
	}

	tmpPairs.quickSort(b3BroadphasePairSortPredicate());

	for (i=0;i<tmpPairs.size();i++)
	{
		addOverlappingPair(tmpPairs[i].x ,tmpPairs[i].y);
	}

	
}


void*	b3SortedOverlappingPairCache::removeOverlappingPair(int proxy0,int proxy1, b3Dispatcher* dispatcher )
{
	if (!hasDeferredRemoval())
	{
		b3BroadphasePair findPair = b3MakeBroadphasePair(proxy0,proxy1);
		

		int findIndex = m_overlappingPairArray.findLinearSearch(findPair);
		if (findIndex < m_overlappingPairArray.size())
		{
			b3g_overlappingPairs--;
			b3BroadphasePair& pair = m_overlappingPairArray[findIndex];
			
			cleanOverlappingPair(pair,dispatcher);
			//if (m_ghostPairCallback)
			//	m_ghostPairCallback->removeOverlappingPair(proxy0, proxy1,dispatcher);
			
			m_overlappingPairArray.swap(findIndex,m_overlappingPairArray.capacity()-1);
			m_overlappingPairArray.pop_back();
			return 0;
		}
	}

	return 0;
}








b3BroadphasePair*	b3SortedOverlappingPairCache::addOverlappingPair(int proxy0,int proxy1)
{
	//don't add overlap with own
	b3Assert(proxy0 != proxy1);

	if (!needsBroadphaseCollision(proxy0,proxy1))
		return 0;
	
	b3BroadphasePair* pair = &m_overlappingPairArray.expandNonInitializing();
	*pair = b3MakeBroadphasePair(proxy0,proxy1);
	
	
	b3g_overlappingPairs++;
	b3g_addedPairs++;
	
//	if (m_ghostPairCallback)
//		m_ghostPairCallback->addOverlappingPair(proxy0, proxy1);
	return pair;
	
}

///this findPair becomes really slow. Either sort the list to speedup the query, or
///use a different solution. It is mainly used for Removing overlapping pairs. Removal could be delayed.
///we could keep a linked list in each proxy, and store pair in one of the proxies (with lowest memory address)
///Also we can use a 2D bitmap, which can be useful for a future GPU implementation
 b3BroadphasePair*	b3SortedOverlappingPairCache::findPair(int proxy0,int proxy1)
{
	if (!needsBroadphaseCollision(proxy0,proxy1))
		return 0;

	b3BroadphasePair tmpPair = b3MakeBroadphasePair(proxy0,proxy1);
	int findIndex = m_overlappingPairArray.findLinearSearch(tmpPair);

	if (findIndex < m_overlappingPairArray.size())
	{
		//b3Assert(it != m_overlappingPairSet.end());
		 b3BroadphasePair* pair = &m_overlappingPairArray[findIndex];
		return pair;
	}
	return 0;
}










//#include <stdio.h>

void	b3SortedOverlappingPairCache::processAllOverlappingPairs(b3OverlapCallback* callback,b3Dispatcher* dispatcher)
{

	int i;

	for (i=0;i<m_overlappingPairArray.size();)
	{
	
		b3BroadphasePair* pair = &m_overlappingPairArray[i];
		if (callback->processOverlap(*pair))
		{
			cleanOverlappingPair(*pair,dispatcher);
			pair->x = -1;
			pair->y = -1;
			m_overlappingPairArray.swap(i,m_overlappingPairArray.size()-1);
			m_overlappingPairArray.pop_back();
			b3g_overlappingPairs--;
		} else
		{
			i++;
		}
	}
}




b3SortedOverlappingPairCache::b3SortedOverlappingPairCache():
	m_blockedForChanges(false),
	m_hasDeferredRemoval(true),
	m_overlapFilterCallback(0)

{
	int initialAllocatedSize= 2;
	m_overlappingPairArray.reserve(initialAllocatedSize);
}

b3SortedOverlappingPairCache::~b3SortedOverlappingPairCache()
{
}

void	b3SortedOverlappingPairCache::cleanOverlappingPair(b3BroadphasePair& pair,b3Dispatcher* dispatcher)
{
/*	if (pair.m_algorithm)
	{
		{
			pair.m_algorithm->~b3CollisionAlgorithm();
			dispatcher->freeCollisionAlgorithm(pair.m_algorithm);
			pair.m_algorithm=0;
			b3g_removePairs--;
		}
	}
	*/
}


void	b3SortedOverlappingPairCache::cleanProxyFromPairs(int proxy,b3Dispatcher* dispatcher)
{

	class	CleanPairCallback : public b3OverlapCallback
	{
		int m_cleanProxy;
		b3OverlappingPairCache*	m_pairCache;
		b3Dispatcher* m_dispatcher;

	public:
		CleanPairCallback(int cleanProxy,b3OverlappingPairCache* pairCache,b3Dispatcher* dispatcher)
			:m_cleanProxy(cleanProxy),
			m_pairCache(pairCache),
			m_dispatcher(dispatcher)
		{
		}
		virtual	bool	processOverlap(b3BroadphasePair& pair)
		{
			if ((pair.x == m_cleanProxy) ||
				(pair.y == m_cleanProxy))
			{
				m_pairCache->cleanOverlappingPair(pair,m_dispatcher);
			}
			return false;
		}
		
	};

	CleanPairCallback cleanPairs(proxy,this,dispatcher);

	processAllOverlappingPairs(&cleanPairs,dispatcher);

}


void	b3SortedOverlappingPairCache::removeOverlappingPairsContainingProxy(int proxy,b3Dispatcher* dispatcher)
{

	class	RemovePairCallback : public b3OverlapCallback
	{
		int m_obsoleteProxy;

	public:
		RemovePairCallback(int obsoleteProxy)
			:m_obsoleteProxy(obsoleteProxy)
		{
		}
		virtual	bool	processOverlap(b3BroadphasePair& pair)
		{
			return ((pair.x == m_obsoleteProxy) ||
				(pair.y == m_obsoleteProxy));
		}
		
	};

	RemovePairCallback removeCallback(proxy);

	processAllOverlappingPairs(&removeCallback,dispatcher);
}

void	b3SortedOverlappingPairCache::sortOverlappingPairs(b3Dispatcher* dispatcher)
{
	//should already be sorted
}


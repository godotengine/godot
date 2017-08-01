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

#ifndef B3_OVERLAPPING_PAIR_CACHE_H
#define B3_OVERLAPPING_PAIR_CACHE_H

#include "Bullet3Common/shared/b3Int2.h"
#include "Bullet3Common/b3AlignedObjectArray.h"

class b3Dispatcher;
#include "b3OverlappingPair.h"



typedef b3AlignedObjectArray<b3BroadphasePair>	b3BroadphasePairArray;

struct	b3OverlapCallback
{
	virtual ~b3OverlapCallback()
	{}
	//return true for deletion of the pair
	virtual bool	processOverlap(b3BroadphasePair& pair) = 0;

};

struct b3OverlapFilterCallback
{
	virtual ~b3OverlapFilterCallback()
	{}
	// return true when pairs need collision
	virtual bool	needBroadphaseCollision(int proxy0,int proxy1) const = 0;
};







extern int b3g_removePairs;
extern int b3g_addedPairs;
extern int b3g_findPairs;

const int B3_NULL_PAIR=0xffffffff;

///The b3OverlappingPairCache provides an interface for overlapping pair management (add, remove, storage), used by the b3BroadphaseInterface broadphases.
///The b3HashedOverlappingPairCache and b3SortedOverlappingPairCache classes are two implementations.
class b3OverlappingPairCache 
{
public:
	virtual ~b3OverlappingPairCache() {} // this is needed so we can get to the derived class destructor

	virtual b3BroadphasePair*	getOverlappingPairArrayPtr() = 0;
	
	virtual const b3BroadphasePair*	getOverlappingPairArrayPtr() const = 0;

	virtual b3BroadphasePairArray&	getOverlappingPairArray() = 0;

	virtual	void	cleanOverlappingPair(b3BroadphasePair& pair,b3Dispatcher* dispatcher) = 0;

	virtual int getNumOverlappingPairs() const = 0;

	virtual void	cleanProxyFromPairs(int proxy,b3Dispatcher* dispatcher) = 0;

	virtual	void setOverlapFilterCallback(b3OverlapFilterCallback* callback) = 0;

	virtual void	processAllOverlappingPairs(b3OverlapCallback*,b3Dispatcher* dispatcher) = 0;

	virtual b3BroadphasePair* findPair(int proxy0, int proxy1) = 0;

	virtual bool	hasDeferredRemoval() = 0;

	//virtual	void	setInternalGhostPairCallback(b3OverlappingPairCallback* ghostPairCallback)=0;

	virtual b3BroadphasePair* 	addOverlappingPair(int proxy0,int proxy1)=0;
	virtual void*	removeOverlappingPair(int proxy0,int proxy1,b3Dispatcher* dispatcher)=0;
	virtual void	removeOverlappingPairsContainingProxy(int /*proxy0*/,b3Dispatcher* /*dispatcher*/)=0;

	virtual void	sortOverlappingPairs(b3Dispatcher* dispatcher) = 0;


};

/// Hash-space based Pair Cache, thanks to Erin Catto, Box2D, http://www.box2d.org, and Pierre Terdiman, Codercorner, http://codercorner.com
class b3HashedOverlappingPairCache : public b3OverlappingPairCache
{
	b3BroadphasePairArray	m_overlappingPairArray;
	b3OverlapFilterCallback* m_overlapFilterCallback;
//	bool		m_blockedForChanges;


public:
	b3HashedOverlappingPairCache();
	virtual ~b3HashedOverlappingPairCache();

	
	virtual void	removeOverlappingPairsContainingProxy(int proxy,b3Dispatcher* dispatcher);

	virtual void*	removeOverlappingPair(int proxy0,int proxy1,b3Dispatcher* dispatcher);
	
	B3_FORCE_INLINE bool needsBroadphaseCollision(int proxy0,int proxy1) const
	{
		if (m_overlapFilterCallback)
			return m_overlapFilterCallback->needBroadphaseCollision(proxy0,proxy1);

		bool collides = true;//(proxy0->m_collisionFilterGroup & proxy1->m_collisionFilterMask) != 0;
		//collides = collides && (proxy1->m_collisionFilterGroup & proxy0->m_collisionFilterMask);
		
		return collides;
	}

	// Add a pair and return the new pair. If the pair already exists,
	// no new pair is created and the old one is returned.
	virtual b3BroadphasePair* 	addOverlappingPair(int proxy0,int proxy1)
	{
		b3g_addedPairs++;

		if (!needsBroadphaseCollision(proxy0,proxy1))
			return 0;

		return internalAddPair(proxy0,proxy1);
	}

	

	void	cleanProxyFromPairs(int proxy,b3Dispatcher* dispatcher);

	
	virtual void	processAllOverlappingPairs(b3OverlapCallback*,b3Dispatcher* dispatcher);

	virtual b3BroadphasePair*	getOverlappingPairArrayPtr()
	{
		return &m_overlappingPairArray[0];
	}

	const b3BroadphasePair*	getOverlappingPairArrayPtr() const
	{
		return &m_overlappingPairArray[0];
	}

	b3BroadphasePairArray&	getOverlappingPairArray()
	{
		return m_overlappingPairArray;
	}

	const b3BroadphasePairArray&	getOverlappingPairArray() const
	{
		return m_overlappingPairArray;
	}

	void	cleanOverlappingPair(b3BroadphasePair& pair,b3Dispatcher* dispatcher);



	b3BroadphasePair* findPair(int proxy0, int proxy1);

	int GetCount() const { return m_overlappingPairArray.size(); }
//	b3BroadphasePair* GetPairs() { return m_pairs; }

	b3OverlapFilterCallback* getOverlapFilterCallback()
	{
		return m_overlapFilterCallback;
	}

	void setOverlapFilterCallback(b3OverlapFilterCallback* callback)
	{
		m_overlapFilterCallback = callback;
	}

	int	getNumOverlappingPairs() const
	{
		return m_overlappingPairArray.size();
	}
private:
	
	b3BroadphasePair* 	internalAddPair(int proxy0,int proxy1);

	void	growTables();

	B3_FORCE_INLINE bool equalsPair(const b3BroadphasePair& pair, int proxyId1, int proxyId2)
	{	
		return pair.x == proxyId1 && pair.y  == proxyId2;
	}

	/*
	// Thomas Wang's hash, see: http://www.concentric.net/~Ttwang/tech/inthash.htm
	// This assumes proxyId1 and proxyId2 are 16-bit.
	B3_FORCE_INLINE int getHash(int proxyId1, int proxyId2)
	{
		int key = (proxyId2 << 16) | proxyId1;
		key = ~key + (key << 15);
		key = key ^ (key >> 12);
		key = key + (key << 2);
		key = key ^ (key >> 4);
		key = key * 2057;
		key = key ^ (key >> 16);
		return key;
	}
	*/


	
	B3_FORCE_INLINE	unsigned int getHash(unsigned int proxyId1, unsigned int proxyId2)
	{
		int key = static_cast<int>(((unsigned int)proxyId1) | (((unsigned int)proxyId2) <<16));
		// Thomas Wang's hash

		key += ~(key << 15);
		key ^=  (key >> 10);
		key +=  (key << 3);
		key ^=  (key >> 6);
		key += ~(key << 11);
		key ^=  (key >> 16);
		return static_cast<unsigned int>(key);
	}
	




	B3_FORCE_INLINE b3BroadphasePair* internalFindPair(int proxy0, int proxy1, int hash)
	{
		int proxyId1 = proxy0;
		int proxyId2 = proxy1;
		#if 0 // wrong, 'equalsPair' use unsorted uids, copy-past devil striked again. Nat.
		if (proxyId1 > proxyId2) 
			b3Swap(proxyId1, proxyId2);
		#endif

		int index = m_hashTable[hash];
		
		while( index != B3_NULL_PAIR && equalsPair(m_overlappingPairArray[index], proxyId1, proxyId2) == false)
		{
			index = m_next[index];
		}

		if ( index == B3_NULL_PAIR )
		{
			return NULL;
		}

		b3Assert(index < m_overlappingPairArray.size());

		return &m_overlappingPairArray[index];
	}

	virtual bool	hasDeferredRemoval()
	{
		return false;
	}

/*	virtual	void	setInternalGhostPairCallback(b3OverlappingPairCallback* ghostPairCallback)
	{
		m_ghostPairCallback = ghostPairCallback;
	}
	*/

	virtual void	sortOverlappingPairs(b3Dispatcher* dispatcher);
	

protected:
	
	b3AlignedObjectArray<int>	m_hashTable;
	b3AlignedObjectArray<int>	m_next;
//	b3OverlappingPairCallback*	m_ghostPairCallback;
	
};




///b3SortedOverlappingPairCache maintains the objects with overlapping AABB
///Typically managed by the Broadphase, Axis3Sweep or b3SimpleBroadphase
class	b3SortedOverlappingPairCache : public b3OverlappingPairCache
{
	protected:
		//avoid brute-force finding all the time
		b3BroadphasePairArray	m_overlappingPairArray;

		//during the dispatch, check that user doesn't destroy/create proxy
		bool		m_blockedForChanges;

		///by default, do the removal during the pair traversal
		bool		m_hasDeferredRemoval;
		
		//if set, use the callback instead of the built in filter in needBroadphaseCollision
		b3OverlapFilterCallback* m_overlapFilterCallback;

//		b3OverlappingPairCallback*	m_ghostPairCallback;

	public:
			
		b3SortedOverlappingPairCache();	
		virtual ~b3SortedOverlappingPairCache();

		virtual void	processAllOverlappingPairs(b3OverlapCallback*,b3Dispatcher* dispatcher);

		void*	removeOverlappingPair(int proxy0,int proxy1,b3Dispatcher* dispatcher);

		void	cleanOverlappingPair(b3BroadphasePair& pair,b3Dispatcher* dispatcher);
		
		b3BroadphasePair*	addOverlappingPair(int proxy0,int proxy1);

		b3BroadphasePair*	findPair(int proxy0,int proxy1);
			
		
		void	cleanProxyFromPairs(int proxy,b3Dispatcher* dispatcher);

		virtual void	removeOverlappingPairsContainingProxy(int proxy,b3Dispatcher* dispatcher);


		inline bool needsBroadphaseCollision(int proxy0,int proxy1) const
		{
			if (m_overlapFilterCallback)
				return m_overlapFilterCallback->needBroadphaseCollision(proxy0,proxy1);

			bool collides = true;//(proxy0->m_collisionFilterGroup & proxy1->m_collisionFilterMask) != 0;
			//collides = collides && (proxy1->m_collisionFilterGroup & proxy0->m_collisionFilterMask);
			
			return collides;
		}
		
		b3BroadphasePairArray&	getOverlappingPairArray()
		{
			return m_overlappingPairArray;
		}

		const b3BroadphasePairArray&	getOverlappingPairArray() const
		{
			return m_overlappingPairArray;
		}

		


		b3BroadphasePair*	getOverlappingPairArrayPtr()
		{
			return &m_overlappingPairArray[0];
		}

		const b3BroadphasePair*	getOverlappingPairArrayPtr() const
		{
			return &m_overlappingPairArray[0];
		}

		int	getNumOverlappingPairs() const
		{
			return m_overlappingPairArray.size();
		}
		
		b3OverlapFilterCallback* getOverlapFilterCallback()
		{
			return m_overlapFilterCallback;
		}

		void setOverlapFilterCallback(b3OverlapFilterCallback* callback)
		{
			m_overlapFilterCallback = callback;
		}

		virtual bool	hasDeferredRemoval()
		{
			return m_hasDeferredRemoval;
		}

/*		virtual	void	setInternalGhostPairCallback(b3OverlappingPairCallback* ghostPairCallback)
		{
			m_ghostPairCallback = ghostPairCallback;
		}
		*/
		virtual void	sortOverlappingPairs(b3Dispatcher* dispatcher);
		

};



///b3NullPairCache skips add/removal of overlapping pairs. Userful for benchmarking and unit testing.
class b3NullPairCache : public b3OverlappingPairCache
{

	b3BroadphasePairArray	m_overlappingPairArray;

public:

	virtual b3BroadphasePair*	getOverlappingPairArrayPtr()
	{
		return &m_overlappingPairArray[0];
	}
	const b3BroadphasePair*	getOverlappingPairArrayPtr() const
	{
		return &m_overlappingPairArray[0];
	}
	b3BroadphasePairArray&	getOverlappingPairArray()
	{
		return m_overlappingPairArray;
	}
	
	virtual	void	cleanOverlappingPair(b3BroadphasePair& /*pair*/,b3Dispatcher* /*dispatcher*/)
	{

	}

	virtual int getNumOverlappingPairs() const
	{
		return 0;
	}

	virtual void	cleanProxyFromPairs(int /*proxy*/,b3Dispatcher* /*dispatcher*/)
	{

	}

	virtual	void setOverlapFilterCallback(b3OverlapFilterCallback* /*callback*/)
	{
	}

	virtual void	processAllOverlappingPairs(b3OverlapCallback*,b3Dispatcher* /*dispatcher*/)
	{
	}

	virtual b3BroadphasePair* findPair(int /*proxy0*/, int /*proxy1*/)
	{
		return 0;
	}

	virtual bool	hasDeferredRemoval()
	{
		return true;
	}

//	virtual	void	setInternalGhostPairCallback(b3OverlappingPairCallback* /* ghostPairCallback */)
//	{
//
//	}

	virtual b3BroadphasePair*	addOverlappingPair(int /*proxy0*/,int /*proxy1*/)
	{
		return 0;
	}

	virtual void*	removeOverlappingPair(int /*proxy0*/,int /*proxy1*/,b3Dispatcher* /*dispatcher*/)
	{
		return 0;
	}

	virtual void	removeOverlappingPairsContainingProxy(int /*proxy0*/,b3Dispatcher* /*dispatcher*/)
	{
	}
	
	virtual void	sortOverlappingPairs(b3Dispatcher* dispatcher)
	{
        (void) dispatcher;
	}


};


#endif //B3_OVERLAPPING_PAIR_CACHE_H



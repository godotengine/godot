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

#ifndef BT_OVERLAPPING_PAIR_CACHE_H
#define BT_OVERLAPPING_PAIR_CACHE_H


#include "btBroadphaseInterface.h"
#include "btBroadphaseProxy.h"
#include "btOverlappingPairCallback.h"

#include "LinearMath/btAlignedObjectArray.h"
class btDispatcher;

typedef btAlignedObjectArray<btBroadphasePair>	btBroadphasePairArray;

struct	btOverlapCallback
{
	virtual ~btOverlapCallback()
	{}
	//return true for deletion of the pair
	virtual bool	processOverlap(btBroadphasePair& pair) = 0;

};

struct btOverlapFilterCallback
{
	virtual ~btOverlapFilterCallback()
	{}
	// return true when pairs need collision
	virtual bool	needBroadphaseCollision(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1) const = 0;
};







const int BT_NULL_PAIR=0xffffffff;

///The btOverlappingPairCache provides an interface for overlapping pair management (add, remove, storage), used by the btBroadphaseInterface broadphases.
///The btHashedOverlappingPairCache and btSortedOverlappingPairCache classes are two implementations.
class btOverlappingPairCache : public btOverlappingPairCallback
{
public:
	virtual ~btOverlappingPairCache() {} // this is needed so we can get to the derived class destructor

	virtual btBroadphasePair*	getOverlappingPairArrayPtr() = 0;
	
	virtual const btBroadphasePair*	getOverlappingPairArrayPtr() const = 0;

	virtual btBroadphasePairArray&	getOverlappingPairArray() = 0;

	virtual	void	cleanOverlappingPair(btBroadphasePair& pair,btDispatcher* dispatcher) = 0;

	virtual int getNumOverlappingPairs() const = 0;

	virtual void	cleanProxyFromPairs(btBroadphaseProxy* proxy,btDispatcher* dispatcher) = 0;

	virtual	void setOverlapFilterCallback(btOverlapFilterCallback* callback) = 0;

	virtual void	processAllOverlappingPairs(btOverlapCallback*,btDispatcher* dispatcher) = 0;

	virtual void    processAllOverlappingPairs(btOverlapCallback* callback,btDispatcher* dispatcher, const struct btDispatcherInfo& dispatchInfo)
	{
		processAllOverlappingPairs(callback, dispatcher);
	}
	virtual btBroadphasePair* findPair(btBroadphaseProxy* proxy0, btBroadphaseProxy* proxy1) = 0;

	virtual bool	hasDeferredRemoval() = 0;

	virtual	void	setInternalGhostPairCallback(btOverlappingPairCallback* ghostPairCallback)=0;

	virtual void	sortOverlappingPairs(btDispatcher* dispatcher) = 0;


};

/// Hash-space based Pair Cache, thanks to Erin Catto, Box2D, http://www.box2d.org, and Pierre Terdiman, Codercorner, http://codercorner.com

ATTRIBUTE_ALIGNED16(class) btHashedOverlappingPairCache : public btOverlappingPairCache
{
	btBroadphasePairArray	m_overlappingPairArray;
	btOverlapFilterCallback* m_overlapFilterCallback;

protected:
	
	btAlignedObjectArray<int>	m_hashTable;
	btAlignedObjectArray<int>	m_next;
	btOverlappingPairCallback*	m_ghostPairCallback;


public:
	BT_DECLARE_ALIGNED_ALLOCATOR();
	
	btHashedOverlappingPairCache();
	virtual ~btHashedOverlappingPairCache();

	
	void	removeOverlappingPairsContainingProxy(btBroadphaseProxy* proxy,btDispatcher* dispatcher);

	virtual void*	removeOverlappingPair(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1,btDispatcher* dispatcher);
	
	SIMD_FORCE_INLINE bool needsBroadphaseCollision(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1) const
	{
		if (m_overlapFilterCallback)
			return m_overlapFilterCallback->needBroadphaseCollision(proxy0,proxy1);

		bool collides = (proxy0->m_collisionFilterGroup & proxy1->m_collisionFilterMask) != 0;
		collides = collides && (proxy1->m_collisionFilterGroup & proxy0->m_collisionFilterMask);
		
		return collides;
	}

	// Add a pair and return the new pair. If the pair already exists,
	// no new pair is created and the old one is returned.
	virtual btBroadphasePair* 	addOverlappingPair(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1)
	{
		if (!needsBroadphaseCollision(proxy0,proxy1))
			return 0;

		return internalAddPair(proxy0,proxy1);
	}

	

	void	cleanProxyFromPairs(btBroadphaseProxy* proxy,btDispatcher* dispatcher);

	
	virtual void	processAllOverlappingPairs(btOverlapCallback*,btDispatcher* dispatcher);

    virtual void    processAllOverlappingPairs(btOverlapCallback* callback,btDispatcher* dispatcher, const struct btDispatcherInfo& dispatchInfo);

	virtual btBroadphasePair*	getOverlappingPairArrayPtr()
	{
		return &m_overlappingPairArray[0];
	}

	const btBroadphasePair*	getOverlappingPairArrayPtr() const
	{
		return &m_overlappingPairArray[0];
	}

	btBroadphasePairArray&	getOverlappingPairArray()
	{
		return m_overlappingPairArray;
	}

	const btBroadphasePairArray&	getOverlappingPairArray() const
	{
		return m_overlappingPairArray;
	}

	void	cleanOverlappingPair(btBroadphasePair& pair,btDispatcher* dispatcher);



	btBroadphasePair* findPair(btBroadphaseProxy* proxy0, btBroadphaseProxy* proxy1);

	int GetCount() const { return m_overlappingPairArray.size(); }
//	btBroadphasePair* GetPairs() { return m_pairs; }

	btOverlapFilterCallback* getOverlapFilterCallback()
	{
		return m_overlapFilterCallback;
	}

	void setOverlapFilterCallback(btOverlapFilterCallback* callback)
	{
		m_overlapFilterCallback = callback;
	}

	int	getNumOverlappingPairs() const
	{
		return m_overlappingPairArray.size();
	}
private:
	
	btBroadphasePair* 	internalAddPair(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1);

	void	growTables();

	SIMD_FORCE_INLINE bool equalsPair(const btBroadphasePair& pair, int proxyId1, int proxyId2)
	{	
		return pair.m_pProxy0->getUid() == proxyId1 && pair.m_pProxy1->getUid() == proxyId2;
	}

	/*
	// Thomas Wang's hash, see: http://www.concentric.net/~Ttwang/tech/inthash.htm
	// This assumes proxyId1 and proxyId2 are 16-bit.
	SIMD_FORCE_INLINE int getHash(int proxyId1, int proxyId2)
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


	SIMD_FORCE_INLINE unsigned int getHash(unsigned int proxyId1, unsigned int proxyId2)
	{
		unsigned int key = proxyId1 | (proxyId2 << 16);
		// Thomas Wang's hash

		key += ~(key << 15);
		key ^=  (key >> 10);
		key +=  (key << 3);
		key ^=  (key >> 6);
		key += ~(key << 11);
		key ^=  (key >> 16);
		return key;
	}
	


	SIMD_FORCE_INLINE btBroadphasePair* internalFindPair(btBroadphaseProxy* proxy0, btBroadphaseProxy* proxy1, int hash)
	{
		int proxyId1 = proxy0->getUid();
		int proxyId2 = proxy1->getUid();
		#if 0 // wrong, 'equalsPair' use unsorted uids, copy-past devil striked again. Nat.
		if (proxyId1 > proxyId2) 
			btSwap(proxyId1, proxyId2);
		#endif

		int index = m_hashTable[hash];
		
		while( index != BT_NULL_PAIR && equalsPair(m_overlappingPairArray[index], proxyId1, proxyId2) == false)
		{
			index = m_next[index];
		}

		if ( index == BT_NULL_PAIR )
		{
			return NULL;
		}

		btAssert(index < m_overlappingPairArray.size());

		return &m_overlappingPairArray[index];
	}

	virtual bool	hasDeferredRemoval()
	{
		return false;
	}

	virtual	void	setInternalGhostPairCallback(btOverlappingPairCallback* ghostPairCallback)
	{
		m_ghostPairCallback = ghostPairCallback;
	}

	virtual void	sortOverlappingPairs(btDispatcher* dispatcher);
	

	
};




///btSortedOverlappingPairCache maintains the objects with overlapping AABB
///Typically managed by the Broadphase, Axis3Sweep or btSimpleBroadphase
class	btSortedOverlappingPairCache : public btOverlappingPairCache
{
	protected:
		//avoid brute-force finding all the time
		btBroadphasePairArray	m_overlappingPairArray;

		//during the dispatch, check that user doesn't destroy/create proxy
		bool		m_blockedForChanges;

		///by default, do the removal during the pair traversal
		bool		m_hasDeferredRemoval;
		
		//if set, use the callback instead of the built in filter in needBroadphaseCollision
		btOverlapFilterCallback* m_overlapFilterCallback;

		btOverlappingPairCallback*	m_ghostPairCallback;

	public:
			
		btSortedOverlappingPairCache();	
		virtual ~btSortedOverlappingPairCache();

		virtual void	processAllOverlappingPairs(btOverlapCallback*,btDispatcher* dispatcher);

		void*	removeOverlappingPair(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1,btDispatcher* dispatcher);

		void	cleanOverlappingPair(btBroadphasePair& pair,btDispatcher* dispatcher);
		
		btBroadphasePair*	addOverlappingPair(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1);

		btBroadphasePair*	findPair(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1);
			
		
		void	cleanProxyFromPairs(btBroadphaseProxy* proxy,btDispatcher* dispatcher);

		void	removeOverlappingPairsContainingProxy(btBroadphaseProxy* proxy,btDispatcher* dispatcher);


		inline bool needsBroadphaseCollision(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1) const
		{
			if (m_overlapFilterCallback)
				return m_overlapFilterCallback->needBroadphaseCollision(proxy0,proxy1);

			bool collides = (proxy0->m_collisionFilterGroup & proxy1->m_collisionFilterMask) != 0;
			collides = collides && (proxy1->m_collisionFilterGroup & proxy0->m_collisionFilterMask);
			
			return collides;
		}
		
		btBroadphasePairArray&	getOverlappingPairArray()
		{
			return m_overlappingPairArray;
		}

		const btBroadphasePairArray&	getOverlappingPairArray() const
		{
			return m_overlappingPairArray;
		}

		


		btBroadphasePair*	getOverlappingPairArrayPtr()
		{
			return &m_overlappingPairArray[0];
		}

		const btBroadphasePair*	getOverlappingPairArrayPtr() const
		{
			return &m_overlappingPairArray[0];
		}

		int	getNumOverlappingPairs() const
		{
			return m_overlappingPairArray.size();
		}
		
		btOverlapFilterCallback* getOverlapFilterCallback()
		{
			return m_overlapFilterCallback;
		}

		void setOverlapFilterCallback(btOverlapFilterCallback* callback)
		{
			m_overlapFilterCallback = callback;
		}

		virtual bool	hasDeferredRemoval()
		{
			return m_hasDeferredRemoval;
		}

		virtual	void	setInternalGhostPairCallback(btOverlappingPairCallback* ghostPairCallback)
		{
			m_ghostPairCallback = ghostPairCallback;
		}

		virtual void	sortOverlappingPairs(btDispatcher* dispatcher);
		

};



///btNullPairCache skips add/removal of overlapping pairs. Userful for benchmarking and unit testing.
class btNullPairCache : public btOverlappingPairCache
{

	btBroadphasePairArray	m_overlappingPairArray;

public:

	virtual btBroadphasePair*	getOverlappingPairArrayPtr()
	{
		return &m_overlappingPairArray[0];
	}
	const btBroadphasePair*	getOverlappingPairArrayPtr() const
	{
		return &m_overlappingPairArray[0];
	}
	btBroadphasePairArray&	getOverlappingPairArray()
	{
		return m_overlappingPairArray;
	}
	
	virtual	void	cleanOverlappingPair(btBroadphasePair& /*pair*/,btDispatcher* /*dispatcher*/)
	{

	}

	virtual int getNumOverlappingPairs() const
	{
		return 0;
	}

	virtual void	cleanProxyFromPairs(btBroadphaseProxy* /*proxy*/,btDispatcher* /*dispatcher*/)
	{

	}

	virtual	void setOverlapFilterCallback(btOverlapFilterCallback* /*callback*/)
	{
	}

	virtual void	processAllOverlappingPairs(btOverlapCallback*,btDispatcher* /*dispatcher*/)
	{
	}

	virtual btBroadphasePair* findPair(btBroadphaseProxy* /*proxy0*/, btBroadphaseProxy* /*proxy1*/)
	{
		return 0;
	}

	virtual bool	hasDeferredRemoval()
	{
		return true;
	}

	virtual	void	setInternalGhostPairCallback(btOverlappingPairCallback* /* ghostPairCallback */)
	{

	}

	virtual btBroadphasePair*	addOverlappingPair(btBroadphaseProxy* /*proxy0*/,btBroadphaseProxy* /*proxy1*/)
	{
		return 0;
	}

	virtual void*	removeOverlappingPair(btBroadphaseProxy* /*proxy0*/,btBroadphaseProxy* /*proxy1*/,btDispatcher* /*dispatcher*/)
	{
		return 0;
	}

	virtual void	removeOverlappingPairsContainingProxy(btBroadphaseProxy* /*proxy0*/,btDispatcher* /*dispatcher*/)
	{
	}
	
	virtual void	sortOverlappingPairs(btDispatcher* dispatcher)
	{
        (void) dispatcher;
	}


};


#endif //BT_OVERLAPPING_PAIR_CACHE_H



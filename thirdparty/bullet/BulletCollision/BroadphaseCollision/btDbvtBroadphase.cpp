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

///btDbvtBroadphase implementation by Nathanael Presson

#include "btDbvtBroadphase.h"
#include "LinearMath/btThreads.h"
btScalar gDbvtMargin = btScalar(0.05);
//
// Profiling
//

#if DBVT_BP_PROFILE || DBVT_BP_ENABLE_BENCHMARK
#include <stdio.h>
#endif

#if DBVT_BP_PROFILE
struct ProfileScope
{
	__forceinline ProfileScope(btClock& clock, unsigned long& value) : m_clock(&clock), m_value(&value), m_base(clock.getTimeMicroseconds())
	{
	}
	__forceinline ~ProfileScope()
	{
		(*m_value) += m_clock->getTimeMicroseconds() - m_base;
	}
	btClock* m_clock;
	unsigned long* m_value;
	unsigned long m_base;
};
#define SPC(_value_) ProfileScope spc_scope(m_clock, _value_)
#else
#define SPC(_value_)
#endif

//
// Helpers
//

//
template <typename T>
static inline void listappend(T* item, T*& list)
{
	item->links[0] = 0;
	item->links[1] = list;
	if (list) list->links[0] = item;
	list = item;
}

//
template <typename T>
static inline void listremove(T* item, T*& list)
{
	if (item->links[0])
		item->links[0]->links[1] = item->links[1];
	else
		list = item->links[1];
	if (item->links[1]) item->links[1]->links[0] = item->links[0];
}

//
template <typename T>
static inline int listcount(T* root)
{
	int n = 0;
	while (root)
	{
		++n;
		root = root->links[1];
	}
	return (n);
}

//
template <typename T>
static inline void clear(T& value)
{
	static const struct ZeroDummy : T
	{
	} zerodummy;
	value = zerodummy;
}

//
// Colliders
//

/* Tree collider	*/
struct btDbvtTreeCollider : btDbvt::ICollide
{
	btDbvtBroadphase* pbp;
	btDbvtProxy* proxy;
	btDbvtTreeCollider(btDbvtBroadphase* p) : pbp(p) {}
	void Process(const btDbvtNode* na, const btDbvtNode* nb)
	{
		if (na != nb)
		{
			btDbvtProxy* pa = (btDbvtProxy*)na->data;
			btDbvtProxy* pb = (btDbvtProxy*)nb->data;
#if DBVT_BP_SORTPAIRS
			if (pa->m_uniqueId > pb->m_uniqueId)
				btSwap(pa, pb);
#endif
			pbp->m_paircache->addOverlappingPair(pa, pb);
			++pbp->m_newpairs;
		}
	}
	void Process(const btDbvtNode* n)
	{
		Process(n, proxy->leaf);
	}
};

//
// btDbvtBroadphase
//

//
btDbvtBroadphase::btDbvtBroadphase(btOverlappingPairCache* paircache)
{
	m_deferedcollide = false;
	m_needcleanup = true;
	m_releasepaircache = (paircache != 0) ? false : true;
	m_prediction = 0;
	m_stageCurrent = 0;
	m_fixedleft = 0;
	m_fupdates = 1;
	m_dupdates = 0;
	m_cupdates = 10;
	m_newpairs = 1;
	m_updates_call = 0;
	m_updates_done = 0;
	m_updates_ratio = 0;
	m_paircache = paircache ? paircache : new (btAlignedAlloc(sizeof(btHashedOverlappingPairCache), 16)) btHashedOverlappingPairCache();
	m_gid = 0;
	m_pid = 0;
	m_cid = 0;
	for (int i = 0; i <= STAGECOUNT; ++i)
	{
		m_stageRoots[i] = 0;
	}
#if BT_THREADSAFE
	m_rayTestStacks.resize(BT_MAX_THREAD_COUNT);
#else
	m_rayTestStacks.resize(1);
#endif
#if DBVT_BP_PROFILE
	clear(m_profiling);
#endif
}

//
btDbvtBroadphase::~btDbvtBroadphase()
{
	if (m_releasepaircache)
	{
		m_paircache->~btOverlappingPairCache();
		btAlignedFree(m_paircache);
	}
}

//
btBroadphaseProxy* btDbvtBroadphase::createProxy(const btVector3& aabbMin,
												 const btVector3& aabbMax,
												 int /*shapeType*/,
												 void* userPtr,
												 int collisionFilterGroup,
												 int collisionFilterMask,
												 btDispatcher* /*dispatcher*/)
{
	btDbvtProxy* proxy = new (btAlignedAlloc(sizeof(btDbvtProxy), 16)) btDbvtProxy(aabbMin, aabbMax, userPtr,
																				   collisionFilterGroup,
																				   collisionFilterMask);

	btDbvtAabbMm aabb = btDbvtVolume::FromMM(aabbMin, aabbMax);

	//bproxy->aabb			=	btDbvtVolume::FromMM(aabbMin,aabbMax);
	proxy->stage = m_stageCurrent;
	proxy->m_uniqueId = ++m_gid;
	proxy->leaf = m_sets[0].insert(aabb, proxy);
	listappend(proxy, m_stageRoots[m_stageCurrent]);
	if (!m_deferedcollide)
	{
		btDbvtTreeCollider collider(this);
		collider.proxy = proxy;
		m_sets[0].collideTV(m_sets[0].m_root, aabb, collider);
		m_sets[1].collideTV(m_sets[1].m_root, aabb, collider);
	}
	return (proxy);
}

//
void btDbvtBroadphase::destroyProxy(btBroadphaseProxy* absproxy,
									btDispatcher* dispatcher)
{
	btDbvtProxy* proxy = (btDbvtProxy*)absproxy;
	if (proxy->stage == STAGECOUNT)
		m_sets[1].remove(proxy->leaf);
	else
		m_sets[0].remove(proxy->leaf);
	listremove(proxy, m_stageRoots[proxy->stage]);
	m_paircache->removeOverlappingPairsContainingProxy(proxy, dispatcher);
	btAlignedFree(proxy);
	m_needcleanup = true;
}

void btDbvtBroadphase::getAabb(btBroadphaseProxy* absproxy, btVector3& aabbMin, btVector3& aabbMax) const
{
	btDbvtProxy* proxy = (btDbvtProxy*)absproxy;
	aabbMin = proxy->m_aabbMin;
	aabbMax = proxy->m_aabbMax;
}

struct BroadphaseRayTester : btDbvt::ICollide
{
	btBroadphaseRayCallback& m_rayCallback;
	BroadphaseRayTester(btBroadphaseRayCallback& orgCallback)
		: m_rayCallback(orgCallback)
	{
	}
	void Process(const btDbvtNode* leaf)
	{
		btDbvtProxy* proxy = (btDbvtProxy*)leaf->data;
		m_rayCallback.process(proxy);
	}
};

void btDbvtBroadphase::rayTest(const btVector3& rayFrom, const btVector3& rayTo, btBroadphaseRayCallback& rayCallback, const btVector3& aabbMin, const btVector3& aabbMax)
{
	BroadphaseRayTester callback(rayCallback);
	btAlignedObjectArray<const btDbvtNode*>* stack = &m_rayTestStacks[0];
#if BT_THREADSAFE
	// for this function to be threadsafe, each thread must have a separate copy
	// of this stack.  This could be thread-local static to avoid dynamic allocations,
	// instead of just a local.
	int threadIndex = btGetCurrentThreadIndex();
	btAlignedObjectArray<const btDbvtNode*> localStack;
	//todo(erwincoumans, "why do we get tsan issue here?")
	if (0)//threadIndex < m_rayTestStacks.size())
	//if (threadIndex < m_rayTestStacks.size())
	{
		// use per-thread preallocated stack if possible to avoid dynamic allocations
		stack = &m_rayTestStacks[threadIndex];
	}
	else
	{
		stack = &localStack;
	}
#endif

	m_sets[0].rayTestInternal(m_sets[0].m_root,
							  rayFrom,
							  rayTo,
							  rayCallback.m_rayDirectionInverse,
							  rayCallback.m_signs,
							  rayCallback.m_lambda_max,
							  aabbMin,
							  aabbMax,
							  *stack,
							  callback);

	m_sets[1].rayTestInternal(m_sets[1].m_root,
							  rayFrom,
							  rayTo,
							  rayCallback.m_rayDirectionInverse,
							  rayCallback.m_signs,
							  rayCallback.m_lambda_max,
							  aabbMin,
							  aabbMax,
							  *stack,
							  callback);
}

struct BroadphaseAabbTester : btDbvt::ICollide
{
	btBroadphaseAabbCallback& m_aabbCallback;
	BroadphaseAabbTester(btBroadphaseAabbCallback& orgCallback)
		: m_aabbCallback(orgCallback)
	{
	}
	void Process(const btDbvtNode* leaf)
	{
		btDbvtProxy* proxy = (btDbvtProxy*)leaf->data;
		m_aabbCallback.process(proxy);
	}
};

void btDbvtBroadphase::aabbTest(const btVector3& aabbMin, const btVector3& aabbMax, btBroadphaseAabbCallback& aabbCallback)
{
	BroadphaseAabbTester callback(aabbCallback);

	const ATTRIBUTE_ALIGNED16(btDbvtVolume) bounds = btDbvtVolume::FromMM(aabbMin, aabbMax);
	//process all children, that overlap with  the given AABB bounds
	m_sets[0].collideTV(m_sets[0].m_root, bounds, callback);
	m_sets[1].collideTV(m_sets[1].m_root, bounds, callback);
}

//
void btDbvtBroadphase::setAabb(btBroadphaseProxy* absproxy,
							   const btVector3& aabbMin,
							   const btVector3& aabbMax,
							   btDispatcher* /*dispatcher*/)
{
	btDbvtProxy* proxy = (btDbvtProxy*)absproxy;
	ATTRIBUTE_ALIGNED16(btDbvtVolume)
	aabb = btDbvtVolume::FromMM(aabbMin, aabbMax);
#if DBVT_BP_PREVENTFALSEUPDATE
	if (NotEqual(aabb, proxy->leaf->volume))
#endif
	{
		bool docollide = false;
		if (proxy->stage == STAGECOUNT)
		{ /* fixed -> dynamic set	*/
			m_sets[1].remove(proxy->leaf);
			proxy->leaf = m_sets[0].insert(aabb, proxy);
			docollide = true;
		}
		else
		{ /* dynamic set				*/
			++m_updates_call;
			if (Intersect(proxy->leaf->volume, aabb))
			{ /* Moving				*/

				const btVector3 delta = aabbMin - proxy->m_aabbMin;
				btVector3 velocity(((proxy->m_aabbMax - proxy->m_aabbMin) / 2) * m_prediction);
				if (delta[0] < 0) velocity[0] = -velocity[0];
				if (delta[1] < 0) velocity[1] = -velocity[1];
				if (delta[2] < 0) velocity[2] = -velocity[2];
				if (
					m_sets[0].update(proxy->leaf, aabb, velocity, gDbvtMargin)

				)
				{
					++m_updates_done;
					docollide = true;
				}
			}
			else
			{ /* Teleporting			*/
				m_sets[0].update(proxy->leaf, aabb);
				++m_updates_done;
				docollide = true;
			}
		}
		listremove(proxy, m_stageRoots[proxy->stage]);
		proxy->m_aabbMin = aabbMin;
		proxy->m_aabbMax = aabbMax;
		proxy->stage = m_stageCurrent;
		listappend(proxy, m_stageRoots[m_stageCurrent]);
		if (docollide)
		{
			m_needcleanup = true;
			if (!m_deferedcollide)
			{
				btDbvtTreeCollider collider(this);
				m_sets[1].collideTTpersistentStack(m_sets[1].m_root, proxy->leaf, collider);
				m_sets[0].collideTTpersistentStack(m_sets[0].m_root, proxy->leaf, collider);
			}
		}
	}
}

//
void btDbvtBroadphase::setAabbForceUpdate(btBroadphaseProxy* absproxy,
										  const btVector3& aabbMin,
										  const btVector3& aabbMax,
										  btDispatcher* /*dispatcher*/)
{
	btDbvtProxy* proxy = (btDbvtProxy*)absproxy;
	ATTRIBUTE_ALIGNED16(btDbvtVolume)
	aabb = btDbvtVolume::FromMM(aabbMin, aabbMax);
	bool docollide = false;
	if (proxy->stage == STAGECOUNT)
	{ /* fixed -> dynamic set	*/
		m_sets[1].remove(proxy->leaf);
		proxy->leaf = m_sets[0].insert(aabb, proxy);
		docollide = true;
	}
	else
	{ /* dynamic set				*/
		++m_updates_call;
		/* Teleporting			*/
		m_sets[0].update(proxy->leaf, aabb);
		++m_updates_done;
		docollide = true;
	}
	listremove(proxy, m_stageRoots[proxy->stage]);
	proxy->m_aabbMin = aabbMin;
	proxy->m_aabbMax = aabbMax;
	proxy->stage = m_stageCurrent;
	listappend(proxy, m_stageRoots[m_stageCurrent]);
	if (docollide)
	{
		m_needcleanup = true;
		if (!m_deferedcollide)
		{
			btDbvtTreeCollider collider(this);
			m_sets[1].collideTTpersistentStack(m_sets[1].m_root, proxy->leaf, collider);
			m_sets[0].collideTTpersistentStack(m_sets[0].m_root, proxy->leaf, collider);
		}
	}
}

//
void btDbvtBroadphase::calculateOverlappingPairs(btDispatcher* dispatcher)
{
	collide(dispatcher);
#if DBVT_BP_PROFILE
	if (0 == (m_pid % DBVT_BP_PROFILING_RATE))
	{
		printf("fixed(%u) dynamics(%u) pairs(%u)\r\n", m_sets[1].m_leaves, m_sets[0].m_leaves, m_paircache->getNumOverlappingPairs());
		unsigned int total = m_profiling.m_total;
		if (total <= 0) total = 1;
		printf("ddcollide: %u%% (%uus)\r\n", (50 + m_profiling.m_ddcollide * 100) / total, m_profiling.m_ddcollide / DBVT_BP_PROFILING_RATE);
		printf("fdcollide: %u%% (%uus)\r\n", (50 + m_profiling.m_fdcollide * 100) / total, m_profiling.m_fdcollide / DBVT_BP_PROFILING_RATE);
		printf("cleanup:   %u%% (%uus)\r\n", (50 + m_profiling.m_cleanup * 100) / total, m_profiling.m_cleanup / DBVT_BP_PROFILING_RATE);
		printf("total:     %uus\r\n", total / DBVT_BP_PROFILING_RATE);
		const unsigned long sum = m_profiling.m_ddcollide +
								  m_profiling.m_fdcollide +
								  m_profiling.m_cleanup;
		printf("leaked: %u%% (%uus)\r\n", 100 - ((50 + sum * 100) / total), (total - sum) / DBVT_BP_PROFILING_RATE);
		printf("job counts: %u%%\r\n", (m_profiling.m_jobcount * 100) / ((m_sets[0].m_leaves + m_sets[1].m_leaves) * DBVT_BP_PROFILING_RATE));
		clear(m_profiling);
		m_clock.reset();
	}
#endif

	performDeferredRemoval(dispatcher);
}

void btDbvtBroadphase::performDeferredRemoval(btDispatcher* dispatcher)
{
	if (m_paircache->hasDeferredRemoval())
	{
		btBroadphasePairArray& overlappingPairArray = m_paircache->getOverlappingPairArray();

		//perform a sort, to find duplicates and to sort 'invalid' pairs to the end
		overlappingPairArray.quickSort(btBroadphasePairSortPredicate());

		int invalidPair = 0;

		int i;

		btBroadphasePair previousPair;
		previousPair.m_pProxy0 = 0;
		previousPair.m_pProxy1 = 0;
		previousPair.m_algorithm = 0;

		for (i = 0; i < overlappingPairArray.size(); i++)
		{
			btBroadphasePair& pair = overlappingPairArray[i];

			bool isDuplicate = (pair == previousPair);

			previousPair = pair;

			bool needsRemoval = false;

			if (!isDuplicate)
			{
				//important to perform AABB check that is consistent with the broadphase
				btDbvtProxy* pa = (btDbvtProxy*)pair.m_pProxy0;
				btDbvtProxy* pb = (btDbvtProxy*)pair.m_pProxy1;
				bool hasOverlap = Intersect(pa->leaf->volume, pb->leaf->volume);

				if (hasOverlap)
				{
					needsRemoval = false;
				}
				else
				{
					needsRemoval = true;
				}
			}
			else
			{
				//remove duplicate
				needsRemoval = true;
				//should have no algorithm
				btAssert(!pair.m_algorithm);
			}

			if (needsRemoval)
			{
				m_paircache->cleanOverlappingPair(pair, dispatcher);

				pair.m_pProxy0 = 0;
				pair.m_pProxy1 = 0;
				invalidPair++;
			}
		}

		//perform a sort, to sort 'invalid' pairs to the end
		overlappingPairArray.quickSort(btBroadphasePairSortPredicate());
		overlappingPairArray.resize(overlappingPairArray.size() - invalidPair);
	}
}

//
void btDbvtBroadphase::collide(btDispatcher* dispatcher)
{
	/*printf("---------------------------------------------------------\n");
	printf("m_sets[0].m_leaves=%d\n",m_sets[0].m_leaves);
	printf("m_sets[1].m_leaves=%d\n",m_sets[1].m_leaves);
	printf("numPairs = %d\n",getOverlappingPairCache()->getNumOverlappingPairs());
	{
		int i;
		for (i=0;i<getOverlappingPairCache()->getNumOverlappingPairs();i++)
		{
			printf("pair[%d]=(%d,%d),",i,getOverlappingPairCache()->getOverlappingPairArray()[i].m_pProxy0->getUid(),
				getOverlappingPairCache()->getOverlappingPairArray()[i].m_pProxy1->getUid());
		}
		printf("\n");
	}
*/

	SPC(m_profiling.m_total);
	/* optimize				*/
	m_sets[0].optimizeIncremental(1 + (m_sets[0].m_leaves * m_dupdates) / 100);
	if (m_fixedleft)
	{
		const int count = 1 + (m_sets[1].m_leaves * m_fupdates) / 100;
		m_sets[1].optimizeIncremental(1 + (m_sets[1].m_leaves * m_fupdates) / 100);
		m_fixedleft = btMax<int>(0, m_fixedleft - count);
	}
	/* dynamic -> fixed set	*/
	m_stageCurrent = (m_stageCurrent + 1) % STAGECOUNT;
	btDbvtProxy* current = m_stageRoots[m_stageCurrent];
	if (current)
	{
#if DBVT_BP_ACCURATESLEEPING
		btDbvtTreeCollider collider(this);
#endif
		do
		{
			btDbvtProxy* next = current->links[1];
			listremove(current, m_stageRoots[current->stage]);
			listappend(current, m_stageRoots[STAGECOUNT]);
#if DBVT_BP_ACCURATESLEEPING
			m_paircache->removeOverlappingPairsContainingProxy(current, dispatcher);
			collider.proxy = current;
			btDbvt::collideTV(m_sets[0].m_root, current->aabb, collider);
			btDbvt::collideTV(m_sets[1].m_root, current->aabb, collider);
#endif
			m_sets[0].remove(current->leaf);
			ATTRIBUTE_ALIGNED16(btDbvtVolume)
			curAabb = btDbvtVolume::FromMM(current->m_aabbMin, current->m_aabbMax);
			current->leaf = m_sets[1].insert(curAabb, current);
			current->stage = STAGECOUNT;
			current = next;
		} while (current);
		m_fixedleft = m_sets[1].m_leaves;
		m_needcleanup = true;
	}
	/* collide dynamics		*/
	{
		btDbvtTreeCollider collider(this);
		if (m_deferedcollide)
		{
			SPC(m_profiling.m_fdcollide);
			m_sets[0].collideTTpersistentStack(m_sets[0].m_root, m_sets[1].m_root, collider);
		}
		if (m_deferedcollide)
		{
			SPC(m_profiling.m_ddcollide);
			m_sets[0].collideTTpersistentStack(m_sets[0].m_root, m_sets[0].m_root, collider);
		}
	}
	/* clean up				*/
	if (m_needcleanup)
	{
		SPC(m_profiling.m_cleanup);
		btBroadphasePairArray& pairs = m_paircache->getOverlappingPairArray();
		if (pairs.size() > 0)
		{
			int ni = btMin(pairs.size(), btMax<int>(m_newpairs, (pairs.size() * m_cupdates) / 100));
			for (int i = 0; i < ni; ++i)
			{
				btBroadphasePair& p = pairs[(m_cid + i) % pairs.size()];
				btDbvtProxy* pa = (btDbvtProxy*)p.m_pProxy0;
				btDbvtProxy* pb = (btDbvtProxy*)p.m_pProxy1;
				if (!Intersect(pa->leaf->volume, pb->leaf->volume))
				{
#if DBVT_BP_SORTPAIRS
					if (pa->m_uniqueId > pb->m_uniqueId)
						btSwap(pa, pb);
#endif
					m_paircache->removeOverlappingPair(pa, pb, dispatcher);
					--ni;
					--i;
				}
			}
			if (pairs.size() > 0)
				m_cid = (m_cid + ni) % pairs.size();
			else
				m_cid = 0;
		}
	}
	++m_pid;
	m_newpairs = 1;
	m_needcleanup = false;
	if (m_updates_call > 0)
	{
		m_updates_ratio = m_updates_done / (btScalar)m_updates_call;
	}
	else
	{
		m_updates_ratio = 0;
	}
	m_updates_done /= 2;
	m_updates_call /= 2;
}

//
void btDbvtBroadphase::optimize()
{
	m_sets[0].optimizeTopDown();
	m_sets[1].optimizeTopDown();
}

//
btOverlappingPairCache* btDbvtBroadphase::getOverlappingPairCache()
{
	return (m_paircache);
}

//
const btOverlappingPairCache* btDbvtBroadphase::getOverlappingPairCache() const
{
	return (m_paircache);
}

//
void btDbvtBroadphase::getBroadphaseAabb(btVector3& aabbMin, btVector3& aabbMax) const
{
	ATTRIBUTE_ALIGNED16(btDbvtVolume)
	bounds;

	if (!m_sets[0].empty())
		if (!m_sets[1].empty())
			Merge(m_sets[0].m_root->volume,
				  m_sets[1].m_root->volume, bounds);
		else
			bounds = m_sets[0].m_root->volume;
	else if (!m_sets[1].empty())
		bounds = m_sets[1].m_root->volume;
	else
		bounds = btDbvtVolume::FromCR(btVector3(0, 0, 0), 0);
	aabbMin = bounds.Mins();
	aabbMax = bounds.Maxs();
}

void btDbvtBroadphase::resetPool(btDispatcher* dispatcher)
{
	int totalObjects = m_sets[0].m_leaves + m_sets[1].m_leaves;
	if (!totalObjects)
	{
		//reset internal dynamic tree data structures
		m_sets[0].clear();
		m_sets[1].clear();

		m_deferedcollide = false;
		m_needcleanup = true;
		m_stageCurrent = 0;
		m_fixedleft = 0;
		m_fupdates = 1;
		m_dupdates = 0;
		m_cupdates = 10;
		m_newpairs = 1;
		m_updates_call = 0;
		m_updates_done = 0;
		m_updates_ratio = 0;

		m_gid = 0;
		m_pid = 0;
		m_cid = 0;
		for (int i = 0; i <= STAGECOUNT; ++i)
		{
			m_stageRoots[i] = 0;
		}
	}
}

//
void btDbvtBroadphase::printStats()
{
}

//
#if DBVT_BP_ENABLE_BENCHMARK

struct btBroadphaseBenchmark
{
	struct Experiment
	{
		const char* name;
		int object_count;
		int update_count;
		int spawn_count;
		int iterations;
		btScalar speed;
		btScalar amplitude;
	};
	struct Object
	{
		btVector3 center;
		btVector3 extents;
		btBroadphaseProxy* proxy;
		btScalar time;
		void update(btScalar speed, btScalar amplitude, btBroadphaseInterface* pbi)
		{
			time += speed;
			center[0] = btCos(time * (btScalar)2.17) * amplitude +
						btSin(time) * amplitude / 2;
			center[1] = btCos(time * (btScalar)1.38) * amplitude +
						btSin(time) * amplitude;
			center[2] = btSin(time * (btScalar)0.777) * amplitude;
			pbi->setAabb(proxy, center - extents, center + extents, 0);
		}
	};
	static int UnsignedRand(int range = RAND_MAX - 1) { return (rand() % (range + 1)); }
	static btScalar UnitRand() { return (UnsignedRand(16384) / (btScalar)16384); }
	static void OutputTime(const char* name, btClock& c, unsigned count = 0)
	{
		const unsigned long us = c.getTimeMicroseconds();
		const unsigned long ms = (us + 500) / 1000;
		const btScalar sec = us / (btScalar)(1000 * 1000);
		if (count > 0)
			printf("%s : %u us (%u ms), %.2f/s\r\n", name, us, ms, count / sec);
		else
			printf("%s : %u us (%u ms)\r\n", name, us, ms);
	}
};

void btDbvtBroadphase::benchmark(btBroadphaseInterface* pbi)
{
	static const btBroadphaseBenchmark::Experiment experiments[] =
		{
			{"1024o.10%", 1024, 10, 0, 8192, (btScalar)0.005, (btScalar)100},
			/*{"4096o.10%",4096,10,0,8192,(btScalar)0.005,(btScalar)100},
		{"8192o.10%",8192,10,0,8192,(btScalar)0.005,(btScalar)100},*/
		};
	static const int nexperiments = sizeof(experiments) / sizeof(experiments[0]);
	btAlignedObjectArray<btBroadphaseBenchmark::Object*> objects;
	btClock wallclock;
	/* Begin			*/
	for (int iexp = 0; iexp < nexperiments; ++iexp)
	{
		const btBroadphaseBenchmark::Experiment& experiment = experiments[iexp];
		const int object_count = experiment.object_count;
		const int update_count = (object_count * experiment.update_count) / 100;
		const int spawn_count = (object_count * experiment.spawn_count) / 100;
		const btScalar speed = experiment.speed;
		const btScalar amplitude = experiment.amplitude;
		printf("Experiment #%u '%s':\r\n", iexp, experiment.name);
		printf("\tObjects: %u\r\n", object_count);
		printf("\tUpdate: %u\r\n", update_count);
		printf("\tSpawn: %u\r\n", spawn_count);
		printf("\tSpeed: %f\r\n", speed);
		printf("\tAmplitude: %f\r\n", amplitude);
		srand(180673);
		/* Create objects	*/
		wallclock.reset();
		objects.reserve(object_count);
		for (int i = 0; i < object_count; ++i)
		{
			btBroadphaseBenchmark::Object* po = new btBroadphaseBenchmark::Object();
			po->center[0] = btBroadphaseBenchmark::UnitRand() * 50;
			po->center[1] = btBroadphaseBenchmark::UnitRand() * 50;
			po->center[2] = btBroadphaseBenchmark::UnitRand() * 50;
			po->extents[0] = btBroadphaseBenchmark::UnitRand() * 2 + 2;
			po->extents[1] = btBroadphaseBenchmark::UnitRand() * 2 + 2;
			po->extents[2] = btBroadphaseBenchmark::UnitRand() * 2 + 2;
			po->time = btBroadphaseBenchmark::UnitRand() * 2000;
			po->proxy = pbi->createProxy(po->center - po->extents, po->center + po->extents, 0, po, 1, 1, 0, 0);
			objects.push_back(po);
		}
		btBroadphaseBenchmark::OutputTime("\tInitialization", wallclock);
		/* First update		*/
		wallclock.reset();
		for (int i = 0; i < objects.size(); ++i)
		{
			objects[i]->update(speed, amplitude, pbi);
		}
		btBroadphaseBenchmark::OutputTime("\tFirst update", wallclock);
		/* Updates			*/
		wallclock.reset();
		for (int i = 0; i < experiment.iterations; ++i)
		{
			for (int j = 0; j < update_count; ++j)
			{
				objects[j]->update(speed, amplitude, pbi);
			}
			pbi->calculateOverlappingPairs(0);
		}
		btBroadphaseBenchmark::OutputTime("\tUpdate", wallclock, experiment.iterations);
		/* Clean up			*/
		wallclock.reset();
		for (int i = 0; i < objects.size(); ++i)
		{
			pbi->destroyProxy(objects[i]->proxy, 0);
			delete objects[i];
		}
		objects.resize(0);
		btBroadphaseBenchmark::OutputTime("\tRelease", wallclock);
	}
}
#else
void btDbvtBroadphase::benchmark(btBroadphaseInterface*)
{
}
#endif

#if DBVT_BP_PROFILE
#undef SPC
#endif

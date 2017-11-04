/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2007 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
///btDbvt implementation by Nathanael Presson

#ifndef BT_DYNAMIC_BOUNDING_VOLUME_TREE_H
#define BT_DYNAMIC_BOUNDING_VOLUME_TREE_H

#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btTransform.h"
#include "LinearMath/btAabbUtil2.h"

//
// Compile time configuration
//


// Implementation profiles
#define DBVT_IMPL_GENERIC		0	// Generic implementation	
#define DBVT_IMPL_SSE			1	// SSE

// Template implementation of ICollide
#ifdef _WIN32
#if (defined (_MSC_VER) && _MSC_VER >= 1400)
#define	DBVT_USE_TEMPLATE		1
#else
#define	DBVT_USE_TEMPLATE		0
#endif
#else
#define	DBVT_USE_TEMPLATE		0
#endif

// Use only intrinsics instead of inline asm
#define DBVT_USE_INTRINSIC_SSE	1

// Using memmov for collideOCL
#define DBVT_USE_MEMMOVE		1

// Enable benchmarking code
#define	DBVT_ENABLE_BENCHMARK	0

// Inlining
#define DBVT_INLINE				SIMD_FORCE_INLINE

// Specific methods implementation

//SSE gives errors on a MSVC 7.1
#if defined (BT_USE_SSE) //&& defined (_WIN32)
#define DBVT_SELECT_IMPL		DBVT_IMPL_SSE
#define DBVT_MERGE_IMPL			DBVT_IMPL_SSE
#define DBVT_INT0_IMPL			DBVT_IMPL_SSE
#else
#define DBVT_SELECT_IMPL		DBVT_IMPL_GENERIC
#define DBVT_MERGE_IMPL			DBVT_IMPL_GENERIC
#define DBVT_INT0_IMPL			DBVT_IMPL_GENERIC
#endif

#if	(DBVT_SELECT_IMPL==DBVT_IMPL_SSE)||	\
	(DBVT_MERGE_IMPL==DBVT_IMPL_SSE)||	\
	(DBVT_INT0_IMPL==DBVT_IMPL_SSE)
#include <emmintrin.h>
#endif

//
// Auto config and checks
//

#if DBVT_USE_TEMPLATE
#define	DBVT_VIRTUAL
#define DBVT_VIRTUAL_DTOR(a)
#define DBVT_PREFIX					template <typename T>
#define DBVT_IPOLICY				T& policy
#define DBVT_CHECKTYPE				static const ICollide&	typechecker=*(T*)1;(void)typechecker;
#else
#define	DBVT_VIRTUAL_DTOR(a)		virtual ~a() {}
#define DBVT_VIRTUAL				virtual
#define DBVT_PREFIX
#define DBVT_IPOLICY				ICollide& policy
#define DBVT_CHECKTYPE
#endif

#if DBVT_USE_MEMMOVE
#if !defined( __CELLOS_LV2__) && !defined(__MWERKS__)
#include <memory.h>
#endif
#include <string.h>
#endif

#ifndef DBVT_USE_TEMPLATE
#error "DBVT_USE_TEMPLATE undefined"
#endif

#ifndef DBVT_USE_MEMMOVE
#error "DBVT_USE_MEMMOVE undefined"
#endif

#ifndef DBVT_ENABLE_BENCHMARK
#error "DBVT_ENABLE_BENCHMARK undefined"
#endif

#ifndef DBVT_SELECT_IMPL
#error "DBVT_SELECT_IMPL undefined"
#endif

#ifndef DBVT_MERGE_IMPL
#error "DBVT_MERGE_IMPL undefined"
#endif

#ifndef DBVT_INT0_IMPL
#error "DBVT_INT0_IMPL undefined"
#endif


//
// Defaults volumes
//

/* btDbvtAabbMm			*/ 
struct	btDbvtAabbMm
{
	DBVT_INLINE btVector3			Center() const	{ return((mi+mx)/2); }
	DBVT_INLINE btVector3			Lengths() const	{ return(mx-mi); }
	DBVT_INLINE btVector3			Extents() const	{ return((mx-mi)/2); }
	DBVT_INLINE const btVector3&	Mins() const	{ return(mi); }
	DBVT_INLINE const btVector3&	Maxs() const	{ return(mx); }
	static inline btDbvtAabbMm		FromCE(const btVector3& c,const btVector3& e);
	static inline btDbvtAabbMm		FromCR(const btVector3& c,btScalar r);
	static inline btDbvtAabbMm		FromMM(const btVector3& mi,const btVector3& mx);
	static inline btDbvtAabbMm		FromPoints(const btVector3* pts,int n);
	static inline btDbvtAabbMm		FromPoints(const btVector3** ppts,int n);
	DBVT_INLINE void				Expand(const btVector3& e);
	DBVT_INLINE void				SignedExpand(const btVector3& e);
	DBVT_INLINE bool				Contain(const btDbvtAabbMm& a) const;
	DBVT_INLINE int					Classify(const btVector3& n,btScalar o,int s) const;
	DBVT_INLINE btScalar			ProjectMinimum(const btVector3& v,unsigned signs) const;
	DBVT_INLINE friend bool			Intersect(	const btDbvtAabbMm& a,
		const btDbvtAabbMm& b);
	
	DBVT_INLINE friend bool			Intersect(	const btDbvtAabbMm& a,
		const btVector3& b);

	DBVT_INLINE friend btScalar		Proximity(	const btDbvtAabbMm& a,
		const btDbvtAabbMm& b);
	DBVT_INLINE friend int			Select(		const btDbvtAabbMm& o,
		const btDbvtAabbMm& a,
		const btDbvtAabbMm& b);
	DBVT_INLINE friend void			Merge(		const btDbvtAabbMm& a,
		const btDbvtAabbMm& b,
		btDbvtAabbMm& r);
	DBVT_INLINE friend bool			NotEqual(	const btDbvtAabbMm& a,
		const btDbvtAabbMm& b);
    
    DBVT_INLINE btVector3&	tMins()	{ return(mi); }
	DBVT_INLINE btVector3&	tMaxs()	{ return(mx); }
    
private:
	DBVT_INLINE void				AddSpan(const btVector3& d,btScalar& smi,btScalar& smx) const;
private:
	btVector3	mi,mx;
};

// Types	
typedef	btDbvtAabbMm	btDbvtVolume;

/* btDbvtNode				*/ 
struct	btDbvtNode
{
	btDbvtVolume	volume;
	btDbvtNode*		parent;
	DBVT_INLINE bool	isleaf() const		{ return(childs[1]==0); }
	DBVT_INLINE bool	isinternal() const	{ return(!isleaf()); }
	union
	{
		btDbvtNode*	childs[2];
		void*	data;
		int		dataAsInt;
	};
};

typedef btAlignedObjectArray<const btDbvtNode*> btNodeStack;


///The btDbvt class implements a fast dynamic bounding volume tree based on axis aligned bounding boxes (aabb tree).
///This btDbvt is used for soft body collision detection and for the btDbvtBroadphase. It has a fast insert, remove and update of nodes.
///Unlike the btQuantizedBvh, nodes can be dynamically moved around, which allows for change in topology of the underlying data structure.
struct	btDbvt
{
	/* Stack element	*/ 
	struct	sStkNN
	{
		const btDbvtNode*	a;
		const btDbvtNode*	b;
		sStkNN() {}
		sStkNN(const btDbvtNode* na,const btDbvtNode* nb) : a(na),b(nb) {}
	};
	struct	sStkNP
	{
		const btDbvtNode*	node;
		int			mask;
		sStkNP(const btDbvtNode* n,unsigned m) : node(n),mask(m) {}
	};
	struct	sStkNPS
	{
		const btDbvtNode*	node;
		int			mask;
		btScalar	value;
		sStkNPS() {}
		sStkNPS(const btDbvtNode* n,unsigned m,btScalar v) : node(n),mask(m),value(v) {}
	};
	struct	sStkCLN
	{
		const btDbvtNode*	node;
		btDbvtNode*		parent;
		sStkCLN(const btDbvtNode* n,btDbvtNode* p) : node(n),parent(p) {}
	};
	// Policies/Interfaces

	/* ICollide	*/ 
	struct	ICollide
	{		
		DBVT_VIRTUAL_DTOR(ICollide)
			DBVT_VIRTUAL void	Process(const btDbvtNode*,const btDbvtNode*)		{}
		DBVT_VIRTUAL void	Process(const btDbvtNode*)					{}
		DBVT_VIRTUAL void	Process(const btDbvtNode* n,btScalar)			{ Process(n); }
		DBVT_VIRTUAL bool	Descent(const btDbvtNode*)					{ return(true); }
		DBVT_VIRTUAL bool	AllLeaves(const btDbvtNode*)					{ return(true); }
	};
	/* IWriter	*/ 
	struct	IWriter
	{
		virtual ~IWriter() {}
		virtual void		Prepare(const btDbvtNode* root,int numnodes)=0;
		virtual void		WriteNode(const btDbvtNode*,int index,int parent,int child0,int child1)=0;
		virtual void		WriteLeaf(const btDbvtNode*,int index,int parent)=0;
	};
	/* IClone	*/ 
	struct	IClone
	{
		virtual ~IClone()	{}
		virtual void		CloneLeaf(btDbvtNode*) {}
	};

	// Constants
	enum	{
		SIMPLE_STACKSIZE	=	64,
		DOUBLE_STACKSIZE	=	SIMPLE_STACKSIZE*2
	};

	// Fields
	btDbvtNode*		m_root;
	btDbvtNode*		m_free;
	int				m_lkhd;
	int				m_leaves;
	unsigned		m_opath;

	
	btAlignedObjectArray<sStkNN>	m_stkStack;


	// Methods
	btDbvt();
	~btDbvt();
	void			clear();
	bool			empty() const { return(0==m_root); }
	void			optimizeBottomUp();
	void			optimizeTopDown(int bu_treshold=128);
	void			optimizeIncremental(int passes);
	btDbvtNode*		insert(const btDbvtVolume& box,void* data);
	void			update(btDbvtNode* leaf,int lookahead=-1);
	void			update(btDbvtNode* leaf,btDbvtVolume& volume);
	bool			update(btDbvtNode* leaf,btDbvtVolume& volume,const btVector3& velocity,btScalar margin);
	bool			update(btDbvtNode* leaf,btDbvtVolume& volume,const btVector3& velocity);
	bool			update(btDbvtNode* leaf,btDbvtVolume& volume,btScalar margin);	
	void			remove(btDbvtNode* leaf);
	void			write(IWriter* iwriter) const;
	void			clone(btDbvt& dest,IClone* iclone=0) const;
	static int		maxdepth(const btDbvtNode* node);
	static int		countLeaves(const btDbvtNode* node);
	static void		extractLeaves(const btDbvtNode* node,btAlignedObjectArray<const btDbvtNode*>& leaves);
#if DBVT_ENABLE_BENCHMARK
	static void		benchmark();
#else
	static void		benchmark(){}
#endif
	// DBVT_IPOLICY must support ICollide policy/interface
	DBVT_PREFIX
		static void		enumNodes(	const btDbvtNode* root,
		DBVT_IPOLICY);
	DBVT_PREFIX
		static void		enumLeaves(	const btDbvtNode* root,
		DBVT_IPOLICY);
	DBVT_PREFIX
		void		collideTT(	const btDbvtNode* root0,
		const btDbvtNode* root1,
		DBVT_IPOLICY);

	DBVT_PREFIX
		void		collideTTpersistentStack(	const btDbvtNode* root0,
		  const btDbvtNode* root1,
		  DBVT_IPOLICY);
#if 0
	DBVT_PREFIX
		void		collideTT(	const btDbvtNode* root0,
		const btDbvtNode* root1,
		const btTransform& xform,
		DBVT_IPOLICY);
	DBVT_PREFIX
		void		collideTT(	const btDbvtNode* root0,
		const btTransform& xform0,
		const btDbvtNode* root1,
		const btTransform& xform1,
		DBVT_IPOLICY);
#endif

	DBVT_PREFIX
		void		collideTV(	const btDbvtNode* root,
		const btDbvtVolume& volume,
		DBVT_IPOLICY) const;
	
	DBVT_PREFIX
	void		collideTVNoStackAlloc(	const btDbvtNode* root,
						  const btDbvtVolume& volume,
						  btNodeStack& stack,
						  DBVT_IPOLICY) const;
	
	
	
	
	///rayTest is a re-entrant ray test, and can be called in parallel as long as the btAlignedAlloc is thread-safe (uses locking etc)
	///rayTest is slower than rayTestInternal, because it builds a local stack, using memory allocations, and it recomputes signs/rayDirectionInverses each time
	DBVT_PREFIX
		static void		rayTest(	const btDbvtNode* root,
		const btVector3& rayFrom,
		const btVector3& rayTo,
		DBVT_IPOLICY);
	///rayTestInternal is faster than rayTest, because it uses a persistent stack (to reduce dynamic memory allocations to a minimum) and it uses precomputed signs/rayInverseDirections
	///rayTestInternal is used by btDbvtBroadphase to accelerate world ray casts
	DBVT_PREFIX
		void		rayTestInternal(	const btDbvtNode* root,
								const btVector3& rayFrom,
								const btVector3& rayTo,
								const btVector3& rayDirectionInverse,
								unsigned int signs[3],
								btScalar lambda_max,
								const btVector3& aabbMin,
								const btVector3& aabbMax,
                                btAlignedObjectArray<const btDbvtNode*>& stack,
								DBVT_IPOLICY) const;

	DBVT_PREFIX
		static void		collideKDOP(const btDbvtNode* root,
		const btVector3* normals,
		const btScalar* offsets,
		int count,
		DBVT_IPOLICY);
	DBVT_PREFIX
		static void		collideOCL(	const btDbvtNode* root,
		const btVector3* normals,
		const btScalar* offsets,
		const btVector3& sortaxis,
		int count,								
		DBVT_IPOLICY,
		bool fullsort=true);
	DBVT_PREFIX
		static void		collideTU(	const btDbvtNode* root,
		DBVT_IPOLICY);
	// Helpers	
	static DBVT_INLINE int	nearest(const int* i,const btDbvt::sStkNPS* a,btScalar v,int l,int h)
	{
		int	m=0;
		while(l<h)
		{
			m=(l+h)>>1;
			if(a[i[m]].value>=v) l=m+1; else h=m;
		}
		return(h);
	}
	static DBVT_INLINE int	allocate(	btAlignedObjectArray<int>& ifree,
		btAlignedObjectArray<sStkNPS>& stock,
		const sStkNPS& value)
	{
		int	i;
		if(ifree.size()>0)
		{ i=ifree[ifree.size()-1];ifree.pop_back();stock[i]=value; }
		else
		{ i=stock.size();stock.push_back(value); }
		return(i); 
	}
	//
private:
	btDbvt(const btDbvt&)	{}	
};

//
// Inline's
//

//
inline btDbvtAabbMm			btDbvtAabbMm::FromCE(const btVector3& c,const btVector3& e)
{
	btDbvtAabbMm box;
	box.mi=c-e;box.mx=c+e;
	return(box);
}

//
inline btDbvtAabbMm			btDbvtAabbMm::FromCR(const btVector3& c,btScalar r)
{
	return(FromCE(c,btVector3(r,r,r)));
}

//
inline btDbvtAabbMm			btDbvtAabbMm::FromMM(const btVector3& mi,const btVector3& mx)
{
	btDbvtAabbMm box;
	box.mi=mi;box.mx=mx;
	return(box);
}

//
inline btDbvtAabbMm			btDbvtAabbMm::FromPoints(const btVector3* pts,int n)
{
	btDbvtAabbMm box;
	box.mi=box.mx=pts[0];
	for(int i=1;i<n;++i)
	{
		box.mi.setMin(pts[i]);
		box.mx.setMax(pts[i]);
	}
	return(box);
}

//
inline btDbvtAabbMm			btDbvtAabbMm::FromPoints(const btVector3** ppts,int n)
{
	btDbvtAabbMm box;
	box.mi=box.mx=*ppts[0];
	for(int i=1;i<n;++i)
	{
		box.mi.setMin(*ppts[i]);
		box.mx.setMax(*ppts[i]);
	}
	return(box);
}

//
DBVT_INLINE void		btDbvtAabbMm::Expand(const btVector3& e)
{
	mi-=e;mx+=e;
}

//
DBVT_INLINE void		btDbvtAabbMm::SignedExpand(const btVector3& e)
{
	if(e.x()>0) mx.setX(mx.x()+e[0]); else mi.setX(mi.x()+e[0]);
	if(e.y()>0) mx.setY(mx.y()+e[1]); else mi.setY(mi.y()+e[1]);
	if(e.z()>0) mx.setZ(mx.z()+e[2]); else mi.setZ(mi.z()+e[2]);
}

//
DBVT_INLINE bool		btDbvtAabbMm::Contain(const btDbvtAabbMm& a) const
{
	return(	(mi.x()<=a.mi.x())&&
		(mi.y()<=a.mi.y())&&
		(mi.z()<=a.mi.z())&&
		(mx.x()>=a.mx.x())&&
		(mx.y()>=a.mx.y())&&
		(mx.z()>=a.mx.z()));
}

//
DBVT_INLINE int		btDbvtAabbMm::Classify(const btVector3& n,btScalar o,int s) const
{
	btVector3			pi,px;
	switch(s)
	{
	case	(0+0+0):	px=btVector3(mi.x(),mi.y(),mi.z());
		pi=btVector3(mx.x(),mx.y(),mx.z());break;
	case	(1+0+0):	px=btVector3(mx.x(),mi.y(),mi.z());
		pi=btVector3(mi.x(),mx.y(),mx.z());break;
	case	(0+2+0):	px=btVector3(mi.x(),mx.y(),mi.z());
		pi=btVector3(mx.x(),mi.y(),mx.z());break;
	case	(1+2+0):	px=btVector3(mx.x(),mx.y(),mi.z());
		pi=btVector3(mi.x(),mi.y(),mx.z());break;
	case	(0+0+4):	px=btVector3(mi.x(),mi.y(),mx.z());
		pi=btVector3(mx.x(),mx.y(),mi.z());break;
	case	(1+0+4):	px=btVector3(mx.x(),mi.y(),mx.z());
		pi=btVector3(mi.x(),mx.y(),mi.z());break;
	case	(0+2+4):	px=btVector3(mi.x(),mx.y(),mx.z());
		pi=btVector3(mx.x(),mi.y(),mi.z());break;
	case	(1+2+4):	px=btVector3(mx.x(),mx.y(),mx.z());
		pi=btVector3(mi.x(),mi.y(),mi.z());break;
	}
	if((btDot(n,px)+o)<0)		return(-1);
	if((btDot(n,pi)+o)>=0)	return(+1);
	return(0);
}

//
DBVT_INLINE btScalar	btDbvtAabbMm::ProjectMinimum(const btVector3& v,unsigned signs) const
{
	const btVector3*	b[]={&mx,&mi};
	const btVector3		p(	b[(signs>>0)&1]->x(),
		b[(signs>>1)&1]->y(),
		b[(signs>>2)&1]->z());
	return(btDot(p,v));
}

//
DBVT_INLINE void		btDbvtAabbMm::AddSpan(const btVector3& d,btScalar& smi,btScalar& smx) const
{
	for(int i=0;i<3;++i)
	{
		if(d[i]<0)
		{ smi+=mx[i]*d[i];smx+=mi[i]*d[i]; }
		else
		{ smi+=mi[i]*d[i];smx+=mx[i]*d[i]; }
	}
}

//
DBVT_INLINE bool		Intersect(	const btDbvtAabbMm& a,
								  const btDbvtAabbMm& b)
{
#if	DBVT_INT0_IMPL == DBVT_IMPL_SSE
	const __m128	rt(_mm_or_ps(	_mm_cmplt_ps(_mm_load_ps(b.mx),_mm_load_ps(a.mi)),
		_mm_cmplt_ps(_mm_load_ps(a.mx),_mm_load_ps(b.mi))));
#if defined (_WIN32)
	const __int32*	pu((const __int32*)&rt);
#else
    const int*	pu((const int*)&rt);
#endif
	return((pu[0]|pu[1]|pu[2])==0);
#else
	return(	(a.mi.x()<=b.mx.x())&&
		(a.mx.x()>=b.mi.x())&&
		(a.mi.y()<=b.mx.y())&&
		(a.mx.y()>=b.mi.y())&&
		(a.mi.z()<=b.mx.z())&&		
		(a.mx.z()>=b.mi.z()));
#endif
}



//
DBVT_INLINE bool		Intersect(	const btDbvtAabbMm& a,
								  const btVector3& b)
{
	return(	(b.x()>=a.mi.x())&&
		(b.y()>=a.mi.y())&&
		(b.z()>=a.mi.z())&&
		(b.x()<=a.mx.x())&&
		(b.y()<=a.mx.y())&&
		(b.z()<=a.mx.z()));
}





//////////////////////////////////////


//
DBVT_INLINE btScalar	Proximity(	const btDbvtAabbMm& a,
								  const btDbvtAabbMm& b)
{
	const btVector3	d=(a.mi+a.mx)-(b.mi+b.mx);
	return(btFabs(d.x())+btFabs(d.y())+btFabs(d.z()));
}



//
DBVT_INLINE int			Select(	const btDbvtAabbMm& o,
							   const btDbvtAabbMm& a,
							   const btDbvtAabbMm& b)
{
#if	DBVT_SELECT_IMPL == DBVT_IMPL_SSE
    
#if defined (_WIN32)
	static ATTRIBUTE_ALIGNED16(const unsigned __int32)	mask[]={0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff};
#else
    static ATTRIBUTE_ALIGNED16(const unsigned int)	mask[]={0x7fffffff,0x7fffffff,0x7fffffff,0x00000000 /*0x7fffffff*/};
#endif
	///@todo: the intrinsic version is 11% slower
#if DBVT_USE_INTRINSIC_SSE

	union btSSEUnion ///NOTE: if we use more intrinsics, move btSSEUnion into the LinearMath directory
	{
	   __m128		ssereg;
	   float		floats[4];
	   int			ints[4];
	};

	__m128	omi(_mm_load_ps(o.mi));
	omi=_mm_add_ps(omi,_mm_load_ps(o.mx));
	__m128	ami(_mm_load_ps(a.mi));
	ami=_mm_add_ps(ami,_mm_load_ps(a.mx));
	ami=_mm_sub_ps(ami,omi);
	ami=_mm_and_ps(ami,_mm_load_ps((const float*)mask));
	__m128	bmi(_mm_load_ps(b.mi));
	bmi=_mm_add_ps(bmi,_mm_load_ps(b.mx));
	bmi=_mm_sub_ps(bmi,omi);
	bmi=_mm_and_ps(bmi,_mm_load_ps((const float*)mask));
	__m128	t0(_mm_movehl_ps(ami,ami));
	ami=_mm_add_ps(ami,t0);
	ami=_mm_add_ss(ami,_mm_shuffle_ps(ami,ami,1));
	__m128 t1(_mm_movehl_ps(bmi,bmi));
	bmi=_mm_add_ps(bmi,t1);
	bmi=_mm_add_ss(bmi,_mm_shuffle_ps(bmi,bmi,1));
	
	btSSEUnion tmp;
	tmp.ssereg = _mm_cmple_ss(bmi,ami);
	return tmp.ints[0]&1;

#else
	ATTRIBUTE_ALIGNED16(__int32	r[1]);
	__asm
	{
		mov		eax,o
			mov		ecx,a
			mov		edx,b
			movaps	xmm0,[eax]
		movaps	xmm5,mask
			addps	xmm0,[eax+16]	
		movaps	xmm1,[ecx]
		movaps	xmm2,[edx]
		addps	xmm1,[ecx+16]
		addps	xmm2,[edx+16]
		subps	xmm1,xmm0
			subps	xmm2,xmm0
			andps	xmm1,xmm5
			andps	xmm2,xmm5
			movhlps	xmm3,xmm1
			movhlps	xmm4,xmm2
			addps	xmm1,xmm3
			addps	xmm2,xmm4
			pshufd	xmm3,xmm1,1
			pshufd	xmm4,xmm2,1
			addss	xmm1,xmm3
			addss	xmm2,xmm4
			cmpless	xmm2,xmm1
			movss	r,xmm2
	}
	return(r[0]&1);
#endif
#else
	return(Proximity(o,a)<Proximity(o,b)?0:1);
#endif
}

//
DBVT_INLINE void		Merge(	const btDbvtAabbMm& a,
							  const btDbvtAabbMm& b,
							  btDbvtAabbMm& r)
{
#if DBVT_MERGE_IMPL==DBVT_IMPL_SSE
	__m128	ami(_mm_load_ps(a.mi));
	__m128	amx(_mm_load_ps(a.mx));
	__m128	bmi(_mm_load_ps(b.mi));
	__m128	bmx(_mm_load_ps(b.mx));
	ami=_mm_min_ps(ami,bmi);
	amx=_mm_max_ps(amx,bmx);
	_mm_store_ps(r.mi,ami);
	_mm_store_ps(r.mx,amx);
#else
	for(int i=0;i<3;++i)
	{
		if(a.mi[i]<b.mi[i]) r.mi[i]=a.mi[i]; else r.mi[i]=b.mi[i];
		if(a.mx[i]>b.mx[i]) r.mx[i]=a.mx[i]; else r.mx[i]=b.mx[i];
	}
#endif
}

//
DBVT_INLINE bool		NotEqual(	const btDbvtAabbMm& a,
								 const btDbvtAabbMm& b)
{
	return(	(a.mi.x()!=b.mi.x())||
		(a.mi.y()!=b.mi.y())||
		(a.mi.z()!=b.mi.z())||
		(a.mx.x()!=b.mx.x())||
		(a.mx.y()!=b.mx.y())||
		(a.mx.z()!=b.mx.z()));
}

//
// Inline's
//

//
DBVT_PREFIX
inline void		btDbvt::enumNodes(	const btDbvtNode* root,
								  DBVT_IPOLICY)
{
	DBVT_CHECKTYPE
		policy.Process(root);
	if(root->isinternal())
	{
		enumNodes(root->childs[0],policy);
		enumNodes(root->childs[1],policy);
	}
}

//
DBVT_PREFIX
inline void		btDbvt::enumLeaves(	const btDbvtNode* root,
								   DBVT_IPOLICY)
{
	DBVT_CHECKTYPE
		if(root->isinternal())
		{
			enumLeaves(root->childs[0],policy);
			enumLeaves(root->childs[1],policy);
		}
		else
		{
			policy.Process(root);
		}
}

//
DBVT_PREFIX
inline void		btDbvt::collideTT(	const btDbvtNode* root0,
								  const btDbvtNode* root1,
								  DBVT_IPOLICY)
{
	DBVT_CHECKTYPE
		if(root0&&root1)
		{
			int								depth=1;
			int								treshold=DOUBLE_STACKSIZE-4;
			btAlignedObjectArray<sStkNN>	stkStack;
			stkStack.resize(DOUBLE_STACKSIZE);
			stkStack[0]=sStkNN(root0,root1);
			do	{		
				sStkNN	p=stkStack[--depth];
				if(depth>treshold)
				{
					stkStack.resize(stkStack.size()*2);
					treshold=stkStack.size()-4;
				}
				if(p.a==p.b)
				{
					if(p.a->isinternal())
					{
						stkStack[depth++]=sStkNN(p.a->childs[0],p.a->childs[0]);
						stkStack[depth++]=sStkNN(p.a->childs[1],p.a->childs[1]);
						stkStack[depth++]=sStkNN(p.a->childs[0],p.a->childs[1]);
					}
				}
				else if(Intersect(p.a->volume,p.b->volume))
				{
					if(p.a->isinternal())
					{
						if(p.b->isinternal())
						{
							stkStack[depth++]=sStkNN(p.a->childs[0],p.b->childs[0]);
							stkStack[depth++]=sStkNN(p.a->childs[1],p.b->childs[0]);
							stkStack[depth++]=sStkNN(p.a->childs[0],p.b->childs[1]);
							stkStack[depth++]=sStkNN(p.a->childs[1],p.b->childs[1]);
						}
						else
						{
							stkStack[depth++]=sStkNN(p.a->childs[0],p.b);
							stkStack[depth++]=sStkNN(p.a->childs[1],p.b);
						}
					}
					else
					{
						if(p.b->isinternal())
						{
							stkStack[depth++]=sStkNN(p.a,p.b->childs[0]);
							stkStack[depth++]=sStkNN(p.a,p.b->childs[1]);
						}
						else
						{
							policy.Process(p.a,p.b);
						}
					}
				}
			} while(depth);
		}
}



DBVT_PREFIX
inline void		btDbvt::collideTTpersistentStack(	const btDbvtNode* root0,
								  const btDbvtNode* root1,
								  DBVT_IPOLICY)
{
	DBVT_CHECKTYPE
		if(root0&&root1)
		{
			int								depth=1;
			int								treshold=DOUBLE_STACKSIZE-4;
			
			m_stkStack.resize(DOUBLE_STACKSIZE);
			m_stkStack[0]=sStkNN(root0,root1);
			do	{		
				sStkNN	p=m_stkStack[--depth];
				if(depth>treshold)
				{
					m_stkStack.resize(m_stkStack.size()*2);
					treshold=m_stkStack.size()-4;
				}
				if(p.a==p.b)
				{
					if(p.a->isinternal())
					{
						m_stkStack[depth++]=sStkNN(p.a->childs[0],p.a->childs[0]);
						m_stkStack[depth++]=sStkNN(p.a->childs[1],p.a->childs[1]);
						m_stkStack[depth++]=sStkNN(p.a->childs[0],p.a->childs[1]);
					}
				}
				else if(Intersect(p.a->volume,p.b->volume))
				{
					if(p.a->isinternal())
					{
						if(p.b->isinternal())
						{
							m_stkStack[depth++]=sStkNN(p.a->childs[0],p.b->childs[0]);
							m_stkStack[depth++]=sStkNN(p.a->childs[1],p.b->childs[0]);
							m_stkStack[depth++]=sStkNN(p.a->childs[0],p.b->childs[1]);
							m_stkStack[depth++]=sStkNN(p.a->childs[1],p.b->childs[1]);
						}
						else
						{
							m_stkStack[depth++]=sStkNN(p.a->childs[0],p.b);
							m_stkStack[depth++]=sStkNN(p.a->childs[1],p.b);
						}
					}
					else
					{
						if(p.b->isinternal())
						{
							m_stkStack[depth++]=sStkNN(p.a,p.b->childs[0]);
							m_stkStack[depth++]=sStkNN(p.a,p.b->childs[1]);
						}
						else
						{
							policy.Process(p.a,p.b);
						}
					}
				}
			} while(depth);
		}
}

#if 0
//
DBVT_PREFIX
inline void		btDbvt::collideTT(	const btDbvtNode* root0,
								  const btDbvtNode* root1,
								  const btTransform& xform,
								  DBVT_IPOLICY)
{
	DBVT_CHECKTYPE
		if(root0&&root1)
		{
			int								depth=1;
			int								treshold=DOUBLE_STACKSIZE-4;
			btAlignedObjectArray<sStkNN>	stkStack;
			stkStack.resize(DOUBLE_STACKSIZE);
			stkStack[0]=sStkNN(root0,root1);
			do	{
				sStkNN	p=stkStack[--depth];
				if(Intersect(p.a->volume,p.b->volume,xform))
				{
					if(depth>treshold)
					{
						stkStack.resize(stkStack.size()*2);
						treshold=stkStack.size()-4;
					}
					if(p.a->isinternal())
					{
						if(p.b->isinternal())
						{					
							stkStack[depth++]=sStkNN(p.a->childs[0],p.b->childs[0]);
							stkStack[depth++]=sStkNN(p.a->childs[1],p.b->childs[0]);
							stkStack[depth++]=sStkNN(p.a->childs[0],p.b->childs[1]);
							stkStack[depth++]=sStkNN(p.a->childs[1],p.b->childs[1]);
						}
						else
						{
							stkStack[depth++]=sStkNN(p.a->childs[0],p.b);
							stkStack[depth++]=sStkNN(p.a->childs[1],p.b);
						}
					}
					else
					{
						if(p.b->isinternal())
						{
							stkStack[depth++]=sStkNN(p.a,p.b->childs[0]);
							stkStack[depth++]=sStkNN(p.a,p.b->childs[1]);
						}
						else
						{
							policy.Process(p.a,p.b);
						}
					}
				}
			} while(depth);
		}
}
//
DBVT_PREFIX
inline void		btDbvt::collideTT(	const btDbvtNode* root0,
								  const btTransform& xform0,
								  const btDbvtNode* root1,
								  const btTransform& xform1,
								  DBVT_IPOLICY)
{
	const btTransform	xform=xform0.inverse()*xform1;
	collideTT(root0,root1,xform,policy);
}
#endif 

DBVT_PREFIX
inline void		btDbvt::collideTV(	const btDbvtNode* root,
								  const btDbvtVolume& vol,
								  DBVT_IPOLICY) const
{
	DBVT_CHECKTYPE
	if(root)
	{
		ATTRIBUTE_ALIGNED16(btDbvtVolume)		volume(vol);
		btAlignedObjectArray<const btDbvtNode*>	stack;
		stack.resize(0);
#ifndef BT_DISABLE_STACK_TEMP_MEMORY
		char tempmemory[SIMPLE_STACKSIZE*sizeof(const btDbvtNode*)];
		stack.initializeFromBuffer(tempmemory, 0, SIMPLE_STACKSIZE);
#else
		stack.reserve(SIMPLE_STACKSIZE);
#endif //BT_DISABLE_STACK_TEMP_MEMORY

		stack.push_back(root);
		do	{
			const btDbvtNode*	n=stack[stack.size()-1];
			stack.pop_back();
			if(Intersect(n->volume,volume))
			{
				if(n->isinternal())
				{
					stack.push_back(n->childs[0]);
					stack.push_back(n->childs[1]);
				}
				else
				{
					policy.Process(n);
				}
			}
		} while(stack.size()>0);
	}
}

//
DBVT_PREFIX
inline void		btDbvt::collideTVNoStackAlloc(	const btDbvtNode* root,
											 const btDbvtVolume& vol,
											 btNodeStack& stack,
											 DBVT_IPOLICY) const
{
	DBVT_CHECKTYPE
	if(root)
	{
		ATTRIBUTE_ALIGNED16(btDbvtVolume)		volume(vol);
		stack.resize(0);
		stack.reserve(SIMPLE_STACKSIZE);
		stack.push_back(root);
		do	{
			const btDbvtNode*	n=stack[stack.size()-1];
			stack.pop_back();
			if(Intersect(n->volume,volume))
			{
				if(n->isinternal())
				{
					stack.push_back(n->childs[0]);
					stack.push_back(n->childs[1]);
				}
				else
				{
					policy.Process(n);
				}
			}
		} while(stack.size()>0);
	}
}


DBVT_PREFIX
inline void		btDbvt::rayTestInternal(	const btDbvtNode* root,
								const btVector3& rayFrom,
								const btVector3& rayTo,
								const btVector3& rayDirectionInverse,
								unsigned int signs[3],
								btScalar lambda_max,
								const btVector3& aabbMin,
								const btVector3& aabbMax,
                                btAlignedObjectArray<const btDbvtNode*>& stack,
                                DBVT_IPOLICY ) const
{
        (void) rayTo;
	DBVT_CHECKTYPE
	if(root)
	{
		btVector3 resultNormal;

		int								depth=1;
		int								treshold=DOUBLE_STACKSIZE-2;
		stack.resize(DOUBLE_STACKSIZE);
		stack[0]=root;
		btVector3 bounds[2];
		do	
		{
			const btDbvtNode*	node=stack[--depth];
			bounds[0] = node->volume.Mins()-aabbMax;
			bounds[1] = node->volume.Maxs()-aabbMin;
			btScalar tmin=1.f,lambda_min=0.f;
			unsigned int result1=false;
			result1 = btRayAabb2(rayFrom,rayDirectionInverse,signs,bounds,tmin,lambda_min,lambda_max);
			if(result1)
			{
				if(node->isinternal())
				{
					if(depth>treshold)
					{
						stack.resize(stack.size()*2);
						treshold=stack.size()-2;
					}
					stack[depth++]=node->childs[0];
					stack[depth++]=node->childs[1];
				}
				else
				{
					policy.Process(node);
				}
			}
		} while(depth);
	}
}

//
DBVT_PREFIX
inline void		btDbvt::rayTest(	const btDbvtNode* root,
								const btVector3& rayFrom,
								const btVector3& rayTo,
								DBVT_IPOLICY)
{
	DBVT_CHECKTYPE
		if(root)
		{
			btVector3 rayDir = (rayTo-rayFrom);
			rayDir.normalize ();

			///what about division by zero? --> just set rayDirection[i] to INF/BT_LARGE_FLOAT
			btVector3 rayDirectionInverse;
			rayDirectionInverse[0] = rayDir[0] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[0];
			rayDirectionInverse[1] = rayDir[1] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[1];
			rayDirectionInverse[2] = rayDir[2] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[2];
			unsigned int signs[3] = { rayDirectionInverse[0] < 0.0, rayDirectionInverse[1] < 0.0, rayDirectionInverse[2] < 0.0};

			btScalar lambda_max = rayDir.dot(rayTo-rayFrom);

			btVector3 resultNormal;

			btAlignedObjectArray<const btDbvtNode*>	stack;

			int								depth=1;
			int								treshold=DOUBLE_STACKSIZE-2;

			char tempmemory[DOUBLE_STACKSIZE * sizeof(const btDbvtNode*)];
#ifndef BT_DISABLE_STACK_TEMP_MEMORY
			stack.initializeFromBuffer(tempmemory, DOUBLE_STACKSIZE, DOUBLE_STACKSIZE);
#else//BT_DISABLE_STACK_TEMP_MEMORY
			stack.resize(DOUBLE_STACKSIZE);
#endif //BT_DISABLE_STACK_TEMP_MEMORY
			stack[0]=root;
			btVector3 bounds[2];
			do	{
				const btDbvtNode*	node=stack[--depth];

				bounds[0] = node->volume.Mins();
				bounds[1] = node->volume.Maxs();
				
				btScalar tmin=1.f,lambda_min=0.f;
				unsigned int result1 = btRayAabb2(rayFrom,rayDirectionInverse,signs,bounds,tmin,lambda_min,lambda_max);

#ifdef COMPARE_BTRAY_AABB2
				btScalar param=1.f;
				bool result2 = btRayAabb(rayFrom,rayTo,node->volume.Mins(),node->volume.Maxs(),param,resultNormal);
				btAssert(result1 == result2);
#endif //TEST_BTRAY_AABB2

				if(result1)
				{
					if(node->isinternal())
					{
						if(depth>treshold)
						{
							stack.resize(stack.size()*2);
							treshold=stack.size()-2;
						}
						stack[depth++]=node->childs[0];
						stack[depth++]=node->childs[1];
					}
					else
					{
						policy.Process(node);
					}
				}
			} while(depth);

		}
}

//
DBVT_PREFIX
inline void		btDbvt::collideKDOP(const btDbvtNode* root,
									const btVector3* normals,
									const btScalar* offsets,
									int count,
									DBVT_IPOLICY)
{
	DBVT_CHECKTYPE
		if(root)
		{
			const int						inside=(1<<count)-1;
			btAlignedObjectArray<sStkNP>	stack;
			int								signs[sizeof(unsigned)*8];
			btAssert(count<int (sizeof(signs)/sizeof(signs[0])));
			for(int i=0;i<count;++i)
			{
				signs[i]=	((normals[i].x()>=0)?1:0)+
					((normals[i].y()>=0)?2:0)+
					((normals[i].z()>=0)?4:0);
			}
			stack.reserve(SIMPLE_STACKSIZE);
			stack.push_back(sStkNP(root,0));
			do	{
				sStkNP	se=stack[stack.size()-1];
				bool	out=false;
				stack.pop_back();
				for(int i=0,j=1;(!out)&&(i<count);++i,j<<=1)
				{
					if(0==(se.mask&j))
					{
						const int	side=se.node->volume.Classify(normals[i],offsets[i],signs[i]);
						switch(side)
						{
						case	-1:	out=true;break;
						case	+1:	se.mask|=j;break;
						}
					}
				}
				if(!out)
				{
					if((se.mask!=inside)&&(se.node->isinternal()))
					{
						stack.push_back(sStkNP(se.node->childs[0],se.mask));
						stack.push_back(sStkNP(se.node->childs[1],se.mask));
					}
					else
					{
						if(policy.AllLeaves(se.node)) enumLeaves(se.node,policy);
					}
				}
			} while(stack.size());
		}
}

//
DBVT_PREFIX
inline void		btDbvt::collideOCL(	const btDbvtNode* root,
								   const btVector3* normals,
								   const btScalar* offsets,
								   const btVector3& sortaxis,
								   int count,
								   DBVT_IPOLICY,
								   bool fsort)
{
	DBVT_CHECKTYPE
		if(root)
		{
			const unsigned					srtsgns=(sortaxis[0]>=0?1:0)+
				(sortaxis[1]>=0?2:0)+
				(sortaxis[2]>=0?4:0);
			const int						inside=(1<<count)-1;
			btAlignedObjectArray<sStkNPS>	stock;
			btAlignedObjectArray<int>		ifree;
			btAlignedObjectArray<int>		stack;
			int								signs[sizeof(unsigned)*8];
			btAssert(count<int (sizeof(signs)/sizeof(signs[0])));
			for(int i=0;i<count;++i)
			{
				signs[i]=	((normals[i].x()>=0)?1:0)+
					((normals[i].y()>=0)?2:0)+
					((normals[i].z()>=0)?4:0);
			}
			stock.reserve(SIMPLE_STACKSIZE);
			stack.reserve(SIMPLE_STACKSIZE);
			ifree.reserve(SIMPLE_STACKSIZE);
			stack.push_back(allocate(ifree,stock,sStkNPS(root,0,root->volume.ProjectMinimum(sortaxis,srtsgns))));
			do	{
				const int	id=stack[stack.size()-1];
				sStkNPS		se=stock[id];
				stack.pop_back();ifree.push_back(id);
				if(se.mask!=inside)
				{
					bool	out=false;
					for(int i=0,j=1;(!out)&&(i<count);++i,j<<=1)
					{
						if(0==(se.mask&j))
						{
							const int	side=se.node->volume.Classify(normals[i],offsets[i],signs[i]);
							switch(side)
							{
							case	-1:	out=true;break;
							case	+1:	se.mask|=j;break;
							}
						}
					}
					if(out) continue;
				}
				if(policy.Descent(se.node))
				{
					if(se.node->isinternal())
					{
						const btDbvtNode* pns[]={	se.node->childs[0],se.node->childs[1]};
						sStkNPS		nes[]={	sStkNPS(pns[0],se.mask,pns[0]->volume.ProjectMinimum(sortaxis,srtsgns)),
							sStkNPS(pns[1],se.mask,pns[1]->volume.ProjectMinimum(sortaxis,srtsgns))};
						const int	q=nes[0].value<nes[1].value?1:0;				
						int			j=stack.size();
						if(fsort&&(j>0))
						{
							/* Insert 0	*/ 
							j=nearest(&stack[0],&stock[0],nes[q].value,0,stack.size());
							stack.push_back(0);
							
							//void * memmove ( void * destination, const void * source, size_t num );
							
#if DBVT_USE_MEMMOVE
                     {
                     int num_items_to_move = stack.size()-1-j;
                     if(num_items_to_move > 0)
                        memmove(&stack[j+1],&stack[j],sizeof(int)*num_items_to_move);
                     }
#else
                     for(int k=stack.size()-1;k>j;--k) {
								stack[k]=stack[k-1];
                     }
#endif
							stack[j]=allocate(ifree,stock,nes[q]);
							/* Insert 1	*/ 
							j=nearest(&stack[0],&stock[0],nes[1-q].value,j,stack.size());
							stack.push_back(0);
#if DBVT_USE_MEMMOVE
                     {
                     int num_items_to_move = stack.size()-1-j;
                     if(num_items_to_move > 0)
                        memmove(&stack[j+1],&stack[j],sizeof(int)*num_items_to_move);
                     }
#else
                     for(int k=stack.size()-1;k>j;--k) {
                        stack[k]=stack[k-1];
                     }
#endif
							stack[j]=allocate(ifree,stock,nes[1-q]);
						}
						else
						{
							stack.push_back(allocate(ifree,stock,nes[q]));
							stack.push_back(allocate(ifree,stock,nes[1-q]));
						}
					}
					else
					{
						policy.Process(se.node,se.value);
					}
				}
			} while(stack.size());
		}
}

//
DBVT_PREFIX
inline void		btDbvt::collideTU(	const btDbvtNode* root,
								  DBVT_IPOLICY)
{
	DBVT_CHECKTYPE
		if(root)
		{
			btAlignedObjectArray<const btDbvtNode*>	stack;
			stack.reserve(SIMPLE_STACKSIZE);
			stack.push_back(root);
			do	{
				const btDbvtNode*	n=stack[stack.size()-1];
				stack.pop_back();
				if(policy.Descent(n))
				{
					if(n->isinternal())
					{ stack.push_back(n->childs[0]);stack.push_back(n->childs[1]); }
					else
					{ policy.Process(n); }
				}
			} while(stack.size()>0);
		}
}

//
// PP Cleanup
//

#undef DBVT_USE_MEMMOVE
#undef DBVT_USE_TEMPLATE
#undef DBVT_VIRTUAL_DTOR
#undef DBVT_VIRTUAL
#undef DBVT_PREFIX
#undef DBVT_IPOLICY
#undef DBVT_CHECKTYPE
#undef DBVT_IMPL_GENERIC
#undef DBVT_IMPL_SSE
#undef DBVT_USE_INTRINSIC_SSE
#undef DBVT_SELECT_IMPL
#undef DBVT_MERGE_IMPL
#undef DBVT_INT0_IMPL

#endif

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
///b3DynamicBvh implementation by Nathanael Presson

#ifndef B3_DYNAMIC_BOUNDING_VOLUME_TREE_H
#define B3_DYNAMIC_BOUNDING_VOLUME_TREE_H

#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3Transform.h"
#include "Bullet3Geometry/b3AabbUtil.h"

//
// Compile time configuration
//


// Implementation profiles
#define B3_DBVT_IMPL_GENERIC		0	// Generic implementation	
#define B3_DBVT_IMPL_SSE			1	// SSE

// Template implementation of ICollide
#ifdef _WIN32
#if (defined (_MSC_VER) && _MSC_VER >= 1400)
#define	B3_DBVT_USE_TEMPLATE		1
#else
#define	B3_DBVT_USE_TEMPLATE		0
#endif
#else
#define	B3_DBVT_USE_TEMPLATE		0
#endif

// Use only intrinsics instead of inline asm
#define B3_DBVT_USE_INTRINSIC_SSE	1

// Using memmov for collideOCL
#define B3_DBVT_USE_MEMMOVE		1

// Enable benchmarking code
#define	B3_DBVT_ENABLE_BENCHMARK	0

// Inlining
#define B3_DBVT_INLINE				B3_FORCE_INLINE

// Specific methods implementation

//SSE gives errors on a MSVC 7.1
#if defined (B3_USE_SSE) //&& defined (_WIN32)
#define B3_DBVT_SELECT_IMPL		B3_DBVT_IMPL_SSE
#define B3_DBVT_MERGE_IMPL			B3_DBVT_IMPL_SSE
#define B3_DBVT_INT0_IMPL			B3_DBVT_IMPL_SSE
#else
#define B3_DBVT_SELECT_IMPL		B3_DBVT_IMPL_GENERIC
#define B3_DBVT_MERGE_IMPL			B3_DBVT_IMPL_GENERIC
#define B3_DBVT_INT0_IMPL			B3_DBVT_IMPL_GENERIC
#endif

#if	(B3_DBVT_SELECT_IMPL==B3_DBVT_IMPL_SSE)||	\
	(B3_DBVT_MERGE_IMPL==B3_DBVT_IMPL_SSE)||	\
	(B3_DBVT_INT0_IMPL==B3_DBVT_IMPL_SSE)
#include <emmintrin.h>
#endif

//
// Auto config and checks
//

#if B3_DBVT_USE_TEMPLATE
#define	B3_DBVT_VIRTUAL
#define B3_DBVT_VIRTUAL_DTOR(a)
#define B3_DBVT_PREFIX					template <typename T>
#define B3_DBVT_IPOLICY				T& policy
#define B3_DBVT_CHECKTYPE				static const ICollide&	typechecker=*(T*)1;(void)typechecker;
#else
#define	B3_DBVT_VIRTUAL_DTOR(a)		virtual ~a() {}
#define B3_DBVT_VIRTUAL				virtual
#define B3_DBVT_PREFIX
#define B3_DBVT_IPOLICY				ICollide& policy
#define B3_DBVT_CHECKTYPE
#endif

#if B3_DBVT_USE_MEMMOVE
#if !defined( __CELLOS_LV2__) && !defined(__MWERKS__)
#include <memory.h>
#endif
#include <string.h>
#endif

#ifndef B3_DBVT_USE_TEMPLATE
#error "B3_DBVT_USE_TEMPLATE undefined"
#endif

#ifndef B3_DBVT_USE_MEMMOVE
#error "B3_DBVT_USE_MEMMOVE undefined"
#endif

#ifndef B3_DBVT_ENABLE_BENCHMARK
#error "B3_DBVT_ENABLE_BENCHMARK undefined"
#endif

#ifndef B3_DBVT_SELECT_IMPL
#error "B3_DBVT_SELECT_IMPL undefined"
#endif

#ifndef B3_DBVT_MERGE_IMPL
#error "B3_DBVT_MERGE_IMPL undefined"
#endif

#ifndef B3_DBVT_INT0_IMPL
#error "B3_DBVT_INT0_IMPL undefined"
#endif

//
// Defaults volumes
//

/* b3DbvtAabbMm			*/ 
struct	b3DbvtAabbMm
{
	B3_DBVT_INLINE b3Vector3			Center() const	{ return((mi+mx)/2); }
	B3_DBVT_INLINE b3Vector3			Lengths() const	{ return(mx-mi); }
	B3_DBVT_INLINE b3Vector3			Extents() const	{ return((mx-mi)/2); }
	B3_DBVT_INLINE const b3Vector3&	Mins() const	{ return(mi); }
	B3_DBVT_INLINE const b3Vector3&	Maxs() const	{ return(mx); }
	static inline b3DbvtAabbMm		FromCE(const b3Vector3& c,const b3Vector3& e);
	static inline b3DbvtAabbMm		FromCR(const b3Vector3& c,b3Scalar r);
	static inline b3DbvtAabbMm		FromMM(const b3Vector3& mi,const b3Vector3& mx);
	static inline b3DbvtAabbMm		FromPoints(const b3Vector3* pts,int n);
	static inline b3DbvtAabbMm		FromPoints(const b3Vector3** ppts,int n);
	B3_DBVT_INLINE void				Expand(const b3Vector3& e);
	B3_DBVT_INLINE void				SignedExpand(const b3Vector3& e);
	B3_DBVT_INLINE bool				Contain(const b3DbvtAabbMm& a) const;
	B3_DBVT_INLINE int					Classify(const b3Vector3& n,b3Scalar o,int s) const;
	B3_DBVT_INLINE b3Scalar			ProjectMinimum(const b3Vector3& v,unsigned signs) const;
	B3_DBVT_INLINE friend bool			b3Intersect(	const b3DbvtAabbMm& a,
		const b3DbvtAabbMm& b);
	
	B3_DBVT_INLINE friend bool			b3Intersect(	const b3DbvtAabbMm& a,
		const b3Vector3& b);

	B3_DBVT_INLINE friend b3Scalar		b3Proximity(	const b3DbvtAabbMm& a,
		const b3DbvtAabbMm& b);
	B3_DBVT_INLINE friend int			b3Select(		const b3DbvtAabbMm& o,
		const b3DbvtAabbMm& a,
		const b3DbvtAabbMm& b);
	B3_DBVT_INLINE friend void			b3Merge(		const b3DbvtAabbMm& a,
		const b3DbvtAabbMm& b,
		b3DbvtAabbMm& r);
	B3_DBVT_INLINE friend bool			b3NotEqual(	const b3DbvtAabbMm& a,
		const b3DbvtAabbMm& b);
    
    B3_DBVT_INLINE b3Vector3&	tMins()	{ return(mi); }
	B3_DBVT_INLINE b3Vector3&	tMaxs()	{ return(mx); }
    
private:
	B3_DBVT_INLINE void				AddSpan(const b3Vector3& d,b3Scalar& smi,b3Scalar& smx) const;
private:
	b3Vector3	mi,mx;
};

// Types	
typedef	b3DbvtAabbMm	b3DbvtVolume;

/* b3DbvtNode				*/ 
struct	b3DbvtNode
{
	b3DbvtVolume	volume;
	b3DbvtNode*		parent;
	B3_DBVT_INLINE bool	isleaf() const		{ return(childs[1]==0); }
	B3_DBVT_INLINE bool	isinternal() const	{ return(!isleaf()); }
	union
	{
		b3DbvtNode*	childs[2];
		void*	data;
		int		dataAsInt;
	};
};

///The b3DynamicBvh class implements a fast dynamic bounding volume tree based on axis aligned bounding boxes (aabb tree).
///This b3DynamicBvh is used for soft body collision detection and for the b3DynamicBvhBroadphase. It has a fast insert, remove and update of nodes.
///Unlike the b3QuantizedBvh, nodes can be dynamically moved around, which allows for change in topology of the underlying data structure.
struct	b3DynamicBvh
{
	/* Stack element	*/ 
	struct	sStkNN
	{
		const b3DbvtNode*	a;
		const b3DbvtNode*	b;
		sStkNN() {}
		sStkNN(const b3DbvtNode* na,const b3DbvtNode* nb) : a(na),b(nb) {}
	};
	struct	sStkNP
	{
		const b3DbvtNode*	node;
		int			mask;
		sStkNP(const b3DbvtNode* n,unsigned m) : node(n),mask(m) {}
	};
	struct	sStkNPS
	{
		const b3DbvtNode*	node;
		int			mask;
		b3Scalar	value;
		sStkNPS() {}
		sStkNPS(const b3DbvtNode* n,unsigned m,b3Scalar v) : node(n),mask(m),value(v) {}
	};
	struct	sStkCLN
	{
		const b3DbvtNode*	node;
		b3DbvtNode*		parent;
		sStkCLN(const b3DbvtNode* n,b3DbvtNode* p) : node(n),parent(p) {}
	};
	// Policies/Interfaces

	/* ICollide	*/ 
	struct	ICollide
	{		
		B3_DBVT_VIRTUAL_DTOR(ICollide)
			B3_DBVT_VIRTUAL void	Process(const b3DbvtNode*,const b3DbvtNode*)		{}
		B3_DBVT_VIRTUAL void	Process(const b3DbvtNode*)					{}
		B3_DBVT_VIRTUAL void	Process(const b3DbvtNode* n,b3Scalar)			{ Process(n); }
		B3_DBVT_VIRTUAL bool	Descent(const b3DbvtNode*)					{ return(true); }
		B3_DBVT_VIRTUAL bool	AllLeaves(const b3DbvtNode*)					{ return(true); }
	};
	/* IWriter	*/ 
	struct	IWriter
	{
		virtual ~IWriter() {}
		virtual void		Prepare(const b3DbvtNode* root,int numnodes)=0;
		virtual void		WriteNode(const b3DbvtNode*,int index,int parent,int child0,int child1)=0;
		virtual void		WriteLeaf(const b3DbvtNode*,int index,int parent)=0;
	};
	/* IClone	*/ 
	struct	IClone
	{
		virtual ~IClone()	{}
		virtual void		CloneLeaf(b3DbvtNode*) {}
	};

	// Constants
	enum	{
		B3_SIMPLE_STACKSIZE	=	64,
		B3_DOUBLE_STACKSIZE	=	B3_SIMPLE_STACKSIZE*2
	};

	// Fields
	b3DbvtNode*		m_root;
	b3DbvtNode*		m_free;
	int				m_lkhd;
	int				m_leaves;
	unsigned		m_opath;

	
	b3AlignedObjectArray<sStkNN>	m_stkStack;
	mutable b3AlignedObjectArray<const b3DbvtNode*>	m_rayTestStack;


	// Methods
	b3DynamicBvh();
	~b3DynamicBvh();
	void			clear();
	bool			empty() const { return(0==m_root); }
	void			optimizeBottomUp();
	void			optimizeTopDown(int bu_treshold=128);
	void			optimizeIncremental(int passes);
	b3DbvtNode*		insert(const b3DbvtVolume& box,void* data);
	void			update(b3DbvtNode* leaf,int lookahead=-1);
	void			update(b3DbvtNode* leaf,b3DbvtVolume& volume);
	bool			update(b3DbvtNode* leaf,b3DbvtVolume& volume,const b3Vector3& velocity,b3Scalar margin);
	bool			update(b3DbvtNode* leaf,b3DbvtVolume& volume,const b3Vector3& velocity);
	bool			update(b3DbvtNode* leaf,b3DbvtVolume& volume,b3Scalar margin);	
	void			remove(b3DbvtNode* leaf);
	void			write(IWriter* iwriter) const;
	void			clone(b3DynamicBvh& dest,IClone* iclone=0) const;
	static int		maxdepth(const b3DbvtNode* node);
	static int		countLeaves(const b3DbvtNode* node);
	static void		extractLeaves(const b3DbvtNode* node,b3AlignedObjectArray<const b3DbvtNode*>& leaves);
#if B3_DBVT_ENABLE_BENCHMARK
	static void		benchmark();
#else
	static void		benchmark(){}
#endif
	// B3_DBVT_IPOLICY must support ICollide policy/interface
	B3_DBVT_PREFIX
		static void		enumNodes(	const b3DbvtNode* root,
		B3_DBVT_IPOLICY);
	B3_DBVT_PREFIX
		static void		enumLeaves(	const b3DbvtNode* root,
		B3_DBVT_IPOLICY);
	B3_DBVT_PREFIX
		void		collideTT(	const b3DbvtNode* root0,
		const b3DbvtNode* root1,
		B3_DBVT_IPOLICY);

	B3_DBVT_PREFIX
		void		collideTTpersistentStack(	const b3DbvtNode* root0,
		  const b3DbvtNode* root1,
		  B3_DBVT_IPOLICY);
#if 0
	B3_DBVT_PREFIX
		void		collideTT(	const b3DbvtNode* root0,
		const b3DbvtNode* root1,
		const b3Transform& xform,
		B3_DBVT_IPOLICY);
	B3_DBVT_PREFIX
		void		collideTT(	const b3DbvtNode* root0,
		const b3Transform& xform0,
		const b3DbvtNode* root1,
		const b3Transform& xform1,
		B3_DBVT_IPOLICY);
#endif

	B3_DBVT_PREFIX
		void		collideTV(	const b3DbvtNode* root,
		const b3DbvtVolume& volume,
		B3_DBVT_IPOLICY) const;
	///rayTest is a re-entrant ray test, and can be called in parallel as long as the b3AlignedAlloc is thread-safe (uses locking etc)
	///rayTest is slower than rayTestInternal, because it builds a local stack, using memory allocations, and it recomputes signs/rayDirectionInverses each time
	B3_DBVT_PREFIX
		static void		rayTest(	const b3DbvtNode* root,
		const b3Vector3& rayFrom,
		const b3Vector3& rayTo,
		B3_DBVT_IPOLICY);
	///rayTestInternal is faster than rayTest, because it uses a persistent stack (to reduce dynamic memory allocations to a minimum) and it uses precomputed signs/rayInverseDirections
	///rayTestInternal is used by b3DynamicBvhBroadphase to accelerate world ray casts
	B3_DBVT_PREFIX
		void		rayTestInternal(	const b3DbvtNode* root,
								const b3Vector3& rayFrom,
								const b3Vector3& rayTo,
								const b3Vector3& rayDirectionInverse,
								unsigned int signs[3],
								b3Scalar lambda_max,
								const b3Vector3& aabbMin,
								const b3Vector3& aabbMax,
								B3_DBVT_IPOLICY) const;

	B3_DBVT_PREFIX
		static void		collideKDOP(const b3DbvtNode* root,
		const b3Vector3* normals,
		const b3Scalar* offsets,
		int count,
		B3_DBVT_IPOLICY);
	B3_DBVT_PREFIX
		static void		collideOCL(	const b3DbvtNode* root,
		const b3Vector3* normals,
		const b3Scalar* offsets,
		const b3Vector3& sortaxis,
		int count,								
		B3_DBVT_IPOLICY,
		bool fullsort=true);
	B3_DBVT_PREFIX
		static void		collideTU(	const b3DbvtNode* root,
		B3_DBVT_IPOLICY);
	// Helpers	
	static B3_DBVT_INLINE int	nearest(const int* i,const b3DynamicBvh::sStkNPS* a,b3Scalar v,int l,int h)
	{
		int	m=0;
		while(l<h)
		{
			m=(l+h)>>1;
			if(a[i[m]].value>=v) l=m+1; else h=m;
		}
		return(h);
	}
	static B3_DBVT_INLINE int	allocate(	b3AlignedObjectArray<int>& ifree,
		b3AlignedObjectArray<sStkNPS>& stock,
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
	b3DynamicBvh(const b3DynamicBvh&)	{}	
};

//
// Inline's
//

//
inline b3DbvtAabbMm			b3DbvtAabbMm::FromCE(const b3Vector3& c,const b3Vector3& e)
{
	b3DbvtAabbMm box;
	box.mi=c-e;box.mx=c+e;
	return(box);
}

//
inline b3DbvtAabbMm			b3DbvtAabbMm::FromCR(const b3Vector3& c,b3Scalar r)
{
	return(FromCE(c,b3MakeVector3(r,r,r)));
}

//
inline b3DbvtAabbMm			b3DbvtAabbMm::FromMM(const b3Vector3& mi,const b3Vector3& mx)
{
	b3DbvtAabbMm box;
	box.mi=mi;box.mx=mx;
	return(box);
}

//
inline b3DbvtAabbMm			b3DbvtAabbMm::FromPoints(const b3Vector3* pts,int n)
{
	b3DbvtAabbMm box;
	box.mi=box.mx=pts[0];
	for(int i=1;i<n;++i)
	{
		box.mi.setMin(pts[i]);
		box.mx.setMax(pts[i]);
	}
	return(box);
}

//
inline b3DbvtAabbMm			b3DbvtAabbMm::FromPoints(const b3Vector3** ppts,int n)
{
	b3DbvtAabbMm box;
	box.mi=box.mx=*ppts[0];
	for(int i=1;i<n;++i)
	{
		box.mi.setMin(*ppts[i]);
		box.mx.setMax(*ppts[i]);
	}
	return(box);
}

//
B3_DBVT_INLINE void		b3DbvtAabbMm::Expand(const b3Vector3& e)
{
	mi-=e;mx+=e;
}

//
B3_DBVT_INLINE void		b3DbvtAabbMm::SignedExpand(const b3Vector3& e)
{
	if(e.x>0) mx.setX(mx.x+e[0]); else mi.setX(mi.x+e[0]);
	if(e.y>0) mx.setY(mx.y+e[1]); else mi.setY(mi.y+e[1]);
	if(e.z>0) mx.setZ(mx.z+e[2]); else mi.setZ(mi.z+e[2]);
}

//
B3_DBVT_INLINE bool		b3DbvtAabbMm::Contain(const b3DbvtAabbMm& a) const
{
	return(	(mi.x<=a.mi.x)&&
		(mi.y<=a.mi.y)&&
		(mi.z<=a.mi.z)&&
		(mx.x>=a.mx.x)&&
		(mx.y>=a.mx.y)&&
		(mx.z>=a.mx.z));
}

//
B3_DBVT_INLINE int		b3DbvtAabbMm::Classify(const b3Vector3& n,b3Scalar o,int s) const
{
	b3Vector3			pi,px;
	switch(s)
	{
	case	(0+0+0):	px=b3MakeVector3(mi.x,mi.y,mi.z);
		pi=b3MakeVector3(mx.x,mx.y,mx.z);break;
	case	(1+0+0):	px=b3MakeVector3(mx.x,mi.y,mi.z);
		pi=b3MakeVector3(mi.x,mx.y,mx.z);break;
	case	(0+2+0):	px=b3MakeVector3(mi.x,mx.y,mi.z);
		pi=b3MakeVector3(mx.x,mi.y,mx.z);break;
	case	(1+2+0):	px=b3MakeVector3(mx.x,mx.y,mi.z);
		pi=b3MakeVector3(mi.x,mi.y,mx.z);break;
	case	(0+0+4):	px=b3MakeVector3(mi.x,mi.y,mx.z);
		pi=b3MakeVector3(mx.x,mx.y,mi.z);break;
	case	(1+0+4):	px=b3MakeVector3(mx.x,mi.y,mx.z);
		pi=b3MakeVector3(mi.x,mx.y,mi.z);break;
	case	(0+2+4):	px=b3MakeVector3(mi.x,mx.y,mx.z);
		pi=b3MakeVector3(mx.x,mi.y,mi.z);break;
	case	(1+2+4):	px=b3MakeVector3(mx.x,mx.y,mx.z);
		pi=b3MakeVector3(mi.x,mi.y,mi.z);break;
	}
	if((b3Dot(n,px)+o)<0)		return(-1);
	if((b3Dot(n,pi)+o)>=0)	return(+1);
	return(0);
}

//
B3_DBVT_INLINE b3Scalar	b3DbvtAabbMm::ProjectMinimum(const b3Vector3& v,unsigned signs) const
{
	const b3Vector3*	b[]={&mx,&mi};
	const b3Vector3		p = b3MakeVector3(	b[(signs>>0)&1]->x,
		b[(signs>>1)&1]->y,
		b[(signs>>2)&1]->z);
	return(b3Dot(p,v));
}

//
B3_DBVT_INLINE void		b3DbvtAabbMm::AddSpan(const b3Vector3& d,b3Scalar& smi,b3Scalar& smx) const
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
B3_DBVT_INLINE bool		b3Intersect(	const b3DbvtAabbMm& a,
								  const b3DbvtAabbMm& b)
{
#if	B3_DBVT_INT0_IMPL == B3_DBVT_IMPL_SSE
	const __m128	rt(_mm_or_ps(	_mm_cmplt_ps(_mm_load_ps(b.mx),_mm_load_ps(a.mi)),
		_mm_cmplt_ps(_mm_load_ps(a.mx),_mm_load_ps(b.mi))));
#if defined (_WIN32)
	const __int32*	pu((const __int32*)&rt);
#else
    const int*	pu((const int*)&rt);
#endif
	return((pu[0]|pu[1]|pu[2])==0);
#else
	return(	(a.mi.x<=b.mx.x)&&
		(a.mx.x>=b.mi.x)&&
		(a.mi.y<=b.mx.y)&&
		(a.mx.y>=b.mi.y)&&
		(a.mi.z<=b.mx.z)&&		
		(a.mx.z>=b.mi.z));
#endif
}



//
B3_DBVT_INLINE bool		b3Intersect(	const b3DbvtAabbMm& a,
								  const b3Vector3& b)
{
	return(	(b.x>=a.mi.x)&&
		(b.y>=a.mi.y)&&
		(b.z>=a.mi.z)&&
		(b.x<=a.mx.x)&&
		(b.y<=a.mx.y)&&
		(b.z<=a.mx.z));
}





//////////////////////////////////////


//
B3_DBVT_INLINE b3Scalar	b3Proximity(	const b3DbvtAabbMm& a,
								  const b3DbvtAabbMm& b)
{
	const b3Vector3	d=(a.mi+a.mx)-(b.mi+b.mx);
	return(b3Fabs(d.x)+b3Fabs(d.y)+b3Fabs(d.z));
}



//
B3_DBVT_INLINE int			b3Select(	const b3DbvtAabbMm& o,
							   const b3DbvtAabbMm& a,
							   const b3DbvtAabbMm& b)
{
#if	B3_DBVT_SELECT_IMPL == B3_DBVT_IMPL_SSE
    
#if defined (_WIN32)
	static B3_ATTRIBUTE_ALIGNED16(const unsigned __int32)	mask[]={0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff};
#else
    static B3_ATTRIBUTE_ALIGNED16(const unsigned int)	mask[]={0x7fffffff,0x7fffffff,0x7fffffff,0x00000000 /*0x7fffffff*/};
#endif
	///@todo: the intrinsic version is 11% slower
#if B3_DBVT_USE_INTRINSIC_SSE

	union b3SSEUnion ///NOTE: if we use more intrinsics, move b3SSEUnion into the LinearMath directory
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
	
	b3SSEUnion tmp;
	tmp.ssereg = _mm_cmple_ss(bmi,ami);
	return tmp.ints[0]&1;

#else
	B3_ATTRIBUTE_ALIGNED16(__int32	r[1]);
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
	return(b3Proximity(o,a)<b3Proximity(o,b)?0:1);
#endif
}

//
B3_DBVT_INLINE void		b3Merge(	const b3DbvtAabbMm& a,
							  const b3DbvtAabbMm& b,
							  b3DbvtAabbMm& r)
{
#if B3_DBVT_MERGE_IMPL==B3_DBVT_IMPL_SSE
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
B3_DBVT_INLINE bool		b3NotEqual(	const b3DbvtAabbMm& a,
								 const b3DbvtAabbMm& b)
{
	return(	(a.mi.x!=b.mi.x)||
		(a.mi.y!=b.mi.y)||
		(a.mi.z!=b.mi.z)||
		(a.mx.x!=b.mx.x)||
		(a.mx.y!=b.mx.y)||
		(a.mx.z!=b.mx.z));
}

//
// Inline's
//

//
B3_DBVT_PREFIX
inline void		b3DynamicBvh::enumNodes(	const b3DbvtNode* root,
								  B3_DBVT_IPOLICY)
{
	B3_DBVT_CHECKTYPE
		policy.Process(root);
	if(root->isinternal())
	{
		enumNodes(root->childs[0],policy);
		enumNodes(root->childs[1],policy);
	}
}

//
B3_DBVT_PREFIX
inline void		b3DynamicBvh::enumLeaves(	const b3DbvtNode* root,
								   B3_DBVT_IPOLICY)
{
	B3_DBVT_CHECKTYPE
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
B3_DBVT_PREFIX
inline void		b3DynamicBvh::collideTT(	const b3DbvtNode* root0,
								  const b3DbvtNode* root1,
								  B3_DBVT_IPOLICY)
{
	B3_DBVT_CHECKTYPE
		if(root0&&root1)
		{
			int								depth=1;
			int								treshold=B3_DOUBLE_STACKSIZE-4;
			b3AlignedObjectArray<sStkNN>	stkStack;
			stkStack.resize(B3_DOUBLE_STACKSIZE);
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
				else if(b3Intersect(p.a->volume,p.b->volume))
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



B3_DBVT_PREFIX
inline void		b3DynamicBvh::collideTTpersistentStack(	const b3DbvtNode* root0,
								  const b3DbvtNode* root1,
								  B3_DBVT_IPOLICY)
{
	B3_DBVT_CHECKTYPE
		if(root0&&root1)
		{
			int								depth=1;
			int								treshold=B3_DOUBLE_STACKSIZE-4;
			
			m_stkStack.resize(B3_DOUBLE_STACKSIZE);
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
				else if(b3Intersect(p.a->volume,p.b->volume))
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
B3_DBVT_PREFIX
inline void		b3DynamicBvh::collideTT(	const b3DbvtNode* root0,
								  const b3DbvtNode* root1,
								  const b3Transform& xform,
								  B3_DBVT_IPOLICY)
{
	B3_DBVT_CHECKTYPE
		if(root0&&root1)
		{
			int								depth=1;
			int								treshold=B3_DOUBLE_STACKSIZE-4;
			b3AlignedObjectArray<sStkNN>	stkStack;
			stkStack.resize(B3_DOUBLE_STACKSIZE);
			stkStack[0]=sStkNN(root0,root1);
			do	{
				sStkNN	p=stkStack[--depth];
				if(b3Intersect(p.a->volume,p.b->volume,xform))
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
B3_DBVT_PREFIX
inline void		b3DynamicBvh::collideTT(	const b3DbvtNode* root0,
								  const b3Transform& xform0,
								  const b3DbvtNode* root1,
								  const b3Transform& xform1,
								  B3_DBVT_IPOLICY)
{
	const b3Transform	xform=xform0.inverse()*xform1;
	collideTT(root0,root1,xform,policy);
}
#endif 

//
B3_DBVT_PREFIX
inline void		b3DynamicBvh::collideTV(	const b3DbvtNode* root,
								  const b3DbvtVolume& vol,
								  B3_DBVT_IPOLICY) const
{
	B3_DBVT_CHECKTYPE
		if(root)
		{
			B3_ATTRIBUTE_ALIGNED16(b3DbvtVolume)		volume(vol);
			b3AlignedObjectArray<const b3DbvtNode*>	stack;
			stack.resize(0);
			stack.reserve(B3_SIMPLE_STACKSIZE);
			stack.push_back(root);
			do	{
				const b3DbvtNode*	n=stack[stack.size()-1];
				stack.pop_back();
				if(b3Intersect(n->volume,volume))
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

B3_DBVT_PREFIX
inline void		b3DynamicBvh::rayTestInternal(	const b3DbvtNode* root,
								const b3Vector3& rayFrom,
								const b3Vector3& rayTo,
								const b3Vector3& rayDirectionInverse,
								unsigned int signs[3],
								b3Scalar lambda_max,
								const b3Vector3& aabbMin,
								const b3Vector3& aabbMax,
								B3_DBVT_IPOLICY) const
{
        (void) rayTo;
	B3_DBVT_CHECKTYPE
	if(root)
	{
		int								depth=1;
		int								treshold=B3_DOUBLE_STACKSIZE-2;
		b3AlignedObjectArray<const b3DbvtNode*>&	stack = m_rayTestStack;
		stack.resize(B3_DOUBLE_STACKSIZE);
		stack[0]=root;
		b3Vector3 bounds[2];
		do	
		{
			const b3DbvtNode*	node=stack[--depth];
			bounds[0] = node->volume.Mins()-aabbMax;
			bounds[1] = node->volume.Maxs()-aabbMin;
			b3Scalar tmin=1.f,lambda_min=0.f;
			unsigned int result1=false;
			result1 = b3RayAabb2(rayFrom,rayDirectionInverse,signs,bounds,tmin,lambda_min,lambda_max);
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
B3_DBVT_PREFIX
inline void		b3DynamicBvh::rayTest(	const b3DbvtNode* root,
								const b3Vector3& rayFrom,
								const b3Vector3& rayTo,
								B3_DBVT_IPOLICY)
{
	B3_DBVT_CHECKTYPE
		if(root)
		{
			b3Vector3 rayDir = (rayTo-rayFrom);
			rayDir.normalize ();

			///what about division by zero? --> just set rayDirection[i] to INF/B3_LARGE_FLOAT
			b3Vector3 rayDirectionInverse;
			rayDirectionInverse[0] = rayDir[0] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDir[0];
			rayDirectionInverse[1] = rayDir[1] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDir[1];
			rayDirectionInverse[2] = rayDir[2] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDir[2];
			unsigned int signs[3] = { rayDirectionInverse[0] < 0.0, rayDirectionInverse[1] < 0.0, rayDirectionInverse[2] < 0.0};

			b3Scalar lambda_max = rayDir.dot(rayTo-rayFrom);
#ifdef COMPARE_BTRAY_AABB2
			b3Vector3 resultNormal;
#endif//COMPARE_BTRAY_AABB2
			
			b3AlignedObjectArray<const b3DbvtNode*>	stack;

			int								depth=1;
			int								treshold=B3_DOUBLE_STACKSIZE-2;

			stack.resize(B3_DOUBLE_STACKSIZE);
			stack[0]=root;
			b3Vector3 bounds[2];
			do	{
				const b3DbvtNode*	node=stack[--depth];

				bounds[0] = node->volume.Mins();
				bounds[1] = node->volume.Maxs();
				
				b3Scalar tmin=1.f,lambda_min=0.f;
				unsigned int result1 = b3RayAabb2(rayFrom,rayDirectionInverse,signs,bounds,tmin,lambda_min,lambda_max);

#ifdef COMPARE_BTRAY_AABB2
				b3Scalar param=1.f;
				bool result2 = b3RayAabb(rayFrom,rayTo,node->volume.Mins(),node->volume.Maxs(),param,resultNormal);
				b3Assert(result1 == result2);
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
B3_DBVT_PREFIX
inline void		b3DynamicBvh::collideKDOP(const b3DbvtNode* root,
									const b3Vector3* normals,
									const b3Scalar* offsets,
									int count,
									B3_DBVT_IPOLICY)
{
	B3_DBVT_CHECKTYPE
		if(root)
		{
			const int						inside=(1<<count)-1;
			b3AlignedObjectArray<sStkNP>	stack;
			int								signs[sizeof(unsigned)*8];
			b3Assert(count<int (sizeof(signs)/sizeof(signs[0])));
			for(int i=0;i<count;++i)
			{
				signs[i]=	((normals[i].x>=0)?1:0)+
					((normals[i].y>=0)?2:0)+
					((normals[i].z>=0)?4:0);
			}
			stack.reserve(B3_SIMPLE_STACKSIZE);
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
B3_DBVT_PREFIX
inline void		b3DynamicBvh::collideOCL(	const b3DbvtNode* root,
								   const b3Vector3* normals,
								   const b3Scalar* offsets,
								   const b3Vector3& sortaxis,
								   int count,
								   B3_DBVT_IPOLICY,
								   bool fsort)
{
	B3_DBVT_CHECKTYPE
		if(root)
		{
			const unsigned					srtsgns=(sortaxis[0]>=0?1:0)+
				(sortaxis[1]>=0?2:0)+
				(sortaxis[2]>=0?4:0);
			const int						inside=(1<<count)-1;
			b3AlignedObjectArray<sStkNPS>	stock;
			b3AlignedObjectArray<int>		ifree;
			b3AlignedObjectArray<int>		stack;
			int								signs[sizeof(unsigned)*8];
			b3Assert(count<int (sizeof(signs)/sizeof(signs[0])));
			for(int i=0;i<count;++i)
			{
				signs[i]=	((normals[i].x>=0)?1:0)+
					((normals[i].y>=0)?2:0)+
					((normals[i].z>=0)?4:0);
			}
			stock.reserve(B3_SIMPLE_STACKSIZE);
			stack.reserve(B3_SIMPLE_STACKSIZE);
			ifree.reserve(B3_SIMPLE_STACKSIZE);
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
						const b3DbvtNode* pns[]={	se.node->childs[0],se.node->childs[1]};
						sStkNPS		nes[]={	sStkNPS(pns[0],se.mask,pns[0]->volume.ProjectMinimum(sortaxis,srtsgns)),
							sStkNPS(pns[1],se.mask,pns[1]->volume.ProjectMinimum(sortaxis,srtsgns))};
						const int	q=nes[0].value<nes[1].value?1:0;				
						int			j=stack.size();
						if(fsort&&(j>0))
						{
							/* Insert 0	*/ 
							j=nearest(&stack[0],&stock[0],nes[q].value,0,stack.size());
							stack.push_back(0);
#if B3_DBVT_USE_MEMMOVE
							memmove(&stack[j+1],&stack[j],sizeof(int)*(stack.size()-j-1));
#else
							for(int k=stack.size()-1;k>j;--k) stack[k]=stack[k-1];
#endif
							stack[j]=allocate(ifree,stock,nes[q]);
							/* Insert 1	*/ 
							j=nearest(&stack[0],&stock[0],nes[1-q].value,j,stack.size());
							stack.push_back(0);
#if B3_DBVT_USE_MEMMOVE
							memmove(&stack[j+1],&stack[j],sizeof(int)*(stack.size()-j-1));
#else
							for(int k=stack.size()-1;k>j;--k) stack[k]=stack[k-1];
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
B3_DBVT_PREFIX
inline void		b3DynamicBvh::collideTU(	const b3DbvtNode* root,
								  B3_DBVT_IPOLICY)
{
	B3_DBVT_CHECKTYPE
		if(root)
		{
			b3AlignedObjectArray<const b3DbvtNode*>	stack;
			stack.reserve(B3_SIMPLE_STACKSIZE);
			stack.push_back(root);
			do	{
				const b3DbvtNode*	n=stack[stack.size()-1];
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

#undef B3_DBVT_USE_MEMMOVE
#undef B3_DBVT_USE_TEMPLATE
#undef B3_DBVT_VIRTUAL_DTOR
#undef B3_DBVT_VIRTUAL
#undef B3_DBVT_PREFIX
#undef B3_DBVT_IPOLICY
#undef B3_DBVT_CHECKTYPE
#undef B3_DBVT_IMPL_GENERIC
#undef B3_DBVT_IMPL_SSE
#undef B3_DBVT_USE_INTRINSIC_SSE
#undef B3_DBVT_SELECT_IMPL
#undef B3_DBVT_MERGE_IMPL
#undef B3_DBVT_INT0_IMPL

#endif

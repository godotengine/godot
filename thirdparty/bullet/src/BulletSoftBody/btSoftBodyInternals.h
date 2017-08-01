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
///btSoftBody implementation by Nathanael Presson

#ifndef _BT_SOFT_BODY_INTERNALS_H
#define _BT_SOFT_BODY_INTERNALS_H

#include "btSoftBody.h"


#include "LinearMath/btQuickprof.h"
#include "LinearMath/btPolarDecomposition.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h"
#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "BulletCollision/CollisionShapes/btConvexInternalShape.h"
#include "BulletCollision/NarrowPhaseCollision/btGjkEpa2.h"
#include <string.h> //for memset
//
// btSymMatrix
//
template <typename T>
struct btSymMatrix
{
	btSymMatrix() : dim(0)					{}
	btSymMatrix(int n,const T& init=T())	{ resize(n,init); }
	void					resize(int n,const T& init=T())			{ dim=n;store.resize((n*(n+1))/2,init); }
	int						index(int c,int r) const				{ if(c>r) btSwap(c,r);btAssert(r<dim);return((r*(r+1))/2+c); }
	T&						operator()(int c,int r)					{ return(store[index(c,r)]); }
	const T&				operator()(int c,int r) const			{ return(store[index(c,r)]); }
	btAlignedObjectArray<T>	store;
	int						dim;
};	

//
// btSoftBodyCollisionShape
//
class btSoftBodyCollisionShape : public btConcaveShape
{
public:
	btSoftBody*						m_body;

	btSoftBodyCollisionShape(btSoftBody* backptr)
	{
		m_shapeType = SOFTBODY_SHAPE_PROXYTYPE;
		m_body=backptr;
	}

	virtual ~btSoftBodyCollisionShape()
	{

	}

	void	processAllTriangles(btTriangleCallback* /*callback*/,const btVector3& /*aabbMin*/,const btVector3& /*aabbMax*/) const
	{
		//not yet
		btAssert(0);
	}

	///getAabb returns the axis aligned bounding box in the coordinate frame of the given transform t.
	virtual void getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const
	{
		/* t is usually identity, except when colliding against btCompoundShape. See Issue 512 */
		const btVector3	mins=m_body->m_bounds[0];
		const btVector3	maxs=m_body->m_bounds[1];
		const btVector3	crns[]={t*btVector3(mins.x(),mins.y(),mins.z()),
			t*btVector3(maxs.x(),mins.y(),mins.z()),
			t*btVector3(maxs.x(),maxs.y(),mins.z()),
			t*btVector3(mins.x(),maxs.y(),mins.z()),
			t*btVector3(mins.x(),mins.y(),maxs.z()),
			t*btVector3(maxs.x(),mins.y(),maxs.z()),
			t*btVector3(maxs.x(),maxs.y(),maxs.z()),
			t*btVector3(mins.x(),maxs.y(),maxs.z())};
		aabbMin=aabbMax=crns[0];
		for(int i=1;i<8;++i)
		{
			aabbMin.setMin(crns[i]);
			aabbMax.setMax(crns[i]);
		}
	}


	virtual void	setLocalScaling(const btVector3& /*scaling*/)
	{		
		///na
	}
	virtual const btVector3& getLocalScaling() const
	{
		static const btVector3 dummy(1,1,1);
		return dummy;
	}
	virtual void	calculateLocalInertia(btScalar /*mass*/,btVector3& /*inertia*/) const
	{
		///not yet
		btAssert(0);
	}
	virtual const char*	getName()const
	{
		return "SoftBody";
	}

};

//
// btSoftClusterCollisionShape
//
class btSoftClusterCollisionShape : public btConvexInternalShape
{
public:
	const btSoftBody::Cluster*	m_cluster;

	btSoftClusterCollisionShape (const btSoftBody::Cluster* cluster) : m_cluster(cluster) { setMargin(0); }


	virtual btVector3	localGetSupportingVertex(const btVector3& vec) const
	{
		btSoftBody::Node* const *						n=&m_cluster->m_nodes[0];
		btScalar										d=btDot(vec,n[0]->m_x);
		int												j=0;
		for(int i=1,ni=m_cluster->m_nodes.size();i<ni;++i)
		{
			const btScalar	k=btDot(vec,n[i]->m_x);
			if(k>d) { d=k;j=i; }
		}
		return(n[j]->m_x);
	}
	virtual btVector3	localGetSupportingVertexWithoutMargin(const btVector3& vec)const
	{
		return(localGetSupportingVertex(vec));
	}
	//notice that the vectors should be unit length
	virtual void	batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const
	{}


	virtual void	calculateLocalInertia(btScalar mass,btVector3& inertia) const
	{}

	virtual void getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const
	{}

	virtual int	getShapeType() const { return SOFTBODY_SHAPE_PROXYTYPE; }

	//debugging
	virtual const char*	getName()const {return "SOFTCLUSTER";}

	virtual void	setMargin(btScalar margin)
	{
		btConvexInternalShape::setMargin(margin);
	}
	virtual btScalar	getMargin() const
	{
		return btConvexInternalShape::getMargin();
	}
};

//
// Inline's
//

//
template <typename T>
static inline void			ZeroInitialize(T& value)
{
	memset(&value,0,sizeof(T));
}
//
template <typename T>
static inline bool			CompLess(const T& a,const T& b)
{ return(a<b); }
//
template <typename T>
static inline bool			CompGreater(const T& a,const T& b)
{ return(a>b); }
//
template <typename T>
static inline T				Lerp(const T& a,const T& b,btScalar t)
{ return(a+(b-a)*t); }
//
template <typename T>
static inline T				InvLerp(const T& a,const T& b,btScalar t)
{ return((b+a*t-b*t)/(a*b)); }
//
static inline btMatrix3x3	Lerp(	const btMatrix3x3& a,
								 const btMatrix3x3& b,
								 btScalar t)
{
	btMatrix3x3	r;
	r[0]=Lerp(a[0],b[0],t);
	r[1]=Lerp(a[1],b[1],t);
	r[2]=Lerp(a[2],b[2],t);
	return(r);
}
//
static inline btVector3		Clamp(const btVector3& v,btScalar maxlength)
{
	const btScalar sql=v.length2();
	if(sql>(maxlength*maxlength))
		return((v*maxlength)/btSqrt(sql));
	else
		return(v);
}
//
template <typename T>
static inline T				Clamp(const T& x,const T& l,const T& h)
{ return(x<l?l:x>h?h:x); }
//
template <typename T>
static inline T				Sq(const T& x)
{ return(x*x); }
//
template <typename T>
static inline T				Cube(const T& x)
{ return(x*x*x); }
//
template <typename T>
static inline T				Sign(const T& x)
{ return((T)(x<0?-1:+1)); }
//
template <typename T>
static inline bool			SameSign(const T& x,const T& y)
{ return((x*y)>0); }
//
static inline btScalar		ClusterMetric(const btVector3& x,const btVector3& y)
{
	const btVector3	d=x-y;
	return(btFabs(d[0])+btFabs(d[1])+btFabs(d[2]));
}
//
static inline btMatrix3x3	ScaleAlongAxis(const btVector3& a,btScalar s)
{
	const btScalar	xx=a.x()*a.x();
	const btScalar	yy=a.y()*a.y();
	const btScalar	zz=a.z()*a.z();
	const btScalar	xy=a.x()*a.y();
	const btScalar	yz=a.y()*a.z();
	const btScalar	zx=a.z()*a.x();
	btMatrix3x3		m;
	m[0]=btVector3(1-xx+xx*s,xy*s-xy,zx*s-zx);
	m[1]=btVector3(xy*s-xy,1-yy+yy*s,yz*s-yz);
	m[2]=btVector3(zx*s-zx,yz*s-yz,1-zz+zz*s);
	return(m);
}
//
static inline btMatrix3x3	Cross(const btVector3& v)
{
	btMatrix3x3	m;
	m[0]=btVector3(0,-v.z(),+v.y());
	m[1]=btVector3(+v.z(),0,-v.x());
	m[2]=btVector3(-v.y(),+v.x(),0);
	return(m);
}
//
static inline btMatrix3x3	Diagonal(btScalar x)
{
	btMatrix3x3	m;
	m[0]=btVector3(x,0,0);
	m[1]=btVector3(0,x,0);
	m[2]=btVector3(0,0,x);
	return(m);
}
//
static inline btMatrix3x3	Add(const btMatrix3x3& a,
								const btMatrix3x3& b)
{
	btMatrix3x3	r;
	for(int i=0;i<3;++i) r[i]=a[i]+b[i];
	return(r);
}
//
static inline btMatrix3x3	Sub(const btMatrix3x3& a,
								const btMatrix3x3& b)
{
	btMatrix3x3	r;
	for(int i=0;i<3;++i) r[i]=a[i]-b[i];
	return(r);
}
//
static inline btMatrix3x3	Mul(const btMatrix3x3& a,
								btScalar b)
{
	btMatrix3x3	r;
	for(int i=0;i<3;++i) r[i]=a[i]*b;
	return(r);
}
//
static inline void			Orthogonalize(btMatrix3x3& m)
{
	m[2]=btCross(m[0],m[1]).normalized();
	m[1]=btCross(m[2],m[0]).normalized();
	m[0]=btCross(m[1],m[2]).normalized();
}
//
static inline btMatrix3x3	MassMatrix(btScalar im,const btMatrix3x3& iwi,const btVector3& r)
{
	const btMatrix3x3	cr=Cross(r);
	return(Sub(Diagonal(im),cr*iwi*cr));
}

//
static inline btMatrix3x3	ImpulseMatrix(	btScalar dt,
										  btScalar ima,
										  btScalar imb,
										  const btMatrix3x3& iwi,
										  const btVector3& r)
{
	return(Diagonal(1/dt)*Add(Diagonal(ima),MassMatrix(imb,iwi,r)).inverse());
}

//
static inline btMatrix3x3	ImpulseMatrix(	btScalar ima,const btMatrix3x3& iia,const btVector3& ra,
										  btScalar imb,const btMatrix3x3& iib,const btVector3& rb)	
{
	return(Add(MassMatrix(ima,iia,ra),MassMatrix(imb,iib,rb)).inverse());
}

//
static inline btMatrix3x3	AngularImpulseMatrix(	const btMatrix3x3& iia,
												 const btMatrix3x3& iib)
{
	return(Add(iia,iib).inverse());
}

//
static inline btVector3		ProjectOnAxis(	const btVector3& v,
										  const btVector3& a)
{
	return(a*btDot(v,a));
}
//
static inline btVector3		ProjectOnPlane(	const btVector3& v,
										   const btVector3& a)
{
	return(v-ProjectOnAxis(v,a));
}

//
static inline void			ProjectOrigin(	const btVector3& a,
										  const btVector3& b,
										  btVector3& prj,
										  btScalar& sqd)
{
	const btVector3	d=b-a;
	const btScalar	m2=d.length2();
	if(m2>SIMD_EPSILON)
	{	
		const btScalar	t=Clamp<btScalar>(-btDot(a,d)/m2,0,1);
		const btVector3	p=a+d*t;
		const btScalar	l2=p.length2();
		if(l2<sqd)
		{
			prj=p;
			sqd=l2;
		}
	}
}
//
static inline void			ProjectOrigin(	const btVector3& a,
										  const btVector3& b,
										  const btVector3& c,
										  btVector3& prj,
										  btScalar& sqd)
{
	const btVector3&	q=btCross(b-a,c-a);
	const btScalar		m2=q.length2();
	if(m2>SIMD_EPSILON)
	{
		const btVector3	n=q/btSqrt(m2);
		const btScalar	k=btDot(a,n);
		const btScalar	k2=k*k;
		if(k2<sqd)
		{
			const btVector3	p=n*k;
			if(	(btDot(btCross(a-p,b-p),q)>0)&&
				(btDot(btCross(b-p,c-p),q)>0)&&
				(btDot(btCross(c-p,a-p),q)>0))
			{			
				prj=p;
				sqd=k2;
			}
			else
			{
				ProjectOrigin(a,b,prj,sqd);
				ProjectOrigin(b,c,prj,sqd);
				ProjectOrigin(c,a,prj,sqd);
			}
		}
	}
}

//
template <typename T>
static inline T				BaryEval(		const T& a,
									 const T& b,
									 const T& c,
									 const btVector3& coord)
{
	return(a*coord.x()+b*coord.y()+c*coord.z());
}
//
static inline btVector3		BaryCoord(	const btVector3& a,
									  const btVector3& b,
									  const btVector3& c,
									  const btVector3& p)
{
	const btScalar	w[]={	btCross(a-p,b-p).length(),
		btCross(b-p,c-p).length(),
		btCross(c-p,a-p).length()};
	const btScalar	isum=1/(w[0]+w[1]+w[2]);
	return(btVector3(w[1]*isum,w[2]*isum,w[0]*isum));
}

//
inline static btScalar				ImplicitSolve(	btSoftBody::ImplicitFn* fn,
										  const btVector3& a,
										  const btVector3& b,
										  const btScalar accuracy,
										  const int maxiterations=256)
{
	btScalar	span[2]={0,1};
	btScalar	values[2]={fn->Eval(a),fn->Eval(b)};
	if(values[0]>values[1])
	{
		btSwap(span[0],span[1]);
		btSwap(values[0],values[1]);
	}
	if(values[0]>-accuracy) return(-1);
	if(values[1]<+accuracy) return(-1);
	for(int i=0;i<maxiterations;++i)
	{
		const btScalar	t=Lerp(span[0],span[1],values[0]/(values[0]-values[1]));
		const btScalar	v=fn->Eval(Lerp(a,b,t));
		if((t<=0)||(t>=1))		break;
		if(btFabs(v)<accuracy)	return(t);
		if(v<0)
		{ span[0]=t;values[0]=v; }
		else
		{ span[1]=t;values[1]=v; }
	}
	return(-1);
}

inline static void					EvaluateMedium(	const btSoftBodyWorldInfo* wfi,
										   const btVector3& x,
										   btSoftBody::sMedium& medium)
{
	medium.m_velocity	=	btVector3(0,0,0);
	medium.m_pressure	=	0;
	medium.m_density	=	wfi->air_density;
	if(wfi->water_density>0)
	{
		const btScalar	depth=-(btDot(x,wfi->water_normal)+wfi->water_offset);
		if(depth>0)
		{
			medium.m_density	=	wfi->water_density;
			medium.m_pressure	=	depth*wfi->water_density*wfi->m_gravity.length();
		}
	}
}


//
static inline btVector3		NormalizeAny(const btVector3& v)
{
	const btScalar l=v.length();
	if(l>SIMD_EPSILON)
		return(v/l);
	else
		return(btVector3(0,0,0));
}

//
static inline btDbvtVolume	VolumeOf(	const btSoftBody::Face& f,
									 btScalar margin)
{
	const btVector3*	pts[]={	&f.m_n[0]->m_x,
		&f.m_n[1]->m_x,
		&f.m_n[2]->m_x};
	btDbvtVolume		vol=btDbvtVolume::FromPoints(pts,3);
	vol.Expand(btVector3(margin,margin,margin));
	return(vol);
}

//
static inline btVector3			CenterOf(	const btSoftBody::Face& f)
{
	return((f.m_n[0]->m_x+f.m_n[1]->m_x+f.m_n[2]->m_x)/3);
}

//
static inline btScalar			AreaOf(		const btVector3& x0,
									   const btVector3& x1,
									   const btVector3& x2)
{
	const btVector3	a=x1-x0;
	const btVector3	b=x2-x0;
	const btVector3	cr=btCross(a,b);
	const btScalar	area=cr.length();
	return(area);
}

//
static inline btScalar		VolumeOf(	const btVector3& x0,
									 const btVector3& x1,
									 const btVector3& x2,
									 const btVector3& x3)
{
	const btVector3	a=x1-x0;
	const btVector3	b=x2-x0;
	const btVector3	c=x3-x0;
	return(btDot(a,btCross(b,c)));
}

//


//
static inline void			ApplyClampedForce(	btSoftBody::Node& n,
											  const btVector3& f,
											  btScalar dt)
{
	const btScalar	dtim=dt*n.m_im;
	if((f*dtim).length2()>n.m_v.length2())
	{/* Clamp	*/ 
		n.m_f-=ProjectOnAxis(n.m_v,f.normalized())/dtim;						
	}
	else
	{/* Apply	*/ 
		n.m_f+=f;
	}
}

//
static inline int		MatchEdge(	const btSoftBody::Node* a,
								  const btSoftBody::Node* b,
								  const btSoftBody::Node* ma,
								  const btSoftBody::Node* mb)
{
	if((a==ma)&&(b==mb)) return(0);
	if((a==mb)&&(b==ma)) return(1);
	return(-1);
}

//
// btEigen : Extract eigen system,
// straitforward implementation of http://math.fullerton.edu/mathews/n2003/JacobiMethodMod.html
// outputs are NOT sorted.
//
struct	btEigen
{
	static int			system(btMatrix3x3& a,btMatrix3x3* vectors,btVector3* values=0)
	{
		static const int		maxiterations=16;
		static const btScalar	accuracy=(btScalar)0.0001;
		btMatrix3x3&			v=*vectors;
		int						iterations=0;
		vectors->setIdentity();
		do	{
			int				p=0,q=1;
			if(btFabs(a[p][q])<btFabs(a[0][2])) { p=0;q=2; }
			if(btFabs(a[p][q])<btFabs(a[1][2])) { p=1;q=2; }
			if(btFabs(a[p][q])>accuracy)
			{
				const btScalar	w=(a[q][q]-a[p][p])/(2*a[p][q]);
				const btScalar	z=btFabs(w);
				const btScalar	t=w/(z*(btSqrt(1+w*w)+z));
				if(t==t)/* [WARNING] let hope that one does not get thrown aways by some compilers... */ 
				{
					const btScalar	c=1/btSqrt(t*t+1);
					const btScalar	s=c*t;
					mulPQ(a,c,s,p,q);
					mulTPQ(a,c,s,p,q);
					mulPQ(v,c,s,p,q);
				} else break;
			} else break;
		} while((++iterations)<maxiterations);
		if(values)
		{
			*values=btVector3(a[0][0],a[1][1],a[2][2]);
		}
		return(iterations);
	}
private:
	static inline void	mulTPQ(btMatrix3x3& a,btScalar c,btScalar s,int p,int q)
	{
		const btScalar	m[2][3]={	{a[p][0],a[p][1],a[p][2]},
		{a[q][0],a[q][1],a[q][2]}};
		int i;

		for(i=0;i<3;++i) a[p][i]=c*m[0][i]-s*m[1][i];
		for(i=0;i<3;++i) a[q][i]=c*m[1][i]+s*m[0][i];
	}
	static inline void	mulPQ(btMatrix3x3& a,btScalar c,btScalar s,int p,int q)
	{
		const btScalar	m[2][3]={	{a[0][p],a[1][p],a[2][p]},
		{a[0][q],a[1][q],a[2][q]}};
		int i;

		for(i=0;i<3;++i) a[i][p]=c*m[0][i]-s*m[1][i];
		for(i=0;i<3;++i) a[i][q]=c*m[1][i]+s*m[0][i];
	}
};

//
// Polar decomposition,
// "Computing the Polar Decomposition with Applications", Nicholas J. Higham, 1986.
//
static inline int			PolarDecompose(	const btMatrix3x3& m,btMatrix3x3& q,btMatrix3x3& s)
{
	static const btPolarDecomposition polar;  
	return polar.decompose(m, q, s);
}

//
// btSoftColliders
//
struct btSoftColliders
{
	//
	// ClusterBase
	//
	struct	ClusterBase : btDbvt::ICollide
	{
		btScalar			erp;
		btScalar			idt;
		btScalar			m_margin;
		btScalar			friction;
		btScalar			threshold;
		ClusterBase()
		{
			erp			=(btScalar)1;
			idt			=0;
			m_margin		=0;
			friction	=0;
			threshold	=(btScalar)0;
		}
		bool				SolveContact(	const btGjkEpaSolver2::sResults& res,
			btSoftBody::Body ba,const btSoftBody::Body bb,
			btSoftBody::CJoint& joint)
		{
			if(res.distance<m_margin)
			{
				btVector3 norm = res.normal;
				norm.normalize();//is it necessary?

				const btVector3		ra=res.witnesses[0]-ba.xform().getOrigin();
				const btVector3		rb=res.witnesses[1]-bb.xform().getOrigin();
				const btVector3		va=ba.velocity(ra);
				const btVector3		vb=bb.velocity(rb);
				const btVector3		vrel=va-vb;
				const btScalar		rvac=btDot(vrel,norm);
				 btScalar		depth=res.distance-m_margin;
				
//				printf("depth=%f\n",depth);
				const btVector3		iv=norm*rvac;
				const btVector3		fv=vrel-iv;
				joint.m_bodies[0]	=	ba;
				joint.m_bodies[1]	=	bb;
				joint.m_refs[0]		=	ra*ba.xform().getBasis();
				joint.m_refs[1]		=	rb*bb.xform().getBasis();
				joint.m_rpos[0]		=	ra;
				joint.m_rpos[1]		=	rb;
				joint.m_cfm			=	1;
				joint.m_erp			=	1;
				joint.m_life		=	0;
				joint.m_maxlife		=	0;
				joint.m_split		=	1;
				
				joint.m_drift		=	depth*norm;

				joint.m_normal		=	norm;
//				printf("normal=%f,%f,%f\n",res.normal.getX(),res.normal.getY(),res.normal.getZ());
				joint.m_delete		=	false;
				joint.m_friction	=	fv.length2()<(rvac*friction*rvac*friction)?1:friction;
				joint.m_massmatrix	=	ImpulseMatrix(	ba.invMass(),ba.invWorldInertia(),joint.m_rpos[0],
					bb.invMass(),bb.invWorldInertia(),joint.m_rpos[1]);

				return(true);
			}
			return(false);
		}
	};
	//
	// CollideCL_RS
	//
	struct	CollideCL_RS : ClusterBase
	{
		btSoftBody*		psb;
		const btCollisionObjectWrapper*	m_colObjWrap;

		void		Process(const btDbvtNode* leaf)
		{
			btSoftBody::Cluster*		cluster=(btSoftBody::Cluster*)leaf->data;
			btSoftClusterCollisionShape	cshape(cluster);
			
			const btConvexShape*		rshape=(const btConvexShape*)m_colObjWrap->getCollisionShape();

			///don't collide an anchored cluster with a static/kinematic object
			if(m_colObjWrap->getCollisionObject()->isStaticOrKinematicObject() && cluster->m_containsAnchor)
				return;

			btGjkEpaSolver2::sResults	res;		
			if(btGjkEpaSolver2::SignedDistance(	&cshape,btTransform::getIdentity(),
				rshape,m_colObjWrap->getWorldTransform(),
				btVector3(1,0,0),res))
			{
				btSoftBody::CJoint	joint;
				if(SolveContact(res,cluster,m_colObjWrap->getCollisionObject(),joint))//prb,joint))
				{
					btSoftBody::CJoint*	pj=new(btAlignedAlloc(sizeof(btSoftBody::CJoint),16)) btSoftBody::CJoint();
					*pj=joint;psb->m_joints.push_back(pj);
					if(m_colObjWrap->getCollisionObject()->isStaticOrKinematicObject())
					{
						pj->m_erp	*=	psb->m_cfg.kSKHR_CL;
						pj->m_split	*=	psb->m_cfg.kSK_SPLT_CL;
					}
					else
					{
						pj->m_erp	*=	psb->m_cfg.kSRHR_CL;
						pj->m_split	*=	psb->m_cfg.kSR_SPLT_CL;
					}
				}
			}
		}
		void		ProcessColObj(btSoftBody* ps,const btCollisionObjectWrapper* colObWrap)
		{
			psb			=	ps;
			m_colObjWrap			=	colObWrap;
			idt			=	ps->m_sst.isdt;
			m_margin		=	m_colObjWrap->getCollisionShape()->getMargin()+psb->getCollisionShape()->getMargin();
			///Bullet rigid body uses multiply instead of minimum to determine combined friction. Some customization would be useful.
			friction	=	btMin(psb->m_cfg.kDF,m_colObjWrap->getCollisionObject()->getFriction());
			btVector3			mins;
			btVector3			maxs;

			ATTRIBUTE_ALIGNED16(btDbvtVolume)		volume;
			colObWrap->getCollisionShape()->getAabb(colObWrap->getWorldTransform(),mins,maxs);
			volume=btDbvtVolume::FromMM(mins,maxs);
			volume.Expand(btVector3(1,1,1)*m_margin);
			ps->m_cdbvt.collideTV(ps->m_cdbvt.m_root,volume,*this);
		}	
	};
	//
	// CollideCL_SS
	//
	struct	CollideCL_SS : ClusterBase
	{
		btSoftBody*	bodies[2];
		void		Process(const btDbvtNode* la,const btDbvtNode* lb)
		{
			btSoftBody::Cluster*		cla=(btSoftBody::Cluster*)la->data;
			btSoftBody::Cluster*		clb=(btSoftBody::Cluster*)lb->data;


			bool connected=false;
			if ((bodies[0]==bodies[1])&&(bodies[0]->m_clusterConnectivity.size()))
			{
				connected = bodies[0]->m_clusterConnectivity[cla->m_clusterIndex+bodies[0]->m_clusters.size()*clb->m_clusterIndex];
			}

			if (!connected)
			{
				btSoftClusterCollisionShape	csa(cla);
				btSoftClusterCollisionShape	csb(clb);
				btGjkEpaSolver2::sResults	res;		
				if(btGjkEpaSolver2::SignedDistance(	&csa,btTransform::getIdentity(),
					&csb,btTransform::getIdentity(),
					cla->m_com-clb->m_com,res))
				{
					btSoftBody::CJoint	joint;
					if(SolveContact(res,cla,clb,joint))
					{
						btSoftBody::CJoint*	pj=new(btAlignedAlloc(sizeof(btSoftBody::CJoint),16)) btSoftBody::CJoint();
						*pj=joint;bodies[0]->m_joints.push_back(pj);
						pj->m_erp	*=	btMax(bodies[0]->m_cfg.kSSHR_CL,bodies[1]->m_cfg.kSSHR_CL);
						pj->m_split	*=	(bodies[0]->m_cfg.kSS_SPLT_CL+bodies[1]->m_cfg.kSS_SPLT_CL)/2;
					}
				}
			} else
			{
				static int count=0;
				count++;
				//printf("count=%d\n",count);
				
			}
		}
		void		ProcessSoftSoft(btSoftBody* psa,btSoftBody* psb)
		{
			idt			=	psa->m_sst.isdt;
			//m_margin		=	(psa->getCollisionShape()->getMargin()+psb->getCollisionShape()->getMargin())/2;
			m_margin		=	(psa->getCollisionShape()->getMargin()+psb->getCollisionShape()->getMargin());
			friction	=	btMin(psa->m_cfg.kDF,psb->m_cfg.kDF);
			bodies[0]	=	psa;
			bodies[1]	=	psb;
			psa->m_cdbvt.collideTT(psa->m_cdbvt.m_root,psb->m_cdbvt.m_root,*this);
		}	
	};
	//
	// CollideSDF_RS
	//
	struct	CollideSDF_RS : btDbvt::ICollide
	{
		void		Process(const btDbvtNode* leaf)
		{
			btSoftBody::Node*	node=(btSoftBody::Node*)leaf->data;
			DoNode(*node);
		}
		void		DoNode(btSoftBody::Node& n) const
		{
			const btScalar			m=n.m_im>0?dynmargin:stamargin;
			btSoftBody::RContact	c;

			if(	(!n.m_battach)&&
				psb->checkContact(m_colObj1Wrap,n.m_x,m,c.m_cti))
			{
				const btScalar	ima=n.m_im;
				const btScalar	imb= m_rigidBody? m_rigidBody->getInvMass() : 0.f;
				const btScalar	ms=ima+imb;
				if(ms>0)
				{
					const btTransform&	wtr=m_rigidBody?m_rigidBody->getWorldTransform() : m_colObj1Wrap->getCollisionObject()->getWorldTransform();
					static const btMatrix3x3	iwiStatic(0,0,0,0,0,0,0,0,0);
					const btMatrix3x3&	iwi=m_rigidBody?m_rigidBody->getInvInertiaTensorWorld() : iwiStatic;
					const btVector3		ra=n.m_x-wtr.getOrigin();
					const btVector3		va=m_rigidBody ? m_rigidBody->getVelocityInLocalPoint(ra)*psb->m_sst.sdt : btVector3(0,0,0);
					const btVector3		vb=n.m_x-n.m_q;	
					const btVector3		vr=vb-va;
					const btScalar		dn=btDot(vr,c.m_cti.m_normal);
					const btVector3		fv=vr-c.m_cti.m_normal*dn;
					const btScalar		fc=psb->m_cfg.kDF*m_colObj1Wrap->getCollisionObject()->getFriction();
					c.m_node	=	&n;
					c.m_c0		=	ImpulseMatrix(psb->m_sst.sdt,ima,imb,iwi,ra);
					c.m_c1		=	ra;
					c.m_c2		=	ima*psb->m_sst.sdt;
			        c.m_c3		=	fv.length2()<(dn*fc*dn*fc)?0:1-fc;
					c.m_c4		=	m_colObj1Wrap->getCollisionObject()->isStaticOrKinematicObject()?psb->m_cfg.kKHR:psb->m_cfg.kCHR;
					psb->m_rcontacts.push_back(c);
					if (m_rigidBody)
						m_rigidBody->activate();
				}
			}
		}
		btSoftBody*		psb;
		const btCollisionObjectWrapper*	m_colObj1Wrap;
		btRigidBody*	m_rigidBody;
		btScalar		dynmargin;
		btScalar		stamargin;
	};
	//
	// CollideVF_SS
	//
	struct	CollideVF_SS : btDbvt::ICollide
	{
		void		Process(const btDbvtNode* lnode,
			const btDbvtNode* lface)
		{
			btSoftBody::Node*	node=(btSoftBody::Node*)lnode->data;
			btSoftBody::Face*	face=(btSoftBody::Face*)lface->data;
			btVector3			o=node->m_x;
			btVector3			p;
			btScalar			d=SIMD_INFINITY;
			ProjectOrigin(	face->m_n[0]->m_x-o,
				face->m_n[1]->m_x-o,
				face->m_n[2]->m_x-o,
				p,d);
			const btScalar	m=mrg+(o-node->m_q).length()*2;
			if(d<(m*m))
			{
				const btSoftBody::Node*	n[]={face->m_n[0],face->m_n[1],face->m_n[2]};
				const btVector3			w=BaryCoord(n[0]->m_x,n[1]->m_x,n[2]->m_x,p+o);
				const btScalar			ma=node->m_im;
				btScalar				mb=BaryEval(n[0]->m_im,n[1]->m_im,n[2]->m_im,w);
				if(	(n[0]->m_im<=0)||
					(n[1]->m_im<=0)||
					(n[2]->m_im<=0))
				{
					mb=0;
				}
				const btScalar	ms=ma+mb;
				if(ms>0)
				{
					btSoftBody::SContact	c;
					c.m_normal		=	p/-btSqrt(d);
					c.m_margin		=	m;
					c.m_node		=	node;
					c.m_face		=	face;
					c.m_weights		=	w;
					c.m_friction	=	btMax(psb[0]->m_cfg.kDF,psb[1]->m_cfg.kDF);
					c.m_cfm[0]		=	ma/ms*psb[0]->m_cfg.kSHR;
					c.m_cfm[1]		=	mb/ms*psb[1]->m_cfg.kSHR;
					psb[0]->m_scontacts.push_back(c);
				}
			}	
		}
		btSoftBody*		psb[2];
		btScalar		mrg;
	};
};

#endif //_BT_SOFT_BODY_INTERNALS_H

/*************************************************************************/
/*  gjk_epa.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "gjk_epa.h"

/* Disabling formatting for thirdparty code snippet */
/* clang-format off */

/*************** Bullet's GJK-EPA2 IMPLEMENTATION *******************/

/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2008 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software in a
product, an acknowledgment in the product documentation would be appreciated
but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

/*
GJK-EPA collision solver by Nathanael Presson, 2008
*/

	// Config

/* GJK	*/
#define GJK_MAX_ITERATIONS	128
#define GJK_ACCURACY		((real_t)0.0001)
#define GJK_MIN_DISTANCE	((real_t)0.0001)
#define GJK_DUPLICATED_EPS	((real_t)0.0001)
#define GJK_SIMPLEX2_EPS	((real_t)0.0)
#define GJK_SIMPLEX3_EPS	((real_t)0.0)
#define GJK_SIMPLEX4_EPS	((real_t)0.0)

/* EPA	*/
#define EPA_MAX_VERTICES	128
#define EPA_MAX_FACES		(EPA_MAX_VERTICES*2)
#define EPA_MAX_ITERATIONS	255
// -- GODOT start --
//#define EPA_ACCURACY		((real_t)0.0001)
#define EPA_ACCURACY		((real_t)0.00001)
// -- GODOT end --
#define EPA_FALLBACK		(10*EPA_ACCURACY)
#define EPA_PLANE_EPS		((real_t)0.00001)
#define EPA_INSIDE_EPS		((real_t)0.01)

namespace GjkEpa2 {


struct sResults	{
	enum eStatus {
		Separated,		/* Shapes doesn't penetrate */
		Penetrating,	/* Shapes are penetrating */
		GJK_Failed,		/* GJK phase fail, no big issue, shapes are probably just 'touching'	*/
		EPA_Failed /* EPA phase fail, bigger problem, need to save parameters, and debug	*/
	} status;

	Vector3	witnesses[2];
	Vector3	normal;
	real_t	distance;
};

// Shorthands
typedef unsigned int	U;
typedef unsigned char	U1;

// MinkowskiDiff
struct	MinkowskiDiff {

	const ShapeSW* m_shapes[2];

	Transform transform_A;
	Transform transform_B;

	// i wonder how this could be sped up... if it can
	_FORCE_INLINE_ Vector3 Support0 ( const Vector3& d ) const {
		return transform_A.xform( m_shapes[0]->get_support( transform_A.basis.xform_inv(d).normalized() ) );
	}

	_FORCE_INLINE_ Vector3 Support1 ( const Vector3& d ) const {
		return transform_B.xform( m_shapes[1]->get_support( transform_B.basis.xform_inv(d).normalized() ) );
	}

	_FORCE_INLINE_ Vector3 Support ( const Vector3& d ) const {
		return ( Support0 ( d )-Support1 ( -d ) );
	}

	_FORCE_INLINE_ Vector3	Support ( const Vector3& d,U index ) const
	{
		if ( index )
			return ( Support1 ( d ) );
		else
			return ( Support0 ( d ) );
	}
};

typedef	MinkowskiDiff tShape;


// GJK
struct	GJK
{
	/* Types		*/
	struct	sSV
	{
		Vector3	d,w;
	};
	struct	sSimplex
	{
		sSV*		c[4];
		real_t	p[4];
		U			rank;
	};
	struct	eStatus	{ enum _ {
		Valid,
		Inside,
		Failed		};};
		/* Fields		*/
		tShape			m_shape;
		Vector3		m_ray;
		real_t		m_distance;
		sSimplex		m_simplices[2];
		sSV				m_store[4];
		sSV*			m_free[4];
		U				m_nfree;
		U				m_current;
		sSimplex*		m_simplex;
		eStatus::_		m_status;
		/* Methods		*/
		GJK()
		{
			Initialize();
		}
		void				Initialize()
		{
			m_ray		=	Vector3(0,0,0);
			m_nfree		=	0;
			m_status	=	eStatus::Failed;
			m_current	=	0;
			m_distance	=	0;
		}
		eStatus::_			Evaluate(const tShape& shapearg,const Vector3& guess)
		{
			U			iterations=0;
			real_t	sqdist=0;
			real_t	alpha=0;
			Vector3	lastw[4];
			U			clastw=0;
			/* Initialize solver		*/
			m_free[0]			=	&m_store[0];
			m_free[1]			=	&m_store[1];
			m_free[2]			=	&m_store[2];
			m_free[3]			=	&m_store[3];
			m_nfree				=	4;
			m_current			=	0;
			m_status			=	eStatus::Valid;
			m_shape				=	shapearg;
			m_distance			=	0;
			/* Initialize simplex		*/
			m_simplices[0].rank	=	0;
			m_ray				=	guess;
			const real_t	sqrl=	m_ray.length_squared();
			appendvertice(m_simplices[0],sqrl>0?-m_ray:Vector3(1,0,0));
			m_simplices[0].p[0]	=	1;
			m_ray				=	m_simplices[0].c[0]->w;
			sqdist				=	sqrl;
			lastw[0]			=
				lastw[1]			=
				lastw[2]			=
				lastw[3]			=	m_ray;
			/* Loop						*/
			do	{
				const U		next=1-m_current;
				sSimplex&	cs=m_simplices[m_current];
				sSimplex&	ns=m_simplices[next];
				/* Check zero							*/
				const real_t	rl=m_ray.length();
				if(rl<GJK_MIN_DISTANCE)
				{/* Touching or inside				*/
					m_status=eStatus::Inside;
					break;
				}
				/* Append new vertice in -'v' direction	*/
				appendvertice(cs,-m_ray);
				const Vector3&	w=cs.c[cs.rank-1]->w;
				bool				found=false;
				for(U i=0;i<4;++i)
				{
					if((w-lastw[i]).length_squared()<GJK_DUPLICATED_EPS)
					{ found=true;break; }
				}
				if(found)
				{/* Return old simplex				*/
					removevertice(m_simplices[m_current]);
					break;
				}
				else
				{/* Update lastw					*/
					lastw[clastw=(clastw+1)&3]=w;
				}
				/* Check for termination				*/
				const real_t	omega=vec3_dot(m_ray,w)/rl;
				alpha=MAX(omega,alpha);
				if(((rl-alpha)-(GJK_ACCURACY*rl))<=0)
				{/* Return old simplex				*/
					removevertice(m_simplices[m_current]);
					break;
				}
				/* Reduce simplex						*/
				real_t	weights[4];
				U			mask=0;
				switch(cs.rank)
				{
				case	2:	sqdist=projectorigin(	cs.c[0]->w,
								cs.c[1]->w,
								weights,mask);break;
				case	3:	sqdist=projectorigin(	cs.c[0]->w,
								cs.c[1]->w,
								cs.c[2]->w,
								weights,mask);break;
				case	4:	sqdist=projectorigin(	cs.c[0]->w,
								cs.c[1]->w,
								cs.c[2]->w,
								cs.c[3]->w,
								weights,mask);break;
				}
				if(sqdist>=0)
				{/* Valid	*/
					ns.rank		=	0;
					m_ray		=	Vector3(0,0,0);
					m_current	=	next;
					for(U i=0,ni=cs.rank;i<ni;++i)
					{
						if(mask&(1<<i))
						{
							ns.c[ns.rank]		=	cs.c[i];
							ns.p[ns.rank++]		=	weights[i];
							m_ray				+=	cs.c[i]->w*weights[i];
						}
						else
						{
							m_free[m_nfree++]	=	cs.c[i];
						}
					}
					if(mask==15) m_status=eStatus::Inside;
				}
				else
				{/* Return old simplex				*/
					removevertice(m_simplices[m_current]);
					break;
				}
				m_status=((++iterations)<GJK_MAX_ITERATIONS)?m_status:eStatus::Failed;
			} while(m_status==eStatus::Valid);
			m_simplex=&m_simplices[m_current];
			switch(m_status)
			{
			case	eStatus::Valid:		m_distance=m_ray.length();break;
			case	eStatus::Inside:	m_distance=0;break;
			default: {}
			}
			return(m_status);
		}
		bool					EncloseOrigin()
		{
			switch(m_simplex->rank)
			{
			case	1:
				{
					for(U i=0;i<3;++i)
					{
						Vector3		axis=Vector3(0,0,0);
						axis[i]=1;
						appendvertice(*m_simplex, axis);
						if(EncloseOrigin())	return(true);
						removevertice(*m_simplex);
						appendvertice(*m_simplex,-axis);
						if(EncloseOrigin())	return(true);
						removevertice(*m_simplex);
					}
				}
				break;
			case	2:
				{
					const Vector3	d=m_simplex->c[1]->w-m_simplex->c[0]->w;
					for(U i=0;i<3;++i)
					{
						Vector3		axis=Vector3(0,0,0);
						axis[i]=1;
						const Vector3	p=vec3_cross(d,axis);
						if(p.length_squared()>0)
						{
							appendvertice(*m_simplex, p);
							if(EncloseOrigin())	return(true);
							removevertice(*m_simplex);
							appendvertice(*m_simplex,-p);
							if(EncloseOrigin())	return(true);
							removevertice(*m_simplex);
						}
					}
				}
				break;
			case	3:
				{
					const Vector3	n=vec3_cross(m_simplex->c[1]->w-m_simplex->c[0]->w,
						m_simplex->c[2]->w-m_simplex->c[0]->w);
					if(n.length_squared()>0)
					{
						appendvertice(*m_simplex,n);
						if(EncloseOrigin())	return(true);
						removevertice(*m_simplex);
						appendvertice(*m_simplex,-n);
						if(EncloseOrigin())	return(true);
						removevertice(*m_simplex);
					}
				}
				break;
			case	4:
				{
					if(Math::abs(det(	m_simplex->c[0]->w-m_simplex->c[3]->w,
						m_simplex->c[1]->w-m_simplex->c[3]->w,
						m_simplex->c[2]->w-m_simplex->c[3]->w))>0)
						return(true);
				}
				break;
			}
			return(false);
		}
		/* Internals	*/
		void				getsupport(const Vector3& d,sSV& sv) const
		{
			sv.d	=	d/d.length();
			sv.w	=	m_shape.Support(sv.d);
		}
		void				removevertice(sSimplex& simplex)
		{
			m_free[m_nfree++]=simplex.c[--simplex.rank];
		}
		void				appendvertice(sSimplex& simplex,const Vector3& v)
		{
			simplex.p[simplex.rank]=0;
			simplex.c[simplex.rank]=m_free[--m_nfree];
			getsupport(v,*simplex.c[simplex.rank++]);
		}
		static real_t		det(const Vector3& a,const Vector3& b,const Vector3& c)
		{
			return(	a.y*b.z*c.x+a.z*b.x*c.y-
				a.x*b.z*c.y-a.y*b.x*c.z+
				a.x*b.y*c.z-a.z*b.y*c.x);
		}
		static real_t		projectorigin(	const Vector3& a,
			const Vector3& b,
			real_t* w,U& m)
		{
			const Vector3	d=b-a;
			const real_t	l=d.length_squared();
			if(l>GJK_SIMPLEX2_EPS)
			{
				const real_t	t(l>0?-vec3_dot(a,d)/l:0);
				if(t>=1)		{ w[0]=0;w[1]=1;m=2;return(b.length_squared()); }
				else if(t<=0)	{ w[0]=1;w[1]=0;m=1;return(a.length_squared()); }
				else			{ w[0]=1-(w[1]=t);m=3;return((a+d*t).length_squared()); }
			}
			return(-1);
		}
		static real_t		projectorigin(	const Vector3& a,
			const Vector3& b,
			const Vector3& c,
			real_t* w,U& m)
		{
			static const U		imd3[]={1,2,0};
			const Vector3*	vt[]={&a,&b,&c};
			const Vector3		dl[]={a-b,b-c,c-a};
			const Vector3		n=vec3_cross(dl[0],dl[1]);
			const real_t		l=n.length_squared();
			if(l>GJK_SIMPLEX3_EPS)
			{
				real_t	mindist=-1;
				real_t	subw[2] = { 0 , 0};
				U 		subm = 0;
				for(U i=0;i<3;++i)
				{
					if(vec3_dot(*vt[i],vec3_cross(dl[i],n))>0)
					{
						const U			j=imd3[i];
						const real_t	subd(projectorigin(*vt[i],*vt[j],subw,subm));
						if((mindist<0)||(subd<mindist))
						{
							mindist		=	subd;
							m			=	static_cast<U>(((subm&1)?1<<i:0)+((subm&2)?1<<j:0));
							w[i]		=	subw[0];
							w[j]		=	subw[1];
							w[imd3[j]]	=	0;
						}
					}
				}
				if(mindist<0)
				{
					const real_t	d=vec3_dot(a,n);
					const real_t	s=Math::sqrt(l);
					const Vector3	p=n*(d/l);
					mindist	=	p.length_squared();
					m		=	7;
					w[0]	=	(vec3_cross(dl[1],b-p)).length()/s;
					w[1]	=	(vec3_cross(dl[2],c-p)).length()/s;
					w[2]	=	1-(w[0]+w[1]);
				}
				return(mindist);
			}
			return(-1);
		}
		static real_t		projectorigin(	const Vector3& a,
			const Vector3& b,
			const Vector3& c,
			const Vector3& d,
			real_t* w,U& m)
		{
			static const U		imd3[]={1,2,0};
			const Vector3*	vt[]={&a,&b,&c,&d};
			const Vector3		dl[]={a-d,b-d,c-d};
			const real_t		vl=det(dl[0],dl[1],dl[2]);
			const bool			ng=(vl*vec3_dot(a,vec3_cross(b-c,a-b)))<=0;
			if(ng&&(Math::abs(vl)>GJK_SIMPLEX4_EPS))
			{
				real_t	mindist=-1;
				real_t	subw[3] = {0.f, 0.f, 0.f};
				U		subm=0;
				for(U i=0;i<3;++i)
				{
					const U			j=imd3[i];
					const real_t	s=vl*vec3_dot(d,vec3_cross(dl[i],dl[j]));
					if(s>0)
					{
						const real_t	subd=projectorigin(*vt[i],*vt[j],d,subw,subm);
						if((mindist<0)||(subd<mindist))
						{
							mindist		=	subd;
							m			=	static_cast<U>((subm&1?1<<i:0)+
								(subm&2?1<<j:0)+
								(subm&4?8:0));
							w[i]		=	subw[0];
							w[j]		=	subw[1];
							w[imd3[j]]	=	0;
							w[3]		=	subw[2];
						}
					}
				}
				if(mindist<0)
				{
					mindist	=	0;
					m		=	15;
					w[0]	=	det(c,b,d)/vl;
					w[1]	=	det(a,c,d)/vl;
					w[2]	=	det(b,a,d)/vl;
					w[3]	=	1-(w[0]+w[1]+w[2]);
				}
				return(mindist);
			}
			return(-1);
		}
};

	// EPA
	struct	EPA
	{
		/* Types		*/
		typedef	GJK::sSV	sSV;
		struct	sFace
		{
			Vector3	n;
			real_t	d;
			sSV*		c[3];
			sFace*		f[3];
			sFace*		l[2];
			U1			e[3];
			U1			pass;
		};
		struct	sList
		{
			sFace*		root;
			U			count;
			sList() : root(0),count(0)	{}
		};
		struct	sHorizon
		{
			sFace*		cf;
			sFace*		ff;
			U			nf;
			sHorizon() : cf(0),ff(0),nf(0)	{}
		};
		struct	eStatus { enum _ {
			Valid,
			Touching,
			Degenerated,
			NonConvex,
			InvalidHull,
			OutOfFaces,
			OutOfVertices,
			AccuraryReached,
			FallBack,
			Failed		};};
			/* Fields		*/
			eStatus::_		m_status;
			GJK::sSimplex	m_result;
			Vector3		m_normal;
			real_t		m_depth;
			sSV				m_sv_store[EPA_MAX_VERTICES];
			sFace			m_fc_store[EPA_MAX_FACES];
			U				m_nextsv;
			sList			m_hull;
			sList			m_stock;
			/* Methods		*/
			EPA()
			{
				Initialize();
			}


			static inline void		bind(sFace* fa,U ea,sFace* fb,U eb)
			{
				fa->e[ea]=(U1)eb;fa->f[ea]=fb;
				fb->e[eb]=(U1)ea;fb->f[eb]=fa;
			}
			static inline void		append(sList& list,sFace* face)
			{
				face->l[0]	=	0;
				face->l[1]	=	list.root;
				if(list.root) list.root->l[0]=face;
				list.root	=	face;
				++list.count;
			}
			static inline void		remove(sList& list,sFace* face)
			{
				if(face->l[1]) face->l[1]->l[0]=face->l[0];
				if(face->l[0]) face->l[0]->l[1]=face->l[1];
				if(face==list.root) list.root=face->l[1];
				--list.count;
			}


			void				Initialize()
			{
				m_status	=	eStatus::Failed;
				m_normal	=	Vector3(0,0,0);
				m_depth		=	0;
				m_nextsv	=	0;
				for(U i=0;i<EPA_MAX_FACES;++i)
				{
					append(m_stock,&m_fc_store[EPA_MAX_FACES-i-1]);
				}
			}
			eStatus::_			Evaluate(GJK& gjk,const Vector3& guess)
			{
				GJK::sSimplex&	simplex=*gjk.m_simplex;
				if((simplex.rank>1)&&gjk.EncloseOrigin())
				{

					/* Clean up				*/
					while(m_hull.root)
					{
						sFace*	f = m_hull.root;
						remove(m_hull,f);
						append(m_stock,f);
					}
					m_status	=	eStatus::Valid;
					m_nextsv	=	0;
					/* Orient simplex		*/
					if(gjk.det(	simplex.c[0]->w-simplex.c[3]->w,
						simplex.c[1]->w-simplex.c[3]->w,
						simplex.c[2]->w-simplex.c[3]->w)<0)
					{
						SWAP(simplex.c[0],simplex.c[1]);
						SWAP(simplex.p[0],simplex.p[1]);
					}
					/* Build initial hull	*/
					sFace*	tetra[]={newface(simplex.c[0],simplex.c[1],simplex.c[2],true),
						newface(simplex.c[1],simplex.c[0],simplex.c[3],true),
						newface(simplex.c[2],simplex.c[1],simplex.c[3],true),
						newface(simplex.c[0],simplex.c[2],simplex.c[3],true)};
					if(m_hull.count==4)
					{
						sFace*		best=findbest();
						sFace		outer=*best;
						U			pass=0;
						U			iterations=0;
						bind(tetra[0],0,tetra[1],0);
						bind(tetra[0],1,tetra[2],0);
						bind(tetra[0],2,tetra[3],0);
						bind(tetra[1],1,tetra[3],2);
						bind(tetra[1],2,tetra[2],1);
						bind(tetra[2],2,tetra[3],1);
						m_status=eStatus::Valid;
						for(;iterations<EPA_MAX_ITERATIONS;++iterations)
						{
							if(m_nextsv<EPA_MAX_VERTICES)
							{
								sHorizon		horizon;
								sSV*			w=&m_sv_store[m_nextsv++];
								bool			valid=true;
								best->pass	=	(U1)(++pass);
								gjk.getsupport(best->n,*w);
								const real_t	wdist=vec3_dot(best->n,w->w)-best->d;
								if(wdist>EPA_ACCURACY)
								{
									for(U j=0;(j<3)&&valid;++j)
									{
										valid&=expand(	pass,w,
											best->f[j],best->e[j],
											horizon);
									}
									if(valid&&(horizon.nf>=3))
									{
										bind(horizon.cf,1,horizon.ff,2);
										remove(m_hull,best);
										append(m_stock,best);
										best=findbest();
										outer=*best;
									} else { m_status=eStatus::InvalidHull;break; }
								} else { m_status=eStatus::AccuraryReached;break; }
							} else { m_status=eStatus::OutOfVertices;break; }
						}
						const Vector3	projection=outer.n*outer.d;
						m_normal	=	outer.n;
						m_depth		=	outer.d;
						m_result.rank	=	3;
						m_result.c[0]	=	outer.c[0];
						m_result.c[1]	=	outer.c[1];
						m_result.c[2]	=	outer.c[2];
						m_result.p[0]	=	vec3_cross(	outer.c[1]->w-projection,
							outer.c[2]->w-projection).length();
						m_result.p[1]	=	vec3_cross(	outer.c[2]->w-projection,
							outer.c[0]->w-projection).length();
						m_result.p[2]	=	vec3_cross(	outer.c[0]->w-projection,
							outer.c[1]->w-projection).length();
						const real_t	sum=m_result.p[0]+m_result.p[1]+m_result.p[2];
						m_result.p[0]	/=	sum;
						m_result.p[1]	/=	sum;
						m_result.p[2]	/=	sum;
						return(m_status);
					}
				}
				/* Fallback		*/
				m_status	=	eStatus::FallBack;
				m_normal	=	-guess;
				const real_t	nl=m_normal.length();
				if(nl>0)
					m_normal	=	m_normal/nl;
				else
					m_normal	=	Vector3(1,0,0);
				m_depth	=	0;
				m_result.rank=1;
				m_result.c[0]=simplex.c[0];
				m_result.p[0]=1;
				return(m_status);
			}

			bool getedgedist(sFace* face, sSV* a, sSV* b, real_t& dist)
			{
				const Vector3 ba = b->w - a->w;
				const Vector3 n_ab = vec3_cross(ba, face->n);   // Outward facing edge normal direction, on triangle plane
				const real_t a_dot_nab = vec3_dot(a->w, n_ab);  // Only care about the sign to determine inside/outside, so not normalization required

				if (a_dot_nab < 0)
				{
					// Outside of edge a->b

					const real_t ba_l2 = ba.length_squared();
					const real_t a_dot_ba = vec3_dot(a->w, ba);
					const real_t b_dot_ba = vec3_dot(b->w, ba);

					if (a_dot_ba > 0)
					{
						// Pick distance vertex a
						dist = a->w.length();
					}
					else if (b_dot_ba < 0)
					{
						// Pick distance vertex b
						dist = b->w.length();
					}
					else
					{
						// Pick distance to edge a->b
						const real_t a_dot_b = vec3_dot(a->w, b->w);
						dist = Math::sqrt(MAX((a->w.length_squared() * b->w.length_squared() - a_dot_b * a_dot_b) / ba_l2, 0.0));
					}

					return true;
				}

				return false;
			}

			sFace*				newface(sSV* a,sSV* b,sSV* c,bool forced)
			{
				if(m_stock.root)
				{
					sFace*	face=m_stock.root;
					remove(m_stock,face);
					append(m_hull,face);
					face->pass	=	0;
					face->c[0]	=	a;
					face->c[1]	=	b;
					face->c[2]	=	c;
					face->n		=	vec3_cross(b->w-a->w,c->w-a->w);
					const real_t	l=face->n.length();
					const bool		v=l>EPA_ACCURACY;
					if(v)
					{
						if (!(getedgedist(face, a, b, face->d) ||
							  getedgedist(face, b, c, face->d) ||
							  getedgedist(face, c, a, face->d)))
						{
							// Origin projects to the interior of the triangle
							// Use distance to triangle plane
							face->d = vec3_dot(a->w, face->n) / l;
						}
						face->n		/=	l;
						if(forced||(face->d>=-EPA_PLANE_EPS))
						{
							return(face);
						} else m_status=eStatus::NonConvex;
					} else m_status=eStatus::Degenerated;
					remove(m_hull,face);
					append(m_stock,face);
					return(0);
				}
				// -- GODOT start --
				//m_status=m_stock.root?eStatus::OutOfVertices:eStatus::OutOfFaces;
				m_status=eStatus::OutOfFaces;
				// -- GODOT end --
				return(0);
			}
			sFace*				findbest()
			{
				sFace*		minf=m_hull.root;
				real_t	mind=minf->d*minf->d;
				for(sFace* f=minf->l[1];f;f=f->l[1])
				{
					const real_t	sqd=f->d*f->d;
					if(sqd<mind)
					{
						minf=f;
						mind=sqd;
					}
				}
				return(minf);
			}
			bool				expand(U pass,sSV* w,sFace* f,U e,sHorizon& horizon)
			{
				static const U	i1m3[]={1,2,0};
				static const U	i2m3[]={2,0,1};
				if(f->pass!=pass)
				{
					const U	e1=i1m3[e];
					if((vec3_dot(f->n,w->w)-f->d)<-EPA_PLANE_EPS)
					{
						sFace*	nf=newface(f->c[e1],f->c[e],w,false);
						if(nf)
						{
							bind(nf,0,f,e);
							if(horizon.cf) bind(horizon.cf,1,nf,2); else horizon.ff=nf;
							horizon.cf=nf;
							++horizon.nf;
							return(true);
						}
					}
					else
					{
						const U	e2=i2m3[e];
						f->pass		=	(U1)pass;
						if(	expand(pass,w,f->f[e1],f->e[e1],horizon)&&
							expand(pass,w,f->f[e2],f->e[e2],horizon))
						{
							remove(m_hull,f);
							append(m_stock,f);
							return(true);
						}
					}
				}
				return(false);
			}

	};

	//
	static void	Initialize(	const ShapeSW* shape0,const Transform& wtrs0,
		const ShapeSW* shape1,const Transform& wtrs1,
		sResults& results,
		tShape& shape,
		bool withmargins)
	{
		/* Results		*/
		results.witnesses[0]	=
			results.witnesses[1]	=	Vector3(0,0,0);
		results.status			=	sResults::Separated;
		/* Shape		*/
		shape.m_shapes[0]		=	shape0;
		shape.m_shapes[1]		=	shape1;
		shape.transform_A		=	wtrs0;
		shape.transform_B		=	wtrs1;

	}



//
// Api
//

//

//
bool Distance(	const ShapeSW*	shape0,
									  const Transform&		wtrs0,
									  const ShapeSW*	shape1,
									  const Transform&		wtrs1,
									  const Vector3&		guess,
									  sResults&				results)
{
	tShape			shape;
	Initialize(shape0,wtrs0,shape1,wtrs1,results,shape,false);
	GJK				gjk;
	GJK::eStatus::_	gjk_status=gjk.Evaluate(shape,guess);
	if(gjk_status==GJK::eStatus::Valid)
	{
		Vector3	w0=Vector3(0,0,0);
		Vector3	w1=Vector3(0,0,0);
		for(U i=0;i<gjk.m_simplex->rank;++i)
		{
			const real_t	p=gjk.m_simplex->p[i];
			w0+=shape.Support( gjk.m_simplex->c[i]->d,0)*p;
			w1+=shape.Support(-gjk.m_simplex->c[i]->d,1)*p;
		}
		results.witnesses[0]	=	w0;
		results.witnesses[1]	=	w1;
		results.normal			=	w0-w1;
		results.distance		=	results.normal.length();
		results.normal			/=	results.distance>GJK_MIN_DISTANCE?results.distance:1;
		return(true);
	}
	else
	{
		results.status	=	gjk_status==GJK::eStatus::Inside?
			sResults::Penetrating	:
		sResults::GJK_Failed	;
		return(false);
	}
}

//
bool Penetration(	const ShapeSW*	shape0,
									 const Transform&		wtrs0,
									 const ShapeSW*	shape1,
									 const Transform&		wtrs1,
									 const Vector3&		guess,
									 sResults&				results
									)
{
	tShape			shape;
	Initialize(shape0,wtrs0,shape1,wtrs1,results,shape,false);
	GJK				gjk;
	GJK::eStatus::_	gjk_status=gjk.Evaluate(shape,-guess);
	switch(gjk_status)
	{
	case	GJK::eStatus::Inside:
		{
			EPA				epa;
			EPA::eStatus::_	epa_status=epa.Evaluate(gjk,-guess);
			if(epa_status!=EPA::eStatus::Failed)
			{
				Vector3	w0=Vector3(0,0,0);
				for(U i=0;i<epa.m_result.rank;++i)
				{
					w0+=shape.Support(epa.m_result.c[i]->d,0)*epa.m_result.p[i];
				}
				results.status			=	sResults::Penetrating;
				results.witnesses[0]	=	w0;
				results.witnesses[1]	=	w0-epa.m_normal*epa.m_depth;
				results.normal			=	-epa.m_normal;
				results.distance		=	-epa.m_depth;
				return(true);
			} else results.status=sResults::EPA_Failed;
		}
		break;
	case	GJK::eStatus::Failed:
		results.status=sResults::GJK_Failed;
		break;
	default: {}
	}
	return(false);
}


/* Symbols cleanup		*/

#undef GJK_MAX_ITERATIONS
#undef GJK_ACCURARY
#undef GJK_MIN_DISTANCE
#undef GJK_DUPLICATED_EPS
#undef GJK_SIMPLEX2_EPS
#undef GJK_SIMPLEX3_EPS
#undef GJK_SIMPLEX4_EPS

#undef EPA_MAX_VERTICES
#undef EPA_MAX_FACES
#undef EPA_MAX_ITERATIONS
#undef EPA_ACCURACY
#undef EPA_FALLBACK
#undef EPA_PLANE_EPS
#undef EPA_INSIDE_EPS


} // end of namespace

/* clang-format on */

bool gjk_epa_calculate_distance(const ShapeSW *p_shape_A, const Transform &p_transform_A, const ShapeSW *p_shape_B, const Transform &p_transform_B, Vector3 &r_result_A, Vector3 &r_result_B) {

	GjkEpa2::sResults res;

	if (GjkEpa2::Distance(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_transform_B.origin - p_transform_A.origin, res)) {

		r_result_A = res.witnesses[0];
		r_result_B = res.witnesses[1];
		return true;
	}

	return false;
}

bool gjk_epa_calculate_penetration(const ShapeSW *p_shape_A, const Transform &p_transform_A, const ShapeSW *p_shape_B, const Transform &p_transform_B, CollisionSolverSW::CallbackResult p_result_callback, void *p_userdata, bool p_swap) {

	GjkEpa2::sResults res;

	if (GjkEpa2::Penetration(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_transform_B.origin - p_transform_A.origin, res)) {
		if (p_result_callback) {
			if (p_swap)
				p_result_callback(res.witnesses[1], res.witnesses[0], p_userdata);
			else
				p_result_callback(res.witnesses[0], res.witnesses[1], p_userdata);
		}
		return true;
	}

	return false;
}

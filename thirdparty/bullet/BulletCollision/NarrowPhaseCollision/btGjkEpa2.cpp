/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2008 Erwin Coumans  https://bulletphysics.org

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
#include "BulletCollision/CollisionShapes/btConvexInternalShape.h"
#include "BulletCollision/CollisionShapes/btSphereShape.h"
#include "btGjkEpa2.h"

#if defined(DEBUG) || defined(_DEBUG)
#include <stdio.h>  //for debug printf
#ifdef __SPU__
#include <spu_printf.h>
#define printf spu_printf
#endif  //__SPU__
#endif

namespace gjkepa2_impl
{
// Config

/* GJK	*/
#define GJK_MAX_ITERATIONS 128

#ifdef BT_USE_DOUBLE_PRECISION
#define GJK_ACCURACY ((btScalar)1e-12)
#define GJK_MIN_DISTANCE ((btScalar)1e-12)
#define GJK_DUPLICATED_EPS ((btScalar)1e-12)
#else
#define GJK_ACCURACY ((btScalar)0.0001)
#define GJK_MIN_DISTANCE ((btScalar)0.0001)
#define GJK_DUPLICATED_EPS ((btScalar)0.0001)
#endif  //BT_USE_DOUBLE_PRECISION

#define GJK_SIMPLEX2_EPS ((btScalar)0.0)
#define GJK_SIMPLEX3_EPS ((btScalar)0.0)
#define GJK_SIMPLEX4_EPS ((btScalar)0.0)

/* EPA	*/
#define EPA_MAX_VERTICES 128
#define EPA_MAX_ITERATIONS 255

#ifdef BT_USE_DOUBLE_PRECISION
#define EPA_ACCURACY ((btScalar)1e-12)
#define EPA_PLANE_EPS ((btScalar)1e-14)
#define EPA_INSIDE_EPS ((btScalar)1e-9)
#else
#define EPA_ACCURACY ((btScalar)0.0001)
#define EPA_PLANE_EPS ((btScalar)0.00001)
#define EPA_INSIDE_EPS ((btScalar)0.01)
#endif

#define EPA_FALLBACK (10 * EPA_ACCURACY)
#define EPA_MAX_FACES (EPA_MAX_VERTICES * 2)

// Shorthands
typedef unsigned int U;
typedef unsigned char U1;

// MinkowskiDiff
struct MinkowskiDiff
{
	const btConvexShape* m_shapes[2];
	btMatrix3x3 m_toshape1;
	btTransform m_toshape0;
#ifdef __SPU__
	bool m_enableMargin;
#else
	btVector3 (btConvexShape::*Ls)(const btVector3&) const;
#endif  //__SPU__

	MinkowskiDiff()
	{
	}
#ifdef __SPU__
	void EnableMargin(bool enable)
	{
		m_enableMargin = enable;
	}
	inline btVector3 Support0(const btVector3& d) const
	{
		if (m_enableMargin)
		{
			return m_shapes[0]->localGetSupportVertexNonVirtual(d);
		}
		else
		{
			return m_shapes[0]->localGetSupportVertexWithoutMarginNonVirtual(d);
		}
	}
	inline btVector3 Support1(const btVector3& d) const
	{
		if (m_enableMargin)
		{
			return m_toshape0 * (m_shapes[1]->localGetSupportVertexNonVirtual(m_toshape1 * d));
		}
		else
		{
			return m_toshape0 * (m_shapes[1]->localGetSupportVertexWithoutMarginNonVirtual(m_toshape1 * d));
		}
	}
#else
	void EnableMargin(bool enable)
	{
		if (enable)
			Ls = &btConvexShape::localGetSupportVertexNonVirtual;
		else
			Ls = &btConvexShape::localGetSupportVertexWithoutMarginNonVirtual;
	}
	inline btVector3 Support0(const btVector3& d) const
	{
		return (((m_shapes[0])->*(Ls))(d));
	}
	inline btVector3 Support1(const btVector3& d) const
	{
		return (m_toshape0 * ((m_shapes[1])->*(Ls))(m_toshape1 * d));
	}
#endif  //__SPU__

	inline btVector3 Support(const btVector3& d) const
	{
		return (Support0(d) - Support1(-d));
	}
	btVector3 Support(const btVector3& d, U index) const
	{
		if (index)
			return (Support1(d));
		else
			return (Support0(d));
	}
};

typedef MinkowskiDiff tShape;

// GJK
struct GJK
{
	/* Types		*/
	struct sSV
	{
		btVector3 d, w;
	};
	struct sSimplex
	{
		sSV* c[4];
		btScalar p[4];
		U rank;
	};
	struct eStatus
	{
		enum _
		{
			Valid,
			Inside,
			Failed
		};
	};
	/* Fields		*/
	tShape m_shape;
	btVector3 m_ray;
	btScalar m_distance;
	sSimplex m_simplices[2];
	sSV m_store[4];
	sSV* m_free[4];
	U m_nfree;
	U m_current;
	sSimplex* m_simplex;
	eStatus::_ m_status;
	/* Methods		*/
	GJK()
	{
		Initialize();
	}
	void Initialize()
	{
		m_ray = btVector3(0, 0, 0);
		m_nfree = 0;
		m_status = eStatus::Failed;
		m_current = 0;
		m_distance = 0;
	}
	eStatus::_ Evaluate(const tShape& shapearg, const btVector3& guess)
	{
		U iterations = 0;
		btScalar sqdist = 0;
		btScalar alpha = 0;
		btVector3 lastw[4];
		U clastw = 0;
		/* Initialize solver		*/
		m_free[0] = &m_store[0];
		m_free[1] = &m_store[1];
		m_free[2] = &m_store[2];
		m_free[3] = &m_store[3];
		m_nfree = 4;
		m_current = 0;
		m_status = eStatus::Valid;
		m_shape = shapearg;
		m_distance = 0;
		/* Initialize simplex		*/
		m_simplices[0].rank = 0;
		m_ray = guess;
		const btScalar sqrl = m_ray.length2();
		appendvertice(m_simplices[0], sqrl > 0 ? -m_ray : btVector3(1, 0, 0));
		m_simplices[0].p[0] = 1;
		m_ray = m_simplices[0].c[0]->w;
		sqdist = sqrl;
		lastw[0] =
			lastw[1] =
				lastw[2] =
					lastw[3] = m_ray;
		/* Loop						*/
		do
		{
			const U next = 1 - m_current;
			sSimplex& cs = m_simplices[m_current];
			sSimplex& ns = m_simplices[next];
			/* Check zero							*/
			const btScalar rl = m_ray.length();
			if (rl < GJK_MIN_DISTANCE)
			{ /* Touching or inside				*/
				m_status = eStatus::Inside;
				break;
			}
			/* Append new vertice in -'v' direction	*/
			appendvertice(cs, -m_ray);
			const btVector3& w = cs.c[cs.rank - 1]->w;
			bool found = false;
			for (U i = 0; i < 4; ++i)
			{
				if ((w - lastw[i]).length2() < GJK_DUPLICATED_EPS)
				{
					found = true;
					break;
				}
			}
			if (found)
			{ /* Return old simplex				*/
				removevertice(m_simplices[m_current]);
				break;
			}
			else
			{ /* Update lastw					*/
				lastw[clastw = (clastw + 1) & 3] = w;
			}
			/* Check for termination				*/
			const btScalar omega = btDot(m_ray, w) / rl;
			alpha = btMax(omega, alpha);
			if (((rl - alpha) - (GJK_ACCURACY * rl)) <= 0)
			{ /* Return old simplex				*/
				removevertice(m_simplices[m_current]);
				break;
			}
			/* Reduce simplex						*/
			btScalar weights[4];
			U mask = 0;
			switch (cs.rank)
			{
				case 2:
					sqdist = projectorigin(cs.c[0]->w,
										   cs.c[1]->w,
										   weights, mask);
					break;
				case 3:
					sqdist = projectorigin(cs.c[0]->w,
										   cs.c[1]->w,
										   cs.c[2]->w,
										   weights, mask);
					break;
				case 4:
					sqdist = projectorigin(cs.c[0]->w,
										   cs.c[1]->w,
										   cs.c[2]->w,
										   cs.c[3]->w,
										   weights, mask);
					break;
			}
			if (sqdist >= 0)
			{ /* Valid	*/
				ns.rank = 0;
				m_ray = btVector3(0, 0, 0);
				m_current = next;
				for (U i = 0, ni = cs.rank; i < ni; ++i)
				{
					if (mask & (1 << i))
					{
						ns.c[ns.rank] = cs.c[i];
						ns.p[ns.rank++] = weights[i];
						m_ray += cs.c[i]->w * weights[i];
					}
					else
					{
						m_free[m_nfree++] = cs.c[i];
					}
				}
				if (mask == 15) m_status = eStatus::Inside;
			}
			else
			{ /* Return old simplex				*/
				removevertice(m_simplices[m_current]);
				break;
			}
			m_status = ((++iterations) < GJK_MAX_ITERATIONS) ? m_status : eStatus::Failed;
		} while (m_status == eStatus::Valid);
		m_simplex = &m_simplices[m_current];
		switch (m_status)
		{
			case eStatus::Valid:
				m_distance = m_ray.length();
				break;
			case eStatus::Inside:
				m_distance = 0;
				break;
			default:
			{
			}
		}
		return (m_status);
	}
	bool EncloseOrigin()
	{
		switch (m_simplex->rank)
		{
			case 1:
			{
				for (U i = 0; i < 3; ++i)
				{
					btVector3 axis = btVector3(0, 0, 0);
					axis[i] = 1;
					appendvertice(*m_simplex, axis);
					if (EncloseOrigin()) return (true);
					removevertice(*m_simplex);
					appendvertice(*m_simplex, -axis);
					if (EncloseOrigin()) return (true);
					removevertice(*m_simplex);
				}
			}
			break;
			case 2:
			{
				const btVector3 d = m_simplex->c[1]->w - m_simplex->c[0]->w;
				for (U i = 0; i < 3; ++i)
				{
					btVector3 axis = btVector3(0, 0, 0);
					axis[i] = 1;
					const btVector3 p = btCross(d, axis);
					if (p.length2() > 0)
					{
						appendvertice(*m_simplex, p);
						if (EncloseOrigin()) return (true);
						removevertice(*m_simplex);
						appendvertice(*m_simplex, -p);
						if (EncloseOrigin()) return (true);
						removevertice(*m_simplex);
					}
				}
			}
			break;
			case 3:
			{
				const btVector3 n = btCross(m_simplex->c[1]->w - m_simplex->c[0]->w,
											m_simplex->c[2]->w - m_simplex->c[0]->w);
				if (n.length2() > 0)
				{
					appendvertice(*m_simplex, n);
					if (EncloseOrigin()) return (true);
					removevertice(*m_simplex);
					appendvertice(*m_simplex, -n);
					if (EncloseOrigin()) return (true);
					removevertice(*m_simplex);
				}
			}
			break;
			case 4:
			{
				if (btFabs(det(m_simplex->c[0]->w - m_simplex->c[3]->w,
							   m_simplex->c[1]->w - m_simplex->c[3]->w,
							   m_simplex->c[2]->w - m_simplex->c[3]->w)) > 0)
					return (true);
			}
			break;
		}
		return (false);
	}
	/* Internals	*/
	void getsupport(const btVector3& d, sSV& sv) const
	{
		sv.d = d / d.length();
		sv.w = m_shape.Support(sv.d);
	}
	void removevertice(sSimplex& simplex)
	{
		m_free[m_nfree++] = simplex.c[--simplex.rank];
	}
	void appendvertice(sSimplex& simplex, const btVector3& v)
	{
		simplex.p[simplex.rank] = 0;
		simplex.c[simplex.rank] = m_free[--m_nfree];
		getsupport(v, *simplex.c[simplex.rank++]);
	}
	static btScalar det(const btVector3& a, const btVector3& b, const btVector3& c)
	{
		return (a.y() * b.z() * c.x() + a.z() * b.x() * c.y() -
				a.x() * b.z() * c.y() - a.y() * b.x() * c.z() +
				a.x() * b.y() * c.z() - a.z() * b.y() * c.x());
	}
	static btScalar projectorigin(const btVector3& a,
								  const btVector3& b,
								  btScalar* w, U& m)
	{
		const btVector3 d = b - a;
		const btScalar l = d.length2();
		if (l > GJK_SIMPLEX2_EPS)
		{
			const btScalar t(l > 0 ? -btDot(a, d) / l : 0);
			if (t >= 1)
			{
				w[0] = 0;
				w[1] = 1;
				m = 2;
				return (b.length2());
			}
			else if (t <= 0)
			{
				w[0] = 1;
				w[1] = 0;
				m = 1;
				return (a.length2());
			}
			else
			{
				w[0] = 1 - (w[1] = t);
				m = 3;
				return ((a + d * t).length2());
			}
		}
		return (-1);
	}
	static btScalar projectorigin(const btVector3& a,
								  const btVector3& b,
								  const btVector3& c,
								  btScalar* w, U& m)
	{
		static const U imd3[] = {1, 2, 0};
		const btVector3* vt[] = {&a, &b, &c};
		const btVector3 dl[] = {a - b, b - c, c - a};
		const btVector3 n = btCross(dl[0], dl[1]);
		const btScalar l = n.length2();
		if (l > GJK_SIMPLEX3_EPS)
		{
			btScalar mindist = -1;
			btScalar subw[2] = {0.f, 0.f};
			U subm(0);
			for (U i = 0; i < 3; ++i)
			{
				if (btDot(*vt[i], btCross(dl[i], n)) > 0)
				{
					const U j = imd3[i];
					const btScalar subd(projectorigin(*vt[i], *vt[j], subw, subm));
					if ((mindist < 0) || (subd < mindist))
					{
						mindist = subd;
						m = static_cast<U>(((subm & 1) ? 1 << i : 0) + ((subm & 2) ? 1 << j : 0));
						w[i] = subw[0];
						w[j] = subw[1];
						w[imd3[j]] = 0;
					}
				}
			}
			if (mindist < 0)
			{
				const btScalar d = btDot(a, n);
				const btScalar s = btSqrt(l);
				const btVector3 p = n * (d / l);
				mindist = p.length2();
				m = 7;
				w[0] = (btCross(dl[1], b - p)).length() / s;
				w[1] = (btCross(dl[2], c - p)).length() / s;
				w[2] = 1 - (w[0] + w[1]);
			}
			return (mindist);
		}
		return (-1);
	}
	static btScalar projectorigin(const btVector3& a,
								  const btVector3& b,
								  const btVector3& c,
								  const btVector3& d,
								  btScalar* w, U& m)
	{
		static const U imd3[] = {1, 2, 0};
		const btVector3* vt[] = {&a, &b, &c, &d};
		const btVector3 dl[] = {a - d, b - d, c - d};
		const btScalar vl = det(dl[0], dl[1], dl[2]);
		const bool ng = (vl * btDot(a, btCross(b - c, a - b))) <= 0;
		if (ng && (btFabs(vl) > GJK_SIMPLEX4_EPS))
		{
			btScalar mindist = -1;
			btScalar subw[3] = {0.f, 0.f, 0.f};
			U subm(0);
			for (U i = 0; i < 3; ++i)
			{
				const U j = imd3[i];
				const btScalar s = vl * btDot(d, btCross(dl[i], dl[j]));
				if (s > 0)
				{
					const btScalar subd = projectorigin(*vt[i], *vt[j], d, subw, subm);
					if ((mindist < 0) || (subd < mindist))
					{
						mindist = subd;
						m = static_cast<U>((subm & 1 ? 1 << i : 0) +
										   (subm & 2 ? 1 << j : 0) +
										   (subm & 4 ? 8 : 0));
						w[i] = subw[0];
						w[j] = subw[1];
						w[imd3[j]] = 0;
						w[3] = subw[2];
					}
				}
			}
			if (mindist < 0)
			{
				mindist = 0;
				m = 15;
				w[0] = det(c, b, d) / vl;
				w[1] = det(a, c, d) / vl;
				w[2] = det(b, a, d) / vl;
				w[3] = 1 - (w[0] + w[1] + w[2]);
			}
			return (mindist);
		}
		return (-1);
	}
};

// EPA
struct EPA
{
	/* Types		*/
	typedef GJK::sSV sSV;
	struct sFace
	{
		btVector3 n;
		btScalar d;
		sSV* c[3];
		sFace* f[3];
		sFace* l[2];
		U1 e[3];
		U1 pass;
	};
	struct sList
	{
		sFace* root;
		U count;
		sList() : root(0), count(0) {}
	};
	struct sHorizon
	{
		sFace* cf;
		sFace* ff;
		U nf;
		sHorizon() : cf(0), ff(0), nf(0) {}
	};
	struct eStatus
	{
		enum _
		{
			Valid,
			Touching,
			Degenerated,
			NonConvex,
			InvalidHull,
			OutOfFaces,
			OutOfVertices,
			AccuraryReached,
			FallBack,
			Failed
		};
	};
	/* Fields		*/
	eStatus::_ m_status;
	GJK::sSimplex m_result;
	btVector3 m_normal;
	btScalar m_depth;
	sSV m_sv_store[EPA_MAX_VERTICES];
	sFace m_fc_store[EPA_MAX_FACES];
	U m_nextsv;
	sList m_hull;
	sList m_stock;
	/* Methods		*/
	EPA()
	{
		Initialize();
	}

	static inline void bind(sFace* fa, U ea, sFace* fb, U eb)
	{
		fa->e[ea] = (U1)eb;
		fa->f[ea] = fb;
		fb->e[eb] = (U1)ea;
		fb->f[eb] = fa;
	}
	static inline void append(sList& list, sFace* face)
	{
		face->l[0] = 0;
		face->l[1] = list.root;
		if (list.root) list.root->l[0] = face;
		list.root = face;
		++list.count;
	}
	static inline void remove(sList& list, sFace* face)
	{
		if (face->l[1]) face->l[1]->l[0] = face->l[0];
		if (face->l[0]) face->l[0]->l[1] = face->l[1];
		if (face == list.root) list.root = face->l[1];
		--list.count;
	}

	void Initialize()
	{
		m_status = eStatus::Failed;
		m_normal = btVector3(0, 0, 0);
		m_depth = 0;
		m_nextsv = 0;
		for (U i = 0; i < EPA_MAX_FACES; ++i)
		{
			append(m_stock, &m_fc_store[EPA_MAX_FACES - i - 1]);
		}
	}
	eStatus::_ Evaluate(GJK& gjk, const btVector3& guess)
	{
		GJK::sSimplex& simplex = *gjk.m_simplex;
		if ((simplex.rank > 1) && gjk.EncloseOrigin())
		{
			/* Clean up				*/
			while (m_hull.root)
			{
				sFace* f = m_hull.root;
				remove(m_hull, f);
				append(m_stock, f);
			}
			m_status = eStatus::Valid;
			m_nextsv = 0;
			/* Orient simplex		*/
			if (gjk.det(simplex.c[0]->w - simplex.c[3]->w,
						simplex.c[1]->w - simplex.c[3]->w,
						simplex.c[2]->w - simplex.c[3]->w) < 0)
			{
				btSwap(simplex.c[0], simplex.c[1]);
				btSwap(simplex.p[0], simplex.p[1]);
			}
			/* Build initial hull	*/
			sFace* tetra[] = {newface(simplex.c[0], simplex.c[1], simplex.c[2], true),
							  newface(simplex.c[1], simplex.c[0], simplex.c[3], true),
							  newface(simplex.c[2], simplex.c[1], simplex.c[3], true),
							  newface(simplex.c[0], simplex.c[2], simplex.c[3], true)};
			if (m_hull.count == 4)
			{
				sFace* best = findbest();
				sFace outer = *best;
				U pass = 0;
				U iterations = 0;
				bind(tetra[0], 0, tetra[1], 0);
				bind(tetra[0], 1, tetra[2], 0);
				bind(tetra[0], 2, tetra[3], 0);
				bind(tetra[1], 1, tetra[3], 2);
				bind(tetra[1], 2, tetra[2], 1);
				bind(tetra[2], 2, tetra[3], 1);
				m_status = eStatus::Valid;
				for (; iterations < EPA_MAX_ITERATIONS; ++iterations)
				{
					if (m_nextsv < EPA_MAX_VERTICES)
					{
						sHorizon horizon;
						sSV* w = &m_sv_store[m_nextsv++];
						bool valid = true;
						best->pass = (U1)(++pass);
						gjk.getsupport(best->n, *w);
						const btScalar wdist = btDot(best->n, w->w) - best->d;
						if (wdist > EPA_ACCURACY)
						{
							for (U j = 0; (j < 3) && valid; ++j)
							{
								valid &= expand(pass, w,
												best->f[j], best->e[j],
												horizon);
							}
							if (valid && (horizon.nf >= 3))
							{
								bind(horizon.cf, 1, horizon.ff, 2);
								remove(m_hull, best);
								append(m_stock, best);
								best = findbest();
								outer = *best;
							}
							else
							{
								m_status = eStatus::InvalidHull;
								break;
							}
						}
						else
						{
							m_status = eStatus::AccuraryReached;
							break;
						}
					}
					else
					{
						m_status = eStatus::OutOfVertices;
						break;
					}
				}
				const btVector3 projection = outer.n * outer.d;
				m_normal = outer.n;
				m_depth = outer.d;
				m_result.rank = 3;
				m_result.c[0] = outer.c[0];
				m_result.c[1] = outer.c[1];
				m_result.c[2] = outer.c[2];
				m_result.p[0] = btCross(outer.c[1]->w - projection,
										outer.c[2]->w - projection)
									.length();
				m_result.p[1] = btCross(outer.c[2]->w - projection,
										outer.c[0]->w - projection)
									.length();
				m_result.p[2] = btCross(outer.c[0]->w - projection,
										outer.c[1]->w - projection)
									.length();
				const btScalar sum = m_result.p[0] + m_result.p[1] + m_result.p[2];
				m_result.p[0] /= sum;
				m_result.p[1] /= sum;
				m_result.p[2] /= sum;
				return (m_status);
			}
		}
		/* Fallback		*/
		m_status = eStatus::FallBack;
		m_normal = -guess;
		const btScalar nl = m_normal.length();
		if (nl > 0)
			m_normal = m_normal / nl;
		else
			m_normal = btVector3(1, 0, 0);
		m_depth = 0;
		m_result.rank = 1;
		m_result.c[0] = simplex.c[0];
		m_result.p[0] = 1;
		return (m_status);
	}
	bool getedgedist(sFace* face, sSV* a, sSV* b, btScalar& dist)
	{
		const btVector3 ba = b->w - a->w;
		const btVector3 n_ab = btCross(ba, face->n);   // Outward facing edge normal direction, on triangle plane
		const btScalar a_dot_nab = btDot(a->w, n_ab);  // Only care about the sign to determine inside/outside, so not normalization required

		if (a_dot_nab < 0)
		{
			// Outside of edge a->b

			const btScalar ba_l2 = ba.length2();
			const btScalar a_dot_ba = btDot(a->w, ba);
			const btScalar b_dot_ba = btDot(b->w, ba);

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
				const btScalar a_dot_b = btDot(a->w, b->w);
				dist = btSqrt(btMax((a->w.length2() * b->w.length2() - a_dot_b * a_dot_b) / ba_l2, (btScalar)0));
			}

			return true;
		}

		return false;
	}
	sFace* newface(sSV* a, sSV* b, sSV* c, bool forced)
	{
		if (m_stock.root)
		{
			sFace* face = m_stock.root;
			remove(m_stock, face);
			append(m_hull, face);
			face->pass = 0;
			face->c[0] = a;
			face->c[1] = b;
			face->c[2] = c;
			face->n = btCross(b->w - a->w, c->w - a->w);
			const btScalar l = face->n.length();
			const bool v = l > EPA_ACCURACY;

			if (v)
			{
				if (!(getedgedist(face, a, b, face->d) ||
					  getedgedist(face, b, c, face->d) ||
					  getedgedist(face, c, a, face->d)))
				{
					// Origin projects to the interior of the triangle
					// Use distance to triangle plane
					face->d = btDot(a->w, face->n) / l;
				}

				face->n /= l;
				if (forced || (face->d >= -EPA_PLANE_EPS))
				{
					return face;
				}
				else
					m_status = eStatus::NonConvex;
			}
			else
				m_status = eStatus::Degenerated;

			remove(m_hull, face);
			append(m_stock, face);
			return 0;
		}
		m_status = m_stock.root ? eStatus::OutOfVertices : eStatus::OutOfFaces;
		return 0;
	}
	sFace* findbest()
	{
		sFace* minf = m_hull.root;
		btScalar mind = minf->d * minf->d;
		for (sFace* f = minf->l[1]; f; f = f->l[1])
		{
			const btScalar sqd = f->d * f->d;
			if (sqd < mind)
			{
				minf = f;
				mind = sqd;
			}
		}
		return (minf);
	}
	bool expand(U pass, sSV* w, sFace* f, U e, sHorizon& horizon)
	{
		static const U i1m3[] = {1, 2, 0};
		static const U i2m3[] = {2, 0, 1};
		if (f->pass != pass)
		{
			const U e1 = i1m3[e];
			if ((btDot(f->n, w->w) - f->d) < -EPA_PLANE_EPS)
			{
				sFace* nf = newface(f->c[e1], f->c[e], w, false);
				if (nf)
				{
					bind(nf, 0, f, e);
					if (horizon.cf)
						bind(horizon.cf, 1, nf, 2);
					else
						horizon.ff = nf;
					horizon.cf = nf;
					++horizon.nf;
					return (true);
				}
			}
			else
			{
				const U e2 = i2m3[e];
				f->pass = (U1)pass;
				if (expand(pass, w, f->f[e1], f->e[e1], horizon) &&
					expand(pass, w, f->f[e2], f->e[e2], horizon))
				{
					remove(m_hull, f);
					append(m_stock, f);
					return (true);
				}
			}
		}
		return (false);
	}
};

//
static void Initialize(const btConvexShape* shape0, const btTransform& wtrs0,
					   const btConvexShape* shape1, const btTransform& wtrs1,
					   btGjkEpaSolver2::sResults& results,
					   tShape& shape,
					   bool withmargins)
{
	/* Results		*/
	results.witnesses[0] =
		results.witnesses[1] = btVector3(0, 0, 0);
	results.status = btGjkEpaSolver2::sResults::Separated;
	/* Shape		*/
	shape.m_shapes[0] = shape0;
	shape.m_shapes[1] = shape1;
	shape.m_toshape1 = wtrs1.getBasis().transposeTimes(wtrs0.getBasis());
	shape.m_toshape0 = wtrs0.inverseTimes(wtrs1);
	shape.EnableMargin(withmargins);
}

}  // namespace gjkepa2_impl

//
// Api
//

using namespace gjkepa2_impl;

//
int btGjkEpaSolver2::StackSizeRequirement()
{
	return (sizeof(GJK) + sizeof(EPA));
}

//
bool btGjkEpaSolver2::Distance(const btConvexShape* shape0,
							   const btTransform& wtrs0,
							   const btConvexShape* shape1,
							   const btTransform& wtrs1,
							   const btVector3& guess,
							   sResults& results)
{
	tShape shape;
	Initialize(shape0, wtrs0, shape1, wtrs1, results, shape, false);
	GJK gjk;
	GJK::eStatus::_ gjk_status = gjk.Evaluate(shape, guess);
	if (gjk_status == GJK::eStatus::Valid)
	{
		btVector3 w0 = btVector3(0, 0, 0);
		btVector3 w1 = btVector3(0, 0, 0);
		for (U i = 0; i < gjk.m_simplex->rank; ++i)
		{
			const btScalar p = gjk.m_simplex->p[i];
			w0 += shape.Support(gjk.m_simplex->c[i]->d, 0) * p;
			w1 += shape.Support(-gjk.m_simplex->c[i]->d, 1) * p;
		}
		results.witnesses[0] = wtrs0 * w0;
		results.witnesses[1] = wtrs0 * w1;
		results.normal = w0 - w1;
		results.distance = results.normal.length();
		results.normal /= results.distance > GJK_MIN_DISTANCE ? results.distance : 1;
		return (true);
	}
	else
	{
		results.status = gjk_status == GJK::eStatus::Inside ? sResults::Penetrating : sResults::GJK_Failed;
		return (false);
	}
}

//
bool btGjkEpaSolver2::Penetration(const btConvexShape* shape0,
								  const btTransform& wtrs0,
								  const btConvexShape* shape1,
								  const btTransform& wtrs1,
								  const btVector3& guess,
								  sResults& results,
								  bool usemargins)
{
	tShape shape;
	Initialize(shape0, wtrs0, shape1, wtrs1, results, shape, usemargins);
	GJK gjk;
	GJK::eStatus::_ gjk_status = gjk.Evaluate(shape, -guess);
	switch (gjk_status)
	{
		case GJK::eStatus::Inside:
		{
			EPA epa;
			EPA::eStatus::_ epa_status = epa.Evaluate(gjk, -guess);
			if (epa_status != EPA::eStatus::Failed)
			{
				btVector3 w0 = btVector3(0, 0, 0);
				for (U i = 0; i < epa.m_result.rank; ++i)
				{
					w0 += shape.Support(epa.m_result.c[i]->d, 0) * epa.m_result.p[i];
				}
				results.status = sResults::Penetrating;
				results.witnesses[0] = wtrs0 * w0;
				results.witnesses[1] = wtrs0 * (w0 - epa.m_normal * epa.m_depth);
				results.normal = -epa.m_normal;
				results.distance = -epa.m_depth;
				return (true);
			}
			else
				results.status = sResults::EPA_Failed;
		}
		break;
		case GJK::eStatus::Failed:
			results.status = sResults::GJK_Failed;
			break;
		default:
		{
		}
	}
	return (false);
}

#ifndef __SPU__
//
btScalar btGjkEpaSolver2::SignedDistance(const btVector3& position,
										 btScalar margin,
										 const btConvexShape* shape0,
										 const btTransform& wtrs0,
										 sResults& results)
{
	tShape shape;
	btSphereShape shape1(margin);
	btTransform wtrs1(btQuaternion(0, 0, 0, 1), position);
	Initialize(shape0, wtrs0, &shape1, wtrs1, results, shape, false);
	GJK gjk;
	GJK::eStatus::_ gjk_status = gjk.Evaluate(shape, btVector3(1, 1, 1));
	if (gjk_status == GJK::eStatus::Valid)
	{
		btVector3 w0 = btVector3(0, 0, 0);
		btVector3 w1 = btVector3(0, 0, 0);
		for (U i = 0; i < gjk.m_simplex->rank; ++i)
		{
			const btScalar p = gjk.m_simplex->p[i];
			w0 += shape.Support(gjk.m_simplex->c[i]->d, 0) * p;
			w1 += shape.Support(-gjk.m_simplex->c[i]->d, 1) * p;
		}
		results.witnesses[0] = wtrs0 * w0;
		results.witnesses[1] = wtrs0 * w1;
		const btVector3 delta = results.witnesses[1] -
								results.witnesses[0];
		const btScalar margin = shape0->getMarginNonVirtual() +
								shape1.getMarginNonVirtual();
		const btScalar length = delta.length();
		results.normal = delta / length;
		results.witnesses[0] += results.normal * margin;
		results.distance = length - margin;
		return results.distance;
	}
	else
	{
		if (gjk_status == GJK::eStatus::Inside)
		{
			if (Penetration(shape0, wtrs0, &shape1, wtrs1, gjk.m_ray, results))
			{
				const btVector3 delta = results.witnesses[0] -
										results.witnesses[1];
				const btScalar length = delta.length();
				if (length >= SIMD_EPSILON)
					results.normal = delta / length;
				return (-length);
			}
		}
	}
	return (SIMD_INFINITY);
}

//
bool btGjkEpaSolver2::SignedDistance(const btConvexShape* shape0,
									 const btTransform& wtrs0,
									 const btConvexShape* shape1,
									 const btTransform& wtrs1,
									 const btVector3& guess,
									 sResults& results)
{
	if (!Distance(shape0, wtrs0, shape1, wtrs1, guess, results))
		return (Penetration(shape0, wtrs0, shape1, wtrs1, guess, results, false));
	else
		return (true);
}
#endif  //__SPU__

/* Symbols cleanup		*/

#undef GJK_MAX_ITERATIONS
#undef GJK_ACCURACY
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

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

#include "b3GjkEpa.h"

#include "b3SupportMappings.h"

namespace gjkepa2_impl2
{
// Config

/* GJK	*/
#define GJK_MAX_ITERATIONS 128
#define GJK_ACCURACY ((b3Scalar)0.0001)
#define GJK_MIN_DISTANCE ((b3Scalar)0.0001)
#define GJK_DUPLICATED_EPS ((b3Scalar)0.0001)
#define GJK_SIMPLEX2_EPS ((b3Scalar)0.0)
#define GJK_SIMPLEX3_EPS ((b3Scalar)0.0)
#define GJK_SIMPLEX4_EPS ((b3Scalar)0.0)

/* EPA	*/
#define EPA_MAX_VERTICES 64
#define EPA_MAX_FACES (EPA_MAX_VERTICES * 2)
#define EPA_MAX_ITERATIONS 255
#define EPA_ACCURACY ((b3Scalar)0.0001)
#define EPA_FALLBACK (10 * EPA_ACCURACY)
#define EPA_PLANE_EPS ((b3Scalar)0.00001)
#define EPA_INSIDE_EPS ((b3Scalar)0.01)

// Shorthands

// MinkowskiDiff
struct b3MinkowskiDiff
{
	const b3ConvexPolyhedronData* m_shapes[2];

	b3Matrix3x3 m_toshape1;
	b3Transform m_toshape0;

	bool m_enableMargin;

	void EnableMargin(bool enable)
	{
		m_enableMargin = enable;
	}
	inline b3Vector3 Support0(const b3Vector3& d, const b3AlignedObjectArray<b3Vector3>& verticesA) const
	{
		if (m_enableMargin)
		{
			return localGetSupportVertexWithMargin(d, m_shapes[0], verticesA, 0.f);
		}
		else
		{
			return localGetSupportVertexWithoutMargin(d, m_shapes[0], verticesA);
		}
	}
	inline b3Vector3 Support1(const b3Vector3& d, const b3AlignedObjectArray<b3Vector3>& verticesB) const
	{
		if (m_enableMargin)
		{
			return m_toshape0 * (localGetSupportVertexWithMargin(m_toshape1 * d, m_shapes[1], verticesB, 0.f));
		}
		else
		{
			return m_toshape0 * (localGetSupportVertexWithoutMargin(m_toshape1 * d, m_shapes[1], verticesB));
		}
	}

	inline b3Vector3 Support(const b3Vector3& d, const b3AlignedObjectArray<b3Vector3>& verticesA, const b3AlignedObjectArray<b3Vector3>& verticesB) const
	{
		return (Support0(d, verticesA) - Support1(-d, verticesB));
	}
	b3Vector3 Support(const b3Vector3& d, unsigned int index, const b3AlignedObjectArray<b3Vector3>& verticesA, const b3AlignedObjectArray<b3Vector3>& verticesB) const
	{
		if (index)
			return (Support1(d, verticesA));
		else
			return (Support0(d, verticesB));
	}
};

typedef b3MinkowskiDiff tShape;

// GJK
struct b3GJK
{
	/* Types		*/
	struct sSV
	{
		b3Vector3 d, w;
	};
	struct sSimplex
	{
		sSV* c[4];
		b3Scalar p[4];
		unsigned int rank;
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
	const b3AlignedObjectArray<b3Vector3>& m_verticesA;
	const b3AlignedObjectArray<b3Vector3>& m_verticesB;
	b3Vector3 m_ray;
	b3Scalar m_distance;
	sSimplex m_simplices[2];
	sSV m_store[4];
	sSV* m_free[4];
	unsigned int m_nfree;
	unsigned int m_current;
	sSimplex* m_simplex;
	eStatus::_ m_status;
	/* Methods		*/
	b3GJK(const b3AlignedObjectArray<b3Vector3>& verticesA, const b3AlignedObjectArray<b3Vector3>& verticesB)
		: m_verticesA(verticesA), m_verticesB(verticesB)
	{
		Initialize();
	}
	void Initialize()
	{
		m_ray = b3MakeVector3(0, 0, 0);
		m_nfree = 0;
		m_status = eStatus::Failed;
		m_current = 0;
		m_distance = 0;
	}
	eStatus::_ Evaluate(const tShape& shapearg, const b3Vector3& guess)
	{
		unsigned int iterations = 0;
		b3Scalar sqdist = 0;
		b3Scalar alpha = 0;
		b3Vector3 lastw[4];
		unsigned int clastw = 0;
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
		const b3Scalar sqrl = m_ray.length2();
		appendvertice(m_simplices[0], sqrl > 0 ? -m_ray : b3MakeVector3(1, 0, 0));
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
			const unsigned int next = 1 - m_current;
			sSimplex& cs = m_simplices[m_current];
			sSimplex& ns = m_simplices[next];
			/* Check zero							*/
			const b3Scalar rl = m_ray.length();
			if (rl < GJK_MIN_DISTANCE)
			{ /* Touching or inside				*/
				m_status = eStatus::Inside;
				break;
			}
			/* Append new vertice in -'v' direction	*/
			appendvertice(cs, -m_ray);
			const b3Vector3& w = cs.c[cs.rank - 1]->w;
			bool found = false;
			for (unsigned int i = 0; i < 4; ++i)
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
			const b3Scalar omega = b3Dot(m_ray, w) / rl;
			alpha = b3Max(omega, alpha);
			if (((rl - alpha) - (GJK_ACCURACY * rl)) <= 0)
			{ /* Return old simplex				*/
				removevertice(m_simplices[m_current]);
				break;
			}
			/* Reduce simplex						*/
			b3Scalar weights[4];
			unsigned int mask = 0;
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
				m_ray = b3MakeVector3(0, 0, 0);
				m_current = next;
				for (unsigned int i = 0, ni = cs.rank; i < ni; ++i)
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
				for (unsigned int i = 0; i < 3; ++i)
				{
					b3Vector3 axis = b3MakeVector3(0, 0, 0);
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
				const b3Vector3 d = m_simplex->c[1]->w - m_simplex->c[0]->w;
				for (unsigned int i = 0; i < 3; ++i)
				{
					b3Vector3 axis = b3MakeVector3(0, 0, 0);
					axis[i] = 1;
					const b3Vector3 p = b3Cross(d, axis);
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
				const b3Vector3 n = b3Cross(m_simplex->c[1]->w - m_simplex->c[0]->w,
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
				if (b3Fabs(det(m_simplex->c[0]->w - m_simplex->c[3]->w,
							   m_simplex->c[1]->w - m_simplex->c[3]->w,
							   m_simplex->c[2]->w - m_simplex->c[3]->w)) > 0)
					return (true);
			}
			break;
		}
		return (false);
	}
	/* Internals	*/
	void getsupport(const b3Vector3& d, sSV& sv) const
	{
		sv.d = d / d.length();
		sv.w = m_shape.Support(sv.d, m_verticesA, m_verticesB);
	}
	void removevertice(sSimplex& simplex)
	{
		m_free[m_nfree++] = simplex.c[--simplex.rank];
	}
	void appendvertice(sSimplex& simplex, const b3Vector3& v)
	{
		simplex.p[simplex.rank] = 0;
		simplex.c[simplex.rank] = m_free[--m_nfree];
		getsupport(v, *simplex.c[simplex.rank++]);
	}
	static b3Scalar det(const b3Vector3& a, const b3Vector3& b, const b3Vector3& c)
	{
		return (a.y * b.z * c.x + a.z * b.x * c.y -
				a.x * b.z * c.y - a.y * b.x * c.z +
				a.x * b.y * c.z - a.z * b.y * c.x);
	}
	static b3Scalar projectorigin(const b3Vector3& a,
								  const b3Vector3& b,
								  b3Scalar* w, unsigned int& m)
	{
		const b3Vector3 d = b - a;
		const b3Scalar l = d.length2();
		if (l > GJK_SIMPLEX2_EPS)
		{
			const b3Scalar t(l > 0 ? -b3Dot(a, d) / l : 0);
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
	static b3Scalar projectorigin(const b3Vector3& a,
								  const b3Vector3& b,
								  const b3Vector3& c,
								  b3Scalar* w, unsigned int& m)
	{
		static const unsigned int imd3[] = {1, 2, 0};
		const b3Vector3* vt[] = {&a, &b, &c};
		const b3Vector3 dl[] = {a - b, b - c, c - a};
		const b3Vector3 n = b3Cross(dl[0], dl[1]);
		const b3Scalar l = n.length2();
		if (l > GJK_SIMPLEX3_EPS)
		{
			b3Scalar mindist = -1;
			b3Scalar subw[2] = {0.f, 0.f};
			unsigned int subm(0);
			for (unsigned int i = 0; i < 3; ++i)
			{
				if (b3Dot(*vt[i], b3Cross(dl[i], n)) > 0)
				{
					const unsigned int j = imd3[i];
					const b3Scalar subd(projectorigin(*vt[i], *vt[j], subw, subm));
					if ((mindist < 0) || (subd < mindist))
					{
						mindist = subd;
						m = static_cast<unsigned int>(((subm & 1) ? 1 << i : 0) + ((subm & 2) ? 1 << j : 0));
						w[i] = subw[0];
						w[j] = subw[1];
						w[imd3[j]] = 0;
					}
				}
			}
			if (mindist < 0)
			{
				const b3Scalar d = b3Dot(a, n);
				const b3Scalar s = b3Sqrt(l);
				const b3Vector3 p = n * (d / l);
				mindist = p.length2();
				m = 7;
				w[0] = (b3Cross(dl[1], b - p)).length() / s;
				w[1] = (b3Cross(dl[2], c - p)).length() / s;
				w[2] = 1 - (w[0] + w[1]);
			}
			return (mindist);
		}
		return (-1);
	}
	static b3Scalar projectorigin(const b3Vector3& a,
								  const b3Vector3& b,
								  const b3Vector3& c,
								  const b3Vector3& d,
								  b3Scalar* w, unsigned int& m)
	{
		static const unsigned int imd3[] = {1, 2, 0};
		const b3Vector3* vt[] = {&a, &b, &c, &d};
		const b3Vector3 dl[] = {a - d, b - d, c - d};
		const b3Scalar vl = det(dl[0], dl[1], dl[2]);
		const bool ng = (vl * b3Dot(a, b3Cross(b - c, a - b))) <= 0;
		if (ng && (b3Fabs(vl) > GJK_SIMPLEX4_EPS))
		{
			b3Scalar mindist = -1;
			b3Scalar subw[3] = {0.f, 0.f, 0.f};
			unsigned int subm(0);
			for (unsigned int i = 0; i < 3; ++i)
			{
				const unsigned int j = imd3[i];
				const b3Scalar s = vl * b3Dot(d, b3Cross(dl[i], dl[j]));
				if (s > 0)
				{
					const b3Scalar subd = projectorigin(*vt[i], *vt[j], d, subw, subm);
					if ((mindist < 0) || (subd < mindist))
					{
						mindist = subd;
						m = static_cast<unsigned int>((subm & 1 ? 1 << i : 0) +
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
struct b3EPA
{
	/* Types		*/
	typedef b3GJK::sSV sSV;
	struct sFace
	{
		b3Vector3 n;
		b3Scalar d;
		sSV* c[3];
		sFace* f[3];
		sFace* l[2];
		unsigned char e[3];
		unsigned char pass;
	};
	struct sList
	{
		sFace* root;
		unsigned int count;
		sList() : root(0), count(0) {}
	};
	struct sHorizon
	{
		sFace* cf;
		sFace* ff;
		unsigned int nf;
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
	b3GJK::sSimplex m_result;
	b3Vector3 m_normal;
	b3Scalar m_depth;
	sSV m_sv_store[EPA_MAX_VERTICES];
	sFace m_fc_store[EPA_MAX_FACES];
	unsigned int m_nextsv;
	sList m_hull;
	sList m_stock;
	/* Methods		*/
	b3EPA()
	{
		Initialize();
	}

	static inline void bind(sFace* fa, unsigned int ea, sFace* fb, unsigned int eb)
	{
		fa->e[ea] = (unsigned char)eb;
		fa->f[ea] = fb;
		fb->e[eb] = (unsigned char)ea;
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
		m_normal = b3MakeVector3(0, 0, 0);
		m_depth = 0;
		m_nextsv = 0;
		for (unsigned int i = 0; i < EPA_MAX_FACES; ++i)
		{
			append(m_stock, &m_fc_store[EPA_MAX_FACES - i - 1]);
		}
	}
	eStatus::_ Evaluate(b3GJK& gjk, const b3Vector3& guess)
	{
		b3GJK::sSimplex& simplex = *gjk.m_simplex;
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
				b3Swap(simplex.c[0], simplex.c[1]);
				b3Swap(simplex.p[0], simplex.p[1]);
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
				unsigned int pass = 0;
				unsigned int iterations = 0;
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
						best->pass = (unsigned char)(++pass);
						gjk.getsupport(best->n, *w);
						const b3Scalar wdist = b3Dot(best->n, w->w) - best->d;
						if (wdist > EPA_ACCURACY)
						{
							for (unsigned int j = 0; (j < 3) && valid; ++j)
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
								m_status = eStatus::Failed;
								//m_status=eStatus::InvalidHull;
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
				const b3Vector3 projection = outer.n * outer.d;
				m_normal = outer.n;
				m_depth = outer.d;
				m_result.rank = 3;
				m_result.c[0] = outer.c[0];
				m_result.c[1] = outer.c[1];
				m_result.c[2] = outer.c[2];
				m_result.p[0] = b3Cross(outer.c[1]->w - projection,
										outer.c[2]->w - projection)
									.length();
				m_result.p[1] = b3Cross(outer.c[2]->w - projection,
										outer.c[0]->w - projection)
									.length();
				m_result.p[2] = b3Cross(outer.c[0]->w - projection,
										outer.c[1]->w - projection)
									.length();
				const b3Scalar sum = m_result.p[0] + m_result.p[1] + m_result.p[2];
				m_result.p[0] /= sum;
				m_result.p[1] /= sum;
				m_result.p[2] /= sum;
				return (m_status);
			}
		}
		/* Fallback		*/
		m_status = eStatus::FallBack;
		m_normal = -guess;
		const b3Scalar nl = m_normal.length();
		if (nl > 0)
			m_normal = m_normal / nl;
		else
			m_normal = b3MakeVector3(1, 0, 0);
		m_depth = 0;
		m_result.rank = 1;
		m_result.c[0] = simplex.c[0];
		m_result.p[0] = 1;
		return (m_status);
	}
	bool getedgedist(sFace* face, sSV* a, sSV* b, b3Scalar& dist)
	{
		const b3Vector3 ba = b->w - a->w;
		const b3Vector3 n_ab = b3Cross(ba, face->n);   // Outward facing edge normal direction, on triangle plane
		const b3Scalar a_dot_nab = b3Dot(a->w, n_ab);  // Only care about the sign to determine inside/outside, so not normalization required

		if (a_dot_nab < 0)
		{
			// Outside of edge a->b

			const b3Scalar ba_l2 = ba.length2();
			const b3Scalar a_dot_ba = b3Dot(a->w, ba);
			const b3Scalar b_dot_ba = b3Dot(b->w, ba);

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
				const b3Scalar a_dot_b = b3Dot(a->w, b->w);
				dist = b3Sqrt(b3Max((a->w.length2() * b->w.length2() - a_dot_b * a_dot_b) / ba_l2, (b3Scalar)0));
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
			face->n = b3Cross(b->w - a->w, c->w - a->w);
			const b3Scalar l = face->n.length();
			const bool v = l > EPA_ACCURACY;

			if (v)
			{
				if (!(getedgedist(face, a, b, face->d) ||
					  getedgedist(face, b, c, face->d) ||
					  getedgedist(face, c, a, face->d)))
				{
					// Origin projects to the interior of the triangle
					// Use distance to triangle plane
					face->d = b3Dot(a->w, face->n) / l;
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
		b3Scalar mind = minf->d * minf->d;
		for (sFace* f = minf->l[1]; f; f = f->l[1])
		{
			const b3Scalar sqd = f->d * f->d;
			if (sqd < mind)
			{
				minf = f;
				mind = sqd;
			}
		}
		return (minf);
	}
	bool expand(unsigned int pass, sSV* w, sFace* f, unsigned int e, sHorizon& horizon)
	{
		static const unsigned int i1m3[] = {1, 2, 0};
		static const unsigned int i2m3[] = {2, 0, 1};
		if (f->pass != pass)
		{
			const unsigned int e1 = i1m3[e];
			if ((b3Dot(f->n, w->w) - f->d) < -EPA_PLANE_EPS)
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
				const unsigned int e2 = i2m3[e];
				f->pass = (unsigned char)pass;
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
static void Initialize(const b3Transform& transA, const b3Transform& transB,
					   const b3ConvexPolyhedronData* hullA, const b3ConvexPolyhedronData* hullB,
					   const b3AlignedObjectArray<b3Vector3>& verticesA,
					   const b3AlignedObjectArray<b3Vector3>& verticesB,
					   b3GjkEpaSolver2::sResults& results,
					   tShape& shape,
					   bool withmargins)
{
	/* Results		*/
	results.witnesses[0] =
		results.witnesses[1] = b3MakeVector3(0, 0, 0);
	results.status = b3GjkEpaSolver2::sResults::Separated;
	/* Shape		*/
	shape.m_shapes[0] = hullA;
	shape.m_shapes[1] = hullB;
	shape.m_toshape1 = transB.getBasis().transposeTimes(transA.getBasis());
	shape.m_toshape0 = transA.inverseTimes(transB);
	shape.EnableMargin(withmargins);
}

}  // namespace gjkepa2_impl2

//
// Api
//

using namespace gjkepa2_impl2;

//
int b3GjkEpaSolver2::StackSizeRequirement()
{
	return (sizeof(b3GJK) + sizeof(b3EPA));
}

//
bool b3GjkEpaSolver2::Distance(const b3Transform& transA, const b3Transform& transB,
							   const b3ConvexPolyhedronData* hullA, const b3ConvexPolyhedronData* hullB,
							   const b3AlignedObjectArray<b3Vector3>& verticesA,
							   const b3AlignedObjectArray<b3Vector3>& verticesB,
							   const b3Vector3& guess,
							   sResults& results)
{
	tShape shape;
	Initialize(transA, transB, hullA, hullB, verticesA, verticesB, results, shape, false);
	b3GJK gjk(verticesA, verticesB);
	b3GJK::eStatus::_ gjk_status = gjk.Evaluate(shape, guess);
	if (gjk_status == b3GJK::eStatus::Valid)
	{
		b3Vector3 w0 = b3MakeVector3(0, 0, 0);
		b3Vector3 w1 = b3MakeVector3(0, 0, 0);
		for (unsigned int i = 0; i < gjk.m_simplex->rank; ++i)
		{
			const b3Scalar p = gjk.m_simplex->p[i];
			w0 += shape.Support(gjk.m_simplex->c[i]->d, 0, verticesA, verticesB) * p;
			w1 += shape.Support(-gjk.m_simplex->c[i]->d, 1, verticesA, verticesB) * p;
		}
		results.witnesses[0] = transA * w0;
		results.witnesses[1] = transA * w1;
		results.normal = w0 - w1;
		results.distance = results.normal.length();
		results.normal /= results.distance > GJK_MIN_DISTANCE ? results.distance : 1;
		return (true);
	}
	else
	{
		results.status = gjk_status == b3GJK::eStatus::Inside ? sResults::Penetrating : sResults::GJK_Failed;
		return (false);
	}
}

//
bool b3GjkEpaSolver2::Penetration(const b3Transform& transA, const b3Transform& transB,
								  const b3ConvexPolyhedronData* hullA, const b3ConvexPolyhedronData* hullB,
								  const b3AlignedObjectArray<b3Vector3>& verticesA,
								  const b3AlignedObjectArray<b3Vector3>& verticesB,
								  const b3Vector3& guess,
								  sResults& results,
								  bool usemargins)
{
	tShape shape;
	Initialize(transA, transB, hullA, hullB, verticesA, verticesB, results, shape, usemargins);
	b3GJK gjk(verticesA, verticesB);
	b3GJK::eStatus::_ gjk_status = gjk.Evaluate(shape, guess);
	switch (gjk_status)
	{
		case b3GJK::eStatus::Inside:
		{
			b3EPA epa;
			b3EPA::eStatus::_ epa_status = epa.Evaluate(gjk, -guess);
			if (epa_status != b3EPA::eStatus::Failed)
			{
				b3Vector3 w0 = b3MakeVector3(0, 0, 0);
				for (unsigned int i = 0; i < epa.m_result.rank; ++i)
				{
					w0 += shape.Support(epa.m_result.c[i]->d, 0, verticesA, verticesB) * epa.m_result.p[i];
				}
				results.status = sResults::Penetrating;
				results.witnesses[0] = transA * w0;
				results.witnesses[1] = transA * (w0 - epa.m_normal * epa.m_depth);
				results.normal = -epa.m_normal;
				results.distance = -epa.m_depth;
				return (true);
			}
			else
				results.status = sResults::EPA_Failed;
		}
		break;
		case b3GJK::eStatus::Failed:
			results.status = sResults::GJK_Failed;
			break;
		default:
		{
		}
	}
	return (false);
}

#if 0
//
b3Scalar	b3GjkEpaSolver2::SignedDistance(const b3Vector3& position,
											b3Scalar margin,
											const b3Transform&	transA,
											const b3ConvexPolyhedronData& hullA, 
											const b3AlignedObjectArray<b3Vector3>& verticesA,
											sResults& results)
{
	tShape			shape;
	btSphereShape	shape1(margin);
	b3Transform		wtrs1(b3Quaternion(0,0,0,1),position);
	Initialize(shape0,wtrs0,&shape1,wtrs1,results,shape,false);
	GJK				gjk;	
	GJK::eStatus::_	gjk_status=gjk.Evaluate(shape,b3Vector3(1,1,1));
	if(gjk_status==GJK::eStatus::Valid)
	{
		b3Vector3	w0=b3Vector3(0,0,0);
		b3Vector3	w1=b3Vector3(0,0,0);
		for(unsigned int i=0;i<gjk.m_simplex->rank;++i)
		{
			const b3Scalar	p=gjk.m_simplex->p[i];
			w0+=shape.Support( gjk.m_simplex->c[i]->d,0)*p;
			w1+=shape.Support(-gjk.m_simplex->c[i]->d,1)*p;
		}
		results.witnesses[0]	=	wtrs0*w0;
		results.witnesses[1]	=	wtrs0*w1;
		const b3Vector3	delta=	results.witnesses[1]-
			results.witnesses[0];
		const b3Scalar	margin=	shape0->getMarginNonVirtual()+
			shape1.getMarginNonVirtual();
		const b3Scalar	length=	delta.length();	
		results.normal			=	delta/length;
		results.witnesses[0]	+=	results.normal*margin;
		return(length-margin);
	}
	else
	{
		if(gjk_status==GJK::eStatus::Inside)
		{
			if(Penetration(shape0,wtrs0,&shape1,wtrs1,gjk.m_ray,results))
			{
				const b3Vector3	delta=	results.witnesses[0]-
					results.witnesses[1];
				const b3Scalar	length=	delta.length();
				if (length >= B3_EPSILON)
					results.normal	=	delta/length;			
				return(-length);
			}
		}	
	}
	return(B3_INFINITY);
}

//
bool	b3GjkEpaSolver2::SignedDistance(const btConvexShape*	shape0,
										const b3Transform&		wtrs0,
										const btConvexShape*	shape1,
										const b3Transform&		wtrs1,
										const b3Vector3&		guess,
										sResults&				results)
{
	if(!Distance(shape0,wtrs0,shape1,wtrs1,guess,results))
		return(Penetration(shape0,wtrs0,shape1,wtrs1,guess,results,false));
	else
		return(true);
}
#endif

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

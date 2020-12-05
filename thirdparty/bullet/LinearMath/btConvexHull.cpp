/*
Stan Melax Convex Hull Computation
Copyright (c) 2003-2006 Stan Melax http://www.melax.com/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include <string.h>

#include "btConvexHull.h"
#include "btAlignedObjectArray.h"
#include "btMinMax.h"
#include "btVector3.h"

//----------------------------------

class int3
{
public:
	int x, y, z;
	int3(){};
	int3(int _x, int _y, int _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}
	const int &operator[](int i) const { return (&x)[i]; }
	int &operator[](int i) { return (&x)[i]; }
};

//------- btPlane ----------

inline btPlane PlaneFlip(const btPlane &plane) { return btPlane(-plane.normal, -plane.dist); }
inline int operator==(const btPlane &a, const btPlane &b) { return (a.normal == b.normal && a.dist == b.dist); }
inline int coplanar(const btPlane &a, const btPlane &b) { return (a == b || a == PlaneFlip(b)); }

//--------- Utility Functions ------

btVector3 PlaneLineIntersection(const btPlane &plane, const btVector3 &p0, const btVector3 &p1);
btVector3 PlaneProject(const btPlane &plane, const btVector3 &point);

btVector3 ThreePlaneIntersection(const btPlane &p0, const btPlane &p1, const btPlane &p2);
btVector3 ThreePlaneIntersection(const btPlane &p0, const btPlane &p1, const btPlane &p2)
{
	btVector3 N1 = p0.normal;
	btVector3 N2 = p1.normal;
	btVector3 N3 = p2.normal;

	btVector3 n2n3;
	n2n3 = N2.cross(N3);
	btVector3 n3n1;
	n3n1 = N3.cross(N1);
	btVector3 n1n2;
	n1n2 = N1.cross(N2);

	btScalar quotient = (N1.dot(n2n3));

	btAssert(btFabs(quotient) > btScalar(0.000001));

	quotient = btScalar(-1.) / quotient;
	n2n3 *= p0.dist;
	n3n1 *= p1.dist;
	n1n2 *= p2.dist;
	btVector3 potentialVertex = n2n3;
	potentialVertex += n3n1;
	potentialVertex += n1n2;
	potentialVertex *= quotient;

	btVector3 result(potentialVertex.getX(), potentialVertex.getY(), potentialVertex.getZ());
	return result;
}

btScalar DistanceBetweenLines(const btVector3 &ustart, const btVector3 &udir, const btVector3 &vstart, const btVector3 &vdir, btVector3 *upoint = NULL, btVector3 *vpoint = NULL);
btVector3 TriNormal(const btVector3 &v0, const btVector3 &v1, const btVector3 &v2);
btVector3 NormalOf(const btVector3 *vert, const int n);

btVector3 PlaneLineIntersection(const btPlane &plane, const btVector3 &p0, const btVector3 &p1)
{
	// returns the point where the line p0-p1 intersects the plane n&d
	btVector3 dif;
	dif = p1 - p0;
	btScalar dn = btDot(plane.normal, dif);
	btScalar t = -(plane.dist + btDot(plane.normal, p0)) / dn;
	return p0 + (dif * t);
}

btVector3 PlaneProject(const btPlane &plane, const btVector3 &point)
{
	return point - plane.normal * (btDot(point, plane.normal) + plane.dist);
}

btVector3 TriNormal(const btVector3 &v0, const btVector3 &v1, const btVector3 &v2)
{
	// return the normal of the triangle
	// inscribed by v0, v1, and v2
	btVector3 cp = btCross(v1 - v0, v2 - v1);
	btScalar m = cp.length();
	if (m == 0) return btVector3(1, 0, 0);
	return cp * (btScalar(1.0) / m);
}

btScalar DistanceBetweenLines(const btVector3 &ustart, const btVector3 &udir, const btVector3 &vstart, const btVector3 &vdir, btVector3 *upoint, btVector3 *vpoint)
{
	btVector3 cp;
	cp = btCross(udir, vdir).normalized();

	btScalar distu = -btDot(cp, ustart);
	btScalar distv = -btDot(cp, vstart);
	btScalar dist = (btScalar)fabs(distu - distv);
	if (upoint)
	{
		btPlane plane;
		plane.normal = btCross(vdir, cp).normalized();
		plane.dist = -btDot(plane.normal, vstart);
		*upoint = PlaneLineIntersection(plane, ustart, ustart + udir);
	}
	if (vpoint)
	{
		btPlane plane;
		plane.normal = btCross(udir, cp).normalized();
		plane.dist = -btDot(plane.normal, ustart);
		*vpoint = PlaneLineIntersection(plane, vstart, vstart + vdir);
	}
	return dist;
}

#define COPLANAR (0)
#define UNDER (1)
#define OVER (2)
#define SPLIT (OVER | UNDER)
#define PAPERWIDTH (btScalar(0.001))

btScalar planetestepsilon = PAPERWIDTH;

typedef ConvexH::HalfEdge HalfEdge;

ConvexH::ConvexH(int vertices_size, int edges_size, int facets_size)
{
	vertices.resize(vertices_size);
	edges.resize(edges_size);
	facets.resize(facets_size);
}

int PlaneTest(const btPlane &p, const btVector3 &v);
int PlaneTest(const btPlane &p, const btVector3 &v)
{
	btScalar a = btDot(v, p.normal) + p.dist;
	int flag = (a > planetestepsilon) ? OVER : ((a < -planetestepsilon) ? UNDER : COPLANAR);
	return flag;
}

int SplitTest(ConvexH &convex, const btPlane &plane);
int SplitTest(ConvexH &convex, const btPlane &plane)
{
	int flag = 0;
	for (int i = 0; i < convex.vertices.size(); i++)
	{
		flag |= PlaneTest(plane, convex.vertices[i]);
	}
	return flag;
}

class VertFlag
{
public:
	unsigned char planetest;
	unsigned char junk;
	unsigned char undermap;
	unsigned char overmap;
};
class EdgeFlag
{
public:
	unsigned char planetest;
	unsigned char fixes;
	short undermap;
	short overmap;
};
class PlaneFlag
{
public:
	unsigned char undermap;
	unsigned char overmap;
};
class Coplanar
{
public:
	unsigned short ea;
	unsigned char v0;
	unsigned char v1;
};

template <class T>
int maxdirfiltered(const T *p, int count, const T &dir, btAlignedObjectArray<int> &allow)
{
	btAssert(count);
	int m = -1;
	for (int i = 0; i < count; i++)
		if (allow[i])
		{
			if (m == -1 || btDot(p[i], dir) > btDot(p[m], dir))
				m = i;
		}
	btAssert(m != -1);
	return m;
}

btVector3 orth(const btVector3 &v);
btVector3 orth(const btVector3 &v)
{
	btVector3 a = btCross(v, btVector3(0, 0, 1));
	btVector3 b = btCross(v, btVector3(0, 1, 0));
	if (a.length() > b.length())
	{
		return a.normalized();
	}
	else
	{
		return b.normalized();
	}
}

template <class T>
int maxdirsterid(const T *p, int count, const T &dir, btAlignedObjectArray<int> &allow)
{
	int m = -1;
	while (m == -1)
	{
		m = maxdirfiltered(p, count, dir, allow);
		if (allow[m] == 3) return m;
		T u = orth(dir);
		T v = btCross(u, dir);
		int ma = -1;
		for (btScalar x = btScalar(0.0); x <= btScalar(360.0); x += btScalar(45.0))
		{
			btScalar s = btSin(SIMD_RADS_PER_DEG * (x));
			btScalar c = btCos(SIMD_RADS_PER_DEG * (x));
			int mb = maxdirfiltered(p, count, dir + (u * s + v * c) * btScalar(0.025), allow);
			if (ma == m && mb == m)
			{
				allow[m] = 3;
				return m;
			}
			if (ma != -1 && ma != mb)  // Yuck - this is really ugly
			{
				int mc = ma;
				for (btScalar xx = x - btScalar(40.0); xx <= x; xx += btScalar(5.0))
				{
					btScalar s = btSin(SIMD_RADS_PER_DEG * (xx));
					btScalar c = btCos(SIMD_RADS_PER_DEG * (xx));
					int md = maxdirfiltered(p, count, dir + (u * s + v * c) * btScalar(0.025), allow);
					if (mc == m && md == m)
					{
						allow[m] = 3;
						return m;
					}
					mc = md;
				}
			}
			ma = mb;
		}
		allow[m] = 0;
		m = -1;
	}
	btAssert(0);
	return m;
}

int operator==(const int3 &a, const int3 &b);
int operator==(const int3 &a, const int3 &b)
{
	for (int i = 0; i < 3; i++)
	{
		if (a[i] != b[i]) return 0;
	}
	return 1;
}

int above(btVector3 *vertices, const int3 &t, const btVector3 &p, btScalar epsilon);
int above(btVector3 *vertices, const int3 &t, const btVector3 &p, btScalar epsilon)
{
	btVector3 n = TriNormal(vertices[t[0]], vertices[t[1]], vertices[t[2]]);
	return (btDot(n, p - vertices[t[0]]) > epsilon);  // EPSILON???
}
int hasedge(const int3 &t, int a, int b);
int hasedge(const int3 &t, int a, int b)
{
	for (int i = 0; i < 3; i++)
	{
		int i1 = (i + 1) % 3;
		if (t[i] == a && t[i1] == b) return 1;
	}
	return 0;
}
int hasvert(const int3 &t, int v);
int hasvert(const int3 &t, int v)
{
	return (t[0] == v || t[1] == v || t[2] == v);
}
int shareedge(const int3 &a, const int3 &b);
int shareedge(const int3 &a, const int3 &b)
{
	int i;
	for (i = 0; i < 3; i++)
	{
		int i1 = (i + 1) % 3;
		if (hasedge(a, b[i1], b[i])) return 1;
	}
	return 0;
}

class btHullTriangle;

class btHullTriangle : public int3
{
public:
	int3 n;
	int id;
	int vmax;
	btScalar rise;
	btHullTriangle(int a, int b, int c) : int3(a, b, c), n(-1, -1, -1)
	{
		vmax = -1;
		rise = btScalar(0.0);
	}
	~btHullTriangle()
	{
	}
	int &neib(int a, int b);
};

int &btHullTriangle::neib(int a, int b)
{
	static int er = -1;
	int i;
	for (i = 0; i < 3; i++)
	{
		int i1 = (i + 1) % 3;
		int i2 = (i + 2) % 3;
		if ((*this)[i] == a && (*this)[i1] == b) return n[i2];
		if ((*this)[i] == b && (*this)[i1] == a) return n[i2];
	}
	btAssert(0);
	return er;
}
void HullLibrary::b2bfix(btHullTriangle *s, btHullTriangle *t)
{
	int i;
	for (i = 0; i < 3; i++)
	{
		int i1 = (i + 1) % 3;
		int i2 = (i + 2) % 3;
		int a = (*s)[i1];
		int b = (*s)[i2];
		btAssert(m_tris[s->neib(a, b)]->neib(b, a) == s->id);
		btAssert(m_tris[t->neib(a, b)]->neib(b, a) == t->id);
		m_tris[s->neib(a, b)]->neib(b, a) = t->neib(b, a);
		m_tris[t->neib(b, a)]->neib(a, b) = s->neib(a, b);
	}
}

void HullLibrary::removeb2b(btHullTriangle *s, btHullTriangle *t)
{
	b2bfix(s, t);
	deAllocateTriangle(s);

	deAllocateTriangle(t);
}

void HullLibrary::checkit(btHullTriangle *t)
{
	(void)t;

	int i;
	btAssert(m_tris[t->id] == t);
	for (i = 0; i < 3; i++)
	{
		int i1 = (i + 1) % 3;
		int i2 = (i + 2) % 3;
		int a = (*t)[i1];
		int b = (*t)[i2];

		// release compile fix
		(void)i1;
		(void)i2;
		(void)a;
		(void)b;

		btAssert(a != b);
		btAssert(m_tris[t->n[i]]->neib(b, a) == t->id);
	}
}

btHullTriangle *HullLibrary::allocateTriangle(int a, int b, int c)
{
	void *mem = btAlignedAlloc(sizeof(btHullTriangle), 16);
	btHullTriangle *tr = new (mem) btHullTriangle(a, b, c);
	tr->id = m_tris.size();
	m_tris.push_back(tr);

	return tr;
}

void HullLibrary::deAllocateTriangle(btHullTriangle *tri)
{
	btAssert(m_tris[tri->id] == tri);
	m_tris[tri->id] = NULL;
	tri->~btHullTriangle();
	btAlignedFree(tri);
}

void HullLibrary::extrude(btHullTriangle *t0, int v)
{
	int3 t = *t0;
	int n = m_tris.size();
	btHullTriangle *ta = allocateTriangle(v, t[1], t[2]);
	ta->n = int3(t0->n[0], n + 1, n + 2);
	m_tris[t0->n[0]]->neib(t[1], t[2]) = n + 0;
	btHullTriangle *tb = allocateTriangle(v, t[2], t[0]);
	tb->n = int3(t0->n[1], n + 2, n + 0);
	m_tris[t0->n[1]]->neib(t[2], t[0]) = n + 1;
	btHullTriangle *tc = allocateTriangle(v, t[0], t[1]);
	tc->n = int3(t0->n[2], n + 0, n + 1);
	m_tris[t0->n[2]]->neib(t[0], t[1]) = n + 2;
	checkit(ta);
	checkit(tb);
	checkit(tc);
	if (hasvert(*m_tris[ta->n[0]], v)) removeb2b(ta, m_tris[ta->n[0]]);
	if (hasvert(*m_tris[tb->n[0]], v)) removeb2b(tb, m_tris[tb->n[0]]);
	if (hasvert(*m_tris[tc->n[0]], v)) removeb2b(tc, m_tris[tc->n[0]]);
	deAllocateTriangle(t0);
}

btHullTriangle *HullLibrary::extrudable(btScalar epsilon)
{
	int i;
	btHullTriangle *t = NULL;
	for (i = 0; i < m_tris.size(); i++)
	{
		if (!t || (m_tris[i] && t->rise < m_tris[i]->rise))
		{
			t = m_tris[i];
		}
	}
	return (t->rise > epsilon) ? t : NULL;
}

int4 HullLibrary::FindSimplex(btVector3 *verts, int verts_count, btAlignedObjectArray<int> &allow)
{
	btVector3 basis[3];
	basis[0] = btVector3(btScalar(0.01), btScalar(0.02), btScalar(1.0));
	int p0 = maxdirsterid(verts, verts_count, basis[0], allow);
	int p1 = maxdirsterid(verts, verts_count, -basis[0], allow);
	basis[0] = verts[p0] - verts[p1];
	if (p0 == p1 || basis[0] == btVector3(0, 0, 0))
		return int4(-1, -1, -1, -1);
	basis[1] = btCross(btVector3(btScalar(1), btScalar(0.02), btScalar(0)), basis[0]);
	basis[2] = btCross(btVector3(btScalar(-0.02), btScalar(1), btScalar(0)), basis[0]);
	if (basis[1].length() > basis[2].length())
	{
		basis[1].normalize();
	}
	else
	{
		basis[1] = basis[2];
		basis[1].normalize();
	}
	int p2 = maxdirsterid(verts, verts_count, basis[1], allow);
	if (p2 == p0 || p2 == p1)
	{
		p2 = maxdirsterid(verts, verts_count, -basis[1], allow);
	}
	if (p2 == p0 || p2 == p1)
		return int4(-1, -1, -1, -1);
	basis[1] = verts[p2] - verts[p0];
	basis[2] = btCross(basis[1], basis[0]).normalized();
	int p3 = maxdirsterid(verts, verts_count, basis[2], allow);
	if (p3 == p0 || p3 == p1 || p3 == p2) p3 = maxdirsterid(verts, verts_count, -basis[2], allow);
	if (p3 == p0 || p3 == p1 || p3 == p2)
		return int4(-1, -1, -1, -1);
	btAssert(!(p0 == p1 || p0 == p2 || p0 == p3 || p1 == p2 || p1 == p3 || p2 == p3));
	if (btDot(verts[p3] - verts[p0], btCross(verts[p1] - verts[p0], verts[p2] - verts[p0])) < 0)
	{
		btSwap(p2, p3);
	}
	return int4(p0, p1, p2, p3);
}

int HullLibrary::calchullgen(btVector3 *verts, int verts_count, int vlimit)
{
	if (verts_count < 4) return 0;
	if (vlimit == 0) vlimit = 1000000000;
	int j;
	btVector3 bmin(*verts), bmax(*verts);
	btAlignedObjectArray<int> isextreme;
	isextreme.reserve(verts_count);
	btAlignedObjectArray<int> allow;
	allow.reserve(verts_count);

	for (j = 0; j < verts_count; j++)
	{
		allow.push_back(1);
		isextreme.push_back(0);
		bmin.setMin(verts[j]);
		bmax.setMax(verts[j]);
	}
	btScalar epsilon = (bmax - bmin).length() * btScalar(0.001);
	btAssert(epsilon != 0.0);

	int4 p = FindSimplex(verts, verts_count, allow);
	if (p.x == -1) return 0;  // simplex failed

	btVector3 center = (verts[p[0]] + verts[p[1]] + verts[p[2]] + verts[p[3]]) / btScalar(4.0);  // a valid interior point
	btHullTriangle *t0 = allocateTriangle(p[2], p[3], p[1]);
	t0->n = int3(2, 3, 1);
	btHullTriangle *t1 = allocateTriangle(p[3], p[2], p[0]);
	t1->n = int3(3, 2, 0);
	btHullTriangle *t2 = allocateTriangle(p[0], p[1], p[3]);
	t2->n = int3(0, 1, 3);
	btHullTriangle *t3 = allocateTriangle(p[1], p[0], p[2]);
	t3->n = int3(1, 0, 2);
	isextreme[p[0]] = isextreme[p[1]] = isextreme[p[2]] = isextreme[p[3]] = 1;
	checkit(t0);
	checkit(t1);
	checkit(t2);
	checkit(t3);

	for (j = 0; j < m_tris.size(); j++)
	{
		btHullTriangle *t = m_tris[j];
		btAssert(t);
		btAssert(t->vmax < 0);
		btVector3 n = TriNormal(verts[(*t)[0]], verts[(*t)[1]], verts[(*t)[2]]);
		t->vmax = maxdirsterid(verts, verts_count, n, allow);
		t->rise = btDot(n, verts[t->vmax] - verts[(*t)[0]]);
	}
	btHullTriangle *te;
	vlimit -= 4;
	while (vlimit > 0 && ((te = extrudable(epsilon)) != 0))
	{
		//int3 ti=*te;
		int v = te->vmax;
		btAssert(v != -1);
		btAssert(!isextreme[v]);  // wtf we've already done this vertex
		isextreme[v] = 1;
		//if(v==p0 || v==p1 || v==p2 || v==p3) continue; // done these already
		j = m_tris.size();
		while (j--)
		{
			if (!m_tris[j]) continue;
			int3 t = *m_tris[j];
			if (above(verts, t, verts[v], btScalar(0.01) * epsilon))
			{
				extrude(m_tris[j], v);
			}
		}
		// now check for those degenerate cases where we have a flipped triangle or a really skinny triangle
		j = m_tris.size();
		while (j--)
		{
			if (!m_tris[j]) continue;
			if (!hasvert(*m_tris[j], v)) break;
			int3 nt = *m_tris[j];
			if (above(verts, nt, center, btScalar(0.01) * epsilon) || btCross(verts[nt[1]] - verts[nt[0]], verts[nt[2]] - verts[nt[1]]).length() < epsilon * epsilon * btScalar(0.1))
			{
				btHullTriangle *nb = m_tris[m_tris[j]->n[0]];
				btAssert(nb);
				btAssert(!hasvert(*nb, v));
				btAssert(nb->id < j);
				extrude(nb, v);
				j = m_tris.size();
			}
		}
		j = m_tris.size();
		while (j--)
		{
			btHullTriangle *t = m_tris[j];
			if (!t) continue;
			if (t->vmax >= 0) break;
			btVector3 n = TriNormal(verts[(*t)[0]], verts[(*t)[1]], verts[(*t)[2]]);
			t->vmax = maxdirsterid(verts, verts_count, n, allow);
			if (isextreme[t->vmax])
			{
				t->vmax = -1;  // already done that vertex - algorithm needs to be able to terminate.
			}
			else
			{
				t->rise = btDot(n, verts[t->vmax] - verts[(*t)[0]]);
			}
		}
		vlimit--;
	}
	return 1;
}

int HullLibrary::calchull(btVector3 *verts, int verts_count, TUIntArray &tris_out, int &tris_count, int vlimit)
{
	int rc = calchullgen(verts, verts_count, vlimit);
	if (!rc) return 0;
	btAlignedObjectArray<int> ts;
	int i;

	for (i = 0; i < m_tris.size(); i++)
	{
		if (m_tris[i])
		{
			for (int j = 0; j < 3; j++)
				ts.push_back((*m_tris[i])[j]);
			deAllocateTriangle(m_tris[i]);
		}
	}
	tris_count = ts.size() / 3;
	tris_out.resize(ts.size());

	for (i = 0; i < ts.size(); i++)
	{
		tris_out[i] = static_cast<unsigned int>(ts[i]);
	}
	m_tris.resize(0);

	return 1;
}

bool HullLibrary::ComputeHull(unsigned int vcount, const btVector3 *vertices, PHullResult &result, unsigned int vlimit)
{
	int tris_count;
	int ret = calchull((btVector3 *)vertices, (int)vcount, result.m_Indices, tris_count, static_cast<int>(vlimit));
	if (!ret) return false;
	result.mIndexCount = (unsigned int)(tris_count * 3);
	result.mFaceCount = (unsigned int)tris_count;
	result.mVertices = (btVector3 *)vertices;
	result.mVcount = (unsigned int)vcount;
	return true;
}

void ReleaseHull(PHullResult &result);
void ReleaseHull(PHullResult &result)
{
	if (result.m_Indices.size())
	{
		result.m_Indices.clear();
	}

	result.mVcount = 0;
	result.mIndexCount = 0;
	result.mVertices = 0;
}

//*********************************************************************
//*********************************************************************
//********  HullLib header
//*********************************************************************
//*********************************************************************

//*********************************************************************
//*********************************************************************
//********  HullLib implementation
//*********************************************************************
//*********************************************************************

HullError HullLibrary::CreateConvexHull(const HullDesc &desc,  // describes the input request
										HullResult &result)    // contains the resulst
{
	HullError ret = QE_FAIL;

	PHullResult hr;

	unsigned int vcount = desc.mVcount;
	if (vcount < 8) vcount = 8;

	btAlignedObjectArray<btVector3> vertexSource;
	vertexSource.resize(static_cast<int>(vcount));

	btVector3 scale;

	unsigned int ovcount;

	bool ok = CleanupVertices(desc.mVcount, desc.mVertices, desc.mVertexStride, ovcount, &vertexSource[0], desc.mNormalEpsilon, scale);  // normalize point cloud, remove duplicates!

	if (ok)
	{
		//		if ( 1 ) // scale vertices back to their original size.
		{
			for (unsigned int i = 0; i < ovcount; i++)
			{
				btVector3 &v = vertexSource[static_cast<int>(i)];
				v[0] *= scale[0];
				v[1] *= scale[1];
				v[2] *= scale[2];
			}
		}

		ok = ComputeHull(ovcount, &vertexSource[0], hr, desc.mMaxVertices);

		if (ok)
		{
			// re-index triangle mesh so it refers to only used vertices, rebuild a new vertex table.
			btAlignedObjectArray<btVector3> vertexScratch;
			vertexScratch.resize(static_cast<int>(hr.mVcount));

			BringOutYourDead(hr.mVertices, hr.mVcount, &vertexScratch[0], ovcount, &hr.m_Indices[0], hr.mIndexCount);

			ret = QE_OK;

			if (desc.HasHullFlag(QF_TRIANGLES))  // if he wants the results as triangle!
			{
				result.mPolygons = false;
				result.mNumOutputVertices = ovcount;
				result.m_OutputVertices.resize(static_cast<int>(ovcount));
				result.mNumFaces = hr.mFaceCount;
				result.mNumIndices = hr.mIndexCount;

				result.m_Indices.resize(static_cast<int>(hr.mIndexCount));

				memcpy(&result.m_OutputVertices[0], &vertexScratch[0], sizeof(btVector3) * ovcount);

				if (desc.HasHullFlag(QF_REVERSE_ORDER))
				{
					const unsigned int *source = &hr.m_Indices[0];
					unsigned int *dest = &result.m_Indices[0];

					for (unsigned int i = 0; i < hr.mFaceCount; i++)
					{
						dest[0] = source[2];
						dest[1] = source[1];
						dest[2] = source[0];
						dest += 3;
						source += 3;
					}
				}
				else
				{
					memcpy(&result.m_Indices[0], &hr.m_Indices[0], sizeof(unsigned int) * hr.mIndexCount);
				}
			}
			else
			{
				result.mPolygons = true;
				result.mNumOutputVertices = ovcount;
				result.m_OutputVertices.resize(static_cast<int>(ovcount));
				result.mNumFaces = hr.mFaceCount;
				result.mNumIndices = hr.mIndexCount + hr.mFaceCount;
				result.m_Indices.resize(static_cast<int>(result.mNumIndices));
				memcpy(&result.m_OutputVertices[0], &vertexScratch[0], sizeof(btVector3) * ovcount);

				//				if ( 1 )
				{
					const unsigned int *source = &hr.m_Indices[0];
					unsigned int *dest = &result.m_Indices[0];
					for (unsigned int i = 0; i < hr.mFaceCount; i++)
					{
						dest[0] = 3;
						if (desc.HasHullFlag(QF_REVERSE_ORDER))
						{
							dest[1] = source[2];
							dest[2] = source[1];
							dest[3] = source[0];
						}
						else
						{
							dest[1] = source[0];
							dest[2] = source[1];
							dest[3] = source[2];
						}

						dest += 4;
						source += 3;
					}
				}
			}
			ReleaseHull(hr);
		}
	}

	return ret;
}

HullError HullLibrary::ReleaseResult(HullResult &result)  // release memory allocated for this result, we are done with it.
{
	if (result.m_OutputVertices.size())
	{
		result.mNumOutputVertices = 0;
		result.m_OutputVertices.clear();
	}
	if (result.m_Indices.size())
	{
		result.mNumIndices = 0;
		result.m_Indices.clear();
	}
	return QE_OK;
}

static void addPoint(unsigned int &vcount, btVector3 *p, btScalar x, btScalar y, btScalar z)
{
	// XXX, might be broken
	btVector3 &dest = p[vcount];
	dest[0] = x;
	dest[1] = y;
	dest[2] = z;
	vcount++;
}

btScalar GetDist(btScalar px, btScalar py, btScalar pz, const btScalar *p2);
btScalar GetDist(btScalar px, btScalar py, btScalar pz, const btScalar *p2)
{
	btScalar dx = px - p2[0];
	btScalar dy = py - p2[1];
	btScalar dz = pz - p2[2];

	return dx * dx + dy * dy + dz * dz;
}

bool HullLibrary::CleanupVertices(unsigned int svcount,
								  const btVector3 *svertices,
								  unsigned int stride,
								  unsigned int &vcount,  // output number of vertices
								  btVector3 *vertices,   // location to store the results.
								  btScalar normalepsilon,
								  btVector3 &scale)
{
	if (svcount == 0) return false;

	m_vertexIndexMapping.resize(0);

#define EPSILON btScalar(0.000001) /* close enough to consider two btScalaring point numbers to be 'the same'. */

	vcount = 0;

	btScalar recip[3] = {0.f, 0.f, 0.f};

	if (scale)
	{
		scale[0] = 1;
		scale[1] = 1;
		scale[2] = 1;
	}

	btScalar bmin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	btScalar bmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	const char *vtx = (const char *)svertices;

	//	if ( 1 )
	{
		for (unsigned int i = 0; i < svcount; i++)
		{
			const btScalar *p = (const btScalar *)vtx;

			vtx += stride;

			for (int j = 0; j < 3; j++)
			{
				if (p[j] < bmin[j]) bmin[j] = p[j];
				if (p[j] > bmax[j]) bmax[j] = p[j];
			}
		}
	}

	btScalar dx = bmax[0] - bmin[0];
	btScalar dy = bmax[1] - bmin[1];
	btScalar dz = bmax[2] - bmin[2];

	btVector3 center;

	center[0] = dx * btScalar(0.5) + bmin[0];
	center[1] = dy * btScalar(0.5) + bmin[1];
	center[2] = dz * btScalar(0.5) + bmin[2];

	if (dx < EPSILON || dy < EPSILON || dz < EPSILON || svcount < 3)
	{
		btScalar len = FLT_MAX;

		if (dx > EPSILON && dx < len) len = dx;
		if (dy > EPSILON && dy < len) len = dy;
		if (dz > EPSILON && dz < len) len = dz;

		if (len == FLT_MAX)
		{
			dx = dy = dz = btScalar(0.01);  // one centimeter
		}
		else
		{
			if (dx < EPSILON) dx = len * btScalar(0.05);  // 1/5th the shortest non-zero edge.
			if (dy < EPSILON) dy = len * btScalar(0.05);
			if (dz < EPSILON) dz = len * btScalar(0.05);
		}

		btScalar x1 = center[0] - dx;
		btScalar x2 = center[0] + dx;

		btScalar y1 = center[1] - dy;
		btScalar y2 = center[1] + dy;

		btScalar z1 = center[2] - dz;
		btScalar z2 = center[2] + dz;

		addPoint(vcount, vertices, x1, y1, z1);
		addPoint(vcount, vertices, x2, y1, z1);
		addPoint(vcount, vertices, x2, y2, z1);
		addPoint(vcount, vertices, x1, y2, z1);
		addPoint(vcount, vertices, x1, y1, z2);
		addPoint(vcount, vertices, x2, y1, z2);
		addPoint(vcount, vertices, x2, y2, z2);
		addPoint(vcount, vertices, x1, y2, z2);

		return true;  // return cube
	}
	else
	{
		if (scale)
		{
			scale[0] = dx;
			scale[1] = dy;
			scale[2] = dz;

			recip[0] = 1 / dx;
			recip[1] = 1 / dy;
			recip[2] = 1 / dz;

			center[0] *= recip[0];
			center[1] *= recip[1];
			center[2] *= recip[2];
		}
	}

	vtx = (const char *)svertices;

	for (unsigned int i = 0; i < svcount; i++)
	{
		const btVector3 *p = (const btVector3 *)vtx;
		vtx += stride;

		btScalar px = p->getX();
		btScalar py = p->getY();
		btScalar pz = p->getZ();

		if (scale)
		{
			px = px * recip[0];  // normalize
			py = py * recip[1];  // normalize
			pz = pz * recip[2];  // normalize
		}

		//		if ( 1 )
		{
			unsigned int j;

			for (j = 0; j < vcount; j++)
			{
				/// XXX might be broken
				btVector3 &v = vertices[j];

				btScalar x = v[0];
				btScalar y = v[1];
				btScalar z = v[2];

				btScalar dx = btFabs(x - px);
				btScalar dy = btFabs(y - py);
				btScalar dz = btFabs(z - pz);

				if (dx < normalepsilon && dy < normalepsilon && dz < normalepsilon)
				{
					// ok, it is close enough to the old one
					// now let us see if it is further from the center of the point cloud than the one we already recorded.
					// in which case we keep this one instead.

					btScalar dist1 = GetDist(px, py, pz, center);
					btScalar dist2 = GetDist(v[0], v[1], v[2], center);

					if (dist1 > dist2)
					{
						v[0] = px;
						v[1] = py;
						v[2] = pz;
					}

					break;
				}
			}

			if (j == vcount)
			{
				btVector3 &dest = vertices[vcount];
				dest[0] = px;
				dest[1] = py;
				dest[2] = pz;
				vcount++;
			}
			m_vertexIndexMapping.push_back(j);
		}
	}

	// ok..now make sure we didn't prune so many vertices it is now invalid.
	//	if ( 1 )
	{
		btScalar bmin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
		btScalar bmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

		for (unsigned int i = 0; i < vcount; i++)
		{
			const btVector3 &p = vertices[i];
			for (int j = 0; j < 3; j++)
			{
				if (p[j] < bmin[j]) bmin[j] = p[j];
				if (p[j] > bmax[j]) bmax[j] = p[j];
			}
		}

		btScalar dx = bmax[0] - bmin[0];
		btScalar dy = bmax[1] - bmin[1];
		btScalar dz = bmax[2] - bmin[2];

		if (dx < EPSILON || dy < EPSILON || dz < EPSILON || vcount < 3)
		{
			btScalar cx = dx * btScalar(0.5) + bmin[0];
			btScalar cy = dy * btScalar(0.5) + bmin[1];
			btScalar cz = dz * btScalar(0.5) + bmin[2];

			btScalar len = FLT_MAX;

			if (dx >= EPSILON && dx < len) len = dx;
			if (dy >= EPSILON && dy < len) len = dy;
			if (dz >= EPSILON && dz < len) len = dz;

			if (len == FLT_MAX)
			{
				dx = dy = dz = btScalar(0.01);  // one centimeter
			}
			else
			{
				if (dx < EPSILON) dx = len * btScalar(0.05);  // 1/5th the shortest non-zero edge.
				if (dy < EPSILON) dy = len * btScalar(0.05);
				if (dz < EPSILON) dz = len * btScalar(0.05);
			}

			btScalar x1 = cx - dx;
			btScalar x2 = cx + dx;

			btScalar y1 = cy - dy;
			btScalar y2 = cy + dy;

			btScalar z1 = cz - dz;
			btScalar z2 = cz + dz;

			vcount = 0;  // add box

			addPoint(vcount, vertices, x1, y1, z1);
			addPoint(vcount, vertices, x2, y1, z1);
			addPoint(vcount, vertices, x2, y2, z1);
			addPoint(vcount, vertices, x1, y2, z1);
			addPoint(vcount, vertices, x1, y1, z2);
			addPoint(vcount, vertices, x2, y1, z2);
			addPoint(vcount, vertices, x2, y2, z2);
			addPoint(vcount, vertices, x1, y2, z2);

			return true;
		}
	}

	return true;
}

void HullLibrary::BringOutYourDead(const btVector3 *verts, unsigned int vcount, btVector3 *overts, unsigned int &ocount, unsigned int *indices, unsigned indexcount)
{
	btAlignedObjectArray<int> tmpIndices;
	tmpIndices.resize(m_vertexIndexMapping.size());
	int i;

	for (i = 0; i < m_vertexIndexMapping.size(); i++)
	{
		tmpIndices[i] = m_vertexIndexMapping[i];
	}

	TUIntArray usedIndices;
	usedIndices.resize(static_cast<int>(vcount));
	memset(&usedIndices[0], 0, sizeof(unsigned int) * vcount);

	ocount = 0;

	for (i = 0; i < int(indexcount); i++)
	{
		unsigned int v = indices[i];  // original array index

		btAssert(v >= 0 && v < vcount);

		if (usedIndices[static_cast<int>(v)])  // if already remapped
		{
			indices[i] = usedIndices[static_cast<int>(v)] - 1;  // index to new array
		}
		else
		{
			indices[i] = ocount;  // new index mapping

			overts[ocount][0] = verts[v][0];  // copy old vert to new vert array
			overts[ocount][1] = verts[v][1];
			overts[ocount][2] = verts[v][2];

			for (int k = 0; k < m_vertexIndexMapping.size(); k++)
			{
				if (tmpIndices[k] == int(v))
					m_vertexIndexMapping[k] = ocount;
			}

			ocount++;  // increment output vert count

			btAssert(ocount >= 0 && ocount <= vcount);

			usedIndices[static_cast<int>(v)] = ocount;  // assign new index remapping
		}
	}
}

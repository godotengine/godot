//
// Copyright (c) 2009-2010 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "Recast.h"
#include "RecastAlloc.h"
#include "RecastAssert.h"


static const unsigned RC_UNSET_HEIGHT = 0xffff;

struct rcHeightPatch
{
	inline rcHeightPatch() : data(0), xmin(0), ymin(0), width(0), height(0) {}
	inline ~rcHeightPatch() { rcFree(data); }
	unsigned short* data;
	int xmin, ymin, width, height;
};


inline float vdot2(const float* a, const float* b)
{
	return a[0]*b[0] + a[2]*b[2];
}

inline float vdistSq2(const float* p, const float* q)
{
	const float dx = q[0] - p[0];
	const float dy = q[2] - p[2];
	return dx*dx + dy*dy;
}

inline float vdist2(const float* p, const float* q)
{
	return sqrtf(vdistSq2(p,q));
}

inline float vcross2(const float* p1, const float* p2, const float* p3)
{
	const float u1 = p2[0] - p1[0];
	const float v1 = p2[2] - p1[2];
	const float u2 = p3[0] - p1[0];
	const float v2 = p3[2] - p1[2];
	return u1 * v2 - v1 * u2;
}

static bool circumCircle(const float* p1, const float* p2, const float* p3,
						 float* c, float& r)
{
	static const float EPS = 1e-6f;
	// Calculate the circle relative to p1, to avoid some precision issues.
	const float v1[3] = {0,0,0};
	float v2[3], v3[3];
	rcVsub(v2, p2,p1);
	rcVsub(v3, p3,p1);
	
	const float cp = vcross2(v1, v2, v3);
	if (fabsf(cp) > EPS)
	{
		const float v1Sq = vdot2(v1,v1);
		const float v2Sq = vdot2(v2,v2);
		const float v3Sq = vdot2(v3,v3);
		c[0] = (v1Sq*(v2[2]-v3[2]) + v2Sq*(v3[2]-v1[2]) + v3Sq*(v1[2]-v2[2])) / (2*cp);
		c[1] = 0;
		c[2] = (v1Sq*(v3[0]-v2[0]) + v2Sq*(v1[0]-v3[0]) + v3Sq*(v2[0]-v1[0])) / (2*cp);
		r = vdist2(c, v1);
		rcVadd(c, c, p1);
		return true;
	}
	
	rcVcopy(c, p1);
	r = 0;
	return false;
}

static float distPtTri(const float* p, const float* a, const float* b, const float* c)
{
	float v0[3], v1[3], v2[3];
	rcVsub(v0, c,a);
	rcVsub(v1, b,a);
	rcVsub(v2, p,a);
	
	const float dot00 = vdot2(v0, v0);
	const float dot01 = vdot2(v0, v1);
	const float dot02 = vdot2(v0, v2);
	const float dot11 = vdot2(v1, v1);
	const float dot12 = vdot2(v1, v2);
	
	// Compute barycentric coordinates
	const float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
	const float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
	
	// If point lies inside the triangle, return interpolated y-coord.
	static const float EPS = 1e-4f;
	if (u >= -EPS && v >= -EPS && (u+v) <= 1+EPS)
	{
		const float y = a[1] + v0[1]*u + v1[1]*v;
		return fabsf(y-p[1]);
	}
	return FLT_MAX;
}

static float distancePtSeg(const float* pt, const float* p, const float* q)
{
	float pqx = q[0] - p[0];
	float pqy = q[1] - p[1];
	float pqz = q[2] - p[2];
	float dx = pt[0] - p[0];
	float dy = pt[1] - p[1];
	float dz = pt[2] - p[2];
	float d = pqx*pqx + pqy*pqy + pqz*pqz;
	float t = pqx*dx + pqy*dy + pqz*dz;
	if (d > 0)
		t /= d;
	if (t < 0)
		t = 0;
	else if (t > 1)
		t = 1;
	
	dx = p[0] + t*pqx - pt[0];
	dy = p[1] + t*pqy - pt[1];
	dz = p[2] + t*pqz - pt[2];
	
	return dx*dx + dy*dy + dz*dz;
}

static float distancePtSeg2d(const float* pt, const float* p, const float* q)
{
	float pqx = q[0] - p[0];
	float pqz = q[2] - p[2];
	float dx = pt[0] - p[0];
	float dz = pt[2] - p[2];
	float d = pqx*pqx + pqz*pqz;
	float t = pqx*dx + pqz*dz;
	if (d > 0)
		t /= d;
	if (t < 0)
		t = 0;
	else if (t > 1)
		t = 1;
	
	dx = p[0] + t*pqx - pt[0];
	dz = p[2] + t*pqz - pt[2];
	
	return dx*dx + dz*dz;
}

static float distToTriMesh(const float* p, const float* verts, const int /*nverts*/, const int* tris, const int ntris)
{
	float dmin = FLT_MAX;
	for (int i = 0; i < ntris; ++i)
	{
		const float* va = &verts[tris[i*4+0]*3];
		const float* vb = &verts[tris[i*4+1]*3];
		const float* vc = &verts[tris[i*4+2]*3];
		float d = distPtTri(p, va,vb,vc);
		if (d < dmin)
			dmin = d;
	}
	if (dmin == FLT_MAX) return -1;
	return dmin;
}

static float distToPoly(int nvert, const float* verts, const float* p)
{
	
	float dmin = FLT_MAX;
	int i, j, c = 0;
	for (i = 0, j = nvert-1; i < nvert; j = i++)
	{
		const float* vi = &verts[i*3];
		const float* vj = &verts[j*3];
		if (((vi[2] > p[2]) != (vj[2] > p[2])) &&
			(p[0] < (vj[0]-vi[0]) * (p[2]-vi[2]) / (vj[2]-vi[2]) + vi[0]) )
			c = !c;
		dmin = rcMin(dmin, distancePtSeg2d(p, vj, vi));
	}
	return c ? -dmin : dmin;
}


static unsigned short getHeight(const float fx, const float fy, const float fz,
								const float /*cs*/, const float ics, const float ch,
								const int radius, const rcHeightPatch& hp)
{
	int ix = (int)floorf(fx*ics + 0.01f);
	int iz = (int)floorf(fz*ics + 0.01f);
	ix = rcClamp(ix-hp.xmin, 0, hp.width - 1);
	iz = rcClamp(iz-hp.ymin, 0, hp.height - 1);
	unsigned short h = hp.data[ix+iz*hp.width];
	if (h == RC_UNSET_HEIGHT)
	{
		// Special case when data might be bad.
		// Walk adjacent cells in a spiral up to 'radius', and look
		// for a pixel which has a valid height.
		int x = 1, z = 0, dx = 1, dz = 0;
		int maxSize = radius * 2 + 1;
		int maxIter = maxSize * maxSize - 1;

		int nextRingIterStart = 8;
		int nextRingIters = 16;

		float dmin = FLT_MAX;
		for (int i = 0; i < maxIter; i++)
		{
			const int nx = ix + x;
			const int nz = iz + z;

			if (nx >= 0 && nz >= 0 && nx < hp.width && nz < hp.height)
			{
				const unsigned short nh = hp.data[nx + nz*hp.width];
				if (nh != RC_UNSET_HEIGHT)
				{
					const float d = fabsf(nh*ch - fy);
					if (d < dmin)
					{
						h = nh;
						dmin = d;
					}
				}
			}

			// We are searching in a grid which looks approximately like this:
			//  __________
			// |2 ______ 2|
			// | |1 __ 1| |
			// | | |__| | |
			// | |______| |
			// |__________|
			// We want to find the best height as close to the center cell as possible. This means that
			// if we find a height in one of the neighbor cells to the center, we don't want to
			// expand further out than the 8 neighbors - we want to limit our search to the closest
			// of these "rings", but the best height in the ring.
			// For example, the center is just 1 cell. We checked that at the entrance to the function.
			// The next "ring" contains 8 cells (marked 1 above). Those are all the neighbors to the center cell.
			// The next one again contains 16 cells (marked 2). In general each ring has 8 additional cells, which
			// can be thought of as adding 2 cells around the "center" of each side when we expand the ring.
			// Here we detect if we are about to enter the next ring, and if we are and we have found
			// a height, we abort the search.
			if (i + 1 == nextRingIterStart)
			{
				if (h != RC_UNSET_HEIGHT)
					break;

				nextRingIterStart += nextRingIters;
				nextRingIters += 8;
			}

			if ((x == z) || ((x < 0) && (x == -z)) || ((x > 0) && (x == 1 - z)))
			{
				int tmp = dx;
				dx = -dz;
				dz = tmp;
			}
			x += dx;
			z += dz;
		}
	}
	return h;
}


enum EdgeValues
{
	EV_UNDEF = -1,
	EV_HULL = -2,
};

static int findEdge(const int* edges, int nedges, int s, int t)
{
	for (int i = 0; i < nedges; i++)
	{
		const int* e = &edges[i*4];
		if ((e[0] == s && e[1] == t) || (e[0] == t && e[1] == s))
			return i;
	}
	return EV_UNDEF;
}

static int addEdge(rcContext* ctx, int* edges, int& nedges, const int maxEdges, int s, int t, int l, int r)
{
	if (nedges >= maxEdges)
	{
		ctx->log(RC_LOG_ERROR, "addEdge: Too many edges (%d/%d).", nedges, maxEdges);
		return EV_UNDEF;
	}
	
	// Add edge if not already in the triangulation.
	int e = findEdge(edges, nedges, s, t);
	if (e == EV_UNDEF)
	{
		int* edge = &edges[nedges*4];
		edge[0] = s;
		edge[1] = t;
		edge[2] = l;
		edge[3] = r;
		return nedges++;
	}
	else
	{
		return EV_UNDEF;
	}
}

static void updateLeftFace(int* e, int s, int t, int f)
{
	if (e[0] == s && e[1] == t && e[2] == EV_UNDEF)
		e[2] = f;
	else if (e[1] == s && e[0] == t && e[3] == EV_UNDEF)
		e[3] = f;
}

static int overlapSegSeg2d(const float* a, const float* b, const float* c, const float* d)
{
	const float a1 = vcross2(a, b, d);
	const float a2 = vcross2(a, b, c);
	if (a1*a2 < 0.0f)
	{
		float a3 = vcross2(c, d, a);
		float a4 = a3 + a2 - a1;
		if (a3 * a4 < 0.0f)
			return 1;
	}
	return 0;
}

static bool overlapEdges(const float* pts, const int* edges, int nedges, int s1, int t1)
{
	for (int i = 0; i < nedges; ++i)
	{
		const int s0 = edges[i*4+0];
		const int t0 = edges[i*4+1];
		// Same or connected edges do not overlap.
		if (s0 == s1 || s0 == t1 || t0 == s1 || t0 == t1)
			continue;
		if (overlapSegSeg2d(&pts[s0*3],&pts[t0*3], &pts[s1*3],&pts[t1*3]))
			return true;
	}
	return false;
}

static void completeFacet(rcContext* ctx, const float* pts, int npts, int* edges, int& nedges, const int maxEdges, int& nfaces, int e)
{
	static const float EPS = 1e-5f;
	
	int* edge = &edges[e*4];
	
	// Cache s and t.
	int s,t;
	if (edge[2] == EV_UNDEF)
	{
		s = edge[0];
		t = edge[1];
	}
	else if (edge[3] == EV_UNDEF)
	{
		s = edge[1];
		t = edge[0];
	}
	else
	{
	    // Edge already completed.
	    return;
	}
    
	// Find best point on left of edge.
	int pt = npts;
	float c[3] = {0,0,0};
	float r = -1;
	for (int u = 0; u < npts; ++u)
	{
		if (u == s || u == t) continue;
		if (vcross2(&pts[s*3], &pts[t*3], &pts[u*3]) > EPS)
		{
			if (r < 0)
			{
				// The circle is not updated yet, do it now.
				pt = u;
				circumCircle(&pts[s*3], &pts[t*3], &pts[u*3], c, r);
				continue;
			}
			const float d = vdist2(c, &pts[u*3]);
			const float tol = 0.001f;
			if (d > r*(1+tol))
			{
				// Outside current circumcircle, skip.
				continue;
			}
			else if (d < r*(1-tol))
			{
				// Inside safe circumcircle, update circle.
				pt = u;
				circumCircle(&pts[s*3], &pts[t*3], &pts[u*3], c, r);
			}
			else
			{
				// Inside epsilon circum circle, do extra tests to make sure the edge is valid.
				// s-u and t-u cannot overlap with s-pt nor t-pt if they exists.
				if (overlapEdges(pts, edges, nedges, s,u))
					continue;
				if (overlapEdges(pts, edges, nedges, t,u))
					continue;
				// Edge is valid.
				pt = u;
				circumCircle(&pts[s*3], &pts[t*3], &pts[u*3], c, r);
			}
		}
	}
	
	// Add new triangle or update edge info if s-t is on hull.
	if (pt < npts)
	{
		// Update face information of edge being completed.
		updateLeftFace(&edges[e*4], s, t, nfaces);
		
		// Add new edge or update face info of old edge.
		e = findEdge(edges, nedges, pt, s);
		if (e == EV_UNDEF)
		    addEdge(ctx, edges, nedges, maxEdges, pt, s, nfaces, EV_UNDEF);
		else
		    updateLeftFace(&edges[e*4], pt, s, nfaces);
		
		// Add new edge or update face info of old edge.
		e = findEdge(edges, nedges, t, pt);
		if (e == EV_UNDEF)
		    addEdge(ctx, edges, nedges, maxEdges, t, pt, nfaces, EV_UNDEF);
		else
		    updateLeftFace(&edges[e*4], t, pt, nfaces);
		
		nfaces++;
	}
	else
	{
		updateLeftFace(&edges[e*4], s, t, EV_HULL);
	}
}

static void delaunayHull(rcContext* ctx, const int npts, const float* pts,
						 const int nhull, const int* hull,
						 rcIntArray& tris, rcIntArray& edges)
{
	int nfaces = 0;
	int nedges = 0;
	const int maxEdges = npts*10;
	edges.resize(maxEdges*4);
	
	for (int i = 0, j = nhull-1; i < nhull; j=i++)
		addEdge(ctx, &edges[0], nedges, maxEdges, hull[j],hull[i], EV_HULL, EV_UNDEF);
	
	int currentEdge = 0;
	while (currentEdge < nedges)
	{
		if (edges[currentEdge*4+2] == EV_UNDEF)
			completeFacet(ctx, pts, npts, &edges[0], nedges, maxEdges, nfaces, currentEdge);
		if (edges[currentEdge*4+3] == EV_UNDEF)
			completeFacet(ctx, pts, npts, &edges[0], nedges, maxEdges, nfaces, currentEdge);
		currentEdge++;
	}
	
	// Create tris
	tris.resize(nfaces*4);
	for (int i = 0; i < nfaces*4; ++i)
		tris[i] = -1;
	
	for (int i = 0; i < nedges; ++i)
	{
		const int* e = &edges[i*4];
		if (e[3] >= 0)
		{
			// Left face
			int* t = &tris[e[3]*4];
			if (t[0] == -1)
			{
				t[0] = e[0];
				t[1] = e[1];
			}
			else if (t[0] == e[1])
				t[2] = e[0];
			else if (t[1] == e[0])
				t[2] = e[1];
		}
		if (e[2] >= 0)
		{
			// Right
			int* t = &tris[e[2]*4];
			if (t[0] == -1)
			{
				t[0] = e[1];
				t[1] = e[0];
			}
			else if (t[0] == e[0])
				t[2] = e[1];
			else if (t[1] == e[1])
				t[2] = e[0];
		}
	}
	
	for (int i = 0; i < tris.size()/4; ++i)
	{
		int* t = &tris[i*4];
		if (t[0] == -1 || t[1] == -1 || t[2] == -1)
		{
			ctx->log(RC_LOG_WARNING, "delaunayHull: Removing dangling face %d [%d,%d,%d].", i, t[0],t[1],t[2]);
			t[0] = tris[tris.size()-4];
			t[1] = tris[tris.size()-3];
			t[2] = tris[tris.size()-2];
			t[3] = tris[tris.size()-1];
			tris.resize(tris.size()-4);
			--i;
		}
	}
}

// Calculate minimum extend of the polygon.
static float polyMinExtent(const float* verts, const int nverts)
{
	float minDist = FLT_MAX;
	for (int i = 0; i < nverts; i++)
	{
		const int ni = (i+1) % nverts;
		const float* p1 = &verts[i*3];
		const float* p2 = &verts[ni*3];
		float maxEdgeDist = 0;
		for (int j = 0; j < nverts; j++)
		{
			if (j == i || j == ni) continue;
			float d = distancePtSeg2d(&verts[j*3], p1,p2);
			maxEdgeDist = rcMax(maxEdgeDist, d);
		}
		minDist = rcMin(minDist, maxEdgeDist);
	}
	return rcSqrt(minDist);
}

// Last time I checked the if version got compiled using cmov, which was a lot faster than module (with idiv).
inline int prev(int i, int n) { return i-1 >= 0 ? i-1 : n-1; }
inline int next(int i, int n) { return i+1 < n ? i+1 : 0; }

static void triangulateHull(const int /*nverts*/, const float* verts, const int nhull, const int* hull, const int nin, rcIntArray& tris)
{
	int start = 0, left = 1, right = nhull-1;
	
	// Start from an ear with shortest perimeter.
	// This tends to favor well formed triangles as starting point.
	float dmin = FLT_MAX;
	for (int i = 0; i < nhull; i++)
	{
		if (hull[i] >= nin) continue; // Ears are triangles with original vertices as middle vertex while others are actually line segments on edges
		int pi = prev(i, nhull);
		int ni = next(i, nhull);
		const float* pv = &verts[hull[pi]*3];
		const float* cv = &verts[hull[i]*3];
		const float* nv = &verts[hull[ni]*3];
		const float d = vdist2(pv,cv) + vdist2(cv,nv) + vdist2(nv,pv);
		if (d < dmin)
		{
			start = i;
			left = ni;
			right = pi;
			dmin = d;
		}
	}
	
	// Add first triangle
	tris.push(hull[start]);
	tris.push(hull[left]);
	tris.push(hull[right]);
	tris.push(0);
	
	// Triangulate the polygon by moving left or right,
	// depending on which triangle has shorter perimeter.
	// This heuristic was chose emprically, since it seems
	// handle tesselated straight edges well.
	while (next(left, nhull) != right)
	{
		// Check to see if se should advance left or right.
		int nleft = next(left, nhull);
		int nright = prev(right, nhull);
		
		const float* cvleft = &verts[hull[left]*3];
		const float* nvleft = &verts[hull[nleft]*3];
		const float* cvright = &verts[hull[right]*3];
		const float* nvright = &verts[hull[nright]*3];
		const float dleft = vdist2(cvleft, nvleft) + vdist2(nvleft, cvright);
		const float dright = vdist2(cvright, nvright) + vdist2(cvleft, nvright);
		
		if (dleft < dright)
		{
			tris.push(hull[left]);
			tris.push(hull[nleft]);
			tris.push(hull[right]);
			tris.push(0);
			left = nleft;
		}
		else
		{
			tris.push(hull[left]);
			tris.push(hull[nright]);
			tris.push(hull[right]);
			tris.push(0);
			right = nright;
		}
	}
}


inline float getJitterX(const int i)
{
	return (((i * 0x8da6b343) & 0xffff) / 65535.0f * 2.0f) - 1.0f;
}

inline float getJitterY(const int i)
{
	return (((i * 0xd8163841) & 0xffff) / 65535.0f * 2.0f) - 1.0f;
}

static bool buildPolyDetail(rcContext* ctx, const float* in, const int nin,
							const float sampleDist, const float sampleMaxError,
							const int heightSearchRadius, const rcCompactHeightfield& chf,
							const rcHeightPatch& hp, float* verts, int& nverts,
							rcIntArray& tris, rcIntArray& edges, rcIntArray& samples)
{
	static const int MAX_VERTS = 127;
	static const int MAX_TRIS = 255;	// Max tris for delaunay is 2n-2-k (n=num verts, k=num hull verts).
	static const int MAX_VERTS_PER_EDGE = 32;
	float edge[(MAX_VERTS_PER_EDGE+1)*3];
	int hull[MAX_VERTS];
	int nhull = 0;
	
	nverts = nin;
	
	for (int i = 0; i < nin; ++i)
		rcVcopy(&verts[i*3], &in[i*3]);
	
	edges.clear();
	tris.clear();
	
	const float cs = chf.cs;
	const float ics = 1.0f/cs;
	
	// Calculate minimum extents of the polygon based on input data.
	float minExtent = polyMinExtent(verts, nverts);
	
	// Tessellate outlines.
	// This is done in separate pass in order to ensure
	// seamless height values across the ply boundaries.
	if (sampleDist > 0)
	{
		for (int i = 0, j = nin-1; i < nin; j=i++)
		{
			const float* vj = &in[j*3];
			const float* vi = &in[i*3];
			bool swapped = false;
			// Make sure the segments are always handled in same order
			// using lexological sort or else there will be seams.
			if (fabsf(vj[0]-vi[0]) < 1e-6f)
			{
				if (vj[2] > vi[2])
				{
					rcSwap(vj,vi);
					swapped = true;
				}
			}
			else
			{
				if (vj[0] > vi[0])
				{
					rcSwap(vj,vi);
					swapped = true;
				}
			}
			// Create samples along the edge.
			float dx = vi[0] - vj[0];
			float dy = vi[1] - vj[1];
			float dz = vi[2] - vj[2];
			float d = sqrtf(dx*dx + dz*dz);
			int nn = 1 + (int)floorf(d/sampleDist);
			if (nn >= MAX_VERTS_PER_EDGE) nn = MAX_VERTS_PER_EDGE-1;
			if (nverts+nn >= MAX_VERTS)
				nn = MAX_VERTS-1-nverts;
			
			for (int k = 0; k <= nn; ++k)
			{
				float u = (float)k/(float)nn;
				float* pos = &edge[k*3];
				pos[0] = vj[0] + dx*u;
				pos[1] = vj[1] + dy*u;
				pos[2] = vj[2] + dz*u;
				pos[1] = getHeight(pos[0],pos[1],pos[2], cs, ics, chf.ch, heightSearchRadius, hp)*chf.ch;
			}
			// Simplify samples.
			int idx[MAX_VERTS_PER_EDGE] = {0,nn};
			int nidx = 2;
			for (int k = 0; k < nidx-1; )
			{
				const int a = idx[k];
				const int b = idx[k+1];
				const float* va = &edge[a*3];
				const float* vb = &edge[b*3];
				// Find maximum deviation along the segment.
				float maxd = 0;
				int maxi = -1;
				for (int m = a+1; m < b; ++m)
				{
					float dev = distancePtSeg(&edge[m*3],va,vb);
					if (dev > maxd)
					{
						maxd = dev;
						maxi = m;
					}
				}
				// If the max deviation is larger than accepted error,
				// add new point, else continue to next segment.
				if (maxi != -1 && maxd > rcSqr(sampleMaxError))
				{
					for (int m = nidx; m > k; --m)
						idx[m] = idx[m-1];
					idx[k+1] = maxi;
					nidx++;
				}
				else
				{
					++k;
				}
			}
			
			hull[nhull++] = j;
			// Add new vertices.
			if (swapped)
			{
				for (int k = nidx-2; k > 0; --k)
				{
					rcVcopy(&verts[nverts*3], &edge[idx[k]*3]);
					hull[nhull++] = nverts;
					nverts++;
				}
			}
			else
			{
				for (int k = 1; k < nidx-1; ++k)
				{
					rcVcopy(&verts[nverts*3], &edge[idx[k]*3]);
					hull[nhull++] = nverts;
					nverts++;
				}
			}
		}
	}
	
	// If the polygon minimum extent is small (sliver or small triangle), do not try to add internal points.
	if (minExtent < sampleDist*2)
	{
		triangulateHull(nverts, verts, nhull, hull, nin, tris);
		return true;
	}
	
	// Tessellate the base mesh.
	// We're using the triangulateHull instead of delaunayHull as it tends to
	// create a bit better triangulation for long thin triangles when there
	// are no internal points.
	triangulateHull(nverts, verts, nhull, hull, nin, tris);
	
	if (tris.size() == 0)
	{
		// Could not triangulate the poly, make sure there is some valid data there.
		ctx->log(RC_LOG_WARNING, "buildPolyDetail: Could not triangulate polygon (%d verts).", nverts);
		return true;
	}
	
	if (sampleDist > 0)
	{
		// Create sample locations in a grid.
		float bmin[3], bmax[3];
		rcVcopy(bmin, in);
		rcVcopy(bmax, in);
		for (int i = 1; i < nin; ++i)
		{
			rcVmin(bmin, &in[i*3]);
			rcVmax(bmax, &in[i*3]);
		}
		int x0 = (int)floorf(bmin[0]/sampleDist);
		int x1 = (int)ceilf(bmax[0]/sampleDist);
		int z0 = (int)floorf(bmin[2]/sampleDist);
		int z1 = (int)ceilf(bmax[2]/sampleDist);
		samples.clear();
		for (int z = z0; z < z1; ++z)
		{
			for (int x = x0; x < x1; ++x)
			{
				float pt[3];
				pt[0] = x*sampleDist;
				pt[1] = (bmax[1]+bmin[1])*0.5f;
				pt[2] = z*sampleDist;
				// Make sure the samples are not too close to the edges.
				if (distToPoly(nin,in,pt) > -sampleDist/2) continue;
				samples.push(x);
				samples.push(getHeight(pt[0], pt[1], pt[2], cs, ics, chf.ch, heightSearchRadius, hp));
				samples.push(z);
				samples.push(0); // Not added
			}
		}
		
		// Add the samples starting from the one that has the most
		// error. The procedure stops when all samples are added
		// or when the max error is within treshold.
		const int nsamples = samples.size()/4;
		for (int iter = 0; iter < nsamples; ++iter)
		{
			if (nverts >= MAX_VERTS)
				break;
			
			// Find sample with most error.
			float bestpt[3] = {0,0,0};
			float bestd = 0;
			int besti = -1;
			for (int i = 0; i < nsamples; ++i)
			{
				const int* s = &samples[i*4];
				if (s[3]) continue; // skip added.
				float pt[3];
				// The sample location is jittered to get rid of some bad triangulations
				// which are cause by symmetrical data from the grid structure.
				pt[0] = s[0]*sampleDist + getJitterX(i)*cs*0.1f;
				pt[1] = s[1]*chf.ch;
				pt[2] = s[2]*sampleDist + getJitterY(i)*cs*0.1f;
				float d = distToTriMesh(pt, verts, nverts, &tris[0], tris.size()/4);
				if (d < 0) continue; // did not hit the mesh.
				if (d > bestd)
				{
					bestd = d;
					besti = i;
					rcVcopy(bestpt,pt);
				}
			}
			// If the max error is within accepted threshold, stop tesselating.
			if (bestd <= sampleMaxError || besti == -1)
				break;
			// Mark sample as added.
			samples[besti*4+3] = 1;
			// Add the new sample point.
			rcVcopy(&verts[nverts*3],bestpt);
			nverts++;
			
			// Create new triangulation.
			// TODO: Incremental add instead of full rebuild.
			edges.clear();
			tris.clear();
			delaunayHull(ctx, nverts, verts, nhull, hull, tris, edges);
		}
	}
	
	const int ntris = tris.size()/4;
	if (ntris > MAX_TRIS)
	{
		tris.resize(MAX_TRIS*4);
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Shrinking triangle count from %d to max %d.", ntris, MAX_TRIS);
	}
	
	return true;
}

static void seedArrayWithPolyCenter(rcContext* ctx, const rcCompactHeightfield& chf,
									const unsigned short* poly, const int npoly,
									const unsigned short* verts, const int bs,
									rcHeightPatch& hp, rcIntArray& array)
{
	// Note: Reads to the compact heightfield are offset by border size (bs)
	// since border size offset is already removed from the polymesh vertices.
	
	static const int offset[9*2] =
	{
		0,0, -1,-1, 0,-1, 1,-1, 1,0, 1,1, 0,1, -1,1, -1,0,
	};
	
	// Find cell closest to a poly vertex
	int startCellX = 0, startCellY = 0, startSpanIndex = -1;
	int dmin = RC_UNSET_HEIGHT;
	for (int j = 0; j < npoly && dmin > 0; ++j)
	{
		for (int k = 0; k < 9 && dmin > 0; ++k)
		{
			const int ax = (int)verts[poly[j]*3+0] + offset[k*2+0];
			const int ay = (int)verts[poly[j]*3+1];
			const int az = (int)verts[poly[j]*3+2] + offset[k*2+1];
			if (ax < hp.xmin || ax >= hp.xmin+hp.width ||
				az < hp.ymin || az >= hp.ymin+hp.height)
				continue;
			
			const rcCompactCell& c = chf.cells[(ax+bs)+(az+bs)*chf.width];
			for (int i = (int)c.index, ni = (int)(c.index+c.count); i < ni && dmin > 0; ++i)
			{
				const rcCompactSpan& s = chf.spans[i];
				int d = rcAbs(ay - (int)s.y);
				if (d < dmin)
				{
					startCellX = ax;
					startCellY = az;
					startSpanIndex = i;
					dmin = d;
				}
			}
		}
	}
	
	rcAssert(startSpanIndex != -1);
	// Find center of the polygon
	int pcx = 0, pcy = 0;
	for (int j = 0; j < npoly; ++j)
	{
		pcx += (int)verts[poly[j]*3+0];
		pcy += (int)verts[poly[j]*3+2];
	}
	pcx /= npoly;
	pcy /= npoly;
	
	// Use seeds array as a stack for DFS
	array.clear();
	array.push(startCellX);
	array.push(startCellY);
	array.push(startSpanIndex);

	int dirs[] = { 0, 1, 2, 3 };
	memset(hp.data, 0, sizeof(unsigned short)*hp.width*hp.height);
	// DFS to move to the center. Note that we need a DFS here and can not just move
	// directly towards the center without recording intermediate nodes, even though the polygons
	// are convex. In very rare we can get stuck due to contour simplification if we do not
	// record nodes.
	int cx = -1, cy = -1, ci = -1;
	while (true)
	{
		if (array.size() < 3)
		{
			ctx->log(RC_LOG_WARNING, "Walk towards polygon center failed to reach center");
			break;
		}

		ci = array.pop();
		cy = array.pop();
		cx = array.pop();

		if (cx == pcx && cy == pcy)
			break;

		// If we are already at the correct X-position, prefer direction
		// directly towards the center in the Y-axis; otherwise prefer
		// direction in the X-axis
		int directDir;
		if (cx == pcx)
			directDir = rcGetDirForOffset(0, pcy > cy ? 1 : -1);
		else
			directDir = rcGetDirForOffset(pcx > cx ? 1 : -1, 0);

		// Push the direct dir last so we start with this on next iteration
		rcSwap(dirs[directDir], dirs[3]);

		const rcCompactSpan& cs = chf.spans[ci];
		for (int i = 0; i < 4; i++)
		{
			int dir = dirs[i];
			if (rcGetCon(cs, dir) == RC_NOT_CONNECTED)
				continue;

			int newX = cx + rcGetDirOffsetX(dir);
			int newY = cy + rcGetDirOffsetY(dir);

			int hpx = newX - hp.xmin;
			int hpy = newY - hp.ymin;
			if (hpx < 0 || hpx >= hp.width || hpy < 0 || hpy >= hp.height)
				continue;

			if (hp.data[hpx+hpy*hp.width] != 0)
				continue;

			hp.data[hpx+hpy*hp.width] = 1;
			array.push(newX);
			array.push(newY);
			array.push((int)chf.cells[(newX+bs)+(newY+bs)*chf.width].index + rcGetCon(cs, dir));
		}

		rcSwap(dirs[directDir], dirs[3]);
	}

	array.clear();
	// getHeightData seeds are given in coordinates with borders
	array.push(cx+bs);
	array.push(cy+bs);
	array.push(ci);

	memset(hp.data, 0xff, sizeof(unsigned short)*hp.width*hp.height);
	const rcCompactSpan& cs = chf.spans[ci];
	hp.data[cx-hp.xmin+(cy-hp.ymin)*hp.width] = cs.y;
}


static void push3(rcIntArray& queue, int v1, int v2, int v3)
{
	queue.resize(queue.size() + 3);
	queue[queue.size() - 3] = v1;
	queue[queue.size() - 2] = v2;
	queue[queue.size() - 1] = v3;
}

static void getHeightData(rcContext* ctx, const rcCompactHeightfield& chf,
						  const unsigned short* poly, const int npoly,
						  const unsigned short* verts, const int bs,
						  rcHeightPatch& hp, rcIntArray& queue,
						  int region)
{
	// Note: Reads to the compact heightfield are offset by border size (bs)
	// since border size offset is already removed from the polymesh vertices.
	
	queue.clear();
	// Set all heights to RC_UNSET_HEIGHT.
	memset(hp.data, 0xff, sizeof(unsigned short)*hp.width*hp.height);

	bool empty = true;
	
	// We cannot sample from this poly if it was created from polys
	// of different regions. If it was then it could potentially be overlapping
	// with polys of that region and the heights sampled here could be wrong.
	if (region != RC_MULTIPLE_REGS)
	{
		// Copy the height from the same region, and mark region borders
		// as seed points to fill the rest.
		for (int hy = 0; hy < hp.height; hy++)
		{
			int y = hp.ymin + hy + bs;
			for (int hx = 0; hx < hp.width; hx++)
			{
				int x = hp.xmin + hx + bs;
				const rcCompactCell& c = chf.cells[x + y*chf.width];
				for (int i = (int)c.index, ni = (int)(c.index + c.count); i < ni; ++i)
				{
					const rcCompactSpan& s = chf.spans[i];
					if (s.reg == region)
					{
						// Store height
						hp.data[hx + hy*hp.width] = s.y;
						empty = false;

						// If any of the neighbours is not in same region,
						// add the current location as flood fill start
						bool border = false;
						for (int dir = 0; dir < 4; ++dir)
						{
							if (rcGetCon(s, dir) != RC_NOT_CONNECTED)
							{
								const int ax = x + rcGetDirOffsetX(dir);
								const int ay = y + rcGetDirOffsetY(dir);
								const int ai = (int)chf.cells[ax + ay*chf.width].index + rcGetCon(s, dir);
								const rcCompactSpan& as = chf.spans[ai];
								if (as.reg != region)
								{
									border = true;
									break;
								}
							}
						}
						if (border)
							push3(queue, x, y, i);
						break;
					}
				}
			}
		}
	}
	
	// if the polygon does not contain any points from the current region (rare, but happens)
	// or if it could potentially be overlapping polygons of the same region,
	// then use the center as the seed point.
	if (empty)
		seedArrayWithPolyCenter(ctx, chf, poly, npoly, verts, bs, hp, queue);
	
	static const int RETRACT_SIZE = 256;
	int head = 0;
	
	// We assume the seed is centered in the polygon, so a BFS to collect
	// height data will ensure we do not move onto overlapping polygons and
	// sample wrong heights.
	while (head*3 < queue.size())
	{
		int cx = queue[head*3+0];
		int cy = queue[head*3+1];
		int ci = queue[head*3+2];
		head++;
		if (head >= RETRACT_SIZE)
		{
			head = 0;
			if (queue.size() > RETRACT_SIZE*3)
				memmove(&queue[0], &queue[RETRACT_SIZE*3], sizeof(int)*(queue.size()-RETRACT_SIZE*3));
			queue.resize(queue.size()-RETRACT_SIZE*3);
		}
		
		const rcCompactSpan& cs = chf.spans[ci];
		for (int dir = 0; dir < 4; ++dir)
		{
			if (rcGetCon(cs, dir) == RC_NOT_CONNECTED) continue;
			
			const int ax = cx + rcGetDirOffsetX(dir);
			const int ay = cy + rcGetDirOffsetY(dir);
			const int hx = ax - hp.xmin - bs;
			const int hy = ay - hp.ymin - bs;
			
			if ((unsigned int)hx >= (unsigned int)hp.width || (unsigned int)hy >= (unsigned int)hp.height)
				continue;
			
			if (hp.data[hx + hy*hp.width] != RC_UNSET_HEIGHT)
				continue;
			
			const int ai = (int)chf.cells[ax + ay*chf.width].index + rcGetCon(cs, dir);
			const rcCompactSpan& as = chf.spans[ai];
			
			hp.data[hx + hy*hp.width] = as.y;
			
			push3(queue, ax, ay, ai);
		}
	}
}

static unsigned char getEdgeFlags(const float* va, const float* vb,
								  const float* vpoly, const int npoly)
{
	// The flag returned by this function matches dtDetailTriEdgeFlags in Detour.
	// Figure out if edge (va,vb) is part of the polygon boundary.
	static const float thrSqr = rcSqr(0.001f);
	for (int i = 0, j = npoly-1; i < npoly; j=i++)
	{
		if (distancePtSeg2d(va, &vpoly[j*3], &vpoly[i*3]) < thrSqr &&
			distancePtSeg2d(vb, &vpoly[j*3], &vpoly[i*3]) < thrSqr)
			return 1;
	}
	return 0;
}

static unsigned char getTriFlags(const float* va, const float* vb, const float* vc,
								 const float* vpoly, const int npoly)
{
	unsigned char flags = 0;
	flags |= getEdgeFlags(va,vb,vpoly,npoly) << 0;
	flags |= getEdgeFlags(vb,vc,vpoly,npoly) << 2;
	flags |= getEdgeFlags(vc,va,vpoly,npoly) << 4;
	return flags;
}

/// @par
///
/// See the #rcConfig documentation for more information on the configuration parameters.
///
/// @see rcAllocPolyMeshDetail, rcPolyMesh, rcCompactHeightfield, rcPolyMeshDetail, rcConfig
bool rcBuildPolyMeshDetail(rcContext* ctx, const rcPolyMesh& mesh, const rcCompactHeightfield& chf,
						   const float sampleDist, const float sampleMaxError,
						   rcPolyMeshDetail& dmesh)
{
	rcAssert(ctx);
	
	rcScopedTimer timer(ctx, RC_TIMER_BUILD_POLYMESHDETAIL);
	
	if (mesh.nverts == 0 || mesh.npolys == 0)
		return true;
	
	const int nvp = mesh.nvp;
	const float cs = mesh.cs;
	const float ch = mesh.ch;
	const float* orig = mesh.bmin;
	const int borderSize = mesh.borderSize;
	const int heightSearchRadius = rcMax(1, (int)ceilf(mesh.maxEdgeError));
	
	rcIntArray edges(64);
	rcIntArray tris(512);
	rcIntArray arr(512);
	rcIntArray samples(512);
	float verts[256*3];
	rcHeightPatch hp;
	int nPolyVerts = 0;
	int maxhw = 0, maxhh = 0;
	
	rcScopedDelete<int> bounds((int*)rcAlloc(sizeof(int)*mesh.npolys*4, RC_ALLOC_TEMP));
	if (!bounds)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'bounds' (%d).", mesh.npolys*4);
		return false;
	}
	rcScopedDelete<float> poly((float*)rcAlloc(sizeof(float)*nvp*3, RC_ALLOC_TEMP));
	if (!poly)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'poly' (%d).", nvp*3);
		return false;
	}
	
	// Find max size for a polygon area.
	for (int i = 0; i < mesh.npolys; ++i)
	{
		const unsigned short* p = &mesh.polys[i*nvp*2];
		int& xmin = bounds[i*4+0];
		int& xmax = bounds[i*4+1];
		int& ymin = bounds[i*4+2];
		int& ymax = bounds[i*4+3];
		xmin = chf.width;
		xmax = 0;
		ymin = chf.height;
		ymax = 0;
		for (int j = 0; j < nvp; ++j)
		{
			if(p[j] == RC_MESH_NULL_IDX) break;
			const unsigned short* v = &mesh.verts[p[j]*3];
			xmin = rcMin(xmin, (int)v[0]);
			xmax = rcMax(xmax, (int)v[0]);
			ymin = rcMin(ymin, (int)v[2]);
			ymax = rcMax(ymax, (int)v[2]);
			nPolyVerts++;
		}
		xmin = rcMax(0,xmin-1);
		xmax = rcMin(chf.width,xmax+1);
		ymin = rcMax(0,ymin-1);
		ymax = rcMin(chf.height,ymax+1);
		if (xmin >= xmax || ymin >= ymax) continue;
		maxhw = rcMax(maxhw, xmax-xmin);
		maxhh = rcMax(maxhh, ymax-ymin);
	}
	
	hp.data = (unsigned short*)rcAlloc(sizeof(unsigned short)*maxhw*maxhh, RC_ALLOC_TEMP);
	if (!hp.data)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'hp.data' (%d).", maxhw*maxhh);
		return false;
	}
	
	dmesh.nmeshes = mesh.npolys;
	dmesh.nverts = 0;
	dmesh.ntris = 0;
	dmesh.meshes = (unsigned int*)rcAlloc(sizeof(unsigned int)*dmesh.nmeshes*4, RC_ALLOC_PERM);
	if (!dmesh.meshes)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'dmesh.meshes' (%d).", dmesh.nmeshes*4);
		return false;
	}
	
	int vcap = nPolyVerts+nPolyVerts/2;
	int tcap = vcap*2;
	
	dmesh.nverts = 0;
	dmesh.verts = (float*)rcAlloc(sizeof(float)*vcap*3, RC_ALLOC_PERM);
	if (!dmesh.verts)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'dmesh.verts' (%d).", vcap*3);
		return false;
	}
	dmesh.ntris = 0;
	dmesh.tris = (unsigned char*)rcAlloc(sizeof(unsigned char)*tcap*4, RC_ALLOC_PERM);
	if (!dmesh.tris)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'dmesh.tris' (%d).", tcap*4);
		return false;
	}
	
	for (int i = 0; i < mesh.npolys; ++i)
	{
		const unsigned short* p = &mesh.polys[i*nvp*2];
		
		// Store polygon vertices for processing.
		int npoly = 0;
		for (int j = 0; j < nvp; ++j)
		{
			if(p[j] == RC_MESH_NULL_IDX) break;
			const unsigned short* v = &mesh.verts[p[j]*3];
			poly[j*3+0] = v[0]*cs;
			poly[j*3+1] = v[1]*ch;
			poly[j*3+2] = v[2]*cs;
			npoly++;
		}
		
		// Get the height data from the area of the polygon.
		hp.xmin = bounds[i*4+0];
		hp.ymin = bounds[i*4+2];
		hp.width = bounds[i*4+1]-bounds[i*4+0];
		hp.height = bounds[i*4+3]-bounds[i*4+2];
		getHeightData(ctx, chf, p, npoly, mesh.verts, borderSize, hp, arr, mesh.regs[i]);
		
		// Build detail mesh.
		int nverts = 0;
		if (!buildPolyDetail(ctx, poly, npoly,
							 sampleDist, sampleMaxError,
							 heightSearchRadius, chf, hp,
							 verts, nverts, tris,
							 edges, samples))
		{
			return false;
		}
		
		// Move detail verts to world space.
		for (int j = 0; j < nverts; ++j)
		{
			verts[j*3+0] += orig[0];
			verts[j*3+1] += orig[1] + chf.ch; // Is this offset necessary?
			verts[j*3+2] += orig[2];
		}
		// Offset poly too, will be used to flag checking.
		for (int j = 0; j < npoly; ++j)
		{
			poly[j*3+0] += orig[0];
			poly[j*3+1] += orig[1];
			poly[j*3+2] += orig[2];
		}
		
		// Store detail submesh.
		const int ntris = tris.size()/4;
		
		dmesh.meshes[i*4+0] = (unsigned int)dmesh.nverts;
		dmesh.meshes[i*4+1] = (unsigned int)nverts;
		dmesh.meshes[i*4+2] = (unsigned int)dmesh.ntris;
		dmesh.meshes[i*4+3] = (unsigned int)ntris;
		
		// Store vertices, allocate more memory if necessary.
		if (dmesh.nverts+nverts > vcap)
		{
			while (dmesh.nverts+nverts > vcap)
				vcap += 256;
			
			float* newv = (float*)rcAlloc(sizeof(float)*vcap*3, RC_ALLOC_PERM);
			if (!newv)
			{
				ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'newv' (%d).", vcap*3);
				return false;
			}
			if (dmesh.nverts)
				memcpy(newv, dmesh.verts, sizeof(float)*3*dmesh.nverts);
			rcFree(dmesh.verts);
			dmesh.verts = newv;
		}
		for (int j = 0; j < nverts; ++j)
		{
			dmesh.verts[dmesh.nverts*3+0] = verts[j*3+0];
			dmesh.verts[dmesh.nverts*3+1] = verts[j*3+1];
			dmesh.verts[dmesh.nverts*3+2] = verts[j*3+2];
			dmesh.nverts++;
		}
		
		// Store triangles, allocate more memory if necessary.
		if (dmesh.ntris+ntris > tcap)
		{
			while (dmesh.ntris+ntris > tcap)
				tcap += 256;
			unsigned char* newt = (unsigned char*)rcAlloc(sizeof(unsigned char)*tcap*4, RC_ALLOC_PERM);
			if (!newt)
			{
				ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'newt' (%d).", tcap*4);
				return false;
			}
			if (dmesh.ntris)
				memcpy(newt, dmesh.tris, sizeof(unsigned char)*4*dmesh.ntris);
			rcFree(dmesh.tris);
			dmesh.tris = newt;
		}
		for (int j = 0; j < ntris; ++j)
		{
			const int* t = &tris[j*4];
			dmesh.tris[dmesh.ntris*4+0] = (unsigned char)t[0];
			dmesh.tris[dmesh.ntris*4+1] = (unsigned char)t[1];
			dmesh.tris[dmesh.ntris*4+2] = (unsigned char)t[2];
			dmesh.tris[dmesh.ntris*4+3] = getTriFlags(&verts[t[0]*3], &verts[t[1]*3], &verts[t[2]*3], poly, npoly);
			dmesh.ntris++;
		}
	}
	
	return true;
}

/// @see rcAllocPolyMeshDetail, rcPolyMeshDetail
bool rcMergePolyMeshDetails(rcContext* ctx, rcPolyMeshDetail** meshes, const int nmeshes, rcPolyMeshDetail& mesh)
{
	rcAssert(ctx);
	
	rcScopedTimer timer(ctx, RC_TIMER_MERGE_POLYMESHDETAIL);
	
	int maxVerts = 0;
	int maxTris = 0;
	int maxMeshes = 0;
	
	for (int i = 0; i < nmeshes; ++i)
	{
		if (!meshes[i]) continue;
		maxVerts += meshes[i]->nverts;
		maxTris += meshes[i]->ntris;
		maxMeshes += meshes[i]->nmeshes;
	}
	
	mesh.nmeshes = 0;
	mesh.meshes = (unsigned int*)rcAlloc(sizeof(unsigned int)*maxMeshes*4, RC_ALLOC_PERM);
	if (!mesh.meshes)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'pmdtl.meshes' (%d).", maxMeshes*4);
		return false;
	}
	
	mesh.ntris = 0;
	mesh.tris = (unsigned char*)rcAlloc(sizeof(unsigned char)*maxTris*4, RC_ALLOC_PERM);
	if (!mesh.tris)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'dmesh.tris' (%d).", maxTris*4);
		return false;
	}
	
	mesh.nverts = 0;
	mesh.verts = (float*)rcAlloc(sizeof(float)*maxVerts*3, RC_ALLOC_PERM);
	if (!mesh.verts)
	{
		ctx->log(RC_LOG_ERROR, "rcBuildPolyMeshDetail: Out of memory 'dmesh.verts' (%d).", maxVerts*3);
		return false;
	}
	
	// Merge datas.
	for (int i = 0; i < nmeshes; ++i)
	{
		rcPolyMeshDetail* dm = meshes[i];
		if (!dm) continue;
		for (int j = 0; j < dm->nmeshes; ++j)
		{
			unsigned int* dst = &mesh.meshes[mesh.nmeshes*4];
			unsigned int* src = &dm->meshes[j*4];
			dst[0] = (unsigned int)mesh.nverts+src[0];
			dst[1] = src[1];
			dst[2] = (unsigned int)mesh.ntris+src[2];
			dst[3] = src[3];
			mesh.nmeshes++;
		}
		
		for (int k = 0; k < dm->nverts; ++k)
		{
			rcVcopy(&mesh.verts[mesh.nverts*3], &dm->verts[k*3]);
			mesh.nverts++;
		}
		for (int k = 0; k < dm->ntris; ++k)
		{
			mesh.tris[mesh.ntris*4+0] = dm->tris[k*4+0];
			mesh.tris[mesh.ntris*4+1] = dm->tris[k*4+1];
			mesh.tris[mesh.ntris*4+2] = dm->tris[k*4+2];
			mesh.tris[mesh.ntris*4+3] = dm->tris[k*4+3];
			mesh.ntris++;
		}
	}
	
	return true;
}

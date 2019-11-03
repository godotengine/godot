//Copyright (C) 2011 by Ivan Fratric
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.

#ifndef TRIANGULATOR_H
#define TRIANGULATOR_H

#include "core/list.h"
#include "core/math/vector2.h"
#include "core/set.h"

//2D point structure

#define TRIANGULATOR_CCW 1
#define TRIANGULATOR_CW -1
//Polygon implemented as an array of points with a 'hole' flag
class TriangulatorPoly {
protected:



	Vector2 *points;
	long numpoints;
	bool hole;

public:

	//constructors/destructors
	TriangulatorPoly();
	~TriangulatorPoly();

	TriangulatorPoly(const TriangulatorPoly &src);
	TriangulatorPoly& operator=(const TriangulatorPoly &src);

	//getters and setters
	long GetNumPoints() {
		return numpoints;
	}

	bool IsHole() {
		return hole;
	}

	void SetHole(bool hole) {
		this->hole = hole;
	}

	Vector2 &GetPoint(long i) {
		return points[i];
	}

	Vector2 *GetPoints() {
		return points;
	}

	Vector2& operator[] (int i) {
		return points[i];
	}

	//clears the polygon points
	void Clear();

	//inits the polygon with numpoints vertices
	void Init(long numpoints);

	//creates a triangle with points p1,p2,p3
	void Triangle(Vector2 &p1, Vector2 &p2, Vector2 &p3);

	//inverts the orfer of vertices
	void Invert();

	//returns the orientation of the polygon
	//possible values:
	//   Triangulator_CCW : polygon vertices are in counter-clockwise order
	//   Triangulator_CW : polygon vertices are in clockwise order
	//       0 : the polygon has no (measurable) area
	int GetOrientation();

	//sets the polygon orientation
	//orientation can be
	//   Triangulator_CCW : sets vertices in counter-clockwise order
	//   Triangulator_CW : sets vertices in clockwise order
	void SetOrientation(int orientation);
};

class TriangulatorPartition {
protected:
	struct PartitionVertex {
		bool isActive;
		bool isConvex;
		bool isEar;

		Vector2 p;
		real_t angle;
		PartitionVertex *previous;
		PartitionVertex *next;
	};

	struct MonotoneVertex {
		Vector2 p;
		long previous;
		long next;
	};

	struct VertexSorter{
		mutable MonotoneVertex *vertices;
		bool operator() (long index1, long index2) const;
	};

	struct Diagonal {
		long index1;
		long index2;
	};

	//dynamic programming state for minimum-weight triangulation
	struct DPState {
		bool visible;
		real_t weight;
		long bestvertex;
	};

	//dynamic programming state for convex partitioning
	struct DPState2 {
		bool visible;
		long weight;
		List<Diagonal> pairs;
	};

	//edge that intersects the scanline
	struct ScanLineEdge {
		mutable long index;
		Vector2 p1;
		Vector2 p2;

		//determines if the edge is to the left of another edge
		bool operator< (const ScanLineEdge & other) const;

		bool IsConvex(const Vector2& p1, const Vector2& p2, const Vector2& p3) const;
	};

	//standard helper functions
	bool IsConvex(Vector2& p1, Vector2& p2, Vector2& p3);
	bool IsReflex(Vector2& p1, Vector2& p2, Vector2& p3);
	bool IsInside(Vector2& p1, Vector2& p2, Vector2& p3, Vector2 &p);

	bool InCone(Vector2 &p1, Vector2 &p2, Vector2 &p3, Vector2 &p);
	bool InCone(PartitionVertex *v, Vector2 &p);

	int Intersects(Vector2 &p11, Vector2 &p12, Vector2 &p21, Vector2 &p22);

	Vector2 Normalize(const Vector2 &p);
	real_t Distance(const Vector2 &p1, const Vector2 &p2);

	//helper functions for Triangulate_EC
	void UpdateVertexReflexity(PartitionVertex *v);
	void UpdateVertex(PartitionVertex *v,PartitionVertex *vertices, long numvertices);

	//helper functions for ConvexPartition_OPT
	void UpdateState(long a, long b, long w, long i, long j, DPState2 **dpstates);
	void TypeA(long i, long j, long k, PartitionVertex *vertices, DPState2 **dpstates);
	void TypeB(long i, long j, long k, PartitionVertex *vertices, DPState2 **dpstates);

	//helper functions for MonotonePartition
	bool Below(Vector2 &p1, Vector2 &p2);
	void AddDiagonal(MonotoneVertex *vertices, long *numvertices, long index1, long index2,
			 char *vertextypes, Set<ScanLineEdge>::Element **edgeTreeIterators,
			 Set<ScanLineEdge> *edgeTree, long *helpers);

	//triangulates a monotone polygon, used in Triangulate_MONO
	int TriangulateMonotone(TriangulatorPoly *inPoly, List<TriangulatorPoly> *triangles);

public:

	//simple heuristic procedure for removing holes from a list of polygons
	//works by creating a diagonal from the rightmost hole vertex to some visible vertex
	//time complexity: O(h*(n^2)), h is the number of holes, n is the number of vertices
	//space complexity: O(n)
	//params:
	//   inpolys : a list of polygons that can contain holes
	//             vertices of all non-hole polys have to be in counter-clockwise order
	//             vertices of all hole polys have to be in clockwise order
	//   outpolys : a list of polygons without holes
	//returns 1 on success, 0 on failure
	int RemoveHoles(List<TriangulatorPoly> *inpolys, List<TriangulatorPoly> *outpolys);

	//triangulates a polygon by ear clipping
	//time complexity O(n^2), n is the number of vertices
	//space complexity: O(n)
	//params:
	//   poly : an input polygon to be triangulated
	//          vertices have to be in counter-clockwise order
	//   triangles : a list of triangles (result)
	//returns 1 on success, 0 on failure
	int Triangulate_EC(TriangulatorPoly *poly, List<TriangulatorPoly> *triangles);

	//triangulates a list of polygons that may contain holes by ear clipping algorithm
	//first calls RemoveHoles to get rid of the holes, and then Triangulate_EC for each resulting polygon
	//time complexity: O(h*(n^2)), h is the number of holes, n is the number of vertices
	//space complexity: O(n)
	//params:
	//   inpolys : a list of polygons to be triangulated (can contain holes)
	//             vertices of all non-hole polys have to be in counter-clockwise order
	//             vertices of all hole polys have to be in clockwise order
	//   triangles : a list of triangles (result)
	//returns 1 on success, 0 on failure
	int Triangulate_EC(List<TriangulatorPoly> *inpolys, List<TriangulatorPoly> *triangles);

	//creates an optimal polygon triangulation in terms of minimal edge length
	//time complexity: O(n^3), n is the number of vertices
	//space complexity: O(n^2)
	//params:
	//   poly : an input polygon to be triangulated
	//          vertices have to be in counter-clockwise order
	//   triangles : a list of triangles (result)
	//returns 1 on success, 0 on failure
	int Triangulate_OPT(TriangulatorPoly *poly, List<TriangulatorPoly> *triangles);

	//triangulates a polygons by firstly partitioning it into monotone polygons
	//time complexity: O(n*log(n)), n is the number of vertices
	//space complexity: O(n)
	//params:
	//   poly : an input polygon to be triangulated
	//          vertices have to be in counter-clockwise order
	//   triangles : a list of triangles (result)
	//returns 1 on success, 0 on failure
	int Triangulate_MONO(TriangulatorPoly *poly, List<TriangulatorPoly> *triangles);

	//triangulates a list of polygons by firstly partitioning them into monotone polygons
	//time complexity: O(n*log(n)), n is the number of vertices
	//space complexity: O(n)
	//params:
	//   inpolys : a list of polygons to be triangulated (can contain holes)
	//             vertices of all non-hole polys have to be in counter-clockwise order
	//             vertices of all hole polys have to be in clockwise order
	//   triangles : a list of triangles (result)
	//returns 1 on success, 0 on failure
	int Triangulate_MONO(List<TriangulatorPoly> *inpolys, List<TriangulatorPoly> *triangles);

	//creates a monotone partition of a list of polygons that can contain holes
	//time complexity: O(n*log(n)), n is the number of vertices
	//space complexity: O(n)
	//params:
	//   inpolys : a list of polygons to be triangulated (can contain holes)
	//             vertices of all non-hole polys have to be in counter-clockwise order
	//             vertices of all hole polys have to be in clockwise order
	//   monotonePolys : a list of monotone polygons (result)
	//returns 1 on success, 0 on failure
	int MonotonePartition(List<TriangulatorPoly> *inpolys, List<TriangulatorPoly> *monotonePolys);

	//partitions a polygon into convex polygons by using Hertel-Mehlhorn algorithm
	//the algorithm gives at most four times the number of parts as the optimal algorithm
	//however, in practice it works much better than that and often gives optimal partition
	//uses triangulation obtained by ear clipping as intermediate result
	//time complexity O(n^2), n is the number of vertices
	//space complexity: O(n)
	//params:
	//   poly : an input polygon to be partitioned
	//          vertices have to be in counter-clockwise order
	//   parts : resulting list of convex polygons
	//returns 1 on success, 0 on failure
	int ConvexPartition_HM(TriangulatorPoly *poly, List<TriangulatorPoly> *parts);

	//partitions a list of polygons into convex parts by using Hertel-Mehlhorn algorithm
	//the algorithm gives at most four times the number of parts as the optimal algorithm
	//however, in practice it works much better than that and often gives optimal partition
	//uses triangulation obtained by ear clipping as intermediate result
	//time complexity O(n^2), n is the number of vertices
	//space complexity: O(n)
	//params:
	//   inpolys : an input list of polygons to be partitioned
	//             vertices of all non-hole polys have to be in counter-clockwise order
	//             vertices of all hole polys have to be in clockwise order
	//   parts : resulting list of convex polygons
	//returns 1 on success, 0 on failure
	int ConvexPartition_HM(List<TriangulatorPoly> *inpolys, List<TriangulatorPoly> *parts);

	//optimal convex partitioning (in terms of number of resulting convex polygons)
	//using the Keil-Snoeyink algorithm
	//M. Keil, J. Snoeyink, "On the time bound for convex decomposition of simple polygons", 1998
	//time complexity O(n^3), n is the number of vertices
	//space complexity: O(n^3)
	//   poly : an input polygon to be partitioned
	//          vertices have to be in counter-clockwise order
	//   parts : resulting list of convex polygons
	//returns 1 on success, 0 on failure
	int ConvexPartition_OPT(TriangulatorPoly *poly, List<TriangulatorPoly> *parts);
};


#endif

/*************************************************************************/
/* Copyright (c) 2011-2021 Ivan Fratric and contributors.                */
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

#ifndef POLYPARTITION_H
#define POLYPARTITION_H

#include "core/math/vector2.h"
#include "core/templates/list.h"
#include "core/templates/set.h"

typedef double tppl_float;

enum TPPLOrientation {
  TPPL_ORIENTATION_CW = -1,
  TPPL_ORIENTATION_NONE = 0,
  TPPL_ORIENTATION_CCW = 1,
};

enum TPPLVertexType {
  TPPL_VERTEXTYPE_REGULAR = 0,
  TPPL_VERTEXTYPE_START = 1,
  TPPL_VERTEXTYPE_END = 2,
  TPPL_VERTEXTYPE_SPLIT = 3,
  TPPL_VERTEXTYPE_MERGE = 4,
};

// 2D point structure.
typedef Vector2 TPPLPoint;

// Polygon implemented as an array of points with a "hole" flag.
class TPPLPoly {
  protected:
  TPPLPoint *points;
  long numpoints;
  bool hole;

  public:
  // Constructors and destructors.
  TPPLPoly();
  ~TPPLPoly();

  TPPLPoly(const TPPLPoly &src);
  TPPLPoly &operator=(const TPPLPoly &src);

  // Getters and setters.
  long GetNumPoints() const {
    return numpoints;
  }

  bool IsHole() const {
    return hole;
  }

  void SetHole(bool hole) {
    this->hole = hole;
  }

  TPPLPoint &GetPoint(long i) {
    return points[i];
  }

  const TPPLPoint &GetPoint(long i) const {
    return points[i];
  }

  TPPLPoint *GetPoints() {
    return points;
  }

  TPPLPoint &operator[](int i) {
    return points[i];
  }

  const TPPLPoint &operator[](int i) const {
    return points[i];
  }

  // Clears the polygon points.
  void Clear();

  // Inits the polygon with numpoints vertices.
  void Init(long numpoints);

  // Creates a triangle with points p1, p2, and p3.
  void Triangle(TPPLPoint &p1, TPPLPoint &p2, TPPLPoint &p3);

  // Inverts the orfer of vertices.
  void Invert();

  // Returns the orientation of the polygon.
  // Possible values:
  //    TPPL_ORIENTATION_CCW: Polygon vertices are in counter-clockwise order.
  //    TPPL_ORIENTATION_CW: Polygon vertices are in clockwise order.
  //    TPPL_ORIENTATION_NONE: The polygon has no (measurable) area.
  TPPLOrientation GetOrientation() const;

  // Sets the polygon orientation.
  // Possible values:
  //    TPPL_ORIENTATION_CCW: Sets vertices in counter-clockwise order.
  //    TPPL_ORIENTATION_CW: Sets vertices in clockwise order.
  //    TPPL_ORIENTATION_NONE: Reverses the orientation of the vertices if there
  //       is one, otherwise does nothing (if orientation is already NONE).
  void SetOrientation(TPPLOrientation orientation);

  // Checks whether a polygon is valid or not.
  inline bool Valid() const { return this->numpoints >= 3; }
};

#ifdef TPPL_ALLOCATOR
typedef List<TPPLPoly, TPPL_ALLOCATOR(TPPLPoly)> TPPLPolyList;
#else
typedef List<TPPLPoly> TPPLPolyList;
#endif

class TPPLPartition {
  protected:
  struct PartitionVertex {
    bool isActive;
    bool isConvex;
    bool isEar;

    TPPLPoint p;
    tppl_float angle;
    PartitionVertex *previous;
    PartitionVertex *next;

    PartitionVertex();
  };

  struct MonotoneVertex {
    TPPLPoint p;
    long previous;
    long next;
  };

  class VertexSorter {
    MonotoneVertex *vertices;

public:
    VertexSorter(MonotoneVertex *v) :
            vertices(v) {}
    bool operator()(long index1, long index2);
  };

  struct Diagonal {
    long index1;
    long index2;
  };

#ifdef TPPL_ALLOCATOR
  typedef List<Diagonal, TPPL_ALLOCATOR(Diagonal)> DiagonalList;
#else
  typedef List<Diagonal> DiagonalList;
#endif

  // Dynamic programming state for minimum-weight triangulation.
  struct DPState {
    bool visible;
    tppl_float weight;
    long bestvertex;
  };

  // Dynamic programming state for convex partitioning.
  struct DPState2 {
    bool visible;
    long weight;
    DiagonalList pairs;
  };

  // Edge that intersects the scanline.
  struct ScanLineEdge {
    mutable long index;
    TPPLPoint p1;
    TPPLPoint p2;

    // Determines if the edge is to the left of another edge.
    bool operator<(const ScanLineEdge &other) const;

    bool IsConvex(const TPPLPoint &p1, const TPPLPoint &p2, const TPPLPoint &p3) const;
  };

  // Standard helper functions.
  bool IsConvex(TPPLPoint &p1, TPPLPoint &p2, TPPLPoint &p3);
  bool IsReflex(TPPLPoint &p1, TPPLPoint &p2, TPPLPoint &p3);
  bool IsInside(TPPLPoint &p1, TPPLPoint &p2, TPPLPoint &p3, TPPLPoint &p);

  bool InCone(TPPLPoint &p1, TPPLPoint &p2, TPPLPoint &p3, TPPLPoint &p);
  bool InCone(PartitionVertex *v, TPPLPoint &p);

  int Intersects(TPPLPoint &p11, TPPLPoint &p12, TPPLPoint &p21, TPPLPoint &p22);

  TPPLPoint Normalize(const TPPLPoint &p);
  tppl_float Distance(const TPPLPoint &p1, const TPPLPoint &p2);

  // Helper functions for Triangulate_EC.
  void UpdateVertexReflexity(PartitionVertex *v);
  void UpdateVertex(PartitionVertex *v, PartitionVertex *vertices, long numvertices);

  // Helper functions for ConvexPartition_OPT.
  void UpdateState(long a, long b, long w, long i, long j, DPState2 **dpstates);
  void TypeA(long i, long j, long k, PartitionVertex *vertices, DPState2 **dpstates);
  void TypeB(long i, long j, long k, PartitionVertex *vertices, DPState2 **dpstates);

  // Helper functions for MonotonePartition.
  bool Below(TPPLPoint &p1, TPPLPoint &p2);
  void AddDiagonal(MonotoneVertex *vertices, long *numvertices, long index1, long index2,
          TPPLVertexType *vertextypes, Set<ScanLineEdge>::Element **edgeTreeIterators,
          Set<ScanLineEdge> *edgeTree, long *helpers);

  // Triangulates a monotone polygon, used in Triangulate_MONO.
  int TriangulateMonotone(TPPLPoly *inPoly, TPPLPolyList *triangles);

  public:
  // Simple heuristic procedure for removing holes from a list of polygons.
  // It works by creating a diagonal from the right-most hole vertex
  // to some other visible vertex.
  // Time complexity: O(h*(n^2)), h is the # of holes, n is the # of vertices.
  // Space complexity: O(n)
  // params:
  //    inpolys:
  //       A list of polygons that can contain holes.
  //       Vertices of all non-hole polys have to be in counter-clockwise order.
  //       Vertices of all hole polys have to be in clockwise order.
  //    outpolys:
  //       A list of polygons without holes.
  // Returns 1 on success, 0 on failure.
  int RemoveHoles(TPPLPolyList *inpolys, TPPLPolyList *outpolys);

  // Triangulates a polygon by ear clipping.
  // Time complexity: O(n^2), n is the number of vertices.
  // Space complexity: O(n)
  // params:
  //    poly:
  //       An input polygon to be triangulated.
  //       Vertices have to be in counter-clockwise order.
  //    triangles:
  //       A list of triangles (result).
  // Returns 1 on success, 0 on failure.
  int Triangulate_EC(TPPLPoly *poly, TPPLPolyList *triangles);

  // Triangulates a list of polygons that may contain holes by ear clipping
  // algorithm. It first calls RemoveHoles to get rid of the holes, and then
  // calls Triangulate_EC for each resulting polygon.
  // Time complexity: O(h*(n^2)), h is the # of holes, n is the # of vertices.
  // Space complexity: O(n)
  // params:
  //    inpolys:
  //       A list of polygons to be triangulated (can contain holes).
  //       Vertices of all non-hole polys have to be in counter-clockwise order.
  //       Vertices of all hole polys have to be in clockwise order.
  //    triangles:
  //       A list of triangles (result).
  // Returns 1 on success, 0 on failure.
  int Triangulate_EC(TPPLPolyList *inpolys, TPPLPolyList *triangles);

  // Creates an optimal polygon triangulation in terms of minimal edge length.
  // Time complexity: O(n^3), n is the number of vertices
  // Space complexity: O(n^2)
  // params:
  //    poly:
  //       An input polygon to be triangulated.
  //       Vertices have to be in counter-clockwise order.
  //    triangles:
  //       A list of triangles (result).
  // Returns 1 on success, 0 on failure.
  int Triangulate_OPT(TPPLPoly *poly, TPPLPolyList *triangles);

  // Triangulates a polygon by first partitioning it into monotone polygons.
  // Time complexity: O(n*log(n)), n is the number of vertices.
  // Space complexity: O(n)
  // params:
  //    poly:
  //       An input polygon to be triangulated.
  //       Vertices have to be in counter-clockwise order.
  //    triangles:
  //       A list of triangles (result).
  // Returns 1 on success, 0 on failure.
  int Triangulate_MONO(TPPLPoly *poly, TPPLPolyList *triangles);

  // Triangulates a list of polygons by first
  // partitioning them into monotone polygons.
  // Time complexity: O(n*log(n)), n is the number of vertices.
  // Space complexity: O(n)
  // params:
  //    inpolys:
  //       A list of polygons to be triangulated (can contain holes).
  //       Vertices of all non-hole polys have to be in counter-clockwise order.
  //       Vertices of all hole polys have to be in clockwise order.
  //    triangles:
  //       A list of triangles (result).
  // Returns 1 on success, 0 on failure.
  int Triangulate_MONO(TPPLPolyList *inpolys, TPPLPolyList *triangles);

  // Creates a monotone partition of a list of polygons that
  // can contain holes. Triangulates a set of polygons by
  // first partitioning them into monotone polygons.
  // Time complexity: O(n*log(n)), n is the number of vertices.
  // Space complexity: O(n)
  // params:
  //    inpolys:
  //       A list of polygons to be triangulated (can contain holes).
  //       Vertices of all non-hole polys have to be in counter-clockwise order.
  //       Vertices of all hole polys have to be in clockwise order.
  //    monotonePolys:
  //       A list of monotone polygons (result).
  // Returns 1 on success, 0 on failure.
  int MonotonePartition(TPPLPolyList *inpolys, TPPLPolyList *monotonePolys);

  // Partitions a polygon into convex polygons by using the
  // Hertel-Mehlhorn algorithm. The algorithm gives at most four times
  // the number of parts as the optimal algorithm, however, in practice
  // it works much better than that and often gives optimal partition.
  // It uses triangulation obtained by ear clipping as intermediate result.
  // Time complexity O(n^2), n is the number of vertices.
  // Space complexity: O(n)
  // params:
  //    poly:
  //       An input polygon to be partitioned.
  //       Vertices have to be in counter-clockwise order.
  //    parts:
  //       Resulting list of convex polygons.
  // Returns 1 on success, 0 on failure.
  int ConvexPartition_HM(TPPLPoly *poly, TPPLPolyList *parts);

  // Partitions a list of polygons into convex parts by using the
  // Hertel-Mehlhorn algorithm. The algorithm gives at most four times
  // the number of parts as the optimal algorithm, however, in practice
  // it works much better than that and often gives optimal partition.
  // It uses triangulation obtained by ear clipping as intermediate result.
  // Time complexity O(n^2), n is the number of vertices.
  // Space complexity: O(n)
  // params:
  //    inpolys:
  //       An input list of polygons to be partitioned. Vertices of
  //       all non-hole polys have to be in counter-clockwise order.
  //       Vertices of all hole polys have to be in clockwise order.
  //    parts:
  //       Resulting list of convex polygons.
  // Returns 1 on success, 0 on failure.
  int ConvexPartition_HM(TPPLPolyList *inpolys, TPPLPolyList *parts);

  // Optimal convex partitioning (in terms of number of resulting
  // convex polygons) using the Keil-Snoeyink algorithm.
  // For reference, see M. Keil, J. Snoeyink, "On the time bound for
  // convex decomposition of simple polygons", 1998.
  // Time complexity O(n^3), n is the number of vertices.
  // Space complexity: O(n^3)
  // params:
  //    poly:
  //       An input polygon to be partitioned.
  //       Vertices have to be in counter-clockwise order.
  //    parts:
  //       Resulting list of convex polygons.
  // Returns 1 on success, 0 on failure.
  int ConvexPartition_OPT(TPPLPoly *poly, TPPLPolyList *parts);
};

#endif

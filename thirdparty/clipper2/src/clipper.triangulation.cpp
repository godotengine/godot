/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  13 December 2025                                                *
* Release   :  BETA RELEASE                                                    *
* Website   :  https://www.angusj.com                                          *
* Copyright :  Angus Johnson 2010-2025                                         *
* Purpose   :  Constrained Delaunay Triangulation                              *
* License   :  https://www.boost.org/LICENSE_1_0.txt                           *
*******************************************************************************/

#include "clipper2/clipper.h"
#include "clipper2/clipper.triangulation.h"

namespace Clipper2Lib
{

  enum class EdgeKind { loose, ascend, descend }; // ascend & descend are 'fixed' edges
  enum class IntersectKind { none, collinear, intersect };
  enum class EdgeContainsResult { neither, left, right };

  //forward definitions
  class Vertex2;
  class Edge;
  class Triangle;

  typedef std::vector<Vertex2*> VertexList;
  typedef std::vector<Edge*> EdgeList;

  class Vertex2 {
  public:
    Point64   pt;
    EdgeList  edges;
    bool      innerLM = false;
    Vertex2(const Point64& p64) : pt(p64) { edges.reserve(2); };
  };

  class Edge {
  public:
    Vertex2*  vL = nullptr;
    Vertex2*  vR = nullptr;
    Vertex2*  vB = nullptr;
    Vertex2*  vT = nullptr;
    EdgeKind  kind = EdgeKind::loose;
    Triangle* triA = nullptr;
    Triangle* triB = nullptr;
    bool      isActive = false;
    Edge*     nextE = nullptr;
    Edge*     prevE = nullptr;
  };

  class Triangle {
  public:
    Edge* edges[3];
    Triangle(Edge* e1, Edge* e2, Edge* e3)
    {
      edges[0] = e1;
      edges[1] = e2;
      edges[2] = e3;
    }
  };

  /////////////////////////////////////////////////////////////////////////////
  // Delaunay class declaration
  /////////////////////////////////////////////////////////////////////////////

  class Delaunay {
  private:
    VertexList                allVertices;
    EdgeList                  allEdges;
    std::vector<Triangle*>    allTriangles;
    std::stack<Edge*>         pendingDelaunayStack;
    std::stack<Edge*>         horzEdgeStack;
    std::stack<Vertex2*>      locMinStack;
    bool                      useDelaunay = true;
    Vertex2*                  lowermostVertex = nullptr;
    Edge*                     firstActive = nullptr;
    void AddPath(const Path64& path);
    bool AddPaths(const Paths64& paths);
    void CleanUp();
    bool FixupEdgeIntersects();
    void MergeDupOrCollinearVertices();
    void SplitEdge(Edge* longE, Edge* shortE);
    bool RemoveIntersection(Edge* e1, Edge* e2);
    Edge* CreateEdge(Vertex2* v1, Vertex2* v2, EdgeKind k);
    Triangle* CreateTriangle(Edge* e1, Edge* e2, Edge* e3);
    Edge* CreateInnerLocMinLooseEdge(Vertex2* vAbove);
    Edge* HorizontalBetween(Vertex2* v1, Vertex2* v2);
    void DoTriangulateLeft(Edge* edge, Vertex2* pivot, int64_t minY);
    void DoTriangulateRight(Edge* edge, Vertex2* pivot, int64_t minY);
    void AddEdgeToActives(Edge* edge);
    void RemoveEdgeFromActives(Edge* edge);
    void ForceLegal(Edge* edge);
  public:
    explicit Delaunay(bool delaunay = true) : useDelaunay(delaunay) {};
    ~Delaunay() { CleanUp(); };
    Paths64 Execute(const Paths64& paths, TriangulateResult& triResult);
  };

  /////////////////////////////////////////////////////////////////////////////
  // Miscellaneous functions
  /////////////////////////////////////////////////////////////////////////////

  static bool VertexListSort(const Vertex2* a, const Vertex2* b)
  {
    return (a->pt.y == b->pt.y) ? (a->pt.x < b->pt.x) : (a->pt.y > b->pt.y);
  }

  static bool EdgeListSort(const Edge* a, const Edge* b)
  {
    return (a->vL->pt.x < b->vL->pt.x);
  }

  static bool IsLooseEdge(const Edge& e)
  {
    return e.kind == EdgeKind::loose;
  }

  static bool IsLeftEdge(const Edge& e)
  {
    // left edge (bound) of a fill region
    // ie fills on the **right** side of the edge
    // precondition - e is never a 'loose' edge
    return e.kind == EdgeKind::ascend;
  }

  static bool IsRightEdge(const Edge& e)
  {
    // right edge (bound) of a fill region
    // but still fills on the **right** side of the edge
    // precondition - e is never a 'loose' edge
    return e.kind == EdgeKind::descend;
  }

  static bool IsHorizontal(const Edge& e)
  {
    return e.vB->pt.y == e.vT->pt.y;
  }

  static bool LeftTurning(const Point64& p1, const Point64& p2, const Point64& p3)
  {
    return CrossProductSign(p1, p2, p3) < 0;
  }

  static bool RightTurning(const Point64& p1, const Point64& p2, const Point64& p3)
  {
    return CrossProductSign(p1, p2, p3) > 0;
  }

  static bool EdgeCompleted(Edge* edge)
  {
    if (!edge->triA) return false;
    if (edge->triB) return true;
    return edge->kind != EdgeKind::loose;
  }

  static EdgeContainsResult EdgeContains(const Edge* edge, const Vertex2* v)
  {
    if (edge->vL == v) return EdgeContainsResult::left;
    else if (edge->vR == v) return EdgeContainsResult::right;
    else return EdgeContainsResult::neither;
  }

  static double GetAngle(const Point64& a, const Point64& b, const Point64& c)
  {
    //https://stackoverflow.com/a/3487062/359538
    double abx = static_cast<double>(b.x - a.x);
    double aby = static_cast<double>(b.y - a.y);
    double bcx = static_cast<double>(b.x - c.x);
    double bcy = static_cast<double>(b.y - c.y);
    double dp = (abx * bcx + aby * bcy);
    double cp = (abx * bcy - aby * bcx);
    return std::atan2(cp, dp); //range between -Pi and Pi
  }

  static double GetLocMinAngle(Vertex2* v)
  {
    // todo - recheck the result's sign compared to left vs right turning
    // (currently assumes left turning => positive values)
    // precondition - this function is called before processing locMin.
    //Assert(v->edges.size() == 2);
    int asc, des;
    if (v->edges[0]->kind == EdgeKind::ascend)
    {
      asc = 0;
      des = 1;
    }
    else
    {
      des = 0;
      asc = 1;
    }
    // winding direction - descending to ascending
    return GetAngle(v->edges[des]->vT->pt, v->pt, v->edges[asc]->vT->pt);
  }

  static void RemoveEdgeFromVertex(Vertex2* vert, Edge* edge)
  {
    auto it = std::find(vert->edges.begin(), vert->edges.end(), edge);
    if (it == vert->edges.end()) throw "oops!";
    vert->edges.erase(it);
  }

  static bool FindLocMinIdx(const Path64& path, size_t len, size_t& idx)
  {
    if (len < 3) return false;
    size_t i0 = idx, n = (idx + 1) % len;
    while (path[n].y <= path[idx].y) 
    {
      idx = n; n = (n + 1) % len;
      if (idx == i0) return false; // fails if the path is completely horizontal
    }
    while (path[n].y >= path[idx].y)
    {
      idx = n; n = (n + 1) % len;
    }
    return true;
  }

  static size_t Prev(size_t& idx, size_t len)
  {
    if (idx == 0) return len - 1; else return idx - 1;
  }

  static size_t Next(size_t& idx, size_t len)
  {
    return (idx + 1) % len;
  }

  static Edge* FindLinkingEdge(const Vertex2* vert1, const Vertex2* vert2, bool preferAscending)
  {
    Edge* res = nullptr;
    for (auto e : vert1->edges)
    {
      if (e->vL == vert2 || e->vR == vert2)
      {
        if (e->kind == EdgeKind::loose ||
          ((e->kind == EdgeKind::ascend) == preferAscending)) return e;
        res = e;
      }
    }
    return res;
  }

  static Path64 PathFromTriangle(Triangle tri)
  { 
    Path64 res;
    res.reserve(3);
    res.push_back(tri.edges[0]->vL->pt);
    res.push_back(tri.edges[0]->vR->pt);
    const Edge& e = *tri.edges[1];
    if (e.vL->pt == res[0] || e.vL->pt == res[1]) 
      res.push_back(e.vR->pt);
    else
      res.push_back(e.vL->pt);
    return res;
  }

  static double InCircleTest(const Point64& ptA, const Point64& ptB, 
    const Point64& ptC, const Point64& ptD)
  {
    // Return the determinant value of 3 x 3 matrix ...
    // | ax-dx    ay-dy    Sqr(ax-dx)+Sqr(ay-dy) |
    // | bx-dx    by-dy    Sqr(bx-dx)+Sqr(by-dy) |
    // | cx-dx    cy-dy    Sqr(cx-dx)+Sqr(cy-dy) |

    // The *sign* of the return value is determined by
    // the orientation (CW vs CCW) of ptA, ptB & ptC.

    // precondition - ptA, ptB & ptC make a *non-empty* triangle
    double m00 = static_cast<double>(ptA.x - ptD.x);
    double m01 = static_cast<double>(ptA.y - ptD.y);
    double m02 = (Sqr(m00) + Sqr(m01));
    double m10 = static_cast<double>(ptB.x - ptD.x);
    double m11 = static_cast<double>(ptB.y - ptD.y);
    double m12 = (Sqr(m10) + Sqr(m11));
    double m20 = static_cast<double>(ptC.x - ptD.x);
    double m21 = static_cast<double>(ptC.y - ptD.y);
    double m22 = (Sqr(m20) + Sqr(m21));
    return m00 * (m11 * m22 - m21 * m12) -
      m10 * (m01 * m22 - m21 * m02) +
      m20 * (m01 * m12 - m11 * m02);
  }

  static double ShortestDistFromSegment(const Point64& pt, const Point64& segPt1, const Point64& segPt2)
  {
    // precondition: segPt1 <> segPt2
    double dx = static_cast<double>(segPt2.x - segPt1.x);
    double dy = static_cast<double>(segPt2.y - segPt1.y);
    //Assert((dx < > 0) or (dy < > 0)); // ie segPt1 <> segPt2

    double ax = static_cast<double>(pt.x - segPt1.x);
    double ay = static_cast<double>(pt.y - segPt1.y);
    //q = (ax * dx + ay * dy) / (dx * dx + dy * dy);
    double qNum = ax * dx + ay * dy;
    if (qNum < 0)  // pt is closest to seq1
      return DistanceSqr(pt, segPt1);
    else if (qNum > (Sqr(dx) + Sqr(dy))) // pt is closest to seq2
      return DistanceSqr(pt, segPt2);
    else
       return Sqr(ax * dy - dx * ay) / (dx * dx + dy * dy);
  }

  static IntersectKind SegsIntersect(const Point64 s1a, const Point64 s1b, 
    const Point64 s2a, const Point64 s2b)
  {
    double dy1 = static_cast<double>(s1b.y - s1a.y);
    double dx1 = static_cast<double>(s1b.x - s1a.x);
    double dy2 = static_cast<double>(s2b.y - s2a.y);
    double dx2 = static_cast<double>(s2b.x - s2a.x);
    double cp = dy1 * dx2 - dy2 * dx1;
    if (cp == 0) return IntersectKind::collinear;

    double t = (static_cast<double>(s1a.x - s2a.x) * dy2 - 
      static_cast<double>(s1a.y - s2a.y) * dx2);
    //ignore segments that 'intersect' at an end-point
    if (t == 0) return IntersectKind::none;
    if (t > 0)
    {
      if (cp < 0 || t >= cp) return IntersectKind::none;
    }
    else
    {
      if (cp > 0 || t <= cp) return IntersectKind::none;
    }

    // so far, the *segment* 's1' intersects the *line* through 's2',
    // but now make sure it also intersects the *segment* 's2'
    t = ((s1a.x - s2a.x) * dy1 - (s1a.y - s2a.y) * dx1);
    if (t == 0) return IntersectKind::none;
    if (t > 0)
    {
      if (cp > 0 && t < cp) return IntersectKind::intersect;
    }
    else
    { 
      if (cp < 0 && t > cp) return IntersectKind::intersect;
    }
    return IntersectKind::none;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Delaunay class definitions
  /////////////////////////////////////////////////////////////////////////////

  void Delaunay::CleanUp()
  {
    for (auto v : allVertices) delete v;
    allVertices.resize(0);
    for (auto e : allEdges) delete e;
    allEdges.resize(0);
    for (auto t : allTriangles) delete t;
    allTriangles.resize(0);

    firstActive = nullptr;
    lowermostVertex = nullptr;
   }

  void Delaunay::ForceLegal(Edge* edge)
  {
    // don't try to make empty triangles legal
    if (!edge->triA || !edge->triB) return;

    // vertA will be assigned the vertex in edge's triangleA
    // that is NOT an end vertex of edge
    // Likewise, vertB will be assigned the vertex in edge's
    // triangleB that is NOT an end vertex of edge
    // If edge is rotated, vertA & vertB will become its end vertices.
    Vertex2* vertA = nullptr;
    Vertex2* vertB = nullptr;

    // Excluding 'edge', edgesA will contain two edges (one from
    // triangleA and one from triangleB) that touch edge.vL.
    // And edgesB will contain the two edges that touch edge.vR.
    
    Edge* edgesA[3] = { nullptr, nullptr, nullptr };
    Edge* edgesB[3] = { nullptr, nullptr, nullptr };
    for (int i = 0; i < 3; ++i)
    {
      if (edge->triA->edges[i] == edge) continue;
      switch (EdgeContains(edge->triA->edges[i], edge->vL))
      {
      case EdgeContainsResult::left:
        edgesA[1] = edge->triA->edges[i];
        vertA = edge->triA->edges[i]->vR;
        break;
      case EdgeContainsResult::right:
        edgesA[1] = edge->triA->edges[i];
        vertA = edge->triA->edges[i]->vL;
        break;
      default:
        edgesB[1] = edge->triA->edges[i];
      }
    }

    for (int i = 0; i < 3; ++i)
    {
      if (edge->triB->edges[i] == edge) continue;
      switch (EdgeContains(edge->triB->edges[i], edge->vL))
      {
      case EdgeContainsResult::left:
        edgesA[2] = edge->triB->edges[i];
        vertB = edge->triB->edges[i]->vR;
        break;
      case EdgeContainsResult::right:
        edgesA[2] = edge->triB->edges[i];
        vertB = edge->triB->edges[i]->vL;
        break;
      default:
        edgesB[2] = edge->triB->edges[i];
      }
    }

    // InCircleTest reqires edge.triangleA to be a valid triangle
    // if IsEmptyTriangle(edge.triangleA) then Exit; // slower
    if (CrossProductSign(vertA->pt, edge->vL->pt, edge->vR->pt) == 0) return;

    // ictResult - result sign is dependant on triangleA's orientation
    double ictResult = InCircleTest(vertA->pt, edge->vL->pt, edge->vR->pt, vertB->pt);
    if (ictResult == 0 || // if on or out of circle then exit
      (RightTurning(vertA->pt, edge->vL->pt, edge->vR->pt) == (ictResult < 0))) return;

    // TRIANGLES HERE ARE **NOT** DELAUNAY COMPLIANT, SO MAKE THEM SO.

    // NOTE: ONCE WE BEGIN DELAUNAY COMPLIANCE, vL & vR WILL
    // NO LONGER REPRESENT LEFT AND RIGHT VERTEX ORIENTATION.
    // THIS IS MINOR PERFORMANCE EFFICIENCY IS SAFE AS LONG AS
    // THE TRIANGULATE() METHOD IS CALLED ONCE ONLY ON A GIVEN
    // SET OF PATHS

    edge->vL = vertA;
    edge->vR = vertB;

    edge->triA->edges[0] = edge;
    for (int i = 1; i < 3; ++i)
    {
      edge->triA->edges[i] = edgesA[i];
      if (!edgesA[i]) throw "oops"; // stops compiler warnings 
      if (IsLooseEdge(*edgesA[i]))
        pendingDelaunayStack.push(edgesA[i]);
      // since each edge has its own triangleA and triangleB, we have to be careful
      // to update the correct one ...
      if (edgesA[i]->triA == edge->triA || edgesA[i]->triB == edge->triA) continue;

      if (edgesA[i]->triA == edge->triB)
        edgesA[i]->triA = edge->triA;
      else if (edgesA[i]->triB == edge->triB)
        edgesA[i]->triB = edge->triA;
      else throw "oops";
    }

    edge->triB->edges[0] = edge;
    for (int i = 1; i < 3; ++i)
    {
      edge->triB->edges[i] = edgesB[i];
      if (!edgesB[i]) throw "oops"; // stops compiler warnings 
      if (IsLooseEdge(*edgesB[i]))
        pendingDelaunayStack.push(edgesB[i]);
      // since each edge has its own triangleA and triangleB, we have to be careful
      // to update the correct one ...
      if (edgesB[i]->triA == edge->triB || edgesB[i]->triB == edge->triB) continue;

      if (edgesB[i]->triA == edge->triA)
        edgesB[i]->triA = edge->triB;
      else if (edgesB[i]->triB == edge->triA)
        edgesB[i]->triB = edge->triB;
      else throw "oops";
    }

  }

  Edge* Delaunay::CreateEdge(Vertex2* v1, Vertex2* v2, EdgeKind k)
  {
    Edge* res = allEdges.emplace_back(new Edge());
    if (v1->pt.y == v2->pt.y)
    {
      res->vB = v1; res->vT = v2;
    }
    else if (v1->pt.y < v2->pt.y)
    {
      res->vB = v2; res->vT = v1;
    }
    else
    {
      res->vB = v1; res->vT = v2;
    }

    if (v1->pt.x <= v2->pt.x)
    {
      res->vL = v1; res->vR = v2;
    }
    else
    {
      res->vL = v2; res->vR = v1;
    }
    res->kind = k;
    v1->edges.push_back(res);
    v2->edges.push_back(res);

    if (k == EdgeKind::loose)
    {
      pendingDelaunayStack.push(res);
      AddEdgeToActives(res);
    }
    return res;
  }

  Triangle* Delaunay::CreateTriangle(Edge* e1, Edge* e2, Edge* e3)
  {
    Triangle* res = allTriangles.emplace_back(new Triangle(e1, e2, e3));
    // nb: only expire loose edges when both sides of these edges have triangles.
    for (int i = 0; i < 3; ++i)       
      if (res->edges[i]->triA)
      {
        res->edges[i]->triB = res;
        // this is the edge's second triangle hence no longer active
        RemoveEdgeFromActives(res->edges[i]);
      }
      else
      {
        res->edges[i]->triA = res;
        // this is the edge's first triangle, so only remove
        // this edge from actives if it's a fixed edge.
        if (!IsLooseEdge(*res->edges[i])) 
          RemoveEdgeFromActives(res->edges[i]);
      }
    return res;
  }

  bool Delaunay::RemoveIntersection(Edge* e1, Edge* e2)
  {
    // find which vertex is closest to the other segment
    // (ie not the vertex closest to the intersection point) ...

    Vertex2* v = e1->vL;
    Edge* tmpE = e2;
    double d = ShortestDistFromSegment(e1->vL->pt, e2->vL->pt, e2->vR->pt);
    double d2 = ShortestDistFromSegment(e1->vR->pt, e2->vL->pt, e2->vR->pt);
    if (d2 < d) { d = d2; v = e1->vR; }
    d2 = ShortestDistFromSegment(e2->vL->pt, e1->vL->pt, e1->vR->pt);
    if (d2 < d) { d = d2; tmpE = e1; v = e2->vL; }
    d2 = ShortestDistFromSegment(e2->vR->pt, e1->vL->pt, e1->vR->pt);
    if (d2 < d) { d = d2; tmpE = e1; v = e2->vR; }
    if (d > 1.000)
      return false; // Oops - this is not just a simple 'rounding' intersection

    // split 'tmpE' into 2 edges at 'v'
    Vertex2* v2 = tmpE->vT;
    RemoveEdgeFromVertex(v2, tmpE);
    // replace v2 in tmpE with v
    if (tmpE->vL == v2) tmpE->vL = v; 
    else tmpE->vR = v;
    tmpE->vT = v;
    v->edges.push_back(tmpE);
    v->innerLM = false; // #47
    // left turning is angle positive
    if (tmpE->vB->innerLM && GetLocMinAngle(tmpE->vB) <= 0)
      tmpE->vB->innerLM = false; // #44, 52
    // finally create a new edge between v and v2 ...
    CreateEdge(v, v2, tmpE->kind);
    return true;
  }

  bool Delaunay::FixupEdgeIntersects()
  {
    // precondition - edgeList must be sorted - ascending on edge.vL.pt.x

    for (size_t i1 = 0; i1 < allEdges.size(); ++i1)
    {
      Edge* e1 = allEdges[i1];
      // nb: we can safely ignore edges newly created inside this for loop
      for (size_t i2 = i1 + 1; i2 < allEdges.size(); ++i2)
      {
        Edge* e2 = allEdges[i2];
        if (e2->vL->pt.x >= e1->vR->pt.x) 
          break; // all 'e' from now on are too far right

        // 'e2' is inside e1's horizontal region. If 'e2' is also inside
        // e1's vertical region, only then check for an intersection ...
        if (e2->vT->pt.y < e1->vB->pt.y && e2->vB->pt.y > e1->vT->pt.y &&
          (SegsIntersect(e2->vL->pt, e2->vR->pt,
            e1->vL->pt, e1->vR->pt) == IntersectKind::intersect))
        {
          if (!RemoveIntersection(e2, e1)) 
            return false; // oops!!
        }
        // nb: collinear edges are managed in MergeDupOrCollinearVertices below
      }
    }
    return true;
  }

  void Delaunay::SplitEdge(Edge* longE, Edge* shortE)
  {
    auto oldT = longE->vT, newT = shortE->vT;
    // remove longEdge from longEdge.vT.edges
    RemoveEdgeFromVertex(oldT, longE);
    // shorten longEdge
    longE->vT = newT;
    if (longE->vL == oldT) longE->vL = newT;
    else longE->vR = newT;
    // add shortened longEdge to newT.edges
    newT->edges.push_back(longE);
    // and create a new edge betweem newV, oldT
    CreateEdge(newT, oldT, longE->kind);
  }

  void Delaunay::MergeDupOrCollinearVertices()
  {
    // note: this procedure may add new edges and change the
    // number of edges connected with a given vertex, but it 
    // won't add or delete vertices (so it's safe to use iterators)
    auto vIter1 = allVertices.begin();
    for (auto vIter2 = allVertices.begin() + 1; vIter2 != allVertices.end(); ++vIter2)
    {
      if ((*vIter1)->pt != (*vIter2)->pt)
      {
        vIter1 = vIter2;
        continue;
      }

      // merge v1 & v2
      Vertex2* v1 = *vIter1, * v2 = *vIter2;
      if (!v1->innerLM || !v2->innerLM)
        v1->innerLM = false;

      // in all of v2's edges, replace links to v2 with links to v1
      for (auto e : v2->edges)
      {
        if (e->vB == v2) e->vB = v1;
        else e->vT = v1;
        if (e->vL == v2) e->vL = v1;
        else e->vR = v1;
      }
      // move all of v2's edges to v1
      std::copy(v2->edges.begin(), v2->edges.end(), back_inserter(v1->edges));
      v2->edges.resize(0);

      // excluding horizontals, if pv.edges contains two edges
      // that are *collinear* and share the same bottom coords
      // but have different lengths, split the longer edge at
      // the top of the shorter edge ...
      for (auto itE = v1->edges.begin(); itE != v1->edges.end(); ++itE)
      {
        if (IsHorizontal(*(*itE)) || (*itE)->vB != v1) continue;
        for (auto itE2 = itE + 1; itE2 != v1->edges.end(); ++itE2)
        {
          auto e1 = *itE, e2 = *itE2;
          if (e2->vB != v1 || e1->vT->pt.y == e2->vT->pt.y ||
            (CrossProductSign(e1->vT->pt, v1->pt, e2->vT->pt) != 0)) continue;
          // we have parallel edges, both heading up from v1.pt.
          // split the longer edge at the top of the shorter edge.
          if (e1->vT->pt.y < e2->vT->pt.y) SplitEdge(e1, e2);
          else SplitEdge(e2, e1);
          break; // because only two edges can be collinear
        }
      }
    }
  }

  Edge* Delaunay::CreateInnerLocMinLooseEdge(Vertex2* vAbove)
  {
    if (!firstActive) return nullptr; // oops!!

    int64_t xAbove = vAbove->pt.x;
    int64_t yAbove = vAbove->pt.y;

    // find the closest 'active' edge with a vertex that's not above vAbove
    Edge* e = firstActive, *eBelow = nullptr;
    double bestD = -1.0;
    while (e) 
    { 
      if (e->vL->pt.x <= xAbove && e->vR->pt.x >= xAbove &&
        e->vB->pt.y >= yAbove && e->vB != vAbove && e->vT != vAbove &&
        !LeftTurning(e->vL->pt, vAbove->pt, e->vR->pt))
      {
        double d = ShortestDistFromSegment(vAbove->pt, e->vL->pt, e->vR->pt);
        if (!eBelow || d < bestD) // compare e with eBelow
        {
          eBelow = e;
          bestD = d;
        }
      }
      e = e->nextE;
    }
    if (!eBelow) return nullptr; // oops!!

    // get the best vertex from 'eBelow'
    Vertex2* vBest = (eBelow->vT->pt.y <= yAbove) ? eBelow->vB : eBelow->vT;
    int64_t xBest = vBest->pt.x;
    int64_t yBest = vBest->pt.y;

    // make sure no edges intersect 'vAbove' and 'vBest' ...
    // todo: fActives is currently *unsorted* but consider making it
    // a tree structure based on each edge's left and right bounds
    e = firstActive;
    if (xBest < xAbove)
    {
      while (e)
      {
        if (e->vR->pt.x > xBest && e->vL->pt.x < xAbove &&
          e->vB->pt.y > yAbove && e->vT->pt.y < yBest &&
          (SegsIntersect(e->vB->pt, e->vT->pt,
            vBest->pt, vAbove->pt) == IntersectKind::intersect))
        {
          vBest = (e->vT->pt.y > yAbove) ? e->vT : e->vB;
          xBest = vBest->pt.x;
          yBest = vBest->pt.y;
        }
        e = e->nextE;
      }
    }
    else
    {
      while (e)
      {
        if (e->vR->pt.x < xBest && e->vL->pt.x > xAbove &&
          e->vB->pt.y > yAbove && e->vT->pt.y < yBest &&
          (SegsIntersect(e->vB->pt, e->vT->pt,
            vBest->pt, vAbove->pt) == IntersectKind::intersect))
        {
          vBest = e->vT->pt.y > yAbove ? e->vT : e->vB;
          xBest = vBest->pt.x;
          yBest = vBest->pt.y;
        }
        e = e->nextE;
      }
    }
    return CreateEdge(vBest, vAbove, EdgeKind::loose);
  }

  Edge* Delaunay::HorizontalBetween(Vertex2* v1, Vertex2* v2)
  {
    int64_t y = v1->pt.y, l, r;
    if (v1->pt.x > v2->pt.x)
    {
      l = v2->pt.x;
      r = v1->pt.x;
    } 
    else
    {
      l = v1->pt.x;
      r = v2->pt.x;
    }

    Edge* res = firstActive;
    while (res)
    {
      if (res->vL->pt.y == y && res->vR->pt.y == y &&
        res->vL->pt.x >= l && res->vR->pt.x <= r &&
        (res->vL->pt.x != l || res->vL->pt.x != r)) break;
      res = res->nextE;
    }
    return res;
  }

  void Delaunay::DoTriangulateLeft(Edge* edge, Vertex2* pivot, int64_t minY)
  {
    // precondition - pivot must be one end of edge (Usually .vB)
    //Assert(!EdgeCompleted(edge));
    Vertex2*  vAlt = nullptr;
    Edge*     eAlt = nullptr;
    Vertex2*  v = (edge->vB == pivot) ? edge->vT : edge->vB;

    for (auto e : pivot->edges)
    {
      if (e == edge || !e->isActive) continue;
      Vertex2* vX = e->vT == pivot ? e->vB : e->vT;
      if (vX == v) continue;

      int cps = CrossProductSign(v->pt, pivot->pt, vX->pt);
      if (cps == 0) //collinear paths
      {
        // if pivot is between v and vX then continue;
        // nb: this is important for both horiz and non-horiz collinear
        if ((v->pt.x > pivot->pt.x) == (pivot->pt.x > vX->pt.x)) continue;
      }
      // else if right-turning or not the best edge, then continue
      else if (cps > 0 || (vAlt && !LeftTurning(vX->pt, pivot->pt, vAlt->pt)))
        continue;
      vAlt = vX;
      eAlt = e;
    }

    if (!vAlt || vAlt->pt.y < minY) return;
    
    // Don't triangulate **across** fixed edges
    if (vAlt->pt.y < pivot->pt.y)
    {
      if (IsLeftEdge(*eAlt)) return;
    }
    else if (vAlt->pt.y > pivot->pt.y)
    {
      if (IsRightEdge(*eAlt)) return;
    }

    Edge* eX = FindLinkingEdge(vAlt, v, (vAlt->pt.y < v->pt.y));
    if (!eX)
    {
      // be very careful creating loose horizontals at minY
      if (vAlt->pt.y == v->pt.y && v->pt.y == minY &&
        HorizontalBetween(vAlt, v)) return;
      eX = CreateEdge(vAlt, v, EdgeKind::loose);
    }

    CreateTriangle(edge, eAlt, eX);
    if (!EdgeCompleted(eX))
      DoTriangulateLeft(eX, vAlt, minY);

  }

  void Delaunay::DoTriangulateRight(Edge* edge, Vertex2* pivot, int64_t minY)
  {
    // precondition - pivot must be one end of edge (Usually .vB)
    //Assert(!EdgeCompleted(edge));
    Vertex2* vAlt = nullptr;
    Edge* eAlt = nullptr;
    Vertex2* v = (edge->vB == pivot) ? edge->vT : edge->vB;

    for (auto e : pivot->edges)
    {
      if (e == edge || !e->isActive) continue;
      Vertex2* vX = e->vT == pivot ? e->vB : e->vT;
      if (vX == v) continue;

      int cps = CrossProductSign(v->pt, pivot->pt, vX->pt);
      if (cps == 0) //collinear paths
      {
        // if pivot is between v and vX then continue;
        // nb: this is important for both horiz and non-horiz collinear
        if ((v->pt.x > pivot->pt.x) == (pivot->pt.x > vX->pt.x)) continue;
      }
      // else if right-turning or not the best edge, then continue
      else if (cps < 0 || (vAlt && !RightTurning(vX->pt, pivot->pt, vAlt->pt)))
        continue;
      vAlt = vX;
      eAlt = e;
    }

    if (!vAlt || vAlt->pt.y < minY) return;

    // Don't triangulate **across** fixed edges
    if (vAlt->pt.y < pivot->pt.y)
    {
      if (IsRightEdge(*eAlt)) return;
    }
    else if (vAlt->pt.y > pivot->pt.y)
    {
      if (IsLeftEdge(*eAlt)) return;
    }

    Edge* eX = FindLinkingEdge(vAlt, v, (vAlt->pt.y > v->pt.y));
    if (!eX)
    {
      // be very careful creating loose horizontals at minY
      if (vAlt->pt.y == v->pt.y && v->pt.y == minY &&
        HorizontalBetween(vAlt, v)) return;
      eX = CreateEdge(vAlt, v, EdgeKind::loose);
    }

    CreateTriangle(edge, eX, eAlt);
    if (!EdgeCompleted(eX))
      DoTriangulateRight(eX, vAlt, minY);


  }

  void Delaunay::AddEdgeToActives(Edge* edge)
  {
    // nb: on occassions this method can get called twice for a given edge
    // This is because, in the Triangulate() method where vertex 'edges'
    // arrays are being parsed, edges can can be removed from the array
    // which changes the index of following edges.
    if (edge->isActive) return;

    edge->prevE = nullptr;
    edge->nextE = firstActive;
    edge->isActive = true;
    if (firstActive)
      firstActive->prevE = edge;
    firstActive = edge;
  }


  void Delaunay::RemoveEdgeFromActives(Edge* edge)
  {
    // first, remove the edge from its vertices
    RemoveEdgeFromVertex(edge->vB, edge);
    RemoveEdgeFromVertex(edge->vT, edge);

    // now remove the edge from double linked list (AEL)
    Edge* prev = edge->prevE;
    Edge* next = edge->nextE;
    if (next) next->prevE = prev;
    if (prev) prev->nextE = next;
    edge->isActive = false;
    if (firstActive == edge) firstActive = next;
  }

  Paths64 Delaunay::Execute(const Paths64& paths, TriangulateResult& triResult)
  {
    if (!AddPaths(paths))
    {
      triResult = TriangulateResult::no_polygons;
      return Paths64(); // oops!
    }

    // if necessary fix path orientation because the algorithm 
    // expects clockwise outer paths and counter-clockwise inner paths
    if (lowermostVertex->innerLM)
    {
      // the orientation of added paths must be wrong, so
      // 1. reverse innerLM flags ...
      Vertex2* lm;
      while (!locMinStack.empty())
      {
        lm = locMinStack.top();
        lm->innerLM = !lm->innerLM;
        locMinStack.pop();
      }
      // 2. swap edge kinds
      for (Edge* e : allEdges)
        if (e->kind == EdgeKind::ascend)
          e->kind = EdgeKind::descend;
        else
          e->kind = EdgeKind::ascend;
    }
    else
    {
      // path orientation is fine so ...
      while (!locMinStack.empty())
        locMinStack.pop();
    }

    std::sort(allEdges.begin(), allEdges.end(), EdgeListSort);
    if (!FixupEdgeIntersects())
    { 
      CleanUp();
      triResult = TriangulateResult::paths_intersect;
      return Paths64(); // oops!
    }

    std::sort(allVertices.begin(), allVertices.end(), VertexListSort);
    MergeDupOrCollinearVertices();
    
    int64_t currY = allVertices[0]->pt.y;
    for (auto vIt = allVertices.begin(); vIt != allVertices.end(); ++vIt)
    {
      Vertex2* v = *vIt;
      if (v->edges.empty()) continue;
      if (v->pt.y != currY)
      {
        // JOIN AN INNER LOCMIN WITH A SUITABLE EDGE BELOW
        while (!locMinStack.empty())
        {
          Vertex2* lm = locMinStack.top();
          locMinStack.pop();
          Edge* e = CreateInnerLocMinLooseEdge(lm);
          if (!e)
          {
            CleanUp();
            triResult = TriangulateResult::fail;
            return Paths64(); // oops!
          }

          if (IsHorizontal(*e))
          {
            if (e->vL == e->vB)
              DoTriangulateLeft(e, e->vB, currY); else
              DoTriangulateRight(e, e->vB, currY);
          }
          else
          {
            DoTriangulateLeft(e, e->vB, currY);
            if (!EdgeCompleted(e))
              DoTriangulateRight(e, e->vB, currY);
          }

          // and because adding locMin edges to Actives was delayed ..
          AddEdgeToActives(lm->edges[0]);
          AddEdgeToActives(lm->edges[1]);
        }

        while (!horzEdgeStack.empty())
        {
          Edge* e = horzEdgeStack.top();
          horzEdgeStack.pop();
          if (EdgeCompleted(e)) continue;
          if (e->vB == e->vL) // #45
          {
            if (IsLeftEdge(*e))
              DoTriangulateLeft(e, e->vB, currY);
          }
          else
            if (IsRightEdge(*e))
              DoTriangulateRight(e, e->vB, currY);
        }
        currY = v->pt.y;
      }

      for (int i = static_cast<int>(v->edges.size()) - 1; i >= 0; --i)
      {
        // the following line may look superfluous, but within this loop  
        // v->edges may be altered with additions and or deletions. 
        // So this line *is* necessary (and why we can't use an iterator).
        // Also, we need to use a *descending* index which is safe because
        // any additions will be loose edges which are ignored here.
        if (i >= static_cast<int>(v->edges.size())) continue;

        Edge* e = v->edges[i];
        if (EdgeCompleted(e) || IsLooseEdge(*e)) continue;

        if (v == e->vB)
        {
          if (IsHorizontal(*e))
            horzEdgeStack.push(e);
          // delay adding locMin edges to actives
          if (!v->innerLM)
            AddEdgeToActives(e);
        }
        else
        {
          if (IsHorizontal(*e))
            horzEdgeStack.push(e);
          else if (IsLeftEdge(*e))
            DoTriangulateLeft(e, e->vB, v->pt.y);
          else
            DoTriangulateRight(e, e->vB, v->pt.y);
        }
      } // for v->edges loop
      
      if (v->innerLM) locMinStack.push(v);

    } // for allVertices loop


    while (!horzEdgeStack.empty())
    {
      Edge* e = horzEdgeStack.top();
      horzEdgeStack.pop();
      if (!EdgeCompleted(e) && e->vB == e->vL) 
        DoTriangulateLeft(e, e->vB, currY);
    }

    if (useDelaunay)
    {
      // Convert triangles to Delaunay conforming
      while (!pendingDelaunayStack.empty())
      {
        Edge* e = pendingDelaunayStack.top();
        pendingDelaunayStack.pop();
        ForceLegal(e);
      }
    }

    Paths64 res;
    res.reserve(allTriangles.size());
    for (auto tri : allTriangles)
    {
      Path64 p = PathFromTriangle(*tri);
      int cps = CrossProductSign(p[0], p[1], p[2]);
      if (cps == 0) continue; // skip any empty triangles
      if (cps < 0) // ccw
        std::reverse(p.begin(), p.end());
      res.push_back(p);
    }

    CleanUp();
    triResult = TriangulateResult::success;
    return res;
  }

  static double DistSqr(const Point64& pt1, const Point64& pt2)
  {
    return Sqr(pt1.x - pt2.x) + Sqr(pt1.y - pt2.y);
  }

  void Delaunay::AddPath(const Path64& path)
  {
    size_t len = path.size(), i0 = 0, iPrev, iNext;
    // find the first locMin for the current path
    if (!FindLocMinIdx(path, len, i0)) return;
    iPrev = Prev(i0, len);
    while (path[iPrev] == path[i0]) iPrev = Prev(iPrev, len);
    iNext = Next(i0, len);

    // it is possible for a locMin here to simply be a
    // collinear spike that should be ignored, so ...
    size_t i = i0;
    while (CrossProductSign(path[iPrev], path[i], path[iNext]) == 0)
    {
      FindLocMinIdx(path, len, i);
      if (i == i0) return; // this is an entirely collinear path
      iPrev = Prev(i, len);
      while (path[iPrev] == path[i]) iPrev = Prev(iPrev, len);
      iNext = Next(i, len);
    }

    size_t vert_cnt = allVertices.size();

    // we are now at the first legitimate locMin
    Vertex2* v0 = allVertices.emplace_back(new Vertex2(path[i]));

    if (LeftTurning(path[iPrev], path[i], path[iNext]))
      v0->innerLM = true;
    Vertex2* vPrev = v0;
    i = iNext;

    for (;;)
    {
      // vPrev is a locMin here
      locMinStack.push(vPrev);
      // ? update lowermostVertex ...
      if (!lowermostVertex ||
        vPrev->pt.y > lowermostVertex->pt.y ||
        (vPrev->pt.y == lowermostVertex->pt.y &&
        vPrev->pt.x < lowermostVertex->pt.x)) 
          lowermostVertex = vPrev;

      iNext = Next(i, len);
      if (CrossProductSign(vPrev->pt, path[i], path[iNext]) == 0)
      {
        i = iNext;
        continue;
      }

      // ascend up next bound to LocMax
      while (path[i].y <= vPrev->pt.y)
      {
        Vertex2* v = allVertices.emplace_back(new Vertex2(path[i]));
        CreateEdge(vPrev, v, EdgeKind::ascend);
        vPrev = v;
        i = iNext;
        iNext = Next(i, len);

        while (CrossProductSign(vPrev->pt, path[i], path[iNext]) == 0)
        {
          i = iNext;
          iNext = Next(i, len);
        }
      }

      // Now at a locMax, so descend to next locMin
      Vertex2* vPrevPrev = vPrev;
      while (i != i0 && path[i].y >= vPrev->pt.y)
      {
        Vertex2* v = allVertices.emplace_back(new Vertex2(path[i]));
        CreateEdge(v, vPrev, EdgeKind::descend);
        vPrevPrev = vPrev;
        vPrev = v;
        i = iNext;
        iNext = Next(i, len);

        while (CrossProductSign(vPrev->pt, path[i], path[iNext]) == 0)
        {
          i = iNext;
          iNext = Next(i, len);
        }
      }

      // now at the next locMin
      if (i == i0) break; // break for(;;) loop
      if (LeftTurning(vPrevPrev->pt, vPrev->pt, path[i]))
        vPrev->innerLM = true;
    }
    CreateEdge(v0, vPrev, EdgeKind::descend);

    // finally, ignore this path if is not a polygon or too small
    len = allVertices.size() - vert_cnt;
    i = vert_cnt;
    if (len < 3 || (len == 3 &&      // or just a very tiny triangle
      ((DistSqr(allVertices[i]->pt, allVertices[i + 1]->pt) <= 1) ||
        (DistSqr(allVertices[i + 1]->pt, allVertices[i + 2]->pt) <= 1) ||
        (DistSqr(allVertices[i + 2]->pt, allVertices[i]->pt) <= 1))))
    {
      for (size_t j = vert_cnt; j < allVertices.size(); ++j)
        allVertices[j]->edges.clear(); // flag to ignore
    }
  }

  bool Delaunay::AddPaths(const Paths64& paths)
  {
    const auto total_vertex_count =
      std::accumulate(paths.begin(), paths.end(), size_t(0),
        [](const auto& a, const Path64& path)
        {return a + path.size(); });
    if (total_vertex_count == 0) return false;
    allVertices.reserve(allVertices.capacity() + total_vertex_count);
    allEdges.reserve(allEdges.capacity() + total_vertex_count);

    for (const Path64& path : paths)
      AddPath(path);
    return (allVertices.size() > 2);
  }

  TriangulateResult Triangulate(const Paths64& pp, Paths64& solution, bool useDelaunay)
  {
    TriangulateResult result;
    Delaunay d(useDelaunay);
    solution = d.Execute(pp, result);
    return result;
  }

  TriangulateResult Triangulate(const PathsD& pp, int decPlaces, PathsD& solution, bool useDelaunay)
  {
    int ec;
    double scale;
    TriangulateResult result;
    if (decPlaces <= 0) scale = 1;
    else if (decPlaces > 8) scale = std::pow(10, 8);
    else scale = std::pow(10, decPlaces);
    Paths64 pp64 = ScalePaths<int64_t, double>(pp, scale, ec);

    Delaunay d(useDelaunay);
    Paths64 sol64 = d.Execute(pp64, result);
    solution = ScalePaths<double, int64_t>(sol64, 1 / scale, ec);
    return result;
  }

} // Clipper2Lib namespace

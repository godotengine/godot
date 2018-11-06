/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Version   :  10.0 (beta)                                                     *
* Date      :  8 Noveber 2017                                                  *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2017                                         *
* Purpose   :  Base clipping module                                            *
* License   : http://www.boost.org/LICENSE_1_0.txt                             *
*******************************************************************************/

#include <stdlib.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <ostream>
#include <functional>
#include "clipper.h"

#include <iostream>

namespace clipperlib {

  enum VertexFlags { vfNone = 0, vfOpenStart = 1, vfOpenEnd = 2, vfLocalMax = 4, vfLocMin = 8 };
  inline VertexFlags operator|(VertexFlags a, VertexFlags b) {
    return static_cast<VertexFlags>(static_cast<int>(a) | static_cast<int>(b));
  }
  inline VertexFlags& operator |=(VertexFlags& a, VertexFlags b) { return a = a | b; }

  struct Vertex {
    Point64      pt;
    Vertex      *next;
    Vertex      *prev;
    VertexFlags  flags;
  };

  struct LocalMinima {
    Vertex      *vertex;
    PathType     polytype;
    bool         is_open;
  };

  struct Scanline {
    int64_t      y;
    Scanline    *next;
  };

  struct IntersectNode {
    Point64      pt;
    Active      *edge1;
    Active      *edge2;
  };

  struct LocMinSorter {
    inline bool operator()(const LocalMinima* locMin1, const LocalMinima* locMin2) {
      return locMin2->vertex->pt.y < locMin1->vertex->pt.y;
    }
  };

  //------------------------------------------------------------------------------
  // PolyPath methods ...
  //------------------------------------------------------------------------------

  void PolyPath::Clear()
  {
    for (size_t i = 0; i < childs_.size(); ++i) {
      childs_[i]->Clear();
      delete childs_[i];
    }
    childs_.resize(0);
  }
  //------------------------------------------------------------------------------

  PolyPath::PolyPath()
  {
    parent_ = nullptr;
  }
  //------------------------------------------------------------------------------

  PolyPath::PolyPath(PolyPath *parent, const Path &path)
  {
    parent_ = parent;
    path_ = path;
  }
  //------------------------------------------------------------------------------

  int PolyPath::ChildCount() const { return (int)childs_.size(); }

  //------------------------------------------------------------------------------

  PolyPath& PolyPath::AddChild(const Path &path)
  {
    PolyPath* child = new PolyPath(this, path);
    childs_.push_back(child);
    return *child;
  }
  //------------------------------------------------------------------------------

  PolyPath& PolyPath::GetChild(unsigned index)
  {
    if (index < 0 || index >= childs_.size())
      throw ClipperException("invalid range in PolyPath::GetChild.");
    return *childs_[index];
  }
  //------------------------------------------------------------------------------

  PolyPath* PolyPath::GetParent() const { return parent_; }

  //------------------------------------------------------------------------------
  Path &PolyPath::GetPath() { return path_; }

  //------------------------------------------------------------------------------
  bool PolyPath::IsHole() const
  {
    bool result = true;
    PolyPath* pp = parent_;
    while (pp) {
      result = !result;
      pp = pp->parent_;
    }
    return result;
  }

  //------------------------------------------------------------------------------
  // miscellaneous functions ...
  //------------------------------------------------------------------------------

  inline int64_t Round(double val)
  {
    if ((val < 0)) return static_cast<int64_t>(val - 0.5);
    else return static_cast<int64_t>(val + 0.5);
  }
  //------------------------------------------------------------------------------

  inline double Abs(double val) { return val < 0 ? -val : val; }
  //------------------------------------------------------------------------------

  inline int Abs(int val) { return val < 0 ? -val : val; }
  //------------------------------------------------------------------------------

  inline bool IsOdd(int val) { return val & 1 ? true : false; }
  //------------------------------------------------------------------------------

  inline bool IsHotEdge(const Active &e) { return (e.outrec); }
  //------------------------------------------------------------------------------

  inline bool IsOpen(const Active &e) { return (e.local_min->is_open); }
  //------------------------------------------------------------------------------

  inline bool IsStartSide(const Active &e) { return (&e == e.outrec->start_e); }
  //------------------------------------------------------------------------------

  inline void SwapSides(OutRec &outrec)
  {
    Active *e2 = outrec.start_e;
    outrec.start_e = outrec.end_e;
    outrec.end_e = e2;
    outrec.pts = outrec.pts->next;
  }
  //------------------------------------------------------------------------------

  bool FixOrientation(Active &e)
  {
    bool result = true;
    Active* e2 = &e;
    while (e2->prev_in_ael) {
      e2 = e2->prev_in_ael;
      if (e2->outrec && !IsOpen(*e2)) result = !result;
    }
    if (result != IsStartSide(e)) {
      if (result) e.outrec->flag = orOuter;
      else e.outrec->flag = orInner;
      SwapSides(*e.outrec);
      return true; //all fixed
    }
    else return false; //no fix needed
  }
  //------------------------------------------------------------------------------

  inline bool IsHorizontal(const Active &e) { return (e.dx == CLIPPER_HORIZONTAL); }
  //------------------------------------------------------------------------------

  inline int64_t TopX(const Active &edge, const int64_t currentY)
  {
    return (currentY == edge.top.y) ?
      edge.top.x :
      edge.bot.x + Round(edge.dx *(currentY - edge.bot.y));
  }
  //------------------------------------------------------------------------------

  inline int64_t GetTopDeltaX(const Active &e1, const Active &e2)
  {
    return (e1.top.y > e2.top.y) ?
      TopX(e2, e1.top.y) - e1.top.x :
      e2.top.x - TopX(e1, e2.top.y);
  }
  //----------------------------------------------------------------------------

  inline void SwapActives(Active *&e1, Active *&e2) {
    Active *e = e1; e1 = e2; e2 = e;
  }
  //----------------------------------------------------------------------

  inline void MoveEdgeToFollowLeftInAEL(Active &e, Active &eLeft)
  {
    Active *aelPrev, *aelNext;
    //extract first ...
    aelPrev = e.prev_in_ael;
    aelNext = e.next_in_ael;
    aelPrev->next_in_ael = aelNext;
    if (aelNext) aelNext->prev_in_ael = aelPrev;
    //now reinsert ...
    e.next_in_ael = eLeft.next_in_ael;
    eLeft.next_in_ael->prev_in_ael = &e;
    e.prev_in_ael = &eLeft;
    eLeft.next_in_ael = &e;
  }
  //----------------------------------------------------------------------------

  inline bool E2InsertsBeforeE1(const Active &e1, const Active &e2, bool prefer_left)
  {
    if (prefer_left) {
      return e2.curr.x == e1.curr.x ?
        GetTopDeltaX(e1, e2) < 0 : e2.curr.x < e1.curr.x;
    }
    else
      return e2.curr.x == e1.curr.x ?
      GetTopDeltaX(e1, e2) <= 0 : e2.curr.x <= e1.curr.x;
  }
  //------------------------------------------------------------------------------

  inline PathType GetPolyType(const Active &e) { return e.local_min->polytype; }
  //------------------------------------------------------------------------------

  inline bool IsSamePolyType(const Active &e1, const Active &e2)
  {
    return e1.local_min->polytype == e2.local_min->polytype;
  }
  //------------------------------------------------------------------------------

  Point64 GetIntersectPoint(const Active &e1, const Active &e2)
  {
    double b1, b2;
    if (e1.dx == e2.dx) return Point64(TopX(e1, e1.curr.y), e1.curr.y);

    if (e1.dx == 0) {
      if (IsHorizontal(e2)) return Point64(e1.bot.x, e2.bot.y);
      b2 = e2.bot.y - (e2.bot.x / e2.dx);
      return Point64(e1.bot.x, Round(e1.bot.x / e2.dx + b2));
    }
    else if (e2.dx == 0) {
      if (IsHorizontal(e1)) return Point64(e2.bot.x, e1.bot.y);
      b1 = e1.bot.y - (e1.bot.x / e1.dx);
      return Point64(e2.bot.x, Round(e2.bot.x / e1.dx + b1));
    }
    else {
      b1 = e1.bot.x - e1.bot.y * e1.dx;
      b2 = e2.bot.x - e2.bot.y * e2.dx;
      double q = (b2 - b1) / (e1.dx - e2.dx);
      return (std::fabs(e1.dx) < std::fabs(e2.dx)) ?
        Point64(Round(e1.dx * q + b1), Round(q)) :
        Point64(Round(e2.dx * q + b2), Round(q));
    }
  }
  //------------------------------------------------------------------------------

  inline void SetDx(Active &e)
  {
    int64_t dy = (e.top.y - e.bot.y);
    e.dx = (dy == 0) ? CLIPPER_HORIZONTAL : (double)(e.top.x - e.bot.x) / dy;
  }
  //---------------------------------------------------------------------------

  inline Vertex& NextVertex(Active &e)
  {
    return (e.wind_dx > 0 ? *e.vertex_top->next : *e.vertex_top->prev);
  }
  //------------------------------------------------------------------------------

  inline bool IsMaxima(const Active &e)
  {
    return (e.vertex_top->flags & vfLocalMax);
  }
  //------------------------------------------------------------------------------

  void TerminateHotOpen(Active &e) {
    if (e.outrec->start_e == &e)
      e.outrec->start_e = NULL;
    else
      e.outrec->end_e = NULL;
    e.outrec = NULL;
  }
  //------------------------------------------------------------------------------

  Active *GetMaximaPair(const Active &e)
  {
    Active *e2;
    if (IsHorizontal(e)) {
      //we can't be sure whether the MaximaPair is on the left or right, so ...
      e2 = e.prev_in_ael;
      while (e2 && e2->curr.x >= e.top.x) {
        if (e2->vertex_top == e.vertex_top) return e2;  //Found!
        e2 = e2->prev_in_ael;
      }
      e2 = e.next_in_ael;
      while (e2 && (TopX(*e2, e.top.y) <= e.top.x)) {
        if (e2->vertex_top == e.vertex_top) return e2;  //Found!
        e2 = e2->next_in_ael;
      }
      return NULL;
    }
    else {
      e2 = e.next_in_ael;
      while (e2) {
        if (e2->vertex_top == e.vertex_top) return e2;  //Found!
        e2 = e2->next_in_ael;
      }
      return NULL;
    }
  }
  //------------------------------------------------------------------------------

  inline int PointCount(OutPt *op)
  {
    if (!op) return 0;
    OutPt *p = op;
    int cnt = 0;
    do {
      cnt++;
      p = p->next;
    } while (p != op);
    return cnt;
  }
  //------------------------------------------------------------------------------

  inline void DisposeOutPts(OutPt *&op)
  {
    if (!op) return;
    op->prev->next = NULL;
    while (op) {
      OutPt *tmpPp = op;
      op = op->next;
      delete tmpPp;
    }
  }
  //------------------------------------------------------------------------------

  bool IntersectListSort(IntersectNode *node1, IntersectNode *node2)
  {
    return (node2->pt.y < node1->pt.y);
  }
  //------------------------------------------------------------------------------

  inline void SetOrientation(OutRec &outrec, Active &e1, Active &e2)
  {
    outrec.start_e = &e1;
    outrec.end_e = &e2;
    e1.outrec = &outrec;
    e2.outrec = &outrec;
  }
  //------------------------------------------------------------------------------

  void SwapOutrecs(Active &e1, Active &e2)
  {
    OutRec *or1 = e1.outrec;
    OutRec *or2 = e2.outrec;
    if (or1 == or2) {
      Active *e = or1->start_e;
      or1->start_e = or1->end_e;
      or1->end_e = e;
      return;
    }
    if (or1) {
      if (&e1 == or1->start_e) or1->start_e = &e2;
      else or1->end_e = &e2;
    }
    if (or2) {
      if (&e2 == or2->start_e) or2->start_e = &e1;
      else or2->end_e = &e1;
    }
    e1.outrec = or2;
    e2.outrec = or1;
  }
  //------------------------------------------------------------------------------

  inline bool EdgesAdjacentInSel(const IntersectNode &inode)
  {
    return (inode.edge1->next_in_sel == inode.edge2) || (inode.edge1->prev_in_sel == inode.edge2);
  }

  //------------------------------------------------------------------------------
  // Clipper class methods ...
  //------------------------------------------------------------------------------

  Clipper::Clipper()
  {
    Clear();
  }
  //--------------------------- ---------------------------------------------------

  Clipper::~Clipper()
  {
    Clear();
  }
  //------------------------------------------------------------------------------

  void Clipper::CleanUp()
  {
    while (actives_) DeleteFromAEL(*actives_);
    scanline_list_ = ScanlineList(); //resets priority_queue
    DisposeAllOutRecs();
  }
  //------------------------------------------------------------------------------

  void Clipper::Clear()
  {
    DisposeVerticesAndLocalMinima();
    curr_loc_min_ = minima_list_.begin();
    minima_list_sorted_ = false;
    has_open_paths_ = false;
  }
  //------------------------------------------------------------------------------

  void Clipper::Reset()
  {
    if (!minima_list_sorted_) {
      std::sort(minima_list_.begin(), minima_list_.end(), LocMinSorter());
      minima_list_sorted_ = true;
    }
    for (MinimaList::const_iterator i = minima_list_.begin(); i != minima_list_.end(); ++i)
      InsertScanline((*i)->vertex->pt.y);
    curr_loc_min_ = minima_list_.begin();

    actives_ = NULL;
    sel_ = NULL;
  }
  //------------------------------------------------------------------------------

  inline void Clipper::InsertScanline(int64_t y) { scanline_list_.push(y); }
  //------------------------------------------------------------------------------

  bool Clipper::PopScanline(int64_t &y)
  {
    if (scanline_list_.empty()) return false;
    y = scanline_list_.top();
    scanline_list_.pop();
    while (!scanline_list_.empty() && y == scanline_list_.top())
      scanline_list_.pop(); // Pop duplicates.
    return true;
  }
  //------------------------------------------------------------------------------

  bool Clipper::PopLocalMinima(int64_t y, LocalMinima *&local_minima)
  {
    if (curr_loc_min_ == minima_list_.end() || (*curr_loc_min_)->vertex->pt.y != y) return false;
    local_minima = (*curr_loc_min_);
    ++curr_loc_min_;
    return true;
  }
  //------------------------------------------------------------------------------

  void Clipper::DisposeAllOutRecs()
  {
    for (OutRecList::const_iterator i = outrec_list_.begin(); i != outrec_list_.end(); ++i) {
      if ((*i)->pts) DisposeOutPts((*i)->pts);
      delete (*i);
    }
    outrec_list_.resize(0);
  }
  //------------------------------------------------------------------------------

  void Clipper::DisposeVerticesAndLocalMinima()
  {
    for (MinimaList::iterator ml_iter = minima_list_.begin();
      ml_iter != minima_list_.end(); ++ml_iter)
        delete (*ml_iter);
    minima_list_.clear();
    VerticesList::iterator vl_iter;
    for (vl_iter = vertex_list_.begin(); vl_iter != vertex_list_.end(); ++vl_iter)
      delete[] (*vl_iter);
    vertex_list_.clear();
  }
  //------------------------------------------------------------------------------

  void Clipper::AddLocMin(Vertex &vert, PathType polytype, bool is_open)
  {
    //make sure the vertex is added only once ...
    if (vfLocMin & vert.flags) return;
    vert.flags |= vfLocMin;

    LocalMinima *lm = new LocalMinima();
    lm->vertex = &vert;
    lm->polytype = polytype;
    lm->is_open = is_open;
    minima_list_.push_back(lm);
  }
  //----------------------------------------------------------------------------

  void Clipper::AddPathToVertexList(const Path &path, PathType polytype, bool is_open)
  {
    int path_len = (int)path.size();
    while (path_len > 1 && (path[path_len - 1] == path[0])) --path_len;
    if (path_len < 2) return;

    int i = 1;
    bool p0_is_minima = false, p0_is_maxima = false, going_up;
    //find the first non-horizontal segment in the path ...
    while ((i < path_len) && (path[i].y == path[0].y)) ++i;
    bool is_flat = (i == path_len);
    if (is_flat) {
      if (!is_open) return;    //Ignore closed paths that have ZERO area.
      going_up = false;           //And this just stops a compiler warning.
    }
    else
    {
      going_up = path[i].y < path[0].y; //because I'm using an inverted Y-axis display
      if (going_up) {
        i = path_len - 1;
        while (path[i].y == path[0].y) --i;
        p0_is_minima = path[i].y < path[0].y; //p[0].y == a minima
      }
      else {
        i = path_len - 1;
        while (path[i].y == path[0].y) --i;
        p0_is_maxima = path[i].y > path[0].y; //p[0].y == a maxima
      }
    }

    Vertex *vertices = new Vertex [path_len];
    vertex_list_.push_back(vertices);

    vertices[0].pt = path[0];
    vertices[0].flags = vfNone;

    if (is_open) {
      vertices[0].flags |= vfOpenStart;
      if (going_up) AddLocMin(vertices[0], polytype, is_open);
      else vertices[0].flags |= vfLocalMax;
    }

    //nb: polygon orientation is determined later (see InsertLocalMinimaIntoAEL).
    i = 0;
    for (int j = 1; j < path_len; ++j) {
      if (path[j] == vertices[i].pt) continue; //ie skips duplicates
      vertices[j].pt = path[j];
      vertices[j].flags = vfNone;
      vertices[i].next = &vertices[j];
      vertices[j].prev = &vertices[i];
      if (path[j].y > path[i].y && going_up) {
        vertices[i].flags |= vfLocalMax;
        going_up = false;
      }
      else if (path[j].y < path[i].y && !going_up) {
        going_up = true;
        AddLocMin(vertices[i], polytype, is_open);
      }
      i = j;
    }
    //i: index of the last vertex in the path.
    vertices[i].next = &vertices[0];
    vertices[0].prev = &vertices[i];

    if (is_open) {
      vertices[i].flags |= vfOpenEnd;
      if (going_up)
        vertices[i].flags |= vfLocalMax;
      else AddLocMin(vertices[i], polytype, is_open);
    }
    else if (going_up) {
      //going up so find local maxima ...
      Vertex *v = &vertices[i];
      while (v->next->pt.y <= v->pt.y) v = v->next;
      v->flags |= vfLocalMax;
      if (p0_is_minima) AddLocMin(vertices[0], polytype, is_open);
    }
    else {
      //going down so find local minima ...
      Vertex *v = &vertices[i];
      while (v->next->pt.y >= v->pt.y) v = v->next;
      AddLocMin(*v, polytype, is_open);
      if (p0_is_maxima)
        vertices[0].flags |= vfLocalMax;
    }
  }
  //------------------------------------------------------------------------------

  void Clipper::AddPath(const Path &path, PathType polytype, bool is_open)
  {
    if (is_open) {
      if (polytype == ptClip)
        throw ClipperException("AddPath: Only subject paths may be open.");
      has_open_paths_ = true;
    }
    minima_list_sorted_ = false;
    AddPathToVertexList(path, polytype, is_open);
  }
  //------------------------------------------------------------------------------

  void Clipper::AddPaths(const Paths &paths, PathType polytype, bool is_open)
  {
    for (Paths::size_type i = 0; i < paths.size(); ++i)
      AddPath(paths[i], polytype, is_open);
  }
  //------------------------------------------------------------------------------

  bool Clipper::IsContributingClosed(const Active& e) const
  {
    switch (fillrule_) {
      case frNonZero:
        if (Abs(e.wind_cnt) != 1) return false;
        break;
      case frPositive:
        if (e.wind_cnt != 1) return false;
        break;
      case frNegative:
        if (e.wind_cnt != -1) return false;
        break;
    }

    switch (cliptype_) {
    case ctIntersection:
      switch (fillrule_) {
        case frEvenOdd:
        case frNonZero: return (e.wind_cnt2 != 0);
        case frPositive: return (e.wind_cnt2 > 0);
        case frNegative: return (e.wind_cnt2 < 0);
      }
      break;
    case ctUnion:
      switch (fillrule_) {
        case frEvenOdd:
        case frNonZero: return (e.wind_cnt2 == 0);
        case frPositive: return (e.wind_cnt2 <= 0);
        case frNegative: return (e.wind_cnt2 >= 0);
      }
      break;
    case ctDifference:
      if (GetPolyType(e) == ptSubject)
        switch (fillrule_) {
          case frEvenOdd:
          case frNonZero: return (e.wind_cnt2 == 0);
          case frPositive: return (e.wind_cnt2 <= 0);
          case frNegative: return (e.wind_cnt2 >= 0);
        }
      else
        switch (fillrule_) {
          case frEvenOdd:
          case frNonZero: return (e.wind_cnt2 != 0);
          case frPositive: return (e.wind_cnt2 > 0);
          case frNegative: return (e.wind_cnt2 < 0);
        };
      break;
    case ctXor: return true; //XOr is always contributing unless open
    }
    return false; //we should never get here
  }
  //------------------------------------------------------------------------------

  inline bool Clipper::IsContributingOpen(const Active& e) const
  {
    switch (cliptype_) {
      case ctIntersection: return (e.wind_cnt2 != 0);
      case ctUnion: return (e.wind_cnt == 0 && e.wind_cnt2 == 0);
      case ctDifference: return (e.wind_cnt2 == 0);
      case ctXor: return (e.wind_cnt != 0) != (e.wind_cnt2 != 0);
    }
    return false; //stops compiler error
  }
  //------------------------------------------------------------------------------

  void Clipper::SetWindingLeftEdgeClosed(Active &e)
  {
    //Wind counts generally refer to polygon regions not edges, so here an edge's
    //WindCnt indicates the higher of the two wind counts of the regions touching
    //the edge. (Note also that adjacent region wind counts only ever differ
    //by one, and open paths have no meaningful wind directions or counts.)

    Active *e2 = e.prev_in_ael;
    //find the nearest closed path edge of the same PolyType in AEL (heading left)
    PathType pt = GetPolyType(e);
    while (e2 && (GetPolyType(*e2) != pt || IsOpen(*e2))) e2 = e2->prev_in_ael;

    if (!e2) {
      e.wind_cnt = e.wind_dx;
      e2 = actives_;
    }
    else if (fillrule_ == frEvenOdd) {
      e.wind_cnt = e.wind_dx;
      e.wind_cnt2 = e2->wind_cnt2;
      e2 = e2->next_in_ael;
    }
    else {
      //NonZero, Positive, or Negative filling here ...
      //if e's WindCnt is in the SAME direction as its WindDx, then e is either
      //an outer left or a hole right boundary, so edge must be inside 'e'.
      //(neither e.WindCnt nor e.WindDx should ever be 0)
      if (e2->wind_cnt * e2->wind_dx < 0) {
        //opposite directions so edge is outside 'e' ...
        if (Abs(e2->wind_cnt) > 1) {
          //outside prev poly but still inside another.
          if (e2->wind_dx * e.wind_dx < 0)
            //reversing direction so use the same WC
            e.wind_cnt = e2->wind_cnt;
          else
            //otherwise keep 'reducing' the WC by 1 (ie towards 0) ...
            e.wind_cnt = e2->wind_cnt + e.wind_dx;
        }
        else
          //now outside all polys of same polytype so set own WC ...
          e.wind_cnt = (IsOpen(e) ? 1 : e.wind_dx);
      }
      else {
        //edge must be inside 'e'
        if (e2->wind_dx * e.wind_dx < 0)
          //reversing direction so use the same WC
          e.wind_cnt = e2->wind_cnt;
        else
          //otherwise keep 'increasing' the WC by 1 (ie away from 0) ...
          e.wind_cnt = e2->wind_cnt + e.wind_dx;
      };
      e.wind_cnt2 = e2->wind_cnt2;
      e2 = e2->next_in_ael; //ie get ready to calc WindCnt2
    }

    //update wind_cnt2 ...
    if (fillrule_ == frEvenOdd)
      while (e2 != &e) {
        if (GetPolyType(*e2) != pt && !IsOpen(*e2))
          e.wind_cnt2 = (e.wind_cnt2 == 0 ? 1 : 0);
        e2 = e2->next_in_ael;
      }
    else
      while (e2 != &e) {
        if (GetPolyType(*e2) != pt && !IsOpen(*e2))
          e.wind_cnt2 += e2->wind_dx;
        e2 = e2->next_in_ael;
      }
  }
  //------------------------------------------------------------------------------

  void Clipper::SetWindingLeftEdgeOpen(Active &e)
  {
    Active *e2 = actives_;
    if (fillrule_ == frEvenOdd) {
      int cnt1 = 0, cnt2 = 0;
      while (e2 != &e) {
        if (GetPolyType(*e2) == ptClip) cnt2++;
        else if (!IsOpen(*e2)) cnt1++;
        e2 = e2->next_in_ael;
      }
      e.wind_cnt = (IsOdd(cnt1) ? 1 : 0);
      e.wind_cnt2 = (IsOdd(cnt2) ? 1 : 0);
    }
    else {
      while (e2 != &e) {
        if (GetPolyType(*e2) == ptClip) e.wind_cnt2 += e2->wind_dx;
        else if (!IsOpen(*e2)) e.wind_cnt += e2->wind_dx;
        e2 = e2->next_in_ael;
      }
    }
  }
  //------------------------------------------------------------------------------

  void Clipper::InsertEdgeIntoAEL(Active &e1, Active *e2)
  {
    if (!actives_) {
      e1.prev_in_ael = NULL;
      e1.next_in_ael = NULL;
      actives_ = &e1;
      return;
    }
    if (!e2) {
      if (E2InsertsBeforeE1(*actives_, e1, false)) {
        e1.prev_in_ael = NULL;
        e1.next_in_ael = actives_;
        actives_->prev_in_ael = &e1;
        actives_ = &e1;
        return;
      }
      e2 = actives_;
      while (e2->next_in_ael &&
        E2InsertsBeforeE1(e1, *e2->next_in_ael, false))
        e2 = e2->next_in_ael;
    }
    else {
      while (e2->next_in_ael &&
        E2InsertsBeforeE1(e1, *e2->next_in_ael, true))
        e2 = e2->next_in_ael;
    }
    e1.next_in_ael = e2->next_in_ael;
    if (e2->next_in_ael) e2->next_in_ael->prev_in_ael = &e1;
    e1.prev_in_ael = e2;
    e2->next_in_ael = &e1;
  }
  //----------------------------------------------------------------------

  void Clipper::InsertLocalMinimaIntoAEL(int64_t bot_y)
  {
    LocalMinima *local_minima;
    Active *left_bound, *right_bound;
    //Add any local minima at BotY ...
    while (PopLocalMinima(bot_y, local_minima)) {
      if ((local_minima->vertex->flags & vfOpenStart) > 0) {
        left_bound = NULL;
      }
      else {
        left_bound = new Active();
        left_bound->bot = local_minima->vertex->pt;
        left_bound->curr = left_bound->bot;
        left_bound->vertex_top = local_minima->vertex->prev; //ie descending
        left_bound->top = left_bound->vertex_top->pt;
        left_bound->wind_dx = -1;
        left_bound->local_min = local_minima;
        SetDx(*left_bound);
      }

      if ((local_minima->vertex->flags & vfOpenEnd) > 0) {
        right_bound = NULL;
      }
      else {
        right_bound = new Active();
        right_bound->bot = local_minima->vertex->pt;
        right_bound->curr = right_bound->bot;
        right_bound->vertex_top = local_minima->vertex->next; //ie ascending
        right_bound->top = right_bound->vertex_top->pt;
        right_bound->wind_dx = 1;
        right_bound->local_min = local_minima;
        SetDx(*right_bound);
      }

      //Currently LeftB is just the descending bound and RightB is the ascending.
      //Now if the LeftB isn't on the left of RightB then we need swap them.
      if (left_bound && right_bound) {
        if (IsHorizontal(*left_bound)) {
          if (left_bound->top.x > left_bound->bot.x) SwapActives(left_bound, right_bound);
        }
        else if (IsHorizontal(*right_bound)) {
          if (right_bound->top.x < right_bound->bot.x) SwapActives(left_bound, right_bound);
        }
        else if (left_bound->dx < right_bound->dx) SwapActives(left_bound, right_bound);
      }
      else if (!left_bound) {
        left_bound = right_bound;
        right_bound = NULL;
      }

      bool contributing;
      InsertEdgeIntoAEL(*left_bound, NULL);              //insert left edge
      if (IsOpen(*left_bound)) {
        SetWindingLeftEdgeOpen(*left_bound);
        contributing = IsContributingOpen(*left_bound);
      }
      else {
        SetWindingLeftEdgeClosed(*left_bound);
        contributing = IsContributingClosed(*left_bound);
      }

      if (right_bound != NULL) {
        right_bound->wind_cnt = left_bound->wind_cnt;
        right_bound->wind_cnt2 = left_bound->wind_cnt2;
        InsertEdgeIntoAEL(*right_bound, left_bound);     //insert right edge
        if (contributing)
          AddLocalMinPoly(*left_bound, *right_bound, left_bound->bot);
        if (IsHorizontal(*right_bound)) PushHorz(*right_bound);
        else InsertScanline(right_bound->top.y);
      }
      else if (contributing)
        StartOpenPath(*left_bound, left_bound->bot);

      if (IsHorizontal(*left_bound)) PushHorz(*left_bound);
      else InsertScanline(left_bound->top.y);

      if (right_bound && left_bound->next_in_ael != right_bound) {
        //intersect edges that are between left and right bounds ...
        Active *e = right_bound->next_in_ael;
        MoveEdgeToFollowLeftInAEL(*right_bound, *left_bound);
        while (right_bound->next_in_ael != e) {
          //nb: For calculating winding counts etc, IntersectEdges() assumes
          //that rightB will be to the right of e ABOVE the intersection ...
          IntersectEdges(*right_bound, *right_bound->next_in_ael, right_bound->bot);
          SwapPositionsInAEL(*right_bound, *right_bound->next_in_ael);
        } //while
      } //if

    } //while (PopLocalMinima())
  }
  //------------------------------------------------------------------------------

  inline void Clipper::PushHorz(Active &e)
  {
    e.next_in_sel = (sel_ ? sel_ : NULL);
    sel_ = &e;
  }
  //------------------------------------------------------------------------------

  inline bool Clipper::PopHorz(Active *&e)
  {
    e = sel_;
    if (!e) return false;
    sel_ = sel_->next_in_sel;
    return true;
  }
  //------------------------------------------------------------------------------

  OutRec* Clipper::GetOwner(const Active *e)
  {
    if (IsHorizontal(*e) && e->top.x < e->bot.x) {
      e = e->next_in_ael;
      while (e && (!IsHotEdge(*e) || IsOpen(*e)))
        e = e->next_in_ael;
      if (!e) return NULL;
      return ((e->outrec->flag == orOuter) == (e->outrec->start_e == e)) ?
        e->outrec->owner : e->outrec;
    }
    else {
      e = e->prev_in_ael;
      while (e && (!IsHotEdge(*e) || IsOpen(*e)))
        e = e->prev_in_ael;
      if (!e) return NULL;
      return ((e->outrec->flag == orOuter) == (e->outrec->end_e == e)) ?
        e->outrec->owner : e->outrec;
    }
  }
  //------------------------------------------------------------------------------

  void Clipper::AddLocalMinPoly(Active &e1, Active &e2, const Point64 pt)
  {
    OutRec *outrec = CreateOutRec();
    outrec->idx = (unsigned)outrec_list_.size();
    outrec_list_.push_back(outrec);
    outrec->owner = GetOwner(&e1);
    outrec->polypath = NULL;

    if (IsOpen(e1))
      outrec->flag = orOpen;
    else if (!outrec->owner || outrec->owner->flag == orInner)
      outrec->flag = orOuter;
    else
      outrec->flag = orInner;

    //now set orientation ...
    bool swap_sides_needed = false;
    if (IsHorizontal(e1)) {
      if (e1.top.x > e1.bot.x) swap_sides_needed = true;
    }
    else if (IsHorizontal(e2)) {
      if (e2.top.x < e2.bot.x) swap_sides_needed = true;
    }
    else if (e1.dx < e2.dx) swap_sides_needed = true;
    if ((outrec->flag == orOuter) != swap_sides_needed)
      SetOrientation(*outrec, e1, e2);
    else
      SetOrientation(*outrec, e2, e1);

    OutPt *op = CreateOutPt();
    op->pt = pt;
    op->next = op;
    op->prev = op;
    outrec->pts = op;
  }
  //------------------------------------------------------------------------------

  void Clipper::AddLocalMaxPoly(Active &e1, Active &e2, const Point64 pt)
  {
    if (!IsHotEdge(e2))
      throw new ClipperException("Error in AddLocalMaxPoly().");
    AddOutPt(e1, pt);
    if (e1.outrec == e2.outrec) {
      e1.outrec->start_e = NULL;
      e1.outrec->end_e = NULL;
      e1.outrec = NULL;
      e2.outrec = NULL;
    }
    //and to preserve the winding orientation of outrec ...
    else if (e1.outrec->idx < e2.outrec->idx)
      JoinOutrecPaths(e1, e2); else
      JoinOutrecPaths(e2, e1);
  }
  //------------------------------------------------------------------------------

  void Clipper::JoinOutrecPaths(Active &e1, Active &e2)
  {

    if (IsStartSide(e1) == IsStartSide(e2)) {
      //one or other edge orientation is wrong...
      if (IsOpen(e1)) SwapSides(*e2.outrec);
      else if (!FixOrientation(e1) && !FixOrientation(e2))
        throw new ClipperException("Error in JoinOutrecPaths()");
      if (e1.outrec->owner == e2.outrec) e1.outrec->owner = e2.outrec->owner;
    }

    //join E2 outrec path onto E1 outrec path and then delete E2 outrec path
    //pointers. (nb: Only very rarely do the joining ends share the same coords.)
    OutPt *p1_st = e1.outrec->pts;
    OutPt *p2_st = e2.outrec->pts;
    OutPt *p1_end = p1_st->next;
    OutPt *p2_end = p2_st->next;
    if (IsStartSide(e1)) {
      p2_end->prev = p1_st;
      p1_st->next = p2_end;
      p2_st->next = p1_end;
      p1_end->prev = p2_st;
      e1.outrec->pts = p2_st;
      e1.outrec->start_e = e2.outrec->start_e;
      if (e1.outrec->start_e) //ie closed path
        e1.outrec->start_e->outrec = e1.outrec;
    }
    else {
      p1_end->prev = p2_st;
      p2_st->next = p1_end;
      p1_st->next = p2_end;
      p2_end->prev = p1_st;
      e1.outrec->end_e = e2.outrec->end_e;
      if (e1.outrec->end_e) //ie closed path
        e1.outrec->end_e->outrec = e1.outrec;
    }

    //after joining, the E2.OutRec contains not vertices ...
    e2.outrec->start_e = NULL;
    e2.outrec->end_e = NULL;
    e2.outrec->pts = NULL;
    e2.outrec->owner = e1.outrec; //this may be redundant

    //and e1 and e2 are maxima and are about to be dropped from the Actives list.
    e1.outrec = NULL;
    e2.outrec = NULL;
  }
  //------------------------------------------------------------------------------

  inline void Clipper::TerminateHotOpen(Active &e)
  {
    if (e.outrec->start_e == &e) e.outrec->start_e = NULL;
    else e.outrec->end_e = NULL;
    e.outrec = NULL;
  }
  //------------------------------------------------------------------------------

  OutPt* Clipper::CreateOutPt()
  {
    //this is a virtual method as descendant classes may need
    //to produce descendant classes of OutPt ...
    return new OutPt();
  }
  //------------------------------------------------------------------------------

  OutRec* Clipper::CreateOutRec()
  {
    //this is a virtual method as descendant classes may need
    //to produce descendant classes of OutRec ...
    return new OutRec();
  }
  //------------------------------------------------------------------------------

  OutPt* Clipper::AddOutPt(Active &e, const Point64 pt)
  {
    //Outrec.pts[0]: a circular double-linked-list of POutPt.
    bool to_start = IsStartSide(e);
    OutPt *start_op = e.outrec->pts;
    OutPt *end_op = start_op->next;
    if (to_start) {
      if (pt == start_op->pt) return start_op;
    }
    else if (pt == end_op->pt) return end_op;

    OutPt *new_op = CreateOutPt();
    new_op->pt = pt;
    end_op->prev = new_op;
    new_op->prev = start_op;
    new_op->next = end_op;
    start_op->next = new_op;
    if (to_start) e.outrec->pts = new_op;
    return new_op;
  }
  //------------------------------------------------------------------------------

  void Clipper::StartOpenPath(Active &e, const Point64 pt)
  {
    OutRec *outrec = CreateOutRec();
    outrec->idx = (unsigned)outrec_list_.size();
    outrec_list_.push_back(outrec);
    outrec->flag = orOpen;
    outrec->owner = NULL;
    outrec->polypath = NULL;
    outrec->end_e = NULL;
    outrec->start_e = NULL;
    e.outrec = outrec;

    OutPt *op = CreateOutPt();
    op->pt = pt;
    op->next = op;
    op->prev = op;
    outrec->pts = op;
  }
  //------------------------------------------------------------------------------

  inline void Clipper::UpdateEdgeIntoAEL(Active *e)
  {
    e->bot = e->top;
    e->vertex_top = &NextVertex(*e);
    e->top = e->vertex_top->pt;
    e->curr = e->bot;
    SetDx(*e);
    if (!IsHorizontal(*e)) InsertScanline(e->top.y);
  }
  //------------------------------------------------------------------------------

  void Clipper::IntersectEdges(Active &e1, Active &e2, const Point64 pt)
  {
    e1.curr = pt;
    e2.curr = pt;

    //if either edge is an OPEN path ...
    if (has_open_paths_ && (IsOpen(e1) || IsOpen(e2))) {
      if (IsOpen(e1) && IsOpen(e2)) return; //ignore lines that intersect
      Active *edge_o, *edge_c;
      if (IsOpen(e1)) { edge_o = &e1; edge_c = &e2; }
      else { edge_o = &e2; edge_c = &e1; }

      switch (cliptype_) {
        case ctIntersection:
        case ctDifference:
          if (IsSamePolyType(*edge_o, *edge_c) || (Abs(edge_c->wind_cnt) != 1)) return;
          break;
        case ctUnion:
          if (IsHotEdge(*edge_o) != ((Abs(edge_c->wind_cnt) != 1) ||
            (IsHotEdge(*edge_o) != (edge_c->wind_cnt != 0)))) return; //just works!
          break;
        case ctXor:
          if (Abs(edge_c->wind_cnt) != 1) return;
          break;
      }
      //toggle contribution ...
      if (IsHotEdge(*edge_o)) {
        AddOutPt(*edge_o, pt);
        TerminateHotOpen(*edge_o);
      }
      else StartOpenPath(*edge_o, pt);
      return;
    }

    //update winding counts...
    //assumes that e1 will be to the right of e2 ABOVE the intersection
    int old_e1_windcnt, old_e2_windcnt;
    if (e1.local_min->polytype == e2.local_min->polytype) {
      if (fillrule_ == frEvenOdd) {
        old_e1_windcnt = e1.wind_cnt;
        e1.wind_cnt = e2.wind_cnt;
        e2.wind_cnt = old_e1_windcnt;
      }
      else {
        if (e1.wind_cnt + e2.wind_dx == 0) e1.wind_cnt = -e1.wind_cnt;
        else e1.wind_cnt += e2.wind_dx;
        if (e2.wind_cnt - e1.wind_dx == 0) e2.wind_cnt = -e2.wind_cnt;
        else e2.wind_cnt -= e1.wind_dx;
      }
    }
    else {
      if (fillrule_ != frEvenOdd) e1.wind_cnt2 += e2.wind_dx;
      else e1.wind_cnt2 = (e1.wind_cnt2 == 0 ? 1 : 0);
      if (fillrule_ != frEvenOdd) e2.wind_cnt2 -= e1.wind_dx;
      else e2.wind_cnt2 = (e2.wind_cnt2 == 0 ? 1 : 0);
    }

    switch (fillrule_) {
      case frPositive:
        old_e1_windcnt = e1.wind_cnt;
        old_e2_windcnt = e2.wind_cnt;
        break;
      case frNegative:
        old_e1_windcnt = -e1.wind_cnt;
        old_e2_windcnt = -e2.wind_cnt;
        break;
      default:
        old_e1_windcnt = Abs(e1.wind_cnt);
        old_e2_windcnt = Abs(e2.wind_cnt);
        break;
    }

    if (IsHotEdge(e1) && IsHotEdge(e2)) {
      if ((old_e1_windcnt != 0 && old_e1_windcnt != 1) || (old_e2_windcnt != 0 && old_e2_windcnt != 1) ||
        (e1.local_min->polytype != e2.local_min->polytype && cliptype_ != ctXor))
      {
        AddLocalMaxPoly(e1, e2, pt);
      }
      else if (IsStartSide(e1)) {
        AddLocalMaxPoly(e1, e2, pt);
        AddLocalMinPoly(e1, e2, pt);
      }
      else {
        AddOutPt(e1, pt);
        AddOutPt(e2, pt);
        SwapOutrecs(e1, e2);
      }
    }
    else if (IsHotEdge(e1)) {
      if (old_e2_windcnt == 0 || old_e2_windcnt == 1) {
        AddOutPt(e1, pt);
        SwapOutrecs(e1, e2);
      }
    }
    else if (IsHotEdge(e2)) {
      if (old_e1_windcnt == 0 || old_e1_windcnt == 1) {
        AddOutPt(e2, pt);
        SwapOutrecs(e1, e2);
      }
    }
    else if ((old_e1_windcnt == 0 || old_e1_windcnt == 1) &&
      (old_e2_windcnt == 0 || old_e2_windcnt == 1))
    {
      //neither edge is currently contributing ...
      int64_t e1Wc2, e2Wc2;
      switch (fillrule_) {
        case frPositive:
          e1Wc2 = e1.wind_cnt2;
          e2Wc2 = e2.wind_cnt2;
          break;
        case frNegative:
          e1Wc2 = -e1.wind_cnt2;
          e2Wc2 = -e2.wind_cnt2;
          break;
        default:
          e1Wc2 = Abs(e1.wind_cnt2);
          e2Wc2 = Abs(e2.wind_cnt2);
          break;
      }

      if (!IsSamePolyType(e1, e2)) {
        AddLocalMinPoly(e1, e2, pt);
      }
      else if (old_e1_windcnt == 1 && old_e2_windcnt == 1)
        switch (cliptype_) {
          case ctIntersection:
            if (e1Wc2 > 0 && e2Wc2 > 0)
              AddLocalMinPoly(e1, e2, pt);
            break;
          case ctUnion:
            if (e1Wc2 <= 0 && e2Wc2 <= 0)
              AddLocalMinPoly(e1, e2, pt);
            break;
          case ctDifference:
            if (((GetPolyType(e1) == ptClip) && (e1Wc2 > 0) && (e2Wc2 > 0)) ||
              ((GetPolyType(e1) == ptSubject) && (e1Wc2 <= 0) && (e2Wc2 <= 0)))
              AddLocalMinPoly(e1, e2, pt);
            break;
          case ctXor:
            AddLocalMinPoly(e1, e2, pt);
            break;
        }
    }
  }
  //------------------------------------------------------------------------------

  inline void Clipper::DeleteFromAEL(Active &e)
  {
    Active* prev = e.prev_in_ael;
    Active* next = e.next_in_ael;
    if (!prev && !next && (&e != actives_)) return; //already deleted
    if (prev) prev->next_in_ael = next;
    else actives_ = next;
    if (next) next->prev_in_ael = prev;
    delete &e;
  }
  //------------------------------------------------------------------------------

  inline void Clipper::CopyAELToSEL()
  {
    Active* e = actives_;
    sel_ = e;
    while (e) {
      e->prev_in_sel = e->prev_in_ael;
      e->next_in_sel = e->next_in_ael;
      e = e->next_in_ael;
    }
  }
  //------------------------------------------------------------------------------

  inline void Clipper::CopyActivesToSELAdjustCurrX(const int64_t top_y)
  {
    Active* e = actives_;
    sel_ = e;
    while (e) {
      e->prev_in_sel = e->prev_in_ael;
      e->next_in_sel = e->next_in_ael;
      e->curr.x = TopX(*e, top_y);
      e = e->next_in_ael;
    }
  }
  //------------------------------------------------------------------------------

  bool Clipper::ExecuteInternal(ClipType ct, FillRule ft)
  {
    if (ct == ctNone) return true;
    fillrule_ = ft;
    cliptype_ = ct;
    Reset();

    int64_t y;
    if (!PopScanline(y)) { return false; }
    for (;;) {
      InsertLocalMinimaIntoAEL(y);
      Active *e;
      while (PopHorz(e)) ProcessHorizontal(*e);
      if (!PopScanline(y)) break;   //Y is now at the top of the scanbeam
      ProcessIntersections(y);
      DoTopOfScanbeam(y);
    }
    return true;
  }
  //------------------------------------------------------------------------------

  bool Clipper::Execute(ClipType clipType, Paths &solution_closed, FillRule ft)
  {
    solution_closed.clear();
    if (!ExecuteInternal(clipType, ft)) return false;
    BuildResult(solution_closed, NULL);
    CleanUp();
    return true;
  }
  //------------------------------------------------------------------------------

  bool Clipper::Execute(ClipType clipType, Paths &solution_closed, Paths &solution_open, FillRule ft)
  {
    solution_closed.clear();
    solution_open.clear();
    if (!ExecuteInternal(clipType, ft)) return false;
    BuildResult(solution_closed, &solution_open);
    CleanUp();
    return true;
  }
  //------------------------------------------------------------------------------

  bool Clipper::Execute(ClipType clipType, PolyPath &solution_closed, Paths &solution_open, FillRule ft)
  {
    solution_closed.Clear();
    if (!ExecuteInternal(clipType, ft)) return false;
    BuildResult2(solution_closed, NULL);
    CleanUp();
    return true;
  }
  //------------------------------------------------------------------------------

  void Clipper::ProcessIntersections(const int64_t top_y)
  {
    BuildIntersectList(top_y);
    if (intersect_list_.size() == 0) return;
    FixupIntersectionOrder();
    ProcessIntersectList();
  }
  //------------------------------------------------------------------------------

  inline void Clipper::DisposeIntersectNodes()
  {
    for (IntersectList::iterator node_iter = intersect_list_.begin();
      node_iter != intersect_list_.end(); ++node_iter)
        delete (*node_iter);
    intersect_list_.resize(0);
  }
  //------------------------------------------------------------------------------

  void Clipper::InsertNewIntersectNode(Active &e1, Active &e2, int64_t top_y)
  {
    Point64 pt = GetIntersectPoint(e1, e2);

    //Rounding errors can occasionally place the calculated intersection
    //point either below or above the scanbeam, so check and correct ...
    if (pt.y > e1.curr.y) {
      pt.y = e1.curr.y;      //e.curr.y is still the bottom of scanbeam
                             //use the more vertical of the 2 edges to derive pt.X ...
      if (Abs(e1.dx) < Abs(e2.dx)) pt.x = TopX(e1, pt.y);
      else pt.x = TopX(e2, pt.y);
    }
    else if (pt.y < top_y) {
      pt.y = top_y;          //top_y is at the top of the scanbeam

      if (e1.top.y == top_y) pt.x = e1.top.x;
      else if (e2.top.y == top_y) pt.x = e2.top.x;
      else if (Abs(e1.dx) < Abs(e2.dx)) pt.x = e1.curr.x;
      else pt.x = e2.curr.x;
    }

    IntersectNode *node = new IntersectNode();
    node->edge1 = &e1;
    node->edge2 = &e2;
    node->pt = pt;
    intersect_list_.push_back(node);
  }
  //------------------------------------------------------------------------------

  void Clipper::BuildIntersectList(const int64_t top_y)
  {
    if (!actives_ || !actives_->next_in_ael) return;
    CopyActivesToSELAdjustCurrX(top_y);

    //Merge sort actives_ into their new positions at the top of scanbeam, and
    //create an intersection node every time an edge crosses over another ...
    //see also https://stackoverflow.com/a/46319131/359538
    int mul = 1;
    while (true) {

      Active *first = sel_, *second = NULL, *baseE, *prev_base = NULL, *tmp;
      //sort successive larger 'mul' count of nodes ...
      while (first) {
        if (mul == 1) {
          second = first->next_in_sel;
          if (!second) {
            first->merge_jump = NULL;
            break;
          }
          first->merge_jump = second->next_in_sel;
        }
        else {
          second = first->merge_jump;
          if (!second) {
            first->merge_jump = NULL;
            break;
          }
          first->merge_jump = second->merge_jump;
        }

        //now sort first and second groups ...
        baseE = first;
        int lCnt = mul, rCnt = mul;
        while (lCnt > 0 && rCnt > 0) {
          if (second->curr.x < first->curr.x) {
            // create one or more Intersect nodes ///////////
            tmp = second->prev_in_sel;
            for (int i = 0; i < lCnt; ++i) {
              //create a new intersect node...
              InsertNewIntersectNode(*tmp, *second, top_y);
              tmp = tmp->prev_in_sel;
            }
            /////////////////////////////////////////////////

            if (first == baseE) {
              if (prev_base) prev_base->merge_jump = second;
              baseE = second;
              baseE->merge_jump = first->merge_jump;
              if (!first->prev_in_sel) sel_ = second;
            }
            tmp = second->next_in_sel;
            //now move the out of place edge to it's new position in SEL ...
            Insert2Before1InSel(*first, *second);
            second = tmp;
            if (!second) break;
            --rCnt;
          }
          else {
            first = first->next_in_sel;
            --lCnt;
          }
        }
        first = baseE->merge_jump;
        prev_base = baseE;
      }
      if (!sel_->merge_jump) break;
      else mul <<= 1;
    }
  }
  //------------------------------------------------------------------------------

  bool Clipper::ProcessIntersectList()
  {
    for (IntersectList::iterator node_iter = intersect_list_.begin();
      node_iter != intersect_list_.end(); ++node_iter) {
      IntersectNode *iNode = *node_iter;
      IntersectEdges(*iNode->edge1, *iNode->edge2, iNode->pt);
      SwapPositionsInAEL(*iNode->edge1, *iNode->edge2);
    }
    DisposeIntersectNodes();
    return true;
  }
  //------------------------------------------------------------------------------

  void Clipper::FixupIntersectionOrder()
  {
    size_t cnt = intersect_list_.size();
    if (cnt < 2) return;
    //It's important that edge intersections are processed from the bottom up,
    //but it's also crucial that intersections only occur between adjacent edges.
    //The first sort here (a quicksort), arranges intersections relative to their
    //vertical positions within the scanbeam ...
    std::sort(intersect_list_.begin(), intersect_list_.end(), IntersectListSort);

    //Now we simulate processing these intersections, and as we do, we make sure
    //that the intersecting edges remain adjacent. If they aren't, this simulated
    //intersection is delayed until such time as these edges do become adjacent.
    CopyAELToSEL();
    for (size_t i = 0; i < cnt; ++i) {
      if (!EdgesAdjacentInSel(*intersect_list_[i])) {
        size_t  j = i + 1;
        while (!EdgesAdjacentInSel(*intersect_list_[j])) j++;
        std::swap(intersect_list_[i], intersect_list_[j]);
      }
      SwapPositionsInSEL(*intersect_list_[i]->edge1, *intersect_list_[i]->edge2);
    }
  }
  //------------------------------------------------------------------------------

  void Clipper::SwapPositionsInAEL(Active &e1, Active &e2)
  {
    //check that one or other edge hasn't already been removed from AEL ...
    if (e1.next_in_ael == e1.prev_in_ael ||
      e2.next_in_ael == e2.prev_in_ael) return;

    Active *next, *prev;
    if (e1.next_in_ael == &e2) {
      next = e2.next_in_ael;
      if (next) next->prev_in_ael = &e1;
      prev = e1.prev_in_ael;
      if (prev) prev->next_in_ael = &e2;
      e2.prev_in_ael = prev;
      e2.next_in_ael = &e1;
      e1.prev_in_ael = &e2;
      e1.next_in_ael = next;
    }
    else if (e2.next_in_ael == &e1) {
      next = e1.next_in_ael;
      if (next) next->prev_in_ael = &e2;
      prev = e2.prev_in_ael;
      if (prev) prev->next_in_ael = &e1;
      e1.prev_in_ael = prev;
      e1.next_in_ael = &e2;
      e2.prev_in_ael = &e1;
      e2.next_in_ael = next;
    }
    else {
      next = e1.next_in_ael;
      prev = e1.prev_in_ael;
      e1.next_in_ael = e2.next_in_ael;
      if (e1.next_in_ael) e1.next_in_ael->prev_in_ael = &e1;
      e1.prev_in_ael = e2.prev_in_ael;
      if (e1.prev_in_ael) e1.prev_in_ael->next_in_ael = &e1;
      e2.next_in_ael = next;
      if (e2.next_in_ael) e2.next_in_ael->prev_in_ael = &e2;
      e2.prev_in_ael = prev;
      if (e2.prev_in_ael) e2.prev_in_ael->next_in_ael = &e2;
    }

    if (!e1.prev_in_ael) actives_ = &e1;
    else if (!e2.prev_in_ael) actives_ = &e2;
  }
  //------------------------------------------------------------------------------

  void Clipper::SwapPositionsInSEL(Active &e1, Active &e2)
  {
    if (!e1.next_in_sel && !e1.prev_in_sel) return;
    if (!e2.next_in_sel && !e2.prev_in_sel) return;

    if (e1.next_in_sel == &e2) {
      Active* Next = e2.next_in_sel;
      if (Next) Next->prev_in_sel = &e1;
      Active* Prev = e1.prev_in_sel;
      if (Prev) Prev->next_in_sel = &e2;
      e2.prev_in_sel = Prev;
      e2.next_in_sel = &e1;
      e1.prev_in_sel = &e2;
      e1.next_in_sel = Next;
    }
    else if (e2.next_in_sel == &e1) {
      Active* Next = e1.next_in_sel;
      if (Next) Next->prev_in_sel = &e2;
      Active* Prev = e2.prev_in_sel;
      if (Prev) Prev->next_in_sel = &e1;
      e1.prev_in_sel = Prev;
      e1.next_in_sel = &e2;
      e2.prev_in_sel = &e1;
      e2.next_in_sel = Next;
    }
    else {
      Active* Next = e1.next_in_sel;
      Active* Prev = e1.prev_in_sel;
      e1.next_in_sel = e2.next_in_sel;
      if (e1.next_in_sel) e1.next_in_sel->prev_in_sel = &e1;
      e1.prev_in_sel = e2.prev_in_sel;
      if (e1.prev_in_sel) e1.prev_in_sel->next_in_sel = &e1;
      e2.next_in_sel = Next;
      if (e2.next_in_sel) e2.next_in_sel->prev_in_sel = &e2;
      e2.prev_in_sel = Prev;
      if (e2.prev_in_sel) e2.prev_in_sel->next_in_sel = &e2;
    }

    if (!e1.prev_in_sel) sel_ = &e1;
    else if (!e2.prev_in_sel) sel_ = &e2;
  }
  //------------------------------------------------------------------------------

  void Clipper::Insert2Before1InSel(Active &first, Active &second)
  {
    //remove second from list ...
    Active *prev = second.prev_in_sel;
    Active *next = second.next_in_sel;
    prev->next_in_sel = next; //always a prev since we're moving from right to left
    if (next) next->prev_in_sel = prev;
    //insert back into list ...
    prev = first.prev_in_sel;
    if (prev) prev->next_in_sel = &second;
    first.prev_in_sel = &second;
    second.prev_in_sel = prev;
    second.next_in_sel = &first;
  }
  //------------------------------------------------------------------------------

  bool Clipper::ResetHorzDirection(Active &horz, Active *max_pair, int64_t &horz_left, int64_t &horz_right)
  {
    if (horz.bot.x == horz.top.x) {
      //the horizontal edge is going nowhere ...
      horz_left = horz.curr.x;
      horz_right = horz.curr.x;
      Active *e = horz.next_in_ael;
      while (e && e != max_pair) e = e->next_in_ael;
      return e != NULL;
    }
    else if (horz.curr.x < horz.top.x) {
      horz_left = horz.curr.x;
      horz_right = horz.top.x;
      return true;
    }
    else {
      horz_left = horz.top.x;
      horz_right = horz.curr.x;
      return false; //right to left
    }
  }
  //------------------------------------------------------------------------------

  void Clipper::ProcessHorizontal(Active &horz)
    /*******************************************************************************
    * Notes: Horizontal edges (HEs) at scanline intersections (ie at the top or    *
    * bottom of a scanbeam) are processed as if layered.The order in which HEs     *
    * are processed doesn't matter. HEs intersect with the bottom vertices of      *
    * other HEs[#] and with non-horizontal edges [*]. Once these intersections     *
    * are completed, intermediate HEs are 'promoted' to the next edge in their     *
    * bounds, and they in turn may be intersected[%] by other HEs.                 *
    *                                                                              *
    * eg: 3 horizontals at a scanline:    /   |                     /           /  *
    *              |                     /    |     (HE3)o ========%========== o   *
    *              o ======= o(HE2)     /     |         /         /                *
    *          o ============#=========*======*========#=========o (HE1)           *
    *         /              |        /       |       /                            *
    *******************************************************************************/
  {
    Point64 pt;
    //with closed paths, simplify consecutive horizontals into a 'single' edge ...
    if (!IsOpen(horz)) {
      pt = horz.bot;
      while (!IsMaxima(horz) && NextVertex(horz).pt.y == pt.y)
        UpdateEdgeIntoAEL(&horz);
      horz.bot = pt;
      horz.curr = pt;
    };

    Active *max_pair = NULL;
    if (IsMaxima(horz) && (!IsOpen(horz) ||
      ((horz.vertex_top->flags & (vfOpenStart | vfOpenEnd)) == 0)))
        max_pair = GetMaximaPair(horz);

    int64_t horz_left, horz_right;
    bool is_left_to_right = ResetHorzDirection(horz, max_pair, horz_left, horz_right);
    if (IsHotEdge(horz)) AddOutPt(horz, horz.curr);

    while (true) { //loops through consec. horizontal edges (if open)
      Active *e;
      bool isMax = IsMaxima(horz);
      if (is_left_to_right)
        e = horz.next_in_ael; else
        e = horz.prev_in_ael;
      while (e) {
        //break if we've gone past the } of the horizontal ...
        if ((is_left_to_right && (e->curr.x > horz_right)) ||
          (!is_left_to_right && (e->curr.x < horz_left))) break;
        //or if we've got to the } of an intermediate horizontal edge ...
        if (e->curr.x == horz.top.x && !isMax && !IsHorizontal(*e)) {
          pt = NextVertex(horz).pt;
          if (is_left_to_right && (TopX(*e, pt.y) >= pt.x) ||
            (!is_left_to_right && (TopX(*e, pt.y) <= pt.x))) break;
        };

        if (e == max_pair) {
          if (IsHotEdge(horz)) {
            if (is_left_to_right)
              AddLocalMaxPoly(horz, *e, horz.top);
            else
              AddLocalMaxPoly(*e, horz, horz.top);
          }
          DeleteFromAEL(*e);
          DeleteFromAEL(horz);
          return;
        };

        if (is_left_to_right) {
          pt = Point64(e->curr.x, horz.curr.y);
          IntersectEdges(horz, *e, pt);
        }
        else {
          pt = Point64(e->curr.x, horz.curr.y);
          IntersectEdges(*e, horz, pt);
        }

        Active *next_e;
        if (is_left_to_right) next_e = e->next_in_ael;
        else next_e = e->prev_in_ael;
        SwapPositionsInAEL(horz, *e);
        e = next_e;
      }

      //check if we've finished with (consecutive) horizontals ...
      if (isMax || NextVertex(horz).pt.y != horz.top.y) break;

      //still more horizontals in bound to process ...
      UpdateEdgeIntoAEL(&horz);
      is_left_to_right = ResetHorzDirection(horz, max_pair, horz_left, horz_right);

      if (IsOpen(horz)) {
        if (IsMaxima(horz)) max_pair = GetMaximaPair(horz);
        if (IsHotEdge(horz)) AddOutPt(horz, horz.bot);
      }
    }

    if (IsHotEdge(horz)) AddOutPt(horz, horz.top);
    if (!IsOpen(horz))
      UpdateEdgeIntoAEL(&horz); //this is the } of an intermediate horiz.
    else if (!IsMaxima(horz))
      UpdateEdgeIntoAEL(&horz);
    else if (!max_pair)      //ie open at top
      DeleteFromAEL(horz);
    else if (IsHotEdge(horz))
      AddLocalMaxPoly(horz, *max_pair, horz.top);
    else {
      DeleteFromAEL(*max_pair); DeleteFromAEL(horz);
    }
  }
  //------------------------------------------------------------------------------

  void Clipper::DoTopOfScanbeam(const int64_t y)
  {
    sel_ = NULL;            //reused to flag horizontals
    Active *e = actives_;
    while (e) {
      //nb: E will never be horizontal at this point
      if (e->top.y == y) {
        e->curr = e->top;   //needed for horizontal processing
        if (IsMaxima(*e)) {
          e = DoMaxima(*e); //TOP OF BOUND (MAXIMA)
          continue;
        }
        else {
          //INTERMEDIATE VERTEX ...
          UpdateEdgeIntoAEL(e);
          if (IsHotEdge(*e))AddOutPt(*e, e->bot);
          if (IsHorizontal(*e))
            PushHorz(*e); //horizontals are processed later
        }
      }
      else {
        e->curr.y = y;
        e->curr.x = TopX(*e, y);
      }
      e = e->next_in_ael;
    }
  }
  //------------------------------------------------------------------------------

  Active* Clipper::DoMaxima(Active &e)
  {
    Active *next_e, *prev_e, *max_pair;
    prev_e = e.prev_in_ael;
    next_e = e.next_in_ael;
    if (IsOpen(e) && ((e.vertex_top->flags & (vfOpenStart | vfOpenEnd)) != 0)) {
      if (IsHotEdge(e)) AddOutPt(e, e.top);
      if (!IsHorizontal(e)) {
        if (IsHotEdge(e)) TerminateHotOpen(e);
        DeleteFromAEL(e);
      }
      return next_e;
    }
    else {
      max_pair = GetMaximaPair(e);
      if (!max_pair) return next_e; //eMaxPair is horizontal
    }

    //only non-horizontal maxima here.
    //process any edges between maxima pair ...
    while (next_e != max_pair) {
      IntersectEdges(e, *next_e, e.top);
      SwapPositionsInAEL(e, *next_e);
      next_e = e.next_in_ael;
    }

    if (IsOpen(e)) {
      if (IsHotEdge(e)) {
        if (max_pair)
          AddLocalMaxPoly(e, *max_pair, e.top); else
          AddOutPt(e, e.top);
      }
      if (max_pair) DeleteFromAEL(*max_pair);
      DeleteFromAEL(e);
      return (prev_e ? prev_e->next_in_ael : actives_);
    }
    //here E.next_in_ael == ENext == EMaxPair ...
    if (IsHotEdge(e))
      AddLocalMaxPoly(e, *max_pair, e.top);

    DeleteFromAEL(e);
    DeleteFromAEL(*max_pair);
    return (prev_e ? prev_e->next_in_ael : actives_);
  }
  //------------------------------------------------------------------------------

  void Clipper::BuildResult(Paths &solution_closed, Paths *solution_open)
  {
    solution_closed.resize(0);
    solution_closed.reserve(outrec_list_.size());
    if (solution_open) {
      solution_open->resize(0);
      solution_open->reserve(outrec_list_.size());
    }

    for (OutRecList::const_iterator ol_iter = outrec_list_.begin();
      ol_iter != outrec_list_.end(); ++ol_iter)
    {
      OutRec *outrec = *ol_iter;
      if (!outrec->pts) continue;
      OutPt *op = outrec->pts->next;
      int cnt = PointCount(op);
      //fixup for duplicate start and } points ...
      if (op->pt == outrec->pts->pt) cnt--;

      bool is_open = (outrec->flag == orOpen);
      if (cnt < 2 || (!is_open && cnt == 2) || (is_open && !solution_open)) continue;
      Path p;
      p.reserve(cnt);
      for (int i = 0; i < cnt; i++) { p.push_back(op->pt); op = op->next; }
      if (is_open) solution_open->push_back(p);
      else solution_closed.push_back(p);
    }
  }
  //------------------------------------------------------------------------------

  void Clipper::BuildResult2(PolyPath &pt, Paths *solution_open)
  {
    pt.Clear();
    if (solution_open) {
      solution_open->resize(0);
      solution_open->reserve(outrec_list_.size());
    }

    for (OutRecList::const_iterator ol_iter = outrec_list_.begin();
      ol_iter != outrec_list_.end(); ++ol_iter)
    {
      OutRec *outrec = *ol_iter;
      if (!outrec->pts) continue;
      OutPt *op = outrec->pts->next;
      int cnt = PointCount(op);
      //fixup for duplicate start and } points ...
      if (op->pt == outrec->pts->pt) cnt--;

      bool is_open = (outrec->flag == orOpen);
      if (cnt < 2 || (!is_open && cnt == 2) || (is_open && !solution_open)) continue;

      Path p;
      p.reserve(cnt);
      for (int i = 0; i < cnt; i++) { p.push_back(op->pt); op = op->next; }
      if (is_open)
        solution_open->push_back(p);
      else if (outrec->owner && outrec->owner->polypath)
        outrec->polypath = &outrec->owner->polypath->AddChild(p);
      else
        outrec->polypath = &pt.AddChild(p);
    }
  }
  //------------------------------------------------------------------------------

  Rect64 Clipper::GetBounds()
  {
    if (vertex_list_.size() == 0) return Rect64(0, 0, 0, 0);
    Rect64 result = Rect64(INT64_MAX, INT64_MAX, INT64_MIN, INT64_MIN);
    VerticesList::const_iterator it = vertex_list_.begin();
    while (it != vertex_list_.end()) {
      Vertex  *v = *it, *v2 = v;
    do {
        if (v2->pt.x < result.left) result.left = v2->pt.x;
        if (v2->pt.x > result.right) result.right = v2->pt.x;
        if (v2->pt.y < result.top) result.top = v2->pt.y;
        if (v2->pt.y > result.bottom) result.bottom = v2->pt.y;
        v2 = v2->next;
      } while (v2 != v);
      ++it;
    }
    return result;
  }
  //------------------------------------------------------------------------------

  std::ostream& operator <<(std::ostream &s, const Point64 &pt)
  {
    s << pt.x << "," << pt.y << " ";
    return s;
  }
  //------------------------------------------------------------------------------

  std::ostream& operator <<(std::ostream &s, const Path &path)
  {
    if (path.empty()) return s;
    Path::size_type last = path.size() -1;
    for (Path::size_type i = 0; i < last; i++) s  << path[i] << " ";
    s << path[last] << "\n";
    return s;
  }
  //------------------------------------------------------------------------------

  std::ostream& operator <<(std::ostream &s, const Paths &paths)
  {
    for (Paths::size_type i = 0; i < paths.size(); i++) s << paths[i];
    s << "\n";
    return s;
  }
  //------------------------------------------------------------------------------

} //clipperlib namespace

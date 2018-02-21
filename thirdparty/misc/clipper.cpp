/*******************************************************************************
*                                                                              *
* Author    :  Angus Johnson                                                   *
* Version   :  6.4.2                                                           *
* Date      :  27 February 2017                                                *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2017                                         *
*                                                                              *
* License:                                                                     *
* Use, modification & distribution is subject to Boost Software License Ver 1. *
* http://www.boost.org/LICENSE_1_0.txt                                         *
*                                                                              *
* Attributions:                                                                *
* The code in this library is an extension of Bala Vatti's clipping algorithm: *
* "A generic solution to polygon clipping"                                     *
* Communications of the ACM, Vol 35, Issue 7 (July 1992) pp 56-63.             *
* http://portal.acm.org/citation.cfm?id=129906                                 *
*                                                                              *
* Computer graphics and geometric modeling: implementation and algorithms      *
* By Max K. Agoston                                                            *
* Springer; 1 edition (January 4, 2005)                                        *
* http://books.google.com/books?q=vatti+clipping+agoston                       *
*                                                                              *
* See also:                                                                    *
* "Polygon Offsetting by Computing Winding Numbers"                            *
* Paper no. DETC2005-85513 pp. 565-575                                         *
* ASME 2005 International Design Engineering Technical Conferences             *
* and Computers and Information in Engineering Conference (IDETC/CIE2005)      *
* September 24-28, 2005 , Long Beach, California, USA                          *
* http://www.me.berkeley.edu/~mcmains/pubs/DAC05OffsetPolygon.pdf              *
*                                                                              *
*******************************************************************************/

/*******************************************************************************
*                                                                              *
* This is a translation of the Delphi Clipper library and the naming style     *
* used has retained a Delphi flavour.                                          *
*                                                                              *
*******************************************************************************/

#include "clipper.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <ostream>
#include <functional>

namespace ClipperLib {

static double const pi = 3.141592653589793238;
static double const two_pi = pi *2;
static double const def_arc_tolerance = 0.25;

enum Direction { dRightToLeft, dLeftToRight };

static int const Unassigned = -1;  //edge not currently 'owning' a solution
static int const Skip = -2;        //edge that would otherwise close a path

#define HORIZONTAL (-1.0E+40)
#define TOLERANCE (1.0e-20)
#define NEAR_ZERO(val) (((val) > -TOLERANCE) && ((val) < TOLERANCE))

struct TEdge {
  IntPoint Bot;
  IntPoint Curr; //current (updated for every new scanbeam)
  IntPoint Top;
  double Dx;
  PolyType PolyTyp;
  EdgeSide Side; //side only refers to current side of solution poly
  int WindDelta; //1 or -1 depending on winding direction
  int WindCnt;
  int WindCnt2; //winding count of the opposite polytype
  int OutIdx;
  TEdge *Next;
  TEdge *Prev;
  TEdge *NextInLML;
  TEdge *NextInAEL;
  TEdge *PrevInAEL;
  TEdge *NextInSEL;
  TEdge *PrevInSEL;
};

struct IntersectNode {
  TEdge          *Edge1;
  TEdge          *Edge2;
  IntPoint        Pt;
};

struct LocalMinimum {
  cInt          Y;
  TEdge        *LeftBound;
  TEdge        *RightBound;
};

struct OutPt;

//OutRec: contains a path in the clipping solution. Edges in the AEL will
//carry a pointer to an OutRec when they are part of the clipping solution.
struct OutRec {
  int       Idx;
  bool      IsHole;
  bool      IsOpen;
  OutRec   *FirstLeft;  //see comments in clipper.pas
  PolyNode *PolyNd;
  OutPt    *Pts;
  OutPt    *BottomPt;
};

struct OutPt {
  int       Idx;
  IntPoint  Pt;
  OutPt    *Next;
  OutPt    *Prev;
};

struct Join {
  OutPt    *OutPt1;
  OutPt    *OutPt2;
  IntPoint  OffPt;
};

struct LocMinSorter
{
  inline bool operator()(const LocalMinimum& locMin1, const LocalMinimum& locMin2)
  {
    return locMin2.Y < locMin1.Y;
  }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

inline cInt Round(double val)
{
  if ((val < 0)) return static_cast<cInt>(val - 0.5); 
  else return static_cast<cInt>(val + 0.5);
}
//------------------------------------------------------------------------------

inline cInt Abs(cInt val)
{
  return val < 0 ? -val : val;
}

//------------------------------------------------------------------------------
// PolyTree methods ...
//------------------------------------------------------------------------------

void PolyTree::Clear()
{
    for (PolyNodes::size_type i = 0; i < AllNodes.size(); ++i)
      delete AllNodes[i];
    AllNodes.resize(0); 
    Childs.resize(0);
}
//------------------------------------------------------------------------------

PolyNode* PolyTree::GetFirst() const
{
  if (!Childs.empty())
      return Childs[0];
  else
      return 0;
}
//------------------------------------------------------------------------------

int PolyTree::Total() const
{
  int result = (int)AllNodes.size();
  //with negative offsets, ignore the hidden outer polygon ...
  if (result > 0 && Childs[0] != AllNodes[0]) result--;
  return result;
}

//------------------------------------------------------------------------------
// PolyNode methods ...
//------------------------------------------------------------------------------

PolyNode::PolyNode(): Parent(0), Index(0), m_IsOpen(false)
{
}
//------------------------------------------------------------------------------

int PolyNode::ChildCount() const
{
  return (int)Childs.size();
}
//------------------------------------------------------------------------------

void PolyNode::AddChild(PolyNode& child)
{
  unsigned cnt = (unsigned)Childs.size();
  Childs.push_back(&child);
  child.Parent = this;
  child.Index = cnt;
}
//------------------------------------------------------------------------------

PolyNode* PolyNode::GetNext() const
{ 
  if (!Childs.empty()) 
      return Childs[0]; 
  else
      return GetNextSiblingUp();    
}  
//------------------------------------------------------------------------------

PolyNode* PolyNode::GetNextSiblingUp() const
{ 
  if (!Parent) //protects against PolyTree.GetNextSiblingUp()
      return 0;
  else if (Index == Parent->Childs.size() - 1)
      return Parent->GetNextSiblingUp();
  else
      return Parent->Childs[Index + 1];
}  
//------------------------------------------------------------------------------

bool PolyNode::IsHole() const
{ 
  bool result = true;
  PolyNode* node = Parent;
  while (node)
  {
      result = !result;
      node = node->Parent;
  }
  return result;
}  
//------------------------------------------------------------------------------

bool PolyNode::IsOpen() const
{ 
  return m_IsOpen;
}  
//------------------------------------------------------------------------------

#ifndef use_int32

//------------------------------------------------------------------------------
// Int128 class (enables safe math on signed 64bit integers)
// eg Int128 val1((long64)9223372036854775807); //ie 2^63 -1
//    Int128 val2((long64)9223372036854775807);
//    Int128 val3 = val1 * val2;
//    val3.AsString => "85070591730234615847396907784232501249" (8.5e+37)
//------------------------------------------------------------------------------

class Int128
{
  public:
    ulong64 lo;
    long64 hi;

    Int128(long64 _lo = 0)
    {
      lo = (ulong64)_lo;   
      if (_lo < 0)  hi = -1; else hi = 0; 
    }


    Int128(const Int128 &val): lo(val.lo), hi(val.hi){}

    Int128(const long64& _hi, const ulong64& _lo): lo(_lo), hi(_hi){}
    
    Int128& operator = (const long64 &val)
    {
      lo = (ulong64)val;
      if (val < 0) hi = -1; else hi = 0;
      return *this;
    }

    bool operator == (const Int128 &val) const
      {return (hi == val.hi && lo == val.lo);}

    bool operator != (const Int128 &val) const
      { return !(*this == val);}

    bool operator > (const Int128 &val) const
    {
      if (hi != val.hi)
        return hi > val.hi;
      else
        return lo > val.lo;
    }

    bool operator < (const Int128 &val) const
    {
      if (hi != val.hi)
        return hi < val.hi;
      else
        return lo < val.lo;
    }

    bool operator >= (const Int128 &val) const
      { return !(*this < val);}

    bool operator <= (const Int128 &val) const
      { return !(*this > val);}

    Int128& operator += (const Int128 &rhs)
    {
      hi += rhs.hi;
      lo += rhs.lo;
      if (lo < rhs.lo) hi++;
      return *this;
    }

    Int128 operator + (const Int128 &rhs) const
    {
      Int128 result(*this);
      result+= rhs;
      return result;
    }

    Int128& operator -= (const Int128 &rhs)
    {
      *this += -rhs;
      return *this;
    }

    Int128 operator - (const Int128 &rhs) const
    {
      Int128 result(*this);
      result -= rhs;
      return result;
    }

    Int128 operator-() const //unary negation
    {
      if (lo == 0)
        return Int128(-hi, 0);
      else
        return Int128(~hi, ~lo + 1);
    }

    operator double() const
    {
      const double shift64 = 18446744073709551616.0; //2^64
      if (hi < 0)
      {
        if (lo == 0) return (double)hi * shift64;
        else return -(double)(~lo + ~hi * shift64);
      }
      else
        return (double)(lo + hi * shift64);
    }

};
//------------------------------------------------------------------------------

Int128 Int128Mul (long64 lhs, long64 rhs)
{
  bool negate = (lhs < 0) != (rhs < 0);

  if (lhs < 0) lhs = -lhs;
  ulong64 int1Hi = ulong64(lhs) >> 32;
  ulong64 int1Lo = ulong64(lhs & 0xFFFFFFFF);

  if (rhs < 0) rhs = -rhs;
  ulong64 int2Hi = ulong64(rhs) >> 32;
  ulong64 int2Lo = ulong64(rhs & 0xFFFFFFFF);

  //nb: see comments in clipper.pas
  ulong64 a = int1Hi * int2Hi;
  ulong64 b = int1Lo * int2Lo;
  ulong64 c = int1Hi * int2Lo + int1Lo * int2Hi;

  Int128 tmp;
  tmp.hi = long64(a + (c >> 32));
  tmp.lo = long64(c << 32);
  tmp.lo += long64(b);
  if (tmp.lo < b) tmp.hi++;
  if (negate) tmp = -tmp;
  return tmp;
};
#endif

//------------------------------------------------------------------------------
// Miscellaneous global functions
//------------------------------------------------------------------------------

bool Orientation(const Path &poly)
{
    return Area(poly) >= 0;
}
//------------------------------------------------------------------------------

double Area(const Path &poly)
{
  int size = (int)poly.size();
  if (size < 3) return 0;

  double a = 0;
  for (int i = 0, j = size -1; i < size; ++i)
  {
    a += ((double)poly[j].X + poly[i].X) * ((double)poly[j].Y - poly[i].Y);
    j = i;
  }
  return -a * 0.5;
}
//------------------------------------------------------------------------------

double Area(const OutPt *op)
{
  const OutPt *startOp = op;
  if (!op) return 0;
  double a = 0;
  do {
    a +=  (double)(op->Prev->Pt.X + op->Pt.X) * (double)(op->Prev->Pt.Y - op->Pt.Y);
    op = op->Next;
  } while (op != startOp);
  return a * 0.5;
}
//------------------------------------------------------------------------------

double Area(const OutRec &outRec)
{
  return Area(outRec.Pts);
}
//------------------------------------------------------------------------------

bool PointIsVertex(const IntPoint &Pt, OutPt *pp)
{
  OutPt *pp2 = pp;
  do
  {
    if (pp2->Pt == Pt) return true;
    pp2 = pp2->Next;
  }
  while (pp2 != pp);
  return false;
}
//------------------------------------------------------------------------------

//See "The Point in Polygon Problem for Arbitrary Polygons" by Hormann & Agathos
//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.88.5498&rep=rep1&type=pdf
int PointInPolygon(const IntPoint &pt, const Path &path)
{
  //returns 0 if false, +1 if true, -1 if pt ON polygon boundary
  int result = 0;
  size_t cnt = path.size();
  if (cnt < 3) return 0;
  IntPoint ip = path[0];
  for(size_t i = 1; i <= cnt; ++i)
  {
    IntPoint ipNext = (i == cnt ? path[0] : path[i]);
    if (ipNext.Y == pt.Y)
    {
        if ((ipNext.X == pt.X) || (ip.Y == pt.Y && 
          ((ipNext.X > pt.X) == (ip.X < pt.X)))) return -1;
    }
    if ((ip.Y < pt.Y) != (ipNext.Y < pt.Y))
    {
      if (ip.X >= pt.X)
      {
        if (ipNext.X > pt.X) result = 1 - result;
        else
        {
          double d = (double)(ip.X - pt.X) * (ipNext.Y - pt.Y) - 
            (double)(ipNext.X - pt.X) * (ip.Y - pt.Y);
          if (!d) return -1;
          if ((d > 0) == (ipNext.Y > ip.Y)) result = 1 - result;
        }
      } else
      {
        if (ipNext.X > pt.X)
        {
          double d = (double)(ip.X - pt.X) * (ipNext.Y - pt.Y) - 
            (double)(ipNext.X - pt.X) * (ip.Y - pt.Y);
          if (!d) return -1;
          if ((d > 0) == (ipNext.Y > ip.Y)) result = 1 - result;
        }
      }
    }
    ip = ipNext;
  } 
  return result;
}
//------------------------------------------------------------------------------

int PointInPolygon (const IntPoint &pt, OutPt *op)
{
  //returns 0 if false, +1 if true, -1 if pt ON polygon boundary
  int result = 0;
  OutPt* startOp = op;
  for(;;)
  {
    if (op->Next->Pt.Y == pt.Y)
    {
        if ((op->Next->Pt.X == pt.X) || (op->Pt.Y == pt.Y && 
          ((op->Next->Pt.X > pt.X) == (op->Pt.X < pt.X)))) return -1;
    }
    if ((op->Pt.Y < pt.Y) != (op->Next->Pt.Y < pt.Y))
    {
      if (op->Pt.X >= pt.X)
      {
        if (op->Next->Pt.X > pt.X) result = 1 - result;
        else
        {
          double d = (double)(op->Pt.X - pt.X) * (op->Next->Pt.Y - pt.Y) - 
            (double)(op->Next->Pt.X - pt.X) * (op->Pt.Y - pt.Y);
          if (!d) return -1;
          if ((d > 0) == (op->Next->Pt.Y > op->Pt.Y)) result = 1 - result;
        }
      } else
      {
        if (op->Next->Pt.X > pt.X)
        {
          double d = (double)(op->Pt.X - pt.X) * (op->Next->Pt.Y - pt.Y) - 
            (double)(op->Next->Pt.X - pt.X) * (op->Pt.Y - pt.Y);
          if (!d) return -1;
          if ((d > 0) == (op->Next->Pt.Y > op->Pt.Y)) result = 1 - result;
        }
      }
    } 
    op = op->Next;
    if (startOp == op) break;
  } 
  return result;
}
//------------------------------------------------------------------------------

bool Poly2ContainsPoly1(OutPt *OutPt1, OutPt *OutPt2)
{
  OutPt* op = OutPt1;
  do
  {
    //nb: PointInPolygon returns 0 if false, +1 if true, -1 if pt on polygon
    int res = PointInPolygon(op->Pt, OutPt2);
    if (res >= 0) return res > 0;
    op = op->Next; 
  }
  while (op != OutPt1);
  return true; 
}
//----------------------------------------------------------------------

bool SlopesEqual(const TEdge &e1, const TEdge &e2, bool UseFullInt64Range)
{
#ifndef use_int32
  if (UseFullInt64Range)
    return Int128Mul(e1.Top.Y - e1.Bot.Y, e2.Top.X - e2.Bot.X) == 
    Int128Mul(e1.Top.X - e1.Bot.X, e2.Top.Y - e2.Bot.Y);
  else 
#endif
    return (e1.Top.Y - e1.Bot.Y) * (e2.Top.X - e2.Bot.X) == 
    (e1.Top.X - e1.Bot.X) * (e2.Top.Y - e2.Bot.Y);
}
//------------------------------------------------------------------------------

bool SlopesEqual(const IntPoint pt1, const IntPoint pt2,
  const IntPoint pt3, bool UseFullInt64Range)
{
#ifndef use_int32
  if (UseFullInt64Range)
    return Int128Mul(pt1.Y-pt2.Y, pt2.X-pt3.X) == Int128Mul(pt1.X-pt2.X, pt2.Y-pt3.Y);
  else 
#endif
    return (pt1.Y-pt2.Y)*(pt2.X-pt3.X) == (pt1.X-pt2.X)*(pt2.Y-pt3.Y);
}
//------------------------------------------------------------------------------

bool SlopesEqual(const IntPoint pt1, const IntPoint pt2,
  const IntPoint pt3, const IntPoint pt4, bool UseFullInt64Range)
{
#ifndef use_int32
  if (UseFullInt64Range)
    return Int128Mul(pt1.Y-pt2.Y, pt3.X-pt4.X) == Int128Mul(pt1.X-pt2.X, pt3.Y-pt4.Y);
  else 
#endif
    return (pt1.Y-pt2.Y)*(pt3.X-pt4.X) == (pt1.X-pt2.X)*(pt3.Y-pt4.Y);
}
//------------------------------------------------------------------------------

inline bool IsHorizontal(TEdge &e)
{
  return e.Dx == HORIZONTAL;
}
//------------------------------------------------------------------------------

inline double GetDx(const IntPoint pt1, const IntPoint pt2)
{
  return (pt1.Y == pt2.Y) ?
    HORIZONTAL : (double)(pt2.X - pt1.X) / (pt2.Y - pt1.Y);
}
//---------------------------------------------------------------------------

inline void SetDx(TEdge &e)
{
  cInt dy  = (e.Top.Y - e.Bot.Y);
  if (dy == 0) e.Dx = HORIZONTAL;
  else e.Dx = (double)(e.Top.X - e.Bot.X) / dy;
}
//---------------------------------------------------------------------------

inline void SwapSides(TEdge &Edge1, TEdge &Edge2)
{
  EdgeSide Side =  Edge1.Side;
  Edge1.Side = Edge2.Side;
  Edge2.Side = Side;
}
//------------------------------------------------------------------------------

inline void SwapPolyIndexes(TEdge &Edge1, TEdge &Edge2)
{
  int OutIdx =  Edge1.OutIdx;
  Edge1.OutIdx = Edge2.OutIdx;
  Edge2.OutIdx = OutIdx;
}
//------------------------------------------------------------------------------

inline cInt TopX(TEdge &edge, const cInt currentY)
{
  return ( currentY == edge.Top.Y ) ?
    edge.Top.X : edge.Bot.X + Round(edge.Dx *(currentY - edge.Bot.Y));
}
//------------------------------------------------------------------------------

void IntersectPoint(TEdge &Edge1, TEdge &Edge2, IntPoint &ip)
{
#ifdef use_xyz  
  ip.Z = 0;
#endif

  double b1, b2;
  if (Edge1.Dx == Edge2.Dx)
  {
    ip.Y = Edge1.Curr.Y;
    ip.X = TopX(Edge1, ip.Y);
    return;
  }
  else if (Edge1.Dx == 0)
  {
    ip.X = Edge1.Bot.X;
    if (IsHorizontal(Edge2))
      ip.Y = Edge2.Bot.Y;
    else
    {
      b2 = Edge2.Bot.Y - (Edge2.Bot.X / Edge2.Dx);
      ip.Y = Round(ip.X / Edge2.Dx + b2);
    }
  }
  else if (Edge2.Dx == 0)
  {
    ip.X = Edge2.Bot.X;
    if (IsHorizontal(Edge1))
      ip.Y = Edge1.Bot.Y;
    else
    {
      b1 = Edge1.Bot.Y - (Edge1.Bot.X / Edge1.Dx);
      ip.Y = Round(ip.X / Edge1.Dx + b1);
    }
  } 
  else 
  {
    b1 = Edge1.Bot.X - Edge1.Bot.Y * Edge1.Dx;
    b2 = Edge2.Bot.X - Edge2.Bot.Y * Edge2.Dx;
    double q = (b2-b1) / (Edge1.Dx - Edge2.Dx);
    ip.Y = Round(q);
    if (std::fabs(Edge1.Dx) < std::fabs(Edge2.Dx))
      ip.X = Round(Edge1.Dx * q + b1);
    else 
      ip.X = Round(Edge2.Dx * q + b2);
  }

  if (ip.Y < Edge1.Top.Y || ip.Y < Edge2.Top.Y) 
  {
    if (Edge1.Top.Y > Edge2.Top.Y)
      ip.Y = Edge1.Top.Y;
    else
      ip.Y = Edge2.Top.Y;
    if (std::fabs(Edge1.Dx) < std::fabs(Edge2.Dx))
      ip.X = TopX(Edge1, ip.Y);
    else
      ip.X = TopX(Edge2, ip.Y);
  } 
  //finally, don't allow 'ip' to be BELOW curr.Y (ie bottom of scanbeam) ...
  if (ip.Y > Edge1.Curr.Y)
  {
    ip.Y = Edge1.Curr.Y;
    //use the more vertical edge to derive X ...
    if (std::fabs(Edge1.Dx) > std::fabs(Edge2.Dx))
      ip.X = TopX(Edge2, ip.Y); else
      ip.X = TopX(Edge1, ip.Y);
  }
}
//------------------------------------------------------------------------------

void ReversePolyPtLinks(OutPt *pp)
{
  if (!pp) return;
  OutPt *pp1, *pp2;
  pp1 = pp;
  do {
  pp2 = pp1->Next;
  pp1->Next = pp1->Prev;
  pp1->Prev = pp2;
  pp1 = pp2;
  } while( pp1 != pp );
}
//------------------------------------------------------------------------------

void DisposeOutPts(OutPt*& pp)
{
  if (pp == 0) return;
    pp->Prev->Next = 0;
  while( pp )
  {
    OutPt *tmpPp = pp;
    pp = pp->Next;
    delete tmpPp;
  }
}
//------------------------------------------------------------------------------

inline void InitEdge(TEdge* e, TEdge* eNext, TEdge* ePrev, const IntPoint& Pt)
{
  std::memset(e, 0, sizeof(TEdge));
  e->Next = eNext;
  e->Prev = ePrev;
  e->Curr = Pt;
  e->OutIdx = Unassigned;
}
//------------------------------------------------------------------------------

void InitEdge2(TEdge& e, PolyType Pt)
{
  if (e.Curr.Y >= e.Next->Curr.Y)
  {
    e.Bot = e.Curr;
    e.Top = e.Next->Curr;
  } else
  {
    e.Top = e.Curr;
    e.Bot = e.Next->Curr;
  }
  SetDx(e);
  e.PolyTyp = Pt;
}
//------------------------------------------------------------------------------

TEdge* RemoveEdge(TEdge* e)
{
  //removes e from double_linked_list (but without removing from memory)
  e->Prev->Next = e->Next;
  e->Next->Prev = e->Prev;
  TEdge* result = e->Next;
  e->Prev = 0; //flag as removed (see ClipperBase.Clear)
  return result;
}
//------------------------------------------------------------------------------

inline void ReverseHorizontal(TEdge &e)
{
  //swap horizontal edges' Top and Bottom x's so they follow the natural
  //progression of the bounds - ie so their xbots will align with the
  //adjoining lower edge. [Helpful in the ProcessHorizontal() method.]
  std::swap(e.Top.X, e.Bot.X);
#ifdef use_xyz  
  std::swap(e.Top.Z, e.Bot.Z);
#endif
}
//------------------------------------------------------------------------------

void SwapPoints(IntPoint &pt1, IntPoint &pt2)
{
  IntPoint tmp = pt1;
  pt1 = pt2;
  pt2 = tmp;
}
//------------------------------------------------------------------------------

bool GetOverlapSegment(IntPoint pt1a, IntPoint pt1b, IntPoint pt2a,
  IntPoint pt2b, IntPoint &pt1, IntPoint &pt2)
{
  //precondition: segments are Collinear.
  if (Abs(pt1a.X - pt1b.X) > Abs(pt1a.Y - pt1b.Y))
  {
    if (pt1a.X > pt1b.X) SwapPoints(pt1a, pt1b);
    if (pt2a.X > pt2b.X) SwapPoints(pt2a, pt2b);
    if (pt1a.X > pt2a.X) pt1 = pt1a; else pt1 = pt2a;
    if (pt1b.X < pt2b.X) pt2 = pt1b; else pt2 = pt2b;
    return pt1.X < pt2.X;
  } else
  {
    if (pt1a.Y < pt1b.Y) SwapPoints(pt1a, pt1b);
    if (pt2a.Y < pt2b.Y) SwapPoints(pt2a, pt2b);
    if (pt1a.Y < pt2a.Y) pt1 = pt1a; else pt1 = pt2a;
    if (pt1b.Y > pt2b.Y) pt2 = pt1b; else pt2 = pt2b;
    return pt1.Y > pt2.Y;
  }
}
//------------------------------------------------------------------------------

bool FirstIsBottomPt(const OutPt* btmPt1, const OutPt* btmPt2)
{
  OutPt *p = btmPt1->Prev;
  while ((p->Pt == btmPt1->Pt) && (p != btmPt1)) p = p->Prev;
  double dx1p = std::fabs(GetDx(btmPt1->Pt, p->Pt));
  p = btmPt1->Next;
  while ((p->Pt == btmPt1->Pt) && (p != btmPt1)) p = p->Next;
  double dx1n = std::fabs(GetDx(btmPt1->Pt, p->Pt));

  p = btmPt2->Prev;
  while ((p->Pt == btmPt2->Pt) && (p != btmPt2)) p = p->Prev;
  double dx2p = std::fabs(GetDx(btmPt2->Pt, p->Pt));
  p = btmPt2->Next;
  while ((p->Pt == btmPt2->Pt) && (p != btmPt2)) p = p->Next;
  double dx2n = std::fabs(GetDx(btmPt2->Pt, p->Pt));

  if (std::max(dx1p, dx1n) == std::max(dx2p, dx2n) &&
    std::min(dx1p, dx1n) == std::min(dx2p, dx2n))
      return Area(btmPt1) > 0; //if otherwise identical use orientation
  else
    return (dx1p >= dx2p && dx1p >= dx2n) || (dx1n >= dx2p && dx1n >= dx2n);
}
//------------------------------------------------------------------------------

OutPt* GetBottomPt(OutPt *pp)
{
  OutPt* dups = 0;
  OutPt* p = pp->Next;
  while (p != pp)
  {
    if (p->Pt.Y > pp->Pt.Y)
    {
      pp = p;
      dups = 0;
    }
    else if (p->Pt.Y == pp->Pt.Y && p->Pt.X <= pp->Pt.X)
    {
      if (p->Pt.X < pp->Pt.X)
      {
        dups = 0;
        pp = p;
      } else
      {
        if (p->Next != pp && p->Prev != pp) dups = p;
      }
    }
    p = p->Next;
  }
  if (dups)
  {
    //there appears to be at least 2 vertices at BottomPt so ...
    while (dups != p)
    {
      if (!FirstIsBottomPt(p, dups)) pp = dups;
      dups = dups->Next;
      while (dups->Pt != pp->Pt) dups = dups->Next;
    }
  }
  return pp;
}
//------------------------------------------------------------------------------

bool Pt2IsBetweenPt1AndPt3(const IntPoint pt1,
  const IntPoint pt2, const IntPoint pt3)
{
  if ((pt1 == pt3) || (pt1 == pt2) || (pt3 == pt2))
    return false;
  else if (pt1.X != pt3.X)
    return (pt2.X > pt1.X) == (pt2.X < pt3.X);
  else
    return (pt2.Y > pt1.Y) == (pt2.Y < pt3.Y);
}
//------------------------------------------------------------------------------

bool HorzSegmentsOverlap(cInt seg1a, cInt seg1b, cInt seg2a, cInt seg2b)
{
  if (seg1a > seg1b) std::swap(seg1a, seg1b);
  if (seg2a > seg2b) std::swap(seg2a, seg2b);
  return (seg1a < seg2b) && (seg2a < seg1b);
}

//------------------------------------------------------------------------------
// ClipperBase class methods ...
//------------------------------------------------------------------------------

ClipperBase::ClipperBase() //constructor
{
  m_CurrentLM = m_MinimaList.begin(); //begin() == end() here
  m_UseFullRange = false;
}
//------------------------------------------------------------------------------

ClipperBase::~ClipperBase() //destructor
{
  Clear();
}
//------------------------------------------------------------------------------

void RangeTest(const IntPoint& Pt, bool& useFullRange)
{
  if (useFullRange)
  {
    if (Pt.X > hiRange || Pt.Y > hiRange || -Pt.X > hiRange || -Pt.Y > hiRange) 
      throw clipperException("Coordinate outside allowed range");
  }
  else if (Pt.X > loRange|| Pt.Y > loRange || -Pt.X > loRange || -Pt.Y > loRange) 
  {
    useFullRange = true;
    RangeTest(Pt, useFullRange);
  }
}
//------------------------------------------------------------------------------

TEdge* FindNextLocMin(TEdge* E)
{
  for (;;)
  {
    while (E->Bot != E->Prev->Bot || E->Curr == E->Top) E = E->Next;
    if (!IsHorizontal(*E) && !IsHorizontal(*E->Prev)) break;
    while (IsHorizontal(*E->Prev)) E = E->Prev;
    TEdge* E2 = E;
    while (IsHorizontal(*E)) E = E->Next;
    if (E->Top.Y == E->Prev->Bot.Y) continue; //ie just an intermediate horz.
    if (E2->Prev->Bot.X < E->Bot.X) E = E2;
    break;
  }
  return E;
}
//------------------------------------------------------------------------------

TEdge* ClipperBase::ProcessBound(TEdge* E, bool NextIsForward)
{
  TEdge *Result = E;
  TEdge *Horz = 0;

  if (E->OutIdx == Skip)
  {
    //if edges still remain in the current bound beyond the skip edge then
    //create another LocMin and call ProcessBound once more
    if (NextIsForward)
    {
      while (E->Top.Y == E->Next->Bot.Y) E = E->Next;
      //don't include top horizontals when parsing a bound a second time,
      //they will be contained in the opposite bound ...
      while (E != Result && IsHorizontal(*E)) E = E->Prev;
    }
    else
    {
      while (E->Top.Y == E->Prev->Bot.Y) E = E->Prev;
      while (E != Result && IsHorizontal(*E)) E = E->Next;
    }

    if (E == Result)
    {
      if (NextIsForward) Result = E->Next;
      else Result = E->Prev;
    }
    else
    {
      //there are more edges in the bound beyond result starting with E
      if (NextIsForward)
        E = Result->Next;
      else
        E = Result->Prev;
      MinimaList::value_type locMin;
      locMin.Y = E->Bot.Y;
      locMin.LeftBound = 0;
      locMin.RightBound = E;
      E->WindDelta = 0;
      Result = ProcessBound(E, NextIsForward);
      m_MinimaList.push_back(locMin);
    }
    return Result;
  }

  TEdge *EStart;

  if (IsHorizontal(*E))
  {
    //We need to be careful with open paths because this may not be a
    //true local minima (ie E may be following a skip edge).
    //Also, consecutive horz. edges may start heading left before going right.
    if (NextIsForward) 
      EStart = E->Prev;
    else 
      EStart = E->Next;
    if (IsHorizontal(*EStart)) //ie an adjoining horizontal skip edge
      {
        if (EStart->Bot.X != E->Bot.X && EStart->Top.X != E->Bot.X)
          ReverseHorizontal(*E);
      }
      else if (EStart->Bot.X != E->Bot.X)
        ReverseHorizontal(*E);
  }
  
  EStart = E;
  if (NextIsForward)
  {
    while (Result->Top.Y == Result->Next->Bot.Y && Result->Next->OutIdx != Skip)
      Result = Result->Next;
    if (IsHorizontal(*Result) && Result->Next->OutIdx != Skip)
    {
      //nb: at the top of a bound, horizontals are added to the bound
      //only when the preceding edge attaches to the horizontal's left vertex
      //unless a Skip edge is encountered when that becomes the top divide
      Horz = Result;
      while (IsHorizontal(*Horz->Prev)) Horz = Horz->Prev;
      if (Horz->Prev->Top.X > Result->Next->Top.X) Result = Horz->Prev;
    }
    while (E != Result) 
    {
      E->NextInLML = E->Next;
      if (IsHorizontal(*E) && E != EStart &&
        E->Bot.X != E->Prev->Top.X) ReverseHorizontal(*E);
      E = E->Next;
    }
    if (IsHorizontal(*E) && E != EStart && E->Bot.X != E->Prev->Top.X) 
      ReverseHorizontal(*E);
    Result = Result->Next; //move to the edge just beyond current bound
  } else
  {
    while (Result->Top.Y == Result->Prev->Bot.Y && Result->Prev->OutIdx != Skip) 
      Result = Result->Prev;
    if (IsHorizontal(*Result) && Result->Prev->OutIdx != Skip)
    {
      Horz = Result;
      while (IsHorizontal(*Horz->Next)) Horz = Horz->Next;
      if (Horz->Next->Top.X == Result->Prev->Top.X ||
          Horz->Next->Top.X > Result->Prev->Top.X) Result = Horz->Next;
    }

    while (E != Result)
    {
      E->NextInLML = E->Prev;
      if (IsHorizontal(*E) && E != EStart && E->Bot.X != E->Next->Top.X) 
        ReverseHorizontal(*E);
      E = E->Prev;
    }
    if (IsHorizontal(*E) && E != EStart && E->Bot.X != E->Next->Top.X) 
      ReverseHorizontal(*E);
    Result = Result->Prev; //move to the edge just beyond current bound
  }

  return Result;
}
//------------------------------------------------------------------------------

bool ClipperBase::AddPath(const Path &pg, PolyType PolyTyp, bool Closed)
{
#ifdef use_lines
  if (!Closed && PolyTyp == ptClip)
    throw clipperException("AddPath: Open paths must be subject.");
#else
  if (!Closed)
    throw clipperException("AddPath: Open paths have been disabled.");
#endif

  int highI = (int)pg.size() -1;
  if (Closed) while (highI > 0 && (pg[highI] == pg[0])) --highI;
  while (highI > 0 && (pg[highI] == pg[highI -1])) --highI;
  if ((Closed && highI < 2) || (!Closed && highI < 1)) return false;

  //create a new edge array ...
  TEdge *edges = new TEdge [highI +1];

  bool IsFlat = true;
  //1. Basic (first) edge initialization ...
  try
  {
    edges[1].Curr = pg[1];
    RangeTest(pg[0], m_UseFullRange);
    RangeTest(pg[highI], m_UseFullRange);
    InitEdge(&edges[0], &edges[1], &edges[highI], pg[0]);
    InitEdge(&edges[highI], &edges[0], &edges[highI-1], pg[highI]);
    for (int i = highI - 1; i >= 1; --i)
    {
      RangeTest(pg[i], m_UseFullRange);
      InitEdge(&edges[i], &edges[i+1], &edges[i-1], pg[i]);
    }
  }
  catch(...)
  {
    delete [] edges;
    throw; //range test fails
  }
  TEdge *eStart = &edges[0];

  //2. Remove duplicate vertices, and (when closed) collinear edges ...
  TEdge *E = eStart, *eLoopStop = eStart;
  for (;;)
  {
    //nb: allows matching start and end points when not Closed ...
    if (E->Curr == E->Next->Curr && (Closed || E->Next != eStart))
    {
      if (E == E->Next) break;
      if (E == eStart) eStart = E->Next;
      E = RemoveEdge(E);
      eLoopStop = E;
      continue;
    }
    if (E->Prev == E->Next) 
      break; //only two vertices
    else if (Closed &&
      SlopesEqual(E->Prev->Curr, E->Curr, E->Next->Curr, m_UseFullRange) && 
      (!m_PreserveCollinear ||
      !Pt2IsBetweenPt1AndPt3(E->Prev->Curr, E->Curr, E->Next->Curr)))
    {
      //Collinear edges are allowed for open paths but in closed paths
      //the default is to merge adjacent collinear edges into a single edge.
      //However, if the PreserveCollinear property is enabled, only overlapping
      //collinear edges (ie spikes) will be removed from closed paths.
      if (E == eStart) eStart = E->Next;
      E = RemoveEdge(E);
      E = E->Prev;
      eLoopStop = E;
      continue;
    }
    E = E->Next;
    if ((E == eLoopStop) || (!Closed && E->Next == eStart)) break;
  }

  if ((!Closed && (E == E->Next)) || (Closed && (E->Prev == E->Next)))
  {
    delete [] edges;
    return false;
  }

  if (!Closed)
  { 
    m_HasOpenPaths = true;
    eStart->Prev->OutIdx = Skip;
  }

  //3. Do second stage of edge initialization ...
  E = eStart;
  do
  {
    InitEdge2(*E, PolyTyp);
    E = E->Next;
    if (IsFlat && E->Curr.Y != eStart->Curr.Y) IsFlat = false;
  }
  while (E != eStart);

  //4. Finally, add edge bounds to LocalMinima list ...

  //Totally flat paths must be handled differently when adding them
  //to LocalMinima list to avoid endless loops etc ...
  if (IsFlat) 
  {
    if (Closed) 
    {
      delete [] edges;
      return false;
    }
    E->Prev->OutIdx = Skip;
    MinimaList::value_type locMin;
    locMin.Y = E->Bot.Y;
    locMin.LeftBound = 0;
    locMin.RightBound = E;
    locMin.RightBound->Side = esRight;
    locMin.RightBound->WindDelta = 0;
    for (;;)
    {
      if (E->Bot.X != E->Prev->Top.X) ReverseHorizontal(*E);
      if (E->Next->OutIdx == Skip) break;
      E->NextInLML = E->Next;
      E = E->Next;
    }
    m_MinimaList.push_back(locMin);
    m_edges.push_back(edges);
	  return true;
  }

  m_edges.push_back(edges);
  bool leftBoundIsForward;
  TEdge* EMin = 0;

  //workaround to avoid an endless loop in the while loop below when
  //open paths have matching start and end points ...
  if (E->Prev->Bot == E->Prev->Top) E = E->Next;

  for (;;)
  {
    E = FindNextLocMin(E);
    if (E == EMin) break;
    else if (!EMin) EMin = E;

    //E and E.Prev now share a local minima (left aligned if horizontal).
    //Compare their slopes to find which starts which bound ...
    MinimaList::value_type locMin;
    locMin.Y = E->Bot.Y;
    if (E->Dx < E->Prev->Dx) 
    {
      locMin.LeftBound = E->Prev;
      locMin.RightBound = E;
      leftBoundIsForward = false; //Q.nextInLML = Q.prev
    } else
    {
      locMin.LeftBound = E;
      locMin.RightBound = E->Prev;
      leftBoundIsForward = true; //Q.nextInLML = Q.next
    }

    if (!Closed) locMin.LeftBound->WindDelta = 0;
    else if (locMin.LeftBound->Next == locMin.RightBound)
      locMin.LeftBound->WindDelta = -1;
    else locMin.LeftBound->WindDelta = 1;
    locMin.RightBound->WindDelta = -locMin.LeftBound->WindDelta;

    E = ProcessBound(locMin.LeftBound, leftBoundIsForward);
    if (E->OutIdx == Skip) E = ProcessBound(E, leftBoundIsForward);

    TEdge* E2 = ProcessBound(locMin.RightBound, !leftBoundIsForward);
    if (E2->OutIdx == Skip) E2 = ProcessBound(E2, !leftBoundIsForward);

    if (locMin.LeftBound->OutIdx == Skip)
      locMin.LeftBound = 0;
    else if (locMin.RightBound->OutIdx == Skip)
      locMin.RightBound = 0;
    m_MinimaList.push_back(locMin);
    if (!leftBoundIsForward) E = E2;
  }
  return true;
}
//------------------------------------------------------------------------------

bool ClipperBase::AddPaths(const Paths &ppg, PolyType PolyTyp, bool Closed)
{
  bool result = false;
  for (Paths::size_type i = 0; i < ppg.size(); ++i)
    if (AddPath(ppg[i], PolyTyp, Closed)) result = true;
  return result;
}
//------------------------------------------------------------------------------

void ClipperBase::Clear()
{
  DisposeLocalMinimaList();
  for (EdgeList::size_type i = 0; i < m_edges.size(); ++i)
  {
    TEdge* edges = m_edges[i];
    delete [] edges;
  }
  m_edges.clear();
  m_UseFullRange = false;
  m_HasOpenPaths = false;
}
//------------------------------------------------------------------------------

void ClipperBase::Reset()
{
  m_CurrentLM = m_MinimaList.begin();
  if (m_CurrentLM == m_MinimaList.end()) return; //ie nothing to process
  std::sort(m_MinimaList.begin(), m_MinimaList.end(), LocMinSorter());

  m_Scanbeam = ScanbeamList(); //clears/resets priority_queue
  //reset all edges ...
  for (MinimaList::iterator lm = m_MinimaList.begin(); lm != m_MinimaList.end(); ++lm)
  {
    InsertScanbeam(lm->Y);
    TEdge* e = lm->LeftBound;
    if (e)
    {
      e->Curr = e->Bot;
      e->Side = esLeft;
      e->OutIdx = Unassigned;
    }

    e = lm->RightBound;
    if (e)
    {
      e->Curr = e->Bot;
      e->Side = esRight;
      e->OutIdx = Unassigned;
    }
  }
  m_ActiveEdges = 0;
  m_CurrentLM = m_MinimaList.begin();
}
//------------------------------------------------------------------------------

void ClipperBase::DisposeLocalMinimaList()
{
  m_MinimaList.clear();
  m_CurrentLM = m_MinimaList.begin();
}
//------------------------------------------------------------------------------

bool ClipperBase::PopLocalMinima(cInt Y, const LocalMinimum *&locMin)
{
  if (m_CurrentLM == m_MinimaList.end() || (*m_CurrentLM).Y != Y) return false;
  locMin = &(*m_CurrentLM);
  ++m_CurrentLM;
  return true;
}
//------------------------------------------------------------------------------

IntRect ClipperBase::GetBounds()
{
  IntRect result;
  MinimaList::iterator lm = m_MinimaList.begin();
  if (lm == m_MinimaList.end())
  {
    result.left = result.top = result.right = result.bottom = 0;
    return result;
  }
  result.left = lm->LeftBound->Bot.X;
  result.top = lm->LeftBound->Bot.Y;
  result.right = lm->LeftBound->Bot.X;
  result.bottom = lm->LeftBound->Bot.Y;
  while (lm != m_MinimaList.end())
  {
    //todo - needs fixing for open paths
    result.bottom = std::max(result.bottom, lm->LeftBound->Bot.Y);
    TEdge* e = lm->LeftBound;
    for (;;) {
      TEdge* bottomE = e;
      while (e->NextInLML)
      {
        if (e->Bot.X < result.left) result.left = e->Bot.X;
        if (e->Bot.X > result.right) result.right = e->Bot.X;
        e = e->NextInLML;
      }
      result.left = std::min(result.left, e->Bot.X);
      result.right = std::max(result.right, e->Bot.X);
      result.left = std::min(result.left, e->Top.X);
      result.right = std::max(result.right, e->Top.X);
      result.top = std::min(result.top, e->Top.Y);
      if (bottomE == lm->LeftBound) e = lm->RightBound;
      else break;
    }
    ++lm;
  }
  return result;
}
//------------------------------------------------------------------------------

void ClipperBase::InsertScanbeam(const cInt Y)
{
  m_Scanbeam.push(Y);
}
//------------------------------------------------------------------------------

bool ClipperBase::PopScanbeam(cInt &Y)
{
  if (m_Scanbeam.empty()) return false;
  Y = m_Scanbeam.top();
  m_Scanbeam.pop();
  while (!m_Scanbeam.empty() && Y == m_Scanbeam.top()) { m_Scanbeam.pop(); } // Pop duplicates.
  return true;
}
//------------------------------------------------------------------------------

void ClipperBase::DisposeAllOutRecs(){
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i)
    DisposeOutRec(i);
  m_PolyOuts.clear();
}
//------------------------------------------------------------------------------

void ClipperBase::DisposeOutRec(PolyOutList::size_type index)
{
  OutRec *outRec = m_PolyOuts[index];
  if (outRec->Pts) DisposeOutPts(outRec->Pts);
  delete outRec;
  m_PolyOuts[index] = 0;
}
//------------------------------------------------------------------------------

void ClipperBase::DeleteFromAEL(TEdge *e)
{
  TEdge* AelPrev = e->PrevInAEL;
  TEdge* AelNext = e->NextInAEL;
  if (!AelPrev &&  !AelNext && (e != m_ActiveEdges)) return; //already deleted
  if (AelPrev) AelPrev->NextInAEL = AelNext;
  else m_ActiveEdges = AelNext;
  if (AelNext) AelNext->PrevInAEL = AelPrev;
  e->NextInAEL = 0;
  e->PrevInAEL = 0;
}
//------------------------------------------------------------------------------

OutRec* ClipperBase::CreateOutRec()
{
  OutRec* result = new OutRec;
  result->IsHole = false;
  result->IsOpen = false;
  result->FirstLeft = 0;
  result->Pts = 0;
  result->BottomPt = 0;
  result->PolyNd = 0;
  m_PolyOuts.push_back(result);
  result->Idx = (int)m_PolyOuts.size() - 1;
  return result;
}
//------------------------------------------------------------------------------

void ClipperBase::SwapPositionsInAEL(TEdge *Edge1, TEdge *Edge2)
{
  //check that one or other edge hasn't already been removed from AEL ...
  if (Edge1->NextInAEL == Edge1->PrevInAEL ||
    Edge2->NextInAEL == Edge2->PrevInAEL) return;

  if (Edge1->NextInAEL == Edge2)
  {
    TEdge* Next = Edge2->NextInAEL;
    if (Next) Next->PrevInAEL = Edge1;
    TEdge* Prev = Edge1->PrevInAEL;
    if (Prev) Prev->NextInAEL = Edge2;
    Edge2->PrevInAEL = Prev;
    Edge2->NextInAEL = Edge1;
    Edge1->PrevInAEL = Edge2;
    Edge1->NextInAEL = Next;
  }
  else if (Edge2->NextInAEL == Edge1)
  {
    TEdge* Next = Edge1->NextInAEL;
    if (Next) Next->PrevInAEL = Edge2;
    TEdge* Prev = Edge2->PrevInAEL;
    if (Prev) Prev->NextInAEL = Edge1;
    Edge1->PrevInAEL = Prev;
    Edge1->NextInAEL = Edge2;
    Edge2->PrevInAEL = Edge1;
    Edge2->NextInAEL = Next;
  }
  else
  {
    TEdge* Next = Edge1->NextInAEL;
    TEdge* Prev = Edge1->PrevInAEL;
    Edge1->NextInAEL = Edge2->NextInAEL;
    if (Edge1->NextInAEL) Edge1->NextInAEL->PrevInAEL = Edge1;
    Edge1->PrevInAEL = Edge2->PrevInAEL;
    if (Edge1->PrevInAEL) Edge1->PrevInAEL->NextInAEL = Edge1;
    Edge2->NextInAEL = Next;
    if (Edge2->NextInAEL) Edge2->NextInAEL->PrevInAEL = Edge2;
    Edge2->PrevInAEL = Prev;
    if (Edge2->PrevInAEL) Edge2->PrevInAEL->NextInAEL = Edge2;
  }

  if (!Edge1->PrevInAEL) m_ActiveEdges = Edge1;
  else if (!Edge2->PrevInAEL) m_ActiveEdges = Edge2;
}
//------------------------------------------------------------------------------

void ClipperBase::UpdateEdgeIntoAEL(TEdge *&e)
{
  if (!e->NextInLML) 
    throw clipperException("UpdateEdgeIntoAEL: invalid call");

  e->NextInLML->OutIdx = e->OutIdx;
  TEdge* AelPrev = e->PrevInAEL;
  TEdge* AelNext = e->NextInAEL;
  if (AelPrev) AelPrev->NextInAEL = e->NextInLML;
  else m_ActiveEdges = e->NextInLML;
  if (AelNext) AelNext->PrevInAEL = e->NextInLML;
  e->NextInLML->Side = e->Side;
  e->NextInLML->WindDelta = e->WindDelta;
  e->NextInLML->WindCnt = e->WindCnt;
  e->NextInLML->WindCnt2 = e->WindCnt2;
  e = e->NextInLML;
  e->Curr = e->Bot;
  e->PrevInAEL = AelPrev;
  e->NextInAEL = AelNext;
  if (!IsHorizontal(*e)) InsertScanbeam(e->Top.Y);
}
//------------------------------------------------------------------------------

bool ClipperBase::LocalMinimaPending()
{
  return (m_CurrentLM != m_MinimaList.end());
}

//------------------------------------------------------------------------------
// TClipper methods ...
//------------------------------------------------------------------------------

Clipper::Clipper(int initOptions) : ClipperBase() //constructor
{
  m_ExecuteLocked = false;
  m_UseFullRange = false;
  m_ReverseOutput = ((initOptions & ioReverseSolution) != 0);
  m_StrictSimple = ((initOptions & ioStrictlySimple) != 0);
  m_PreserveCollinear = ((initOptions & ioPreserveCollinear) != 0);
  m_HasOpenPaths = false;
#ifdef use_xyz  
  m_ZFill = 0;
#endif
}
//------------------------------------------------------------------------------

#ifdef use_xyz  
void Clipper::ZFillFunction(ZFillCallback zFillFunc)
{  
  m_ZFill = zFillFunc;
}
//------------------------------------------------------------------------------
#endif

bool Clipper::Execute(ClipType clipType, Paths &solution, PolyFillType fillType)
{
    return Execute(clipType, solution, fillType, fillType);
}
//------------------------------------------------------------------------------

bool Clipper::Execute(ClipType clipType, PolyTree &polytree, PolyFillType fillType)
{
    return Execute(clipType, polytree, fillType, fillType);
}
//------------------------------------------------------------------------------

bool Clipper::Execute(ClipType clipType, Paths &solution,
    PolyFillType subjFillType, PolyFillType clipFillType)
{
  if( m_ExecuteLocked ) return false;
  if (m_HasOpenPaths)
    throw clipperException("Error: PolyTree struct is needed for open path clipping.");
  m_ExecuteLocked = true;
  solution.resize(0);
  m_SubjFillType = subjFillType;
  m_ClipFillType = clipFillType;
  m_ClipType = clipType;
  m_UsingPolyTree = false;
  bool succeeded = ExecuteInternal();
  if (succeeded) BuildResult(solution);
  DisposeAllOutRecs();
  m_ExecuteLocked = false;
  return succeeded;
}
//------------------------------------------------------------------------------

bool Clipper::Execute(ClipType clipType, PolyTree& polytree,
    PolyFillType subjFillType, PolyFillType clipFillType)
{
  if( m_ExecuteLocked ) return false;
  m_ExecuteLocked = true;
  m_SubjFillType = subjFillType;
  m_ClipFillType = clipFillType;
  m_ClipType = clipType;
  m_UsingPolyTree = true;
  bool succeeded = ExecuteInternal();
  if (succeeded) BuildResult2(polytree);
  DisposeAllOutRecs();
  m_ExecuteLocked = false;
  return succeeded;
}
//------------------------------------------------------------------------------

void Clipper::FixHoleLinkage(OutRec &outrec)
{
  //skip OutRecs that (a) contain outermost polygons or
  //(b) already have the correct owner/child linkage ...
  if (!outrec.FirstLeft ||                
      (outrec.IsHole != outrec.FirstLeft->IsHole &&
      outrec.FirstLeft->Pts)) return;

  OutRec* orfl = outrec.FirstLeft;
  while (orfl && ((orfl->IsHole == outrec.IsHole) || !orfl->Pts))
      orfl = orfl->FirstLeft;
  outrec.FirstLeft = orfl;
}
//------------------------------------------------------------------------------

bool Clipper::ExecuteInternal()
{
  bool succeeded = true;
  try {
    Reset();
    m_Maxima = MaximaList();
    m_SortedEdges = 0;

    succeeded = true;
    cInt botY, topY;
    if (!PopScanbeam(botY)) return false;
    InsertLocalMinimaIntoAEL(botY);
    while (PopScanbeam(topY) || LocalMinimaPending())
    {
      ProcessHorizontals();
	    ClearGhostJoins();
      if (!ProcessIntersections(topY))
      {
        succeeded = false;
        break;
      }
      ProcessEdgesAtTopOfScanbeam(topY);
      botY = topY;
      InsertLocalMinimaIntoAEL(botY);
    }
  }
  catch(...) 
  {
    succeeded = false;
  }

  if (succeeded)
  {
    //fix orientations ...
    for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i)
    {
      OutRec *outRec = m_PolyOuts[i];
      if (!outRec->Pts || outRec->IsOpen) continue;
      if ((outRec->IsHole ^ m_ReverseOutput) == (Area(*outRec) > 0))
        ReversePolyPtLinks(outRec->Pts);
    }

    if (!m_Joins.empty()) JoinCommonEdges();

    //unfortunately FixupOutPolygon() must be done after JoinCommonEdges()
    for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i)
    {
      OutRec *outRec = m_PolyOuts[i];
      if (!outRec->Pts) continue;
      if (outRec->IsOpen)
        FixupOutPolyline(*outRec);
      else
        FixupOutPolygon(*outRec);
    }

    if (m_StrictSimple) DoSimplePolygons();
  }

  ClearJoins();
  ClearGhostJoins();
  return succeeded;
}
//------------------------------------------------------------------------------

void Clipper::SetWindingCount(TEdge &edge)
{
  TEdge *e = edge.PrevInAEL;
  //find the edge of the same polytype that immediately preceeds 'edge' in AEL
  while (e  && ((e->PolyTyp != edge.PolyTyp) || (e->WindDelta == 0))) e = e->PrevInAEL;
  if (!e)
  {
    if (edge.WindDelta == 0)
    {
      PolyFillType pft = (edge.PolyTyp == ptSubject ? m_SubjFillType : m_ClipFillType);
      edge.WindCnt = (pft == pftNegative ? -1 : 1);
    }
    else
      edge.WindCnt = edge.WindDelta;
    edge.WindCnt2 = 0;
    e = m_ActiveEdges; //ie get ready to calc WindCnt2
  }   
  else if (edge.WindDelta == 0 && m_ClipType != ctUnion)
  {
    edge.WindCnt = 1;
    edge.WindCnt2 = e->WindCnt2;
    e = e->NextInAEL; //ie get ready to calc WindCnt2
  }
  else if (IsEvenOddFillType(edge))
  {
    //EvenOdd filling ...
    if (edge.WindDelta == 0)
    {
      //are we inside a subj polygon ...
      bool Inside = true;
      TEdge *e2 = e->PrevInAEL;
      while (e2)
      {
        if (e2->PolyTyp == e->PolyTyp && e2->WindDelta != 0) 
          Inside = !Inside;
        e2 = e2->PrevInAEL;
      }
      edge.WindCnt = (Inside ? 0 : 1);
    }
    else
    {
      edge.WindCnt = edge.WindDelta;
    }
    edge.WindCnt2 = e->WindCnt2;
    e = e->NextInAEL; //ie get ready to calc WindCnt2
  } 
  else
  {
    //nonZero, Positive or Negative filling ...
    if (e->WindCnt * e->WindDelta < 0)
    {
      //prev edge is 'decreasing' WindCount (WC) toward zero
      //so we're outside the previous polygon ...
      if (Abs(e->WindCnt) > 1)
      {
        //outside prev poly but still inside another.
        //when reversing direction of prev poly use the same WC 
        if (e->WindDelta * edge.WindDelta < 0) edge.WindCnt = e->WindCnt;
        //otherwise continue to 'decrease' WC ...
        else edge.WindCnt = e->WindCnt + edge.WindDelta;
      } 
      else
        //now outside all polys of same polytype so set own WC ...
        edge.WindCnt = (edge.WindDelta == 0 ? 1 : edge.WindDelta);
    } else
    {
      //prev edge is 'increasing' WindCount (WC) away from zero
      //so we're inside the previous polygon ...
      if (edge.WindDelta == 0) 
        edge.WindCnt = (e->WindCnt < 0 ? e->WindCnt - 1 : e->WindCnt + 1);
      //if wind direction is reversing prev then use same WC
      else if (e->WindDelta * edge.WindDelta < 0) edge.WindCnt = e->WindCnt;
      //otherwise add to WC ...
      else edge.WindCnt = e->WindCnt + edge.WindDelta;
    }
    edge.WindCnt2 = e->WindCnt2;
    e = e->NextInAEL; //ie get ready to calc WindCnt2
  }

  //update WindCnt2 ...
  if (IsEvenOddAltFillType(edge))
  {
    //EvenOdd filling ...
    while (e != &edge)
    {
      if (e->WindDelta != 0)
        edge.WindCnt2 = (edge.WindCnt2 == 0 ? 1 : 0);
      e = e->NextInAEL;
    }
  } else
  {
    //nonZero, Positive or Negative filling ...
    while ( e != &edge )
    {
      edge.WindCnt2 += e->WindDelta;
      e = e->NextInAEL;
    }
  }
}
//------------------------------------------------------------------------------

bool Clipper::IsEvenOddFillType(const TEdge& edge) const
{
  if (edge.PolyTyp == ptSubject)
    return m_SubjFillType == pftEvenOdd; else
    return m_ClipFillType == pftEvenOdd;
}
//------------------------------------------------------------------------------

bool Clipper::IsEvenOddAltFillType(const TEdge& edge) const
{
  if (edge.PolyTyp == ptSubject)
    return m_ClipFillType == pftEvenOdd; else
    return m_SubjFillType == pftEvenOdd;
}
//------------------------------------------------------------------------------

bool Clipper::IsContributing(const TEdge& edge) const
{
  PolyFillType pft, pft2;
  if (edge.PolyTyp == ptSubject)
  {
    pft = m_SubjFillType;
    pft2 = m_ClipFillType;
  } else
  {
    pft = m_ClipFillType;
    pft2 = m_SubjFillType;
  }

  switch(pft)
  {
    case pftEvenOdd: 
      //return false if a subj line has been flagged as inside a subj polygon
      if (edge.WindDelta == 0 && edge.WindCnt != 1) return false;
      break;
    case pftNonZero:
      if (Abs(edge.WindCnt) != 1) return false;
      break;
    case pftPositive: 
      if (edge.WindCnt != 1) return false;
      break;
    default: //pftNegative
      if (edge.WindCnt != -1) return false;
  }

  switch(m_ClipType)
  {
    case ctIntersection:
      switch(pft2)
      {
        case pftEvenOdd: 
        case pftNonZero: 
          return (edge.WindCnt2 != 0);
        case pftPositive: 
          return (edge.WindCnt2 > 0);
        default: 
          return (edge.WindCnt2 < 0);
      }
      break;
    case ctUnion:
      switch(pft2)
      {
        case pftEvenOdd: 
        case pftNonZero: 
          return (edge.WindCnt2 == 0);
        case pftPositive: 
          return (edge.WindCnt2 <= 0);
        default: 
          return (edge.WindCnt2 >= 0);
      }
      break;
    case ctDifference:
      if (edge.PolyTyp == ptSubject)
        switch(pft2)
        {
          case pftEvenOdd: 
          case pftNonZero: 
            return (edge.WindCnt2 == 0);
          case pftPositive: 
            return (edge.WindCnt2 <= 0);
          default: 
            return (edge.WindCnt2 >= 0);
        }
      else
        switch(pft2)
        {
          case pftEvenOdd: 
          case pftNonZero: 
            return (edge.WindCnt2 != 0);
          case pftPositive: 
            return (edge.WindCnt2 > 0);
          default: 
            return (edge.WindCnt2 < 0);
        }
      break;
    case ctXor:
      if (edge.WindDelta == 0) //XOr always contributing unless open
        switch(pft2)
        {
          case pftEvenOdd: 
          case pftNonZero: 
            return (edge.WindCnt2 == 0);
          case pftPositive: 
            return (edge.WindCnt2 <= 0);
          default: 
            return (edge.WindCnt2 >= 0);
        }
      else 
        return true;
      break;
    default:
      return true;
  }
}
//------------------------------------------------------------------------------

OutPt* Clipper::AddLocalMinPoly(TEdge *e1, TEdge *e2, const IntPoint &Pt)
{
  OutPt* result;
  TEdge *e, *prevE;
  if (IsHorizontal(*e2) || ( e1->Dx > e2->Dx ))
  {
    result = AddOutPt(e1, Pt);
    e2->OutIdx = e1->OutIdx;
    e1->Side = esLeft;
    e2->Side = esRight;
    e = e1;
    if (e->PrevInAEL == e2)
      prevE = e2->PrevInAEL; 
    else
      prevE = e->PrevInAEL;
  } else
  {
    result = AddOutPt(e2, Pt);
    e1->OutIdx = e2->OutIdx;
    e1->Side = esRight;
    e2->Side = esLeft;
    e = e2;
    if (e->PrevInAEL == e1)
        prevE = e1->PrevInAEL;
    else
        prevE = e->PrevInAEL;
  }

  if (prevE && prevE->OutIdx >= 0 && prevE->Top.Y < Pt.Y && e->Top.Y < Pt.Y) 
  {
    cInt xPrev = TopX(*prevE, Pt.Y);
    cInt xE = TopX(*e, Pt.Y);
    if (xPrev == xE && (e->WindDelta != 0) && (prevE->WindDelta != 0) &&
      SlopesEqual(IntPoint(xPrev, Pt.Y), prevE->Top, IntPoint(xE, Pt.Y), e->Top, m_UseFullRange))
    {
      OutPt* outPt = AddOutPt(prevE, Pt);
      AddJoin(result, outPt, e->Top);
    }
  }
  return result;
}
//------------------------------------------------------------------------------

void Clipper::AddLocalMaxPoly(TEdge *e1, TEdge *e2, const IntPoint &Pt)
{
  AddOutPt( e1, Pt );
  if (e2->WindDelta == 0) AddOutPt(e2, Pt);
  if( e1->OutIdx == e2->OutIdx )
  {
    e1->OutIdx = Unassigned;
    e2->OutIdx = Unassigned;
  }
  else if (e1->OutIdx < e2->OutIdx) 
    AppendPolygon(e1, e2); 
  else 
    AppendPolygon(e2, e1);
}
//------------------------------------------------------------------------------

void Clipper::AddEdgeToSEL(TEdge *edge)
{
  //SEL pointers in PEdge are reused to build a list of horizontal edges.
  //However, we don't need to worry about order with horizontal edge processing.
  if( !m_SortedEdges )
  {
    m_SortedEdges = edge;
    edge->PrevInSEL = 0;
    edge->NextInSEL = 0;
  }
  else
  {
    edge->NextInSEL = m_SortedEdges;
    edge->PrevInSEL = 0;
    m_SortedEdges->PrevInSEL = edge;
    m_SortedEdges = edge;
  }
}
//------------------------------------------------------------------------------

bool Clipper::PopEdgeFromSEL(TEdge *&edge)
{
  if (!m_SortedEdges) return false;
  edge = m_SortedEdges;
  DeleteFromSEL(m_SortedEdges);
  return true;
}
//------------------------------------------------------------------------------

void Clipper::CopyAELToSEL()
{
  TEdge* e = m_ActiveEdges;
  m_SortedEdges = e;
  while ( e )
  {
    e->PrevInSEL = e->PrevInAEL;
    e->NextInSEL = e->NextInAEL;
    e = e->NextInAEL;
  }
}
//------------------------------------------------------------------------------

void Clipper::AddJoin(OutPt *op1, OutPt *op2, const IntPoint OffPt)
{
  Join* j = new Join;
  j->OutPt1 = op1;
  j->OutPt2 = op2;
  j->OffPt = OffPt;
  m_Joins.push_back(j);
}
//------------------------------------------------------------------------------

void Clipper::ClearJoins()
{
  for (JoinList::size_type i = 0; i < m_Joins.size(); i++)
    delete m_Joins[i];
  m_Joins.resize(0);
}
//------------------------------------------------------------------------------

void Clipper::ClearGhostJoins()
{
  for (JoinList::size_type i = 0; i < m_GhostJoins.size(); i++)
    delete m_GhostJoins[i];
  m_GhostJoins.resize(0);
}
//------------------------------------------------------------------------------

void Clipper::AddGhostJoin(OutPt *op, const IntPoint OffPt)
{
  Join* j = new Join;
  j->OutPt1 = op;
  j->OutPt2 = 0;
  j->OffPt = OffPt;
  m_GhostJoins.push_back(j);
}
//------------------------------------------------------------------------------

void Clipper::InsertLocalMinimaIntoAEL(const cInt botY)
{
  const LocalMinimum *lm;
  while (PopLocalMinima(botY, lm))
  {
    TEdge* lb = lm->LeftBound;
    TEdge* rb = lm->RightBound;
    
    OutPt *Op1 = 0;
    if (!lb)
    {
      //nb: don't insert LB into either AEL or SEL
      InsertEdgeIntoAEL(rb, 0);
      SetWindingCount(*rb);
      if (IsContributing(*rb))
        Op1 = AddOutPt(rb, rb->Bot); 
    } 
    else if (!rb)
    {
      InsertEdgeIntoAEL(lb, 0);
      SetWindingCount(*lb);
      if (IsContributing(*lb))
        Op1 = AddOutPt(lb, lb->Bot);
      InsertScanbeam(lb->Top.Y);
    }
    else
    {
      InsertEdgeIntoAEL(lb, 0);
      InsertEdgeIntoAEL(rb, lb);
      SetWindingCount( *lb );
      rb->WindCnt = lb->WindCnt;
      rb->WindCnt2 = lb->WindCnt2;
      if (IsContributing(*lb))
        Op1 = AddLocalMinPoly(lb, rb, lb->Bot);      
      InsertScanbeam(lb->Top.Y);
    }

     if (rb)
     {
		 if (IsHorizontal(*rb))
		 {
			 AddEdgeToSEL(rb);
			 if (rb->NextInLML) 
				 InsertScanbeam(rb->NextInLML->Top.Y);
		 }
		 else InsertScanbeam( rb->Top.Y );
     }

    if (!lb || !rb) continue;

    //if any output polygons share an edge, they'll need joining later ...
    if (Op1 && IsHorizontal(*rb) && 
      m_GhostJoins.size() > 0 && (rb->WindDelta != 0))
    {
      for (JoinList::size_type i = 0; i < m_GhostJoins.size(); ++i)
      {
        Join* jr = m_GhostJoins[i];
        //if the horizontal Rb and a 'ghost' horizontal overlap, then convert
        //the 'ghost' join to a real join ready for later ...
        if (HorzSegmentsOverlap(jr->OutPt1->Pt.X, jr->OffPt.X, rb->Bot.X, rb->Top.X))
          AddJoin(jr->OutPt1, Op1, jr->OffPt);
      }
    }

    if (lb->OutIdx >= 0 && lb->PrevInAEL && 
      lb->PrevInAEL->Curr.X == lb->Bot.X &&
      lb->PrevInAEL->OutIdx >= 0 &&
      SlopesEqual(lb->PrevInAEL->Bot, lb->PrevInAEL->Top, lb->Curr, lb->Top, m_UseFullRange) &&
      (lb->WindDelta != 0) && (lb->PrevInAEL->WindDelta != 0))
    {
        OutPt *Op2 = AddOutPt(lb->PrevInAEL, lb->Bot);
        AddJoin(Op1, Op2, lb->Top);
    }

    if(lb->NextInAEL != rb)
    {

      if (rb->OutIdx >= 0 && rb->PrevInAEL->OutIdx >= 0 &&
        SlopesEqual(rb->PrevInAEL->Curr, rb->PrevInAEL->Top, rb->Curr, rb->Top, m_UseFullRange) &&
        (rb->WindDelta != 0) && (rb->PrevInAEL->WindDelta != 0))
      {
          OutPt *Op2 = AddOutPt(rb->PrevInAEL, rb->Bot);
          AddJoin(Op1, Op2, rb->Top);
      }

      TEdge* e = lb->NextInAEL;
      if (e)
      {
        while( e != rb )
        {
          //nb: For calculating winding counts etc, IntersectEdges() assumes
          //that param1 will be to the Right of param2 ABOVE the intersection ...
          IntersectEdges(rb , e , lb->Curr); //order important here
          e = e->NextInAEL;
        }
      }
    }
    
  }
}
//------------------------------------------------------------------------------

void Clipper::DeleteFromSEL(TEdge *e)
{
  TEdge* SelPrev = e->PrevInSEL;
  TEdge* SelNext = e->NextInSEL;
  if( !SelPrev &&  !SelNext && (e != m_SortedEdges) ) return; //already deleted
  if( SelPrev ) SelPrev->NextInSEL = SelNext;
  else m_SortedEdges = SelNext;
  if( SelNext ) SelNext->PrevInSEL = SelPrev;
  e->NextInSEL = 0;
  e->PrevInSEL = 0;
}
//------------------------------------------------------------------------------

#ifdef use_xyz
void Clipper::SetZ(IntPoint& pt, TEdge& e1, TEdge& e2)
{
  if (pt.Z != 0 || !m_ZFill) return;
  else if (pt == e1.Bot) pt.Z = e1.Bot.Z;
  else if (pt == e1.Top) pt.Z = e1.Top.Z;
  else if (pt == e2.Bot) pt.Z = e2.Bot.Z;
  else if (pt == e2.Top) pt.Z = e2.Top.Z;
  else (*m_ZFill)(e1.Bot, e1.Top, e2.Bot, e2.Top, pt); 
}
//------------------------------------------------------------------------------
#endif

void Clipper::IntersectEdges(TEdge *e1, TEdge *e2, IntPoint &Pt)
{
  bool e1Contributing = ( e1->OutIdx >= 0 );
  bool e2Contributing = ( e2->OutIdx >= 0 );

#ifdef use_xyz
        SetZ(Pt, *e1, *e2);
#endif

#ifdef use_lines
  //if either edge is on an OPEN path ...
  if (e1->WindDelta == 0 || e2->WindDelta == 0)
  {
    //ignore subject-subject open path intersections UNLESS they
    //are both open paths, AND they are both 'contributing maximas' ...
	if (e1->WindDelta == 0 && e2->WindDelta == 0) return;

    //if intersecting a subj line with a subj poly ...
    else if (e1->PolyTyp == e2->PolyTyp && 
      e1->WindDelta != e2->WindDelta && m_ClipType == ctUnion)
    {
      if (e1->WindDelta == 0)
      {
        if (e2Contributing)
        {
          AddOutPt(e1, Pt);
          if (e1Contributing) e1->OutIdx = Unassigned;
        }
      }
      else
      {
        if (e1Contributing)
        {
          AddOutPt(e2, Pt);
          if (e2Contributing) e2->OutIdx = Unassigned;
        }
      }
    }
    else if (e1->PolyTyp != e2->PolyTyp)
    {
      //toggle subj open path OutIdx on/off when Abs(clip.WndCnt) == 1 ...
      if ((e1->WindDelta == 0) && abs(e2->WindCnt) == 1 && 
        (m_ClipType != ctUnion || e2->WindCnt2 == 0))
      {
        AddOutPt(e1, Pt);
        if (e1Contributing) e1->OutIdx = Unassigned;
      }
      else if ((e2->WindDelta == 0) && (abs(e1->WindCnt) == 1) && 
        (m_ClipType != ctUnion || e1->WindCnt2 == 0))
      {
        AddOutPt(e2, Pt);
        if (e2Contributing) e2->OutIdx = Unassigned;
      }
    }
    return;
  }
#endif

  //update winding counts...
  //assumes that e1 will be to the Right of e2 ABOVE the intersection
  if ( e1->PolyTyp == e2->PolyTyp )
  {
    if ( IsEvenOddFillType( *e1) )
    {
      int oldE1WindCnt = e1->WindCnt;
      e1->WindCnt = e2->WindCnt;
      e2->WindCnt = oldE1WindCnt;
    } else
    {
      if (e1->WindCnt + e2->WindDelta == 0 ) e1->WindCnt = -e1->WindCnt;
      else e1->WindCnt += e2->WindDelta;
      if ( e2->WindCnt - e1->WindDelta == 0 ) e2->WindCnt = -e2->WindCnt;
      else e2->WindCnt -= e1->WindDelta;
    }
  } else
  {
    if (!IsEvenOddFillType(*e2)) e1->WindCnt2 += e2->WindDelta;
    else e1->WindCnt2 = ( e1->WindCnt2 == 0 ) ? 1 : 0;
    if (!IsEvenOddFillType(*e1)) e2->WindCnt2 -= e1->WindDelta;
    else e2->WindCnt2 = ( e2->WindCnt2 == 0 ) ? 1 : 0;
  }

  PolyFillType e1FillType, e2FillType, e1FillType2, e2FillType2;
  if (e1->PolyTyp == ptSubject)
  {
    e1FillType = m_SubjFillType;
    e1FillType2 = m_ClipFillType;
  } else
  {
    e1FillType = m_ClipFillType;
    e1FillType2 = m_SubjFillType;
  }
  if (e2->PolyTyp == ptSubject)
  {
    e2FillType = m_SubjFillType;
    e2FillType2 = m_ClipFillType;
  } else
  {
    e2FillType = m_ClipFillType;
    e2FillType2 = m_SubjFillType;
  }

  cInt e1Wc, e2Wc;
  switch (e1FillType)
  {
    case pftPositive: e1Wc = e1->WindCnt; break;
    case pftNegative: e1Wc = -e1->WindCnt; break;
    default: e1Wc = Abs(e1->WindCnt);
  }
  switch(e2FillType)
  {
    case pftPositive: e2Wc = e2->WindCnt; break;
    case pftNegative: e2Wc = -e2->WindCnt; break;
    default: e2Wc = Abs(e2->WindCnt);
  }

  if ( e1Contributing && e2Contributing )
  {
    if ((e1Wc != 0 && e1Wc != 1) || (e2Wc != 0 && e2Wc != 1) ||
      (e1->PolyTyp != e2->PolyTyp && m_ClipType != ctXor) )
    {
      AddLocalMaxPoly(e1, e2, Pt); 
    }
    else
    {
      AddOutPt(e1, Pt);
      AddOutPt(e2, Pt);
      SwapSides( *e1 , *e2 );
      SwapPolyIndexes( *e1 , *e2 );
    }
  }
  else if ( e1Contributing )
  {
    if (e2Wc == 0 || e2Wc == 1) 
    {
      AddOutPt(e1, Pt);
      SwapSides(*e1, *e2);
      SwapPolyIndexes(*e1, *e2);
    }
  }
  else if ( e2Contributing )
  {
    if (e1Wc == 0 || e1Wc == 1) 
    {
      AddOutPt(e2, Pt);
      SwapSides(*e1, *e2);
      SwapPolyIndexes(*e1, *e2);
    }
  } 
  else if ( (e1Wc == 0 || e1Wc == 1) && (e2Wc == 0 || e2Wc == 1))
  {
    //neither edge is currently contributing ...

    cInt e1Wc2, e2Wc2;
    switch (e1FillType2)
    {
      case pftPositive: e1Wc2 = e1->WindCnt2; break;
      case pftNegative : e1Wc2 = -e1->WindCnt2; break;
      default: e1Wc2 = Abs(e1->WindCnt2);
    }
    switch (e2FillType2)
    {
      case pftPositive: e2Wc2 = e2->WindCnt2; break;
      case pftNegative: e2Wc2 = -e2->WindCnt2; break;
      default: e2Wc2 = Abs(e2->WindCnt2);
    }

    if (e1->PolyTyp != e2->PolyTyp)
    {
      AddLocalMinPoly(e1, e2, Pt);
    }
    else if (e1Wc == 1 && e2Wc == 1)
      switch( m_ClipType ) {
        case ctIntersection:
          if (e1Wc2 > 0 && e2Wc2 > 0)
            AddLocalMinPoly(e1, e2, Pt);
          break;
        case ctUnion:
          if ( e1Wc2 <= 0 && e2Wc2 <= 0 )
            AddLocalMinPoly(e1, e2, Pt);
          break;
        case ctDifference:
          if (((e1->PolyTyp == ptClip) && (e1Wc2 > 0) && (e2Wc2 > 0)) ||
              ((e1->PolyTyp == ptSubject) && (e1Wc2 <= 0) && (e2Wc2 <= 0)))
                AddLocalMinPoly(e1, e2, Pt);
          break;
        case ctXor:
          AddLocalMinPoly(e1, e2, Pt);
      }
    else
      SwapSides( *e1, *e2 );
  }
}
//------------------------------------------------------------------------------

void Clipper::SetHoleState(TEdge *e, OutRec *outrec)
{
  TEdge *e2 = e->PrevInAEL;
  TEdge *eTmp = 0;
  while (e2)
  {
    if (e2->OutIdx >= 0 && e2->WindDelta != 0)
    {
      if (!eTmp) eTmp = e2;
      else if (eTmp->OutIdx == e2->OutIdx) eTmp = 0;        
    }
    e2 = e2->PrevInAEL;
  }
  if (!eTmp)
  {
    outrec->FirstLeft = 0;
    outrec->IsHole = false;
  }
  else
  {
    outrec->FirstLeft = m_PolyOuts[eTmp->OutIdx];
    outrec->IsHole = !outrec->FirstLeft->IsHole;
  }
}
//------------------------------------------------------------------------------

OutRec* GetLowermostRec(OutRec *outRec1, OutRec *outRec2)
{
  //work out which polygon fragment has the correct hole state ...
  if (!outRec1->BottomPt) 
    outRec1->BottomPt = GetBottomPt(outRec1->Pts);
  if (!outRec2->BottomPt) 
    outRec2->BottomPt = GetBottomPt(outRec2->Pts);
  OutPt *OutPt1 = outRec1->BottomPt;
  OutPt *OutPt2 = outRec2->BottomPt;
  if (OutPt1->Pt.Y > OutPt2->Pt.Y) return outRec1;
  else if (OutPt1->Pt.Y < OutPt2->Pt.Y) return outRec2;
  else if (OutPt1->Pt.X < OutPt2->Pt.X) return outRec1;
  else if (OutPt1->Pt.X > OutPt2->Pt.X) return outRec2;
  else if (OutPt1->Next == OutPt1) return outRec2;
  else if (OutPt2->Next == OutPt2) return outRec1;
  else if (FirstIsBottomPt(OutPt1, OutPt2)) return outRec1;
  else return outRec2;
}
//------------------------------------------------------------------------------

bool OutRec1RightOfOutRec2(OutRec* outRec1, OutRec* outRec2)
{
  do
  {
    outRec1 = outRec1->FirstLeft;
    if (outRec1 == outRec2) return true;
  } while (outRec1);
  return false;
}
//------------------------------------------------------------------------------

OutRec* Clipper::GetOutRec(int Idx)
{
  OutRec* outrec = m_PolyOuts[Idx];
  while (outrec != m_PolyOuts[outrec->Idx])
    outrec = m_PolyOuts[outrec->Idx];
  return outrec;
}
//------------------------------------------------------------------------------

void Clipper::AppendPolygon(TEdge *e1, TEdge *e2)
{
  //get the start and ends of both output polygons ...
  OutRec *outRec1 = m_PolyOuts[e1->OutIdx];
  OutRec *outRec2 = m_PolyOuts[e2->OutIdx];

  OutRec *holeStateRec;
  if (OutRec1RightOfOutRec2(outRec1, outRec2))
    holeStateRec = outRec2;
  else if (OutRec1RightOfOutRec2(outRec2, outRec1))
    holeStateRec = outRec1;
  else 
    holeStateRec = GetLowermostRec(outRec1, outRec2);

  //get the start and ends of both output polygons and
  //join e2 poly onto e1 poly and delete pointers to e2 ...

  OutPt* p1_lft = outRec1->Pts;
  OutPt* p1_rt = p1_lft->Prev;
  OutPt* p2_lft = outRec2->Pts;
  OutPt* p2_rt = p2_lft->Prev;

  //join e2 poly onto e1 poly and delete pointers to e2 ...
  if(  e1->Side == esLeft )
  {
    if(  e2->Side == esLeft )
    {
      //z y x a b c
      ReversePolyPtLinks(p2_lft);
      p2_lft->Next = p1_lft;
      p1_lft->Prev = p2_lft;
      p1_rt->Next = p2_rt;
      p2_rt->Prev = p1_rt;
      outRec1->Pts = p2_rt;
    } else
    {
      //x y z a b c
      p2_rt->Next = p1_lft;
      p1_lft->Prev = p2_rt;
      p2_lft->Prev = p1_rt;
      p1_rt->Next = p2_lft;
      outRec1->Pts = p2_lft;
    }
  } else
  {
    if(  e2->Side == esRight )
    {
      //a b c z y x
      ReversePolyPtLinks(p2_lft);
      p1_rt->Next = p2_rt;
      p2_rt->Prev = p1_rt;
      p2_lft->Next = p1_lft;
      p1_lft->Prev = p2_lft;
    } else
    {
      //a b c x y z
      p1_rt->Next = p2_lft;
      p2_lft->Prev = p1_rt;
      p1_lft->Prev = p2_rt;
      p2_rt->Next = p1_lft;
    }
  }

  outRec1->BottomPt = 0;
  if (holeStateRec == outRec2)
  {
    if (outRec2->FirstLeft != outRec1)
      outRec1->FirstLeft = outRec2->FirstLeft;
    outRec1->IsHole = outRec2->IsHole;
  }
  outRec2->Pts = 0;
  outRec2->BottomPt = 0;
  outRec2->FirstLeft = outRec1;

  int OKIdx = e1->OutIdx;
  int ObsoleteIdx = e2->OutIdx;

  e1->OutIdx = Unassigned; //nb: safe because we only get here via AddLocalMaxPoly
  e2->OutIdx = Unassigned;

  TEdge* e = m_ActiveEdges;
  while( e )
  {
    if( e->OutIdx == ObsoleteIdx )
    {
      e->OutIdx = OKIdx;
      e->Side = e1->Side;
      break;
    }
    e = e->NextInAEL;
  }

  outRec2->Idx = outRec1->Idx;
}
//------------------------------------------------------------------------------

OutPt* Clipper::AddOutPt(TEdge *e, const IntPoint &pt)
{
  if(  e->OutIdx < 0 )
  {
    OutRec *outRec = CreateOutRec();
    outRec->IsOpen = (e->WindDelta == 0);
    OutPt* newOp = new OutPt;
    outRec->Pts = newOp;
    newOp->Idx = outRec->Idx;
    newOp->Pt = pt;
    newOp->Next = newOp;
    newOp->Prev = newOp;
    if (!outRec->IsOpen)
      SetHoleState(e, outRec);
    e->OutIdx = outRec->Idx;
    return newOp;
  } else
  {
    OutRec *outRec = m_PolyOuts[e->OutIdx];
    //OutRec.Pts is the 'Left-most' point & OutRec.Pts.Prev is the 'Right-most'
    OutPt* op = outRec->Pts;

	bool ToFront = (e->Side == esLeft);
	if (ToFront && (pt == op->Pt)) return op;
    else if (!ToFront && (pt == op->Prev->Pt)) return op->Prev;

    OutPt* newOp = new OutPt;
    newOp->Idx = outRec->Idx;
    newOp->Pt = pt;
    newOp->Next = op;
    newOp->Prev = op->Prev;
    newOp->Prev->Next = newOp;
    op->Prev = newOp;
    if (ToFront) outRec->Pts = newOp;
    return newOp;
  }
}
//------------------------------------------------------------------------------

OutPt* Clipper::GetLastOutPt(TEdge *e)
{
	OutRec *outRec = m_PolyOuts[e->OutIdx];
	if (e->Side == esLeft)
		return outRec->Pts;
	else
		return outRec->Pts->Prev;
}
//------------------------------------------------------------------------------

void Clipper::ProcessHorizontals()
{
  TEdge* horzEdge;
  while (PopEdgeFromSEL(horzEdge))
    ProcessHorizontal(horzEdge);
}
//------------------------------------------------------------------------------

inline bool IsMinima(TEdge *e)
{
  return e  && (e->Prev->NextInLML != e) && (e->Next->NextInLML != e);
}
//------------------------------------------------------------------------------

inline bool IsMaxima(TEdge *e, const cInt Y)
{
  return e && e->Top.Y == Y && !e->NextInLML;
}
//------------------------------------------------------------------------------

inline bool IsIntermediate(TEdge *e, const cInt Y)
{
  return e->Top.Y == Y && e->NextInLML;
}
//------------------------------------------------------------------------------

TEdge *GetMaximaPair(TEdge *e)
{
  if ((e->Next->Top == e->Top) && !e->Next->NextInLML)
    return e->Next;
  else if ((e->Prev->Top == e->Top) && !e->Prev->NextInLML)
    return e->Prev;
  else return 0;
}
//------------------------------------------------------------------------------

TEdge *GetMaximaPairEx(TEdge *e)
{
  //as GetMaximaPair() but returns 0 if MaxPair isn't in AEL (unless it's horizontal)
  TEdge* result = GetMaximaPair(e);
  if (result && (result->OutIdx == Skip ||
    (result->NextInAEL == result->PrevInAEL && !IsHorizontal(*result)))) return 0;
  return result;
}
//------------------------------------------------------------------------------

void Clipper::SwapPositionsInSEL(TEdge *Edge1, TEdge *Edge2)
{
  if(  !( Edge1->NextInSEL ) &&  !( Edge1->PrevInSEL ) ) return;
  if(  !( Edge2->NextInSEL ) &&  !( Edge2->PrevInSEL ) ) return;

  if(  Edge1->NextInSEL == Edge2 )
  {
    TEdge* Next = Edge2->NextInSEL;
    if( Next ) Next->PrevInSEL = Edge1;
    TEdge* Prev = Edge1->PrevInSEL;
    if( Prev ) Prev->NextInSEL = Edge2;
    Edge2->PrevInSEL = Prev;
    Edge2->NextInSEL = Edge1;
    Edge1->PrevInSEL = Edge2;
    Edge1->NextInSEL = Next;
  }
  else if(  Edge2->NextInSEL == Edge1 )
  {
    TEdge* Next = Edge1->NextInSEL;
    if( Next ) Next->PrevInSEL = Edge2;
    TEdge* Prev = Edge2->PrevInSEL;
    if( Prev ) Prev->NextInSEL = Edge1;
    Edge1->PrevInSEL = Prev;
    Edge1->NextInSEL = Edge2;
    Edge2->PrevInSEL = Edge1;
    Edge2->NextInSEL = Next;
  }
  else
  {
    TEdge* Next = Edge1->NextInSEL;
    TEdge* Prev = Edge1->PrevInSEL;
    Edge1->NextInSEL = Edge2->NextInSEL;
    if( Edge1->NextInSEL ) Edge1->NextInSEL->PrevInSEL = Edge1;
    Edge1->PrevInSEL = Edge2->PrevInSEL;
    if( Edge1->PrevInSEL ) Edge1->PrevInSEL->NextInSEL = Edge1;
    Edge2->NextInSEL = Next;
    if( Edge2->NextInSEL ) Edge2->NextInSEL->PrevInSEL = Edge2;
    Edge2->PrevInSEL = Prev;
    if( Edge2->PrevInSEL ) Edge2->PrevInSEL->NextInSEL = Edge2;
  }

  if( !Edge1->PrevInSEL ) m_SortedEdges = Edge1;
  else if( !Edge2->PrevInSEL ) m_SortedEdges = Edge2;
}
//------------------------------------------------------------------------------

TEdge* GetNextInAEL(TEdge *e, Direction dir)
{
  return dir == dLeftToRight ? e->NextInAEL : e->PrevInAEL;
}
//------------------------------------------------------------------------------

void GetHorzDirection(TEdge& HorzEdge, Direction& Dir, cInt& Left, cInt& Right)
{
  if (HorzEdge.Bot.X < HorzEdge.Top.X)
  {
    Left = HorzEdge.Bot.X;
    Right = HorzEdge.Top.X;
    Dir = dLeftToRight;
  } else
  {
    Left = HorzEdge.Top.X;
    Right = HorzEdge.Bot.X;
    Dir = dRightToLeft;
  }
}
//------------------------------------------------------------------------

/*******************************************************************************
* Notes: Horizontal edges (HEs) at scanline intersections (ie at the Top or    *
* Bottom of a scanbeam) are processed as if layered. The order in which HEs    *
* are processed doesn't matter. HEs intersect with other HE Bot.Xs only [#]    *
* (or they could intersect with Top.Xs only, ie EITHER Bot.Xs OR Top.Xs),      *
* and with other non-horizontal edges [*]. Once these intersections are        *
* processed, intermediate HEs then 'promote' the Edge above (NextInLML) into   *
* the AEL. These 'promoted' edges may in turn intersect [%] with other HEs.    *
*******************************************************************************/

void Clipper::ProcessHorizontal(TEdge *horzEdge)
{
  Direction dir;
  cInt horzLeft, horzRight;
  bool IsOpen = (horzEdge->WindDelta == 0);

  GetHorzDirection(*horzEdge, dir, horzLeft, horzRight);

  TEdge* eLastHorz = horzEdge, *eMaxPair = 0;
  while (eLastHorz->NextInLML && IsHorizontal(*eLastHorz->NextInLML)) 
    eLastHorz = eLastHorz->NextInLML;
  if (!eLastHorz->NextInLML)
    eMaxPair = GetMaximaPair(eLastHorz);

  MaximaList::const_iterator maxIt;
  MaximaList::const_reverse_iterator maxRit;
  if (m_Maxima.size() > 0)
  {
      //get the first maxima in range (X) ...
      if (dir == dLeftToRight)
      {
          maxIt = m_Maxima.begin();
          while (maxIt != m_Maxima.end() && *maxIt <= horzEdge->Bot.X) maxIt++;
          if (maxIt != m_Maxima.end() && *maxIt >= eLastHorz->Top.X)
              maxIt = m_Maxima.end();
      }
      else
      {
          maxRit = m_Maxima.rbegin();
          while (maxRit != m_Maxima.rend() && *maxRit > horzEdge->Bot.X) maxRit++;
          if (maxRit != m_Maxima.rend() && *maxRit <= eLastHorz->Top.X)
              maxRit = m_Maxima.rend();
      }
  }

  OutPt* op1 = 0;

  for (;;) //loop through consec. horizontal edges
  {
		  
    bool IsLastHorz = (horzEdge == eLastHorz);
    TEdge* e = GetNextInAEL(horzEdge, dir);
    while(e)
    {

        //this code block inserts extra coords into horizontal edges (in output
        //polygons) whereever maxima touch these horizontal edges. This helps
        //'simplifying' polygons (ie if the Simplify property is set).
        if (m_Maxima.size() > 0)
        {
            if (dir == dLeftToRight)
            {
                while (maxIt != m_Maxima.end() && *maxIt < e->Curr.X) 
                {
                  if (horzEdge->OutIdx >= 0 && !IsOpen)
                    AddOutPt(horzEdge, IntPoint(*maxIt, horzEdge->Bot.Y));
                  maxIt++;
                }
            }
            else
            {
                while (maxRit != m_Maxima.rend() && *maxRit > e->Curr.X)
                {
                  if (horzEdge->OutIdx >= 0 && !IsOpen)
                    AddOutPt(horzEdge, IntPoint(*maxRit, horzEdge->Bot.Y));
                  maxRit++;
                }
            }
        };

        if ((dir == dLeftToRight && e->Curr.X > horzRight) ||
			(dir == dRightToLeft && e->Curr.X < horzLeft)) break;

		//Also break if we've got to the end of an intermediate horizontal edge ...
		//nb: Smaller Dx's are to the right of larger Dx's ABOVE the horizontal.
		if (e->Curr.X == horzEdge->Top.X && horzEdge->NextInLML && 
			e->Dx < horzEdge->NextInLML->Dx) break;

    if (horzEdge->OutIdx >= 0 && !IsOpen)  //note: may be done multiple times
		{
#ifdef use_xyz
			if (dir == dLeftToRight) SetZ(e->Curr, *horzEdge, *e);
			else SetZ(e->Curr, *e, *horzEdge);
#endif      
			op1 = AddOutPt(horzEdge, e->Curr);
			TEdge* eNextHorz = m_SortedEdges;
			while (eNextHorz)
			{
				if (eNextHorz->OutIdx >= 0 &&
					HorzSegmentsOverlap(horzEdge->Bot.X,
					horzEdge->Top.X, eNextHorz->Bot.X, eNextHorz->Top.X))
				{
                    OutPt* op2 = GetLastOutPt(eNextHorz);
                    AddJoin(op2, op1, eNextHorz->Top);
				}
				eNextHorz = eNextHorz->NextInSEL;
			}
			AddGhostJoin(op1, horzEdge->Bot);
		}
		
		//OK, so far we're still in range of the horizontal Edge  but make sure
        //we're at the last of consec. horizontals when matching with eMaxPair
        if(e == eMaxPair && IsLastHorz)
        {
          if (horzEdge->OutIdx >= 0)
            AddLocalMaxPoly(horzEdge, eMaxPair, horzEdge->Top);
          DeleteFromAEL(horzEdge);
          DeleteFromAEL(eMaxPair);
          return;
        }
        
		if(dir == dLeftToRight)
        {
          IntPoint Pt = IntPoint(e->Curr.X, horzEdge->Curr.Y);
          IntersectEdges(horzEdge, e, Pt);
        }
        else
        {
          IntPoint Pt = IntPoint(e->Curr.X, horzEdge->Curr.Y);
          IntersectEdges( e, horzEdge, Pt);
        }
        TEdge* eNext = GetNextInAEL(e, dir);
        SwapPositionsInAEL( horzEdge, e );
        e = eNext;
    } //end while(e)

	//Break out of loop if HorzEdge.NextInLML is not also horizontal ...
	if (!horzEdge->NextInLML || !IsHorizontal(*horzEdge->NextInLML)) break;

	UpdateEdgeIntoAEL(horzEdge);
    if (horzEdge->OutIdx >= 0) AddOutPt(horzEdge, horzEdge->Bot);
    GetHorzDirection(*horzEdge, dir, horzLeft, horzRight);

  } //end for (;;)

  if (horzEdge->OutIdx >= 0 && !op1)
  {
      op1 = GetLastOutPt(horzEdge);
      TEdge* eNextHorz = m_SortedEdges;
      while (eNextHorz)
      {
          if (eNextHorz->OutIdx >= 0 &&
              HorzSegmentsOverlap(horzEdge->Bot.X,
              horzEdge->Top.X, eNextHorz->Bot.X, eNextHorz->Top.X))
          {
              OutPt* op2 = GetLastOutPt(eNextHorz);
              AddJoin(op2, op1, eNextHorz->Top);
          }
          eNextHorz = eNextHorz->NextInSEL;
      }
      AddGhostJoin(op1, horzEdge->Top);
  }

  if (horzEdge->NextInLML)
  {
    if(horzEdge->OutIdx >= 0)
    {
      op1 = AddOutPt( horzEdge, horzEdge->Top);
      UpdateEdgeIntoAEL(horzEdge);
      if (horzEdge->WindDelta == 0) return;
      //nb: HorzEdge is no longer horizontal here
      TEdge* ePrev = horzEdge->PrevInAEL;
      TEdge* eNext = horzEdge->NextInAEL;
      if (ePrev && ePrev->Curr.X == horzEdge->Bot.X &&
        ePrev->Curr.Y == horzEdge->Bot.Y && ePrev->WindDelta != 0 &&
        (ePrev->OutIdx >= 0 && ePrev->Curr.Y > ePrev->Top.Y &&
        SlopesEqual(*horzEdge, *ePrev, m_UseFullRange)))
      {
        OutPt* op2 = AddOutPt(ePrev, horzEdge->Bot);
        AddJoin(op1, op2, horzEdge->Top);
      }
      else if (eNext && eNext->Curr.X == horzEdge->Bot.X &&
        eNext->Curr.Y == horzEdge->Bot.Y && eNext->WindDelta != 0 &&
        eNext->OutIdx >= 0 && eNext->Curr.Y > eNext->Top.Y &&
        SlopesEqual(*horzEdge, *eNext, m_UseFullRange))
      {
        OutPt* op2 = AddOutPt(eNext, horzEdge->Bot);
        AddJoin(op1, op2, horzEdge->Top);
      }
    }
    else
      UpdateEdgeIntoAEL(horzEdge); 
  }
  else
  {
    if (horzEdge->OutIdx >= 0) AddOutPt(horzEdge, horzEdge->Top);
    DeleteFromAEL(horzEdge);
  }
}
//------------------------------------------------------------------------------

bool Clipper::ProcessIntersections(const cInt topY)
{
  if( !m_ActiveEdges ) return true;
  try {
    BuildIntersectList(topY);
    size_t IlSize = m_IntersectList.size();
    if (IlSize == 0) return true;
    if (IlSize == 1 || FixupIntersectionOrder()) ProcessIntersectList();
    else return false;
  }
  catch(...) 
  {
    m_SortedEdges = 0;
    DisposeIntersectNodes();
    throw clipperException("ProcessIntersections error");
  }
  m_SortedEdges = 0;
  return true;
}
//------------------------------------------------------------------------------

void Clipper::DisposeIntersectNodes()
{
  for (size_t i = 0; i < m_IntersectList.size(); ++i )
    delete m_IntersectList[i];
  m_IntersectList.clear();
}
//------------------------------------------------------------------------------

void Clipper::BuildIntersectList(const cInt topY)
{
  if ( !m_ActiveEdges ) return;

  //prepare for sorting ...
  TEdge* e = m_ActiveEdges;
  m_SortedEdges = e;
  while( e )
  {
    e->PrevInSEL = e->PrevInAEL;
    e->NextInSEL = e->NextInAEL;
    e->Curr.X = TopX( *e, topY );
    e = e->NextInAEL;
  }

  //bubblesort ...
  bool isModified;
  do
  {
    isModified = false;
    e = m_SortedEdges;
    while( e->NextInSEL )
    {
      TEdge *eNext = e->NextInSEL;
      IntPoint Pt;
      if(e->Curr.X > eNext->Curr.X)
      {
        IntersectPoint(*e, *eNext, Pt);
        if (Pt.Y < topY) Pt = IntPoint(TopX(*e, topY), topY);
        IntersectNode * newNode = new IntersectNode;
        newNode->Edge1 = e;
        newNode->Edge2 = eNext;
        newNode->Pt = Pt;
        m_IntersectList.push_back(newNode);

        SwapPositionsInSEL(e, eNext);
        isModified = true;
      }
      else
        e = eNext;
    }
    if( e->PrevInSEL ) e->PrevInSEL->NextInSEL = 0;
    else break;
  }
  while ( isModified );
  m_SortedEdges = 0; //important
}
//------------------------------------------------------------------------------


void Clipper::ProcessIntersectList()
{
  for (size_t i = 0; i < m_IntersectList.size(); ++i)
  {
    IntersectNode* iNode = m_IntersectList[i];
    {
      IntersectEdges( iNode->Edge1, iNode->Edge2, iNode->Pt);
      SwapPositionsInAEL( iNode->Edge1 , iNode->Edge2 );
    }
    delete iNode;
  }
  m_IntersectList.clear();
}
//------------------------------------------------------------------------------

bool IntersectListSort(IntersectNode* node1, IntersectNode* node2)
{
  return node2->Pt.Y < node1->Pt.Y;
}
//------------------------------------------------------------------------------

inline bool EdgesAdjacent(const IntersectNode &inode)
{
  return (inode.Edge1->NextInSEL == inode.Edge2) ||
    (inode.Edge1->PrevInSEL == inode.Edge2);
}
//------------------------------------------------------------------------------

bool Clipper::FixupIntersectionOrder()
{
  //pre-condition: intersections are sorted Bottom-most first.
  //Now it's crucial that intersections are made only between adjacent edges,
  //so to ensure this the order of intersections may need adjusting ...
  CopyAELToSEL();
  std::sort(m_IntersectList.begin(), m_IntersectList.end(), IntersectListSort);
  size_t cnt = m_IntersectList.size();
  for (size_t i = 0; i < cnt; ++i) 
  {
    if (!EdgesAdjacent(*m_IntersectList[i]))
    {
      size_t j = i + 1;
      while (j < cnt && !EdgesAdjacent(*m_IntersectList[j])) j++;
      if (j == cnt)  return false;
      std::swap(m_IntersectList[i], m_IntersectList[j]);
    }
    SwapPositionsInSEL(m_IntersectList[i]->Edge1, m_IntersectList[i]->Edge2);
  }
  return true;
}
//------------------------------------------------------------------------------

void Clipper::DoMaxima(TEdge *e)
{
  TEdge* eMaxPair = GetMaximaPairEx(e);
  if (!eMaxPair)
  {
    if (e->OutIdx >= 0)
      AddOutPt(e, e->Top);
    DeleteFromAEL(e);
    return;
  }

  TEdge* eNext = e->NextInAEL;
  while(eNext && eNext != eMaxPair)
  {
    IntersectEdges(e, eNext, e->Top);
    SwapPositionsInAEL(e, eNext);
    eNext = e->NextInAEL;
  }

  if(e->OutIdx == Unassigned && eMaxPair->OutIdx == Unassigned)
  {
    DeleteFromAEL(e);
    DeleteFromAEL(eMaxPair);
  }
  else if( e->OutIdx >= 0 && eMaxPair->OutIdx >= 0 )
  {
    if (e->OutIdx >= 0) AddLocalMaxPoly(e, eMaxPair, e->Top);
    DeleteFromAEL(e);
    DeleteFromAEL(eMaxPair);
  }
#ifdef use_lines
  else if (e->WindDelta == 0)
  {
    if (e->OutIdx >= 0) 
    {
      AddOutPt(e, e->Top);
      e->OutIdx = Unassigned;
    }
    DeleteFromAEL(e);

    if (eMaxPair->OutIdx >= 0)
    {
      AddOutPt(eMaxPair, e->Top);
      eMaxPair->OutIdx = Unassigned;
    }
    DeleteFromAEL(eMaxPair);
  } 
#endif
  else throw clipperException("DoMaxima error");
}
//------------------------------------------------------------------------------

void Clipper::ProcessEdgesAtTopOfScanbeam(const cInt topY)
{
  TEdge* e = m_ActiveEdges;
  while( e )
  {
    //1. process maxima, treating them as if they're 'bent' horizontal edges,
    //   but exclude maxima with horizontal edges. nb: e can't be a horizontal.
    bool IsMaximaEdge = IsMaxima(e, topY);

    if(IsMaximaEdge)
    {
      TEdge* eMaxPair = GetMaximaPairEx(e);
      IsMaximaEdge = (!eMaxPair || !IsHorizontal(*eMaxPair));
    }

    if(IsMaximaEdge)
    {
      if (m_StrictSimple) m_Maxima.push_back(e->Top.X);
      TEdge* ePrev = e->PrevInAEL;
      DoMaxima(e);
      if( !ePrev ) e = m_ActiveEdges;
      else e = ePrev->NextInAEL;
    }
    else
    {
      //2. promote horizontal edges, otherwise update Curr.X and Curr.Y ...
      if (IsIntermediate(e, topY) && IsHorizontal(*e->NextInLML))
      {
        UpdateEdgeIntoAEL(e);
        if (e->OutIdx >= 0)
          AddOutPt(e, e->Bot);
        AddEdgeToSEL(e);
      } 
      else
      {
        e->Curr.X = TopX( *e, topY );
        e->Curr.Y = topY;
#ifdef use_xyz
		e->Curr.Z = topY == e->Top.Y ? e->Top.Z : (topY == e->Bot.Y ? e->Bot.Z : 0);
#endif
	  }

      //When StrictlySimple and 'e' is being touched by another edge, then
      //make sure both edges have a vertex here ...
      if (m_StrictSimple)
      {  
        TEdge* ePrev = e->PrevInAEL;
        if ((e->OutIdx >= 0) && (e->WindDelta != 0) && ePrev && (ePrev->OutIdx >= 0) &&
          (ePrev->Curr.X == e->Curr.X) && (ePrev->WindDelta != 0))
        {
          IntPoint pt = e->Curr;
#ifdef use_xyz
          SetZ(pt, *ePrev, *e);
#endif
          OutPt* op = AddOutPt(ePrev, pt);
          OutPt* op2 = AddOutPt(e, pt);
          AddJoin(op, op2, pt); //StrictlySimple (type-3) join
        }
      }

      e = e->NextInAEL;
    }
  }

  //3. Process horizontals at the Top of the scanbeam ...
  m_Maxima.sort();
  ProcessHorizontals();
  m_Maxima.clear();

  //4. Promote intermediate vertices ...
  e = m_ActiveEdges;
  while(e)
  {
    if(IsIntermediate(e, topY))
    {
      OutPt* op = 0;
      if( e->OutIdx >= 0 ) 
        op = AddOutPt(e, e->Top);
      UpdateEdgeIntoAEL(e);

      //if output polygons share an edge, they'll need joining later ...
      TEdge* ePrev = e->PrevInAEL;
      TEdge* eNext = e->NextInAEL;
      if (ePrev && ePrev->Curr.X == e->Bot.X &&
        ePrev->Curr.Y == e->Bot.Y && op &&
        ePrev->OutIdx >= 0 && ePrev->Curr.Y > ePrev->Top.Y &&
        SlopesEqual(e->Curr, e->Top, ePrev->Curr, ePrev->Top, m_UseFullRange) &&
        (e->WindDelta != 0) && (ePrev->WindDelta != 0))
      {
        OutPt* op2 = AddOutPt(ePrev, e->Bot);
        AddJoin(op, op2, e->Top);
      }
      else if (eNext && eNext->Curr.X == e->Bot.X &&
        eNext->Curr.Y == e->Bot.Y && op &&
        eNext->OutIdx >= 0 && eNext->Curr.Y > eNext->Top.Y &&
        SlopesEqual(e->Curr, e->Top, eNext->Curr, eNext->Top, m_UseFullRange) &&
        (e->WindDelta != 0) && (eNext->WindDelta != 0))
      {
        OutPt* op2 = AddOutPt(eNext, e->Bot);
        AddJoin(op, op2, e->Top);
      }
    }
    e = e->NextInAEL;
  }
}
//------------------------------------------------------------------------------

void Clipper::FixupOutPolyline(OutRec &outrec)
{
  OutPt *pp = outrec.Pts;
  OutPt *lastPP = pp->Prev;
  while (pp != lastPP)
  {
    pp = pp->Next;
    if (pp->Pt == pp->Prev->Pt)
    {
      if (pp == lastPP) lastPP = pp->Prev;
      OutPt *tmpPP = pp->Prev;
      tmpPP->Next = pp->Next;
      pp->Next->Prev = tmpPP;
      delete pp;
      pp = tmpPP;
    }
  }

  if (pp == pp->Prev)
  {
    DisposeOutPts(pp);
    outrec.Pts = 0;
    return;
  }
}
//------------------------------------------------------------------------------

void Clipper::FixupOutPolygon(OutRec &outrec)
{
    //FixupOutPolygon() - removes duplicate points and simplifies consecutive
    //parallel edges by removing the middle vertex.
    OutPt *lastOK = 0;
    outrec.BottomPt = 0;
    OutPt *pp = outrec.Pts;
    bool preserveCol = m_PreserveCollinear || m_StrictSimple;

    for (;;)
    {
        if (pp->Prev == pp || pp->Prev == pp->Next)
        {
            DisposeOutPts(pp);
            outrec.Pts = 0;
            return;
        }

        //test for duplicate points and collinear edges ...
        if ((pp->Pt == pp->Next->Pt) || (pp->Pt == pp->Prev->Pt) ||
            (SlopesEqual(pp->Prev->Pt, pp->Pt, pp->Next->Pt, m_UseFullRange) &&
            (!preserveCol || !Pt2IsBetweenPt1AndPt3(pp->Prev->Pt, pp->Pt, pp->Next->Pt))))
        {
            lastOK = 0;
            OutPt *tmp = pp;
            pp->Prev->Next = pp->Next;
            pp->Next->Prev = pp->Prev;
            pp = pp->Prev;
            delete tmp;
        }
        else if (pp == lastOK) break;
        else
        {
            if (!lastOK) lastOK = pp;
            pp = pp->Next;
        }
    }
    outrec.Pts = pp;
}
//------------------------------------------------------------------------------

int PointCount(OutPt *Pts)
{
    if (!Pts) return 0;
    int result = 0;
    OutPt* p = Pts;
    do
    {
        result++;
        p = p->Next;
    }
    while (p != Pts);
    return result;
}
//------------------------------------------------------------------------------

void Clipper::BuildResult(Paths &polys)
{
  polys.reserve(m_PolyOuts.size());
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i)
  {
    if (!m_PolyOuts[i]->Pts) continue;
    Path pg;
    OutPt* p = m_PolyOuts[i]->Pts->Prev;
    int cnt = PointCount(p);
    if (cnt < 2) continue;
    pg.reserve(cnt);
    for (int i = 0; i < cnt; ++i)
    {
      pg.push_back(p->Pt);
      p = p->Prev;
    }
    polys.push_back(pg);
  }
}
//------------------------------------------------------------------------------

void Clipper::BuildResult2(PolyTree& polytree)
{
    polytree.Clear();
    polytree.AllNodes.reserve(m_PolyOuts.size());
    //add each output polygon/contour to polytree ...
    for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); i++)
    {
        OutRec* outRec = m_PolyOuts[i];
        int cnt = PointCount(outRec->Pts);
        if ((outRec->IsOpen && cnt < 2) || (!outRec->IsOpen && cnt < 3)) continue;
        FixHoleLinkage(*outRec);
        PolyNode* pn = new PolyNode();
        //nb: polytree takes ownership of all the PolyNodes
        polytree.AllNodes.push_back(pn);
        outRec->PolyNd = pn;
        pn->Parent = 0;
        pn->Index = 0;
        pn->Contour.reserve(cnt);
        OutPt *op = outRec->Pts->Prev;
        for (int j = 0; j < cnt; j++)
        {
            pn->Contour.push_back(op->Pt);
            op = op->Prev;
        }
    }

    //fixup PolyNode links etc ...
    polytree.Childs.reserve(m_PolyOuts.size());
    for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); i++)
    {
        OutRec* outRec = m_PolyOuts[i];
        if (!outRec->PolyNd) continue;
        if (outRec->IsOpen) 
        {
          outRec->PolyNd->m_IsOpen = true;
          polytree.AddChild(*outRec->PolyNd);
        }
        else if (outRec->FirstLeft && outRec->FirstLeft->PolyNd) 
          outRec->FirstLeft->PolyNd->AddChild(*outRec->PolyNd);
        else
          polytree.AddChild(*outRec->PolyNd);
    }
}
//------------------------------------------------------------------------------

void SwapIntersectNodes(IntersectNode &int1, IntersectNode &int2)
{
  //just swap the contents (because fIntersectNodes is a single-linked-list)
  IntersectNode inode = int1; //gets a copy of Int1
  int1.Edge1 = int2.Edge1;
  int1.Edge2 = int2.Edge2;
  int1.Pt = int2.Pt;
  int2.Edge1 = inode.Edge1;
  int2.Edge2 = inode.Edge2;
  int2.Pt = inode.Pt;
}
//------------------------------------------------------------------------------

inline bool E2InsertsBeforeE1(TEdge &e1, TEdge &e2)
{
  if (e2.Curr.X == e1.Curr.X) 
  {
    if (e2.Top.Y > e1.Top.Y)
      return e2.Top.X < TopX(e1, e2.Top.Y); 
      else return e1.Top.X > TopX(e2, e1.Top.Y);
  } 
  else return e2.Curr.X < e1.Curr.X;
}
//------------------------------------------------------------------------------

bool GetOverlap(const cInt a1, const cInt a2, const cInt b1, const cInt b2, 
    cInt& Left, cInt& Right)
{
  if (a1 < a2)
  {
    if (b1 < b2) {Left = std::max(a1,b1); Right = std::min(a2,b2);}
    else {Left = std::max(a1,b2); Right = std::min(a2,b1);}
  } 
  else
  {
    if (b1 < b2) {Left = std::max(a2,b1); Right = std::min(a1,b2);}
    else {Left = std::max(a2,b2); Right = std::min(a1,b1);}
  }
  return Left < Right;
}
//------------------------------------------------------------------------------

inline void UpdateOutPtIdxs(OutRec& outrec)
{  
  OutPt* op = outrec.Pts;
  do
  {
    op->Idx = outrec.Idx;
    op = op->Prev;
  }
  while(op != outrec.Pts);
}
//------------------------------------------------------------------------------

void Clipper::InsertEdgeIntoAEL(TEdge *edge, TEdge* startEdge)
{
  if(!m_ActiveEdges)
  {
    edge->PrevInAEL = 0;
    edge->NextInAEL = 0;
    m_ActiveEdges = edge;
  }
  else if(!startEdge && E2InsertsBeforeE1(*m_ActiveEdges, *edge))
  {
      edge->PrevInAEL = 0;
      edge->NextInAEL = m_ActiveEdges;
      m_ActiveEdges->PrevInAEL = edge;
      m_ActiveEdges = edge;
  } 
  else
  {
    if(!startEdge) startEdge = m_ActiveEdges;
    while(startEdge->NextInAEL  && 
      !E2InsertsBeforeE1(*startEdge->NextInAEL , *edge))
        startEdge = startEdge->NextInAEL;
    edge->NextInAEL = startEdge->NextInAEL;
    if(startEdge->NextInAEL) startEdge->NextInAEL->PrevInAEL = edge;
    edge->PrevInAEL = startEdge;
    startEdge->NextInAEL = edge;
  }
}
//----------------------------------------------------------------------

OutPt* DupOutPt(OutPt* outPt, bool InsertAfter)
{
  OutPt* result = new OutPt;
  result->Pt = outPt->Pt;
  result->Idx = outPt->Idx;
  if (InsertAfter)
  {
    result->Next = outPt->Next;
    result->Prev = outPt;
    outPt->Next->Prev = result;
    outPt->Next = result;
  } 
  else
  {
    result->Prev = outPt->Prev;
    result->Next = outPt;
    outPt->Prev->Next = result;
    outPt->Prev = result;
  }
  return result;
}
//------------------------------------------------------------------------------

bool JoinHorz(OutPt* op1, OutPt* op1b, OutPt* op2, OutPt* op2b,
  const IntPoint Pt, bool DiscardLeft)
{
  Direction Dir1 = (op1->Pt.X > op1b->Pt.X ? dRightToLeft : dLeftToRight);
  Direction Dir2 = (op2->Pt.X > op2b->Pt.X ? dRightToLeft : dLeftToRight);
  if (Dir1 == Dir2) return false;

  //When DiscardLeft, we want Op1b to be on the Left of Op1, otherwise we
  //want Op1b to be on the Right. (And likewise with Op2 and Op2b.)
  //So, to facilitate this while inserting Op1b and Op2b ...
  //when DiscardLeft, make sure we're AT or RIGHT of Pt before adding Op1b,
  //otherwise make sure we're AT or LEFT of Pt. (Likewise with Op2b.)
  if (Dir1 == dLeftToRight) 
  {
    while (op1->Next->Pt.X <= Pt.X && 
      op1->Next->Pt.X >= op1->Pt.X && op1->Next->Pt.Y == Pt.Y)  
        op1 = op1->Next;
    if (DiscardLeft && (op1->Pt.X != Pt.X)) op1 = op1->Next;
    op1b = DupOutPt(op1, !DiscardLeft);
    if (op1b->Pt != Pt) 
    {
      op1 = op1b;
      op1->Pt = Pt;
      op1b = DupOutPt(op1, !DiscardLeft);
    }
  } 
  else
  {
    while (op1->Next->Pt.X >= Pt.X && 
      op1->Next->Pt.X <= op1->Pt.X && op1->Next->Pt.Y == Pt.Y) 
        op1 = op1->Next;
    if (!DiscardLeft && (op1->Pt.X != Pt.X)) op1 = op1->Next;
    op1b = DupOutPt(op1, DiscardLeft);
    if (op1b->Pt != Pt)
    {
      op1 = op1b;
      op1->Pt = Pt;
      op1b = DupOutPt(op1, DiscardLeft);
    }
  }

  if (Dir2 == dLeftToRight)
  {
    while (op2->Next->Pt.X <= Pt.X && 
      op2->Next->Pt.X >= op2->Pt.X && op2->Next->Pt.Y == Pt.Y)
        op2 = op2->Next;
    if (DiscardLeft && (op2->Pt.X != Pt.X)) op2 = op2->Next;
    op2b = DupOutPt(op2, !DiscardLeft);
    if (op2b->Pt != Pt)
    {
      op2 = op2b;
      op2->Pt = Pt;
      op2b = DupOutPt(op2, !DiscardLeft);
    };
  } else
  {
    while (op2->Next->Pt.X >= Pt.X && 
      op2->Next->Pt.X <= op2->Pt.X && op2->Next->Pt.Y == Pt.Y) 
        op2 = op2->Next;
    if (!DiscardLeft && (op2->Pt.X != Pt.X)) op2 = op2->Next;
    op2b = DupOutPt(op2, DiscardLeft);
    if (op2b->Pt != Pt)
    {
      op2 = op2b;
      op2->Pt = Pt;
      op2b = DupOutPt(op2, DiscardLeft);
    };
  };

  if ((Dir1 == dLeftToRight) == DiscardLeft)
  {
    op1->Prev = op2;
    op2->Next = op1;
    op1b->Next = op2b;
    op2b->Prev = op1b;
  }
  else
  {
    op1->Next = op2;
    op2->Prev = op1;
    op1b->Prev = op2b;
    op2b->Next = op1b;
  }
  return true;
}
//------------------------------------------------------------------------------

bool Clipper::JoinPoints(Join *j, OutRec* outRec1, OutRec* outRec2)
{
  OutPt *op1 = j->OutPt1, *op1b;
  OutPt *op2 = j->OutPt2, *op2b;

  //There are 3 kinds of joins for output polygons ...
  //1. Horizontal joins where Join.OutPt1 & Join.OutPt2 are vertices anywhere
  //along (horizontal) collinear edges (& Join.OffPt is on the same horizontal).
  //2. Non-horizontal joins where Join.OutPt1 & Join.OutPt2 are at the same
  //location at the Bottom of the overlapping segment (& Join.OffPt is above).
  //3. StrictSimple joins where edges touch but are not collinear and where
  //Join.OutPt1, Join.OutPt2 & Join.OffPt all share the same point.
  bool isHorizontal = (j->OutPt1->Pt.Y == j->OffPt.Y);

  if (isHorizontal  && (j->OffPt == j->OutPt1->Pt) &&
  (j->OffPt == j->OutPt2->Pt))
  {
    //Strictly Simple join ...
    if (outRec1 != outRec2) return false;
    op1b = j->OutPt1->Next;
    while (op1b != op1 && (op1b->Pt == j->OffPt)) 
      op1b = op1b->Next;
    bool reverse1 = (op1b->Pt.Y > j->OffPt.Y);
    op2b = j->OutPt2->Next;
    while (op2b != op2 && (op2b->Pt == j->OffPt)) 
      op2b = op2b->Next;
    bool reverse2 = (op2b->Pt.Y > j->OffPt.Y);
    if (reverse1 == reverse2) return false;
    if (reverse1)
    {
      op1b = DupOutPt(op1, false);
      op2b = DupOutPt(op2, true);
      op1->Prev = op2;
      op2->Next = op1;
      op1b->Next = op2b;
      op2b->Prev = op1b;
      j->OutPt1 = op1;
      j->OutPt2 = op1b;
      return true;
    } else
    {
      op1b = DupOutPt(op1, true);
      op2b = DupOutPt(op2, false);
      op1->Next = op2;
      op2->Prev = op1;
      op1b->Prev = op2b;
      op2b->Next = op1b;
      j->OutPt1 = op1;
      j->OutPt2 = op1b;
      return true;
    }
  } 
  else if (isHorizontal)
  {
    //treat horizontal joins differently to non-horizontal joins since with
    //them we're not yet sure where the overlapping is. OutPt1.Pt & OutPt2.Pt
    //may be anywhere along the horizontal edge.
    op1b = op1;
    while (op1->Prev->Pt.Y == op1->Pt.Y && op1->Prev != op1b && op1->Prev != op2)
      op1 = op1->Prev;
    while (op1b->Next->Pt.Y == op1b->Pt.Y && op1b->Next != op1 && op1b->Next != op2)
      op1b = op1b->Next;
    if (op1b->Next == op1 || op1b->Next == op2) return false; //a flat 'polygon'

    op2b = op2;
    while (op2->Prev->Pt.Y == op2->Pt.Y && op2->Prev != op2b && op2->Prev != op1b)
      op2 = op2->Prev;
    while (op2b->Next->Pt.Y == op2b->Pt.Y && op2b->Next != op2 && op2b->Next != op1)
      op2b = op2b->Next;
    if (op2b->Next == op2 || op2b->Next == op1) return false; //a flat 'polygon'

    cInt Left, Right;
    //Op1 --> Op1b & Op2 --> Op2b are the extremites of the horizontal edges
    if (!GetOverlap(op1->Pt.X, op1b->Pt.X, op2->Pt.X, op2b->Pt.X, Left, Right))
      return false;

    //DiscardLeftSide: when overlapping edges are joined, a spike will created
    //which needs to be cleaned up. However, we don't want Op1 or Op2 caught up
    //on the discard Side as either may still be needed for other joins ...
    IntPoint Pt;
    bool DiscardLeftSide;
    if (op1->Pt.X >= Left && op1->Pt.X <= Right) 
    {
      Pt = op1->Pt; DiscardLeftSide = (op1->Pt.X > op1b->Pt.X);
    } 
    else if (op2->Pt.X >= Left&& op2->Pt.X <= Right) 
    {
      Pt = op2->Pt; DiscardLeftSide = (op2->Pt.X > op2b->Pt.X);
    } 
    else if (op1b->Pt.X >= Left && op1b->Pt.X <= Right)
    {
      Pt = op1b->Pt; DiscardLeftSide = op1b->Pt.X > op1->Pt.X;
    } 
    else
    {
      Pt = op2b->Pt; DiscardLeftSide = (op2b->Pt.X > op2->Pt.X);
    }
    j->OutPt1 = op1; j->OutPt2 = op2;
    return JoinHorz(op1, op1b, op2, op2b, Pt, DiscardLeftSide);
  } else
  {
    //nb: For non-horizontal joins ...
    //    1. Jr.OutPt1.Pt.Y == Jr.OutPt2.Pt.Y
    //    2. Jr.OutPt1.Pt > Jr.OffPt.Y

    //make sure the polygons are correctly oriented ...
    op1b = op1->Next;
    while ((op1b->Pt == op1->Pt) && (op1b != op1)) op1b = op1b->Next;
    bool Reverse1 = ((op1b->Pt.Y > op1->Pt.Y) ||
      !SlopesEqual(op1->Pt, op1b->Pt, j->OffPt, m_UseFullRange));
    if (Reverse1)
    {
      op1b = op1->Prev;
      while ((op1b->Pt == op1->Pt) && (op1b != op1)) op1b = op1b->Prev;
      if ((op1b->Pt.Y > op1->Pt.Y) ||
        !SlopesEqual(op1->Pt, op1b->Pt, j->OffPt, m_UseFullRange)) return false;
    };
    op2b = op2->Next;
    while ((op2b->Pt == op2->Pt) && (op2b != op2))op2b = op2b->Next;
    bool Reverse2 = ((op2b->Pt.Y > op2->Pt.Y) ||
      !SlopesEqual(op2->Pt, op2b->Pt, j->OffPt, m_UseFullRange));
    if (Reverse2)
    {
      op2b = op2->Prev;
      while ((op2b->Pt == op2->Pt) && (op2b != op2)) op2b = op2b->Prev;
      if ((op2b->Pt.Y > op2->Pt.Y) ||
        !SlopesEqual(op2->Pt, op2b->Pt, j->OffPt, m_UseFullRange)) return false;
    }

    if ((op1b == op1) || (op2b == op2) || (op1b == op2b) ||
      ((outRec1 == outRec2) && (Reverse1 == Reverse2))) return false;

    if (Reverse1)
    {
      op1b = DupOutPt(op1, false);
      op2b = DupOutPt(op2, true);
      op1->Prev = op2;
      op2->Next = op1;
      op1b->Next = op2b;
      op2b->Prev = op1b;
      j->OutPt1 = op1;
      j->OutPt2 = op1b;
      return true;
    } else
    {
      op1b = DupOutPt(op1, true);
      op2b = DupOutPt(op2, false);
      op1->Next = op2;
      op2->Prev = op1;
      op1b->Prev = op2b;
      op2b->Next = op1b;
      j->OutPt1 = op1;
      j->OutPt2 = op1b;
      return true;
    }
  }
}
//----------------------------------------------------------------------

static OutRec* ParseFirstLeft(OutRec* FirstLeft)
{
  while (FirstLeft && !FirstLeft->Pts)
    FirstLeft = FirstLeft->FirstLeft;
  return FirstLeft;
}
//------------------------------------------------------------------------------

void Clipper::FixupFirstLefts1(OutRec* OldOutRec, OutRec* NewOutRec)
{ 
  //tests if NewOutRec contains the polygon before reassigning FirstLeft
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i)
  {
    OutRec* outRec = m_PolyOuts[i];
    OutRec* firstLeft = ParseFirstLeft(outRec->FirstLeft);
    if (outRec->Pts  && firstLeft == OldOutRec)
    {
      if (Poly2ContainsPoly1(outRec->Pts, NewOutRec->Pts))
        outRec->FirstLeft = NewOutRec;
    }
  }
}
//----------------------------------------------------------------------

void Clipper::FixupFirstLefts2(OutRec* InnerOutRec, OutRec* OuterOutRec)
{
  //A polygon has split into two such that one is now the inner of the other.
  //It's possible that these polygons now wrap around other polygons, so check
  //every polygon that's also contained by OuterOutRec's FirstLeft container
  //(including 0) to see if they've become inner to the new inner polygon ...
  OutRec* orfl = OuterOutRec->FirstLeft;
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i)
  {
    OutRec* outRec = m_PolyOuts[i];

    if (!outRec->Pts || outRec == OuterOutRec || outRec == InnerOutRec)
      continue;
    OutRec* firstLeft = ParseFirstLeft(outRec->FirstLeft);
    if (firstLeft != orfl && firstLeft != InnerOutRec && firstLeft != OuterOutRec)
      continue;
    if (Poly2ContainsPoly1(outRec->Pts, InnerOutRec->Pts))
      outRec->FirstLeft = InnerOutRec;
    else if (Poly2ContainsPoly1(outRec->Pts, OuterOutRec->Pts))
      outRec->FirstLeft = OuterOutRec;
    else if (outRec->FirstLeft == InnerOutRec || outRec->FirstLeft == OuterOutRec)
      outRec->FirstLeft = orfl;
  }
}
//----------------------------------------------------------------------
void Clipper::FixupFirstLefts3(OutRec* OldOutRec, OutRec* NewOutRec)
{
  //reassigns FirstLeft WITHOUT testing if NewOutRec contains the polygon
  for (PolyOutList::size_type i = 0; i < m_PolyOuts.size(); ++i)
  {
    OutRec* outRec = m_PolyOuts[i];
    OutRec* firstLeft = ParseFirstLeft(outRec->FirstLeft);
    if (outRec->Pts && firstLeft == OldOutRec)
      outRec->FirstLeft = NewOutRec;
  }
}
//----------------------------------------------------------------------

void Clipper::JoinCommonEdges()
{
  for (JoinList::size_type i = 0; i < m_Joins.size(); i++)
  {
    Join* join = m_Joins[i];

    OutRec *outRec1 = GetOutRec(join->OutPt1->Idx);
    OutRec *outRec2 = GetOutRec(join->OutPt2->Idx);

    if (!outRec1->Pts || !outRec2->Pts) continue;
    if (outRec1->IsOpen || outRec2->IsOpen) continue;

    //get the polygon fragment with the correct hole state (FirstLeft)
    //before calling JoinPoints() ...
    OutRec *holeStateRec;
    if (outRec1 == outRec2) holeStateRec = outRec1;
    else if (OutRec1RightOfOutRec2(outRec1, outRec2)) holeStateRec = outRec2;
    else if (OutRec1RightOfOutRec2(outRec2, outRec1)) holeStateRec = outRec1;
    else holeStateRec = GetLowermostRec(outRec1, outRec2);

    if (!JoinPoints(join, outRec1, outRec2)) continue;

    if (outRec1 == outRec2)
    {
      //instead of joining two polygons, we've just created a new one by
      //splitting one polygon into two.
      outRec1->Pts = join->OutPt1;
      outRec1->BottomPt = 0;
      outRec2 = CreateOutRec();
      outRec2->Pts = join->OutPt2;

      //update all OutRec2.Pts Idx's ...
      UpdateOutPtIdxs(*outRec2);

      if (Poly2ContainsPoly1(outRec2->Pts, outRec1->Pts))
      {
        //outRec1 contains outRec2 ...
        outRec2->IsHole = !outRec1->IsHole;
        outRec2->FirstLeft = outRec1;

        if (m_UsingPolyTree) FixupFirstLefts2(outRec2, outRec1);

        if ((outRec2->IsHole ^ m_ReverseOutput) == (Area(*outRec2) > 0))
          ReversePolyPtLinks(outRec2->Pts);
            
      } else if (Poly2ContainsPoly1(outRec1->Pts, outRec2->Pts))
      {
        //outRec2 contains outRec1 ...
        outRec2->IsHole = outRec1->IsHole;
        outRec1->IsHole = !outRec2->IsHole;
        outRec2->FirstLeft = outRec1->FirstLeft;
        outRec1->FirstLeft = outRec2;

        if (m_UsingPolyTree) FixupFirstLefts2(outRec1, outRec2);

        if ((outRec1->IsHole ^ m_ReverseOutput) == (Area(*outRec1) > 0))
          ReversePolyPtLinks(outRec1->Pts);
      } 
      else
      {
        //the 2 polygons are completely separate ...
        outRec2->IsHole = outRec1->IsHole;
        outRec2->FirstLeft = outRec1->FirstLeft;

        //fixup FirstLeft pointers that may need reassigning to OutRec2
        if (m_UsingPolyTree) FixupFirstLefts1(outRec1, outRec2);
      }
     
    } else
    {
      //joined 2 polygons together ...

      outRec2->Pts = 0;
      outRec2->BottomPt = 0;
      outRec2->Idx = outRec1->Idx;

      outRec1->IsHole = holeStateRec->IsHole;
      if (holeStateRec == outRec2) 
        outRec1->FirstLeft = outRec2->FirstLeft;
      outRec2->FirstLeft = outRec1;

      if (m_UsingPolyTree) FixupFirstLefts3(outRec2, outRec1);
    }
  }
}

//------------------------------------------------------------------------------
// ClipperOffset support functions ...
//------------------------------------------------------------------------------

DoublePoint GetUnitNormal(const IntPoint &pt1, const IntPoint &pt2)
{
  if(pt2.X == pt1.X && pt2.Y == pt1.Y) 
    return DoublePoint(0, 0);

  double Dx = (double)(pt2.X - pt1.X);
  double dy = (double)(pt2.Y - pt1.Y);
  double f = 1 *1.0/ std::sqrt( Dx*Dx + dy*dy );
  Dx *= f;
  dy *= f;
  return DoublePoint(dy, -Dx);
}

//------------------------------------------------------------------------------
// ClipperOffset class
//------------------------------------------------------------------------------

ClipperOffset::ClipperOffset(double miterLimit, double arcTolerance)
{
  this->MiterLimit = miterLimit;
  this->ArcTolerance = arcTolerance;
  m_lowest.X = -1;
}
//------------------------------------------------------------------------------

ClipperOffset::~ClipperOffset()
{
  Clear();
}
//------------------------------------------------------------------------------

void ClipperOffset::Clear()
{
  for (int i = 0; i < m_polyNodes.ChildCount(); ++i)
    delete m_polyNodes.Childs[i];
  m_polyNodes.Childs.clear();
  m_lowest.X = -1;
}
//------------------------------------------------------------------------------

void ClipperOffset::AddPath(const Path& path, JoinType joinType, EndType endType)
{
  int highI = (int)path.size() - 1;
  if (highI < 0) return;
  PolyNode* newNode = new PolyNode();
  newNode->m_jointype = joinType;
  newNode->m_endtype = endType;

  //strip duplicate points from path and also get index to the lowest point ...
  if (endType == etClosedLine || endType == etClosedPolygon)
    while (highI > 0 && path[0] == path[highI]) highI--;
  newNode->Contour.reserve(highI + 1);
  newNode->Contour.push_back(path[0]);
  int j = 0, k = 0;
  for (int i = 1; i <= highI; i++)
    if (newNode->Contour[j] != path[i])
    {
      j++;
      newNode->Contour.push_back(path[i]);
      if (path[i].Y > newNode->Contour[k].Y ||
        (path[i].Y == newNode->Contour[k].Y &&
        path[i].X < newNode->Contour[k].X)) k = j;
    }
  if (endType == etClosedPolygon && j < 2)
  {
    delete newNode;
    return;
  }
  m_polyNodes.AddChild(*newNode);

  //if this path's lowest pt is lower than all the others then update m_lowest
  if (endType != etClosedPolygon) return;
  if (m_lowest.X < 0)
    m_lowest = IntPoint(m_polyNodes.ChildCount() - 1, k);
  else
  {
    IntPoint ip = m_polyNodes.Childs[(int)m_lowest.X]->Contour[(int)m_lowest.Y];
    if (newNode->Contour[k].Y > ip.Y ||
      (newNode->Contour[k].Y == ip.Y &&
      newNode->Contour[k].X < ip.X))
      m_lowest = IntPoint(m_polyNodes.ChildCount() - 1, k);
  }
}
//------------------------------------------------------------------------------

void ClipperOffset::AddPaths(const Paths& paths, JoinType joinType, EndType endType)
{
  for (Paths::size_type i = 0; i < paths.size(); ++i)
    AddPath(paths[i], joinType, endType);
}
//------------------------------------------------------------------------------

void ClipperOffset::FixOrientations()
{
  //fixup orientations of all closed paths if the orientation of the
  //closed path with the lowermost vertex is wrong ...
  if (m_lowest.X >= 0 && 
    !Orientation(m_polyNodes.Childs[(int)m_lowest.X]->Contour))
  {
    for (int i = 0; i < m_polyNodes.ChildCount(); ++i)
    {
      PolyNode& node = *m_polyNodes.Childs[i];
      if (node.m_endtype == etClosedPolygon ||
        (node.m_endtype == etClosedLine && Orientation(node.Contour)))
          ReversePath(node.Contour);
    }
  } else
  {
    for (int i = 0; i < m_polyNodes.ChildCount(); ++i)
    {
      PolyNode& node = *m_polyNodes.Childs[i];
      if (node.m_endtype == etClosedLine && !Orientation(node.Contour))
        ReversePath(node.Contour);
    }
  }
}
//------------------------------------------------------------------------------

void ClipperOffset::Execute(Paths& solution, double delta)
{
  solution.clear();
  FixOrientations();
  DoOffset(delta);
  
  //now clean up 'corners' ...
  Clipper clpr;
  clpr.AddPaths(m_destPolys, ptSubject, true);
  if (delta > 0)
  {
    clpr.Execute(ctUnion, solution, pftPositive, pftPositive);
  }
  else
  {
    IntRect r = clpr.GetBounds();
    Path outer(4);
    outer[0] = IntPoint(r.left - 10, r.bottom + 10);
    outer[1] = IntPoint(r.right + 10, r.bottom + 10);
    outer[2] = IntPoint(r.right + 10, r.top - 10);
    outer[3] = IntPoint(r.left - 10, r.top - 10);

    clpr.AddPath(outer, ptSubject, true);
    clpr.ReverseSolution(true);
    clpr.Execute(ctUnion, solution, pftNegative, pftNegative);
    if (solution.size() > 0) solution.erase(solution.begin());
  }
}
//------------------------------------------------------------------------------

void ClipperOffset::Execute(PolyTree& solution, double delta)
{
  solution.Clear();
  FixOrientations();
  DoOffset(delta);

  //now clean up 'corners' ...
  Clipper clpr;
  clpr.AddPaths(m_destPolys, ptSubject, true);
  if (delta > 0)
  {
    clpr.Execute(ctUnion, solution, pftPositive, pftPositive);
  }
  else
  {
    IntRect r = clpr.GetBounds();
    Path outer(4);
    outer[0] = IntPoint(r.left - 10, r.bottom + 10);
    outer[1] = IntPoint(r.right + 10, r.bottom + 10);
    outer[2] = IntPoint(r.right + 10, r.top - 10);
    outer[3] = IntPoint(r.left - 10, r.top - 10);

    clpr.AddPath(outer, ptSubject, true);
    clpr.ReverseSolution(true);
    clpr.Execute(ctUnion, solution, pftNegative, pftNegative);
    //remove the outer PolyNode rectangle ...
    if (solution.ChildCount() == 1 && solution.Childs[0]->ChildCount() > 0)
    {
      PolyNode* outerNode = solution.Childs[0];
      solution.Childs.reserve(outerNode->ChildCount());
      solution.Childs[0] = outerNode->Childs[0];
      solution.Childs[0]->Parent = outerNode->Parent;
      for (int i = 1; i < outerNode->ChildCount(); ++i)
        solution.AddChild(*outerNode->Childs[i]);
    }
    else
      solution.Clear();
  }
}
//------------------------------------------------------------------------------

void ClipperOffset::DoOffset(double delta)
{
  m_destPolys.clear();
  m_delta = delta;

  //if Zero offset, just copy any CLOSED polygons to m_p and return ...
  if (NEAR_ZERO(delta)) 
  {
    m_destPolys.reserve(m_polyNodes.ChildCount());
    for (int i = 0; i < m_polyNodes.ChildCount(); i++)
    {
      PolyNode& node = *m_polyNodes.Childs[i];
      if (node.m_endtype == etClosedPolygon)
        m_destPolys.push_back(node.Contour);
    }
    return;
  }

  //see offset_triginometry3.svg in the documentation folder ...
  if (MiterLimit > 2) m_miterLim = 2/(MiterLimit * MiterLimit);
  else m_miterLim = 0.5;

  double y;
  if (ArcTolerance <= 0.0) y = def_arc_tolerance;
  else if (ArcTolerance > std::fabs(delta) * def_arc_tolerance) 
    y = std::fabs(delta) * def_arc_tolerance;
  else y = ArcTolerance;
  //see offset_triginometry2.svg in the documentation folder ...
  double steps = pi / std::acos(1 - y / std::fabs(delta));
  if (steps > std::fabs(delta) * pi) 
    steps = std::fabs(delta) * pi;  //ie excessive precision check
  m_sin = std::sin(two_pi / steps);
  m_cos = std::cos(two_pi / steps);
  m_StepsPerRad = steps / two_pi;
  if (delta < 0.0) m_sin = -m_sin;

  m_destPolys.reserve(m_polyNodes.ChildCount() * 2);
  for (int i = 0; i < m_polyNodes.ChildCount(); i++)
  {
    PolyNode& node = *m_polyNodes.Childs[i];
    m_srcPoly = node.Contour;

    int len = (int)m_srcPoly.size();
    if (len == 0 || (delta <= 0 && (len < 3 || node.m_endtype != etClosedPolygon)))
        continue;

    m_destPoly.clear();
    if (len == 1)
    {
      if (node.m_jointype == jtRound)
      {
        double X = 1.0, Y = 0.0;
        for (cInt j = 1; j <= steps; j++)
        {
          m_destPoly.push_back(IntPoint(
            Round(m_srcPoly[0].X + X * delta),
            Round(m_srcPoly[0].Y + Y * delta)));
          double X2 = X;
          X = X * m_cos - m_sin * Y;
          Y = X2 * m_sin + Y * m_cos;
        }
      }
      else
      {
        double X = -1.0, Y = -1.0;
        for (int j = 0; j < 4; ++j)
        {
          m_destPoly.push_back(IntPoint(
            Round(m_srcPoly[0].X + X * delta),
            Round(m_srcPoly[0].Y + Y * delta)));
          if (X < 0) X = 1;
          else if (Y < 0) Y = 1;
          else X = -1;
        }
      }
      m_destPolys.push_back(m_destPoly);
      continue;
    }
    //build m_normals ...
    m_normals.clear();
    m_normals.reserve(len);
    for (int j = 0; j < len - 1; ++j)
      m_normals.push_back(GetUnitNormal(m_srcPoly[j], m_srcPoly[j + 1]));
    if (node.m_endtype == etClosedLine || node.m_endtype == etClosedPolygon)
      m_normals.push_back(GetUnitNormal(m_srcPoly[len - 1], m_srcPoly[0]));
    else
      m_normals.push_back(DoublePoint(m_normals[len - 2]));

    if (node.m_endtype == etClosedPolygon)
    {
      int k = len - 1;
      for (int j = 0; j < len; ++j)
        OffsetPoint(j, k, node.m_jointype);
      m_destPolys.push_back(m_destPoly);
    }
    else if (node.m_endtype == etClosedLine)
    {
      int k = len - 1;
      for (int j = 0; j < len; ++j)
        OffsetPoint(j, k, node.m_jointype);
      m_destPolys.push_back(m_destPoly);
      m_destPoly.clear();
      //re-build m_normals ...
      DoublePoint n = m_normals[len -1];
      for (int j = len - 1; j > 0; j--)
        m_normals[j] = DoublePoint(-m_normals[j - 1].X, -m_normals[j - 1].Y);
      m_normals[0] = DoublePoint(-n.X, -n.Y);
      k = 0;
      for (int j = len - 1; j >= 0; j--)
        OffsetPoint(j, k, node.m_jointype);
      m_destPolys.push_back(m_destPoly);
    }
    else
    {
      int k = 0;
      for (int j = 1; j < len - 1; ++j)
        OffsetPoint(j, k, node.m_jointype);

      IntPoint pt1;
      if (node.m_endtype == etOpenButt)
      {
        int j = len - 1;
        pt1 = IntPoint((cInt)Round(m_srcPoly[j].X + m_normals[j].X *
          delta), (cInt)Round(m_srcPoly[j].Y + m_normals[j].Y * delta));
        m_destPoly.push_back(pt1);
        pt1 = IntPoint((cInt)Round(m_srcPoly[j].X - m_normals[j].X *
          delta), (cInt)Round(m_srcPoly[j].Y - m_normals[j].Y * delta));
        m_destPoly.push_back(pt1);
      }
      else
      {
        int j = len - 1;
        k = len - 2;
        m_sinA = 0;
        m_normals[j] = DoublePoint(-m_normals[j].X, -m_normals[j].Y);
        if (node.m_endtype == etOpenSquare)
          DoSquare(j, k);
        else
          DoRound(j, k);
      }

      //re-build m_normals ...
      for (int j = len - 1; j > 0; j--)
        m_normals[j] = DoublePoint(-m_normals[j - 1].X, -m_normals[j - 1].Y);
      m_normals[0] = DoublePoint(-m_normals[1].X, -m_normals[1].Y);

      k = len - 1;
      for (int j = k - 1; j > 0; --j) OffsetPoint(j, k, node.m_jointype);

      if (node.m_endtype == etOpenButt)
      {
        pt1 = IntPoint((cInt)Round(m_srcPoly[0].X - m_normals[0].X * delta),
          (cInt)Round(m_srcPoly[0].Y - m_normals[0].Y * delta));
        m_destPoly.push_back(pt1);
        pt1 = IntPoint((cInt)Round(m_srcPoly[0].X + m_normals[0].X * delta),
          (cInt)Round(m_srcPoly[0].Y + m_normals[0].Y * delta));
        m_destPoly.push_back(pt1);
      }
      else
      {
        k = 1;
        m_sinA = 0;
        if (node.m_endtype == etOpenSquare)
          DoSquare(0, 1);
        else
          DoRound(0, 1);
      }
      m_destPolys.push_back(m_destPoly);
    }
  }
}
//------------------------------------------------------------------------------

void ClipperOffset::OffsetPoint(int j, int& k, JoinType jointype)
{
  //cross product ...
  m_sinA = (m_normals[k].X * m_normals[j].Y - m_normals[j].X * m_normals[k].Y);
  if (std::fabs(m_sinA * m_delta) < 1.0) 
  {
    //dot product ...
    double cosA = (m_normals[k].X * m_normals[j].X + m_normals[j].Y * m_normals[k].Y ); 
    if (cosA > 0) // angle => 0 degrees
    {
      m_destPoly.push_back(IntPoint(Round(m_srcPoly[j].X + m_normals[k].X * m_delta),
        Round(m_srcPoly[j].Y + m_normals[k].Y * m_delta)));
      return; 
    }
    //else angle => 180 degrees   
  }
  else if (m_sinA > 1.0) m_sinA = 1.0;
  else if (m_sinA < -1.0) m_sinA = -1.0;

  if (m_sinA * m_delta < 0)
  {
    m_destPoly.push_back(IntPoint(Round(m_srcPoly[j].X + m_normals[k].X * m_delta),
      Round(m_srcPoly[j].Y + m_normals[k].Y * m_delta)));
    m_destPoly.push_back(m_srcPoly[j]);
    m_destPoly.push_back(IntPoint(Round(m_srcPoly[j].X + m_normals[j].X * m_delta),
      Round(m_srcPoly[j].Y + m_normals[j].Y * m_delta)));
  }
  else
    switch (jointype)
    {
      case jtMiter:
        {
          double r = 1 + (m_normals[j].X * m_normals[k].X +
            m_normals[j].Y * m_normals[k].Y);
          if (r >= m_miterLim) DoMiter(j, k, r); else DoSquare(j, k);
          break;
        }
      case jtSquare: DoSquare(j, k); break;
      case jtRound: DoRound(j, k); break;
    }
  k = j;
}
//------------------------------------------------------------------------------

void ClipperOffset::DoSquare(int j, int k)
{
  double dx = std::tan(std::atan2(m_sinA,
      m_normals[k].X * m_normals[j].X + m_normals[k].Y * m_normals[j].Y) / 4);
  m_destPoly.push_back(IntPoint(
      Round(m_srcPoly[j].X + m_delta * (m_normals[k].X - m_normals[k].Y * dx)),
      Round(m_srcPoly[j].Y + m_delta * (m_normals[k].Y + m_normals[k].X * dx))));
  m_destPoly.push_back(IntPoint(
      Round(m_srcPoly[j].X + m_delta * (m_normals[j].X + m_normals[j].Y * dx)),
      Round(m_srcPoly[j].Y + m_delta * (m_normals[j].Y - m_normals[j].X * dx))));
}
//------------------------------------------------------------------------------

void ClipperOffset::DoMiter(int j, int k, double r)
{
  double q = m_delta / r;
  m_destPoly.push_back(IntPoint(Round(m_srcPoly[j].X + (m_normals[k].X + m_normals[j].X) * q),
      Round(m_srcPoly[j].Y + (m_normals[k].Y + m_normals[j].Y) * q)));
}
//------------------------------------------------------------------------------

void ClipperOffset::DoRound(int j, int k)
{
  double a = std::atan2(m_sinA,
  m_normals[k].X * m_normals[j].X + m_normals[k].Y * m_normals[j].Y);
  int steps = std::max((int)Round(m_StepsPerRad * std::fabs(a)), 1);

  double X = m_normals[k].X, Y = m_normals[k].Y, X2;
  for (int i = 0; i < steps; ++i)
  {
    m_destPoly.push_back(IntPoint(
        Round(m_srcPoly[j].X + X * m_delta),
        Round(m_srcPoly[j].Y + Y * m_delta)));
    X2 = X;
    X = X * m_cos - m_sin * Y;
    Y = X2 * m_sin + Y * m_cos;
  }
  m_destPoly.push_back(IntPoint(
  Round(m_srcPoly[j].X + m_normals[j].X * m_delta),
  Round(m_srcPoly[j].Y + m_normals[j].Y * m_delta)));
}

//------------------------------------------------------------------------------
// Miscellaneous public functions
//------------------------------------------------------------------------------

void Clipper::DoSimplePolygons()
{
  PolyOutList::size_type i = 0;
  while (i < m_PolyOuts.size()) 
  {
    OutRec* outrec = m_PolyOuts[i++];
    OutPt* op = outrec->Pts;
    if (!op || outrec->IsOpen) continue;
    do //for each Pt in Polygon until duplicate found do ...
    {
      OutPt* op2 = op->Next;
      while (op2 != outrec->Pts) 
      {
        if ((op->Pt == op2->Pt) && op2->Next != op && op2->Prev != op) 
        {
          //split the polygon into two ...
          OutPt* op3 = op->Prev;
          OutPt* op4 = op2->Prev;
          op->Prev = op4;
          op4->Next = op;
          op2->Prev = op3;
          op3->Next = op2;

          outrec->Pts = op;
          OutRec* outrec2 = CreateOutRec();
          outrec2->Pts = op2;
          UpdateOutPtIdxs(*outrec2);
          if (Poly2ContainsPoly1(outrec2->Pts, outrec->Pts))
          {
            //OutRec2 is contained by OutRec1 ...
            outrec2->IsHole = !outrec->IsHole;
            outrec2->FirstLeft = outrec;
            if (m_UsingPolyTree) FixupFirstLefts2(outrec2, outrec);
          }
          else
            if (Poly2ContainsPoly1(outrec->Pts, outrec2->Pts))
          {
            //OutRec1 is contained by OutRec2 ...
            outrec2->IsHole = outrec->IsHole;
            outrec->IsHole = !outrec2->IsHole;
            outrec2->FirstLeft = outrec->FirstLeft;
            outrec->FirstLeft = outrec2;
            if (m_UsingPolyTree) FixupFirstLefts2(outrec, outrec2);
            }
            else
          {
            //the 2 polygons are separate ...
            outrec2->IsHole = outrec->IsHole;
            outrec2->FirstLeft = outrec->FirstLeft;
            if (m_UsingPolyTree) FixupFirstLefts1(outrec, outrec2);
            }
          op2 = op; //ie get ready for the Next iteration
        }
        op2 = op2->Next;
      }
      op = op->Next;
    }
    while (op != outrec->Pts);
  }
}
//------------------------------------------------------------------------------

void ReversePath(Path& p)
{
  std::reverse(p.begin(), p.end());
}
//------------------------------------------------------------------------------

void ReversePaths(Paths& p)
{
  for (Paths::size_type i = 0; i < p.size(); ++i)
    ReversePath(p[i]);
}
//------------------------------------------------------------------------------

void SimplifyPolygon(const Path &in_poly, Paths &out_polys, PolyFillType fillType)
{
  Clipper c;
  c.StrictlySimple(true);
  c.AddPath(in_poly, ptSubject, true);
  c.Execute(ctUnion, out_polys, fillType, fillType);
}
//------------------------------------------------------------------------------

void SimplifyPolygons(const Paths &in_polys, Paths &out_polys, PolyFillType fillType)
{
  Clipper c;
  c.StrictlySimple(true);
  c.AddPaths(in_polys, ptSubject, true);
  c.Execute(ctUnion, out_polys, fillType, fillType);
}
//------------------------------------------------------------------------------

void SimplifyPolygons(Paths &polys, PolyFillType fillType)
{
  SimplifyPolygons(polys, polys, fillType);
}
//------------------------------------------------------------------------------

inline double DistanceSqrd(const IntPoint& pt1, const IntPoint& pt2)
{
  double Dx = ((double)pt1.X - pt2.X);
  double dy = ((double)pt1.Y - pt2.Y);
  return (Dx*Dx + dy*dy);
}
//------------------------------------------------------------------------------

double DistanceFromLineSqrd(
  const IntPoint& pt, const IntPoint& ln1, const IntPoint& ln2)
{
  //The equation of a line in general form (Ax + By + C = 0)
  //given 2 points (x¹,y¹) & (x²,y²) is ...
  //(y¹ - y²)x + (x² - x¹)y + (y² - y¹)x¹ - (x² - x¹)y¹ = 0
  //A = (y¹ - y²); B = (x² - x¹); C = (y² - y¹)x¹ - (x² - x¹)y¹
  //perpendicular distance of point (x³,y³) = (Ax³ + By³ + C)/Sqrt(A² + B²)
  //see http://en.wikipedia.org/wiki/Perpendicular_distance
  double A = double(ln1.Y - ln2.Y);
  double B = double(ln2.X - ln1.X);
  double C = A * ln1.X  + B * ln1.Y;
  C = A * pt.X + B * pt.Y - C;
  return (C * C) / (A * A + B * B);
}
//---------------------------------------------------------------------------

bool SlopesNearCollinear(const IntPoint& pt1, 
    const IntPoint& pt2, const IntPoint& pt3, double distSqrd)
{
  //this function is more accurate when the point that's geometrically
  //between the other 2 points is the one that's tested for distance.
  //ie makes it more likely to pick up 'spikes' ...
	if (Abs(pt1.X - pt2.X) > Abs(pt1.Y - pt2.Y))
	{
    if ((pt1.X > pt2.X) == (pt1.X < pt3.X))
      return DistanceFromLineSqrd(pt1, pt2, pt3) < distSqrd;
    else if ((pt2.X > pt1.X) == (pt2.X < pt3.X))
      return DistanceFromLineSqrd(pt2, pt1, pt3) < distSqrd;
		else
	    return DistanceFromLineSqrd(pt3, pt1, pt2) < distSqrd;
	}
	else
	{
    if ((pt1.Y > pt2.Y) == (pt1.Y < pt3.Y))
      return DistanceFromLineSqrd(pt1, pt2, pt3) < distSqrd;
    else if ((pt2.Y > pt1.Y) == (pt2.Y < pt3.Y))
      return DistanceFromLineSqrd(pt2, pt1, pt3) < distSqrd;
		else
      return DistanceFromLineSqrd(pt3, pt1, pt2) < distSqrd;
	}
}
//------------------------------------------------------------------------------

bool PointsAreClose(IntPoint pt1, IntPoint pt2, double distSqrd)
{
    double Dx = (double)pt1.X - pt2.X;
    double dy = (double)pt1.Y - pt2.Y;
    return ((Dx * Dx) + (dy * dy) <= distSqrd);
}
//------------------------------------------------------------------------------

OutPt* ExcludeOp(OutPt* op)
{
  OutPt* result = op->Prev;
  result->Next = op->Next;
  op->Next->Prev = result;
  result->Idx = 0;
  return result;
}
//------------------------------------------------------------------------------

void CleanPolygon(const Path& in_poly, Path& out_poly, double distance)
{
  //distance = proximity in units/pixels below which vertices
  //will be stripped. Default ~= sqrt(2).
  
  size_t size = in_poly.size();
  
  if (size == 0) 
  {
    out_poly.clear();
    return;
  }

  OutPt* outPts = new OutPt[size];
  for (size_t i = 0; i < size; ++i)
  {
    outPts[i].Pt = in_poly[i];
    outPts[i].Next = &outPts[(i + 1) % size];
    outPts[i].Next->Prev = &outPts[i];
    outPts[i].Idx = 0;
  }

  double distSqrd = distance * distance;
  OutPt* op = &outPts[0];
  while (op->Idx == 0 && op->Next != op->Prev) 
  {
    if (PointsAreClose(op->Pt, op->Prev->Pt, distSqrd))
    {
      op = ExcludeOp(op);
      size--;
    } 
    else if (PointsAreClose(op->Prev->Pt, op->Next->Pt, distSqrd))
    {
      ExcludeOp(op->Next);
      op = ExcludeOp(op);
      size -= 2;
    }
    else if (SlopesNearCollinear(op->Prev->Pt, op->Pt, op->Next->Pt, distSqrd))
    {
      op = ExcludeOp(op);
      size--;
    }
    else
    {
      op->Idx = 1;
      op = op->Next;
    }
  }

  if (size < 3) size = 0;
  out_poly.resize(size);
  for (size_t i = 0; i < size; ++i)
  {
    out_poly[i] = op->Pt;
    op = op->Next;
  }
  delete [] outPts;
}
//------------------------------------------------------------------------------

void CleanPolygon(Path& poly, double distance)
{
  CleanPolygon(poly, poly, distance);
}
//------------------------------------------------------------------------------

void CleanPolygons(const Paths& in_polys, Paths& out_polys, double distance)
{
  out_polys.resize(in_polys.size());
  for (Paths::size_type i = 0; i < in_polys.size(); ++i)
    CleanPolygon(in_polys[i], out_polys[i], distance);
}
//------------------------------------------------------------------------------

void CleanPolygons(Paths& polys, double distance)
{
  CleanPolygons(polys, polys, distance);
}
//------------------------------------------------------------------------------

void Minkowski(const Path& poly, const Path& path, 
  Paths& solution, bool isSum, bool isClosed)
{
  int delta = (isClosed ? 1 : 0);
  size_t polyCnt = poly.size();
  size_t pathCnt = path.size();
  Paths pp;
  pp.reserve(pathCnt);
  if (isSum)
    for (size_t i = 0; i < pathCnt; ++i)
    {
      Path p;
      p.reserve(polyCnt);
      for (size_t j = 0; j < poly.size(); ++j)
        p.push_back(IntPoint(path[i].X + poly[j].X, path[i].Y + poly[j].Y));
      pp.push_back(p);
    }
  else
    for (size_t i = 0; i < pathCnt; ++i)
    {
      Path p;
      p.reserve(polyCnt);
      for (size_t j = 0; j < poly.size(); ++j)
        p.push_back(IntPoint(path[i].X - poly[j].X, path[i].Y - poly[j].Y));
      pp.push_back(p);
    }

  solution.clear();
  solution.reserve((pathCnt + delta) * (polyCnt + 1));
  for (size_t i = 0; i < pathCnt - 1 + delta; ++i)
    for (size_t j = 0; j < polyCnt; ++j)
    {
      Path quad;
      quad.reserve(4);
      quad.push_back(pp[i % pathCnt][j % polyCnt]);
      quad.push_back(pp[(i + 1) % pathCnt][j % polyCnt]);
      quad.push_back(pp[(i + 1) % pathCnt][(j + 1) % polyCnt]);
      quad.push_back(pp[i % pathCnt][(j + 1) % polyCnt]);
      if (!Orientation(quad)) ReversePath(quad);
      solution.push_back(quad);
    }
}
//------------------------------------------------------------------------------

void MinkowskiSum(const Path& pattern, const Path& path, Paths& solution, bool pathIsClosed)
{
  Minkowski(pattern, path, solution, true, pathIsClosed);
  Clipper c;
  c.AddPaths(solution, ptSubject, true);
  c.Execute(ctUnion, solution, pftNonZero, pftNonZero);
}
//------------------------------------------------------------------------------

void TranslatePath(const Path& input, Path& output, const IntPoint delta)
{
  //precondition: input != output
  output.resize(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    output[i] = IntPoint(input[i].X + delta.X, input[i].Y + delta.Y);
}
//------------------------------------------------------------------------------

void MinkowskiSum(const Path& pattern, const Paths& paths, Paths& solution, bool pathIsClosed)
{
  Clipper c;
  for (size_t i = 0; i < paths.size(); ++i)
  {
    Paths tmp;
    Minkowski(pattern, paths[i], tmp, true, pathIsClosed);
    c.AddPaths(tmp, ptSubject, true);
    if (pathIsClosed)
    {
      Path tmp2;
      TranslatePath(paths[i], tmp2, pattern[0]);
      c.AddPath(tmp2, ptClip, true);
    }
  }
    c.Execute(ctUnion, solution, pftNonZero, pftNonZero);
}
//------------------------------------------------------------------------------

void MinkowskiDiff(const Path& poly1, const Path& poly2, Paths& solution)
{
  Minkowski(poly1, poly2, solution, false, true);
  Clipper c;
  c.AddPaths(solution, ptSubject, true);
  c.Execute(ctUnion, solution, pftNonZero, pftNonZero);
}
//------------------------------------------------------------------------------

enum NodeType {ntAny, ntOpen, ntClosed};

void AddPolyNodeToPaths(const PolyNode& polynode, NodeType nodetype, Paths& paths)
{
  bool match = true;
  if (nodetype == ntClosed) match = !polynode.IsOpen();
  else if (nodetype == ntOpen) return;

  if (!polynode.Contour.empty() && match)
    paths.push_back(polynode.Contour);
  for (int i = 0; i < polynode.ChildCount(); ++i)
    AddPolyNodeToPaths(*polynode.Childs[i], nodetype, paths);
}
//------------------------------------------------------------------------------

void PolyTreeToPaths(const PolyTree& polytree, Paths& paths)
{
  paths.resize(0); 
  paths.reserve(polytree.Total());
  AddPolyNodeToPaths(polytree, ntAny, paths);
}
//------------------------------------------------------------------------------

void ClosedPathsFromPolyTree(const PolyTree& polytree, Paths& paths)
{
  paths.resize(0); 
  paths.reserve(polytree.Total());
  AddPolyNodeToPaths(polytree, ntClosed, paths);
}
//------------------------------------------------------------------------------

void OpenPathsFromPolyTree(PolyTree& polytree, Paths& paths)
{
  paths.resize(0); 
  paths.reserve(polytree.Total());
  //Open paths are top level only, so ...
  for (int i = 0; i < polytree.ChildCount(); ++i)
    if (polytree.Childs[i]->IsOpen())
      paths.push_back(polytree.Childs[i]->Contour);
}
//------------------------------------------------------------------------------

std::ostream& operator <<(std::ostream &s, const IntPoint &p)
{
  s << "(" << p.X << "," << p.Y << ")";
  return s;
}
//------------------------------------------------------------------------------

std::ostream& operator <<(std::ostream &s, const Path &p)
{
  if (p.empty()) return s;
  Path::size_type last = p.size() -1;
  for (Path::size_type i = 0; i < last; i++)
    s << "(" << p[i].X << "," << p[i].Y << "), ";
  s << "(" << p[last].X << "," << p[last].Y << ")\n";
  return s;
}
//------------------------------------------------------------------------------

std::ostream& operator <<(std::ostream &s, const Paths &p)
{
  for (Paths::size_type i = 0; i < p.size(); i++)
    s << p[i];
  s << "\n";
  return s;
}
//------------------------------------------------------------------------------

} //ClipperLib namespace

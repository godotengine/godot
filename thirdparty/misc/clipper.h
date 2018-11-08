/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Version   :  10.0 (beta)                                                     *
* Date      :  8 Noveber 2017                                                  *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2017                                         *
* Purpose   :  Base clipping module                                            *
* License   : http://www.boost.org/LICENSE_1_0.txt                             *
*******************************************************************************/

#ifndef clipper_h
#define clipper_h

#define CLIPPER_VERSION "10.0.0"

#include <vector>
#include <queue>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <float.h>

namespace clipperlib {

enum ClipType { ctNone,  ctIntersection, ctUnion, ctDifference, ctXor };
enum PathType { ptSubject, ptClip };

//By far the most widely used winding rules for polygon filling are EvenOdd
//and NonZero (see GDI, GDI+, XLib, OpenGL, Cairo, AGG, Quartz, SVG, Gr32).
//https://www.w3.org/TR/SVG/painting.html
enum FillRule { frEvenOdd, frNonZero, frPositive, frNegative };

struct Point64 {
  int64_t x;
  int64_t y;
  Point64(int64_t x = 0, int64_t y = 0): x(x), y(y) {};

  friend inline bool operator== (const Point64 &a, const Point64 &b)
  {
    return a.x == b.x && a.y == b.y;
  }
  friend inline bool operator!= (const Point64 &a, const Point64 &b)
  {
    return a.x != b.x || a.y != b.y;
  }
  inline bool operator<(const Point64 &b) const {
    return (x == b.x) ? (y < b.y) : (x < b.x);
  }
  inline Point64 operator+(const Point64 &b) const {
    return Point64(x + b.x, y + b.y);
  }
};

typedef std::vector< Point64 > Path;
typedef std::vector< Path > Paths;

inline Path& operator <<(Path &path, const Point64 &pt) {path.push_back(pt); return path;}
inline Paths& operator <<(Paths &paths, const Path &path) {paths.push_back(path); return paths;}

std::ostream& operator <<(std::ostream &s, const Point64 &p);
std::ostream& operator <<(std::ostream &s, const Path &p);
std::ostream& operator <<(std::ostream &s, const Paths &p);

class PolyPath
{
  private:
	  PolyPath *parent_;
	  Path path_;
	  std::vector< PolyPath* > childs_;
  public:
    PolyPath();
	  PolyPath(PolyPath *parent, const Path &path);
	  virtual ~PolyPath(){};
    PolyPath &AddChild(const Path &path);
	  PolyPath& GetChild(unsigned index);
    int ChildCount() const;
    PolyPath* GetParent() const;
	  Path& GetPath();
	  bool IsHole() const;
	  void Clear();

    inline bool operator<(const PolyPath* p) const { return this < p; }
};

struct Rect64 {
	int64_t left;
	int64_t top;
	int64_t right;
	int64_t bottom;
  Rect64(int64_t l, int64_t t, int64_t r, int64_t b) : left(l), top(t), right(r), bottom(b) {}
};

struct Scanline;
struct IntersectNode;
struct Active;
struct Vertex;
struct LocalMinima;

class OutPt {
public:
  Point64      pt;
  OutPt       *next;
  OutPt       *prev;
};

enum OutRecFlag { orInner, orOuter, orOpen};

//OutRec: contains a path in the clipping solution. Edges in the active edge list (AEL)
//will carry a pointer to an OutRec when they are part of the clipping solution.
struct OutRec {
  unsigned     idx;
  OutRec      *owner;
  Active      *start_e;
  Active      *end_e;
  OutPt       *pts;
  PolyPath    *polypath;
  OutRecFlag  flag;
};

//Active: an edge in the AEL that may or may not be 'hot' (part of the clip solution).
struct Active {
  Point64      bot;
  Point64      curr;         //current (updated at every new scanline)
  Point64      top;
  double       dx;
  int          wind_dx;      //1 or -1 depending on winding direction
  int          wind_cnt;
  int          wind_cnt2;    //winding count of the opposite polytype
  OutRec      *outrec;
  Active      *next_in_ael;
  Active      *prev_in_ael;
  Active      *next_in_sel;
  Active      *prev_in_sel;
  Active      *merge_jump;
  Vertex      *vertex_top;
  LocalMinima *local_min;    //bottom of bound
};

class Clipper {
  private:
    typedef std::vector < OutRec* > OutRecList;
	  typedef std::vector < IntersectNode* > IntersectList;
	  typedef std::priority_queue< int64_t > ScanlineList;
    typedef std::vector< LocalMinima* > MinimaList;
    typedef std::vector< Vertex* > VerticesList;

	  ClipType          cliptype_;
    FillRule          fillrule_;
    Active	         *actives_;
    Active           *sel_;
    bool			        has_open_paths_;
    MinimaList        minima_list_;
    MinimaList::iterator curr_loc_min_;
    bool			        minima_list_sorted_;
    OutRecList		    outrec_list_;
    IntersectList     intersect_list_;
    VerticesList      vertex_list_;
    ScanlineList		  scanline_list_;
    void Reset();
    void InsertScanline(int64_t y);
    bool PopScanline(int64_t &y);
    bool PopLocalMinima(int64_t y, LocalMinima *&local_minima);
    void DisposeAllOutRecs();
    void DisposeVerticesAndLocalMinima();
    void AddLocMin(Vertex &vert, PathType polytype, bool is_open);
    void AddPathToVertexList(const Path &p, PathType polytype, bool is_open);
    bool IsContributingClosed(const Active &e) const;
    inline bool IsContributingOpen(const Active &e) const;
    void SetWindingLeftEdgeClosed(Active &edge);
    void SetWindingLeftEdgeOpen(Active &e);
    void InsertEdgeIntoAEL(Active &edge, Active *startEdge);
    virtual void InsertLocalMinimaIntoAEL(int64_t bot_y);
    inline void PushHorz(Active &e);
    inline bool PopHorz(Active *&e);
    inline OutRec* GetOwner(const Active *e);
    void JoinOutrecPaths(Active &e1, Active &e2);
    inline void TerminateHotOpen(Active &e);
    inline void StartOpenPath(Active &e, const Point64 pt);
    inline void UpdateEdgeIntoAEL(Active *e);
    virtual void IntersectEdges(Active &e1, Active &e2, const Point64 pt);
    inline void DeleteFromAEL(Active &e);
    inline void CopyAELToSEL();
    inline void CopyActivesToSELAdjustCurrX(const int64_t top_y);
    void ProcessIntersections(const int64_t top_y);
    void DisposeIntersectNodes();
    void InsertNewIntersectNode(Active &e1, Active &e2, const int64_t top_y);
    void BuildIntersectList(const int64_t top_y);
    bool ProcessIntersectList();
    void FixupIntersectionOrder();
    void SwapPositionsInAEL(Active &edge1, Active &edge2);
    void SwapPositionsInSEL(Active &edge1, Active &edge2);
    inline void Insert2Before1InSel(Active &first, Active &second);
    bool ResetHorzDirection(Active &horz, Active *max_pair, int64_t &horz_left, int64_t &horz_right);
    void ProcessHorizontal(Active &horz);
    void DoTopOfScanbeam(const int64_t top_y);
    Active* DoMaxima(Active &e);
    void BuildResult(Paths &paths_closed, Paths *paths_open);
    void BuildResult2(PolyPath &pt, Paths *solution_open);
  protected:
    void CleanUp();
    virtual OutPt* CreateOutPt();
    virtual OutRec* CreateOutRec();
    virtual OutPt* AddOutPt(Active &e, const Point64 pt);
    virtual void AddLocalMinPoly(Active &e1, Active &e2, const Point64 pt);
    virtual void AddLocalMaxPoly(Active &e1, Active &e2, const Point64 pt);
    bool ExecuteInternal(ClipType ct, FillRule ft);
    /*get properties ... */
    OutRecList& outrec_list() { return outrec_list_; };
  public:
    Clipper();
    virtual ~Clipper();
    virtual void AddPath(const Path &path, PathType polytype, bool is_open = false);
    virtual void AddPaths(const Paths &paths, PathType polytype, bool is_open = false);
    virtual bool Execute(ClipType clipType, Paths &solution_closed, FillRule fr = frEvenOdd);
    virtual bool Execute(ClipType clipType, Paths &solution_closed, Paths &solution_open, FillRule fr = frEvenOdd);
    virtual bool Execute(ClipType clipType, PolyPath &solution_closed, Paths &solution_open, FillRule fr = frEvenOdd);
    void Clear();
    Rect64 GetBounds();
};
//------------------------------------------------------------------------------

#define CLIPPER_HORIZONTAL (-DBL_MAX)

} //clipperlib namespace

#endif //clipper_h



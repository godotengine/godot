/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Version   :  10.0 (beta)                                                     *
* Date      :  8 Noveber 2017                                                  *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2017                                         *
* Purpose   :  Triangulate clipping solutions                                  *
* License   : http://www.boost.org/LICENSE_1_0.txt                             *
*******************************************************************************/

#ifndef clipper_tri_h
#define clipper_tri_h

#include <cstdlib>
#include "clipper.h"

namespace clipperlib {

  class OutRecTri;

  class OutPtTri : public OutPt
  {
  public:
    OutRec     *outrec;
    OutRecTri  *right_outrec;
  };

  class OutRecTri : public OutRec
  {
  public:
    OutPtTri  *left_outpt;
  };

  class ClipperTri : public virtual Clipper
  {
  private:
    OutPt *last_op_;
    Paths triangles_;
    void  AddPolygon(const Point64 pt1, const Point64 pt2, const Point64 pt3);
    void  Triangulate(OutRec *outrec);
    void  BuildResult(Paths &triangles);
  protected:
    OutPt* CreateOutPt();
    OutRec* CreateOutRec();
    OutPt* AddOutPt(Active &e, const Point64 pt);
    void AddLocalMinPoly(Active &e1, Active &e2, const Point64 pt);
    void AddLocalMaxPoly(Active &e1, Active &e2, const Point64 pt);
  public:
    bool Execute(ClipType clipType, Paths &solution, FillRule fr = frEvenOdd);
    bool Execute(ClipType clipType, Paths &solution_closed, Paths &solution_open, FillRule fr = frEvenOdd)
      { return false; } //it's pointless triangulating open paths
    bool Execute(ClipType clipType, PolyPath &solution_closed, Paths &solution_open, FillRule fr = frEvenOdd)
    { return false; } //the PolyPath structure is of no benefit when triangulating
  };

} //namespace

#endif //clipper_tri_h

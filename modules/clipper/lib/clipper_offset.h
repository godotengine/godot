/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Version   :  10.0 (beta)                                                     *
* Date      :  8 Noveber 2017                                                  *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2017                                         *
* Purpose   :  Offset clipping solutions                                       *
* License   : http://www.boost.org/LICENSE_1_0.txt                             *
*******************************************************************************/

#ifndef clipper_offset_h
#define clipper_offset_h

#include <vector>
#include <cstdlib>
#include "clipper.h"

namespace clipperlib {

  enum JoinType { kSquare, kRound, kMiter };
  enum EndType { kPolygon, kOpenJoined, kOpenButt, kOpenSquare, kOpenRound };

  void OffsetPaths(Paths &paths_in, Paths &paths_out, double delta, JoinType jt, EndType et);

  class ClipperOffset
  {
  private:

    struct PointD
    {
      double x;
      double y;
      PointD(double x_ = 0, double y_ = 0) : x(x_), y(y_) {};
      PointD(const Point64 &pt) : x((double)pt.x), y((double)pt.y) {};
    };

    struct PathNode
    {
      Path path;
      JoinType join_type;
      EndType end_type;
      int lowest_idx;
      PathNode(const Path &p, JoinType jt, EndType et);
    };

    typedef std::vector< PointD > NormalsList;
    typedef std::vector< PathNode* > NodeList;

    Paths solution_;
    Path path_in_, path_out_;
    NormalsList norms_;
    NodeList nodes_;
    double arc_tolerance_;
    double miter_limit_;

    //nb: miter_lim_ below is a temp field that differs from miter_limit
    double delta_, sin_a_, sin_, cos_, miter_lim_, steps_per_radian_;
    int lowest_idx_;
    void GetLowestPolygonIdx();
    void OffsetPoint(size_t j, size_t &k, JoinType join_type);
    void DoSquare(int j, int k);
    void DoMiter(int j, int k, double cos_a_plus_1);
    void DoRound(int j, int k);
    void DoOffset(double d);
    static PointD GetUnitNormal(const Point64 &pt1, const Point64 &pt2);

  public:
    ClipperOffset(double miter_limit = 2.0, double arc_tolerance = 0) :
      miter_limit_(miter_limit), arc_tolerance_(arc_tolerance) {};
    void Clear();
    void AddPath(const Path &path, JoinType jt, EndType et);
    void AddPaths(const Paths &paths, JoinType jt, EndType et);
    void Execute(Paths &sol, double delta);
  };

} //clipperlib namespace

#endif //clipper_offset_h



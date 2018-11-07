/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Version   :  10.0 (beta)                                                     *
* Date      :  8 Noveber 2017                                                  *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2017                                         *
* Purpose   :  Offset clipping solutions                                       *
* License   : http://www.boost.org/LICENSE_1_0.txt                             *
*******************************************************************************/

#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include "clipper.h"
#include "clipper_offset.h"

namespace clipperlib {

  #define PI                (3.14159265358979323846) 
  #define TWO_PI            (PI * 2)
  #define DEFAULT_ARC_FRAC  (0.02)
  #define TOLERANCE         (1.0E-12)

  inline int64_t Round(double val)
  {
    if ((val < 0)) return static_cast<int64_t>(val - 0.5);
    else return static_cast<int64_t>(val + 0.5);
  }
  //---------------------------------------------------------------------------

  double Area(Path path)
  {
    int cnt = (int)path.size();
    if (cnt < 3) return 0;
    double a = 0;
    for (int i = 0, j = cnt - 1; i < cnt; ++i)
    {
      a += ((double)path[j].x + path[i].x) * ((double)path[j].y - path[i].y);
      j = i;
    }
    return -a * 0.5;
  }
  //---------------------------------------------------------------------------

  ClipperOffset::PointD ClipperOffset::GetUnitNormal(const Point64 &pt1, const Point64 &pt2)
  {
    double dx = (double)(pt2.x - pt1.x);
    double dy = (double)(pt2.y - pt1.y);
    if ((dx == 0) && (dy == 0)) return PointD(0,0);
    double f = 1 * 1.0 / sqrt(dx * dx + dy * dy);
    dx *= f;
    dy *= f;
    return PointD(dy, -dx);
  }
  //---------------------------------------------------------------------------

  ClipperOffset::PathNode::PathNode(const Path &p, JoinType jt, EndType et)
  {
    join_type = jt;
    end_type = et;

    size_t len_path = p.size();
    if (et == kPolygon || et == kOpenJoined)
      while (len_path > 1 && p[len_path - 1] == p[0]) len_path--;
    else if (len_path == 2 && p[1] == p[0])
      len_path = 1;
    if (len_path == 0) return;

    if (len_path < 3 && (et == kPolygon || et == kOpenJoined))
    {
      if (jt == kRound) end_type = kOpenRound;
      else end_type = kOpenSquare;
    }

    path.reserve(len_path);
    path.push_back(p[0]);

    Point64 last_pt = p[0];
    lowest_idx = 0;
    for (size_t i = 1, last = 0; i < len_path; ++i)
    {
      if (last_pt == p[i]) continue;
      last++; 
      path.push_back(p[i]);
      last_pt = p[i];
      //j == path.size() -1;
      if (et != kPolygon) continue;
      if (path[last].y >= path[lowest_idx].y &&
        (path[last].y > path[lowest_idx].y || path[last].x < path[lowest_idx].x))
        lowest_idx = last;
    }
    if (end_type == kPolygon && path.size() < 3) path.clear();
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::GetLowestPolygonIdx()
  {
    lowest_idx_ = -1;
    Point64 ip1 = Point64(0,0), ip2;
    for (size_t i = 0; i < nodes_.size(); ++i)
    {
      PathNode *node = nodes_[i];
      if (node->end_type != kPolygon) continue;
      if (lowest_idx_ < 0)
      {
        ip1 = node->path[node->lowest_idx];
        lowest_idx_ = i;
      }
      else
      {
        ip2 = node->path[node->lowest_idx];
        if (ip2.y >= ip1.y && (ip2.y > ip1.y || ip2.x < ip1.x))
        {
          lowest_idx_ = i;
          ip1 = ip2;
        }
      }
    }
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::OffsetPoint(size_t j, size_t &k, JoinType join_type)
  {
    //A: angle between adjoining paths on left side (left WRT winding direction).
    //A == 0 deg (or A == 360 deg): collinear edges heading in same direction
    //A == 180 deg: collinear edges heading in opposite directions (ie a 'spike')
    //sin(A) < 0: convex on left.
    //cos(A) > 0: angles on both left and right sides > 90 degrees

    //cross product ...
    sin_a_ = (norms_[k].x * norms_[j].y - norms_[j].x * norms_[k].y);

    if (abs(sin_a_ * delta_) < 1.0) //angle is approaching 180 or 360 deg.
    {
      //dot product ...
      double cos_a = (norms_[k].x * norms_[j].x + norms_[j].y * norms_[k].y);
      if (cos_a > 0) //given condition above the angle is approaching 360 deg.
      {
        //with angles approaching 360 deg collinear (whether concave or convex),
        //offsetting with two or more vertices (that would be so close together)
        //occasionally causes tiny self-intersections due to rounding.
        //So we offset with just a single vertex here ...
        path_out_.push_back(Point64(Round(path_in_[j].x + norms_[k].x * delta_),
          Round(path_in_[j].y + norms_[k].y * delta_)));
        return;
      }
    }
    else if (sin_a_ > 1.0) sin_a_ = 1.0;
    else if (sin_a_ < -1.0) sin_a_ = -1.0;

    if (sin_a_ * delta_ < 0) //ie a concave offset
    {
      path_out_.push_back(Point64(Round(path_in_[j].x + norms_[k].x * delta_),
        Round(path_in_[j].y + norms_[k].y * delta_)));
      path_out_.push_back(path_in_[j]);
      path_out_.push_back(Point64(Round(path_in_[j].x + norms_[j].x * delta_),
        Round(path_in_[j].y + norms_[j].y * delta_)));
    }
    else
    {
      double cos_a;
      //convex offsets here ...
      switch (join_type)
      {
      case kMiter:
        cos_a = (norms_[j].x * norms_[k].x + norms_[j].y * norms_[k].y);
        //see offset_triginometry3.svg
        if (1 + cos_a < miter_lim_) DoSquare(j, k);
        else DoMiter(j, k, 1 + cos_a);
        break;
      case kSquare:
        cos_a = (norms_[j].x * norms_[k].x + norms_[j].y * norms_[k].y);
        if (cos_a >= 0) DoMiter(j, k, 1 + cos_a); //angles >= 90 deg. don't need squaring
        else DoSquare(j, k);
        break;
      case kRound:
        DoRound(j, k);
        break;
      }
    }
    k = j;
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::DoSquare(int j, int k)
  {
    //Two vertices, one using the prior offset's (k) normal one the current (j).
    //Do a 'normal' offset (by delta_) and then another by 'de-normaling' the
    //normal hence parallel to the direction of the respective edges.
    if (delta_ > 0)
    {
      path_out_.push_back(Point64(
        Round(path_in_[j].x + delta_ * (norms_[k].x - norms_[k].y)),
        Round(path_in_[j].y + delta_ * (norms_[k].y + norms_[k].x))));
      path_out_.push_back(Point64(
        Round(path_in_[j].x + delta_ * (norms_[j].x + norms_[j].y)),
        Round(path_in_[j].y + delta_ * (norms_[j].y - norms_[j].x))));
    }
    else
    {
      path_out_.push_back(Point64(
        Round(path_in_[j].x + delta_ * (norms_[k].x + norms_[k].y)),
        Round(path_in_[j].y + delta_ * (norms_[k].y - norms_[k].x))));
      path_out_.push_back(Point64(
        Round(path_in_[j].x + delta_ * (norms_[j].x - norms_[j].y)),
        Round(path_in_[j].y + delta_ * (norms_[j].y + norms_[j].x))));
    }
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::DoMiter(int j, int k, double cos_a_plus_1)
  {
    //see offset_triginometry4.svg
    double q = delta_ / cos_a_plus_1; //0 < cosAplus1 <= 2
    path_out_.push_back(Point64(Round(path_in_[j].x + (norms_[k].x + norms_[j].x) * q),
      Round(path_in_[j].y + (norms_[k].y + norms_[j].y) * q)));
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::DoRound(int j, int k)
  {
    double a = atan2(sin_a_, norms_[k].x * norms_[j].x + norms_[k].y * norms_[j].y);
    int steps = (std::max)((int)Round(steps_per_radian_ * abs(a)), 1);

    double x = norms_[k].x, y = norms_[k].y, x2;
    for (int i = 0; i < steps; ++i)
    {
      path_out_.push_back(Point64(
        Round(path_in_[j].x + x * delta_),
        Round(path_in_[j].y + y * delta_)));
      x2 = x;
      x = x * cos_ - sin_ * y;
      y = x2 * sin_ + y * cos_;
    }
    path_out_.push_back(Point64(
      Round(path_in_[j].x + norms_[j].x * delta_),
      Round(path_in_[j].y + norms_[j].y * delta_)));
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::DoOffset(double d)
  {
    delta_ = d;
    double abs_delta = abs(d);

    //if a Zero offset, then just copy CLOSED polygons to FSolution and return ...
    if (abs_delta < TOLERANCE)
    {
      solution_.reserve(nodes_.size());
      for (NodeList::iterator nl_iter = nodes_.begin(); nl_iter != nodes_.end(); ++nl_iter)
        if ((*nl_iter)->end_type == kPolygon) solution_.push_back((*nl_iter)->path);
      return;
    }

    //MiterLimit: see offset_triginometry3.svg in the documentation folder ...
    if (miter_limit_ > 2)
      miter_lim_ = 2 / (miter_limit_ * miter_limit_);
    else
      miter_lim_ = 0.5;

    double arc_tol;
    if (arc_tolerance_ < DEFAULT_ARC_FRAC)
      arc_tol = abs_delta * DEFAULT_ARC_FRAC; else
      arc_tol = arc_tolerance_;

    //see offset_triginometry2.svg in the documentation folder ...
    double steps = PI / acos(1 - arc_tol / abs_delta);  //steps per 360 degrees
    if (steps > abs_delta * PI) steps = abs_delta * PI; //ie excessive precision check

    sin_ = sin(TWO_PI / steps);
    cos_ = cos(TWO_PI / steps);
    if (d < 0) sin_ = -sin_;
    steps_per_radian_ = steps / TWO_PI;

    solution_.reserve(nodes_.size() * 2);
    for (NodeList::iterator nl_iter = nodes_.begin(); nl_iter != nodes_.end(); ++nl_iter)
    {
      PathNode *node = *nl_iter;
      path_in_ = node->path;
      path_out_.clear();
      size_t path_in_size = path_in_.size();

      //if a single vertex then build circle or a square ...
      if (path_in_size == 1)
      {
        if (node->join_type == kRound)
        {
          double x = 1.0, y = 0.0;
          for (int j = 1; j <= steps; j++)
          {
            path_out_.push_back(Point64(
              Round(path_in_[0].x + x * delta_),
              Round(path_in_[0].y + y * delta_)));
            double x2 = x;
            x = x * cos_ - sin_ * y;
            y = x2 * sin_ + y * cos_;
          }
        }
        else
        {
          double x = -1.0, y = -1.0;
          for (int j = 0; j < 4; ++j)
          {
            path_out_.push_back(Point64(
              Round(path_in_[0].x + x * delta_),
              Round(path_in_[0].y + y * delta_)));
            if (x < 0) x = 1;
            else if (y < 0) y = 1;
            else x = -1;
          }
        }
        solution_.push_back(path_out_);
        continue;
      } //end of single vertex offsetting

        //build norms_ ...
      norms_.clear();
      norms_.reserve(path_in_size);
      for (size_t j = 0; j < path_in_size - 1; ++j)
        norms_.push_back(GetUnitNormal(path_in_[j], path_in_[j + 1]));
      if (node->end_type == kOpenJoined || node->end_type == kPolygon)
        norms_.push_back(GetUnitNormal(path_in_[path_in_size - 1], path_in_[0]));
      else
        norms_.push_back(PointD(norms_[path_in_size - 2]));

      if (node->end_type == kPolygon)
      {
        size_t k = path_in_size - 1;
        for (size_t j = 0; j < path_in_size; j++)
          OffsetPoint(j, k, node->join_type);
        solution_.push_back(path_out_);
      }
      else if (node->end_type == kOpenJoined)
      {
        size_t k = path_in_size - 1;
        for (size_t j = 0; j < path_in_size; j++)
          OffsetPoint(j, k, node->join_type);
        solution_.push_back(path_out_);
        path_out_.clear();
        //re-build norms_ ...
        PointD n = norms_[path_in_size - 1];
        for (size_t j = path_in_size - 1; j > 0; --j)
          norms_[j] = PointD(-norms_[j - 1].x, -norms_[j - 1].y);
        norms_[0] = PointD(-n.x, -n.y);
        k = 0;
        for (size_t j = path_in_size - 1; j >= 0; j--)
          OffsetPoint(j, k, node->join_type);
        solution_.push_back(path_out_);
      }
      else
      {
        size_t k = 0;
        for (size_t j = 1; j < path_in_size - 1; ++j)
          OffsetPoint(j, k, node->join_type);

        Point64 pt1;
        if (node->end_type == kOpenButt)
        {
          size_t j = path_in_size - 1;
          pt1 = Point64(Round(path_in_[j].x + norms_[j].x *
            delta_), Round(path_in_[j].y + norms_[j].y * delta_));
          path_out_.push_back(pt1);
          pt1 = Point64(Round(path_in_[j].x - norms_[j].x * delta_), 
            Round(path_in_[j].y - norms_[j].y * delta_));
          path_out_.push_back(pt1);
        }
        else
        {
          size_t j = path_in_size - 1;
          k = path_in_size - 2;
          sin_a_ = 0;
          norms_[j] = PointD(-norms_[j].x, -norms_[j].y);
          if (node->end_type == kOpenSquare) DoSquare(j, k);
          else DoRound(j, k);
        }

        //reverse norms_ ...
        for (size_t j = path_in_size - 1; j > 0; j--)
          norms_[j] = PointD(-norms_[j - 1].x, -norms_[j - 1].y);
        norms_[0] = PointD(-norms_[1].x, -norms_[1].y);

        k = path_in_size - 1;
        for (size_t j = k - 1; j > 0; --j) OffsetPoint(j, k, node->join_type);

        if (node->end_type == kOpenButt)
        {
          pt1 = Point64(Round(path_in_[0].x - norms_[0].x * delta_),
            Round(path_in_[0].y - norms_[0].y * delta_));
          path_out_.push_back(pt1);
          pt1 = Point64(Round(path_in_[0].x + norms_[0].x * delta_),
            Round(path_in_[0].y + norms_[0].y * delta_));
          path_out_.push_back(pt1);
        }
        else
        {
          k = 1;
          sin_a_ = 0;
          if (node->end_type == kOpenSquare) DoSquare(0, 1);
          else DoRound(0, 1);
        }
        solution_.push_back(path_out_);
      }
    }
    norms_.clear();
    path_in_.clear();
    path_out_.clear();
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::Clear()
  {
    for (NodeList::iterator nl_iter = nodes_.begin(); nl_iter != nodes_.end(); ++nl_iter)
      delete (*nl_iter);
    nodes_.clear();
    norms_.clear();
    solution_.clear();
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::AddPath(const Path &path, JoinType jt, EndType et)
  {
    PathNode *pn = new PathNode(path, jt, et);
    if (pn->path.empty()) delete pn;
    else nodes_.push_back(pn);
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::AddPaths(const Paths &paths, JoinType jt, EndType et)
  {
    for (Paths::const_iterator p_iter = paths.begin(); p_iter != paths.end(); ++p_iter)
      AddPath(*p_iter, jt, et);
  }
  //---------------------------------------------------------------------------

  void ClipperOffset::Execute(Paths &sol, double delta)
  {
    solution_.clear();
    if (nodes_.size() == 0) return;

    GetLowestPolygonIdx();
    bool negate = (lowest_idx_ >= 0 && Area(nodes_[lowest_idx_]->path) < 0);
    //if polygon orientations are reversed, then 'negate' ...
    if (negate) delta_ = -delta;
    else delta_ = delta;
    DoOffset(delta_);

    //now clean up 'corners' ...
    Clipper clpr;
    clpr.AddPaths(solution_, ptSubject);
    if (negate) clpr.Execute(ctUnion, sol, frNegative);
    else clpr.Execute(ctUnion, sol, frPositive);
  }
  //---------------------------------------------------------------------------

  void OffsetPaths(Paths &paths_in, Paths &paths_out, double delta, JoinType jt, EndType et)
  {
    ClipperOffset co;
    co.AddPaths(paths_in, jt, et);
    co.Execute(paths_out, delta);
  }
  //---------------------------------------------------------------------------

} //clipperlib namespace



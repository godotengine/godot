/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  14 January 2023                                                 *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2022                                         *
* Purpose   :  FAST rectangular clipping                                       *
* License   :  http://www.boost.org/LICENSE_1_0.txt                            *
*******************************************************************************/

#include <cmath>
#include "clipper2/clipper.h"
#include "clipper2/clipper.rectclip.h"

namespace Clipper2Lib {

  //------------------------------------------------------------------------------
  // Miscellaneous methods
  //------------------------------------------------------------------------------

  inline PointInPolygonResult Path1ContainsPath2(const Path64& path1, const Path64& path2)
  {
    PointInPolygonResult result = PointInPolygonResult::IsOn;
    for(const Point64& pt : path2)
    {
      result = PointInPolygon(pt, path1);
      if (result != PointInPolygonResult::IsOn) break;
    }
    return result;
  }

  inline bool GetLocation(const Rect64& rec,
    const Point64& pt, Location& loc)
  {
    if (pt.x == rec.left && pt.y >= rec.top && pt.y <= rec.bottom)
    {
      loc = Location::Left;
      return false;
    }
    else if (pt.x == rec.right && pt.y >= rec.top && pt.y <= rec.bottom)
    {
      loc = Location::Right;
      return false;
    }
    else if (pt.y == rec.top && pt.x >= rec.left && pt.x <= rec.right)
    {
      loc = Location::Top;
      return false;
    }
    else if (pt.y == rec.bottom && pt.x >= rec.left && pt.x <= rec.right)
    {
      loc = Location::Bottom;
      return false;
    }
    else if (pt.x < rec.left) loc = Location::Left;
    else if (pt.x > rec.right) loc = Location::Right;
    else if (pt.y < rec.top) loc = Location::Top;
    else if (pt.y > rec.bottom) loc = Location::Bottom;
    else loc = Location::Inside;
    return true;
  }

  inline bool GetIntersection(const Path64& rectPath,
    const Point64& p, const Point64& p2, Location& loc, Point64& ip)
  {
    // gets the intersection closest to 'p'
    // when Result = false, loc will remain unchanged
    switch (loc)
    {
    case Location::Left:
      if (SegmentsIntersect(p, p2, rectPath[0], rectPath[3], true))
        GetIntersectPoint(p, p2, rectPath[0], rectPath[3], ip);
      else if (p.y < rectPath[0].y &&
        SegmentsIntersect(p, p2, rectPath[0], rectPath[1], true))
      {
        GetIntersectPoint(p, p2, rectPath[0], rectPath[1], ip);
        loc = Location::Top;
      }
      else if (SegmentsIntersect(p, p2, rectPath[2], rectPath[3], true))
      {
        GetIntersectPoint(p, p2, rectPath[2], rectPath[3], ip);
        loc = Location::Bottom;
      }
      else return false;
      break;

    case Location::Top:
      if (SegmentsIntersect(p, p2, rectPath[0], rectPath[1], true))
        GetIntersectPoint(p, p2, rectPath[0], rectPath[1], ip);
      else if (p.x < rectPath[0].x &&
        SegmentsIntersect(p, p2, rectPath[0], rectPath[3], true))
      {
        GetIntersectPoint(p, p2, rectPath[0], rectPath[3], ip);
        loc = Location::Left;
      }
      else if (p.x > rectPath[1].x &&
        SegmentsIntersect(p, p2, rectPath[1], rectPath[2], true))
      {
        GetIntersectPoint(p, p2, rectPath[1], rectPath[2], ip);
        loc = Location::Right;
      }
      else return false;
        break;

    case Location::Right:
      if (SegmentsIntersect(p, p2, rectPath[1], rectPath[2], true))
        GetIntersectPoint(p, p2, rectPath[1], rectPath[2], ip);
      else if (p.y < rectPath[0].y &&
        SegmentsIntersect(p, p2, rectPath[0], rectPath[1], true))
      {
        GetIntersectPoint(p, p2, rectPath[0], rectPath[1], ip);
        loc = Location::Top;
      }
      else if (SegmentsIntersect(p, p2, rectPath[2], rectPath[3], true))
      {
        GetIntersectPoint(p, p2, rectPath[2], rectPath[3], ip);
        loc = Location::Bottom;
      }
      else return false;
      break;

    case Location::Bottom:
      if (SegmentsIntersect(p, p2, rectPath[2], rectPath[3], true))
        GetIntersectPoint(p, p2, rectPath[2], rectPath[3], ip);
      else if (p.x < rectPath[3].x &&
        SegmentsIntersect(p, p2, rectPath[0], rectPath[3], true))
      {
        GetIntersectPoint(p, p2, rectPath[0], rectPath[3], ip);
        loc = Location::Left;
      }
      else if (p.x > rectPath[2].x &&
        SegmentsIntersect(p, p2, rectPath[1], rectPath[2], true))
      {
        GetIntersectPoint(p, p2, rectPath[1], rectPath[2], ip);
        loc = Location::Right;
      }
      else return false;
      break;

    default: // loc == rInside
      if (SegmentsIntersect(p, p2, rectPath[0], rectPath[3], true))
      {
        GetIntersectPoint(p, p2, rectPath[0], rectPath[3], ip);
        loc = Location::Left;
      }
      else if (SegmentsIntersect(p, p2, rectPath[0], rectPath[1], true))
      {
        GetIntersectPoint(p, p2, rectPath[0], rectPath[1], ip);
        loc = Location::Top;
      }
      else if (SegmentsIntersect(p, p2, rectPath[1], rectPath[2], true))
      {
        GetIntersectPoint(p, p2, rectPath[1], rectPath[2], ip);
        loc = Location::Right;
      }
      else if (SegmentsIntersect(p, p2, rectPath[2], rectPath[3], true))
      {
        GetIntersectPoint(p, p2, rectPath[2], rectPath[3], ip);
        loc = Location::Bottom;
      }
      else return false;
      break;
    }
    return true;
  }

  inline Location GetAdjacentLocation(Location loc, bool isClockwise)
  {
    int delta = (isClockwise) ? 1 : 3;
    return static_cast<Location>((static_cast<int>(loc) + delta) % 4);
  }

  inline bool HeadingClockwise(Location prev, Location curr)
  {
    return (static_cast<int>(prev) + 1) % 4 == static_cast<int>(curr);
  }

  inline bool AreOpposites(Location prev, Location curr)
  {
    return abs(static_cast<int>(prev) - static_cast<int>(curr)) == 2;
  }

  inline bool IsClockwise(Location prev, Location curr,
    const Point64& prev_pt, const Point64& curr_pt, const Point64& rect_mp)
  {
    if (AreOpposites(prev, curr))
      return CrossProduct(prev_pt, rect_mp, curr_pt) < 0;
    else
      return HeadingClockwise(prev, curr);
  }

  //----------------------------------------------------------------------------
  // RectClip64
  //----------------------------------------------------------------------------

  void RectClip::AddCorner(Location prev, Location curr)
  {
    if (HeadingClockwise(prev, curr))
      result_.push_back(rectPath_[static_cast<int>(prev)]);
    else
      result_.push_back(rectPath_[static_cast<int>(curr)]);
  }

  void RectClip::AddCorner(Location& loc, bool isClockwise)
  {
    if (isClockwise)
    {
      result_.push_back(rectPath_[static_cast<int>(loc)]);
      loc = GetAdjacentLocation(loc, true);
    }
    else
    {
      loc = GetAdjacentLocation(loc, false);
      result_.push_back(rectPath_[static_cast<int>(loc)]);
    }
  }

  void RectClip::GetNextLocation(const Path64& path,
      Location& loc, int& i, int highI)
  {
    switch (loc)
    {
    case Location::Left:
      while (i <= highI && path[i].x <= rect_.left) ++i;
      if (i > highI) break;
      else if (path[i].x >= rect_.right) loc = Location::Right;
      else if (path[i].y <= rect_.top) loc = Location::Top;
      else if (path[i].y >= rect_.bottom) loc = Location::Bottom;
      else loc = Location::Inside;
      break;

    case Location::Top:
      while (i <= highI && path[i].y <= rect_.top) ++i;
      if (i > highI) break;
      else if (path[i].y >= rect_.bottom) loc = Location::Bottom;
      else if (path[i].x <= rect_.left) loc = Location::Left;
      else if (path[i].x >= rect_.right) loc = Location::Right;
      else loc = Location::Inside;
      break;

    case Location::Right:
      while (i <= highI && path[i].x >= rect_.right) ++i;
      if (i > highI) break;
      else if (path[i].x <= rect_.left) loc = Location::Left;
      else if (path[i].y <= rect_.top) loc = Location::Top;
      else if (path[i].y >= rect_.bottom) loc = Location::Bottom;
      else loc = Location::Inside;
      break;

    case Location::Bottom:
      while (i <= highI && path[i].y >= rect_.bottom) ++i;
      if (i > highI) break;
      else if (path[i].y <= rect_.top) loc = Location::Top;
      else if (path[i].x <= rect_.left) loc = Location::Left;
      else if (path[i].x >= rect_.right) loc = Location::Right;
      else loc = Location::Inside;
      break;

    case Location::Inside:
      while (i <= highI)
      {
        if (path[i].x < rect_.left) loc = Location::Left;
        else if (path[i].x > rect_.right) loc = Location::Right;
        else if (path[i].y > rect_.bottom) loc = Location::Bottom;
        else if (path[i].y < rect_.top) loc = Location::Top;
        else { result_.push_back(path[i]); ++i; continue; }
        break; //inner loop
      }
      break;
    } //switch          
  }

  Path64 RectClip::Execute(const Path64& path)
  {
    if (rect_.IsEmpty() || path.size() < 3) return Path64();

    result_.clear();
    start_locs_.clear();
    int i = 0, highI = static_cast<int>(path.size()) - 1;
    Location prev = Location::Inside, loc;
    Location crossing_loc = Location::Inside;
    Location first_cross_ = Location::Inside;
    if (!GetLocation(rect_, path[highI], loc))
    {
      i = highI - 1;
      while (i >= 0 && !GetLocation(rect_, path[i], prev)) --i;
      if (i < 0) return path;
      if (prev == Location::Inside) loc = Location::Inside;
      i = 0;
    }
    Location starting_loc = loc;

    ///////////////////////////////////////////////////
    while (i <= highI)
    {
      prev = loc;
      Location crossing_prev = crossing_loc;

      GetNextLocation(path, loc, i, highI);

      if (i > highI) break;
      Point64 ip, ip2;
      Point64 prev_pt = (i) ? path[static_cast<size_t>(i - 1)] : path[highI];

      crossing_loc = loc;
      if (!GetIntersection(rectPath_, path[i], prev_pt, crossing_loc, ip))
      {
        // ie remaining outside

        if (crossing_prev == Location::Inside)
        {
          bool isClockw = IsClockwise(prev, loc, prev_pt, path[i], mp_);
          do {
            start_locs_.push_back(prev);
            prev = GetAdjacentLocation(prev, isClockw);
          } while (prev != loc);
          crossing_loc = crossing_prev; // still not crossed 
        }
        else if (prev != Location::Inside && prev != loc)
        {
          bool isClockw = IsClockwise(prev, loc, prev_pt, path[i], mp_);
          do {
            AddCorner(prev, isClockw);
          } while (prev != loc);
        }
        ++i;
        continue;
      }

      ////////////////////////////////////////////////////
      // we must be crossing the rect boundary to get here
      ////////////////////////////////////////////////////

      if (loc == Location::Inside) // path must be entering rect
      {
        if (first_cross_ == Location::Inside)
        {
          first_cross_ = crossing_loc;
          start_locs_.push_back(prev);
        }
        else if (prev != crossing_loc)
        {
          bool isClockw = IsClockwise(prev, crossing_loc, prev_pt, path[i], mp_);
          do {
            AddCorner(prev, isClockw);
          } while (prev != crossing_loc);
        }
      }
      else if (prev != Location::Inside)
      {
        // passing right through rect. 'ip' here will be the second 
        // intersect pt but we'll also need the first intersect pt (ip2)
        loc = prev;
        GetIntersection(rectPath_, prev_pt, path[i], loc, ip2);
        if (crossing_prev != Location::Inside)
          AddCorner(crossing_prev, loc);

        if (first_cross_ == Location::Inside)
        {
          first_cross_ = loc;
          start_locs_.push_back(prev);
        }

        loc = crossing_loc;
        result_.push_back(ip2);
        if (ip == ip2)
        {
          // it's very likely that path[i] is on rect
          GetLocation(rect_, path[i], loc);
          AddCorner(crossing_loc, loc);
          crossing_loc = loc;
          continue;
        }
      }
      else // path must be exiting rect
      {
        loc = crossing_loc;
        if (first_cross_ == Location::Inside)
          first_cross_ = crossing_loc;
      }

      result_.push_back(ip);

    } //while i <= highI
    ///////////////////////////////////////////////////

    if (first_cross_ == Location::Inside)
    {
      if (starting_loc == Location::Inside) return path;
      Rect64 tmp_rect = Bounds(path);
      if (tmp_rect.Contains(rect_) &&
        Path1ContainsPath2(path, rectPath_) !=
        PointInPolygonResult::IsOutside) return rectPath_;
      else
        return Path64();
    }

    if (loc != Location::Inside &&
      (loc != first_cross_ || start_locs_.size() > 2))
    {
      if (start_locs_.size() > 0)
      {
        prev = loc;
        for (auto loc2 : start_locs_)
        {
          if (prev == loc2) continue;
          AddCorner(prev, HeadingClockwise(prev, loc2));
          prev = loc2;
        }
        loc = prev;
      }
      if (loc != first_cross_)
        AddCorner(loc, HeadingClockwise(loc, first_cross_));
    }

    if (result_.size() < 3) return Path64();

    // tidy up duplicates and collinear segments
    Path64 res;
    res.reserve(result_.size());
    size_t k = 0; highI = static_cast<int>(result_.size()) - 1;
    Point64 prev_pt = result_[highI];
    res.push_back(result_[0]);
    Path64::const_iterator cit;
    for (cit = result_.cbegin() + 1; cit != result_.cend(); ++cit)
    {
      if (CrossProduct(prev_pt, res[k], *cit))
      {
        prev_pt = res[k++];
        res.push_back(*cit);
      }
      else
        res[k] = *cit;
    }

    if (k < 2) return Path64();
    // and a final check for collinearity
    else if (!CrossProduct(res[0], res[k - 1], res[k])) res.pop_back();
    return res;
  }

  Paths64 RectClipLines::Execute(const Path64& path)
  {
    result_.clear();
    Paths64 result;
    if (rect_.IsEmpty() || path.size() == 0) return result;

    int i = 1, highI = static_cast<int>(path.size()) - 1;

    Location prev = Location::Inside, loc;
    Location crossing_loc;
    if (!GetLocation(rect_, path[0], loc))
    {
      while (i <= highI && !GetLocation(rect_, path[i], prev)) ++i;
      if (i > highI) {
        result.push_back(path);
        return result;
      }
      if (prev == Location::Inside) loc = Location::Inside;
      i = 1;
    }
    if (loc == Location::Inside) result_.push_back(path[0]);

    ///////////////////////////////////////////////////
    while (i <= highI)
    {
      prev = loc;
      GetNextLocation(path, loc, i, highI);
      if (i > highI) break;
      Point64 ip, ip2;
      Point64 prev_pt = path[static_cast<size_t>(i - 1)];

      crossing_loc = loc;
      if (!GetIntersection(rectPath_, path[i], prev_pt, crossing_loc, ip))
      {
        // ie remaining outside
        ++i;
        continue;
      }

      ////////////////////////////////////////////////////
      // we must be crossing the rect boundary to get here
      ////////////////////////////////////////////////////

      if (loc == Location::Inside) // path must be entering rect
      {
        result_.push_back(ip);
      }
      else if (prev != Location::Inside)
      {
        // passing right through rect. 'ip' here will be the second 
        // intersect pt but we'll also need the first intersect pt (ip2)
        crossing_loc = prev;
        GetIntersection(rectPath_, prev_pt, path[i], crossing_loc, ip2);
        result_.push_back(ip2);
        result_.push_back(ip);
        result.push_back(result_);
        result_.clear();
      }
      else // path must be exiting rect
      {
        result_.push_back(ip);
        result.push_back(result_);
        result_.clear();
      }
    } //while i <= highI
    ///////////////////////////////////////////////////

    if (result_.size() > 1)
      result.push_back(result_);
    return result;
  }

} // namespace

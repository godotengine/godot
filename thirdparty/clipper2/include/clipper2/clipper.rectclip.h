/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  26 October 2022                                                 *
* Website   :  http://www.angusj.com                                           *
* Copyright :  Angus Johnson 2010-2022                                         *
* Purpose   :  FAST rectangular clipping                                       *
* License   :  http://www.boost.org/LICENSE_1_0.txt                            *
*******************************************************************************/

#ifndef CLIPPER_RECTCLIP_H
#define CLIPPER_RECTCLIP_H

#include <cstdlib>
#include <vector>
#include "clipper.h"
#include "clipper.core.h"

namespace Clipper2Lib 
{

  enum class Location { Left, Top, Right, Bottom, Inside };

  class RectClip {
  protected:
    const Rect64 rect_;
    const Point64 mp_;
    const Path64 rectPath_;
    Path64 result_;
    std::vector<Location> start_locs_;

    void GetNextLocation(const Path64& path,
      Location& loc, int& i, int highI);
    void AddCorner(Location prev, Location curr);
    void AddCorner(Location& loc, bool isClockwise);
  public:
    explicit RectClip(const Rect64& rect) :
      rect_(rect),
      mp_(rect.MidPoint()),
      rectPath_(rect.AsPath()) {}
    Path64 Execute(const Path64& path);
  };

  class RectClipLines : public RectClip {
  public:
    explicit RectClipLines(const Rect64& rect) : RectClip(rect) {};
    Paths64 Execute(const Path64& path);
  };

} // Clipper2Lib namespace
#endif  // CLIPPER_RECTCLIP_H

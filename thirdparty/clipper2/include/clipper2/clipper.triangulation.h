/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  6 December 2025                                                 *
* Release   :  BETA RELEASE                                                    *
* Website   :  https://www.angusj.com                                          *
* Copyright :  Angus Johnson 2010-2025                                         *
* Purpose   :  Delaunay Triangulation                                          *
* License   :  https://www.boost.org/LICENSE_1_0.txt                           *
*******************************************************************************/

#ifndef CLIPPER_TRIANGULATION_H
#define CLIPPER_TRIANGULATION_H

#include <stack>
#include "clipper2/clipper.core.h"

namespace Clipper2Lib
{

  enum class TriangulateResult { success, fail, no_polygons, paths_intersect };

  // Triangulate - this function will not accept intesecting paths
  TriangulateResult Triangulate(const Paths64& pp, Paths64& solution, bool useDelaunay = true);
  TriangulateResult Triangulate(const PathsD& pp, int decPlaces, PathsD& solution, bool useDelaunay = true);

} // Clipper2Lib namespace
#endif  // CLIPPER_TRIANGULATION_H

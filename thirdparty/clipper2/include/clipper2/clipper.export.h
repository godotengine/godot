/*******************************************************************************
* Author    :  Angus Johnson                                                   *
* Date      :  24 January 2025                                                 *
* Website   :  https://www.angusj.com                                          *
* Copyright :  Angus Johnson 2010-2025                                         *
* Purpose   :  This module exports the Clipper2 Library (ie DLL/so)            *
* License   :  https://www.boost.org/LICENSE_1_0.txt                           *
*******************************************************************************/


/*
 Boolean clipping:
 cliptype: NoClip=0, Intersection=1, Union=2, Difference=3, Xor=4
 fillrule: EvenOdd=0, NonZero=1, Positive=2, Negative=3

 Polygon offsetting (inflate/deflate):
 jointype: Square=0, Bevel=1, Round=2, Miter=3
 endtype: Polygon=0, Joined=1, Butt=2, Square=3, Round=4

The path structures used extensively in other parts of this library are all
based on std::vector classes. Since C++ classes can't be accessed by other
languages, these paths are exported here as very simple array structures 
(either of int64_t or double) that can be parsed by just about any 
programming language.

These 2D paths are defined by series of x and y coordinates together with an
optional user-defined 'z' value (see Z-values below). Hence, a vertex refers
to a single x and y coordinate (+/- a user-defined value). Data structures 
have names with suffixes that indicate the array type (either int64_t or 
double). For example, the data structure CPath64 contains an array of int64_t 
values, whereas the data structure CPathD contains an array of double. 
Where documentation omits the type suffix (eg CPath), it is referring to an 
array whose data type could be either int64_t or double.

For conciseness, the following letters are used in the diagrams below:
N: Number of vertices in a given path
C: Count (ie number) of paths (or PolyPaths) in the structure
A: Number of elements in an array


CPath64 and CPathD:
These are arrays of either int64_t or double values. Apart from 
the first two elements, these arrays are a series of vertices 
that together define a path. The very first element contains the 
number of vertices (N) in the path, while second element should 
contain a 0 value.
_______________________________________________________________
| counters | vertex1      | vertex2      | ... | vertexN      |
| N, 0     | x1, y1, (z1) | x2, y2, (z2) | ... | xN, yN, (zN) |
---------------------------------------------------------------


CPaths64 and CPathsD:
These are also arrays of either int64_t or double values that
contain any number of consecutive CPath structures. However, 
preceding the first path is a pair of values. The first value
contains the length of the entire array structure (A), and the 
second contains the number (ie count) of contained paths (C).
  Memory allocation for CPaths64 = A * sizeof(int64_t)
  Memory allocation for CPathsD  = A * sizeof(double)
__________________________________________
| counters | path1 | path2 | ... | pathC |
| A, C     |       |       | ... |       |
------------------------------------------


CPolytree64 and CPolytreeD:
The entire polytree structure is an array of int64_t or double. The 
first element in the array indicates the array's total length (A). 
The second element indicates the number (C) of CPolyPath structures 
that are the TOP LEVEL CPolyPath in the polytree, and these top
level CPolyPath immediately follow these first two array elements. 
These top level CPolyPath structures may, in turn, contain nested 
CPolyPath children, and these collectively make a tree structure.
_________________________________________________________
| counters | CPolyPath1 | CPolyPath2 | ... | CPolyPathC |
| A, C     |            |            | ... |            |
---------------------------------------------------------


CPolyPath64 and CPolyPathD:
These array structures consist of a pair of counter values followed by a
series of polygon vertices and a series of nested CPolyPath children.
The first counter values indicates the number of vertices in the
polygon (N), and the second counter indicates the CPolyPath child count (C).
_____________________________________________________________________________
|cntrs |vertex1     |vertex2      |...|vertexN     |child1|child2|...|childC|
|N, C  |x1, y1, (z1)| x2, y2, (z2)|...|xN, yN, (zN)|      |      |...|      |
-----------------------------------------------------------------------------


DisposeArray64 & DisposeArrayD:
All array structures are allocated in heap memory which will eventually
need to be released. However, since applications linking to these DLL
functions may use different memory managers, the only safe way to release
this memory is to use the exported DisposeArray functions.


(Optional) Z-Values:
Structures will only contain user-defined z-values when the USINGZ
pre-processor identifier is used. The library does not assign z-values
because this field is intended for users to assign custom values to vertices.
Z-values in input paths (subject and clip) will be copied to solution paths.
New vertices at path intersections will generate a callback event that allows
users to assign z-values at these new vertices. The user's callback function
must conform with the DLLZCallback definition and be registered with the
DLL via SetZCallback. To assist the user in assigning z-values, the library
passes in the callback function the new intersection point together with
the four vertices that define the two segments that are intersecting.

*/
#ifndef CLIPPER2_EXPORT_H
#define CLIPPER2_EXPORT_H

#include "clipper2/clipper.core.h"
#include "clipper2/clipper.engine.h"
#include "clipper2/clipper.offset.h"
#include "clipper2/clipper.rectclip.h"
#include <cstdlib>

namespace Clipper2Lib {

typedef int64_t* CPath64;
typedef int64_t* CPaths64;
typedef double*  CPathD;
typedef double*  CPathsD;

typedef int64_t* CPolyPath64;
typedef int64_t* CPolyTree64;
typedef double* CPolyPathD;
typedef double* CPolyTreeD;

template <typename T>
struct CRect {
  T left;
  T top;
  T right;
  T bottom;
};

typedef CRect<int64_t> CRect64;
typedef CRect<double> CRectD;

template <typename T>
inline bool CRectIsEmpty(const CRect<T>& rect)
{
  return (rect.right <= rect.left) || (rect.bottom <= rect.top);
}

template <typename T>
inline Rect<T> CRectToRect(const CRect<T>& rect)
{
  Rect<T> result;
  result.left = rect.left;
  result.top = rect.top;
  result.right = rect.right;
  result.bottom = rect.bottom;
  return result;
}

template <typename T1, typename T2>
inline T1 Reinterpret(T2 value) {
  return *reinterpret_cast<T1*>(&value);
}


#ifdef _WIN32
  #define EXTERN_DLL_EXPORT extern "C" __declspec(dllexport)
#else
  #define EXTERN_DLL_EXPORT extern "C"
#endif


//////////////////////////////////////////////////////
// EXPORTED FUNCTION DECLARATIONS
//////////////////////////////////////////////////////

EXTERN_DLL_EXPORT const char* Version();

EXTERN_DLL_EXPORT void DisposeArray64(int64_t*& p)
{
  delete[] p;
}

EXTERN_DLL_EXPORT void DisposeArrayD(double*& p)
{
  delete[] p;
}

EXTERN_DLL_EXPORT int BooleanOp64(uint8_t cliptype,
  uint8_t fillrule, const CPaths64 subjects,
  const CPaths64 subjects_open, const CPaths64 clips,
  CPaths64& solution, CPaths64& solution_open,
  bool preserve_collinear = true, bool reverse_solution = false);

EXTERN_DLL_EXPORT int BooleanOp_PolyTree64(uint8_t cliptype,
  uint8_t fillrule, const CPaths64 subjects,
  const CPaths64 subjects_open, const CPaths64 clips,
  CPolyTree64& sol_tree, CPaths64& solution_open,
  bool preserve_collinear = true, bool reverse_solution = false);

EXTERN_DLL_EXPORT int BooleanOpD(uint8_t cliptype,
  uint8_t fillrule, const CPathsD subjects,
  const CPathsD subjects_open, const CPathsD clips,
  CPathsD& solution, CPathsD& solution_open, int precision = 2,
  bool preserve_collinear = true, bool reverse_solution = false);

EXTERN_DLL_EXPORT int BooleanOp_PolyTreeD(uint8_t cliptype,
  uint8_t fillrule, const CPathsD subjects,
  const CPathsD subjects_open, const CPathsD clips,
  CPolyTreeD& solution, CPathsD& solution_open, int precision = 2,
  bool preserve_collinear = true, bool reverse_solution = false);

EXTERN_DLL_EXPORT CPaths64 InflatePaths64(const CPaths64 paths,
  double delta, uint8_t jointype, uint8_t endtype,
  double miter_limit = 2.0, double arc_tolerance = 0.0,
  bool reverse_solution = false);

EXTERN_DLL_EXPORT CPathsD InflatePathsD(const CPathsD paths,
  double delta, uint8_t jointype, uint8_t endtype,
  int precision = 2, double miter_limit = 2.0,
  double arc_tolerance = 0.0, bool reverse_solution = false);

EXTERN_DLL_EXPORT CPaths64 InflatePath64(const CPath64 path,
    double delta, uint8_t jointype, uint8_t endtype,
    double miter_limit = 2.0, double arc_tolerance = 0.0,
    bool reverse_solution = false);

EXTERN_DLL_EXPORT CPathsD InflatePathD(const CPathD path,
    double delta, uint8_t jointype, uint8_t endtype,
    int precision = 2, double miter_limit = 2.0,
    double arc_tolerance = 0.0, bool reverse_solution = false);

// RectClip & RectClipLines:
EXTERN_DLL_EXPORT CPaths64 RectClip64(const CRect64& rect,
  const CPaths64 paths);
EXTERN_DLL_EXPORT CPathsD RectClipD(const CRectD& rect,
  const CPathsD paths, int precision = 2);
EXTERN_DLL_EXPORT CPaths64 RectClipLines64(const CRect64& rect,
  const CPaths64 paths);
EXTERN_DLL_EXPORT CPathsD RectClipLinesD(const CRectD& rect,
  const CPathsD paths, int precision = 2);

//////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
//////////////////////////////////////////////////////

#ifdef USINGZ
ZCallback64 dllCallback64 = nullptr;
ZCallbackD  dllCallbackD  = nullptr;

constexpr int EXPORT_VERTEX_DIMENSIONALITY = 3;
#else    
constexpr int EXPORT_VERTEX_DIMENSIONALITY  = 2;
#endif 

template <typename T>
static void GetPathCountAndCPathsArrayLen(const Paths<T>& paths,
  size_t& cnt, size_t& array_len)
{
  array_len = 2;
  cnt = 0;
  for (const Path<T>& path : paths)
    if (path.size())
    {
      array_len += path.size() * EXPORT_VERTEX_DIMENSIONALITY + 2;
      ++cnt;
    }
}

static size_t GetPolyPathArrayLen64(const PolyPath64& pp)
{
  size_t result = 2; // poly_length + child_count
  result += pp.Polygon().size() * EXPORT_VERTEX_DIMENSIONALITY;
  //plus nested children :)
  for (size_t i = 0; i < pp.Count(); ++i)
    result += GetPolyPathArrayLen64(*pp[i]);
  return result;
}

static size_t GetPolyPathArrayLenD(const PolyPathD& pp)
{
  size_t result = 2; // poly_length + child_count
  result += pp.Polygon().size() * EXPORT_VERTEX_DIMENSIONALITY;
  //plus nested children :)
  for (size_t i = 0; i < pp.Count(); ++i)
    result += GetPolyPathArrayLenD(*pp[i]);
  return result;
}

static void GetPolytreeCountAndCStorageSize64(const PolyTree64& tree,
  size_t& cnt, size_t& array_len)
{
  cnt = tree.Count(); // nb: top level count only
  array_len = GetPolyPathArrayLen64(tree);
}

static void GetPolytreeCountAndCStorageSizeD(const PolyTreeD& tree,
  size_t& cnt, size_t& array_len)
{
  cnt = tree.Count(); // nb: top level count only
  array_len = GetPolyPathArrayLenD(tree);
}

template <typename T>
static T* CreateCPathsFromPathsT(const Paths<T>& paths)
{
  size_t cnt = 0, array_len = 0;
  GetPathCountAndCPathsArrayLen(paths, cnt, array_len);
  T* result = new T[array_len], * v = result;
  *v++ = array_len;
  *v++ = cnt;
  for (const Path<T>& path : paths)
  {
    if (!path.size()) continue;
    *v++ = path.size();
    *v++ = 0;
    for (const Point<T>& pt : path)
    {
      *v++ = pt.x;
      *v++ = pt.y;
#ifdef USINGZ
      *v++ = Reinterpret<T>(pt.z);
#endif
    }
  }
  return result;
}

CPathsD CreateCPathsDFromPathsD(const PathsD& paths)
{
  if (!paths.size()) return nullptr;
  size_t cnt, array_len;
  GetPathCountAndCPathsArrayLen(paths, cnt, array_len);
  CPathsD result = new double[array_len], v = result;
  *v++ = (double)array_len;
  *v++ = (double)cnt;
  for (const PathD& path : paths)
  {
    if (!path.size()) continue;
    *v = (double)path.size();
    ++v; *v++ = 0;
    for (const PointD& pt : path)
    {
      *v++ = pt.x;
      *v++ = pt.y;
#ifdef USINGZ
      * v++ = Reinterpret<double>(pt.z);
#endif
    }
  }
  return result;
}

CPathsD CreateCPathsDFromPaths64(const Paths64& paths, double scale)
{
  if (!paths.size()) return nullptr;
  size_t cnt, array_len;
  GetPathCountAndCPathsArrayLen(paths, cnt, array_len);
  CPathsD result = new double[array_len], v = result;
  *v++ = (double)array_len;
  *v++ = (double)cnt;
  for (const Path64& path : paths)
  {
    if (!path.size()) continue;
    *v = (double)path.size();
    ++v; *v++ = 0;
    for (const Point64& pt : path)
    {
      *v++ = pt.x * scale;
      *v++ = pt.y * scale;
#ifdef USINGZ
      *v++ = Reinterpret<double>(pt.z);
#endif
    }
  }
  return result;
}

template <typename T>
static Path<T> ConvertCPathToPathT(T* path)
{
  Path<T> result;
  if (!path) return result;
  T* v = path;
  size_t cnt = static_cast<size_t>(*v);
  v += 2; // skip 0 value
  result.reserve(cnt);
  for (size_t j = 0; j < cnt; ++j)
  {
      T x = *v++, y = *v++;
#ifdef USINGZ
      z_type z = Reinterpret<z_type>(*v++);
      result.emplace_back(x, y, z);
#else  
      result.emplace_back(x, y);
#endif
  }
  return result;
}

template <typename T>
static Paths<T> ConvertCPathsToPathsT(T* paths)
{
  Paths<T> result;
  if (!paths) return result;
  T* v = paths; ++v;
  size_t cnt = static_cast<size_t>(*v++);
  result.reserve(cnt);
  for (size_t i = 0; i < cnt; ++i)
  {
    size_t cnt2 = static_cast<size_t>(*v);
    v += 2; 
    Path<T> path;
    path.reserve(cnt2);
    for (size_t j = 0; j < cnt2; ++j)
    {
      T x = *v++, y = *v++;
#ifdef USINGZ
      z_type z = Reinterpret<z_type>(*v++);
      path.emplace_back(x, y, z);
#else
      path.emplace_back(x, y);
#endif
    }
    result.emplace_back(std::move(path));
  }
  return result;
}

static Path64 ConvertCPathDToPath64WithScale(const CPathD path, double scale)
{
    Path64 result;
    if (!path) return result;
    double* v = path;
    size_t cnt = static_cast<size_t>(*v);
    v += 2; // skip 0 value
    result.reserve(cnt);
    for (size_t j = 0; j < cnt; ++j)
    {
        double x = *v++ * scale;
        double y = *v++ * scale;
#ifdef USINGZ
        z_type z = Reinterpret<z_type>(*v++);
        result.emplace_back(x, y, z);
#else  
        result.emplace_back(x, y);
#endif
    }
    return result;
}

static Paths64 ConvertCPathsDToPaths64(const CPathsD paths, double scale)
{
  Paths64 result;
  if (!paths) return result;
  double* v = paths;
  ++v; // skip the first value (0)
  size_t cnt = static_cast<size_t>(*v++);
  result.reserve(cnt);
  for (size_t i = 0; i < cnt; ++i)
  {
    size_t cnt2 = static_cast<size_t>(*v);
    v += 2;
    Path64 path;
    path.reserve(cnt2);
    for (size_t j = 0; j < cnt2; ++j)
    {
      double x = *v++ * scale;
      double y = *v++ * scale;
#ifdef USINGZ
      z_type z = Reinterpret<z_type>(*v++);
      path.emplace_back(x, y, z);
#else
      path.emplace_back(x, y);
#endif
    }
    result.emplace_back(std::move(path));
  }
  return result;
}

static void CreateCPolyPath64(const PolyPath64* pp, int64_t*& v)
{
  *v++ = static_cast<int64_t>(pp->Polygon().size());
  *v++ = static_cast<int64_t>(pp->Count());
  for (const Point64& pt : pp->Polygon())
  {
    *v++ = pt.x;
    *v++ = pt.y;
#ifdef USINGZ   
    * v++ = Reinterpret<int64_t>(pt.z); // raw memory copy
#endif
  }
  for (size_t i = 0; i < pp->Count(); ++i)
    CreateCPolyPath64(pp->Child(i), v);
}

static void CreateCPolyPathD(const PolyPathD* pp, double*& v)
{
  *v++ = static_cast<double>(pp->Polygon().size());
  *v++ = static_cast<double>(pp->Count());
  for (const PointD& pt : pp->Polygon())
  {
    *v++ = pt.x;
    *v++ = pt.y;
#ifdef USINGZ   
    * v++ = Reinterpret<double>(pt.z); // raw memory copy
#endif
  }
  for (size_t i = 0; i < pp->Count(); ++i)
    CreateCPolyPathD(pp->Child(i), v);
}

static int64_t* CreateCPolyTree64(const PolyTree64& tree)
{
  size_t cnt, array_len;
  GetPolytreeCountAndCStorageSize64(tree, cnt, array_len);
  if (!cnt) return nullptr;
  // allocate storage
  int64_t* result = new int64_t[array_len];
  int64_t* v = result;
  *v++ = static_cast<int64_t>(array_len);
  *v++ = static_cast<int64_t>(tree.Count());
  for (size_t i = 0; i < tree.Count(); ++i)
    CreateCPolyPath64(tree.Child(i), v);
  return result;
}

static double* CreateCPolyTreeD(const PolyTreeD& tree)
{
  double scale = std::log10(tree.Scale());
  size_t cnt, array_len;
  GetPolytreeCountAndCStorageSizeD(tree, cnt, array_len);
  if (!cnt) return nullptr;
  // allocate storage
  double* result = new double[array_len];
  double* v = result;
  *v++ = static_cast<double>(array_len);
  *v++ = static_cast<double>(tree.Count());
  for (size_t i = 0; i < tree.Count(); ++i)
    CreateCPolyPathD(tree.Child(i), v);
  return result;
}

//////////////////////////////////////////////////////
// EXPORTED FUNCTION DEFINITIONS
//////////////////////////////////////////////////////

EXTERN_DLL_EXPORT const char* Version()
{
  return CLIPPER2_VERSION;
}

EXTERN_DLL_EXPORT int BooleanOp64(uint8_t cliptype,
  uint8_t fillrule, const CPaths64 subjects,
  const CPaths64 subjects_open, const CPaths64 clips,
  CPaths64& solution, CPaths64& solution_open,
  bool preserve_collinear, bool reverse_solution)
{
  if (cliptype > static_cast<uint8_t>(ClipType::Xor)) return -4;
  if (fillrule > static_cast<uint8_t>(FillRule::Negative)) return -3;

  Paths64 sub, sub_open, clp, sol, sol_open;
  sub       = ConvertCPathsToPathsT(subjects);
  sub_open  = ConvertCPathsToPathsT(subjects_open);
  clp       = ConvertCPathsToPathsT(clips);

  Clipper64 clipper;
  clipper.PreserveCollinear(preserve_collinear);
  clipper.ReverseSolution(reverse_solution);
#ifdef USINGZ
  if (dllCallback64)
    clipper.SetZCallback(dllCallback64);
#endif
  if (sub.size() > 0) clipper.AddSubject(sub);
  if (sub_open.size() > 0) clipper.AddOpenSubject(sub_open);
  if (clp.size() > 0) clipper.AddClip(clp);
  if (!clipper.Execute(ClipType(cliptype), FillRule(fillrule), sol, sol_open))
    return -1; // clipping bug - should never happen :)
  solution = CreateCPathsFromPathsT(sol);
  solution_open = CreateCPathsFromPathsT(sol_open);
  return 0; //success !!
}

EXTERN_DLL_EXPORT int BooleanOp_PolyTree64(uint8_t cliptype,
  uint8_t fillrule, const CPaths64 subjects,
  const CPaths64 subjects_open, const CPaths64 clips,
  CPolyTree64& sol_tree, CPaths64& solution_open,
  bool preserve_collinear, bool reverse_solution)
{
  if (cliptype > static_cast<uint8_t>(ClipType::Xor)) return -4;
  if (fillrule > static_cast<uint8_t>(FillRule::Negative)) return -3;
  Paths64 sub, sub_open, clp, sol_open;
  sub = ConvertCPathsToPathsT(subjects);
  sub_open = ConvertCPathsToPathsT(subjects_open);
  clp = ConvertCPathsToPathsT(clips);

  PolyTree64 tree;
  Clipper64 clipper;
  clipper.PreserveCollinear(preserve_collinear);
  clipper.ReverseSolution(reverse_solution);
#ifdef USINGZ
  if (dllCallback64)
    clipper.SetZCallback(dllCallback64);
#endif
  if (sub.size() > 0) clipper.AddSubject(sub);
  if (sub_open.size() > 0) clipper.AddOpenSubject(sub_open);
  if (clp.size() > 0) clipper.AddClip(clp);
  if (!clipper.Execute(ClipType(cliptype), FillRule(fillrule), tree, sol_open))
    return -1; // clipping bug - should never happen :)

  sol_tree = CreateCPolyTree64(tree);
  solution_open = CreateCPathsFromPathsT(sol_open);
  return 0; //success !!
}

EXTERN_DLL_EXPORT int BooleanOpD(uint8_t cliptype,
  uint8_t fillrule, const CPathsD subjects,
  const CPathsD subjects_open, const CPathsD clips,
  CPathsD& solution, CPathsD& solution_open, int precision,
  bool preserve_collinear, bool reverse_solution)
{
  if (precision < -8 || precision > 8) return -5;
  if (cliptype > static_cast<uint8_t>(ClipType::Xor)) return -4;
  if (fillrule > static_cast<uint8_t>(FillRule::Negative)) return -3;
  //const double scale = std::pow(10, precision);

  PathsD sub, sub_open, clp, sol, sol_open;
  sub       = ConvertCPathsToPathsT(subjects);
  sub_open  = ConvertCPathsToPathsT(subjects_open);
  clp       = ConvertCPathsToPathsT(clips);

  ClipperD clipper(precision);
  clipper.PreserveCollinear(preserve_collinear);
  clipper.ReverseSolution(reverse_solution);
#ifdef USINGZ
  if (dllCallbackD)
    clipper.SetZCallback(dllCallbackD);
#endif
  if (sub.size() > 0) clipper.AddSubject(sub);
  if (sub_open.size() > 0) clipper.AddOpenSubject(sub_open);
  if (clp.size() > 0) clipper.AddClip(clp);
  if (!clipper.Execute(ClipType(cliptype),
    FillRule(fillrule), sol, sol_open)) return -1;
  solution = CreateCPathsDFromPathsD(sol);
  solution_open = CreateCPathsDFromPathsD(sol_open);
  return 0;
}

EXTERN_DLL_EXPORT int BooleanOp_PolyTreeD(uint8_t cliptype,
  uint8_t fillrule, const CPathsD subjects,
  const CPathsD subjects_open, const CPathsD clips,
  CPolyTreeD& solution, CPathsD& solution_open, int precision,
  bool preserve_collinear, bool reverse_solution)
{
  if (precision < -8 || precision > 8) return -5;
  if (cliptype > static_cast<uint8_t>(ClipType::Xor)) return -4;
  if (fillrule > static_cast<uint8_t>(FillRule::Negative)) return -3;
  //double scale = std::pow(10, precision);

  int err = 0;
  PathsD sub, sub_open, clp, sol_open;
  sub       = ConvertCPathsToPathsT(subjects);
  sub_open  = ConvertCPathsToPathsT(subjects_open);
  clp       = ConvertCPathsToPathsT(clips);

  PolyTreeD tree;
  ClipperD clipper(precision);
  clipper.PreserveCollinear(preserve_collinear);
  clipper.ReverseSolution(reverse_solution);
#ifdef USINGZ
  if (dllCallbackD)
    clipper.SetZCallback(dllCallbackD);
#endif
  if (sub.size() > 0) clipper.AddSubject(sub);
  if (sub_open.size() > 0) clipper.AddOpenSubject(sub_open);
  if (clp.size() > 0) clipper.AddClip(clp);
  if (!clipper.Execute(ClipType(cliptype), FillRule(fillrule), tree, sol_open))
    return -1; // clipping bug - should never happen :)

  solution = CreateCPolyTreeD(tree);
  solution_open = CreateCPathsDFromPathsD(sol_open);
  return 0; //success !!
}

EXTERN_DLL_EXPORT CPaths64 InflatePaths64(const CPaths64 paths,
  double delta, uint8_t jointype, uint8_t endtype, double miter_limit,
  double arc_tolerance, bool reverse_solution)
{
  Paths64 pp;
  pp = ConvertCPathsToPathsT(paths);
  ClipperOffset clip_offset( miter_limit,
    arc_tolerance, reverse_solution);
  clip_offset.AddPaths(pp, JoinType(jointype), EndType(endtype));
  Paths64 result;
  clip_offset.Execute(delta, result);
  return CreateCPathsFromPathsT(result);
}

EXTERN_DLL_EXPORT CPathsD InflatePathsD(const CPathsD paths,
  double delta, uint8_t jointype, uint8_t endtype,
  int precision, double miter_limit,
  double arc_tolerance, bool reverse_solution)
{
  if (precision < -8 || precision > 8 || !paths) return nullptr;

  const double scale = std::pow(10, precision);
  ClipperOffset clip_offset(miter_limit, arc_tolerance, reverse_solution);
  Paths64 pp = ConvertCPathsDToPaths64(paths, scale);
  clip_offset.AddPaths(pp, JoinType(jointype), EndType(endtype));
  Paths64 result;
  clip_offset.Execute(delta * scale, result);
  return CreateCPathsDFromPaths64(result, 1 / scale);
}


EXTERN_DLL_EXPORT CPaths64 InflatePath64(const CPath64 path,
    double delta, uint8_t jointype, uint8_t endtype, double miter_limit,
    double arc_tolerance, bool reverse_solution)
{
    Path64 pp;
    pp = ConvertCPathToPathT(path);
    ClipperOffset clip_offset(miter_limit,
        arc_tolerance, reverse_solution);
    clip_offset.AddPath(pp, JoinType(jointype), EndType(endtype));
    Paths64 result;
    clip_offset.Execute(delta, result);
    return CreateCPathsFromPathsT(result);
}

EXTERN_DLL_EXPORT CPathsD InflatePathD(const CPathD path,
    double delta, uint8_t jointype, uint8_t endtype,
    int precision, double miter_limit,
    double arc_tolerance, bool reverse_solution)
{
    if (precision < -8 || precision > 8 || !path) return nullptr;

    const double scale = std::pow(10, precision);
    ClipperOffset clip_offset(miter_limit, arc_tolerance, reverse_solution);
    Path64 pp = ConvertCPathDToPath64WithScale(path, scale);
    clip_offset.AddPath(pp, JoinType(jointype), EndType(endtype));
    Paths64 result;
    clip_offset.Execute(delta * scale, result);

    return CreateCPathsDFromPaths64(result, 1 / scale);
}

EXTERN_DLL_EXPORT CPaths64 RectClip64(const CRect64& rect, const CPaths64 paths)
{
  if (CRectIsEmpty(rect) || !paths) return nullptr;
  Rect64 r64 = CRectToRect(rect);
  class RectClip64 rc(r64);
  Paths64 pp = ConvertCPathsToPathsT(paths);
  Paths64 result = rc.Execute(pp);
  return CreateCPathsFromPathsT(result);
}

EXTERN_DLL_EXPORT CPathsD RectClipD(const CRectD& rect, const CPathsD paths, int precision)
{
  if (CRectIsEmpty(rect) || !paths) return nullptr;
  if (precision < -8 || precision > 8) return nullptr;
  const double scale = std::pow(10, precision);

  RectD r = CRectToRect(rect);
  Rect64 rec = ScaleRect<int64_t, double>(r, scale);
  Paths64 pp = ConvertCPathsDToPaths64(paths, scale);
  class RectClip64 rc(rec);
  Paths64 result = rc.Execute(pp);

  return CreateCPathsDFromPaths64(result, 1 / scale);
}

EXTERN_DLL_EXPORT CPaths64 RectClipLines64(const CRect64& rect,
  const CPaths64 paths)
{
  if (CRectIsEmpty(rect) || !paths) return nullptr;
  Rect64 r = CRectToRect(rect);
  class RectClipLines64 rcl (r);
  Paths64 pp = ConvertCPathsToPathsT(paths);
  Paths64 result = rcl.Execute(pp);
  return CreateCPathsFromPathsT(result);
}

EXTERN_DLL_EXPORT CPathsD RectClipLinesD(const CRectD& rect,
  const CPathsD paths, int precision)
{
  if (CRectIsEmpty(rect) || !paths) return nullptr;
  if (precision < -8 || precision > 8) return nullptr;

  const double scale = std::pow(10, precision);
  Rect64 r = ScaleRect<int64_t, double>(CRectToRect(rect), scale);
  class RectClipLines64 rcl(r);
  Paths64 pp = ConvertCPathsDToPaths64(paths, scale);
  Paths64 result = rcl.Execute(pp);
  return CreateCPathsDFromPaths64(result, 1 / scale);
}

EXTERN_DLL_EXPORT CPaths64 MinkowskiSum64(const CPath64& cpattern, const CPath64& cpath, bool is_closed)
{
  Path64 path = ConvertCPathToPathT(cpath);
  Path64 pattern = ConvertCPathToPathT(cpattern);
  Paths64 solution = MinkowskiSum(pattern, path, is_closed);
  return CreateCPathsFromPathsT(solution);
}

EXTERN_DLL_EXPORT CPaths64 MinkowskiDiff64(const CPath64& cpattern, const CPath64& cpath, bool is_closed)
{
  Path64 path = ConvertCPathToPathT(cpath);
  Path64 pattern = ConvertCPathToPathT(cpattern);
  Paths64 solution = MinkowskiDiff(pattern, path, is_closed);
  return CreateCPathsFromPathsT(solution);
}

#ifdef USINGZ
typedef void (*DLLZCallback64)(const Point64& e1bot, const Point64& e1top, const Point64& e2bot, const Point64& e2top, Point64& pt);
typedef void (*DLLZCallbackD)(const PointD& e1bot, const PointD& e1top, const PointD& e2bot, const PointD& e2top, PointD& pt);

EXTERN_DLL_EXPORT void SetZCallback64(DLLZCallback64 callback)
{
  dllCallback64 = callback;
}

EXTERN_DLL_EXPORT void SetZCallbackD(DLLZCallbackD callback)
{
  dllCallbackD = callback;
}

#endif

}
#endif  // CLIPPER2_EXPORT_H

## Earcut

A C++ port of [earcut.js](https://github.com/mapbox/earcut), a fast, [header-only](https://github.com/mapbox/earcut.hpp/blob/master/include/mapbox/earcut.hpp) polygon triangulation library.

[![Travis](https://img.shields.io/travis/com/mapbox/earcut.hpp.svg)](https://travis-ci.com/github/mapbox/earcut.hpp)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/a1ysrqd69mqn7coo/branch/master?svg=true)](https://ci.appveyor.com/project/Mapbox/earcut-hpp-8wm4o/branch/master)
[![Coverage](https://img.shields.io/coveralls/github/mapbox/earcut.hpp.svg)](https://coveralls.io/github/mapbox/earcut.hpp)
[![Coverity Scan](https://img.shields.io/coverity/scan/14000.svg)](https://scan.coverity.com/projects/14000)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/mapbox/earcut.hpp.svg)](http://isitmaintained.com/project/mapbox/earcut.hpp "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/mapbox/earcut.hpp.svg)](http://isitmaintained.com/project/mapbox/earcut.hpp "Percentage of issues still open")
[![Mourner](https://img.shields.io/badge/simply-awesome-brightgreen.svg)](https://github.com/mourner/projects)

The library implements a modified ear slicing algorithm, optimized by [z-order curve](http://en.wikipedia.org/wiki/Z-order_curve) hashing and extended to handle holes, twisted polygons, degeneracies and self-intersections in a way that doesn't _guarantee_ correctness of triangulation, but attempts to always produce acceptable results for practical data like geographical shapes.

It's based on ideas from [FIST: Fast Industrial-Strength Triangulation of Polygons](http://www.cosy.sbg.ac.at/~held/projects/triang/triang.html) by Martin Held and [Triangulation by Ear Clipping](http://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf) by David Eberly.

## Usage

```cpp
#include <earcut.hpp>
```
```cpp
// The number type to use for tessellation
using Coord = double;

// The index type. Defaults to uint32_t, but you can also pass uint16_t if you know that your
// data won't have more than 65536 vertices.
using N = uint32_t;

// Create array
using Point = std::array<Coord, 2>;
std::vector<std::vector<Point>> polygon;

// Fill polygon structure with actual data. Any winding order works.
// The first polyline defines the main polygon.
polygon.push_back({{100, 0}, {100, 100}, {0, 100}, {0, 0}});
// Following polylines define holes.
polygon.push_back({{75, 25}, {75, 75}, {25, 75}, {25, 25}});

// Run tessellation
// Returns array of indices that refer to the vertices of the input polygon.
// e.g: the index 6 would refer to {25, 75} in this example.
// Three subsequent indices form a triangle. Output triangles are clockwise.
std::vector<N> indices = mapbox::earcut<N>(polygon);
```

Earcut can triangulate a simple, planar polygon of any winding order including holes. It will even return a robust, acceptable solution for non-simple poygons. Earcut works on a 2D plane. If you have three or more dimensions, you can project them onto a 2D surface before triangulation, or use a more suitable library for the task (e.g [CGAL](https://doc.cgal.org/latest/Triangulation_3/index.html)).


It is also possible to use your custom point type as input. There are default accessors defined for `std::tuple`, `std::pair`, and `std::array`. For a custom type (like Clipper's `IntPoint` type), do this:

```cpp
// struct IntPoint {
//     int64_t X, Y;
// };

namespace mapbox {
namespace util {

template <>
struct nth<0, IntPoint> {
    inline static auto get(const IntPoint &t) {
        return t.X;
    };
};
template <>
struct nth<1, IntPoint> {
    inline static auto get(const IntPoint &t) {
        return t.Y;
    };
};

} // namespace util
} // namespace mapbox
```

You can also use a custom container type for your polygon. Similar to std::vector<T>, it has to meet the requirements of [Container](https://en.cppreference.com/w/cpp/named_req/Container), in particular `size()`, `empty()` and `operator[]`.

<p align="center">
  <img src="https://camo.githubusercontent.com/01836f8ba21af844c93d8d3145f4e9976025a696/68747470733a2f2f692e696d6775722e636f6d2f67314e704c54712e706e67" alt="example triangulation"/>
</p>

## Additional build instructions
In case you just want to use the earcut triangulation library; copy and include the header file [`<earcut.hpp>`](https://github.com/mapbox/earcut.hpp/blob/master/include/mapbox/earcut.hpp) in your project and follow the steps documented in the section [Usage](#usage).

If you want to build the test, benchmark and visualization programs instead, follow these instructions:

### Dependencies

Before you continue, make sure to have the following tools and libraries installed:
 * git ([Ubuntu](https://help.ubuntu.com/lts/serverguide/git.html)/[Windows/macOS](http://git-scm.com/downloads))
 * cmake 3.2+ ([Ubuntu](https://launchpad.net/~george-edison55/+archive/ubuntu/cmake-3.x)/[Windows/macOS](https://cmake.org/download/))
 * OpenGL SDK ([Ubuntu](http://packages.ubuntu.com/de/trusty/libgl1-mesa-dev)/[Windows](https://dev.windows.com/en-us/downloads/windows-10-sdk)/[macOS](https://developer.apple.com/opengl/))
 * Compiler such as [GCC 4.9+, Clang 3.4+](https://launchpad.net/~ubuntu-toolchain-r/+archive/ubuntu/test), [MSVC12+](https://www.visualstudio.com/)

Note: On some operating systems such as Windows, manual steps are required to add cmake and [git](http://blog.countableset.ch/2012/06/07/adding-git-to-windows-7-path/) to your PATH environment variable.

### Manual compilation

```bash
git clone --recursive https://github.com/mapbox/earcut.hpp.git
cd earcut.hpp
mkdir build
cd build
cmake ..
make
# ./tests
# ./bench
# ./viz
```

### [Visual Studio](https://www.visualstudio.com/), [Eclipse](https://eclipse.org/), [XCode](https://developer.apple.com/xcode/), ...

```batch
git clone --recursive https://github.com/mapbox/earcut.hpp.git
cd earcut.hpp
mkdir project
cd project
cmake .. -G "Visual Studio 14 2015"
::you can also generate projects for "Visual Studio 12 2013", "XCode", "Eclipse CDT4 - Unix Makefiles"
```
After completion, open the generated project with your IDE.


### [CLion](https://www.jetbrains.com/clion/), [Visual Studio 2017+](https://www.visualstudio.com/)

Import the project from https://github.com/mapbox/earcut.hpp.git and you should be good to go!

## Status

This is currently based on [earcut 2.2.4](https://github.com/mapbox/earcut#224-jul-5-2022).

cd ../../thirdparty/

rm -rf embree
git clone https://github.com/embree/embree.git

cd embree

HASH=$(git rev-parse HEAD)
echo "$HASH"

#rm -rf CHANGELOG.md
rm -rf CMakeLists.txt
#rm -rf common
rm -rf CTestConfig.cmake
rm -rf doc
rm -rf .git
rm -rf .gitattributes
rm -rf .gitignore
rm -rf .gitlab-ci.yml
#rm -rf include
#rm -rf kernels
rm -rf LICENSE.txt
rm -rf man
rm -rf README.md
rm -rf readme.pdf
rm -rf scripts
rm -rf tutorials

rm -rf common/cmake
rm -rf common/CMakeLists.txt
rm -rf common/algorithms/CMakeLists.txt

printf "// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the \"License\");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an \"AS IS\" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#define RTC_HASH \"%s\"
" "$HASH" > kernels/hash.h

printf "// ======================================================================== //
// Copyright 2009-2016 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the \"License\");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an \"AS IS\" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/* #undef EMBREE_RAY_MASK */
/* #undef EMBREE_STAT_COUNTERS */
/* #undef EMBREE_BACKFACE_CULLING */
#define EMBREE_FILTER_FUNCTION
/* #undef EMBREE_RETURN_SUBDIV_NORMAL */
/* #undef EMBREE_IGNORE_INVALID_RAYS */
#define EMBREE_GEOMETRY_TRIANGLE
#define EMBREE_GEOMETRY_QUAD
#define EMBREE_GEOMETRY_CURVE
#define EMBREE_GEOMETRY_SUBDIVISION
#define EMBREE_GEOMETRY_USER
#define EMBREE_GEOMETRY_INSTANCE
#define EMBREE_GEOMETRY_GRID
#define EMBREE_GEOMETRY_POINT
#define EMBREE_RAY_PACKETS

#define EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR 2.0

#if defined(EMBREE_GEOMETRY_TRIANGLE)
  #define IF_ENABLED_TRIS(x) x
#else
  #define IF_ENABLED_TRIS(x)
#endif

#if defined(EMBREE_GEOMETRY_QUAD)
  #define IF_ENABLED_QUADS(x) x
#else
  #define IF_ENABLED_QUADS(x)
#endif

#if defined(EMBREE_GEOMETRY_CURVE) || defined(EMBREE_GEOMETRY_POINT)
  #define IF_ENABLED_CURVES(x) x
#else
  #define IF_ENABLED_CURVES(x)
#endif

#if defined(EMBREE_GEOMETRY_SUBDIVISION)
  #define IF_ENABLED_SUBDIV(x) x
#else
  #define IF_ENABLED_SUBDIV(x)
#endif

#if defined(EMBREE_GEOMETRY_USER)
  #define IF_ENABLED_USER(x) x
#else
  #define IF_ENABLED_USER(x)
#endif

#if defined(EMBREE_GEOMETRY_INSTANCE)
  #define IF_ENABLED_INSTANCE(x) x
#else
  #define IF_ENABLED_INSTANCE(x)
#endif

#if defined(EMBREE_GEOMETRY_GRID)
  #define IF_ENABLED_GRIDS(x) x
#else
  #define IF_ENABLED_GRIDS(x)
#endif
"  > kernels/config.h

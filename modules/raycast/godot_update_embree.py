import glob, os, shutil, subprocess, re

git_tag = "v3.13.5"

include_dirs = [
    "common/tasking",
    "kernels/bvh",
    "kernels/builders",
    "common/sys",
    "kernels",
    "kernels/common",
    "common/math",
    "common/algorithms",
    "common/lexers",
    "common/simd",
    "common/simd/arm",
    "common/simd/wasm",
    "include/embree3",
    "kernels/subdiv",
    "kernels/geometry",
]

cpp_files = [
    "common/sys/sysinfo.cpp",
    "common/sys/alloc.cpp",
    "common/sys/filename.cpp",
    "common/sys/library.cpp",
    "common/sys/thread.cpp",
    "common/sys/string.cpp",
    "common/sys/regression.cpp",
    "common/sys/mutex.cpp",
    "common/sys/condition.cpp",
    "common/sys/barrier.cpp",
    "common/math/constants.cpp",
    "common/simd/sse.cpp",
    "common/lexers/stringstream.cpp",
    "common/lexers/tokenstream.cpp",
    "common/tasking/taskschedulerinternal.cpp",
    "kernels/common/device.cpp",
    "kernels/common/stat.cpp",
    "kernels/common/acceln.cpp",
    "kernels/common/accelset.cpp",
    "kernels/common/state.cpp",
    "kernels/common/rtcore.cpp",
    "kernels/common/rtcore_builder.cpp",
    "kernels/common/scene.cpp",
    "kernels/common/alloc.cpp",
    "kernels/common/geometry.cpp",
    "kernels/common/scene_triangle_mesh.cpp",
    "kernels/geometry/primitive4.cpp",
    "kernels/builders/primrefgen.cpp",
    "kernels/bvh/bvh.cpp",
    "kernels/bvh/bvh_statistics.cpp",
    "kernels/bvh/bvh4_factory.cpp",
    "kernels/bvh/bvh8_factory.cpp",
    "kernels/bvh/bvh_collider.cpp",
    "kernels/bvh/bvh_rotate.cpp",
    "kernels/bvh/bvh_refit.cpp",
    "kernels/bvh/bvh_builder.cpp",
    "kernels/bvh/bvh_builder_morton.cpp",
    "kernels/bvh/bvh_builder_sah.cpp",
    "kernels/bvh/bvh_builder_sah_spatial.cpp",
    "kernels/bvh/bvh_builder_sah_mb.cpp",
    "kernels/bvh/bvh_builder_twolevel.cpp",
    "kernels/bvh/bvh_intersector1.cpp",
    "kernels/bvh/bvh_intersector1_bvh4.cpp",
    "kernels/bvh/bvh_intersector_hybrid4_bvh4.cpp",
    "kernels/bvh/bvh_intersector_stream_bvh4.cpp",
    "kernels/bvh/bvh_intersector_stream_filters.cpp",
    "kernels/bvh/bvh_intersector_hybrid.cpp",
    "kernels/bvh/bvh_intersector_stream.cpp",
]

os.chdir("../../thirdparty")

dir_name = "embree"
if os.path.exists(dir_name):
    shutil.rmtree(dir_name)

subprocess.run(["git", "clone", "https://github.com/embree/embree.git", "embree-tmp"])
os.chdir("embree-tmp")
subprocess.run(["git", "checkout", git_tag])

commit_hash = str(subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True)).strip()

all_files = set(cpp_files)

dest_dir = os.path.join("..", dir_name)
for include_dir in include_dirs:
    headers = glob.iglob(os.path.join(include_dir, "*.h"))
    all_files.update(headers)

for f in all_files:
    d = os.path.join(dest_dir, os.path.dirname(f))
    if not os.path.exists(d):
        os.makedirs(d)
    shutil.copy2(f, d)

with open(os.path.join(dest_dir, "kernels/hash.h"), "w") as hash_file:
    hash_file.write(
        f"""// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define RTC_HASH "{commit_hash}"
"""
    )

with open(os.path.join(dest_dir, "kernels/config.h"), "w") as config_file:
    config_file.write(
        """// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/* #undef EMBREE_RAY_MASK */
/* #undef EMBREE_STAT_COUNTERS */
/* #undef EMBREE_BACKFACE_CULLING */
/* #undef EMBREE_BACKFACE_CULLING_CURVES */
#define EMBREE_FILTER_FUNCTION
/* #undef EMBREE_IGNORE_INVALID_RAYS */
#define EMBREE_GEOMETRY_TRIANGLE
/* #undef EMBREE_GEOMETRY_QUAD */
/* #undef EMBREE_GEOMETRY_CURVE */
/* #undef EMBREE_GEOMETRY_SUBDIVISION */
/* #undef EMBREE_GEOMETRY_USER */
/* #undef EMBREE_GEOMETRY_INSTANCE */
/* #undef EMBREE_GEOMETRY_GRID */
/* #undef EMBREE_GEOMETRY_POINT */
#define EMBREE_RAY_PACKETS
/* #undef EMBREE_COMPACT_POLYS */

#define EMBREE_CURVE_SELF_INTERSECTION_AVOIDANCE_FACTOR 2.0
#define EMBREE_DISC_POINT_SELF_INTERSECTION_AVOIDANCE

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
  #define IF_ENABLED_CURVES_OR_POINTS(x) x
#else
  #define IF_ENABLED_CURVES_OR_POINTS(x)
#endif

#if defined(EMBREE_GEOMETRY_CURVE)
  #define IF_ENABLED_CURVES(x) x
#else
  #define IF_ENABLED_CURVES(x)
#endif

#if defined(EMBREE_GEOMETRY_POINT)
  #define IF_ENABLED_POINTS(x) x
#else
  #define IF_ENABLED_POINTS(x)
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
"""
    )


with open("CMakeLists.txt", "r") as cmake_file:
    cmake_content = cmake_file.read()
    major_version = int(re.compile(r"EMBREE_VERSION_MAJOR\s(\d+)").findall(cmake_content)[0])
    minor_version = int(re.compile(r"EMBREE_VERSION_MINOR\s(\d+)").findall(cmake_content)[0])
    patch_version = int(re.compile(r"EMBREE_VERSION_PATCH\s(\d+)").findall(cmake_content)[0])

with open(os.path.join(dest_dir, "include/embree3/rtcore_config.h"), "w") as config_file:
    config_file.write(
        f"""// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define RTC_VERSION_MAJOR {major_version}
#define RTC_VERSION_MINOR {minor_version}
#define RTC_VERSION_PATCH {patch_version}
#define RTC_VERSION {major_version}{minor_version:02d}{patch_version:02d}
#define RTC_VERSION_STRING "{major_version}.{minor_version}.{patch_version}"

#define RTC_MAX_INSTANCE_LEVEL_COUNT 1

#define EMBREE_MIN_WIDTH 0
#define RTC_MIN_WIDTH EMBREE_MIN_WIDTH

#if !defined(EMBREE_STATIC_LIB)
#   define EMBREE_STATIC_LIB
#endif
/* #undef EMBREE_API_NAMESPACE*/

#if defined(EMBREE_API_NAMESPACE)
#  define RTC_NAMESPACE
#  define RTC_NAMESPACE_BEGIN namespace {{
#  define RTC_NAMESPACE_END }}
#  define RTC_NAMESPACE_USE using namespace;
#  define RTC_API_EXTERN_C
#  undef EMBREE_API_NAMESPACE
#else
#  define RTC_NAMESPACE_BEGIN
#  define RTC_NAMESPACE_END
#  define RTC_NAMESPACE_USE
#  if defined(__cplusplus)
#    define RTC_API_EXTERN_C extern "C"
#  else
#    define RTC_API_EXTERN_C
#  endif
#endif

#if defined(ISPC)
#  define RTC_API_IMPORT extern "C" unmasked
#  define RTC_API_EXPORT extern "C" unmasked
#elif defined(EMBREE_STATIC_LIB)
#  define RTC_API_IMPORT RTC_API_EXTERN_C
#  define RTC_API_EXPORT RTC_API_EXTERN_C
#elif defined(_WIN32)
#  define RTC_API_IMPORT RTC_API_EXTERN_C __declspec(dllimport)
#  define RTC_API_EXPORT RTC_API_EXTERN_C __declspec(dllexport)
#else
#  define RTC_API_IMPORT RTC_API_EXTERN_C
#  define RTC_API_EXPORT RTC_API_EXTERN_C __attribute__ ((visibility ("default")))
#endif

#if defined(RTC_EXPORT_API)
#  define RTC_API RTC_API_EXPORT
#else
#  define RTC_API RTC_API_IMPORT
#endif
"""
    )

os.chdir("..")
shutil.rmtree("embree-tmp")

subprocess.run(["git", "restore", "embree/patches"])

for patch in os.listdir("embree/patches"):
    subprocess.run(["git", "apply", "embree/patches/" + patch])

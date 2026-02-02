// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// UsdSkel API implementations

#include "usdSkel.hh"

#include <sstream>

#include "common-macros.inc"
#include "tiny-format.hh"
#include "prim-types.hh"
#include "path-util.hh"

namespace tinyusdz {
namespace {}  // namespace

constexpr auto kInbetweensNamespace = "inbetweens";

bool BlendShape::add_inbetweenBlendShape(const double weight, Attribute &&attr) {

  if (attr.name().empty()) {
    return false;
  }

  if (attr.is_uniform()) {
    return false;
  }

  if (!attr.is_value()) {
    return false;
  }

  std::string attr_name = fmt::format("{}:{}", kInbetweensNamespace, attr.name());
  attr.set_name(attr_name);

  attr.metas().weight = weight;

  props[attr_name] = Property(attr, /* custom */false);

  return true;
}

bool SkelAnimation::get_blendShapes(std::vector<value::token> *toks) {
  return blendShapes.get_value(toks);
}

bool SkelAnimation::get_joints(std::vector<value::token> *dst) {
  return joints.get_value(dst);
}

bool SkelAnimation::get_blendShapeWeights(
    std::vector<float> *vals, const double t,
    const value::TimeSampleInterpolationType tinterp) {
  Animatable<std::vector<float>> v;
  if (blendShapeWeights.get_value(&v)) {
    // Evaluate at time `t` with `tinterp` interpolation
    return v.get(t, vals, tinterp);
  }

  return false;
}

bool SkelAnimation::get_rotations(std::vector<value::quatf> *vals,
                                  const double t,
                                  const value::TimeSampleInterpolationType tinterp) {
  Animatable<std::vector<value::quatf>> v;
  if (rotations.get_value(&v)) {
    // Evaluate at time `t` with `tinterp` interpolation
    return v.get(t, vals, tinterp);
  }

  return false;
}

bool SkelAnimation::get_scales(std::vector<value::half3> *vals, const double t,
                               const value::TimeSampleInterpolationType tinterp) {
  Animatable<std::vector<value::half3>> v;
  if (scales.get_value(&v)) {
    // Evaluate at time `t` with `tinterp` interpolation
    return v.get(t, vals, tinterp);
  }

  return false;
}

bool SkelAnimation::get_translations(
    std::vector<value::float3> *vals, const double t,
    const value::TimeSampleInterpolationType tinterp) {
  Animatable<std::vector<value::float3>> v;
  if (translations.get_value(&v)) {
    // Evaluate at time `t` with `tinterp` interpolation
    return v.get(t, vals, tinterp);
  }

  return false;
}

bool BuildSkelTopology(
  const std::vector<value::token> &joints,
  std::vector<int> &dst,
  std::string *err) {

  if (joints.empty()) {
    return true;
  }

  std::vector<Path> paths(joints.size());
  for (size_t i = 0; i < joints.size(); i++) {
    Path p = Path(joints[i].str(), "");

    if (!p.is_valid()) {
      if (err) {
        (*err) += fmt::format("joints[{}] is invalid Prim path: `{}`", i, joints[i].str());
      }
      return false;
    }

    if (p.is_root_path()) {
      if (err) {
        (*err) += fmt::format("joints[{}] Root Prim path '/' cannot be used for joint Prim path.", i);
      }
      return false;
    }

    std::string _err;

    if (!pathutil::ValidatePrimPath(p, &_err)) {
      if (err) {
        (*err) += fmt::format("joints[{}] is not a valid Prim path: `{}`, reason = {}", i, joints[i].str(), _err);
      }
      return false;
    }
    
    paths[i] = p;
  }

  // path name <-> index map
  std::map<std::string, int> pathMap;
  for (size_t i = 0; i < paths.size(); i++) {
    pathMap[paths[i].prim_part()] = int(i); 
  }

  auto GetParentIndex = [](const std::map<std::string, int> &_pathMap, const Path &path) -> int {
    if (path.is_root_path()) {
      return -1;
    }
  
    // from pxrUSD's comment...
    //
    // Recurse over all ancestor paths, not just the direct parent.
    // For instance, if the map includes only paths 'a' and 'a/b/c',
    // 'a' will be treated as the parent of 'a/b/c'.
    //
    Path parentPath = path.get_parent_prim_path();
     
    uint32_t kMaxRec = 1024 * 128; // to avoid infinite loop.

    uint32_t depth = 0;
    while (parentPath.is_valid() && !parentPath.is_root_path()) {

      if (_pathMap.count(parentPath.prim_part())) {
        return _pathMap.at(parentPath.prim_part());
      } else {
      }

      parentPath = parentPath.get_parent_prim_path();
      depth++;

      if (depth >= kMaxRec) {
        // TODO: Report error
        return -1;
      } 
    }

    return -1;
  };

  dst.resize(joints.size());
  for (size_t i = 0; i < paths.size(); i++) {
    dst[i] = GetParentIndex(pathMap, paths[i]);
  }

  return true;
}

}  // namespace tinyusdz


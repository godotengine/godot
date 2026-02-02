// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// UsdSkel(includes BlendShapes)
#pragma once

#include "prim-types.hh"
#include "value-types.hh"
#include "xform.hh"

namespace tinyusdz {

constexpr auto kSkelRoot = "SkelRoot";
constexpr auto kSkeleton = "Skeleton";
constexpr auto kSkelAnimation = "SkelAnimation";
constexpr auto kBlendShape = "BlendShape";

// BlendShapes
struct BlendShape {
  std::string name;
  Specifier spec{Specifier::Def};
  int64_t parent_id{-1}; 

  void set_name(const std::string &name_) {
    name = name_;
  }

  const std::string &get_name() const {
    return name;
  }

  Specifier &specifier() { return spec; }
  const Specifier &specifier() const { return spec; }

  TypedAttribute<std::vector<value::vector3f>>
      offsets;  // uniform vector3f[]. required property
  TypedAttribute<std::vector<value::vector3f>>
      normalOffsets;  // uniform vector3f[]. required property

  TypedAttribute<std::vector<int>>
      pointIndices;  // uniform int[]. optional. vertex indices to the original
                     // mesh for each values in `offsets` and `normalOffsets`.

  std::pair<ListEditQual, std::vector<Reference>> references;
  std::pair<ListEditQual, std::vector<Payload>> payload;
  std::map<std::string, VariantSet> variantSet;
  std::map<std::string, Property> props;

  ///
  /// Add attribute as in-beteen BlendShape attribute.
  ///
  /// - add `inbetweens` namespace prefix
  /// - add `weight` attribute as Attribute meta.
  ///
  bool add_inbetweenBlendShape(double weight, Attribute &&attr);

  const std::vector<value::token> &primChildrenNames() const { return _primChildren; }
  const std::vector<value::token> &propertyNames() const { return _properties; }
  std::vector<value::token> &primChildrenNames() { return _primChildren; }
  std::vector<value::token> &propertyNames() { return _properties; }

  PrimMeta meta;

  PrimMeta &metas() {
    return meta;
  }

  const PrimMeta &metas() const {
    return meta;
  }

 private:
  std::vector<value::token> _primChildren;
  std::vector<value::token> _properties;
};

// Skeleton
struct Skeleton : Xformable {
  std::string name;
  Specifier spec{Specifier::Def};
  int64_t parent_id{-1};

  void set_name(const std::string &name_) {
    name = name_;
  }

  const std::string &get_name() const {
    return name;
  }

  Specifier &specifier() { return spec; }
  const Specifier &specifier() const { return spec; }


  TypedAttribute<std::vector<value::matrix4d>>
      bindTransforms;  // uniform matrix4d[]. bind-pose transform of each joint
                       // in world coordinate.

  TypedAttribute<std::vector<value::token>> jointNames;  // uniform token[]
  TypedAttribute<std::vector<value::token>> joints;      // uniform token[]

  TypedAttribute<std::vector<value::matrix4d>>
      restTransforms;  // uniform matrix4d[] rest-pose transforms of each
                       // joint in local coordinate.

  nonstd::optional<Relationship> proxyPrim;  // rel proxyPrim

  // SkelBindingAPI
  nonstd::optional<Relationship>
      animationSource;  // rel skel:animationSource = </path/...>

  TypedAttributeWithFallback<Animatable<Visibility>> visibility{
      Visibility::Inherited};  // "token visibility"
  TypedAttribute<Animatable<Extent>>
      extent;  // bounding extent. When authorized, the extent is the bounding
               // box of whole its children.
  TypedAttributeWithFallback<Purpose> purpose{
      Purpose::Default};  // "uniform token purpose"

  std::pair<ListEditQual, std::vector<Reference>> references;
  std::pair<ListEditQual, std::vector<Payload>> payload;
  std::map<std::string, VariantSet> variantSet;
  std::map<std::string, Property> props;
  //std::vector<value::token> xformOpOrder;

  PrimMeta meta;

  PrimMeta &metas() {
    return meta;
  }

  const PrimMeta &metas() const {
    return meta;
  }


  bool get_animationSource(Path *path, ListEditQual *qual = nullptr) {
    if (!path) {
      return false;
    }

    const Relationship &rel = animationSource.value();
    if (qual) {
      (*qual) = rel.get_listedit_qual();
    }

    if (rel.is_path()) {
      (*path) = rel.targetPath;
    } else if (rel.is_pathvector()) {
      if (rel.targetPathVector.size()) {
        (*path) = rel.targetPathVector[0];
      }
    } else {
      return false;
    }


    return false;
  }

  const std::vector<value::token> &primChildrenNames() const { return _primChildren; }
  const std::vector<value::token> &propertyNames() const { return _properties; }
  std::vector<value::token> &primChildrenNames() { return _primChildren; }
  std::vector<value::token> &propertyNames() { return _properties; }

  private:
  std::vector<value::token> _primChildren;
  std::vector<value::token> _properties;
};

// NOTE: SkelRoot itself does not have dedicated attributes in the schema.
struct SkelRoot : Xformable {
  std::string name;
  Specifier spec{Specifier::Def};
  int64_t parent_id{-1};

  void set_name(const std::string &name_) {
    name = name_;
  }

  const std::string &get_name() const {
    return name;
  }

  Specifier &specifier() { return spec; }
  const Specifier &specifier() const { return spec; }


  TypedAttribute<Animatable<Extent>>
    extent;  // bounding extent. When authorized, the extent is the bounding
  // box of whole its children.
  TypedAttributeWithFallback<Purpose> purpose{
    Purpose::Default};  // "uniform token purpose"
  TypedAttributeWithFallback<Animatable<Visibility>> visibility{
    Visibility::Inherited};  // "token visibility"

  nonstd::optional<Relationship> proxyPrim;  // rel proxyPrim
  //std::vector<XformOp> xformOps;

  // TODO: Add function to check if SkelRoot contains `Skeleton` and `GeomMesh`
  // node?;


  std::pair<ListEditQual, std::vector<Reference>> references;
  std::pair<ListEditQual, std::vector<Payload>> payload;
  std::map<std::string, VariantSet> variantSet;
  std::map<std::string, Property> props;

  const std::vector<value::token> &primChildrenNames() const { return _primChildren; }
  const std::vector<value::token> &propertyNames() const { return _properties; }
  std::vector<value::token> &primChildrenNames() { return _primChildren; }
  std::vector<value::token> &propertyNames() { return _properties; }

  PrimMeta meta;

  PrimMeta &metas() {
    return meta;
  }

  const PrimMeta &metas() const {
    return meta;
  }


 private:
  std::vector<value::token> _primChildren;
  std::vector<value::token> _properties;

};

struct SkelAnimation {
  std::string name;
  Specifier spec{Specifier::Def};
  int64_t parent_id{-1};

  void set_name(const std::string &name_) {
    name = name_;
  }

  const std::string &get_name() const {
    return name;
  }

  Specifier &specifier() { return spec; }
  const Specifier &specifier() const { return spec; }

  TypedAttribute<std::vector<value::token>> blendShapes;  // uniform token[]
  TypedAttribute<Animatable<std::vector<float>>> blendShapeWeights;  // float[]
  TypedAttribute<std::vector<value::token>> joints;  // uniform token[]
  TypedAttribute<Animatable<std::vector<value::quatf>>>
      rotations;  // quatf[] Joint-local unit quaternion rotations
  TypedAttribute<Animatable<std::vector<value::half3>>>
      scales;  // half3[] Joint-local scaling in 16bit half float. TODO: Use
               // float3 for TinyUSDZ for convenience?
  TypedAttribute<Animatable<std::vector<value::float3>>>
      translations;  // float3[] Joint-local translation.

  bool get_blendShapes(std::vector<value::token> *toks);
  bool get_blendShapeWeights(std::vector<float> *vals,
                             const double t = value::TimeCode::Default(),
                             const value::TimeSampleInterpolationType tinterp =
                                 value::TimeSampleInterpolationType::Held);
  bool get_joints(std::vector<value::token> *toks);
  bool get_rotations(std::vector<value::quatf> *vals,
                     const double t = value::TimeCode::Default(),
                     const value::TimeSampleInterpolationType tinterp =
                         value::TimeSampleInterpolationType::Held);
  bool get_scales(std::vector<value::half3> *vals,
                  const double t = value::TimeCode::Default(),
                  const value::TimeSampleInterpolationType tinterp =
                      value::TimeSampleInterpolationType::Held);
  bool get_translations(std::vector<value::float3> *vals,
                        const double t = value::TimeCode::Default(),
                        const value::TimeSampleInterpolationType tinterp =
                            value::TimeSampleInterpolationType::Held);

  std::pair<ListEditQual, std::vector<Reference>> references;
  std::pair<ListEditQual, std::vector<Payload>> payload;
  std::map<std::string, VariantSet> variantSet;
  std::map<std::string, Property> props;

  const std::vector<value::token> &primChildrenNames() const { return _primChildren; }
  const std::vector<value::token> &propertyNames() const { return _properties; }
  std::vector<value::token> &primChildrenNames() { return _primChildren; }
  std::vector<value::token> &propertyNames() { return _properties; }

  PrimMeta meta;

  PrimMeta &metas() {
    return meta;
  }

  const PrimMeta &metas() const {
    return meta;
  }

 private:
  std::vector<value::token> _primChildren;
  std::vector<value::token> _properties;
};

// PackedJointAnimation is deprecated(Convert to SkelAnimation)
// struct PackedJointAnimation {
// };

//
// Some usdSkel utility functions
//

// Equivalent to pxrUSd's UsdSkelNormalizeWeights
bool SkelNormalizeWeights(const std::vector<float> &weights, int numInfluencesPerComponent, const float eps = std::numeric_limits<float>::epsilon());
bool SkelSortInfluences(const std::vector<int> indices, const std::vector<float> &weights, int numInfluencesPerComponent);

#if 0 // move to Tydra
struct SkelNode
{
  std::string joint;
  std::string jointName;
  int32_t parentIndex{-1}; // Index of parent SkelNode.
  int32_t index; // Index of this SkelNode.
  
  value::matrix4d bindTransform{value::matrix4d::identity()};
  value::matrix4d restTransform{value::matrix4d::identity()};
};
#endif

//
// Build Skeleleton Topology(hierarchy) from Skeleton's joints.
// (Usually from Skeleton's `joints`attribute).
// 
// If you want to get handy, full Skeleton hierarchy information, Use Tydra's BuildSkelHierarchy() API.
//
// @param[in] `joints` Joint paths
// @param[out] `dst` Built SkelTopology.  dst[i] = parent joint index. -1 for root joint.
// @param[out] `err` Error message when `joints` info is invalid.
//
// @return true upon success. false when error.
// 
bool BuildSkelTopology(
  const std::vector<value::token> &joints,
  std::vector<int> &dst,
  std::string *err);

// import DEFINE_TYPE_TRAIT and DEFINE_ROLE_TYPE_TRAIT
#include "define-type-trait.inc"

namespace value {

// Register usdSkel Prim type.
DEFINE_TYPE_TRAIT(SkelRoot, kSkelRoot, TYPE_ID_SKEL_ROOT, 1);
DEFINE_TYPE_TRAIT(Skeleton, kSkeleton, TYPE_ID_SKELETON, 1);
DEFINE_TYPE_TRAIT(SkelAnimation, kSkelAnimation, TYPE_ID_SKELANIMATION, 1);
DEFINE_TYPE_TRAIT(BlendShape, kBlendShape, TYPE_ID_BLENDSHAPE, 1);

#undef DEFINE_TYPE_TRAIT
#undef DEFINE_ROLE_TYPE_TRAIT

}  // namespace value

}  // namespace tinyusdz

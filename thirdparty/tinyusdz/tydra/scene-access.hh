// SPDX-License-Identifier: Apache 2.0
// Copyright 2022-Present Light Transport Entertainment, Inc.
//
// Scene access API
//
// NOTE: Tydra API does not use nonstd::optional and nonstd::expected,
// std::functions and other non basic STL feature for easier language bindings.
//
#pragma once

#include <map>

#include "prim-type-macros.inc"
#include "prim-types.hh"
#include "stage.hh"
#include "tiny-format.hh"
#include "usdGeom.hh"
#include "usdLux.hh"
#include "usdShade.hh"
#include "usdSkel.hh"
#include "value-type-macros.inc"
#include "value-types.hh"

namespace tinyusdz {
namespace tydra {

// key = fully absolute Prim path in string(e.g. "/xform/geom0")
template <typename T>
using PathPrimMap = std::map<std::string, const T *>;

//
// value = pair of Shader Prim which contains the Shader type T("info:id") and
// its concrete Shader type(UsdPreviewSurface)
//
template <typename T>
using PathShaderMap =
    std::map<std::string, std::pair<const Shader *, const T *>>;

// TODO: extern template to suppress possible `-Wundefined-func-template`?

///
/// List Prim of type T from the Stage.
/// Returns false when unsupported/unimplemented Prim type T is given.
///
template <typename T>
bool ListPrims(const tinyusdz::Stage &stage, PathPrimMap<T> &m /* output */);

#define EXTERN_LISTPRIMS(__ty)                                 \
  extern template bool ListPrims(const tinyusdz::Stage &stage, \
                                 PathPrimMap<__ty> &m);

APPLY_FUNC_TO_PRIM_TYPES(EXTERN_LISTPRIMS)

#undef EXTERN_LISTPRIMS

///
/// List Shader of shader type T from the Stage.
/// Returns false when unsupported/unimplemented Shader type T is given.
/// TODO: User defined shader type("info:id")
///
template <typename T>
bool ListShaders(const tinyusdz::Stage &stage,
                 PathShaderMap<T> &m /* output */);

extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdPreviewSurface> &m);
extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdUVTexture> &m);

extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdPrimvarReader_string> &m);
extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdPrimvarReader_int> &m);
extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdPrimvarReader_float> &m);
extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdPrimvarReader_float2> &m);
extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdPrimvarReader_float3> &m);
extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdPrimvarReader_float4> &m);
extern template bool ListShaders(const tinyusdz::Stage &stage,
                                 PathShaderMap<UsdPrimvarReader_matrix> &m);

///
/// Get parent Prim from Path.
/// Path must be fully expanded absolute path.
///
/// Example: Return "/xform" Prim for "/xform/mesh0" path
///
/// Returns nullptr when the given Path is a root Prim or invalid Path(`err`
/// will be filled when failed).
///
const Prim *GetParentPrim(const tinyusdz::Stage &stage,
                          const tinyusdz::Path &path, std::string *err);

///
/// Visit Stage and invoke callback functions for each Prim.
/// Can be used for alternative method of Stage::Traverse() in pxrUSD
///

///
/// Use old-style Callback function approach for easier language bindings
///
/// @param[in] abs_path Prim's absolute path(e.g. "/xform/mesh0")
/// @param[in] prim Prim
/// @param[in] tree_depth Tree depth of this Prim. 0 = root prim.
/// @param[inout] userdata User data.
/// @param[out] error message.
///
/// @return Usually true. return false + no error message to notify early
/// termination of visiting Prims.
///
typedef bool (*VisitPrimFunction)(const Path &abs_path, const Prim &prim,
                                  const int32_t tree_depth, void *userdata,
                                  std::string *err);

///
/// Visit Prims in Stage.
/// Use `primChildren` metadatum to determine traversal order of Prims if
/// exists(USDC usually contains `primChildren`) Traversal will be failed when
/// no Prim found specified in `primChildren`(if exists)
///
/// @param[out] err Error message.
///
bool VisitPrims(const tinyusdz::Stage &stage, VisitPrimFunction visitor_fun,
                void *userdata = nullptr, std::string *err = nullptr);

///
/// Get Property(Attribute or Relationship) of given Prim by name.
/// Similar to UsdPrim::GetProperty() in pxrUSD.
///
/// @param[in] prim Prim
/// @param[in] prop_name Property name
/// @param[out] prop Property
/// @param[out] err Error message(filled when returning false)
///
/// @return true if Property found in given Prim.
/// @return false if Property is not found in given Prim.
///
bool GetProperty(const tinyusdz::Prim &prim, const std::string &prop_name,
                 Property *prop, std::string *err);

///
/// Get List of Property(Attribute and Relationship) names of given Prim by
/// name. It includes authored builtin Property names.
///
/// @param[in] prim Prim
/// @param[out] prop_names Property names
/// @param[out] err Error message(filled when returning false)
///
/// @return true upon success.
/// @return false when something go wrong.
///
bool GetPropertyNames(const tinyusdz::Prim &prim,
                      std::vector<std::string> *prop_names, std::string *err);

///
/// Get List of Attribute names of given Prim.
/// It includes authored builtin Attribute names(e.g. "points" for `GeomMesh`).
///
/// @param[in] prim Prim
/// @param[out] attr_names Attribute names
/// @param[out] err Error message(filled when returning false)
///
/// @return true upon success.
/// @return false when something go wrong.
///
bool GetAttributeNames(const tinyusdz::Prim &prim,
                       std::vector<std::string> *attr_names, std::string *err);

///
/// Get List of Relationship names of given Prim.
/// It includes authored builtin Relationship names(e.g. "proxyPrim" for
/// `GeomMesh`).
///
/// @param[in] prim Prim
/// @param[out] rel_names Relationship names
/// @param[out] err Error message(filled when returning false)
///
/// @return true upon success.
/// @return false when something go wrong.
///
bool GetRelationshipNames(const tinyusdz::Prim &prim,
                          std::vector<std::string> *rel_names,
                          std::string *err);

///
/// Get Attribute of given Prim by name.
/// Similar to UsdPrim::GetAttribute() in pxrUSD.
///
/// @param[in] prim Prim
/// @param[in] attr_name Attribute name
/// @param[out] attr Attribute
/// @param[out] err Error message(filled when returning false)
///
/// @return true if Attribute found in given Prim.
/// @return false if Attribute is not found in given Prim, or `attr_name` is a
/// Relationship.
///
bool GetAttribute(const tinyusdz::Prim &prim, const std::string &attr_name,
                  Attribute *attr, std::string *err);

///
/// Check if Prim has Attribute.
///
/// @param[in] prim Prim
/// @param[in] attr_name Attribute name to query.
///
/// @return true if `attr_name` Attribute exists in the Prim.
///
bool HasAttribute(const tinyusdz::Prim &prim, const std::string &attr_name);

///
/// Get Relationship of given Prim by name.
/// Similar to UsdPrim::GetRelationship() in pxrUSD.
///
/// @param[in] prim Prim
/// @param[in] rel_name Relationship name
/// @param[out] rel Relationship
/// @param[out] err Error message(filled when returning false)
///
/// @return true if Relationship found in given Prim.
/// @return false if Relationship is not found in given Prim, or `rel_name` is a
/// Attribute.
///
bool GetRelationship(const tinyusdz::Prim &prim, const std::string &rel_name,
                     Relationship *rel, std::string *err);

///
/// Check if Prim has Relationship.
///
/// @param[in] prim Prim
/// @param[in] rel_name Relationship name to query.
///
/// @return true if `rel_name` Relationship exists in the Prim.
///
bool HasRelationship(const tinyusdz::Prim &prim, const std::string &rel_name);

///
/// For efficient Xform retrieval from Stage.
///
/// XformNode's pointer value and hierarchy become invalid when Prim is
/// removed/added from/to Stage. If you change the content of Stage, please
/// rebuild XformNode using BuildXformNodeFromStage() again
///
/// TODO: Use prim_id and deprecate the pointer to Prim.
///
struct XformNode {
  std::string element_name;  // e.g. "geom0"
  Path absolute_path;        // e.g. "/xform/geom0"

  const Prim *prim{nullptr};  // The pointer to Prim.
  int64_t prim_id{-1};        // Prim id(1 or greater for valid Prim ID)

  XformNode *parent{nullptr};  // pointer to parent
  std::vector<XformNode> children;

  const value::matrix4d &get_local_matrix() const { return _local_matrix; }

  // world matrix = parent_world_matrix x local_matrix
  // Equivalent to GetLocalToWorldMatrix in pxrUSD
  // if !resetXformStack! exists in Prim's xformOpOrder, this returns Prim's
  // local matrix (clears parent's world matrix)
  const value::matrix4d &get_world_matrix() const { return _world_matrix; }

  const value::matrix4d &get_parent_world_matrix() const {
    return _parent_world_matrix;
  }

  // TODO: accessible only from Friend class?
  void set_local_matrix(const value::matrix4d &m) { _local_matrix = m; }

  void set_world_matrix(const value::matrix4d &m) { _world_matrix = m; }

  void set_parent_world_matrix(const value::matrix4d &m) {
    _parent_world_matrix = m;
  }

  // true: Prim with Xform(e.g. GeomMesh)
  // false: Prim with no Xform(e.g. Stage root("/"), Scope, Material, ...)
  bool has_xform() const { return _has_xform; }
  bool &has_xform() { return _has_xform; }

  bool has_resetXformStack() const { return _has_resetXformStack; }
  bool &has_resetXformStack() { return _has_resetXformStack; }

 private:
  bool _has_xform{false};
  bool _has_resetXformStack{false};  // !resetXformStack! in xformOps
  value::matrix4d _local_matrix{value::matrix4d::identity()};
  value::matrix4d _world_matrix{value::matrix4d::identity()};
  value::matrix4d _parent_world_matrix{value::matrix4d::identity()};
};

///
/// Build Xform hierachy from Stage.
///
/// Xform value is evaluated at specified time and timeSample interpolation
/// type.
///
///
bool BuildXformNodeFromStage(
    const tinyusdz::Stage &stage, XformNode *root, /* out */
    const double t = tinyusdz::value::TimeCode::Default(),
    const tinyusdz::value::TimeSampleInterpolationType tinterp =
        tinyusdz::value::TimeSampleInterpolationType::Linear);

std::string DumpXformNode(const XformNode &root);

///
/// Get GeomSubset children of the given Prim path
///
/// The pointer address is valid until Stage's content is unchanged.
///
/// @param[in] familyName Get GeomSubset having this `familyName`. empty token =
/// return all GeomSubsets.
/// @param[in] prim_must_be_geommesh Prim path must point to GeomMesh Prim.
///
/// (TODO: Return id of GeomSubset Prim object, instead of the ponter address)
///
/// @return array of GeomSubset pointers. Empty array when failed or no
/// GeomSubset Prim(with `familyName`) attached to the Prim.
///
///
std::vector<const GeomSubset *> GetGeomSubsets(
    const tinyusdz::Stage &stage, const tinyusdz::Path &prim_path,
    const tinyusdz::value::token &familyName,
    bool prim_must_be_geommesh = true);

///
/// Get GeomSubset children of the given Prim
///
/// The pointer address is valid until Stage's content is unchanged.
///
/// @param[in] familyName Get GeomSubset having this `familyName`. empty token =
/// return all GeomSubsets.
/// @param[in] prim_must_be_geommesh Prim must be GeomMesh Prim type.
///
/// (TODO: Return id of GeomSubset Prim object, instead of the ponter address)
///
/// @return array of GeomSubset pointers. Empty array when failed or no
/// GeomSubset Prim(with `familyName`) attached to the Prim.
///
std::vector<const GeomSubset *> GetGeomSubsetChildren(
    const tinyusdz::Prim &prim, const tinyusdz::value::token &familyName,
    bool prim_must_be_geommesh = true);

//
// Get BlendShape prims in this GeomMesh Prim
// (`skel:blendShapes`, `skel:blendShapeTargets`)
//
std::vector<std::pair<std::string, const tinyusdz::BlendShape *>>
GetBlendShapes(const tinyusdz::Stage &stage, const tinyusdz::Prim &prim,
                std::string *err = nullptr);

#if 0  // TODO
///
/// Get list of GeomSubset PrimSpecs attached to the PrimSpec
/// Prim path must point to GeomMesh PrimSpec.
///
/// The pointer address is valid until Layer's content is unchanged.
///
/// (TODO: Return PrimSpec index instead of the ponter address)
///
std::vector<const PrimSpec *> GetGeomSubsetPrimSpecs(const tinyusdz::Layer &layer, const tinyusdz::Path &prim_path);

std::vector<const PrimSpec *> GetGeomSubsetChildren(const tinyusdz::Path &prim_path);
#endif

///
/// For composition. Convert Concrete Prim(Xform, GeomMesh, ...) to PrimSpec,
/// generic Prim container.
/// TODO: Move to *core* module?
///
bool PrimToPrimSpec(const Prim &prim, PrimSpec &ps, std::string *err);

///
/// For MaterialX
/// TODO: Move to shader-network.hh?
///
bool ShaderToPrimSpec(const UsdUVTexture &node, PrimSpec &ps, std::string *warn,
                      std::string *err);
bool ShaderToPrimSpec(const UsdTransform2d &node, PrimSpec &ps,
                      std::string *warn, std::string *err);

template <typename T>
bool ShaderToPrimSpec(const UsdPrimvarReader<T> &node, PrimSpec &ps,
                      std::string *warn, std::string *err);

//
// Utilities and Query for CollectionAPI
//

///
/// Get `Collection` object(properties defined in Collection API) from a given
/// Prim.
///
/// @param[in] prim Prim
/// @param[out] Pointer to the pointer of found Collection.
/// @return true upon success.
///
bool GetCollection(const Prim &prim, const Collection **collection);

class CollectionMembershipQuery {
 public:
 private:
  std::map<Path, CollectionInstance::ExpansionRule> _expansionRuleMap;
};

///
/// Get terminal Attribute. Similar to GetValueProducingAttribute in pxrUSD.
///
/// On the contrary to EvaluateAttribute, Do not evaluate Attribute value at
/// specified time.
///
/// - if Attribute is connection, follow its targetPath recursively until
/// encountering non-connection Attribute.
/// - if Attribute is blocked, return Attribute ValueBlock.
/// - if Attribute is timesamples, return TimeSamples Attribute.
/// - if Attribute is scalar, return scalar Attribute.
///
/// @return true upon success.
bool GetTerminalAttribute(const Stage &stage, const Attribute &attr,
                          const std::string &attr_name, Attribute *attr_out,
                          std::string *err);

template <typename T>
bool GetTerminalAttribute(const Stage &stage, const TypedAttribute<T> &attr,
                          const std::string &attr_name, Attribute *attr_out,
                          std::string *err) {
  if (!attr_out) {
    return false;
  }

  Attribute value;
  if (attr.is_connection()) {
    Attribute input;
    input.set_connections(attr.connections());
    return GetTerminalAttribute(stage, input, attr_name, attr_out, err);
  } else if (attr.is_blocked()) {
    value.metas() = attr.metas();
    value.variability() = Variability::Uniform;
    value.set_type_name(value::TypeTraits<T>::type_name());
    value.set_blocked(true);
    (*attr_out) = std::move(value);
    return true;
  } else if (attr.is_value_empty()) {
    value.set_type_name(value::TypeTraits<T>::type_name());
    value.metas() = attr.metas();
    value.variability() = Variability::Uniform;
  } else {
    value.set_value(attr.get_value());
    value.metas() = attr.metas();
    value.variability() = Variability::Uniform;
  }

  (*attr_out) = std::move(value);
  return true;
}

///
/// Get Geom Primvar.
///
/// This API supports Connection Attribute(which requires finding Prim of
/// targetPath in Stage).
///
/// example of Primvar with Connection Attribute:
///
///   texCoord2f[] primvars:uvs = </root/geom0.uvs>
///   int[] primvars:uvs:indices.connection = </root/geom0.indices>
///
/// @param[in] stage Stage
/// @param[in] prim The pointer to GPrim.
/// @param[in] name Primvar name(`primvars:` prefix omitted)
/// @param[out] primvar GeomPrimvar output.
/// @param[out] err Error message.
///
/// @return true upon success.
///
bool GetGeomPrimvar(const Stage &stage, const GPrim *prim,
                    const std::string &name, GeomPrimvar *primvar,
                    std::string *err = nullptr);

///
/// Get Primvars in GPrim.
///
/// This API supports Connection Attribute(which requires finding Prim of
/// targetPath in Stage).
///
std::vector<GeomPrimvar> GetGeomPrimvars(const Stage &stage, const GPrim &prim);

///
/// Build Collection Membership
///
/// It traverse collection paths starting from `seedCollectionInstance` in the
/// Stage. Note: No circular referencing path allowed.
///
/// @returns CollectionMembershipQuery object. When encountered an error,
/// CollectionMembershipQuery contains empty info(i.e, all query will fail)
///
CollectionMembershipQuery BuildCollectionMembershipQuery(
    const Stage &stage, const CollectionInstance &seedCollectionInstance);

bool IsPathIncluded(const CollectionMembershipQuery &query, const Stage &stage,
                    const Path &abs_path,
                    const CollectionInstance::ExpansionRule expansionRule =
                        CollectionInstance::ExpansionRule::ExpandPrims);

// TODO: Layer version
// bool IsPathIncluded(const Layer &layer, const Path &abs_path, const
// CollectionInstance::ExpansionRule expansionRule =
// CollectionInstance::ExpansionRule::ExpandPrims);

//
// usdSkel
//

struct SkelNode {
  //std::string jointElementName;  // elementName(leaf node name) of jointPath.
  std::string joint_path;         // joints in UsdSkel. Relative or Absolute Prim
                                 // path(e.g. "root/head", "/root/head")
  std::string joint_name;         // jointNames in UsdSkel
  int joint_id{-1};               // jointIndex(array index in UsdSkel joints)

  value::matrix4d bind_transform{value::matrix4d::identity()};
  value::matrix4d rest_transform{value::matrix4d::identity()};
  //int parentNodeIndex{-1};

  std::vector<SkelNode> children;
};

class SkelHierarchy {
 public:
  SkelHierarchy() = default;

  std::string prim_name;                  // Skeleleton Prim name
  std::string abs_path;                   // Absolute path to Skeleleton Prim
  std::string display_name;               // `displayName` Prim meta

  SkelNode root_node; 

  int anim_id{-1};                        // Default animation(SkelAnimation) attached to Skeleton

 private:

};

std::map<std::string, int> BuildSkelNameToIndexMap(const SkelHierarchy &skel);

///
/// Extract skeleleton info from Skeleton and build skeleton(bone) hierarchy.
///
bool BuildSkelHierarchy(const Skeleton &skel,
                        SkelNode &dst, std::string *err = nullptr);

//
// For USDZ AR extensions
//

///
/// List up `sceneName` of given Prim's children
/// https://developer.apple.com/documentation/realitykit/usdz-schemas-for-ar
///
/// Prim's Kind must be `sceneLibrary`
/// @param[out] List of pair of (Is Specifier `over`, sceneName). For `def`
/// Specifier(primary scene), it is set to false.
///
///
bool ListSceneNames(const tinyusdz::Prim &root,
                    std::vector<std::pair<bool, std::string>> *sceneNames);

}  // namespace tydra
}  // namespace tinyusdz

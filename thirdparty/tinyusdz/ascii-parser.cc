// SPDX-License-Identifier: Apache 2.0
// Copyright 2021 - 2022, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment, Inc.
//
// To reduce compilation time and sections generated in .obj(object file),
// We split implementaion to multiple of .cc for ascii-parser.hh

#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
//
#include <algorithm>
#include <atomic>
#include <cstdio>
//#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#if defined(__wasi__)
#else
#include <mutex>
#include <thread>
#endif
#include <vector>

#include "ascii-parser.hh"
#include "path-util.hh"
#include "str-util.hh"
#include "tiny-format.hh"

//
#if !defined(TINYUSDZ_DISABLE_MODULE_USDA_READER)

//

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// external
#include "nonstd/expected.hpp"

//

#ifdef __clang__
#pragma clang diagnostic pop
#endif

//

#include "common-macros.inc"
#include "io-util.hh"
#include "pprinter.hh"
#include "prim-types.hh"
#include "str-util.hh"
#include "stream-reader.hh"
#include "tinyusdz.hh"
#include "value-pprint.hh"
#include "value-types.hh"

namespace tinyusdz {

namespace ascii {

constexpr auto kRel = "rel";
constexpr auto kTimeSamplesSuffix = ".timeSamples";
constexpr auto kConnectSuffix = ".connect";

constexpr auto kAscii = "[ASCII]";

extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<bool>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<int32_t>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::int2>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::int3>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::int4>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<uint32_t>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::uint2>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::uint3>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::uint4>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<int64_t>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<uint64_t>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::half>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::half2>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::half3>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::half4>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<float>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::float2>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::float3>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::float4>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<double>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::double2>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::double3>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::double4>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::quath>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::quatf>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::quatd>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::texcoord2h>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::texcoord2f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::texcoord2d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::texcoord3h>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::texcoord3f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::texcoord3d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::point3h>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::point3f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::point3d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::normal3h>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::normal3f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::normal3d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::vector3h>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::vector3f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::vector3d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::color3h>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::color3f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::color3d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::color4h>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::color4f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::color4d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::matrix2f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::matrix3f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::matrix4f>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::matrix2d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::matrix3d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::matrix4d>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::token>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::StringData>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<std::string>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<Reference>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<Payload>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<Path>> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<nonstd::optional<value::AssetPath>> *result);

extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<bool> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<int32_t> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::int2> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::int3> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::int4> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<uint32_t> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::uint2> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::uint3> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::uint4> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<int64_t> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<uint64_t> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::half> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::half2> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::half3> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::half4> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<float> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::float2> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::float3> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::float4> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<double> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::double2> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::double3> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::double4> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::quath> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::quatf> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::quatd> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::texcoord2h> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::texcoord2f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::texcoord2d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::texcoord3h> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::texcoord3f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::texcoord3d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::point3h> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::point3f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::point3d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::normal3h> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::normal3f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::normal3d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::vector3h> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::vector3f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::vector3d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::color3h> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::color3f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::color3d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::color4h> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::color4f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::color4d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::matrix2f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::matrix3f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::matrix4f> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::matrix2d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::matrix3d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::matrix4d> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::token> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::StringData> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<std::string> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<Reference> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<Payload> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<Path> *result);
extern template bool AsciiParser::ParseBasicTypeArray(
    std::vector<value::AssetPath> *result);

static void RegisterStageMetas(
    std::map<std::string, AsciiParser::VariableDef> &metas) {
  metas.clear();
  metas["doc"] = AsciiParser::VariableDef(value::kString, "doc");
  metas["documentation"] =
      AsciiParser::VariableDef(value::kString, "doc");  // alias to 'doc'

  metas["comment"] = AsciiParser::VariableDef(value::kString, "comment");

  // TODO: both support float and double?
  metas["metersPerUnit"] =
      AsciiParser::VariableDef(value::kDouble, "metersPerUnit");
  metas["timeCodesPerSecond"] =
      AsciiParser::VariableDef(value::kDouble, "timeCodesPerSecond");
  metas["framesPerSecond"] =
      AsciiParser::VariableDef(value::kDouble, "framesPerSecond");

  metas["startTimeCode"] =
      AsciiParser::VariableDef(value::kDouble, "startTimeCode");
  metas["endTimeCode"] =
      AsciiParser::VariableDef(value::kDouble, "endTimeCode");

  metas["defaultPrim"] = AsciiParser::VariableDef(value::kToken, "defaultPrim");
  metas["upAxis"] = AsciiParser::VariableDef(value::kToken, "upAxis");
  metas["customLayerData"] =
      AsciiParser::VariableDef(value::kDictionary, "customLayerData");

  // Composition arc.
  // Type can be array. i.e. asset, asset[]
  metas["subLayers"] = AsciiParser::VariableDef(value::kAssetPath, "subLayers",
                                                /* allow array type */ true);

  // UsdPhysics
  metas["kilogramsPerUnit"] =
      AsciiParser::VariableDef(value::kDouble, "kilogramsPerUnit");

  // USDZ extension
  metas["autoPlay"] = AsciiParser::VariableDef(value::kBool, "autoPlay");
  metas["playbackMode"] =
      AsciiParser::VariableDef(value::kToken, "playbackMode");
}

static void RegisterPrimMetas(
    std::map<std::string, AsciiParser::VariableDef> &metas) {
  metas.clear();

  metas["kind"] = AsciiParser::VariableDef(value::kToken, "kind");
  metas["doc"] = AsciiParser::VariableDef(value::kString, "doc");

  //
  // Composition arcs -----------------------
  //

  // Type can be array. i.e. path, path[]
  metas["references"] = AsciiParser::VariableDef("Reference", "references",
                                                 /* allow array type */ true);

  // TODO: Use relatioship type?
  metas["inherits"] = AsciiParser::VariableDef(value::kPath, "inherits", true);
  metas["payload"] = AsciiParser::VariableDef("Payload", "payload", true);
  metas["specializes"] =
      AsciiParser::VariableDef(value::kPath, "specializes", true);

  // Use `string`
  metas["variantSets"] = AsciiParser::VariableDef(value::kString, "variantSets",
                                                  /* allow array type */ true);

  // Parse as dict. TODO: Use ParseVariants()
  metas["variants"] = AsciiParser::VariableDef(value::kDictionary, "variants");

  // ------------------------------------------

  metas["assetInfo"] =
      AsciiParser::VariableDef(value::kDictionary, "assetInfo");
  metas["customData"] =
      AsciiParser::VariableDef(value::kDictionary, "customData");

  metas["active"] = AsciiParser::VariableDef(value::kBool, "active");
  metas["hidden"] = AsciiParser::VariableDef(value::kBool, "hidden");
  metas["instanceable"] =
      AsciiParser::VariableDef(value::kBool, "instanceable");

  // ListOp
  metas["apiSchemas"] = AsciiParser::VariableDef(
      value::Add1DArraySuffix(value::kToken), "apiSchemas");

  // usdShade
  // NOTE: items are expected to be all string type.
  metas["sdrMetadata"] =
      AsciiParser::VariableDef(value::kDictionary, "sdrMetadata");

  metas["clips"] = AsciiParser::VariableDef(value::kDictionary, "clips");

  // USDZ extension
  metas["sceneName"] = AsciiParser::VariableDef(value::kString, "sceneName");

  // Builtin from pxrUSD 23.xx
  metas["displayName"] =
      AsciiParser::VariableDef(value::kString, "displayName");

}

static void RegisterPropMetas(
    std::map<std::string, AsciiParser::VariableDef> &metas) {
  metas.clear();

  metas["doc"] = AsciiParser::VariableDef(value::kString, "doc");
  metas["active"] = AsciiParser::VariableDef(value::kBool, "active");
  metas["hidden"] = AsciiParser::VariableDef(value::kBool, "hidden");
  metas["customData"] =
      AsciiParser::VariableDef(value::kDictionary, "customData");

  // for sparse primvars
  metas["unauthoredValuesIndex"] =
      AsciiParser::VariableDef(value::kInt, "unauthoredValuesIndex");

  // usdSkel
  metas["elementSize"] = AsciiParser::VariableDef(value::kInt, "elementSize");

  // usdSkel inbetween BlendShape
  // use Double in TinyUSDZ. its float type in pxrUSD.
  metas["weight"] = AsciiParser::VariableDef(value::kDouble, "weight");

  // usdShade?
  metas["colorSpace"] = AsciiParser::VariableDef(value::kToken, "colorSpace");

  metas["interpolation"] =
      AsciiParser::VariableDef(value::kToken, "interpolation");

  // usdShade
  metas["bindMaterialAs"] =
      AsciiParser::VariableDef(value::kToken, "bindMaterialAs");
  metas["connectability"] =
      AsciiParser::VariableDef(value::kToken, "connectability");
  metas["renderType"] = AsciiParser::VariableDef(value::kToken, "renderType");
  metas["outputName"] = AsciiParser::VariableDef(value::kToken, "outputName");
  metas["sdrMetadata"] =
      AsciiParser::VariableDef(value::kDictionary, "sdrMetadata");

  // Builtin from pxrUSD 23.xx
  metas["displayName"] =
      AsciiParser::VariableDef(value::kString, "displayName");

  // Builtin from pxrUSD 24.xx?
  metas["displayGroup"] =
      AsciiParser::VariableDef(value::kString, "displayGroup");
}

static void RegisterPrimAttrTypes(std::set<std::string> &d) {
  d.clear();

  d.insert(value::kBool);

  d.insert(value::kInt64);

  d.insert(value::kInt);
  d.insert(value::kInt2);
  d.insert(value::kInt3);
  d.insert(value::kInt4);

  d.insert(value::kUInt64);

  d.insert(value::kUInt);
  d.insert(value::kUInt2);
  d.insert(value::kUInt3);
  d.insert(value::kUInt4);

  d.insert(value::kFloat);
  d.insert(value::kFloat2);
  d.insert(value::kFloat3);
  d.insert(value::kFloat4);

  d.insert(value::kDouble);
  d.insert(value::kDouble2);
  d.insert(value::kDouble3);
  d.insert(value::kDouble4);

  d.insert(value::kHalf);
  d.insert(value::kHalf2);
  d.insert(value::kHalf3);
  d.insert(value::kHalf4);

  d.insert(value::kQuath);
  d.insert(value::kQuatf);
  d.insert(value::kQuatd);

  d.insert(value::kNormal3f);
  d.insert(value::kPoint3f);
  d.insert(value::kTexCoord2h);
  d.insert(value::kTexCoord3h);
  d.insert(value::kTexCoord4h);
  d.insert(value::kTexCoord2f);
  d.insert(value::kTexCoord3f);
  d.insert(value::kTexCoord4f);
  d.insert(value::kTexCoord2d);
  d.insert(value::kTexCoord3d);
  d.insert(value::kTexCoord4d);
  d.insert(value::kVector3f);
  d.insert(value::kVector4f);
  d.insert(value::kVector3d);
  d.insert(value::kVector4d);
  d.insert(value::kColor3h);
  d.insert(value::kColor3f);
  d.insert(value::kColor3d);
  d.insert(value::kColor4h);
  d.insert(value::kColor4f);
  d.insert(value::kColor4d);

  d.insert(value::kMatrix2f);
  d.insert(value::kMatrix3f);
  d.insert(value::kMatrix4f);

  d.insert(value::kMatrix2d);
  d.insert(value::kMatrix3d);
  d.insert(value::kMatrix4d);

  d.insert(value::kToken);
  d.insert(value::kString);

  d.insert(value::kRelationship);
  d.insert(value::kAssetPath);

  d.insert(value::kDictionary);

  // variantSet. Require special treatment.
  d.insert("variantSet");

  // TODO: Add more types...
}

static void RegisterPrimTypes(std::set<std::string> &d) {
  d.insert("Xform");
  d.insert("Sphere");
  d.insert("Cube");
  d.insert("Cone");
  d.insert("Cylinder");
  d.insert("Capsule");
  d.insert("BasisCurves");
  d.insert("Mesh");
  d.insert("Points");
  d.insert("GeomSubset");
  d.insert("Scope");
  d.insert("Material");
  d.insert("NodeGraph");
  d.insert("Shader");
  d.insert("SphereLight");
  d.insert("DomeLight");
  d.insert("DiskLight");
  d.insert("DistantLight");
  d.insert("CylinderLight");
  // d.insert("PortalLight");
  d.insert("Camera");
  d.insert("SkelRoot");
  d.insert("Skeleton");
  d.insert("SkelAnimation");
  d.insert("BlendShape");

  d.insert("GPrim");
}

// TinyUSDZ does not allow user-defined API schema at the moment
// (Primarily for security reason, secondary it requires re-design of Prim
// classes to support user-defined API schema)
static void RegisterAPISchemas(std::set<std::string> &d) {
  d.insert("MaterialBindingAPI");
  d.insert("SkelBindingAPI");

  // TODO:
  // d.insert("PhysicsCollisionAPI");
  // d.insert("PhysicsRigidBodyAPI");

  // TODO: Support Multi-apply API(`CollectionAPI`)
  // d.insert("PhysicsLimitAPI");
  // d.insert("PhysicsDriveAPI");
  // d.insert("CollectionAPI");
}

namespace {

using ReferenceList = std::vector<std::pair<ListEditQual, Reference>>;

// https://www.techiedelight.com/trim-string-cpp-remove-leading-trailing-spaces/
std::string TrimString(const std::string &str) {
  const std::string WHITESPACE = " \n\r\t\f\v";

  // remove leading and trailing whitespaces
  std::string s = str;
  {
    size_t start = s.find_first_not_of(WHITESPACE);
    s = (start == std::string::npos) ? "" : s.substr(start);
  }

  {
    size_t end = s.find_last_not_of(WHITESPACE);
    s = (end == std::string::npos) ? "" : s.substr(0, end + 1);
  }

  return s;
}

}  // namespace

inline bool isChar(char c) { return std::isalpha(int(c)); }

inline bool hasConnect(const std::string &str) {
  return endsWith(str, ".connect");
}

inline bool hasInputs(const std::string &str) {
  return startsWith(str, "inputs:");
}

inline bool hasOutputs(const std::string &str) {
  return startsWith(str, "outputs:");
}

inline bool is_digit(char x) {
  return (static_cast<unsigned int>((x) - '0') < static_cast<unsigned int>(10));
}

void AsciiParser::SetBaseDir(const std::string &str) { _base_dir = str; }

void AsciiParser::SetStream(StreamReader *sr) { _sr = sr; }

std::string AsciiParser::GetError() {
  if (err_stack.empty()) {
    return std::string();
  }

  std::stringstream ss;
  while (!err_stack.empty()) {
    ErrorDiagnostic diag = err_stack.top();

    ss << "err_stack[" << (err_stack.size() - 1) << "] USDA source near line "
       << (diag.cursor.row + 1) << ", col " << (diag.cursor.col + 1) << ": ";
    ss << diag.err;  // assume message contains newline.

    err_stack.pop();
  }

  return ss.str();
}

std::string AsciiParser::GetWarning() {
  if (warn_stack.empty()) {
    return std::string();
  }

  std::stringstream ss;
  while (!warn_stack.empty()) {
    ErrorDiagnostic diag = warn_stack.top();

    ss << "USDA source near line " << (diag.cursor.row + 1) << ", col "
       << (diag.cursor.col + 1) << ": ";
    ss << diag.err;  // assume message contains newline.

    warn_stack.pop();
  }

  return ss.str();
}

// -- end basic

// types: Allowd in dict.
// std::string is not included since its represented as StringData or
// std::string.
// TODO: Include timecode?
#define APPLY_TO_METAVARIABLE_TYPE(__FUNC) \
  __FUNC(value::token)                     \
  __FUNC(bool)                             \
  __FUNC(value::half)                      \
  __FUNC(value::half2)                     \
  __FUNC(value::half3)                     \
  __FUNC(value::half4)                     \
  __FUNC(int32_t)                          \
  __FUNC(uint32_t)                         \
  __FUNC(value::int2)                      \
  __FUNC(value::int3)                      \
  __FUNC(value::int4)                      \
  __FUNC(value::uint2)                     \
  __FUNC(value::uint3)                     \
  __FUNC(value::uint4)                     \
  __FUNC(int64_t)                          \
  __FUNC(uint64_t)                         \
  __FUNC(float)                            \
  __FUNC(value::float2)                    \
  __FUNC(value::float3)                    \
  __FUNC(value::float4)                    \
  __FUNC(double)                           \
  __FUNC(value::double2)                   \
  __FUNC(value::double3)                   \
  __FUNC(value::double4)                   \
  __FUNC(value::matrix2f)                  \
  __FUNC(value::matrix3f)                  \
  __FUNC(value::matrix4f)                  \
  __FUNC(value::matrix2d)                  \
  __FUNC(value::matrix3d)                  \
  __FUNC(value::matrix4d)                  \
  __FUNC(value::quath)                     \
  __FUNC(value::quatf)                     \
  __FUNC(value::quatd)                     \
  __FUNC(value::normal3h)                  \
  __FUNC(value::normal3f)                  \
  __FUNC(value::normal3d)                  \
  __FUNC(value::vector3h)                  \
  __FUNC(value::vector3f)                  \
  __FUNC(value::vector3d)                  \
  __FUNC(value::point3h)                   \
  __FUNC(value::point3f)                   \
  __FUNC(value::point3d)                   \
  __FUNC(value::color3f)                   \
  __FUNC(value::color3d)                   \
  __FUNC(value::color4f)                   \
  __FUNC(value::color4d)                   \
  __FUNC(value::texcoord2h)                \
  __FUNC(value::texcoord2f)                \
  __FUNC(value::texcoord2d)                \
  __FUNC(value::texcoord3h)                \
  __FUNC(value::texcoord3f)                \
  __FUNC(value::texcoord3d)

bool AsciiParser::ParseDictElement(std::string *out_key,
                                   MetaVariable *out_var) {
  (void)out_key;
  (void)out_var;

  // dict_element: type (array_qual?) name '=' value
  //           ;

  std::string type_name;

  if (!ReadIdentifier(&type_name)) {
    return false;
  }

  if (!SkipWhitespace()) {
    return false;
  }

  if (!IsSupportedPrimAttrType(type_name)) {
    PUSH_ERROR_AND_RETURN("Unknown or unsupported type `" + type_name + "`\n");
  }

  // Has array qualifier? `[]`
  bool array_qual = false;
  {
    char c0, c1;
    if (!Char1(&c0)) {
      return false;
    }

    if (c0 == '[') {
      if (!Char1(&c1)) {
        return false;
      }

      if (c1 == ']') {
        array_qual = true;
      } else {
        // Invalid syntax
        PUSH_ERROR_AND_RETURN("Invalid syntax found.");
      }

    } else {
      if (!Rewind(1)) {
        return false;
      }
    }
  }

  if (!SkipWhitespace()) {
    return false;
  }

  std::string key_name;
  if (!ReadIdentifier(&key_name)) {
    // string literal is also supported. e.g. "0"
    if (ReadStringLiteral(&key_name)) {
      // ok
    } else {
      PUSH_ERROR_AND_RETURN("Failed to parse dictionary key identifier.\n");
    }
  }

  if (!SkipWhitespace()) {
    return false;
  }

  if (!Expect('=')) {
    return false;
  }

  if (!SkipWhitespace()) {
    return false;
  }

  uint32_t tyid = value::GetTypeId(type_name);

  primvar::PrimVar var;

  //
  // Supports limited types for customData/Dictionary.
  //

  // TODO: Unify code with ParseMetaValue()

#define PARSE_BASE_TYPE(__ty)                                     \
  case value::TypeTraits<__ty>::type_id(): {                      \
    if (array_qual) {                                             \
      std::vector<__ty> vss;                                      \
      if (!ParseBasicTypeArray(&vss)) {                           \
        PUSH_ERROR_AND_RETURN(                                    \
            fmt::format("Failed to parse a value of type `{}[]`", \
                        value::TypeTraits<__ty>::type_name()));   \
      }                                                           \
      var.set_value(vss);                                         \
    } else {                                                      \
      __ty val;                                                   \
      if (!ReadBasicType(&val)) {                                 \
        PUSH_ERROR_AND_RETURN(                                    \
            fmt::format("Failed to parse a value of type `{}`",   \
                        value::TypeTraits<__ty>::type_name()));   \
      }                                                           \
      var.set_value(val);                                         \
    }                                                             \
    break;                                                        \
  }

  switch (tyid) {
    APPLY_TO_METAVARIABLE_TYPE(PARSE_BASE_TYPE)
    case value::TYPE_ID_STRING: {
      // FIXME: Use std::string
      if (array_qual) {
        std::vector<value::StringData> strs;
        if (!ParseBasicTypeArray(&strs)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `string[]`");
        }
        var.set_value(strs);
      } else {
        value::StringData str;
        if (!ReadBasicType(&str)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `string`");
        }
        var.set_value(str);
      }
      break;
    }
    case value::TYPE_ID_ASSET_PATH: {
      if (array_qual) {
        std::vector<value::AssetPath> arrs;
        if (!ParseBasicTypeArray(&arrs)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `asset[]`");
        }
        var.set_value(arrs);
      } else {
        value::AssetPath asset;
        if (!ReadBasicType(&asset)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `asset`");
        }
        var.set_value(asset);
      }
      break;
    }
    case value::TYPE_ID_DICT: {
      Dictionary dict;

      DCOUT("Parse dictionary");
      if (!ParseDict(&dict)) {
        PUSH_ERROR_AND_RETURN("Failed to parse `dictionary`");
      }
      var.set_value(dict);
      break;
    }
    default: {
      PUSH_ERROR_AND_RETURN("Unsupported or invalid type for Metadatum:" +
                            type_name);
    }
  }

#undef PARSE_BASE_TYPE

  MetaVariable metavar;
  metavar.set_value(key_name, var.value_raw());

  DCOUT("key: " << key_name << ", type: " << type_name);

  (*out_key) = key_name;
  (*out_var) = metavar;

  return true;
}

bool AsciiParser::MaybeCustom() {
  std::string tok;

  auto loc = CurrLoc();
  bool ok = ReadIdentifier(&tok);

  if (!ok) {
    // revert
    SeekTo(loc);
    return false;
  }

  if (tok == "custom") {
    // cosume `custom` token.
    return true;
  }

  // revert
  SeekTo(loc);
  return false;
}

bool AsciiParser::ParseDict(std::map<std::string, MetaVariable> *out_dict) {
  // '{' comment | (type name '=' value)+ '}'
  if (!Expect('{')) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == '}') {
      break;
    } else {
      if (!Rewind(1)) {
        return false;
      }

      std::string key;
      MetaVariable var;
      if (!ParseDictElement(&key, &var)) {
        PUSH_ERROR_AND_RETURN("Failed to parse dict element.");
      }

      if (!SkipCommentAndWhitespaceAndNewline()) {
        return false;
      }

      if (!var.is_valid()) {
        PUSH_ERROR_AND_RETURN("Invalid Dict element(probably internal issue).");
      }

      DCOUT("Add to dict: " << key);
      (*out_dict)[key] = var;
    }
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  return true;
}

bool AsciiParser::ParseVariantsElement(std::string *out_key,
                                       std::string *out_var) {
  // variants_element: string name '=' value
  //           ;

  std::string type_name;

  if (!ReadIdentifier(&type_name)) {
    return false;
  }

  // must be `string`
  if (type_name != value::kString) {
    PUSH_ERROR_AND_RETURN(
        "TinyUSDZ only accepts type `string` for `variants` element.");
  }

  if (!SkipWhitespace()) {
    return false;
  }

  std::string key_name;
  if (!ReadIdentifier(&key_name)) {
    // string literal is also supported. e.g. "0"
    if (ReadStringLiteral(&key_name)) {
      // ok
    } else {
      PUSH_ERROR_AND_RETURN("Failed to parse dictionary key identifier.\n");
    }
  }

  if (!SkipWhitespace()) {
    return false;
  }

  if (!Expect('=')) {
    return false;
  }

  if (!SkipWhitespace()) {
    return false;
  }

  std::string var;
  if (!ReadBasicType(&var)) {
    PUSH_ERROR_AND_RETURN("Failed to parse `string`");
  }

  DCOUT("key: " << key_name << ", value: " << var);

  (*out_key) = key_name;
  (*out_var) = var;

  return true;
}

bool AsciiParser::ParseVariants(VariantSelectionMap *out_map) {
  // '{' (string name '=' value)+ '}'
  if (!Expect('{')) {
    return false;
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == '}') {
      break;
    } else {
      if (!Rewind(1)) {
        return false;
      }

      std::string key;
      std::string var;
      if (!ParseVariantsElement(&key, &var)) {
        PUSH_ERROR_AND_RETURN("Failed to parse an element of `variants`.");
      }

      if (!SkipWhitespaceAndNewline()) {
        return false;
      }

      DCOUT("Add to variants: " << key);
      (*out_map)[key] = var;
    }
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  return true;
}

// 'None'
bool AsciiParser::MaybeNone() {
  std::vector<char> buf;

  auto loc = CurrLoc();

  if (!CharN(4, &buf)) {
    SeekTo(loc);
    return false;
  }

  if ((buf[0] == 'N') && (buf[1] == 'o') && (buf[2] == 'n') &&
      (buf[3] == 'e')) {
    // got it
    return true;
  }

  SeekTo(loc);

  return false;
}

bool AsciiParser::MaybeListEditQual(tinyusdz::ListEditQual *qual) {
  if (!SkipWhitespace()) {
    return false;
  }

  std::string tok;

  auto loc = CurrLoc();
  if (!ReadIdentifier(&tok)) {
    SeekTo(loc);
    return false;
  }

  if (tok == "prepend") {
    DCOUT("`prepend` list edit qualifier.");
    (*qual) = tinyusdz::ListEditQual::Prepend;
  } else if (tok == "append") {
    DCOUT("`append` list edit qualifier.");
    (*qual) = tinyusdz::ListEditQual::Append;
  } else if (tok == "add") {
    DCOUT("`add` list edit qualifier.");
    (*qual) = tinyusdz::ListEditQual::Add;
  } else if (tok == "delete") {
    DCOUT("`delete` list edit qualifier.");
    (*qual) = tinyusdz::ListEditQual::Delete;
  } else if (tok == "order") {
    DCOUT("`order` list edit qualifier.");
    (*qual) = tinyusdz::ListEditQual::Order;
  } else {
    DCOUT("No ListEdit qualifier.");
    // unqualified
    // rewind
    SeekTo(loc);
    (*qual) = tinyusdz::ListEditQual::ResetToExplicit;
  }

  if (!SkipWhitespace()) {
    return false;
  }

  return true;
}

bool AsciiParser::MaybeVariability(tinyusdz::Variability *variability,
                                   bool *varying_authored) {
  if (!SkipWhitespace()) {
    return false;
  }

  std::string tok;

  auto loc = CurrLoc();
  if (!ReadIdentifier(&tok)) {
    SeekTo(loc);
    return false;
  }

  if (tok == "uniform") {
    (*variability) = tinyusdz::Variability::Uniform;
    (*varying_authored) = false;
  } else if (tok == "varying") {
    (*variability) = tinyusdz::Variability::Varying;
    (*varying_authored) = true;
  } else {
    (*varying_authored) = false;
    // rewind
    SeekTo(loc);
  }

  if (!SkipWhitespace()) {
    return false;
  }

  return true;
}

bool AsciiParser::IsSupportedPrimType(const std::string &ty) {
  return _supported_prim_types.count(ty);
}

bool AsciiParser::IsSupportedPrimAttrType(const std::string &ty) {
  return _supported_prim_attr_types.count(ty);
}

bool AsciiParser::IsSupportedAPISchema(const std::string &ty) {
  return _supported_api_schemas.count(ty);
}

bool AsciiParser::ReadStringLiteral(std::string *literal) {
  std::stringstream ss;

  char c0;
  if (!Char1(&c0)) {
    return false;
  }

  // TODO: Allow triple-quotated string?

  bool single_quote{false};

  if (c0 == '"') {
    // ok
  } else if (c0 == '\'') {
    // ok
    single_quote = true;
  } else {
    DCOUT("c0 = " << c0);
    PUSH_ERROR_AND_RETURN(
        "String or Token literal expected but it does not start with \" or '");
  }

  bool end_with_quotation{false};

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    if ((c == '\n') || (c == '\r')) {
      PUSH_ERROR_AND_RETURN("New line in string literal.");
    }

    if (single_quote) {
      if (c == '\'') {
        end_with_quotation = true;
        break;
      }
    } else if (c == '"') {
      end_with_quotation = true;
      break;
    }

    ss << c;
  }

  if (!end_with_quotation) {
    PUSH_ERROR_AND_RETURN(
        fmt::format("String literal expected but it does not end with {}.",
                    single_quote ? "'" : "\""));
  }

  (*literal) = ss.str();

  _curr_cursor.col += int(literal->size() + 2);  // +2 for quotation chars

  return true;
}

bool AsciiParser::MaybeString(value::StringData *str) {
  std::stringstream ss;

  if (!str) {
    return false;
  }

  auto loc = CurrLoc();
  auto start_cursor = _curr_cursor;

  char c0;
  if (!Char1(&c0)) {
    SeekTo(loc);
    return false;
  }

  // ' or " allowed.
  if ((c0 != '"') && (c0 != '\'')) {
    SeekTo(loc);
    return false;
  }

  bool single_quote = (c0 == '\'');

  bool end_with_quotation{false};

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      SeekTo(loc);
      return false;
    }

    if ((c == '\n') || (c == '\r')) {
      SeekTo(loc);
      return false;
    }

    if (c == '\\') {
      // escaped quote? \" \'
      char nc;
      if (!LookChar1(&nc)) {
        return false;
      }

      if (nc == '\'') {
        ss << "'";
        _sr->seek_from_current(1);  // advance 1 char
        continue;
      } else if (nc == '"') {
        ss << "\"";
        _sr->seek_from_current(1);  // advance 1 char
        continue;
      }
    }

    if (single_quote) {
      if (c == '\'') {
        end_with_quotation = true;
        break;
      }
    } else {
      if (c == '"') {
        end_with_quotation = true;
        break;
      }
    }

    ss << c;
  }

  if (!end_with_quotation) {
    SeekTo(loc);
    return false;
  }

  DCOUT("Single quoted string found. col " << start_cursor.col << ", row "
                                           << start_cursor.row);

  size_t displayed_string_len = ss.str().size();
  str->value = unescapeControlSequence(ss.str());
  str->line_col = start_cursor.col;
  str->line_row = start_cursor.row;
  str->is_triple_quoted = false;

  _curr_cursor.col += int(displayed_string_len + 2);  // +2 for quotation chars

  return true;
}

bool AsciiParser::MaybeTripleQuotedString(value::StringData *str) {
  std::stringstream ss;

  auto loc = CurrLoc();
  auto start_cursor = _curr_cursor;

  std::vector<char> triple_quote;
  if (!CharN(3, &triple_quote)) {
    SeekTo(loc);
    return false;
  }

  if (triple_quote.size() != 3) {
    SeekTo(loc);
    return false;
  }

  bool single_quote = false;

  if (triple_quote[0] == '"' && triple_quote[1] == '"' &&
      triple_quote[2] == '"') {
    // ok
  } else if (triple_quote[0] == '\'' && triple_quote[1] == '\'' &&
             triple_quote[2] == '\'') {
    // ok
    single_quote = true;
  } else {
    SeekTo(loc);
    return false;
  }

  // Read until next triple-quote `"""` or "'''"
  std::stringstream str_buf;

  auto locinfo = _curr_cursor;

  int single_quote_count = 0;  // '
  int double_quote_count = 0;  // "

  bool got_closing_triple_quote{false};

  while (!Eof()) {
    char c;

    if (!Char1(&c)) {
      SeekTo(loc);
      return false;
    }

    // Seek \""" or \'''
    // Unescape '\'
    if (c == '\\') {
      std::vector<char> buf(3, '\0');
      if (!LookCharN(3, &buf)) {
        // at least 3 chars should be read
        return false;
      }

      if (buf[0] == '\'' && buf[1] == '\'' && buf[2] == '\'') {
        str_buf << "'''";
        // advance
        _sr->seek_from_current(3);
        locinfo.col += 3;
        continue;
      } else if (buf[0] == '"' && buf[1] == '"' && buf[2] == '"') {
        str_buf << "\"\"\"";
        // advance
        _sr->seek_from_current(3);
        locinfo.col += 3;
        continue;
      }
    }

    str_buf << c;

    if (c == '"') {
      double_quote_count++;
      single_quote_count = 0;
    } else if (c == '\'') {
      double_quote_count = 0;
      single_quote_count++;
    } else {
      double_quote_count = 0;
      single_quote_count = 0;
    }

    // Update loc info
    locinfo.col++;
    if (c == '\n') {
      locinfo.col = 0;
      locinfo.row++;
    } else if (c == '\r') {
      // CRLF?
      if (_sr->tell() < (_sr->size() - 1)) {
        char d;
        if (!Char1(&d)) {
          // this should not happen.
          SeekTo(loc);
          return false;
        }

        if (d == '\n') {
          // CRLF
          str_buf << d;
        } else {
          // unwind 1 char
          if (!_sr->seek_from_current(-1)) {
            // this should not happen.
            SeekTo(loc);
            return false;
          }
        }
      }
      locinfo.col = 0;
      locinfo.row++;
    }

    if (double_quote_count == 3) {
      // got '"""'
      if (single_quote) {
        // continue
      } else {
        got_closing_triple_quote = true;
        break;
      }
    }
    if (single_quote_count == 3) {
      // got '''
      if (double_quote_count) {
        // continue
      } else {
        got_closing_triple_quote = true;
        break;
      }
    }
  }

  if (!got_closing_triple_quote) {
    SeekTo(loc);
    return false;
  }

  DCOUT("single_quote = " << single_quote);
  DCOUT("Triple quoted string found. col " << start_cursor.col << ", row "
                                           << start_cursor.row);

  // remove last '"""' or '''
  str->single_quote = single_quote;
  std::string s = str_buf.str();
  if (s.size() > 3) {  // just in case
    s.erase(s.size() - 3);
  }

  DCOUT("str = " << s);

  str->value = unescapeControlSequence(s);

  DCOUT("unescape str = " << str->value);

  str->line_col = start_cursor.col;
  str->line_row = start_cursor.row;
  str->is_triple_quoted = true;

  _curr_cursor = locinfo;

  return true;
}

bool AsciiParser::ReadPrimAttrIdentifier(std::string *token) {
  // Example:
  // - xformOp:transform
  // - primvars:uvmap1

  std::stringstream ss;

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    if (c == '_') {
      // ok
    } else if (c == ':') {  // namespace
      // ':' must lie in the middle of string literal
      if (ss.str().size() == 0) {
        PUSH_ERROR_AND_RETURN("PrimAttr name must not starts with `:`");
      }
    } else if (c == '.') {  // delimiter for `connect`
      // '.' must lie in the middle of string literal
      if (ss.str().size() == 0) {
        PUSH_ERROR_AND_RETURN("PrimAttr name must not starts with `.`");
      }
    } else if (std::isalnum(int(c))) {
      // number must not be allowed for the first char.
      if (ss.str().size() == 0) {
        if (!std::isalpha(int(c))) {
          PUSH_ERROR_AND_RETURN("PrimAttr name must not starts with number.");
        }
      }
    } else {
      _sr->seek_from_current(-1);
      break;
    }

    _curr_cursor.col++;

    ss << c;
  }

  {
    std::string name_err;
    if (!pathutil::ValidatePropPath(Path("", ss.str()), &name_err)) {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii,
          fmt::format("Invalid Property name `{}`: {}", ss.str(), name_err));
    }
  }

  // '.' must lie in the middle of string literal
  if (ss.str().back() == '.') {
    PUSH_ERROR_AND_RETURN("PrimAttr name must not ends with `.`\n");
    return false;
  }

  std::string tok = ss.str();

  if (contains(tok, '.')) {
    if (endsWith(tok, ".connect") || endsWith(tok, ".timeSamples")) {
      // OK
    } else {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii, fmt::format("Must ends with `.connect` or `.timeSamples` for "
                              "attrbute name: `{}`",
                              tok));
    }

    // Multiple `.` is not allowed(e.g. attr.connect.timeSamples)
    if (counts(tok, '.') > 1) {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii, fmt::format("Attribute identifier `{}` containing multiple "
                              "`.` is not allowed.",
                              tok));
    }
  }

  (*token) = ss.str();
  DCOUT("primAttr identifier = " << (*token));
  return true;
}

bool AsciiParser::ReadIdentifier(std::string *token) {
  // identifier = (`_` | [a-zA-Z]) (`_` | [a-zA-Z0-9]+)
  std::stringstream ss;

  // The first character.
  {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      DCOUT("read1 failed.");
      return false;
    }

    if (c == '_') {
      // ok
    } else if (!std::isalpha(int(c))) {
      DCOUT(fmt::format("Invalid identiefier: '{}'", c));
      _sr->seek_from_current(-1);
      return false;
    }
    _curr_cursor.col++;

    ss << c;
  }

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    if (c == '_') {
      // ok
    } else if (!std::isalnum(int(c))) {
      _sr->seek_from_current(-1);
      break;  // end of identifier(e.g. ' ')
    }

    _curr_cursor.col++;

    ss << c;
  }

  (*token) = ss.str();
  return true;
}

bool AsciiParser::ReadPathIdentifier(std::string *path_identifier) {
  // path_identifier = `<` string `>`
  std::stringstream ss;

  if (!Expect('<')) {
    return false;
  }

  if (!SkipWhitespace()) {
    return false;
  }

  // read until '>'
  bool ok = false;
  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    if (c == '>') {
      // end
      ok = true;
      _curr_cursor.col++;
      break;
    }

    // TODO: Check if character is valid for path identifier
    ss << c;
  }

  if (!ok) {
    return false;
  }

  (*path_identifier) = TrimString(ss.str());
  // std::cout << "PathIdentifier: " << (*path_identifier) << "\n";

  return true;
}

bool AsciiParser::ReadUntilNewline(std::string *str) {
  std::stringstream ss;

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    if (c == '\n') {
      break;
    } else if (c == '\r') {
      // CRLF?
      if (_sr->tell() < (_sr->size() - 1)) {
        char d;
        if (!Char1(&d)) {
          // this should not happen.
          return false;
        }

        if (d == '\n') {
          break;
        }

        // unwind 1 char
        if (!_sr->seek_from_current(-1)) {
          // this should not happen.
          return false;
        }

        break;
      }
    }

    ss << c;
  }

  _curr_cursor.row++;
  _curr_cursor.col = 0;

  (*str) = ss.str();

  return true;
}

bool AsciiParser::SkipUntilNewline() {
  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    if (c == '\n') {
      break;
    } else if (c == '\r') {
      // CRLF?
      if (_sr->tell() < (_sr->size() - 1)) {
        char d;
        if (!Char1(&d)) {
          // this should not happen.
          return false;
        }

        if (d == '\n') {
          break;
        }

        // unwind 1 char
        if (!_sr->seek_from_current(-1)) {
          // this should not happen.
          return false;
        }

        break;
      }

    } else {
      // continue
    }
  }

  _curr_cursor.row++;
  _curr_cursor.col = 0;
  return true;
}

// metadata_opt := string_literal '\n'
//              |  var '=' value '\n'
//
bool AsciiParser::ParseStageMetaOpt() {
  // Maybe string-only comment.
  // Comment cannot have multiple lines. The last one wins
  {
    value::StringData str;
    if (MaybeTripleQuotedString(&str)) {
      _stage_metas.comment = str;
      return true;
    } else if (MaybeString(&str)) {
      _stage_metas.comment = str;
      return true;
    }
  }

  std::string varname;
  if (!ReadIdentifier(&varname)) {
    return false;
  }

  DCOUT("varname = " << varname);

  if (!IsStageMeta(varname)) {
    std::string msg = "'" + varname + "' is not a Stage Metadata variable.\n";
    PUSH_ERROR_AND_RETURN(msg);
    return false;
  }

  if (!Expect('=')) {
    PUSH_ERROR_AND_RETURN("'=' expected in Stage Metadata opt.");
    return false;
  }

  if (!SkipWhitespace()) {
    return false;
  }

  const VariableDef &vardef = _supported_stage_metas.at(varname);
  MetaVariable var;
  if (!ParseMetaValue(vardef, &var)) {
    PUSH_ERROR_AND_RETURN("Failed to parse meta value.\n");
    return false;
  }
  var.set_name(varname);

  if (varname == "defaultPrim") {
    value::token tok;
    if (var.get_value(&tok)) {
      DCOUT("defaultPrim = " << tok);
      _stage_metas.defaultPrim = tok;
    } else {
      PUSH_ERROR_AND_RETURN("`defaultPrim` isn't a token value.");
    }
  } else if (varname == "subLayers") {
    std::vector<value::AssetPath> paths;
    if (var.get_value(&paths)) {
      DCOUT("subLayers = " << paths);
      for (const auto &item : paths) {
        _stage_metas.subLayers.push_back(item);
      }
    } else {
      PUSH_ERROR_AND_RETURN("`subLayers` isn't an array of asset path");
    }
  } else if (varname == "upAxis") {
    if (auto pv = var.get_value<value::token>()) {
      DCOUT("upAxis = " << pv.value());
      const std::string s = pv.value().str();
      if (s == "X") {
        _stage_metas.upAxis = Axis::X;
      } else if (s == "Y") {
        _stage_metas.upAxis = Axis::Y;
      } else if (s == "Z") {
        _stage_metas.upAxis = Axis::Z;
      } else {
        if (_option.strict_allowedToken_check) {
          PUSH_ERROR_AND_RETURN(
              "Invalid `upAxis` value. Must be \"X\", \"Y\" or \"Z\", but got "
              "\"" +
              s + "\"(Note: Case sensitive)");
        } else {
          PUSH_WARN(
              "Ignore unknown `upAxis` value. Must be \"X\", \"Y\" or \"Z\", "
              "but got "
              "\"" +
              s + "\"(Note: Case sensitive). Use default upAxis `Y`.");
          _stage_metas.upAxis = Axis::Y;
        }
      }
    } else {
      PUSH_ERROR_AND_RETURN("`upAxis` isn't a token value.");
    }
  } else if ((varname == "doc") || (varname == "documentation")) {
    // `documentation` will be shorten to `doc`
    if (auto pv = var.get_value<value::StringData>()) {
      DCOUT("doc = " << to_string(pv.value()));
      _stage_metas.doc = pv.value();
    } else if (auto pvs = var.get_value<std::string>()) {
      value::StringData sdata;
      sdata.value = pvs.value();
      sdata.is_triple_quoted = false;
      _stage_metas.doc = sdata;
    } else {
      PUSH_ERROR_AND_RETURN(fmt::format("`{}` isn't a string value.", varname));
    }
  } else if (varname == "metersPerUnit") {
    DCOUT("ty = " << var.type_name());
    if (auto pv = var.get_value<float>()) {
      DCOUT("metersPerUnit = " << pv.value());
      _stage_metas.metersPerUnit = double(pv.value());
    } else if (auto pvd = var.get_value<double>()) {
      DCOUT("metersPerUnit = " << pvd.value());
      _stage_metas.metersPerUnit = pvd.value();
    } else {
      PUSH_ERROR_AND_RETURN("`metersPerUnit` isn't a floating-point value.");
    }
  } else if (varname == "kilogramsPerUnit") {
    DCOUT("ty = " << var.type_name());
    if (auto pv = var.get_value<float>()) {
      DCOUT("kilogramsPerUnit = " << pv.value());
      _stage_metas.kilogramsPerUnit = double(pv.value());
    } else if (auto pvd = var.get_value<double>()) {
      DCOUT("kilogramsPerUnit = " << pvd.value());
      _stage_metas.kilogramsPerUnit = pvd.value();
    } else {
      PUSH_ERROR_AND_RETURN("`kilogramsPerUnit` isn't a floating-point value.");
    }
  } else if (varname == "timeCodesPerSecond") {
    DCOUT("ty = " << var.type_name());
    if (auto pv = var.get_value<float>()) {
      DCOUT("metersPerUnit = " << pv.value());
      _stage_metas.timeCodesPerSecond = double(pv.value());
    } else if (auto pvd = var.get_value<double>()) {
      DCOUT("metersPerUnit = " << pvd.value());
      _stage_metas.timeCodesPerSecond = pvd.value();
    } else {
      PUSH_ERROR_AND_RETURN(
          "`timeCodesPerSecond` isn't a floating-point value.");
    }
  } else if (varname == "startTimeCode") {
    if (auto pv = var.get_value<float>()) {
      DCOUT("startTimeCode = " << pv.value());
      _stage_metas.startTimeCode = double(pv.value());
    } else if (auto pvd = var.get_value<double>()) {
      DCOUT("startTimeCode = " << pvd.value());
      _stage_metas.startTimeCode = pvd.value();
    }
  } else if (varname == "endTimeCode") {
    if (auto pv = var.get_value<float>()) {
      DCOUT("endTimeCode = " << pv.value());
      _stage_metas.endTimeCode = double(pv.value());
    } else if (auto pvd = var.get_value<double>()) {
      DCOUT("endTimeCode = " << pvd.value());
      _stage_metas.endTimeCode = pvd.value();
    }
  } else if (varname == "framesPerSecond") {
    if (auto pv = var.get_value<float>()) {
      DCOUT("framesPerSecond = " << pv.value());
      _stage_metas.framesPerSecond = double(pv.value());
    } else if (auto pvd = var.get_value<double>()) {
      DCOUT("framesPerSecond = " << pvd.value());
      _stage_metas.framesPerSecond = pvd.value();
    }
  } else if (varname == "apiSchemas") {
    // TODO: ListEdit qualifer check
    if (auto pv = var.get_value<std::vector<value::token>>()) {
      for (auto &item : pv.value()) {
        if (IsSupportedAPISchema(item.str())) {
          // OK
        } else {
          PUSH_ERROR_AND_RETURN("\"" << item.str()
                                     << "\" is not supported(at the moment) "
                                        "for `apiSchemas` in TinyUSDZ.");
        }
      }
    } else {
      PUSH_ERROR_AND_RETURN("`apiSchemas` isn't an `token[]` type.");
    }
  } else if (varname == "customLayerData") {
    if (auto pv = var.get_value<Dictionary>()) {
      _stage_metas.customLayerData = pv.value();
    } else {
      PUSH_ERROR_AND_RETURN("`customLayerData` isn't a dictionary value.");
    }
  } else if (varname == "comment") {
    if (auto pv = var.get_value<value::StringData>()) {
      DCOUT("comment = " << to_string(pv.value()));
      _stage_metas.comment = pv.value();
    } else if (auto pvs = var.get_value<std::string>()) {
      value::StringData sdata;
      sdata.value = pvs.value();
      sdata.is_triple_quoted = false;
      _stage_metas.comment = sdata;
    } else {
      PUSH_ERROR_AND_RETURN(fmt::format("`{}` isn't a string value.", varname));
    }
  } else {
    DCOUT("TODO: Stage meta: " << varname);
    PUSH_WARN("TODO: Stage meta: " << varname);
  }

  return true;
}

// Parse Stage meta
// meta = '(' (comment | metadata_opt)+ ')'
//      ;
bool AsciiParser::ParseStageMetas() {
  if (!Expect('(')) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  while (!Eof()) {
    char c;
    if (!LookChar1(&c)) {
      return false;
    }

    if (c == ')') {
      if (!SeekTo(CurrLoc() + 1)) {
        return false;
      }

      if (!SkipCommentAndWhitespaceAndNewline()) {
        return false;
      }

      DCOUT("Stage metas end");

      // end
      return true;

    } else {
      if (!SkipCommentAndWhitespaceAndNewline()) {
        // eof
        return false;
      }

      if (!ParseStageMetaOpt()) {
        // parse error
        return false;
      }
    }

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }
  }

  DCOUT("ParseStageMetas end");
  return true;
}

// `#` style comment
bool AsciiParser::ParseSharpComment() {
  char c;
  if (!Char1(&c)) {
    // eol
    return false;
  }

  if (c != '#') {
    return false;
  }

  return true;
}

// Fetch 1 char. Do not change input stream position.
bool AsciiParser::LookChar1(char *c) {
  if (!Char1(c)) {
    return false;
  }

  Rewind(1);

  return true;
}

// Fetch N chars. Do not change input stream position.
bool AsciiParser::LookCharN(size_t n, std::vector<char> *nc) {
  std::vector<char> buf(n);

  auto loc = CurrLoc();

  bool ok = _sr->read(n, n, reinterpret_cast<uint8_t *>(buf.data()));
  if (ok) {
    (*nc) = buf;
  }

  SeekTo(loc);

  return ok;
}

bool AsciiParser::Char1(char *c) { return _sr->read1(c); }

bool AsciiParser::CharN(size_t n, std::vector<char> *nc) {
  std::vector<char> buf(n);

  bool ok = _sr->read(n, n, reinterpret_cast<uint8_t *>(buf.data()));
  if (ok) {
    (*nc) = buf;
  }

  return ok;
}

bool AsciiParser::Rewind(size_t offset) {
  if (!_sr->seek_from_current(-int64_t(offset))) {
    return false;
  }

  return true;
}

uint64_t AsciiParser::CurrLoc() { return _sr->tell(); }

bool AsciiParser::SeekTo(uint64_t pos) {
  if (!_sr->seek_set(pos)) {
    return false;
  }

  return true;
}

bool AsciiParser::PushParserState() {
  // Stack size must be less than the number of input bytes.
  if (parse_stack.size() >= _sr->size()) {
    PUSH_ERROR_AND_RETURN_TAG(kAscii, "Parser state stack become too deep.");
  }

  uint64_t loc = _sr->tell();

  ParseState state;
  state.loc = int64_t(loc);
  parse_stack.push(state);

  return true;
}

bool AsciiParser::PopParserState(ParseState *state) {
  if (parse_stack.empty()) {
    return false;
  }

  (*state) = parse_stack.top();

  parse_stack.pop();

  return true;
}

bool AsciiParser::SkipWhitespace() {
  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }
    _curr_cursor.col++;

    if ((c == ' ') || (c == '\t') || (c == '\f')) {
      // continue
    } else {
      break;
    }
  }

  // unwind 1 char
  if (!_sr->seek_from_current(-1)) {
    return false;
  }
  _curr_cursor.col--;

  return true;
}

bool AsciiParser::SkipWhitespaceAndNewline(const bool allow_semicolon) {
  // USDA also allow C-style ';' as a newline separator.
  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    // printf("sws c = %c\n", c);

    if ((c == ' ') || (c == '\t') || (c == '\f')) {
      _curr_cursor.col++;
      // continue
    } else if (allow_semicolon && (c == ';')) {
      _curr_cursor.col++;
      // continue
    } else if (c == '\n') {
      _curr_cursor.col = 0;
      _curr_cursor.row++;
      // continue
    } else if (c == '\r') {
      // CRLF?
      if (_sr->tell() < (_sr->size() - 1)) {
        char d;
        if (!Char1(&d)) {
          // this should not happen.
          return false;
        }

        if (d == '\n') {
          // CRLF
        } else {
          // unwind 1 char
          if (!_sr->seek_from_current(-1)) {
            // this should not happen.
            return false;
          }
        }
      }
      _curr_cursor.col = 0;
      _curr_cursor.row++;
      // continue
    } else {
      // end loop
      if (!_sr->seek_from_current(-1)) {
        return false;
      }
      break;
    }
  }

  return true;
}

bool AsciiParser::SkipCommentAndWhitespaceAndNewline(
    const bool allow_semicolon) {
  // Skip multiple line of comments.
  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    // printf("sws c = %c\n", c);

    if (c == '#') {
      if (!SkipUntilNewline()) {
        return false;
      }
    } else if ((c == ' ') || (c == '\t') || (c == '\f')) {
      _curr_cursor.col++;
      // continue
    } else if (allow_semicolon && (c == ';')) {
      _curr_cursor.col++;
      // continue
    } else if (c == '\n') {
      _curr_cursor.col = 0;
      _curr_cursor.row++;
      // continue
    } else if (c == '\r') {
      // CRLF?
      if (_sr->tell() < (_sr->size() - 1)) {
        char d;
        if (!Char1(&d)) {
          // this should not happen.
          return false;
        }

        if (d == '\n') {
          // CRLF
        } else {
          // unwind 1 char
          if (!_sr->seek_from_current(-1)) {
            // this should not happen.
            return false;
          }
        }
      }
      _curr_cursor.col = 0;
      _curr_cursor.row++;
      // continue
    } else {
      // std::cout << "unwind\n";
      // end loop
      if (!_sr->seek_from_current(-1)) {
        return false;
      }
      break;
    }
  }

  return true;
}

bool AsciiParser::Expect(char expect_c) {
  if (!SkipWhitespace()) {
    return false;
  }

  char c;
  if (!Char1(&c)) {
    // this should not happen.
    return false;
  }

  bool ret = (c == expect_c);

  if (!ret) {
    std::string msg = "Expected `" + std::string(&expect_c, 1) + "` but got `" +
                      std::string(&c, 1) + "`\n";
    PUSH_ERROR_AND_RETURN(msg);

    // unwind
    _sr->seek_from_current(-1);
  } else {
    _curr_cursor.col++;
  }

  return ret;
}

// Parse magic
// #usda FLOAT
bool AsciiParser::ParseMagicHeader() {
  if (!SkipWhitespace()) {
    return false;
  }

  if (Eof()) {
    return false;
  }

  {
    char magic[6];
    if (!_sr->read(6, 6, reinterpret_cast<uint8_t *>(magic))) {
      // eol
      return false;
    }

    if ((magic[0] == '#') && (magic[1] == 'u') && (magic[2] == 's') &&
        (magic[3] == 'd') && (magic[4] == 'a') && (magic[5] == ' ')) {
      // ok
    } else {
      PUSH_ERROR_AND_RETURN(
          "Magic header must start with `#usda `(at least single whitespace "
          "after 'a') but got `" +
          std::string(magic, 6));
    }
  }

  if (!SkipWhitespace()) {
    // eof
    return false;
  }

  // current we only accept "1.0"
  {
    char ver[3];
    if (!_sr->read(3, 3, reinterpret_cast<uint8_t *>(ver))) {
      return false;
    }

    if ((ver[0] == '1') && (ver[1] == '.') && (ver[2] == '0')) {
      // ok
      _version = 1.0f;
    } else {
      PUSH_ERROR_AND_RETURN("Version must be `1.0` but got `" +
                            std::string(ver, 3) + "`");
    }
  }

  SkipUntilNewline();

  return true;
}

bool AsciiParser::ParseCustomMetaValue() {
  // type identifier '=' value

  // return ParseAttributeMeta();
  PUSH_ERROR_AND_RETURN("TODO");
}

bool AsciiParser::ParseAssetIdentifier(value::AssetPath *out,
                                       bool *triple_deliminated) {
  // '..' or "..." are also allowed.
  // @...@
  // or @@@...@@@ (Triple '@'-deliminated asset identifier.)
  // @@@ = Path containing '@'. '@@@' in Path is encoded as '\@@@'
  //
  // Example:
  //   @bora@
  //   @@@bora@@@
  //   @@@bora\@@@dora@@@

  // TODO: Correctly support escape characters

  // look ahead.
  std::vector<char> buf;
  uint64_t curr = _sr->tell();
  bool maybe_triple{false};

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  if (CharN(3, &buf)) {
    if (buf[0] == '@' && buf[1] == '@' && buf[2] == '@') {
      maybe_triple = true;
    }
  }

  bool valid{false};

  if (!maybe_triple) {
    // delimiter = " ' @

    SeekTo(curr);
    char s;
    if (!Char1(&s)) {
      return false;
    }

    char delim = s;

    if ((s == '@') || (s == '\'') || (s == '"')) {
      // ok
    } else {
      std::string sstr{s};
      PUSH_ERROR_AND_RETURN(
          "Asset must start with '@', '\'' or '\"', but got '" + sstr + "'");
    }

    std::string tok;

    // Read until next delimiter
    bool found_delimiter = false;
    while (!Eof()) {
      char c;

      if (!Char1(&c)) {
        return false;
      }

      if (c == delim) {
        found_delimiter = true;
        break;
      }

      tok += c;
    }

    if (found_delimiter) {
      (*out) = tok;
      (*triple_deliminated) = false;

      valid = true;
    }

  } else {
    bool found_delimiter{false};
    bool escape_sequence{false};
    int at_cnt{0};
    std::string tok;

    // Read until '@@@' appears
    // Need to escaped '@@@'("\\@@@")
    while (!Eof()) {
      char c;

      if (!Char1(&c)) {
        return false;
      }

      if (c == '\\') {
        escape_sequence = true;
      }

      if (c == '@') {
        at_cnt++;
      } else {
        at_cnt--;
        if (at_cnt < 0) {
          at_cnt = 0;
        }
      }

      tok += c;

      if (at_cnt == 3) {
        if (escape_sequence) {
          // Still in path identifier...
          // Unescape "\\@@@"

          if (tok.size() > 3) {            // this should be true.
            if (endsWith(tok, "\\@@@")) {  // this also should be true.
              tok.erase(tok.size() - 4);
              tok.append("@@@");
            }
          }
          at_cnt = 0;
          escape_sequence = false;
        } else {
          // Got it. '@@@'
          found_delimiter = true;
          break;
        }
      }
    }

    if (found_delimiter) {
      // remote last '@@@'
      (*out) = removeSuffix(tok, "@@@");
      (*triple_deliminated) = true;

      valid = true;
    }
  }

  return valid;
}

bool AsciiParser::ParseReference(Reference *out, bool *triple_deliminated) {
  /*
    Asset reference = AsssetIdentifier + optially followd by prim path

    AssetIdentifier could be empty(self-reference?)

    Example:
     "bora"
     @bora@
     @bora@</dora>
     </bora>
  */

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  // Parse AssetIdentifier
  {
    char nc;
    if (!LookChar1(&nc)) {
      return false;
    }

    if (nc == '<') {
      // No Asset Identifier.
      out->asset_path = value::AssetPath("");
    } else {
      value::AssetPath ap;
      if (!ParseAssetIdentifier(&ap, triple_deliminated)) {
        PUSH_ERROR_AND_RETURN_TAG(kAscii,
                                  "Failed to parse asset path identifier.");
      }
      out->asset_path = ap;
    }
  }

  // Parse optional prim_path
  if (!SkipWhitespace()) {
    return false;
  }

  {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == '<') {
      if (!Rewind(1)) {
        return false;
      }

      std::string path;
      if (!ReadPathIdentifier(&path)) {
        return false;
      }

      out->prim_path = Path(path, "");
    } else {
      if (!Rewind(1)) {
        return false;
      }
    }
  }

  // TODO: LayerOffset and CustomData

  return true;
}

bool AsciiParser::ParsePayload(Payload *out, bool *triple_deliminated) {
  // Reference, but no customData.

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  // Parse AssetIdentifier
  {
    char nc;
    if (!LookChar1(&nc)) {
      return false;
    }

    if (nc == '<') {
      // No Asset Identifier.
      out->asset_path = value::AssetPath("");
    } else {
      value::AssetPath ap;
      if (!ParseAssetIdentifier(&ap, triple_deliminated)) {
        PUSH_ERROR_AND_RETURN_TAG(kAscii,
                                  "Failed to parse asset path identifier.");
      }
      out->asset_path = ap;
    }
  }

  // Parse optional prim_path
  if (!SkipWhitespace()) {
    return false;
  }

  {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == '<') {
      if (!Rewind(1)) {
        return false;
      }

      std::string path;
      if (!ReadPathIdentifier(&path)) {
        return false;
      }

      out->prim_path = Path(path, "");
    } else {
      if (!Rewind(1)) {
        return false;
      }
    }
  }

  // TODO: LayerOffset

  return true;
}

bool AsciiParser::ParseMetaValue(const VariableDef &def, MetaVariable *outvar) {
  std::string vartype = def.type;
  const std::string varname = def.name;

  MetaVariable var;

  bool array_qual{false};

  DCOUT("parseMeta: vartype " << vartype);

  if (endsWith(vartype, "[]")) {
    vartype = removeSuffix(vartype, "[]");
    array_qual = true;
  } else if (def.allow_array_type) {  // variable can be array
    // Seek '['
    char c;
    if (LookChar1(&c)) {
      if (c == '[') {
        array_qual = true;
      }
    }
  }

  uint32_t tyid = value::GetTypeId(vartype);

#define PARSE_BASE_TYPE(__ty)                                     \
  case value::TypeTraits<__ty>::type_id(): {                      \
    if (array_qual) {                                             \
      std::vector<__ty> vss;                                      \
      if (!ParseBasicTypeArray(&vss)) {                           \
        PUSH_ERROR_AND_RETURN(                                    \
            fmt::format("Failed to parse a value of type `{}[]`", \
                        value::TypeTraits<__ty>::type_name()));   \
      }                                                           \
      var.set_value(vss);                                         \
    } else {                                                      \
      __ty val;                                                   \
      if (!ReadBasicType(&val)) {                                 \
        PUSH_ERROR_AND_RETURN(                                    \
            fmt::format("Failed to parse a value of type `{}`",   \
                        value::TypeTraits<__ty>::type_name()));   \
      }                                                           \
      var.set_value(val);                                         \
    }                                                             \
    break;                                                        \
  }

  // Special treatment for "Reference" and "Payload"
  if (vartype == "Reference") {
    if (array_qual) {
      std::vector<Reference> refs;
      if (!ParseBasicTypeArray(&refs)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Failed to parse `{}` in Prim metadataum.", def.name));
      }
      var.set_value(refs);
    } else {
      nonstd::optional<Reference> ref;
      if (!ReadBasicType(&ref)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Failed to parse `{}` in Prim metadataum.", def.name));
      }
      if (ref) {
        var.set_value(ref.value());
      } else {
        // None
        var.set_value(value::ValueBlock());
      }
    }
  } else if (vartype == "Payload") {
    if (array_qual) {
      std::vector<Payload> refs;
      if (!ParseBasicTypeArray(&refs)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Failed to parse `{}` in Prim metadataum.", def.name));
      }
      var.set_value(refs);
    } else {
      nonstd::optional<Payload> ref;
      if (!ReadBasicType(&ref)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Failed to parse `{}` in Prim metadataum.", def.name));
      }
      if (ref) {
        var.set_value(ref.value());
      } else {
        // None
        var.set_value(value::ValueBlock());
      }
    }
  } else if (vartype == value::kPath) {
    if (array_qual) {
      std::vector<Path> paths;
      if (!ParseBasicTypeArray(&paths)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Failed to parse `{}` in Prim metadatum.", def.name));
      }
      var.set_value(paths);

    } else {
      Path path;
      if (!ReadBasicType(&path)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Failed to parse `{}` in Prim metadatum.", def.name));
      }
      var.set_value(path);
    }
  } else {
    switch (tyid) {
      APPLY_TO_METAVARIABLE_TYPE(PARSE_BASE_TYPE)
      case value::TYPE_ID_STRING: {
        if (array_qual) {
          std::vector<std::string> strs;
          if (!ParseBasicTypeArray(&strs)) {
            PUSH_ERROR_AND_RETURN("Failed to parse `string[]`");
          }
          var.set_value(strs);
        } else {
          std::string str;
          if (!ReadBasicType(&str)) {
            PUSH_ERROR_AND_RETURN("Failed to parse `string`");
          }
          var.set_value(str);
        }
        break;
      }
      case value::TYPE_ID_ASSET_PATH: {
        if (array_qual) {
          std::vector<value::AssetPath> arrs;
          if (!ParseBasicTypeArray(&arrs)) {
            PUSH_ERROR_AND_RETURN("Failed to parse `asset[]`");
          }
          var.set_value(arrs);
        } else {
          value::AssetPath asset;
          if (!ReadBasicType(&asset)) {
            PUSH_ERROR_AND_RETURN("Failed to parse `asset`");
          }
          var.set_value(asset);
        }
        break;
      }
      case value::TYPE_ID_DICT: {
        Dictionary dict;

        DCOUT("Parse dictionary");
        if (!ParseDict(&dict)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `dictionary`");
        }
        var.set_value(dict);
        break;
      }
      default: {
        std::string tyname = vartype;
        if (array_qual) {
          tyname += "[]";
        }
        PUSH_ERROR_AND_RETURN("Unsupported or invalid type for Metadatum:" +
                              tyname);
      }
    }
  }

#undef PARSE_BASE_TYPE

  (*outvar) = var;

  return true;
}

bool AsciiParser::LexFloat(std::string *result) {
  // FLOATVAL : ('+' or '-')? FLOAT
  // FLOAT
  //     :   ('0'..'9')+ '.' ('0'..'9')* EXPONENT?
  //     |   '.' ('0'..'9')+ EXPONENT?
  //     |   ('0'..'9')+ EXPONENT
  //     ;
  // EXPONENT : ('e'|'E') ('+'|'-')? ('0'..'9')+ ;

  std::stringstream ss;

  bool has_sign{false};
  bool leading_decimal_dots{false};
  {
    char sc;
    if (!Char1(&sc)) {
      return false;
    }
    _curr_cursor.col++;

    // sign, '.' or [0-9]
    if ((sc == '+') || (sc == '-')) {
      ss << sc;
      has_sign = true;

      char c;
      if (!Char1(&c)) {
        return false;
      }

      if (c == '.') {
        // ok. something like `+.7`, `-.53`
        leading_decimal_dots = true;
        _curr_cursor.col++;
        ss << c;

      } else {
        // unwind and continue
        _sr->seek_from_current(-1);
      }

    } else if ((sc >= '0') && (sc <= '9')) {
      // ok
      ss << sc;
    } else if (sc == '.') {
      // ok but rescan again in 2.
      leading_decimal_dots = true;
      if (!Rewind(1)) {
        return false;
      }
      _curr_cursor.col--;
    } else {
      PUSH_ERROR_AND_RETURN("Sign or `.` or 0-9 expected.");
    }
  }

  (void)has_sign;

  // 1. Read the integer part
  char curr;
  if (!leading_decimal_dots) {
    // std::cout << "1 read int part: ss = " << ss.str() << "\n";

    while (!Eof()) {
      if (!Char1(&curr)) {
        return false;
      }

      // std::cout << "1 curr = " << curr << "\n";
      if ((curr >= '0') && (curr <= '9')) {
        // continue
        ss << curr;
      } else {
        _sr->seek_from_current(-1);
        break;
      }
    }
  }

  if (Eof()) {
    (*result) = ss.str();
    return true;
  }

  if (!Char1(&curr)) {
    return false;
  }

  // std::cout << "before 2: ss = " << ss.str() << ", curr = " << curr <<
  // "\n";

  // 2. Read the decimal part
  if (curr == '.') {
    ss << curr;

    while (!Eof()) {
      if (!Char1(&curr)) {
        return false;
      }

      if ((curr >= '0') && (curr <= '9')) {
        ss << curr;
      } else {
        break;
      }
    }

  } else if ((curr == 'e') || (curr == 'E')) {
    // go to 3.
  } else {
    // end
    (*result) = ss.str();
    _sr->seek_from_current(-1);
    return true;
  }

  if (Eof()) {
    (*result) = ss.str();
    return true;
  }

  // 3. Read the exponent part
  bool has_exp_sign{false};
  if ((curr == 'e') || (curr == 'E')) {
    ss << curr;

    if (!Char1(&curr)) {
      return false;
    }

    if ((curr == '+') || (curr == '-')) {
      // exp sign
      ss << curr;
      has_exp_sign = true;

    } else if ((curr >= '0') && (curr <= '9')) {
      // ok
      ss << curr;
    } else {
      // Empty E is not allowed.
      PUSH_ERROR_AND_RETURN("Empty `E' is not allowed.");
    }

    while (!Eof()) {
      if (!Char1(&curr)) {
        return false;
      }

      if ((curr >= '0') && (curr <= '9')) {
        // ok
        ss << curr;

      } else if ((curr == '+') || (curr == '-')) {
        if (has_exp_sign) {
          // No multiple sign characters
          PUSH_ERROR_AND_RETURN("No multiple exponential sign characters.");
        }

        ss << curr;
        has_exp_sign = true;
      } else {
        // end
        _sr->seek_from_current(-1);
        break;
      }
    }
  } else {
    _sr->seek_from_current(-1);
  }

  (*result) = ss.str();
  return true;
}

nonstd::optional<AsciiParser::VariableDef> AsciiParser::GetStageMetaDefinition(
    const std::string &name) {
  if (_supported_stage_metas.count(name)) {
    return _supported_stage_metas.at(name);
  }

  return nonstd::nullopt;
}

nonstd::optional<AsciiParser::VariableDef> AsciiParser::GetPrimMetaDefinition(
    const std::string &name) {
  if (_supported_prim_metas.count(name)) {
    return _supported_prim_metas.at(name);
  }

  return nonstd::nullopt;
}

nonstd::optional<AsciiParser::VariableDef> AsciiParser::GetPropMetaDefinition(
    const std::string &name) {
  if (_supported_prop_metas.count(name)) {
    return _supported_prop_metas.at(name);
  }

  return nonstd::nullopt;
}

bool AsciiParser::ParseStageMeta(std::pair<ListEditQual, MetaVariable> *out) {
  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  tinyusdz::ListEditQual qual{ListEditQual::ResetToExplicit};
  if (!MaybeListEditQual(&qual)) {
    return false;
  }

  DCOUT("list-edit qual: " << tinyusdz::to_string(qual));

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  std::string varname;
  if (!ReadIdentifier(&varname)) {
    return false;
  }

  // std::cout << "varname = `" << varname << "`\n";

  if (!IsStageMeta(varname)) {
    PUSH_ERROR_AND_RETURN("Unsupported or invalid/empty variable name `" +
                          varname + "` for Stage metadatum");
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  if (!Expect('=')) {
    PUSH_ERROR_AND_RETURN("`=` expected.");
    return false;
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  auto pvardef = GetStageMetaDefinition(varname);
  if (!pvardef) {
    // This should not happen though;
    return false;
  }

  auto vardef = (*pvardef);

  MetaVariable var;
  if (!ParseMetaValue(vardef, &var)) {
    return false;
  }
  var.set_name(varname);

  std::get<0>(*out) = qual;
  std::get<1>(*out) = var;

  return true;
}

nonstd::optional<std::pair<ListEditQual, MetaVariable>>
AsciiParser::ParsePrimMeta() {
  if (!SkipCommentAndWhitespaceAndNewline()) {
    return nonstd::nullopt;
  }

  tinyusdz::ListEditQual qual{ListEditQual::ResetToExplicit};

  // May be string only(varname is "comment")
  // For some reason, string-only data is just stored in `MetaVariable` and
  // reconstructed in ReconstructPrimMeta in usda-reader.cc later
  //
  {
    value::StringData sdata;
    if (MaybeTripleQuotedString(&sdata)) {
      MetaVariable var;
      // empty name
      var.set_value("comment", sdata);

      return std::make_pair(qual, var);

    } else if (MaybeString(&sdata)) {
      MetaVariable var;
      var.set_value("comment", sdata);

      return std::make_pair(qual, var);
    }
  }

  if (!MaybeListEditQual(&qual)) {
    return nonstd::nullopt;
  }

  DCOUT("list-edit qual: " << tinyusdz::to_string(qual));

  if (!SkipWhitespaceAndNewline()) {
    return nonstd::nullopt;
  }

  std::string varname;
  if (!ReadIdentifier(&varname)) {
    return nonstd::nullopt;
  }

  DCOUT("Identifier = " << varname);

  bool registered_meta = IsRegisteredPrimMeta(varname);

  if (!Expect('=')) {
    PUSH_ERROR("'=' expected in Prim Metadata line.");
    return nonstd::nullopt;
  }
  SkipWhitespace();

  if (!registered_meta) {
    // parse as string until newline

    std::string content;
    if (!ReadUntilNewline(&content)) {
      PUSH_ERROR("Failed to parse unregistered Prim metadata.");
      return nonstd::nullopt;
    }

    MetaVariable var;
    var.set_value(varname, content);

    return std::make_pair(qual, var);
  } else {
    if (auto pv = GetPrimMetaDefinition(varname)) {
      MetaVariable var;
      const auto vardef = pv.value();
      if (!ParseMetaValue(vardef, &var)) {
        PUSH_ERROR("Failed to parse Prim meta value.");
        return nonstd::nullopt;
      }
      var.set_name(varname);

      return std::make_pair(qual, var);
    } else {
      PUSH_ERROR(fmt::format(
          "[Internal error] Unsupported/unimplemented PrimSpec metadata {}",
          varname));
      return nonstd::nullopt;
    }
  }
}

bool AsciiParser::ParsePrimMetas(PrimMetaMap *args) {
  // '(' args ')'
  // args = list of argument, separated by newline.

  if (!Expect('(')) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    // std::cout << "skip comment/whitespace/nl failed\n";
    DCOUT("SkipCommentAndWhitespaceAndNewline failed.");
    return false;
  }

  while (!Eof()) {
    if (!SkipCommentAndWhitespaceAndNewline()) {
      // std::cout << "2: skip comment/whitespace/nl failed\n";
      return false;
    }

    char s;
    if (!Char1(&s)) {
      return false;
    }

    if (s == ')') {
      DCOUT("Prim meta end");
      // End
      break;
    }

    Rewind(1);

    DCOUT("Start PrimMeta parse.");

    // ty = std::pair<ListEditQual, MetaVariable>;
    if (auto m = ParsePrimMeta()) {
      DCOUT("PrimMeta: list-edit qual = "
            << tinyusdz::to_string(std::get<0>(m.value()))
            << ", name = " << std::get<1>(m.value()).get_name());

      if (std::get<1>(m.value()).get_name().empty()) {
        PUSH_ERROR_AND_RETURN("[InternalError] Metadataum name is empty.");
      }

      (*args)[std::get<1>(m.value()).get_name()] = m.value();
    } else {
      PUSH_ERROR_AND_RETURN("Failed to parse Meta value.");
    }
  }

  return true;
}

bool AsciiParser::ParseAttrMeta(AttrMeta *out_meta) {
  // '(' metas ')'
  //
  // currently we only support 'interpolation', 'elementSize' and 'cutomData'

  if (!SkipWhitespace()) {
    return false;
  }

  // The first character.
  {
    char c;
    if (!Char1(&c)) {
      // this should not happen.
      return false;
    }

    if (c == '(') {
      // ok
    } else {
      _sr->seek_from_current(-1);

      // Still ok. No meta
      DCOUT("No attribute meta.");
      return true;
    }
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  while (!Eof()) {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == ')') {
      // end meta
      break;
    } else {
      if (!Rewind(1)) {
        return false;
      }

      // May be string only
      {
        value::StringData sdata;
        if (MaybeTripleQuotedString(&sdata)) {
          out_meta->stringData.push_back(sdata);

          DCOUT("Add triple-quoted string to attr meta:" << to_string(sdata));
          if (!SkipWhitespaceAndNewline()) {
            return false;
          }
          continue;
        } else if (MaybeString(&sdata)) {
          out_meta->stringData.push_back(sdata);

          DCOUT("Add string to attr meta:" << to_string(sdata));
          if (!SkipWhitespaceAndNewline()) {
            return false;
          }
          continue;
        }
      }

      std::string varname;
      if (!ReadIdentifier(&varname)) {
        return false;
      }

      DCOUT("Property/Attribute meta name: " << varname);

      bool supported = _supported_prop_metas.count(varname);
      if (!supported) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Unsupported Property metadatum name: {}", varname));
      }

      {
        std::string name_err;
        if (!pathutil::ValidatePropPath(Path("", varname), &name_err)) {
          PUSH_ERROR_AND_RETURN_TAG(
              kAscii,
              fmt::format("Invalid Property name `{}`: {}", varname, name_err));
        }
      }

      if (!SkipWhitespaceAndNewline()) {
        return false;
      }

      if (!Expect('=')) {
        return false;
      }

      if (!SkipWhitespaceAndNewline()) {
        return false;
      }

      //
      // First-class predefind prop metas.
      //
      if (varname == "interpolation") {
        std::string value;
        if (!ReadStringLiteral(&value)) {
          return false;
        }

        DCOUT("Got `interpolation` meta : " << value);
        out_meta->interpolation = InterpolationFromString(value);
      } else if (varname == "elementSize") {
        uint32_t value;
        if (!ReadBasicType(&value)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `elementSize`");
        }

        DCOUT("Got `elementSize` meta : " << value);
        out_meta->elementSize = value;
      } else if (varname == "colorSpace") {
        value::token tok;
        if (!ReadBasicType(&tok)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `colorSpace`");
        }
        // Add as custom meta value.
        MetaVariable metavar;
        metavar.set_value("colorSpace", tok);
        out_meta->meta["colorSpace"] = metavar;
      } else if (varname == "unauthoredValuesIndex") {
        int value;
        if (!ReadBasicType(&value)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `unauthoredValuesIndex`");
        }

        DCOUT("Got `unauthoredValuesIndex` meta : " << value);
        MetaVariable metavar;
        metavar.set_value("unauthoredValuesIndex", value);
        out_meta->meta["unauthoredValuesIndex"] = metavar;
      } else if (varname == "customData") {
        Dictionary dict;

        if (!ParseDict(&dict)) {
          return false;
        }

        DCOUT("Got `customData` meta");
        out_meta->customData = dict;

      } else if (varname == "weight") {
        double value;
        if (!ReadBasicType(&value)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `weight`");
        }

        DCOUT("Got `weight` meta : " << value);
        out_meta->weight = value;
      } else if (varname == "bindMaterialAs") {
        value::token tok;
        if (!ReadBasicType(&tok)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `bindMaterialAs`");
        }
        if ((tok.str() == kWeaderThanDescendants) ||
            (tok.str() == kStrongerThanDescendants)) {
          // ok
        } else {
          // still valid though
          PUSH_WARN("Unsupported token for bindMaterialAs: " << tok.str());
        }
        DCOUT("bindMaterialAs: " << tok);
        out_meta->bindMaterialAs = tok;
      } else if (varname == "displayName") {
        std::string str;
        if (!ReadStringLiteral(&str)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `displayName`(string type)");
        }
        DCOUT("displayName: " << str);
        out_meta->displayName = str;
      } else if (varname == "displayGroup") {
        std::string str;
        if (!ReadStringLiteral(&str)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `displayGroup`(string type)");
        }
        DCOUT("displayGroup: " << str);
        out_meta->displayGroup = str;

      } else if (varname == "connectability") {
        value::token tok;
        if (!ReadBasicType(&tok)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `connectability`");
        }
        DCOUT("connectability: " << tok);
        out_meta->connectability = tok;
      } else if (varname == "renderType") {
        value::token tok;
        if (!ReadBasicType(&tok)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `renderType`");
        }
        DCOUT("renderType: " << tok);
        out_meta->renderType = tok;
      } else if (varname == "outputName") {
        value::token tok;
        if (!ReadBasicType(&tok)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `outputName`");
        }
        DCOUT("outputName: " << tok);
        out_meta->outputName = tok;
      } else if (varname == "sdrMetadata") {
        Dictionary dict;

        if (!ParseDict(&dict)) {
          return false;
        }

        out_meta->sdrMetadata = dict;
      } else {
        if (auto pv = GetPropMetaDefinition(varname)) {
          // Parse as generic metadata variable
          MetaVariable metavar;
          const auto &vardef = pv.value();

          if (!ParseMetaValue(vardef, &metavar)) {
            return false;
          }
          metavar.set_name(varname);

          // add to custom meta
          out_meta->meta[varname] = metavar;

        } else {
          // This should not happen though.
          PUSH_ERROR_AND_RETURN_TAG(
              kAscii,
              fmt::format(
                  "[InternalErrror] Failed to parse Property metadataum `{}`",
                  varname));
        }
      }

      if (!SkipWhitespaceAndNewline()) {
        return false;
      }
    }
  }

  return true;
}

bool IsUSDA(const std::string &filename, size_t max_filesize) {
  // TODO: Read only first N bytes
  std::vector<uint8_t> data;
  std::string err;

  if (!io::ReadWholeFile(&data, &err, filename, max_filesize)) {
    return false;
  }

  tinyusdz::StreamReader sr(data.data(), data.size(), /* swap endian */ false);
  tinyusdz::ascii::AsciiParser parser(&sr);

  return parser.CheckHeader();
}

//
// -- Impl
//

///
/// Parse `rel`
///
bool AsciiParser::ParseRelationship(Relationship *result) {
  char c;
  if (!LookChar1(&c)) {
    return false;
  }

  if (c == '<') {
    // Path
    Path value;
    if (!ReadBasicType(&value)) {
      PUSH_ERROR_AND_RETURN("Failed to parse Path.");
    }

    // Resolve relative path here.
    // NOTE: Internally, USD(Crate) does not allow relative path.
    Path base_prim_path(GetCurrentPrimPath(), "");
    Path abs_path;
    std::string err;
    if (!pathutil::ResolveRelativePath(base_prim_path, value, &abs_path,
                                       &err)) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Invalid relative Path: {}. error = {}", value, err));
    }

    result->set(abs_path);
  } else if (c == '[') {
    // PathVector
    std::vector<Path> values;
    if (!ParseBasicTypeArray(&values)) {
      PUSH_ERROR_AND_RETURN("Failed to parse PathVector.");
    }

    // Resolve relative path here.
    // NOTE: Internally, USD(Crate) does not allow relative path.
    for (size_t i = 0; i < values.size(); i++) {
      Path base_prim_path(GetCurrentPrimPath(), "");
      Path abs_path;
      if (!pathutil::ResolveRelativePath(base_prim_path, values[i],
                                         &abs_path)) {
        PUSH_ERROR_AND_RETURN(fmt::format("Invalid relative Path: {}.",
                                          values[i].full_path_name()));
      }

      // replace
      values[i] = abs_path;
    }

    result->set(values);
  } else if (c == 'N') {
    // None
    nonstd::optional<Path> value;
    if (!ReadBasicType(&value)) {
      PUSH_ERROR_AND_RETURN("Failed to parse None.");
    }

    // Should be empty for None.
    if (value.has_value()) {
      PUSH_ERROR_AND_RETURN("Failed to parse None.");
    }

    DCOUT("Relationship valueblock.");
    result->set_blocked();
  } else {
    PUSH_ERROR_AND_RETURN("Unexpected char \"" + std::to_string(c) +
                          "\" found. Expects Path or PathVector.");
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  return true;
}

template <typename T>
bool AsciiParser::ParseBasicPrimAttr(bool array_qual,
                                     const std::string &primattr_name,
                                     Attribute *out_attr) {
  Attribute attr;
  primvar::PrimVar var;
  bool blocked{false};

  if (array_qual) {
    if (MaybeNone()) {
    } else {
      std::vector<T> value;
      if (!ParseBasicTypeArray(&value)) {
        PUSH_ERROR_AND_RETURN(fmt::format("Failed to parse Primtive Attribute {} type = {}[]", primattr_name,
                              std::string(value::TypeTraits<T>::type_name())));
      }

      // Empty array allowed.
      DCOUT("Got it: primatrr " << primattr_name << ", ty = " + std::string(value::TypeTraits<T>::type_name()) +
            ", sz = " + std::to_string(value.size()));
      var.set_value(value);
    }

#if 0
  // FIXME: Disable duplicated parsing attribute connection here, since parsing attribute connection will be handled in ParsePrimProps().
  } else if (hasConnect(primattr_name)) {
    std::string value;  // TODO: Use Path
    if (!ReadPathIdentifier(&value)) {
      PUSH_ERROR_AND_RETURN("Failed to parse path identifier.");
    }

    // validate.
    Path connectionPath = pathutil::FromString(value);
    if (!connectionPath.is_valid()) {
      PUSH_ERROR_AND_RETURN(fmt::format("Invalid connectionPath: {}.", value));
    }

    // Resolve relative path here.
    // NOTE: Internally, USD(Crate) does not allow relative path.
    Path base_prim_path(GetCurrentPrimPath(), "");
    Path abs_path;
    if (!pathutil::ResolveRelativePath(base_prim_path, connectionPath,
                                       &abs_path)) {
      PUSH_ERROR_AND_RETURN(fmt::format("Invalid relative Path: {}.", value));
    }

    // TODO: Use Path
    var.set_value(abs_path.full_path_name());

    // Check if attribute metadatum is not authored.
    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!LookChar1(&c)) {
      return false;
    }

    if (c == '(') {
      PUSH_ERROR_AND_RETURN(fmt::format("Attribute connection cannot have attribute metadataum: {}", primattr_name));
    }

#endif
  } else {
    nonstd::optional<T> value;
    if (!ReadBasicType(&value)) {
      PUSH_ERROR_AND_RETURN("Failed to parse " +
                            std::string(value::TypeTraits<T>::type_name()));
    }

    if (value) {
      DCOUT("ParseBasicPrimAttr: " << value::TypeTraits<T>::type_name() << " = "
                                   << (*value));

      var.set_value(value.value());

    } else {
      blocked = true;
      // std::cout << "ParseBasicPrimAttr: " <<
      // value::TypeTraits<T>::type_name()
      //           << " = None\n";
    }
  }

  // optional: attribute meta.
  AttrMeta meta;
  if (!ParseAttrMeta(&meta)) {
    PUSH_ERROR_AND_RETURN("Failed to parse Attribute meta.");
  }
  attr.metas() = meta;

  if (blocked) {
    // There is still have a type for ValueBlock.
    value::ValueBlock noneval;
    attr.set_value(noneval);
    attr.set_blocked(true);
    if (array_qual) {
      attr.set_type_name(value::TypeTraits<T>::type_name() + "[]");
    } else {
      attr.set_type_name(value::TypeTraits<T>::type_name());
    }
  } else {
    attr.set_var(std::move(var));
  }

  (*out_attr) = std::move(attr);

  return true;
}

bool AsciiParser::ParsePrimProps(std::map<std::string, Property> *props,
                                 std::vector<value::token> *propNames) {
  (void)propNames;

  // prim_prop : (custom?) (variability?) type (array_qual?) name '=' value
  //           | (custom?) type (array_qual?) name '=' value interpolation?
  //           | (custom?) (variability?) type (array_qual?) name interpolation?
  //           | (custom?) (listeditqual?) (variability?) rel attr_name = None
  //           | (custom?) (listeditqual?) (variability?) rel attr_name = string
  //           meta | (custom?) (listeditqual?) (variability?) rel attr_name =
  //           path meta | (custom?) (listeditqual?) (variability?) rel
  //           attr_name = pathvector meta | (custom?) (listeditqual?)
  //           (variability?) rel attr_name meta
  //           ;

  // NOTE:
  //  custom append varying ... is not allowed.
  //  append varying custom ... is not allowed.
  //  append custom varying ... is allowed(decomposed into `custom varying ...`
  //  and `append varying ...`

  // Skip comment
  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  // Parse `custom`
  bool custom_qual = MaybeCustom();

  if (!SkipWhitespace()) {
    return false;
  }

  ListEditQual listop_qual;
  if (!MaybeListEditQual(&listop_qual)) {
    return false;
  }

  // `custom` then listop is not allowed.
  if (listop_qual != ListEditQual::ResetToExplicit) {
    if (custom_qual) {
      PUSH_ERROR_AND_RETURN("`custom` then ListEdit qualifier is not allowed.");
    }

    // listop then `custom` is allowed.
    custom_qual = MaybeCustom();
  }

  bool varying_authored{false};
  tinyusdz::Variability variability{tinyusdz::Variability::Varying};

  if (!MaybeVariability(&variability, &varying_authored)) {
    return false;
  }
  DCOUT("variability = " << to_string(variability) << ", varying_authored "
                         << varying_authored);

  std::string type_name;

  if (!ReadIdentifier(&type_name)) {
    return false;
  }

  if (!SkipWhitespace()) {
    return false;
  }

  DCOUT("type_name = " << type_name);

  // `uniform` or `varying`

  // Relation('rel')
  if (type_name == kRel) {
    DCOUT("relation");

    if (variability == Variability::Uniform) {
      PUSH_ERROR_AND_RETURN(
          "Explicit `uniform` variability keyword is not allowed for "
          "Relationship.");
    }

    // - prim_identifier
    // - prim_identifier, '(' metadataum ')'
    // - prim_identifier, '=', (None|string|path|pathvector)
    // NOTE: There should be no 'uniform rel'

    std::string attr_name;

    if (!ReadPrimAttrIdentifier(&attr_name)) {
      PUSH_ERROR_AND_RETURN(
          "Attribute name(Identifier) expected but got non-identifier.");
    }

    if (!SkipWhitespace()) {
      return false;
    }

    char c;
    if (!LookChar1(&c)) {
      return false;
    }

    nonstd::optional<AttrMeta> metap;

    if (c == '(') {
      // FIXME: Implement Relation specific metadatum parser?
      AttrMeta meta;
      if (!ParseAttrMeta(&meta)) {
        PUSH_ERROR_AND_RETURN("Failed to parse metadataum.");
      }

      metap = meta;

      if (!LookChar1(&c)) {
        return false;
      }
    }

    if (c != '=') {
      DCOUT("Relationship with no target: " << attr_name);

      // No targets. Define only.
      Property p;
      p.set_property_type(Property::Type::NoTargetsRelation);
      p.set_listedit_qual(listop_qual);

      if (varying_authored) {
        p.relationship().set_varying_authored();
      }

      if (metap) {
        // TODO: metadataum for Rel
        p.relationship().metas() = metap.value();
      }

      (*props)[attr_name] = p;

      return true;
    }

    // has targets
    if (!Expect('=')) {
      return false;
    }

    if (metap) {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii,
          "Syntax error. Property metadatum must be defined after `=` and "
          "relationship target(s).");
    }

    if (!SkipWhitespaceAndNewline()) {
      return false;
    }

    Relationship rel;
    if (!ParseRelationship(&rel)) {
      PUSH_ERROR_AND_RETURN("Failed to parse `rel` property.");
    }

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    if (!LookChar1(&c)) {
      return false;
    }

    if (c == '(') {
      if (metap) {
        PUSH_ERROR_AND_RETURN_TAG(kAscii, "[InternalError] parser error.");
      }

      AttrMeta meta;

      // FIXME: Implement Relation specific metadatum parser?
      if (!ParseAttrMeta(&meta)) {
        PUSH_ERROR_AND_RETURN("Failed to parse metadataum.");
      }

      metap = meta;
    }

    DCOUT("Relationship with target: " << attr_name);
    Property p(rel, custom_qual);
    p.set_listedit_qual(listop_qual);

    if (varying_authored) {
      p.relationship().set_varying_authored();
    }

    if (metap) {
      p.relationship().metas() = metap.value();
    }

    (*props)[attr_name] = p;

    return true;
  }

  //
  // Attrib.
  //
  
  // Attribute cannot have 'varying' keyword
  if (varying_authored) {
    PUSH_ERROR_AND_RETURN_TAG(
        kAscii, "Syntax error. `varying` keyword is not allowed for Attribute.");
  }

  if (listop_qual != ListEditQual::ResetToExplicit) {
    PUSH_ERROR_AND_RETURN_TAG(
        kAscii, "List editing qualifier is not allowed for Attribute.");
  }

  if (!IsSupportedPrimAttrType(type_name)) {
    PUSH_ERROR_AND_RETURN("Unknown or unsupported primtive attribute type `" +
                          type_name);
  }

  // Has array qualifier? `[]`
  bool array_qual = false;
  {
    char c0, c1;
    if (!Char1(&c0)) {
      return false;
    }

    if (c0 == '[') {
      if (!Char1(&c1)) {
        return false;
      }

      if (c1 == ']') {
        array_qual = true;
      } else {
        // Invalid syntax
        PUSH_ERROR_AND_RETURN("Invalid syntax found.");
      }

    } else {
      if (!Rewind(1)) {
        return false;
      }
    }
  }

  if (!SkipWhitespace()) {
    return false;
  }

  std::string primattr_name;
  if (!ReadPrimAttrIdentifier(&primattr_name)) {
    PUSH_ERROR_AND_RETURN("Failed to parse primAttr identifier.");
  }

  if (!SkipWhitespace()) {
    return false;
  }

  bool isTimeSample = endsWith(primattr_name, kTimeSamplesSuffix);
  bool isConnection = endsWith(primattr_name, kConnectSuffix);

  // Remove suffix
  std::string attr_name = primattr_name;
  if (isTimeSample) {
    attr_name = removeSuffix(primattr_name, kTimeSamplesSuffix);
  }
  if (isConnection) {
    attr_name = removeSuffix(primattr_name, kConnectSuffix);
  }

  bool define_only = false;
  {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c != '=') {
      // Define only(e.g. output variable)
      define_only = true;
    }
  }

  DCOUT("define only:" << define_only);

  if (define_only) {
    Rewind(1);

    // optional: attribute meta.
    AttrMeta meta;
    if (!ParseAttrMeta(&meta)) {
      PUSH_ERROR_AND_RETURN("Failed to parse Attribute meta.");
    }

    DCOUT("Define only property = " + primattr_name);

    // Empty Attribute. type info only
    Property p;
    p.set_property_type(Property::Type::EmptyAttrib);
    p.set_custom(custom_qual);
    std::string typeName = type_name;
    if (array_qual) {
      typeName += "[]";
    }
    p.attribute().set_type_name(typeName);

    p.attribute().variability() = variability;
    if (varying_authored) {
      p.attribute().set_varying_authored();
    }

    p.attribute().metas() = meta;

    (*props)[attr_name] = p;

    return true;
  }

  // Continue to parse argument
  if (!SkipWhitespace()) {
    return false;
  }

  bool value_blocked{false};

  if (MaybeNone()) {
    value_blocked = true;
  }

  if (isConnection) {
    // atribute connection
    DCOUT("isConnection");

    Path path;
    if (!value_blocked) {
      // Target Must be Path
      if (!ReadBasicType(&path)) {
        PUSH_ERROR_AND_RETURN("Path expected for .connect target.");
      }
    }

    // Resolve relative path.
    Path base_abs_path(GetCurrentPrimPath(), "");
    Path abs_path;
    std::string err;
    if (!pathutil::ResolveRelativePath(base_abs_path, path, &abs_path, &err)) {
      PUSH_ERROR_AND_RETURN(fmt::format("Invalid relative Path: {}. error = {}",
                                        path.full_path_name(), err));
    }

    // Check if attribute metadatum is not authored.
    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!LookChar1(&c)) {
      return false;
    }

    if (c == '(') {
      PUSH_ERROR_AND_RETURN(fmt::format("Attribute connection cannot have attribute metadataum: {}", attr_name));
    }

    bool attr_exists = props->count(attr_name) && props->at(attr_name).is_attribute();
    if (attr_exists) {

      // TODO: Check if type is the same.

      // Check if variability is the same
      if (props->at(attr_name).attribute().variability() != variability) {
        PUSH_ERROR_AND_RETURN(fmt::format("Variability mismatch. Attribute `{}` already has variability `{}`, but timeSampled value has variability `{}`.", attr_name, to_string(props->at(attr_name).attribute().variability()), to_string(variability)));
      }

      props->at(attr_name).attribute().set_connection(abs_path);

      // Set PropType to Attrib(since previously created Property may have EmptyAttrib).
      props->at(attr_name).set_property_type(Property::Type::Attrib);
    } else {

      Attribute attr;
      attr.set_type_name(type_name);
      attr.set_connection(abs_path);
      attr.variability() = variability;

      //Property p(abs_path, /* value typename */ type_name, custom_qual);

      //p.attribute().variability() = variability;
      //if (varying_authored) {
      //  p.attribute().set_varying_authored();
      //}

      Property p(std::move(attr), custom_qual);
      (*props)[attr_name] = p;
    }

    DCOUT(fmt::format("Added attribute connection to `{}`", attr_name));

    return true;

  } else if (isTimeSample) {
    // float.timeSamples = None is syntax error.
    if (value_blocked) {
      PUSH_ERROR_AND_RETURN(fmt::format("Syntax error. ValueBlock to .timeSamples is invalid: {}", attr_name));
    }

    //
    // TODO(syoyo): Refactror and implement value parser dispatcher.
    //
    if (array_qual) {
      DCOUT("timeSample data. type = " << type_name << "[]");
    } else {
      DCOUT("timeSample data. type = " << type_name);
    }

    value::TimeSamples ts;
    if (array_qual) {
      if (!ParseTimeSamplesOfArray(type_name, &ts)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Failed to parse TimeSamples of type {}[]", type_name));
      }
    } else {
      if (!ParseTimeSamples(type_name, &ts)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kAscii,
            fmt::format("Failed to parse TimeSamples of type {}", type_name));
      }
    }

    // Attribute metadatum is not allowed for timeSamples.
    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!LookChar1(&c)) {
      return false;
    }

    if (c == '(') {
      PUSH_ERROR_AND_RETURN(fmt::format("TimeSampled Attribute cannot have attribute metadataum: {}", attr_name));
    }

    DCOUT("timeSamples primattr: type = " << type_name
                                          << ", name = " << attr_name);

    Attribute attr;
    Attribute *pattr{nullptr};
    bool attr_exists = props->count(attr_name) && props->at(attr_name).is_attribute();
    if (attr_exists) {
      DCOUT("Attr exists");
      // Add timeSamples to existing Attribute
      pattr = &(props->at(attr_name).attribute());

      // Check if variability is the same
      if (pattr->variability() != variability) {
        PUSH_ERROR_AND_RETURN(fmt::format("Variability mismatch. Attribute `{}` already has variability `{}`, but timeSampled value has variability `{}`.", attr_name, to_string(pattr->variability()), to_string(variability)));
      }

      pattr->get_var().set_timesamples(ts);

      // Set PropType to Attrib(since previously created Property may have EmptyAttrib).
      props->at(attr_name).set_property_type(Property::Type::Attrib);

    } else {
      // new Attribute
      pattr = &attr;  

      primvar::PrimVar var;
      var.set_timesamples(ts);
      if (array_qual) {
        pattr->set_type_name(type_name + "[]");
      } else {
        pattr->set_type_name(type_name);
      }
      pattr->set_var(std::move(var));
      pattr->variability() = variability;

      //if (varying_authored) {
      //  pattr->set_varying_authored();
      //}

      pattr->name() = attr_name;

      Property p(attr, custom_qual);
      p.set_property_type(Property::Type::Attrib);
      (*props)[attr_name] = p;
    }

    return true;

  } else {

    Attribute _attr;
    Attribute *pattr{nullptr};
    bool attr_exists = props->count(attr_name) && props->at(attr_name).is_attribute();
    DCOUT("attr_exists " << attr_exists);
    if (attr_exists) {
      pattr = &(props->at(attr_name).attribute());
    } else {
      pattr = &_attr;
      pattr->set_name(primattr_name);
    }

    if (!value_blocked) {
      // TODO: Refactor. ParseAttrMeta is currently called inside
      // ParseBasicPrimAttr()
      if (type_name == value::kBool) {
        if (!ParseBasicPrimAttr<bool>(array_qual, primattr_name, pattr)) {
          return false;
        }
      } else if (type_name == value::kInt) {
        if (!ParseBasicPrimAttr<int>(array_qual, primattr_name, pattr)) {
          return false;
        }
      } else if (type_name == value::kInt2) {
        if (!ParseBasicPrimAttr<value::int2>(array_qual, primattr_name,
                                             pattr)) {
          return false;
        }
      } else if (type_name == value::kInt3) {
        if (!ParseBasicPrimAttr<value::int3>(array_qual, primattr_name,
                                             pattr)) {
          return false;
        }
      } else if (type_name == value::kInt4) {
        if (!ParseBasicPrimAttr<value::int4>(array_qual, primattr_name,
                                             pattr)) {
          return false;
        }
      } else if (type_name == value::kUInt) {
        if (!ParseBasicPrimAttr<uint32_t>(array_qual, primattr_name, pattr)) {
          return false;
        }
      } else if (type_name == value::kUInt2) {
        if (!ParseBasicPrimAttr<value::uint2>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kUInt3) {
        if (!ParseBasicPrimAttr<value::uint3>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kUInt4) {
        if (!ParseBasicPrimAttr<value::uint4>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kInt64) {
        if (!ParseBasicPrimAttr<int64_t>(array_qual, primattr_name, pattr)) {
          return false;
        }
      } else if (type_name == value::kUInt64) {
        if (!ParseBasicPrimAttr<uint64_t>(array_qual, primattr_name, pattr)) {
          return false;
        }
      } else if (type_name == value::kDouble) {
        if (!ParseBasicPrimAttr<double>(array_qual, primattr_name, pattr)) {
          return false;
        }
      } else if (type_name == value::kString) {
        if (!ParseBasicPrimAttr<std::string>(array_qual, primattr_name,
                                                   pattr)) {
          return false;
        }
      } else if (type_name == value::kToken) {
        if (!ParseBasicPrimAttr<value::token>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kHalf) {
        if (!ParseBasicPrimAttr<value::half>(array_qual, primattr_name,
                                             pattr)) {
          return false;
        }
      } else if (type_name == value::kHalf2) {
        if (!ParseBasicPrimAttr<value::half2>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kHalf3) {
        if (!ParseBasicPrimAttr<value::half3>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kHalf4) {
        if (!ParseBasicPrimAttr<value::half4>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kFloat) {
        if (!ParseBasicPrimAttr<float>(array_qual, primattr_name, pattr)) {
          return false;
        }
      } else if (type_name == value::kFloat2) {
        if (!ParseBasicPrimAttr<value::float2>(array_qual, primattr_name,
                                               pattr)) {
          return false;
        }
      } else if (type_name == value::kFloat3) {
        if (!ParseBasicPrimAttr<value::float3>(array_qual, primattr_name,
                                               pattr)) {
          return false;
        }
      } else if (type_name == value::kFloat4) {
        if (!ParseBasicPrimAttr<value::float4>(array_qual, primattr_name,
                                               pattr)) {
          return false;
        }
      } else if (type_name == value::kDouble2) {
        if (!ParseBasicPrimAttr<value::double2>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kDouble3) {
        if (!ParseBasicPrimAttr<value::double3>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kDouble4) {
        if (!ParseBasicPrimAttr<value::double4>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kQuath) {
        if (!ParseBasicPrimAttr<value::quath>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kQuatf) {
        if (!ParseBasicPrimAttr<value::quatf>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kQuatd) {
        if (!ParseBasicPrimAttr<value::quatd>(array_qual, primattr_name,
                                              pattr)) {
          return false;
        }
      } else if (type_name == value::kPoint3f) {
        if (!ParseBasicPrimAttr<value::point3f>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kColor3f) {
        if (!ParseBasicPrimAttr<value::color3f>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kColor4f) {
        if (!ParseBasicPrimAttr<value::color4f>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kPoint3d) {
        if (!ParseBasicPrimAttr<value::point3d>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kNormal3f) {
        if (!ParseBasicPrimAttr<value::normal3f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kNormal3d) {
        if (!ParseBasicPrimAttr<value::normal3d>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kVector3f) {
        if (!ParseBasicPrimAttr<value::vector3f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kVector3d) {
        if (!ParseBasicPrimAttr<value::vector3d>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kColor3d) {
        if (!ParseBasicPrimAttr<value::color3d>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kColor4d) {
        if (!ParseBasicPrimAttr<value::color4d>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix2f) {
        if (!ParseBasicPrimAttr<value::matrix2f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix3f) {
        if (!ParseBasicPrimAttr<value::matrix3f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix4f) {
        if (!ParseBasicPrimAttr<value::matrix4f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix2d) {
        if (!ParseBasicPrimAttr<value::matrix2d>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kFloat3) {
        if (!ParseBasicPrimAttr<value::float3>(array_qual, primattr_name,
                                               pattr)) {
          return false;
        }
      } else if (type_name == value::kFloat4) {
        if (!ParseBasicPrimAttr<value::float4>(array_qual, primattr_name,
                                               pattr)) {
          return false;
        }
      } else if (type_name == value::kDouble2) {
        if (!ParseBasicPrimAttr<value::double2>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kDouble3) {
        if (!ParseBasicPrimAttr<value::double3>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kDouble4) {
        if (!ParseBasicPrimAttr<value::double4>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kPoint3f) {
        if (!ParseBasicPrimAttr<value::point3f>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kColor3f) {
        if (!ParseBasicPrimAttr<value::color3f>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kColor4f) {
        if (!ParseBasicPrimAttr<value::color4f>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kPoint3d) {
        if (!ParseBasicPrimAttr<value::point3d>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kNormal3f) {
        if (!ParseBasicPrimAttr<value::normal3f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kNormal3d) {
        if (!ParseBasicPrimAttr<value::normal3d>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kVector3f) {
        if (!ParseBasicPrimAttr<value::vector3f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kVector3d) {
        if (!ParseBasicPrimAttr<value::vector3d>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kColor3d) {
        if (!ParseBasicPrimAttr<value::color3d>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kColor4d) {
        if (!ParseBasicPrimAttr<value::color4d>(array_qual, primattr_name,
                                                pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix2f) {
        if (!ParseBasicPrimAttr<value::matrix2f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix3f) {
        if (!ParseBasicPrimAttr<value::matrix3f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix4f) {
        if (!ParseBasicPrimAttr<value::matrix4f>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }

      } else if (type_name == value::kMatrix2d) {
        if (!ParseBasicPrimAttr<value::matrix2d>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix3d) {
        if (!ParseBasicPrimAttr<value::matrix3d>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }
      } else if (type_name == value::kMatrix4d) {
        if (!ParseBasicPrimAttr<value::matrix4d>(array_qual, primattr_name,
                                                 pattr)) {
          return false;
        }

      } else if (type_name == value::kTexCoord2f) {
        if (!ParseBasicPrimAttr<value::texcoord2f>(array_qual, primattr_name,
                                                   pattr)) {
          return false;
        }

      } else if (type_name == value::kAssetPath) {
        if (!ParseBasicPrimAttr<value::AssetPath>(array_qual, primattr_name,
                                                  pattr)) {
          return false;
        }
      } else {
        PUSH_ERROR_AND_RETURN("TODO: type = " + type_name);
      }
    }


    if (varying_authored) {
      pattr->set_varying_authored();
    }

    // TODO: Check if type is the same with existing attribute.
    if (value_blocked) {
      if (array_qual) {
        pattr->set_type_name(type_name + "[]");
      } else {
        pattr->set_type_name(type_name);
      }
      pattr->set_blocked(true);
    }

    DCOUT("primattr: type = " << type_name << ", name = " << primattr_name);
    DCOUT(" value_blocked " << value_blocked);

    if (attr_exists) {
      // Check if variability is the same
      if (pattr->variability() != variability) {
        PUSH_ERROR_AND_RETURN(fmt::format("Variability mismatch. Attribute `{}` already has variability `{}`, but 'default' value has variability `{}`.", attr_name, to_string(pattr->variability()), to_string(variability)));
      }

      // Set PropType to Attrib(since previously created Property may have EmptyAttrib).
      props->at(attr_name).set_property_type(Property::Type::Attrib);
    } else {
      pattr->variability() = variability;
      Property p(*pattr, custom_qual);

      (*props)[primattr_name] = p;
    }

    return true;
  }
}

// propNames stores list of property name in its appearance order.
bool AsciiParser::ParseProperties(std::map<std::string, Property> *props,
                                  std::vector<value::token> *propNames) {
  // property : primm_attr
  //          | 'rel' name '=' path
  //          ;

  if (!SkipWhitespace()) {
    return false;
  }

  // rel?
  {
    uint64_t loc = CurrLoc();
    std::string tok;

    if (!ReadIdentifier(&tok)) {
      return false;
    }

    if (tok == "rel") {
      PUSH_ERROR_AND_RETURN("TODO: Parse rel");
    } else {
      SeekTo(loc);
    }
  }

  // attribute
  return ParsePrimProps(props, propNames);
}

std::string AsciiParser::GetCurrentPrimPath() {
  if (_path_stack.empty()) {
    return "/";
  }

  return _path_stack.top();
}

//
// -- ctor, dtor
//

AsciiParser::AsciiParser() { Setup(); }

AsciiParser::AsciiParser(StreamReader *sr) : _sr(sr) { Setup(); }

void AsciiParser::Setup() {
  RegisterStageMetas(_supported_stage_metas);
  RegisterPrimMetas(_supported_prim_metas);
  RegisterPropMetas(_supported_prop_metas);
  RegisterPrimAttrTypes(_supported_prim_attr_types);
  RegisterPrimTypes(_supported_prim_types);
  RegisterAPISchemas(_supported_api_schemas);
}

AsciiParser::~AsciiParser() {}

bool AsciiParser::CheckHeader() { return ParseMagicHeader(); }

bool AsciiParser::IsRegisteredPrimMeta(const std::string &name) {
  return _supported_prim_metas.count(name) ? true : false;
}

bool AsciiParser::IsStageMeta(const std::string &name) {
  return _supported_stage_metas.count(name) ? true : false;
}

bool AsciiParser::ParseVariantSet(
    const int64_t primIdx, const int64_t parentPrimIdx, const uint32_t depth,
    std::map<std::string, VariantContent> *variantSetOut) {
  if (!variantSetOut) {
    PUSH_ERROR_AND_RETURN_TAG(kAscii,
                              "[InternalError] variantSetOut arg is nullptr.");
  }

  // variantSet =
  // {
  //   "variantName0" ( metas ) { ... }
  //   "variantName1" ( metas ) { ... }
  //   ...
  // }
  if (!Expect('{')) {
    return false;
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  std::map<std::string, VariantContent> variantContentMap;

  // for each variantStatement
  while (!Eof()) {
    {
      char c;
      if (!Char1(&c)) {
        return false;
      }

      if (c == '}') {
        // end
        break;
      }

      Rewind(1);
    }

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    // string
    std::string variantName;
    if (!ReadBasicType(&variantName)) {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii, "Failed to parse variant name for `variantSet` statement.");
    }

    if (!SkipWhitespace()) {
      return false;
    }

    // Optional: PrimSpec meta
    PrimMetaMap metas;
    {
      char mc;
      if (!LookChar1(&mc)) {
        return false;
      }

      if (mc == '(') {
        if (!ParsePrimMetas(&metas)) {
          PUSH_ERROR_AND_RETURN_TAG(
              kAscii, "Failed to parse PrimSpec metas in variant statement.");
        }
      }
    }

    if (!Expect('{')) {
      return false;
    }

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    VariantContent variantContent;

    while (!Eof()) {
      {
        char c;
        if (!Char1(&c)) {
          return false;
        }

        if (c == '}') {
          DCOUT("End block in variantSet stmt.");
          // end block
          break;
        }
      }

      if (!Rewind(1)) {
        return false;
      }

      DCOUT("Read first token in VariantSet stmt");
      Identifier tok;
      if (!ReadBasicType(&tok)) {
        PUSH_ERROR_AND_RETURN(
            "Failed to parse an identifier in variantSet block statement.");
      }

      if (!Rewind(tok.size())) {
        return false;
      }

      if (tok == "variantSet") {
        PUSH_ERROR_AND_RETURN("Nested `variantSet` is not supported yet.");
      }

      Specifier child_spec{Specifier::Invalid};
      if (tok == "def") {
        child_spec = Specifier::Def;
      } else if (tok == "over") {
        child_spec = Specifier::Over;
      } else if (tok == "class") {
        child_spec = Specifier::Class;
      }

      // No specifier => Assume properties only.
      // Has specifier => Prim
      if (child_spec != Specifier::Invalid) {
        // FIXME: Assign idx dedicated for variant.
        int64_t idx = _prim_idx_assign_fun(parentPrimIdx);
        DCOUT("enter parseBlock in variantSet. spec = "
              << to_string(child_spec) << ", idx = " << idx
              << ", rootIdx = " << primIdx);

        // recusive call
        if (!ParseBlock(child_spec, idx, primIdx, depth + 1,
                        /* in_variantStmt */ true)) {
          PUSH_ERROR_AND_RETURN(
              fmt::format("`{}` block parse failed.", to_string(child_spec)));
        }
        DCOUT(fmt::format("Done parse `{}` block.", to_string(child_spec)));

        DCOUT(fmt::format("Add primIdx {} to variant {}", idx, variantName));
        variantContent.primIndices.push_back(idx);

      } else {
        DCOUT("Enter ParsePrimProps.");
        if (!ParsePrimProps(&variantContent.props,
                            &variantContent.properties)) {
          PUSH_ERROR_AND_RETURN("Failed to parse Prim attribute.");
        }
        DCOUT(fmt::format("Done parse ParsePrimProps."));
      }

      if (!SkipCommentAndWhitespaceAndNewline()) {
        return false;
      }
    }

    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    DCOUT(fmt::format("variantSet item {} parsed.", variantName));

    variantContent.metas = metas;
    variantContentMap.emplace(variantName, variantContent);
  }

  (*variantSetOut) = std::move(variantContentMap);

  return true;
}

///
/// Parse block.
///
/// block = spec prim_type? token metas? { ... }
/// metas = '(' args ')'
///
/// spec = `def`, `over` or `class`
///
///
bool AsciiParser::ParseBlock(const Specifier spec, const int64_t primIdx,
                             const int64_t parentPrimIdx, const uint32_t depth,
                             const bool in_variantStaement) {
  (void)in_variantStaement;

  DCOUT("ParseBlock");

  if (!SkipCommentAndWhitespaceAndNewline()) {
    DCOUT("SkipCommentAndWhitespaceAndNewline failed");
    return false;
  }

  Identifier def;
  if (!ReadIdentifier(&def)) {
    DCOUT("ReadIdentifier failed");
    return false;
  }
  DCOUT("spec = " << def);

  if ((def == "def") || (def == "over") || (def == "class")) {
    // ok
  } else {
    PUSH_ERROR_AND_RETURN("Invalid specifier.");
  }

  // Ensure spec and def is same.
  if (def == "def") {
    if (spec != Specifier::Def) {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii, "Internal error. Invalid Specifier token combination. def = "
                      << def << ", spec = " << to_string(spec));
    }
  } else if (def == "over") {
    if (spec != Specifier::Over) {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii, "Internal error. Invalid Specifier token combination. def = "
                      << def << ", spec = " << to_string(spec));
    }
  } else if (def == "class") {
    if (spec != Specifier::Class) {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii, "Internal error. Invalid Specifier token combination. def = "
                      << def << ", spec = " << to_string(spec));
    }
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  // look ahead
  bool has_primtype = false;
  {
    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (!Rewind(1)) {
      return false;
    }

    if (c == '"') {
      // token
      has_primtype = false;
    } else {
      has_primtype = true;
    }
  }

  Identifier prim_type;

  DCOUT("has_primtype = " << has_primtype);

  if (has_primtype) {
    if (!ReadIdentifier(&prim_type)) {
      return false;
    }
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  std::string prim_name;
  if (!ReadBasicType(&prim_name)) {
    return false;
  }

  DCOUT("prim name = " << prim_name);
  if (!ValidatePrimElementName(prim_name)) {
    PUSH_ERROR_AND_RETURN_TAG(kAscii, "Prim name contains invalid chacracter.");
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  std::map<std::string, std::pair<ListEditQual, MetaVariable>> in_metas;
  {
    // look ahead
    char c;
    if (!LookChar1(&c)) {
      return false;
    }

    if (c == '(') {
      // meta

      if (!ParsePrimMetas(&in_metas)) {
        DCOUT("Parse Prim metas failed.");
        return false;
      }

      if (!SkipWhitespaceAndNewline()) {
        return false;
      }
    }
  }

  if (!SkipCommentAndWhitespaceAndNewline()) {
    return false;
  }

  if (!Expect('{')) {
    return false;
  }

  if (!SkipWhitespaceAndNewline()) {
    return false;
  }

  std::map<std::string, Property> props;
  std::vector<value::token> propNames;
  VariantSetList variantSetList;

  {
    std::string full_path = GetCurrentPrimPath();
    if (full_path == "/") {
      full_path += prim_name;
    } else {
      full_path += "/" + prim_name;
    }
    PushPrimPath(full_path);
  }

  // expect = '}'
  //        | def_block
  //        | prim_attr+
  //        | variantSet '{' ... '}'
  while (!Eof()) {
    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    char c;
    if (!Char1(&c)) {
      return false;
    }

    if (c == '}') {
      // end block
      break;
    } else {
      if (!Rewind(1)) {
        return false;
      }

      DCOUT("Read stmt token");
      Identifier tok;
      if (!ReadBasicType(&tok)) {
        // maybe ';'?

        if (LookChar1(&c)) {
          if (c == ';') {
            PUSH_ERROR_AND_RETURN(
                "Semicolon is not allowd in `def` block statement.");
          }
        }
        PUSH_ERROR_AND_RETURN(
            "Failed to parse an identifier in `def` block statement.");
      }

      if (tok == "variantSet") {
        if (!SkipWhitespace()) {
          return false;
        }

        std::string variantName;
        if (!ReadBasicType(&variantName)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `variantSet` statement.");
        }

        DCOUT("variantName = " << variantName);

        if (!SkipWhitespace()) {
          return false;
        }

        if (!Expect('=')) {
          return false;
        }

        if (!SkipWhitespace()) {
          return false;
        }

        std::map<std::string, VariantContent> vmap;
        if (!ParseVariantSet(primIdx, parentPrimIdx, depth, &vmap)) {
          PUSH_ERROR_AND_RETURN("Failed to parse `variantSet` statement.");
        }

        variantSetList.emplace(variantName, vmap);

        continue;
      }

      if (!Rewind(tok.size())) {
        return false;
      }

      Specifier child_spec{Specifier::Invalid};
      if (tok == "def") {
        child_spec = Specifier::Def;
      } else if (tok == "over") {
        child_spec = Specifier::Over;
      } else if (tok == "class") {
        child_spec = Specifier::Class;
      }

      if (child_spec != Specifier::Invalid) {
        int64_t idx = _prim_idx_assign_fun(parentPrimIdx);
        DCOUT("enter parseDef. spec = " << to_string(child_spec) << ", idx = "
                                        << idx << ", rootIdx = " << primIdx);

        // recusive call
        if (!ParseBlock(child_spec, idx, primIdx, depth + 1)) {
          PUSH_ERROR_AND_RETURN(
              fmt::format("`{}` block parse failed.", to_string(child_spec)));
        }
        DCOUT(fmt::format("Done parse `{}` block.", to_string(child_spec)));
      } else {
        DCOUT("Enter ParsePrimProps.");
        // Assume PrimAttr
        if (!ParsePrimProps(&props, &propNames)) {
          PUSH_ERROR_AND_RETURN("Failed to parse Prim attribute.");
        }
      }

      if (!SkipWhitespaceAndNewline()) {
        return false;
      }
    }
  }

  std::string pTy = prim_type;

  if (_primspec_mode) {
    // Load scene as PrimSpec tree
    if (_primspec_fun) {
      Path fullpath(GetCurrentPrimPath(), "");
      Path pname(prim_name, "");

      // pass prim_type as is(empty = empty string)
      nonstd::expected<bool, std::string> ret =
          _primspec_fun(fullpath, spec, prim_type, pname, primIdx,
                        parentPrimIdx, props, in_metas, variantSetList);

      if (!ret) {
        // construction failed.
        PUSH_ERROR_AND_RETURN(fmt::format(
            "Constructing PrimSpec typeName `{}`, elementName `{}` failed: {}",
            prim_type, prim_name, ret.error()));
      }
    } else {
      PUSH_ERROR_AND_RETURN_TAG(
          kAscii, "[Internal Error] PrimSpec handler is not found.");
    }

  } else {
    // Create typed Prim.

    if (prim_type.empty()) {
      // No Prim type specified. Treat it as Model

      pTy = "Model";
    }

    if (!_prim_construct_fun_map.count(pTy)) {
      if (_option.allow_unknown_prim) {
        // Unknown Prim type specified. Treat it as Model
        // Prim's type name will be storead in Model::prim_type_name
        pTy = "Model";
      }
    }

    if (_prim_construct_fun_map.count(pTy)) {
      auto construct_fun = _prim_construct_fun_map[pTy];

      Path fullpath(GetCurrentPrimPath(), "");
      Path pname(prim_name, "");
      nonstd::expected<bool, std::string> ret =
          construct_fun(fullpath, spec, prim_type, pname, primIdx,
                        parentPrimIdx, props, in_metas, variantSetList);

      if (!ret) {
        // construction failed.
        PUSH_ERROR_AND_RETURN("Constructing Prim type `" + pTy +
                              "` failed: " + ret.error());
      }

    } else {
      PUSH_WARN(fmt::format(
          "TODO: Unsupported/Unimplemented Prim type: `{}`. Skipping parsing.",
          pTy));
    }
  }

  PopPrimPath();

  return true;
}

///
/// Parser entry point
/// TODO: Refactor and use unified code path regardless of LoadState.
///
bool AsciiParser::Parse(const uint32_t load_states,
                        const AsciiParserOption &parser_option) {
  _toplevel = (load_states & static_cast<uint32_t>(LoadState::Toplevel));
  _sub_layered = (load_states & static_cast<uint32_t>(LoadState::Sublayer));
  _referenced = (load_states & static_cast<uint32_t>(LoadState::Reference));
  _payloaded = (load_states & static_cast<uint32_t>(LoadState::Payload));
  _option = parser_option;

  bool header_ok = ParseMagicHeader();
  if (!header_ok) {
    PUSH_ERROR_AND_RETURN("Failed to parse USDA magic header.\n");
  }

  SkipCommentAndWhitespaceAndNewline();

  if (Eof()) {
    // Empty USDA
    return true;
  }

  {
    char c;
    if (!LookChar1(&c)) {
      return false;
    }

    if (c == '(') {
      // stage meta.
      if (!ParseStageMetas()) {
        PUSH_ERROR_AND_RETURN("Failed to parse Stage metas.");
      }
    }
  }

  if (_stage_meta_process_fun) {
    DCOUT("Invoke StageMeta callback.");
    bool ret = _stage_meta_process_fun(_stage_metas);
    if (!ret) {
      PUSH_ERROR_AND_RETURN("Failed to reconstruct Stage metas.");
    }
  } else {
    // TODO: Report error when StageMeta callback is not set?
    PUSH_WARN("Stage metadata processing callback is not set.");
  }

  PushPrimPath("/");

  // parse blocks
  while (!Eof()) {
    if (!SkipCommentAndWhitespaceAndNewline()) {
      return false;
    }

    if (Eof()) {
      // Whitespaces in the end of line.
      break;
    }

    // Look ahead token
    auto curr_loc = _sr->tell();

    Identifier tok;
    if (!ReadBasicType(&tok)) {
      PUSH_ERROR_AND_RETURN("Identifier expected.\n");
    }

    // Rewind
    if (!SeekTo(curr_loc)) {
      return false;
    }

    Specifier spec{Specifier::Invalid};
    if (tok == "def") {
      spec = Specifier::Def;
    } else if (tok == "over") {
      spec = Specifier::Over;
    } else if (tok == "class") {
      spec = Specifier::Class;
    } else {
      PUSH_ERROR_AND_RETURN("Invalid specifier token '" + tok + "'");
    }

    int64_t primIdx = _prim_idx_assign_fun(-1);
    DCOUT("Enter parseDef. primIdx = " << primIdx
                                       << ", parentPrimIdx = root(-1)");
    bool block_ok = ParseBlock(spec, primIdx, /* parent */ -1, /* depth */ 0,
                               /* in_variantStmt */ false);
    if (!block_ok) {
      PUSH_ERROR_AND_RETURN("Failed to parse `def` block.");
    }
  }

  return true;
}

bool ParseUnregistredValue(const std::string &_typeName, const std::string &str,
                           value::Value *value, std::string *err) {
  if (!value) {
    if (err) {
      (*err) += "`value` argument is nullptr.\n";
    }
    return false;
  }

  bool array_qual = false;
  std::string typeName = _typeName;
  if (endsWith(typeName, "[]")) {
    typeName = removeSuffix(typeName, "[]");
    array_qual = true;
  }

  nonstd::optional<uint32_t> typeId = value::TryGetTypeId(typeName);

  if (!typeId) {
    if (err) {
      (*err) += "Unsupported type: " + typeName + "\n";
    }
    return false;
  }

  tinyusdz::StreamReader sr(reinterpret_cast<const uint8_t *>(str.data()),
                            str.size(), /* swap endian */ false);
  tinyusdz::ascii::AsciiParser parser(&sr);

#define PARSE_BASE_TYPE(__ty)                                            \
  case value::TypeTraits<__ty>::type_id(): {                             \
    if (array_qual) {                                                    \
      std::vector<__ty> vss;                                             \
      if (!parser.ParseBasicTypeArray(&vss)) {                           \
        if (err) {                                                       \
          (*err) = fmt::format("Failed to parse a value of type `{}[]`", \
                               value::TypeTraits<__ty>::type_name());    \
        }                                                                \
        return false;                                                    \
      }                                                                  \
      dst = vss;                                                         \
    } else {                                                             \
      __ty val;                                                          \
      if (!parser.ReadBasicType(&val)) {                                 \
        if (err) {                                                       \
          (*err) = fmt::format("Failed to parse a value of type `{}`",   \
                               value::TypeTraits<__ty>::type_name());    \
        }                                                                \
        return false;                                                    \
      }                                                                  \
      dst = val;                                                         \
    }                                                                    \
    break;                                                               \
  }

  value::Value dst;

  switch (typeId.value()) {
    PARSE_BASE_TYPE(value::uint2)
    PARSE_BASE_TYPE(value::uint3)
    PARSE_BASE_TYPE(value::uint4)
    default: {
      if (err) {
        (*err) =
            fmt::format("Unsupported or unimplemeneted type `{}`", typeName);
      }
      return false;
    }
  }

  (*value) = std::move(dst);

  return true;
}

}  // namespace ascii
}  // namespace tinyusdz

#else  // TINYUSDZ_DISABLE_MODULE_USDA_READER

bool ParseUnregistredValue(const std::string &typeName, const std::string &str,
                           value::Value *value, std::string *err) {
  if (err) {
    (*err) += "USDA_READER module is disabled.\n";
  }
  return false;
}

#endif  // TINYUSDZ_DISABLE_MODULE_USDA_READER

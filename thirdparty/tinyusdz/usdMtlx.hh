// SPDX-License-Identifier: Apache 2.0
// Copyright 2023 - Present, Light Transport Entertainment, Inc.
//
// Predefined MaterialX shadingmodel & Built-in MaterialX XML import plugIn.
// Import only. Export is not supported(yet).
//
// example usage
//
// def Shader "mesh" (
//   prepend references = @myshader.mtlx@
// )
// {
//    ...
// }
//
// Based on MaterialX spec v1.38

#pragma once

#include <string>

#include "asset-resolution.hh"
#include "usdShade.hh"


namespace tinyusdz {

constexpr auto kMtlxUsdPreviewSurface = "MtlxUsdPreviewSurface";
constexpr auto kMtlxAutodeskStandardSurface = "MtlxAutodeskStandaradSurface";


namespace mtlx {

enum class ColorSpace {
  Lin_rec709, // lin_rec709
  Unknown
};

} // namespace mtlx

// <surfacematerial>
struct MtlxMaterial {
  std::string name;
  std::string typeName;
  std::string nodename;
};

struct MtlxModel {
  std::string asset_name;

  std::string version;
  std::string cms;
  std::string cmsconfig; // filename
  std::string color_space; // colorspace
  std::string name_space; // namespace

  //mtlx::ColorSpace colorspace{Lin_rec709};
  // TODO

  std::string shader_name;

  // Content of shader.
  // MtlxUsdPreviewSurface or MtlxAutodeskStandaradSurface
  value::Value shader; 

  std::map<std::string, MtlxMaterial> surface_materials;
  std::map<std::string, value::Value> shaders; // MtlxUsdPreviewSurface or MtlxAutodeskStandaradSurface
};

struct MtlxUsdPreviewSurface : UsdPreviewSurface {
  //  TODO: add mtlx specific attribute.
};

// https://github.com/Autodesk/standard-surface/blob/master/reference/standard_surface.mtlx
// We only support v1.0.1
struct MtlxAutodeskStandardSurface : ShaderNode {
  TypedAttributeWithFallback<Animatable<float>> base{1.0f};
  TypedAttributeWithFallback<Animatable<value::color3f>> baseColor{
      value::color3f{0.8f, 0.8f, 0.8f}};  // color3

  // TODO
  // ...

  // (coat_affect_roughness * coat) * coat_roughness
  TypedAttribute<Animatable<float>> coat_affect_roughness;
  TypedAttribute<Animatable<float>> coat;
  TypedAttribute<Animatable<float>> coat_roughness;

  // (specular_roughness + transmission_extra_roughness)
  TypedAttribute<Animatable<float>> specular_roughness;
  TypedAttribute<Animatable<float>> transmission_extra_roughness;
  TypedAttribute<Animatable<float>> transmission_roughness_add;

  // tangent_rotate_normalize
  // normalize(rotate3d(/* in */tangent, /*amount*/(specular_rotation * 360), /*
  // axis */normal))
  TypedAttribute<Animatable<float>> specular_rotation;

  // Output
  TypedTerminalAttribute<value::token> out;  // 'out'
};

//
// IO
//

///
/// Load MaterialX XML from a string.
///
/// @param[in] str String representation of XML data.
/// @param[in] asset_name Corresponding asset name. Can be empty.
/// @param[out] mtlx Output
/// @param[out] warn Warning message
/// @param[out] err Error message
///
/// @return true upon success.
bool ReadMaterialXFromString(const std::string &str, const std::string &asset_name, MtlxModel *mtlx,
                             std::string *warn, std::string *err);

///
/// Load MaterialX XML from a file.
///
/// @param[in] str String representation of XML data.
/// @param[in] asset_name Corresponding asset name. Can be empty.
/// @param[out] mtlx Output
/// @param[out] err Error message
///
/// @return true upon success.
///
/// TODO: Use FileSystem handler

bool ReadMaterialXFromFile(const AssetResolutionResolver &resolver,
                            const std::string &asset_path, MtlxModel *mtlx,
                            std::string *warn, std::string *err);

bool WriteMaterialXToString(const MtlxModel &mtlx, std::string &xml_str,
                             std::string *warn, std::string *err);

bool ToPrimSpec(const MtlxModel &model, PrimSpec &ps, std::string *err);

///
/// Load MaterialX from Asset and construct USD PrimSpec
///
bool LoadMaterialXFromAsset(const Asset &asset,
                            const std::string &asset_path, PrimSpec &ps /* inout */,
                            std::string *warn, std::string *err);

// import DEFINE_TYPE_TRAIT and DEFINE_ROLE_TYPE_TRAIT
#include "define-type-trait.inc"

namespace value {

// ShaderNodes
DEFINE_TYPE_TRAIT(MtlxUsdPreviewSurface, kMtlxUsdPreviewSurface,
                  TYPE_ID_IMAGING_MTLX_PREVIEWSURFACE, 1);
DEFINE_TYPE_TRAIT(MtlxAutodeskStandardSurface, kMtlxAutodeskStandardSurface,
                  TYPE_ID_IMAGING_MTLX_STANDARDSURFACE, 1);

#undef DEFINE_TYPE_TRAIT
#undef DEFINE_ROLE_TYPE_TRAIT

}  // namespace value

}  // namespace tinyusdz

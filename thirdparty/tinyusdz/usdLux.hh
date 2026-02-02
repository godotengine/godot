// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// UsdLux LightSource
#pragma once

#include "prim-types.hh"
#include "xform.hh"

namespace tinyusdz {

constexpr auto kSphereLight = "SphereLight";
constexpr auto kCylinderLight = "CylinderLight";
constexpr auto kDomeLight = "DomeLight";
constexpr auto kDiskLight = "DiskLight";
constexpr auto kRectLight = "RectLight";
constexpr auto kDistantLight = "DistantLight";
constexpr auto kGeometryLight = "GeometryLight";
constexpr auto kPortalLight = "PortalLight";
constexpr auto kPluginLight = "PluginLight";

class BoundableLight : public Xformable, public Collection {

 public:
  std::string name;
  Specifier spec{Specifier::Def};
  int64_t parent_id{-1};  // Index to xform node

  TypedAttribute<Animatable<Extent>> extent; // float3[]
  TypedAttributeWithFallback<Animatable<Visibility>> visibility{Visibility::Inherited};
  TypedAttributeWithFallback<Purpose> purpose{Purpose::Default};

  // Light API
  TypedAttributeWithFallback<Animatable<value::color3f>> color{value::color3f({1.0f, 1.0f, 1.0f})}; // inputs.color Light energy in linear color space.
  TypedAttributeWithFallback<Animatable<float>> colorTemperature{6500.0f};  // inputs:colorTemperature
  TypedAttributeWithFallback<Animatable<float>> diffuse{1.0f}; // inputs:diffuse diffuse multiplier
  TypedAttributeWithFallback<Animatable<bool>> enableColorTemperature{false}; // inputs:enableColorTemperature
  TypedAttributeWithFallback<Animatable<float>> exposure{0.0f}; // inputs:exposure EV
  TypedAttributeWithFallback<Animatable<float>> intensity{1.0f}; // inputs:intensity
  TypedAttributeWithFallback<Animatable<bool>> normalize{false}; // inputs:normalize normalize power by the surface area of the light.
  TypedAttributeWithFallback<Animatable<float>> specular{1.0f}; // inputs:specular specular multiplier
  // rel light:filters


  std::pair<ListEditQual, std::vector<Reference>> references;
  std::pair<ListEditQual, std::vector<Payload>> payload;
  std::map<std::string, VariantSet> variantSet;
  std::map<std::string, Property> props;
  PrimMeta meta; // TODO: move to private

  const PrimMeta &metas() const { return meta; }
  PrimMeta &metas() { return meta; }

  const std::vector<value::token> &primChildrenNames() const { return _primChildren; }
  const std::vector<value::token> &propertyNames() const { return _properties; }
  std::vector<value::token> &primChildrenNames() { return _primChildren; }
  std::vector<value::token> &propertyNames() { return _properties; }

 private:

  std::vector<value::token> _primChildren;
  std::vector<value::token> _properties;
};

// non-boundable still inherits Xformable. 
// Differences with boundable is just `extent` attribute is omitted.
class NonboundableLight : public Xformable, public Collection {

 public:
  std::string name;
  Specifier spec{Specifier::Def};
  int64_t parent_id{-1};  // Index to xform node

  TypedAttributeWithFallback<Animatable<Visibility>> visibility{Visibility::Inherited};
  TypedAttributeWithFallback<Purpose> purpose{Purpose::Default};

  // Light API
  TypedAttributeWithFallback<Animatable<value::color3f>> color{value::color3f({1.0f, 1.0f, 1.0f})}; // inputs.color Light energy in linear color space.
  TypedAttributeWithFallback<Animatable<float>> colorTemperature{6500.0f};  // inputs:colorTemperature
  TypedAttributeWithFallback<Animatable<float>> diffuse{1.0f}; // inputs:diffuse diffuse multiplier
  TypedAttributeWithFallback<Animatable<bool>> enableColorTemperature{false}; // inputs:enableColorTemperature
  TypedAttributeWithFallback<Animatable<float>> exposure{0.0f}; // inputs:exposure EV
  TypedAttributeWithFallback<Animatable<float>> intensity{1.0f}; // inputs:intensity
  TypedAttributeWithFallback<Animatable<bool>> normalize{false}; // inputs:normalize normalize power by the surface area of the light.
  TypedAttributeWithFallback<Animatable<float>> specular{1.0f}; // inputs:specular specular multiplier
  // rel light:filters


  std::pair<ListEditQual, std::vector<Reference>> references;
  std::pair<ListEditQual, std::vector<Payload>> payload;
  std::map<std::string, VariantSet> variantSet;
  std::map<std::string, Property> props;
  PrimMeta meta; // TODO: move to private

  const PrimMeta &metas() const { return meta; }
  PrimMeta &metas() { return meta; }

  const std::vector<value::token> &primChildrenNames() const { return _primChildren; }
  const std::vector<value::token> &propertyNames() const { return _properties; }
  std::vector<value::token> &primChildrenNames() { return _primChildren; }
  std::vector<value::token> &propertyNames() { return _properties; }

 private:

  std::vector<value::token> _primChildren;
  std::vector<value::token> _properties;
};

struct SphereLight : public BoundableLight {

  TypedAttributeWithFallback<Animatable<float>> radius{0.5f}; // inputs:radius

};

struct CylinderLight : public BoundableLight {

  TypedAttributeWithFallback<Animatable<float>> length{1.0f}; // inputs:length size in Y axis
  TypedAttributeWithFallback<Animatable<float>> radius{0.5f}; // inputs:radius  size in X axis

};


struct RectLight : public BoundableLight {

  TypedAttribute<Animatable<value::AssetPath>> file; // asset inputs:texture:file
  TypedAttributeWithFallback<Animatable<float>> height{1.0f}; // inputs:height size in Y axis
  TypedAttributeWithFallback<Animatable<float>> width{1.0f}; // inputs:width  size in X axis

};

struct DiskLight : public BoundableLight {

  TypedAttributeWithFallback<Animatable<float>> radius{0.5f}; // inputs:radius

};

// NOTE: Make Distance xformable?
struct DistantLight : public NonboundableLight {

  TypedAttributeWithFallback<Animatable<float>> angle{0.53f}; // inputs:angle in degrees

};

struct DomeLight : public NonboundableLight {

  enum class TextureFormat {
    Automatic, // "automatic"
    Latlong, // "latlong"
    MirroredBall, // "mirroredBall"
    Angular // "angular"
  };

  // DomeLight specific
  TypedAttributeWithFallback<Animatable<float>> guideRadius{1.0e5f};
  TypedAttribute<Animatable<value::AssetPath>> file; // asset inputs:texture:file
  TypedAttributeWithFallback<Animatable<TextureFormat>> textureFormat{TextureFormat::Automatic}; // token inputs:texture:format
  // rel portals
  // rel proxyPrim
  
};

// TODO: Deprecate
struct GeometryLight : public NonboundableLight {

  RelationshipProperty geometry; // `rel geometry`

};

// TODO
struct PortalLight : public NonboundableLight {

};

// TODO
struct PluginLight : public Xformable, public Collection {
};

#if 0 // TODO
struct PluginLightFilter : public Light {
};
#endif

inline bool IsLightPrim(const Prim &prim) {
  return (prim.type_id() > value::TYPE_ID_LUX_BEGIN) && (prim.type_id() < value::TYPE_ID_LUX_END);
}

// import DEFINE_TYPE_TRAIT and DEFINE_ROLE_TYPE_TRAIT
#include "define-type-trait.inc"

namespace value {

DEFINE_TYPE_TRAIT(DomeLight, kDomeLight, TYPE_ID_LUX_DOME, 1);
DEFINE_TYPE_TRAIT(CylinderLight, kCylinderLight, TYPE_ID_LUX_CYLINDER, 1);
DEFINE_TYPE_TRAIT(SphereLight, kSphereLight, TYPE_ID_LUX_SPHERE, 1);
DEFINE_TYPE_TRAIT(DiskLight, kDiskLight, TYPE_ID_LUX_DISK, 1);
DEFINE_TYPE_TRAIT(DistantLight, kDistantLight, TYPE_ID_LUX_DISTANT, 1);
DEFINE_TYPE_TRAIT(RectLight, kRectLight, TYPE_ID_LUX_RECT, 1);
DEFINE_TYPE_TRAIT(GeometryLight, kGeometryLight, TYPE_ID_LUX_GEOMETRY, 1);
DEFINE_TYPE_TRAIT(PortalLight, kPortalLight, TYPE_ID_LUX_PORTAL, 1);
DEFINE_TYPE_TRAIT(PluginLight, kPluginLight, TYPE_ID_LUX_PLUGIN, 1);

#undef DEFINE_TYPE_TRAIT
#undef DEFINE_ROLE_TYPE_TRAIT

} // namespace value

}  // namespace tinyusdz

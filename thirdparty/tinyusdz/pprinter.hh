#pragma once

//
// pretty-print routine(using iostream) in non-intrusive way.
// Some build configuration may not want I/O module(e.g. mobile/embedded
// device), so provide print routines in separated file.
//
//

#include <ostream>
#include <sstream>
#include <string>

#include "prim-types.hh"
#include "usdGeom.hh"
#include "usdLux.hh"
#include "usdShade.hh"
#include "usdSkel.hh"
#include "value-pprint.hh"

namespace tinyusdz {

namespace pprint {

void SetIndentString(const std::string &s);
std::string Indent(uint32_t level);

}  // namespace pprint

std::string to_string(Visibility v);
std::string to_string(Orientation o);
std::string to_string(Extent e);
std::string to_string(Interpolation interp);
std::string to_string(Axis axis);
std::string to_string(ListEditQual qual);
std::string to_string(Specifier specifier);
std::string to_string(Purpose purpose);
std::string to_string(Permission permission);
std::string to_string(Variability variability);
std::string to_string(SpecType spec_type);
std::string to_string(Kind kind);
std::string to_string(const Reference &reference);
std::string to_string(const Payload &payload);

std::string to_string(const XformOp::OpType &ty);

std::string to_string(GeomMesh::InterpolateBoundary interp_boundary);
std::string to_string(GeomMesh::SubdivisionScheme subd_scheme);
std::string to_string(GeomMesh::FaceVaryingLinearInterpolation fv);

std::string to_string(const Path &path, bool show_full_path = true);
std::string to_string(const std::vector<Path> &v, bool show_full_path = true);

// For debugging
std::string dump_path(const Path &p);

template <typename T>
std::string to_string(const std::vector<T> &v, const uint32_t level = 0) {
  std::stringstream ss;
  ss << pprint::Indent(level) << "[";

  // TODO(syoyo): indent for large array
  for (size_t i = 0; i < v.size(); i++) {
    ss << to_string(v[i]);
    if (i != (v.size() - 1)) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

#if 0
template<>
std::string to_string(const std::vector<std::string> &v, const uint32_t level) {
  std::stringstream ss;
  ss << pprint::Indent(level) << "[";

  // TODO(syoyo): indent for large array
  for (size_t i = 0; i < v.size(); i++) {
    ss << quote(v[i]);
    if (i != (v.size() -1)) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}
#endif

template <typename T>
std::string to_string(const ListOp<T> &op, const uint32_t indent_level = 0) {
  std::stringstream ss;
  ss << pprint::Indent(indent_level) << "ListOp(isExplicit " << op.IsExplicit()
     << ") {\n";
  ss << pprint::Indent(indent_level)
     << "  explicit_items = " << to_string(op.GetExplicitItems()) << "\n";
  ss << pprint::Indent(indent_level)
     << "  added_items = " << to_string(op.GetAddedItems()) << "\n";
  ss << pprint::Indent(indent_level)
     << "  prepended_items = " << to_string(op.GetPrependedItems()) << "\n";
  ss << pprint::Indent(indent_level)
     << "  deleted_items = " << to_string(op.GetDeletedItems()) << "\n";
  ss << pprint::Indent(indent_level)
     << "  ordered_items = " << to_string(op.GetOrderedItems()) << "\n";
  ss << pprint::Indent(indent_level) << "}";

  return ss.str();
}

//
// Setting `closing_brace` false won't emit `}`(for printing USD scene graph
// recursively).
//

std::string to_string(const Model &model, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const Scope &scope, const uint32_t indent = 0,
                      bool closing_brace = true);
// std::string to_string(const Klass &klass, const uint32_t indent = 0, bool
// closing_brace = true);
std::string to_string(const GPrim &gprim, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const Xform &xform, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomSphere &sphere, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomMesh &mesh, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomPoints &pts, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomBasisCurves &curves, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomNurbsCurves &curves, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomCapsule &geom, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomCone &geom, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomCylinder &geom, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomCube &geom, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomCamera &camera, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomSubset &subset, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeomSubset::ElementType ty);
std::string to_string(const GeomSubset::FamilyType ty);

std::string to_string(const GeomBasisCurves::Wrap &v);
std::string to_string(const GeomBasisCurves::Type &v);
std::string to_string(const GeomBasisCurves::Basis &v);

std::string to_string(const PointInstancer &instancer, const uint32_t indent = 0,
                      bool closing_brace = true);


std::string to_string(const SkelRoot &root, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const Skeleton &skel, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const SkelAnimation &anim, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const BlendShape &bs, const uint32_t indent = 0,
                      bool closing_brace = true);

std::string to_string(const SphereLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const DomeLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const DiskLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const DistantLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const CylinderLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const RectLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const GeometryLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const PortalLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const PluginLight &light, const uint32_t indent = 0,
                      bool closing_brace = true);

std::string to_string(const DomeLight::TextureFormat &texformat);

std::string to_string(const Material &material, const uint32_t indent = 0,
                      bool closing_brace = true);

// It will delegate to to_string() of concrete Shader type(e.g.
// UsdPreviewSurface)
std::string to_string(const Shader &shader, const uint32_t indent = 0,
                      bool closing_brace = true);

std::string to_string(const UsdPreviewSurface &shader,
                      const uint32_t indent = 0, bool closing_brace = true);
std::string to_string(const UsdUVTexture &shader, const uint32_t indent = 0,
                      bool closing_brace = true);
std::string to_string(const UsdPrimvarReader_float &shader,
                      const uint32_t indent = 0, bool closing_brace = true);
std::string to_string(const UsdPrimvarReader_float2 &shader,
                      const uint32_t indent = 0, bool closing_brace = true);
std::string to_string(const UsdPrimvarReader_float3 &shader,
                      const uint32_t indent = 0, bool closing_brace = true);
std::string to_string(const UsdPrimvarReader_float4 &shader,
                      const uint32_t indent = 0, bool closing_brace = true);
std::string to_string(const UsdPrimvarReader_int &shader,
                      const uint32_t indent = 0, bool closing_brace = true);

std::string to_string(const UsdPreviewSurface::OpacityMode v);

std::string to_string(const UsdUVTexture::SourceColorSpace v);
std::string to_string(const UsdUVTexture::Wrap v);

std::string to_string(const GeomCamera::Projection &proj);
std::string to_string(const GeomCamera::StereoRole &role);

std::string to_string(const tinyusdz::Animatable<Visibility> &v,
                      const uint32_t indent = 0, bool closing_brace = true);

std::string to_string(const APISchemas::APIName &name);
std::string to_string(const CustomDataType &customData);

std::string to_string(const Layer &layer, const uint32_t indent = 0, bool closing_brace = true);
std::string to_string(const PrimSpec &primspec, const uint32_t indent = 0, bool closing_brace = true);

std::string to_string(const CollectionInstance::ExpansionRule rule);

std::string print_xformOpOrder(const std::vector<XformOp> &xformOps,
                               const uint32_t indent);
std::string print_xformOps(const std::vector<XformOp> &xformOps,
                           const uint32_t indent);
std::string print_attr_metas(const AttrMeta &meta, const uint32_t indent);

// varname = optional variable name which is used when meta.get_name() is empty.
std::string print_meta(const MetaVariable &meta, const uint32_t indent, bool emit_type_name,
                       const std::string &varname = std::string());
std::string print_prim_metas(const PrimMeta &meta, const uint32_t indent);
std::string print_customData(const CustomDataType &customData,
                             const std::string &name, const uint32_t indent);
std::string print_variantSelectionMap(const VariantSelectionMap &m,
                                      const uint32_t indent);
std::string print_variantSetStmt(
    const std::map<std::string, VariantSet> &vslist, const uint32_t indent);
std::string print_variantSetSpecStmt(
    const std::map<std::string, VariantSetSpec> &vslist, const uint32_t indent);
std::string print_payload(const prim::PayloadList &payload,
                          const uint32_t indent);
std::string print_timesamples(const value::TimeSamples &v,
                              const uint32_t indent);
std::string print_rel_prop(const Property &prop, const std::string &name,
                           uint32_t indent);

std::string print_prop(const Property &prop, const std::string &prop_name,
                       uint32_t indent);

// Print properties.
// TODO: Deprecate this function.
std::string print_props(const std::map<std::string, Property> &props,
                        uint32_t indent);

// tok_table: Manages property is already printed(built-in props) or not.
// propNames: Specify the order of property to print
// When `propNames` is empty, print all of items in `props`.
std::string print_props(const std::map<std::string, Property> &props,
                        /* input */ std::set<std::string> &tok_table,
                        const std::vector<value::token> &propNames,
                        uint32_t indent);

std::string print_layer_metas(const LayerMetas &metas, const uint32_t indent);
std::string print_layer(const Layer &layer, const uint32_t indent);

std::string print_material_binding(const MaterialBinding *mb, const uint32_t indent);
std::string print_collection(const Collection *coll, const uint32_t indent);

}  // namespace tinyusdz

namespace std {

std::ostream &operator<<(std::ostream &ofs, const tinyusdz::Visibility v);
std::ostream &operator<<(std::ostream &ofs, const tinyusdz::Extent v);
std::ostream &operator<<(std::ostream &ofs, const tinyusdz::Interpolation v);
std::ostream &operator<<(std::ostream &ofs, const tinyusdz::Layer &layer);

}  // namespace std

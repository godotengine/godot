// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// UsdGeom
//
// TODO
//
// - [ ] Replace nonstd::optional<T> member to RelationshipProperty or TypedAttribute***<T>
//
#pragma once

#include "prim-types.hh"
#include "value-types.hh"
#include "xform.hh"
#include "usdShade.hh"

namespace tinyusdz {

// From schema definition.
constexpr auto kGPrim = "GPrim";
constexpr auto kGeomCube = "Cube";
constexpr auto kGeomXform = "Xform";
constexpr auto kGeomMesh = "Mesh";
constexpr auto kGeomSubset = "GeomSubset";
constexpr auto kGeomBasisCurves = "BasisCurves";
constexpr auto kGeomNurbsCurves = "NurbsCurves";
constexpr auto kGeomCylinder = "Cylinder";
constexpr auto kGeomCapsule = "Capsule";
constexpr auto kGeomPoints = "Points";
constexpr auto kGeomCone = "Cone";
constexpr auto kGeomSphere = "Sphere";
constexpr auto kGeomCamera = "Camera";
constexpr auto kPointInstancer = "PointInstancer";

constexpr auto kMaterialBinding = "material:binding";
constexpr auto kMaterialBindingCollection = "material:binding:collection";
constexpr auto kMaterialBindingPreview = "material:binding:preview";
constexpr auto kMaterialBindingFull = "material:binding:full";

struct GPrim;

bool IsSupportedGeomPrimvarType(uint32_t tyid);
bool IsSupportedGeomPrimvarType(const std::string &type_name);

//
// GeomPrimvar is a wrapper class for Attribute and indices(for Indexed Primvar)
// - Attribute with `primvars` prefix. e.g. "primvars:
// - Optional: indices.
//
// GeomPrimvar is only constructable from GPrim.
// This class COPIES variable from GPrim for `get` operation.
//
// Currently read-only operation is well provided. writing feature is not well tested(`set_value` may have issue)
// (If you struggled to ue GeomPrimvar, please operate on `GPrim::props` directly)
//
// Limitation:
// TimeSamples are not supported for indices.
// Also, TimeSamples are not supported both when constructing GeomPrimvar with Typed Attribute value and retrieving Attribute value.
//
//
class GeomPrimvar {

 friend GPrim;

 public:
  GeomPrimvar() : _has_value(false) {
  }

  GeomPrimvar(const Attribute &attr) : _attr(attr) {
    _has_value = true;
  }

  GeomPrimvar(const Attribute &attr, const std::vector<int32_t> &indices) : _attr(attr)
  {
    _indices = indices;
    _has_value = true;
  }

  GeomPrimvar(const Attribute &attr, const TypedTimeSamples<std::vector<int32_t>> &indices) : _attr(attr)
  {
    _ts_indices = indices;
    _has_value = true;
  }

  GeomPrimvar(const Attribute &attr, TypedTimeSamples<std::vector<int32_t>> &&indices) : _attr(attr)
  {
    _ts_indices = std::move(indices);
    _has_value = true;
  }

  GeomPrimvar(const GeomPrimvar &rhs) {
    _name = rhs._name;
    _attr = rhs._attr;
    _indices = rhs._indices;
    _ts_indices = rhs._ts_indices;
    _has_value = rhs._has_value;
    if (rhs._elementSize) {
      _elementSize = rhs._elementSize;
    }

    if (rhs._interpolation) {
      _interpolation = rhs._interpolation;
    }
    _unauthoredValuesIndex = rhs._unauthoredValuesIndex;
  }

  GeomPrimvar &operator=(const GeomPrimvar &rhs) {
    _name = rhs._name;
    _attr = rhs._attr;
    _indices = rhs._indices;
    _ts_indices = rhs._ts_indices;
    _has_value = rhs._has_value;
    if (rhs._elementSize) {
      _elementSize = rhs._elementSize;
    }

    if (rhs._interpolation) {
      _interpolation = rhs._interpolation;
    }
    _unauthoredValuesIndex = rhs._unauthoredValuesIndex;

    return *this;
  }

  ///
  /// For Indexed Primvar(array value + indices)
  ///
  /// equivalent to ComputeFlattened in pxrUSD.
  ///
  /// ```
  /// for i in len(indices):
  ///   dest[i] = values[indices[i]]
  /// ```
  ///
  /// Use Default time and Linear interpolation when `indices` and/or primvar is timesamples.
  ///
  /// If Primvar does not have indices, return attribute value as is(same with `get_value`).
  /// For now, we only support Attribute with 1D array type.
  ///
  /// Return false when operation failed or if the attribute type is not supported for Indexed Primvar.
  ///
  ///
  template <typename T>
  bool flatten_with_indices(std::vector<T> *dst, std::string *err = nullptr) const;

  ///
  /// Specify time and interpolation type.
  ///
  template <typename T>
  bool flatten_with_indices(double t, std::vector<T> *dst, value::TimeSampleInterpolationType tinerp = value::TimeSampleInterpolationType::Linear, std::string *err = nullptr) const;


  // Generic Value version.
  // TODO: return Attribute?
  bool flatten_with_indices(value::Value *dst, std::string *err = nullptr) const;
  bool flatten_with_indices(double t, value::Value *dst, value::TimeSampleInterpolationType tinterp = value::TimeSampleInterpolationType::Linear, std::string *err = nullptr) const;

  bool has_elementSize() const;
  uint32_t get_elementSize() const;

  bool has_interpolation() const;
  Interpolation get_interpolation() const;

  void set_elementSize(uint32_t n) {
    _elementSize = n;
  }

  void set_interpolation(const Interpolation interp) {
    _interpolation = interp;
  }

  bool has_unauthoredValuesIndex() const {
    return _unauthoredValuesIndex.has_value();
  }

  int get_unauthoredValuesIndex() const {
    return _unauthoredValuesIndex.value_or(-1);
  }

  void set_unauthoredValuesIndex(int n) {
    _unauthoredValuesIndex = n;
  }

  ///
  /// Get indices at specified timecode.
  /// Returns empty when appropriate indices does not exist for timecode 't'.
  ///
  std::vector<int32_t> get_indices(const double t = value::TimeCode::Default()) const;
  
  const std::vector<int32_t> &get_default_indices() const {
    return _indices;
  }

  const TypedTimeSamples<std::vector<int32_t>> &get_timesampled_indices() const {
    return _ts_indices;
  }

  bool has_default_indices() const { return !_indices.empty(); }
  bool has_timesampled_indices() const { return _ts_indices.size() > 0; }

  bool has_indices() const {
    return has_default_indices() || has_timesampled_indices();
  }

  uint32_t type_id() const { return _attr.type_id(); }
  std::string type_name() const { return _attr.type_name(); }

  // Name of Primvar. "primvars:" prefix(namespace) is omitted.
  const std::string name() const { return _name; }

  ///
  /// Attribute has value?(Not empty)
  ///
  bool has_value() const {
    return _has_value;
  }

  ///
  /// Get type name of primvar.
  ///
  std::string get_type_name() const {
    if (!_has_value) {
      return "null";
    }

    return _attr.type_name();
  }

  ///
  /// Get type id of primvar.
  ///
  uint32_t get_type_id() const {
    if (!_has_value) {
      return value::TYPE_ID_NULL;
    }
    return _attr.type_id();
  }

  ///
  /// Get Attribute value.
  ///
  template <typename T>
  bool get_value(T *dst, std::string *err = nullptr) const;

  bool get_value(value::Value *dst, std::string *err = nullptr) const;


  ///
  /// Get Attribute value at specified time.
  ///
  template <typename T>
  bool get_value(double timecode, T *dst, const value::TimeSampleInterpolationType interp = value::TimeSampleInterpolationType::Linear, std::string *err = nullptr) const;

  bool get_value(double timecode, value::Value *dst, const value::TimeSampleInterpolationType interp = value::TimeSampleInterpolationType::Linear, std::string *err = nullptr) const;

  ///
  /// Set Attribute value.
  ///
  template <typename T>
  void set_value(const T &val) {
    _attr.set_value(val);
    _has_value = true;
  }

  void set_value(const Attribute &attr) {
    _attr = attr;
    _has_value = true;
  }

  void set_value(Attribute &&attr) {
    _attr = std::move(attr);
    _has_value = true;
  }

  void set_name(const std::string &name) { _name = name; }

  // Set indices for specified timecode 't'
  // indices will be replaced when there is an indices at timecode 't'.
  void set_indices(const std::vector<int32_t> &indices, const double t = value::TimeCode::Default());


  void set_default_indices(const std::vector<int32_t> &indices) {
    _indices = indices;
  }

  void set_default_indices(std::vector<int32_t> &&indices) {
    _indices = std::move(indices);
  }

  void set_timesampled_indices(const TypedTimeSamples<std::vector<int32_t>> &indices) {
    _ts_indices = indices;
  }

  const Attribute &get_attribute() const {
    return _attr;
  }

 private:

  std::string _name;
  bool _has_value{false};
  Attribute _attr;
  std::vector<int32_t> _indices;  // 'default' indices
  TypedTimeSamples<std::vector<int32_t>> _ts_indices;

  // Store Attribute meta separately.
  nonstd::optional<int32_t> _unauthoredValuesIndex; // for sparse primvars in some DCC. default = -1.
  nonstd::optional<uint32_t> _elementSize;
  nonstd::optional<Interpolation> _interpolation;

#if 0 // TODO
  bool get_value(const value::Value *value,
                 const double t = value::TimeCode::Default(),
                 const value::TimeSampleInterpolationType tinterp =
                     value::TimeSampleInterpolationType::Held);
#endif

};

// Geometric Prim. Encapsulates Imagable + Boundable in pxrUSD schema.
// <pxrUSD>/pxr/usd/usdGeom/schema.udsa
//
// TODO: inherit UsdShagePrim?

struct GPrim : Xformable, MaterialBinding, Collection {
  std::string name;
  Specifier spec{Specifier::Def};

  int64_t parent_id{-1};  // Index to parent node

  std::string prim_type;  // Primitive type(if specified by `def`)

  void set_name(const std::string &name_) {
    name = name_;
  }

  const std::string &get_name() const {
    return name;
  }

  Specifier &specifier() { return spec; }
  const Specifier &specifier() const { return spec; }

  // Gprim

  TypedAttribute<Animatable<Extent>>
      extent;  // bounding extent. When authorized, the extent is the bounding
               // box of whole its children.

  TypedAttributeWithFallback<bool> doubleSided{
      false};  // "uniform bool doubleSided"

  TypedAttributeWithFallback<Orientation> orientation{
      Orientation::RightHanded};  // "uniform token orientation"
  TypedAttributeWithFallback<Animatable<Visibility>> visibility{
      Visibility::Inherited};  // "token visibility"
  TypedAttributeWithFallback<Purpose> purpose{
      Purpose::Default};  // "uniform token purpose"

  // Handy API to get `primvars:displayColor` and `primvars:displayOpacity`
  bool get_displayColor(value::color3f *col, const double t = value::TimeCode::Default(), const value::TimeSampleInterpolationType tinterp = value::TimeSampleInterpolationType::Linear) const;

  bool get_displayOpacity(float *opacity, const double t = value::TimeCode::Default(), const value::TimeSampleInterpolationType tinterp = value::TimeSampleInterpolationType::Linear) const;

  const std::vector<value::color3f> get_displayColors(
      double time = value::TimeCode::Default(),
      value::TimeSampleInterpolationType interp =
          value::TimeSampleInterpolationType::Linear) const;

  Interpolation get_displayColorsInterpolation() const;

  RelationshipProperty proxyPrim;

#if 0
  // Some frequently used materialBindings
  nonstd::optional<Relationship> materialBinding; // material:binding
  nonstd::optional<Relationship> materialBindingCollection; // material:binding:collection  TODO: deprecate?(seems `material:binding:collection` without leaf NAME seems ignored in pxrUSD.
  nonstd::optional<Relationship> materialBindingPreview; // material:binding:preview
  nonstd::optional<Relationship> materialBindingFull; // material:binding:full
#endif

  std::map<std::string, Property> props;

  std::pair<ListEditQual, std::vector<Reference>> references;
  std::pair<ListEditQual, std::vector<Payload>> payload;
  std::map<std::string, VariantSet> variantSet;

  // For GeomPrimvar.

  ///
  /// Get Attribute(+ indices Attribute for Indexed Primvar) with "primvars:" suffix(namespace) in `props`
  ///
  /// NOTE: This API does not support Connection Atttribute(e.g. `int[] primvars:uvs:indices = </root/geom0.indices>`)
  /// If you want to get Primvar with possible Connection Attribute, use Tydra API: `GetGeomPrimvar`
  ///
  /// @param[in] name Primvar name(`primvars:` prefix omitted. e.g. "normals", "st0", ...)
  /// @param[out] primvar GeomPrimvar output.
  /// @param[out] err Optional Error message(filled when returning false)
  ///
  bool get_primvar(const std::string &name, GeomPrimvar *primvar, std::string *err = nullptr) const;

  ///
  /// Check if primvar exists with given name
  ///
  /// @param[in] name Primvar name(`primvars:` prefix omitted. e.g. "normals", "st0", ...)
  ///
  bool has_primvar(const std::string &name) const;

  ///
  /// Return List of Primvar in this GPrim contains.
  ///
  /// NOTE: This API does not support Connection Atttribute(e.g. `int[] primvars:uvs:indices = </root/geom0.indices>`)
  /// If you want to get Primvar with possible Connection Attribute, use Tydra API: `GetGeomPrimvars`
  ///
  std::vector<GeomPrimvar> get_primvars() const;

  ///
  /// Set Attribute(+ indices Attribute for Indexed Primvar) with "primvars:" suffix(namespace) to `props`
  ///
  /// @param[in] primvar GeomPrimvar
  /// @param[out] err Optional Error message(filled when returning false)
  ///
  /// Returns true when success to add primvar. Return false on error(e.g. `primvar` does not contain valid name).
  ///
  bool set_primvar(const GeomPrimvar &primvar, std::string *err = nullptr);


  ///
  /// Aux infos
  ///
  std::vector<value::token> &primChildrenNames() {
    return _primChildrenNames;
  }

  std::vector<value::token> &propertyNames() {
    return _propertyNames;
  }

  const std::vector<value::token> &primChildrenNames() const {
    return _primChildrenNames;
  }

  const std::vector<value::token> &propertyNames() const {
    return _propertyNames;
  }

  const std::map<std::string, VariantSet> &variantSetList() const {
    return _variantSetMap;
  }

  std::map<std::string, VariantSet> &variantSetList() {
    return _variantSetMap;
  }

  // Prim metadataum.
  PrimMeta meta; // TODO: Move to private

  const PrimMeta &metas() const {
    return meta;
  }

  PrimMeta &metas() {
    return meta;
  }

#if 0
  //
  // NOTE on material binding.
  // https://openusd.org/release/wp_usdshade.html
  //
  //  - "all purpose", direct binding, material:binding. single relationship target only
  //  - a purpose-restricted, direct, fallback binding, e.g. material:binding:preview
  //  - an all-purpose, collection-based binding, e.g. material:binding:collection:metalBits
  //  - a purpose-restricted, collection-based binding, e.g. material:binding:collection:full:metalBits
  //
  // In TinyUSDZ, treat empty purpose token as "all purpose"
  //

  bool has_materialBinding() const {
    return materialBinding.has_value();
  }

  bool has_materialBindingPreview() const {
    return materialBindingPreview.has_value();
  }

  bool has_materialBindingFull() const {
    return materialBindingFull.has_value();
  }

  bool has_materialBinding(const value::token &mat_purpose) const {
    if (mat_purpose.str() == "full") {
      return has_materialBindingFull();
    } else if (mat_purpose.str() == "preview") {
      return has_materialBindingPreview();
    } else {
      return _materialBindingMap.count(mat_purpose.str());
    }
  }

  void clear_materialBinding() {
    materialBinding.reset();
  }

  void clear_materialBindingPreview() {
    materialBindingPreview.reset();
  }

  void clear_materialBindingFull() {
    materialBindingFull.reset();
  }

  void set_materialBinding(const Relationship &rel) {
    materialBinding = rel;
  }

  void set_materialBinding(const Relationship &rel, const MaterialBindingStrength strength) {
    value::token strength_tok(to_string(strength));
    materialBinding = rel;
    materialBinding.value().metas().bindMaterialAs = strength_tok;
  }

  void set_materialBindingPreview(const Relationship &rel) {
    materialBindingPreview = rel;
  }

  void set_materialBindingPreview(const Relationship &rel, const MaterialBindingStrength strength) {
    value::token strength_tok(to_string(strength));
    materialBindingPreview = rel;
    materialBindingPreview.value().metas().bindMaterialAs = strength_tok;
  }

  void set_materialBindingFull(const Relationship &rel) {
    materialBindingFull = rel;
  }

  void set_materialBindingFull(const Relationship &rel, const MaterialBindingStrength strength) {
    value::token strength_tok(to_string(strength));
    materialBindingFull = rel;
    materialBindingFull.value().metas().bindMaterialAs = strength_tok;
  }

  void set_materialBinding(const Relationship &rel, const value::token &mat_purpose) {

    if (mat_purpose.str().empty()) {
      return set_materialBinding(rel);
    } else if (mat_purpose.str() == "full") {
      return set_materialBindingFull(rel);
    } else if (mat_purpose.str() == "preview") {
      return set_materialBindingFull(rel);
    } else {
      _materialBindingMap[mat_purpose.str()] = rel;
    }
  }

  void set_materialBinding(const Relationship &rel, const value::token &mat_purpose, const MaterialBindingStrength strength) {
    value::token strength_tok(to_string(strength));

    if (mat_purpose.str().empty()) {
      return set_materialBinding(rel, strength);
    } else if (mat_purpose.str() == "full") {
      return set_materialBindingFull(rel, strength);
    } else if (mat_purpose.str() == "preview") {
      return set_materialBindingFull(rel, strength);
    } else {
      _materialBindingMap[mat_purpose.str()] = rel;
      _materialBindingMap[mat_purpose.str()].metas().bindMaterialAs = strength_tok;
    }
  }

  bool has_materialBindingCollection(const std::string &tok) {

    if (!_materialBindingCollectionMap.count(tok)) {
      return false;
    }

    return _materialBindingCollectionMap.count(tok);
  }

  void set_materialBindingCollection(const value::token &tok, const value::token &mat_purpose, const Relationship &rel) {

    // NOTE:
    // https://openusd.org/release/wp_usdshade.html#basic-proposal-for-collection-based-assignment
    // says: material:binding:collection defines a namespace of binding relationships to be applied in namespace order, with the earliest ordered binding relationship the strongest
    //
    // so the app is better first check if `tok` element alreasy exists(using has_materialBindingCollection)

    auto &m = _materialBindingCollectionMap[tok.str()];

    m[mat_purpose.str()] = rel;
  }

  void clear_materialBindingCollection(const value::token &tok, const value::token &mat_purpose) {
    if (_materialBindingCollectionMap.count(tok.str())) {
      _materialBindingCollectionMap[tok.str()].erase(mat_purpose.str());
    }
  }

  void set_materialBindingCollection(const value::token &tok, const value::token &mat_purpose, const Relationship &rel, MaterialBindingStrength strength) {
    value::token strength_tok(to_string(strength));

    _materialBindingCollectionMap[tok.str()][mat_purpose.str()] = rel;
    _materialBindingCollectionMap[tok.str()][mat_purpose.str()].metas().bindMaterialAs = strength_tok;

  }

  const std::map<std::string, std::map<std::string, Relationship>> materialBindingCollectionMap() const {
    return _materialBindingCollectionMap;
  }
#endif


 private:

  //bool _valid{true};  // default behavior is valid(allow empty GPrim)

  std::vector<value::token> _primChildrenNames;
  std::vector<value::token> _propertyNames;

  // For Variants
  std::map<std::string, VariantSet> _variantSetMap;

#if 0
  // For material:binding(excludes frequently used `material:binding`, `material:binding:full` and `material:binding:preview`)
  // key = PURPOSE, value = rel
  std::map<std::string, Relationship> _materialBindingMap;

  // For material:binding:collection
  // key = NAME, value = map<PURPOSE, Rel>
  // TODO: Use multi-index map
  std::map<std::string, std::map<std::string, Relationship>> _materialBindingCollectionMap;
#endif

};

struct Xform : GPrim {
  // Xform() {}
};

// GeomSubset
struct GeomSubset : public MaterialBinding, Collection {
  enum class ElementType { Face, Point };

  enum class FamilyType {
    Partition,       // 'partition'
    NonOverlapping,  // 'nonOverlapping'
    Unrestricted,    // 'unrestricted' (fallback)
  };

  std::string name;
  Specifier spec{Specifier::Def};

  int64_t parent_id{-1};  // Index to parent node

  TypedAttributeWithFallback<ElementType> elementType{ElementType::Face};
  TypedAttribute<value::token> familyName;  // "uniform token familyName"

  // FamilyType attribute is described in parent GeomMesh's `subsetFamily:<FAMILYNAME>:familyType` attribute.
  //TypedAttributeWithFallback<FamilyType> familyType{FamilyType::Unrestricted};

  nonstd::expected<bool, std::string> SetElementType(const std::string &str) {
    if (str == "face") {
      elementType = ElementType::Face;
      return true;
    } else if (str == "point") {
      elementType = ElementType::Point;
      return true;
    }

    return nonstd::make_unexpected(
        "`face` or `point` is supported for `elementType`, but `" + str +
        "` specified");
  }

#if 0
  // Some frequently used materialBindings
  nonstd::optional<Relationship> materialBinding; // rel material:binding
  nonstd::optional<Relationship> materialBindingCollection; // rel material:binding:collection
  nonstd::optional<Relationship> materialBindingPreview; // rel material:binding:preview
#endif

  TypedAttribute<Animatable<std::vector<int32_t>>> indices; // int[] indices

  std::map<std::string, Property> props;  // custom Properties
  PrimMeta meta;

  std::vector<value::token> &primChildrenNames() {
    return _primChildrenNames;
  }

  std::vector<value::token> &propertyNames() {
    return _propertyNames;
  }

  const std::vector<value::token> &primChildrenNames() const {
    return _primChildrenNames;
  }

  const std::vector<value::token> &propertyNames() const {
    return _propertyNames;
  }

  static bool ValidateSubsets(
    const std::vector<const GeomSubset *> &subsets,
    const size_t elementCount,
    const FamilyType &familyType, std::string *err);


 private:
  std::vector<value::token> _primChildrenNames;
  std::vector<value::token> _propertyNames;
};

// Polygon mesh geometry
// X11's X.h uses `None` macro, so add extra prefix to `None` enum
struct GeomMesh : GPrim {
  enum class InterpolateBoundary {
    InterpolateBoundaryNone,  // "none"
    EdgeAndCorner,            // "edgeAndCorner"
    EdgeOnly                  // "edgeOnly"
  };

  enum class FaceVaryingLinearInterpolation {
    CornersPlus1,                        // "cornersPlus1"
    CornersPlus2,                        // "cornersPlus2"
    CornersOnly,                         // "cornersOnly"
    Boundaries,                          // "boundaries"
    FaceVaryingLinearInterpolationNone,  // "none"
    All,                                 // "all"
  };

  enum class SubdivisionScheme {
    CatmullClark,           // "catmullClark"
    Loop,                   // "loop"
    Bilinear,               // "bilinear"
    SubdivisionSchemeNone,  // "none"
  };

  //
  // Predefined attribs.
  //
  TypedAttribute<Animatable<std::vector<value::point3f>>> points;  // point3f[]
  TypedAttribute<Animatable<std::vector<value::normal3f>>>
      normals;  // normal3f[] (NOTE: "primvars:normals" are stored in
                // `GPrim::props`)

  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      velocities;  // vector3f[]

  TypedAttribute<Animatable<std::vector<int32_t>>>
      faceVertexCounts;  // int[] faceVertexCounts
  TypedAttribute<Animatable<std::vector<int32_t>>>
      faceVertexIndices;  // int[] faceVertexIndices

  // Make SkelBindingAPI first citizen.
  nonstd::optional<Relationship> skeleton;  // rel skel:skeleton

  //
  // Utility functions
  //


  ///
  /// @brief Returns `points`.
  ///
  /// NOTE: No support for connected attribute. Using tydra::EvaluateTypedAttribute preferred.
  ///
  /// @param[in] time Time for TimeSampled `points` data.
  /// @param[in] interp Interpolation type for TimeSampled `points` data
  /// @return points vector(copied). Returns empty when `points` attribute is
  /// not defined.
  ///
  const std::vector<value::point3f> get_points(
      double time = value::TimeCode::Default(),
      value::TimeSampleInterpolationType interp =
          value::TimeSampleInterpolationType::Linear) const;

  ///
  /// @brief Returns normals vector. Precedence order: `primvars:normals` then
  /// `normals`.
  ///
  /// NOTE: No support for connected attribute. Using tydra::GetGeomPrimvar preferred.
  ///
  /// @return normals vector(copied). Returns empty normals vector when neither
  /// `primvars:normals` nor `normals` attribute defined, attribute is a
  /// Relationship, Connection Attribute, or normals attribute have invalid type(other than `normal3f`).
  ///
  const std::vector<value::normal3f> get_normals(
      double time = value::TimeCode::Default(),
      value::TimeSampleInterpolationType interp =
          value::TimeSampleInterpolationType::Linear) const;

  ///
  /// @brief Get interpolation of `primvars:normals`, then `normals`.
  /// @return Interpolation of normals. `vertex` by defaut.
  ///
  Interpolation get_normalsInterpolation() const;

  ///
  /// @brief Returns `faceVertexCounts`.
  ///
  /// NOTE: No support for connected attribute. Using tydra::EvaluateTypedAttribute preferred.
  ///
  /// @return face vertex counts vector(copied)
  ///
  const std::vector<int32_t> get_faceVertexCounts(double time = value::TimeCode::Default()) const;

  ///
  /// @brief Returns `faceVertexIndices`.
  ///
  /// @return face vertex indices vector(copied)
  ///
  const std::vector<int32_t> get_faceVertexIndices(double time = value::TimeCode::Default()) const;

  //
  // SubD attribs.
  //
  TypedAttribute<Animatable<std::vector<int32_t>>>
      cornerIndices;  // int[] cornerIndices
  TypedAttribute<Animatable<std::vector<float>>>
      cornerSharpnesses;  // float[] cornerSharpnesses
  TypedAttribute<Animatable<std::vector<int32_t>>>
      creaseIndices;  // int[] creaseIndices
  TypedAttribute<Animatable<std::vector<int32_t>>>
      creaseLengths;  // int[] creaseLengths
  TypedAttribute<Animatable<std::vector<float>>>
      creaseSharpnesses;  // float[] creaseSharpnesses
  TypedAttribute<Animatable<std::vector<int32_t>>>
      holeIndices;  // int[] holeIndices
  TypedAttributeWithFallback<Animatable<InterpolateBoundary>>
      interpolateBoundary{
          InterpolateBoundary::EdgeAndCorner};  // token interpolateBoundary
  TypedAttributeWithFallback<SubdivisionScheme> subdivisionScheme{
      SubdivisionScheme::CatmullClark};  // uniform token subdivisionScheme
  TypedAttributeWithFallback<Animatable<FaceVaryingLinearInterpolation>>
      faceVaryingLinearInterpolation{
          FaceVaryingLinearInterpolation::
              CornersPlus1};  // token faceVaryingLinearInterpolation

  TypedAttribute<std::vector<value::token>> blendShapes; // uniform token[] skel:blendShapes
  nonstd::optional<Relationship> blendShapeTargets; // rel skel:blendShapeTargets (Path[])

  //
  // TODO: Make these primvars first citizen?
  // - int[] primvars:skel:jointIndices
  // - float[] primvars:skel:jointWeights


  ///
  /// For GeomSubset
  ///
  /// This creates `uniform token subsetFamily:<familyName>:familyType = familyType` attribute when serialized.
  ///
  void set_subsetFamilyType(const value::token &familyName, GeomSubset::FamilyType familyType) {
    subsetFamilyTypeMap[familyName] = familyType;
  }

  ///
  /// This look ups `uniform token subsetFamily:<familyName>:familyType = familyType` attribute.
  ///
  /// @return true upon success, false when corresponding attribute not found or invalid.
  bool get_subsetFamilyType(const value::token &familyName, GeomSubset::FamilyType *familyType) {
    if (!familyType) {
      return false;
    }

    if (subsetFamilyTypeMap.count(familyName)) {
      (*familyType) = subsetFamilyTypeMap[familyName];
      return true;
    }
    return false;

  }

  ///
  /// Return the list of subet familyNames in this GeomMesh.
  ///
  /// This lists `uniform token subsetFamily:<familyName>:familyType` attributes.
  ///
  /// @return The list familyNames. Empty when no familyName attribute found.
  std::vector<value::token> get_subsetFamilyNames() {
    std::vector<value::token> toks;
    for (const auto &item : subsetFamilyTypeMap) {
      toks.push_back(item.first);
    }
    return toks;
  }


  // familyName -> familyType map
  std::map<value::token, GeomSubset::FamilyType> subsetFamilyTypeMap;

#if 0 // GeomSubset Prim is now managed as a child Prim
  //
  // GeomSubset
  //
  // uniform token `subsetFamily:materialBind:familyType`
  GeomSubset::FamilyType materialBindFamilyType{
      GeomSubset::FamilyType::Partition};

  std::vector<GeomSubset> geom_subset_children;

#endif

  // Get Explicit Joint orders: `uniform token[] skel:joints`
  std::vector<value::token> get_joints() const;

#if 0 // Deprecated: Use tydra::GetGeomSubsets() instead.
  ///
  /// Get GeomSubset list assgied to this GeomMesh(child Prim).
  ///
  /// The pointer points to the address of child Prim,
  /// so should not free it and this GeomMesh object must be valid during using the pointer to GeomSubset.
  ///
  std::vector<const GeomSubset *> GetGeomSubsets();
#endif

};

struct GeomCamera : public GPrim {
  enum class Projection {
    Perspective,   // "perspective"
    Orthographic,  // "orthographic"
  };

  enum class StereoRole {
    Mono,   // "mono"
    Left,   // "left"
    Right,  // "right"
  };

  //
  // Properties
  // 
  // NOTE: fallback value is in [mm](tenth of scene unit)
  //

  TypedAttribute<Animatable<std::vector<value::float4>>> clippingPlanes; // float4[]
  TypedAttributeWithFallback<Animatable<value::float2>> clippingRange{
      value::float2({0.1f, 1000000.0f})};
  TypedAttributeWithFallback<Animatable<float>> exposure{0.0f};  // in EV
  TypedAttributeWithFallback<Animatable<float>> focalLength{50.0f};
  TypedAttributeWithFallback<Animatable<float>> focusDistance{0.0f};
  TypedAttributeWithFallback<Animatable<float>> horizontalAperture{20.965f};
  TypedAttributeWithFallback<Animatable<float>> horizontalApertureOffset{0.0f};
  TypedAttributeWithFallback<Animatable<float>> verticalAperture{15.2908f};
  TypedAttributeWithFallback<Animatable<float>> verticalApertureOffset{0.0f};
  TypedAttributeWithFallback<Animatable<float>> fStop{
      0.0f};  // 0.0 = no focusing
  TypedAttributeWithFallback<Animatable<Projection>> projection{
      Projection::Perspective};  // "token projection" Animatable

  TypedAttributeWithFallback<StereoRole> stereoRole{
      StereoRole::Mono};  // "uniform token stereoRole"

  TypedAttributeWithFallback<Animatable<double>> shutterClose{
      0.0};  // double shutter:close
  TypedAttributeWithFallback<Animatable<double>> shutterOpen{
      0.0};  // double shutter:open
};

// struct GeomBoundable : GPrim {};

struct GeomCone : public GPrim {
  //
  // Properties
  //
  TypedAttributeWithFallback<Animatable<double>> height{2.0};
  TypedAttributeWithFallback<Animatable<double>> radius{1.0};

  TypedAttributeWithFallback<Axis> axis{Axis::Z};
};

struct GeomCapsule : public GPrim {
  //
  // Properties
  //
  TypedAttributeWithFallback<Animatable<double>> height{2.0};
  TypedAttributeWithFallback<Animatable<double>> radius{0.5};
  TypedAttributeWithFallback<Axis> axis{Axis::Z};  // uniform token axis
};

struct GeomCylinder : public GPrim {
  //
  // Properties
  //
  TypedAttributeWithFallback<Animatable<double>> height{2.0};
  TypedAttributeWithFallback<Animatable<double>> radius{1.0};
  TypedAttributeWithFallback<Axis> axis{Axis::Z};  // uniform token axis
};

struct GeomCube : public GPrim {
  //
  // Properties
  //
  TypedAttributeWithFallback<Animatable<double>> size{2.0};
};

struct GeomSphere : public GPrim {
  //
  // Predefined attribs.
  //
  TypedAttributeWithFallback<Animatable<double>> radius{2.0};
};

//
// Basis Curves(for hair/fur)
//
struct GeomBasisCurves : public GPrim {
  enum class Type {
    Cubic,   // "cubic"(default)
    Linear,  // "linear"
  };

  enum class Basis {
    Bezier,      // "bezier"(default)
    Bspline,     // "bspline"
    CatmullRom,  // "catmullRom"
  };

  enum class Wrap {
    Nonperiodic,  // "nonperiodic"(default)
    Periodic,     // "periodic"
    Pinned,       // "pinned"
  };

  TypedAttributeWithFallback<Type> type{Type::Cubic};
  TypedAttributeWithFallback<Basis> basis{Basis::Bezier};
  TypedAttributeWithFallback<Wrap> wrap{Wrap::Nonperiodic};

  //
  // Predefined attribs.
  //
  TypedAttribute<Animatable<std::vector<value::point3f>>> points;    // point3f
  TypedAttribute<Animatable<std::vector<value::normal3f>>> normals;  // normal3f
  TypedAttribute<Animatable<std::vector<int>>> curveVertexCounts;
  TypedAttribute<Animatable<std::vector<float>>> widths;
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      velocities;  // vector3f
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      accelerations;  // vector3f
};

struct GeomNurbsCurves : public GPrim {

  //
  // Predefined attribs.
  //
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      accelerations;
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      velocities;
  TypedAttribute<Animatable<std::vector<int>>>
      curveVertexCounts;
  TypedAttribute<Animatable<std::vector<value::normal3f>>>
      normals;
  TypedAttribute<Animatable<std::vector<value::point3f>>>
      points;
  TypedAttribute<Animatable<std::vector<float>>>
      widths;


  TypedAttribute<Animatable<std::vector<int>>> order;
  TypedAttribute<Animatable<std::vector<double>>> knots;
  TypedAttribute<Animatable<std::vector<value::double2>>> ranges;
  TypedAttribute<Animatable<std::vector<double>>> pointWeights;
};

//
// Points primitive.
//
struct GeomPoints : public GPrim {
  //
  // Predefined attribs.
  //
  TypedAttribute<Animatable<std::vector<value::point3f>>> points;  // point3f[]
  TypedAttribute<Animatable<std::vector<value::normal3f>>>
      normals;                                            // normal3f[]
  TypedAttribute<Animatable<std::vector<float>>> widths;  // float[]
  TypedAttribute<Animatable<std::vector<int64_t>>>
      ids;  // int64[] per-point ids.
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      velocities;  // vector3f[]
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      accelerations;  // vector3f[]
};

//
// Point instancer(TODO).
//
struct PointInstancer : public GPrim {
  nonstd::optional<Relationship> prototypes;  // rel prototypes

  TypedAttribute<Animatable<std::vector<int32_t>>>
      protoIndices;                                      // int[] protoIndices
  TypedAttribute<Animatable<std::vector<int64_t>>> ids;  // int64[] ids
  TypedAttribute<Animatable<std::vector<value::point3f>>>
      positions;  // point3f[] positions
  TypedAttribute<Animatable<std::vector<value::quath>>>
      orientations;  // quath[] orientations
  TypedAttribute<Animatable<std::vector<value::float3>>>
      scales;  // float3[] scales
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      velocities;  // vector3f[] velocities
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      accelerations;  // vector3f[] accelerations
  TypedAttribute<Animatable<std::vector<value::vector3f>>>
      angularVelocities;  // vector3f[] angularVelocities
  TypedAttribute<Animatable<std::vector<int64_t>>>
      invisibleIds;  // int64[] invisibleIds
};


// import DEFINE_TYPE_TRAIT and DEFINE_ROLE_TYPE_TRAIT
#include "define-type-trait.inc"

namespace value {

// Geom
DEFINE_TYPE_TRAIT(GPrim, kGPrim, TYPE_ID_GPRIM, 1);

DEFINE_TYPE_TRAIT(Xform, kGeomXform, TYPE_ID_GEOM_XFORM, 1);
DEFINE_TYPE_TRAIT(GeomMesh, kGeomMesh, TYPE_ID_GEOM_MESH, 1);
DEFINE_TYPE_TRAIT(GeomBasisCurves, kGeomBasisCurves, TYPE_ID_GEOM_BASIS_CURVES,
                  1);
DEFINE_TYPE_TRAIT(GeomNurbsCurves, kGeomNurbsCurves, TYPE_ID_GEOM_NURBS_CURVES,
                  1);
DEFINE_TYPE_TRAIT(GeomSphere, kGeomSphere, TYPE_ID_GEOM_SPHERE, 1);
DEFINE_TYPE_TRAIT(GeomCube, kGeomCube, TYPE_ID_GEOM_CUBE, 1);
DEFINE_TYPE_TRAIT(GeomCone, kGeomCone, TYPE_ID_GEOM_CONE, 1);
DEFINE_TYPE_TRAIT(GeomCylinder, kGeomCylinder, TYPE_ID_GEOM_CYLINDER, 1);
DEFINE_TYPE_TRAIT(GeomCapsule, kGeomCapsule, TYPE_ID_GEOM_CAPSULE, 1);
DEFINE_TYPE_TRAIT(GeomPoints, kGeomPoints, TYPE_ID_GEOM_POINTS, 1);
DEFINE_TYPE_TRAIT(GeomSubset, kGeomSubset, TYPE_ID_GEOM_GEOMSUBSET, 1);
DEFINE_TYPE_TRAIT(GeomCamera, kGeomCamera, TYPE_ID_GEOM_CAMERA, 1);
DEFINE_TYPE_TRAIT(PointInstancer, kPointInstancer, TYPE_ID_GEOM_POINT_INSTANCER,
                  1);

#undef DEFINE_TYPE_TRAIT
#undef DEFINE_ROLE_TYPE_TRAIT

}  // namespace value

// Relation is supported as geomprimvar.
// example:
//
// rel primvar:myrel = [</a>, </b>]
//

// NOTE: `bool` type seems not supported on pxrUSD
// NOTE: `string` type need special treatment when `idFrom` Relationship exists( https://github.com/syoyo/tinyusdz/issues/113 )
#define APPLY_GEOMPRIVAR_TYPE(__FUNC) \
  __FUNC(bool)                        \
  __FUNC(std::string)                 \
  __FUNC(value::half)                 \
  __FUNC(value::half2)                \
  __FUNC(value::half3)                \
  __FUNC(value::half4)                \
  __FUNC(int)                         \
  __FUNC(value::int2)                 \
  __FUNC(value::int3)                 \
  __FUNC(value::int4)                 \
  __FUNC(uint32_t)                    \
  __FUNC(value::uint2)                \
  __FUNC(value::uint3)                \
  __FUNC(value::uint4)                \
  __FUNC(float)                       \
  __FUNC(value::float2)               \
  __FUNC(value::float3)               \
  __FUNC(value::float4)               \
  __FUNC(double)                      \
  __FUNC(value::double2)              \
  __FUNC(value::double3)              \
  __FUNC(value::double4)              \
  __FUNC(value::matrix2d)             \
  __FUNC(value::matrix3d)             \
  __FUNC(value::matrix4d)             \
  __FUNC(value::quath)                \
  __FUNC(value::quatf)                \
  __FUNC(value::quatd)                \
  __FUNC(value::normal3h)             \
  __FUNC(value::normal3f)             \
  __FUNC(value::normal3d)             \
  __FUNC(value::vector3h)             \
  __FUNC(value::vector3f)             \
  __FUNC(value::vector3d)             \
  __FUNC(value::point3h)              \
  __FUNC(value::point3f)              \
  __FUNC(value::point3d)              \
  __FUNC(value::color3f)              \
  __FUNC(value::color3d)              \
  __FUNC(value::color4f)              \
  __FUNC(value::color4d)              \
  __FUNC(value::texcoord2h)           \
  __FUNC(value::texcoord2f)           \
  __FUNC(value::texcoord2d)           \
  __FUNC(value::texcoord3h)           \
  __FUNC(value::texcoord3f)           \
  __FUNC(value::texcoord3d)

// TODO: 64bit int/uint seems not supported on pxrUSD. Enable it in TinyUSDZ?
#if 0
  __FUNC(int64_t) \
  __FUNC(uint64_t)
#endif

#define EXTERN_TEMPLATE_GET_VALUE(__ty) \
  extern template bool GeomPrimvar::get_value(__ty *dest, std::string *err) const; \
  extern template bool GeomPrimvar::get_value(double, __ty *dest, value::TimeSampleInterpolationType, std::string *err) const; \
  extern template bool GeomPrimvar::get_value(std::vector<__ty> *dest, std::string *err) const; \
  extern template bool GeomPrimvar::get_value(double, std::vector<__ty> *dest, value::TimeSampleInterpolationType, std::string *err) const; \
  extern template bool GeomPrimvar::flatten_with_indices(std::vector<__ty> *dest, std::string *err) const; \
  extern template bool GeomPrimvar::flatten_with_indices(double, std::vector<__ty> *dest, value::TimeSampleInterpolationType, std::string *err) const;

APPLY_GEOMPRIVAR_TYPE(EXTERN_TEMPLATE_GET_VALUE)

#undef EXTERN_TEMPLATE_GET_VALUE

//#undef APPLY_GEOMPRIVAR_TYPE


}  // namespace tinyusdz

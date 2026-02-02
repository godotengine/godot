// SPDX-License-Identifier: Apache 2.0
#pragma once

#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(TINYUSDZ_ENABLE_THREAD)
#include <mutex>
#include <thread>
#endif

//
#include "value-types.hh"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "nonstd/expected.hpp"
#include "nonstd/optional.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "handle-allocator.hh"
#include "primvar.hh"
//
#include "value-eval-util.hh"
#include "math-util.inc"

namespace tinyusdz {

// Simple Python-like OrderedDict
template <typename T>
class ordered_dict {
 public:

  bool at(const size_t idx, T *dst) const {
    if (idx >= _keys.size()) {
      return false;
    }

    if (!_m.count(_keys[idx])) {
      // This should not happen though.
      return false;
    }

    (*dst) = _m.at(_keys[idx]);

    return true;
  }

  bool at(const size_t idx, const T **dst) const {
    if (idx >= _keys.size()) {
      return false;
    }

    if (!_m.count(_keys[idx])) {
      // This should not happen though.
      return false;
    }

    (*dst) = &(_m.at(_keys[idx]));

    return true;
  }

  bool count(const std::string &key) const {
    return _m.count(key) > 0;
  }

  void insert(const std::string &key, const T &value) {
    if (_m.count(key)) {
      // overwrite existing value
    } else {
      _keys.push_back(key);
    }

    _m[key] = value;
  }

  T &get_or_add(const std::string &key) {
    if (!_m.count(key)) {
      _keys.push_back(key);
    }

    return _m[key];
  }


  void insert(const std::string &key, T &&value) {
    if (_m.count(key)) {
      // overwrite existing value
    } else {
      _keys.push_back(key);
    }

    _m[key] = std::move(value);
  }

  bool erase(const std::string &key) {

    if (!_m.count(key)) {
      return false;
    }

    // linear search
    bool erased = false;
    size_t idx = 0;
    for (size_t i = 0; i < _keys.size(); i++) {
      if (key == _keys[i]) {
        idx = i;
        erased = true;
      }
    }

    if (!erased) {
      return false;
    }

    _keys.erase(_keys.begin() + std::ptrdiff_t(idx));
    _m.erase(key);

    return true;
  }

  bool at(const std::string &key, T **dst) {
    if (!_m.count(key)) {
      // This should not happen though.
      return false;
    }

    (*dst) = &_m.at(key);

    return true;
  }


  bool at(const std::string &key, const T *dst) const {
    if (!_m.count(key)) {
      // This should not happen though.
      return false;
    }

    (*dst) = _m.at(key);

    return true;
  }

  bool at(const std::string &key, const T **dst) const {
    if (!_m.count(key)) {
      // This should not happen though.
      return false;
    }

    (*dst) = &_m.at(key);

    return true;
  }

  const std::vector<std::string> &keys() const {
    return _keys;
  }

  size_t size() const { return _m.size(); }

  // No operator[] for safety.

 private:
  std::vector<std::string> _keys;
  std::map<std::string, T> _m;
};

// SpecType enum must be same order with pxrUSD's SdfSpecType(since enum value
// is stored in Crate directly)
enum class SpecType {
  Unknown = 0,  // must be 0
  Attribute,
  Connection,
  Expression,
  Mapper,
  MapperArg,
  Prim,
  PseudoRoot,
  Relationship,
  RelationshipTarget,
  Variant,
  VariantSet,
  Invalid,  // or NumSpecTypes
};

enum class Orientation {
  RightHanded,  // 0
  LeftHanded,
  Invalid
};

enum class Visibility {
  Inherited,  // "inherited" (default)
  Invisible,  // "invisible"
  Invalid
};

enum class Purpose {
  Default,  // 0
  Render,   // "render"
  Proxy,    // "proxy"
  Guide,    // "guide"
};

//
// USDZ extension: sceneLibrary
// https://developer.apple.com/documentation/arkit/usdz_schemas_for_ar/scenelibrary
//

enum class Kind {
  Model,
  Group,
  Assembly,
  Component,
  Subcomponent,
  SceneLibrary,  // USDZ extension
  UserDef, // Unknown or user defined Kind
  Invalid
};

// Attribute interpolation
enum class Interpolation {
  Constant,     // "constant"
  Uniform,      // "uniform"
  Varying,      // "varying"
  Vertex,       // "vertex"
  FaceVarying,  // "faceVarying"
  Invalid
};

// NOTE: Attribute cannot have ListEdit qualifier
enum class ListEditQual {
  ResetToExplicit,  // "unqualified"(no qualifier)
  Append,           // "append"
  Add,              // "add"
  Delete,           // "delete"
  Prepend,          // "prepend"
  Order,            // "order"
  Invalid
};

enum class Axis { X, Y, Z, Invalid };

// metrics(UsdGeomLinearUnits in pxrUSD)
// To avoid linkage error, defined as static constexpr function.
struct Units {
  static constexpr double Nanometers = 1e-9;
  static constexpr double Micrometers = 1e-6;
  static constexpr double Millimeters = 0.001;
  static constexpr double Centimeters = 0.01;
  static constexpr double Meters = 1.0;
  static constexpr double Kilometers = 1000;
  static constexpr double LightYears = 9.4607304725808e15;
  static constexpr double Inches = 0.0254;
  static constexpr double Feet = 0.3048;
  static constexpr double Yards = 0.9144;
  static constexpr double Miles = 1609.344;
};

// For PrimSpec
enum class Specifier {
  Def,  // 0
  Over,
  Class,
  Invalid
};

enum class Permission {
  Public,  // 0
  Private,
  Invalid
};

enum class Variability {
  Varying,  // 0
  Uniform,
  Config,
  Invalid
};

// Return false when invalid character(e.g. '%') exists in a given string.
// This function only validates `elementName` of a Prim(e.g. "dora", "xform1").
// If you want to validate a Prim path(e.g. "/root/xform1"),
// Use ValidatePrimPath() in path-util.hh
bool ValidatePrimElementName(const std::string &tok);

///
/// Simlar to SdfPath.
/// NOTE: We are doging refactoring of Path class, so the following comment may
/// not be correct.
///
/// We don't need the performance for USDZ, so use naiive implementation
/// to represent Path.
/// Path is something like Unix path, delimited by `/`, ':' and '.'
/// Square brackets('<', '>' is not included)
///
/// Root path is represented as prim path "/" and elementPath ""(empty).
///
/// Example:
///
/// - `/muda/bora.dora` : prim_part is `/muda/bora`, prop_part is `.dora`.
/// - `bora` : Could be Element(leaf) path or Relative path
///
/// ':' is a namespce delimiter(example `input:muda`).
///
/// Limitations:
///
/// - Relational attribute path(`[` `]`. e.g. `/muda/bora[/ari].dora`) is not
/// supported.
/// - Variant chars('{' '}') is not supported(yet).
/// - Relative path(e.g. '../') is not yet supported(TODO)
///
/// and have more limitatons.
///
class Path {
 public:
  // Similar to SdfPathNode
  enum class PathType {
    Prim,
    PrimProperty,
    RelationalAttribute,
    MapperArg,
    Target,
    Mapper,
    PrimVariantSelection,
    Expression,
    Root,
  };

  Path() : _valid(false) {}

  static Path make_root_path() {
    Path p = Path("/", "");
    // elementPath is empty for root.
    p._element = "";
    p._valid = true;
    return p;
  }

  // Create Path both from Prim Path and Prop
  // If `prim` starts
  // "/aaa", "bora" => /aaa.bora
  // "/aaa", "" => /aaa (prim only)
  // "", "bora" => .bora (property only)
  //
  // Note: This constructor may fail to extract elementName from given `prim`
  // and `prop`. It is highly recommended to use AppendPrim() and AppendProperty
  // to. construct Path hierarchy(e.g. `/aaa/xform/geom.points`) so that
  // elementName is set correctly.
  Path(const std::string &prim, const std::string &prop);

  // : prim_part(prim), valid(true) {}
  // Path(const std::string &prim, const std::string &prop)
  //    : prim_part(prim), prop_part(prop) {}

  Path(const Path &rhs) = default;

  Path &operator=(const Path &rhs) {
    this->_valid = rhs._valid;

    this->_prim_part = rhs._prim_part;
    this->_prop_part = rhs._prop_part;
    this->_element = rhs._element;

    return (*this);
  }

  std::string full_path_name() const {
    std::string s;
    if (!_valid) {
      s += "#INVALID#";
    }

    s += _prim_part;
    if (_prop_part.empty()) {
      return s;
    }

    s += "." + _prop_part;

    return s;
  }

  const std::string &prim_part() const { return _prim_part; }
  const std::string &prop_part() const { return _prop_part; }

  const std::string &variant_part() const {
    _variant_part_str =
        "{" + _variant_part + "=" + _variant_selection_part + "}";
    return _variant_part_str;
  }

  void set_path_type(const PathType ty) { _path_type = ty; }

  bool get_path_type(PathType &ty) {
    if (_path_type) {
      ty = _path_type.value();
    }
    return false;
  }

  // IsPropertyPath: PrimProperty or RelationalAttribute
  bool is_property_path() const {
    if (_path_type) {
      if ((_path_type.value() == PathType::PrimProperty ||
           (_path_type.value() == PathType::RelationalAttribute))) {
        return true;
      }
    }

    // TODO: RelationalAttribute
    if (_prim_part.empty()) {
      return false;
    }

    if (_prop_part.size()) {
      return true;
    }

    return false;
  }

  // Is Prim path?
  bool is_prim_path() const {
    if (_prop_part.size()) {
      return false;
    }

    if (_prim_part.size()) {
      return true;
    }

    return false;
  }

  // Is Prim's property path?
  // True when both PrimPart and PropPart are not empty.
  bool is_prim_property_path() const {
    if (_prim_part.empty()) {
      return false;
    }
    if (_prop_part.size()) {
      return true;
    }
    return false;
  }

  bool is_valid() const { return _valid; }

  bool is_empty() {
    return (_prim_part.empty() && _variant_part.empty() && _prop_part.empty());
  }

  // static Path RelativePath() { return Path("."); }

  // Append property path(change internal state)
  Path append_property(const std::string &elem);

  // Append prim or variantSelection path(change internal state)
  Path append_element(const std::string &elem);
  Path append_prim(const std::string &elem) {
    return append_element(elem);
  }  // for legacy

  // Const version. Does not change internal state.
  const Path AppendProperty(const std::string &elem) const;
  const Path AppendPrim(const std::string &elem) const;
  const Path AppendElement(const std::string &elem) const;

  // Get element name(the last element of Path. i.e. Prim's name, Property's
  // name)
  const std::string &element_name() const;

  ///
  /// Split a path to the root(common ancestor) and its siblings
  ///
  /// example:
  ///
  /// - / -> [/, Empty]
  /// - /bora -> [/bora, Empty]
  /// - /bora/dora -> [/bora, /dora]
  /// - /bora/dora/muda -> [/bora, /dora/muda]
  /// - bora -> [Empty, bora]
  /// - .muda -> [Empty, .muda]
  ///
  std::pair<Path, Path> split_at_root() const;

  ///
  /// TODO: Deprecate(use get_parent_path() instead)
  ///
  /// Get parent Prim path.
  /// If the given path is a root Prim path(e.g. "/bora"), same Path is
  /// returned.
  ///
  /// example:
  ///
  /// - / -> invalid Path
  /// - /bora -> /bora
  /// - /bora/dora -> /bora
  /// - /bora/dora.prop -> /bora/dora
  /// - dora/bora -> dora
  /// - dora -> invalid Path
  /// - .dora -> invalid Path(path is property path)
  Path get_parent_prim_path() const;

  ///
  /// Get parent Path.
  /// If the given path is the root path("/") same Path is returned.
  ///
  /// example:
  ///
  /// - / -> invalid Path
  /// - /bora -> /
  /// - /bora/dora -> /bora
  /// - /bora/dora.prop -> /bora/dora
  /// - dora/bora -> dora
  /// - dora -> invalid Path
  /// - .dora -> invalid Path(path is property path)
  Path get_parent_path() const;

  ///
  /// Check if this Path has same prefix for given Path
  ///
  /// example.
  /// rhs path: /bora/dora
  ///
  /// /bora/dora/muda -> true
  /// /bora/dora2 -> fase
  ///
  /// If the prefix path contains prop part, compare it with ==
  /// (assume no hierarchy in property part)
  ///
  bool has_prefix(const Path &rhs) const;

  ///
  /// Replace Prim path prefix.
  /// example.
  /// srcPrefix = /bora/dora
  /// dstPrefix = /bora2/dora2
  /// 
  /// /bora/dora/muda -> /bora2/dora2/muda 
  ///
  bool replace_prefix(const Path &srcPrefix, const Path &dstPrefix);

  ///
  /// @returns true if a path is '/' only
  ///
  bool is_root_path() const {
    if (!_valid) {
      return false;
    }

    if ((_prim_part.size() == 1) && (_prim_part[0] == '/')) {
      return true;
    }

    return false;
  }

  ///
  /// @returns true if a path is root prim: e.g. '/bora'
  ///
  bool is_root_prim() const {
    if (!_valid) {
      return false;
    }

    if (is_root_path()) {
      return false;
    }

    if ((_prim_part.size() > 1) && (_prim_part[0] == '/')) {
      // no other '/' except for the fist one
      if (_prim_part.find_last_of('/') == 0) {
        return true;
      }
    }

    return false;
  }

  bool is_absolute_path() const {
    if (_prim_part.size() && _prim_part[0] == '/') {
      return true;
    }

    return false;
  }

  bool is_relative_path() const {
    if (_prim_part.size()) {
      return !is_absolute_path();
    }

    return true;  // prop part only
  }

#if 0 // TODO: rmove
  bool is_variant_selection_path() const {
    if (!is_valid()) {
      return false;
    }

    if (_variant_part.size()) {
      return true;
    }

    return false;
  }
#endif

  // Strip '/'
  Path &make_relative() {
    if (is_absolute_path() && (_prim_part.size() > 1)) {
      // Remove first '/'
      _prim_part.erase(0, 1);
    }
    return *this;
  }

  const Path make_relative(Path &&rhs) {
    (*this) = std::move(rhs);

    return make_relative();
  }

  static const Path make_relative(const Path &rhs) {
    Path p = rhs;  // copy
    return p.make_relative();
  }

  static bool LessThan(const Path &lhs, const Path &rhs);

  // To sort paths lexicographically.
  // TODO: consider abs and relative path correctly
  bool operator<(const Path &rhs) const {
    if (full_path_name() == rhs.full_path_name()) {
      return false;
    }

    if (prim_part().empty() || rhs.prim_part().empty()) {
      return prim_part().empty() && rhs.prim_part().size();
    }

    return LessThan(*this, rhs);
  }

 private:
  void _update(const std::string &p, const std::string &prop);

  std::string _prim_part;     // e.g. /Model/MyMesh, MySphere
  std::string _prop_part;     // e.g. visibility (`.` is not included)
  std::string _variant_part;  // e.g. `variantColor` for {variantColor=green}
  std::string _variant_selection_part;  // e.g. `green` for {variantColor=green}
                                        // . Could be empty({variantColor=}).
  mutable std::string _variant_part_str;  // str buffer for variant_part()
  mutable std::string _element;           // Element name

  nonstd::optional<PathType> _path_type;  // Currently optional.

  bool _valid{false};
};

#if 0
///
/// Split Path by the delimiter(e.g. "/") then create lists.
///
class TokenizedPath {
 public:
  TokenizedPath() {}

  TokenizedPath(const Path &path) {
    std::string s = path.prop_part();
    if (s.empty()) {
      // ???
      return;
    }

    if (s[0] != '/') {
      // Path must start with "/"
      return;
    }

    s.erase(0, 1);

    char delimiter = '/';
    size_t pos{0};
    while ((pos = s.find(delimiter)) != std::string::npos) {
      std::string token = s.substr(0, pos);
      _tokens.push_back(token);
      s.erase(0, pos + sizeof(char));
    }

    if (!s.empty()) {
      // leaf element
      _tokens.push_back(s);
    }
  }

 private:
  std::vector<std::string> _tokens;
};
#endif

bool operator==(const Path &lhs, const Path &rhs);

// variants in Prim Meta.
//
// e.g.
// variants = {
//   string variant0 = "bora"
//   string variant1 = "dora"
// }
// pxrUSD uses dict type for the content, but TinyUSDZ only accepts list of
// strings for now
//
using VariantSelectionMap = std::map<std::string, std::string>;

class MetaVariable;

// TODO: Use `Dictionary` and deprecate CustomDataType
using CustomDataType = std::map<std::string, MetaVariable>;

using Dictionary = CustomDataType;  // alias to CustomDataType

///
/// Helper function to access CustomData(dictionary).
/// Recursively process into subdictionaries when a key contains namespaces(':')
///
bool HasCustomDataKey(const Dictionary &customData, const std::string &key);
bool GetCustomDataByKey(const Dictionary &customData, const std::string &key,
                        /* out */ MetaVariable *dst);
bool SetCustomDataByKey(const std::string &key, const MetaVariable &val,
                        /* inout */ Dictionary &customData);

void OverrideDictionary(Dictionary &customData, const Dictionary &src, const bool override_existing = true);

// Variable class for Prim and Attribute Metadataum.
//
// - Accepts limited number of types for value
// - No 'custom' keyword
// - 'None'(Value Block) is supported for some type(at least `references` and
// `payload` accepts None)
// - No TimeSamples, No Connection, No Relationship(`rel`)
// - Value must be assigned(e.g. "float myval = 1.3"). So no definition only
// syntax("float myval")
// - Can be string only(no type information)
//   - Its variable name is interpreted as "comment"
//
class MetaVariable {
 public:
  MetaVariable &operator=(const MetaVariable &rhs) {
    _name = rhs._name;
    _value = rhs._value;

    return *this;
  }

  template <typename T>
  MetaVariable(const T &v) {
    set_value(v);
  }

  MetaVariable(const MetaVariable &rhs) {
    _name = rhs._name;
    _value = rhs._value;
  }

  template <typename T>
  MetaVariable(const std::string &name, const T &v) {
    set_value(name, v);
  }

  // template <typename T>
  // bool is() const {
  //   return value.index() == ValueType::index_of<T>();
  // }

  bool is_valid() const {
    return _value.type_id() != value::TypeTraits<std::nullptr_t>::type_id();
  }

  //// TODO
  // bool is_timesamples() const { return false; }

  MetaVariable() = default;

  //
  // custom data must have some value, so no set_type()
  // OK "float myval = 1"
  // NG "float myval"
  //
  template <typename T>
  void set_value(const T &v) {
    // TODO: Check T is supported type for Metadatum.
    _value = v;

    _name = std::string();  // empty
  }

  template <typename T>
  void set_value(const std::string &name, const T &v) {
    // TODO: Check T is supported type for Metadatum.
    _value = v;

    _name = name;
  }

  template <typename T>
  bool get_value(T *dst) const {
    if (!dst) {
      return false;
    }

    if (const T *v = _value.as<T>()) {
      (*dst) = *v;
      return true;
    }

    return false;
  }

  template <typename T>
  nonstd::optional<T> get_value() const {
    if (const T *v = _value.as<T>()) {
      return *v;
    }

    return nonstd::nullopt;
  }

  void set_name(const std::string &name) { _name = name; }
  const std::string &get_name() const { return _name; }

  const value::Value &get_raw_value() const { return _value; }
  value::Value &get_raw_value() { return _value; }

  // No set_type_name()
  const std::string type_name() const { return TypeName(*this); }

  uint32_t type_id() const { return TypeId(*this); }

  bool is_blocked() const {
    return type_id() == value::TypeId::TYPE_ID_VALUEBLOCK;
  }

 private:
  static std::string TypeName(const MetaVariable &v) {
    return v._value.type_name();
  }

  static uint32_t TypeId(const MetaVariable &v) { return v._value.type_id(); }

 private:
  value::Value _value{nullptr};
  std::string _name;
};

struct AssetInfo {
  // builtin fields
  value::AssetPath identifier;
  std::string name;
  std::vector<value::AssetPath> payloadAssetDependencies;
  std::string version;

  // Other fields
  Dictionary _fields;
};


// USDZ AR class?
// Preliminary_Trigger,
// Preliminary_PhysicsGravitationalForce,
// Preliminary_InfiniteColliderPlane,
// Preliminary_ReferenceImage,
// Preliminary_Action,
// Preliminary_Text,

struct APISchemas {
  // TinyUSDZ does not allow user-supplied API schema for now
  enum class APIName {
    // usdShade
    MaterialBindingAPI,  // "MaterialBindingAPI"
    ConnectableAPI, // "ConnectableAPI"
    CoordSysAPI, // "CoordSysAPI"
    NodeDefAPI, // "NodeDefAPI"

    CollectionAPI,      // "CollectionAPI"
    // usdGeom
    GeomModelAPI, // "GeomModelAPI"
    MotionAPI, // "MotionAPI"
    PrimvarsAPI, // "PrimvarsAPI"
    VisibilityAPI, // "VisibilityAPI"
    XformCommonAPI, // "XformCommonAPI"

    // usdLux
    LightAPI, // "LightAPI"
    LightListAPI, // "LightListAPI"
    ListAPI, // "ListAPI"
    MeshLightAPI, // "MeshLightAPI"
    ShapingAPI, // "ShapingAPI"
    ShadowAPI,  // "ShadowAPI"
    VolumeLightAPI,  // "VolumeLightAPI"

    // usdSkel
    SkelBindingAPI,      // "SkelBindingAPI"
    
    // USDZ AR extensions
    Preliminary_AnchoringAPI,
    Preliminary_PhysicsColliderAPI,
    Preliminary_PhysicsMaterialAPI,
    Preliminary_PhysicsRigidBodyAPI,
  };

  ListEditQual listOpQual{ListEditQual::ResetToExplicit};  // must be 'prepend'

  // std::get<1>: instance name. For Multi-apply API Schema e.g.
  // `material:MainMaterial` for `CollectionAPI:material:MainMaterial`
  std::vector<std::pair<APIName, std::string>> names;
};

// SdfLayerOffset
struct LayerOffset {
  double _offset{0.0};
  double _scale{1.0};
};

// SdfReference
struct Reference {
  value::AssetPath asset_path;
  Path prim_path;
  LayerOffset layerOffset;
  Dictionary customData;
};

// SdfPayload
struct Payload {
  value::AssetPath asset_path;  // std::string in SdfPayload
  Path prim_path;
  LayerOffset layerOffset;  // from 0.8.0
  // No customData for Payload

  // NOTE: pxrUSD encodes `payload = None` as Payload with empty paths in USDC(Crate).
  // (Not ValueBlock)
  bool is_none() const {
    return asset_path.GetAssetPath().empty() && !prim_path.is_valid();
  }
};

// Metadata for Prim
struct PrimMetas {
  nonstd::optional<bool> active;  // 'active'
  nonstd::optional<bool> hidden;  // 'hidden'
  nonstd::optional<Kind> kind;    // 'kind'. user-defined kind value is stored in _kind_str;
  std::string _kind_str;

  nonstd::optional<Dictionary>
      assetInfo;  // 'assetInfo' // TODO: Use AssetInfo?
  nonstd::optional<Dictionary> customData;  // `customData`
  nonstd::optional<value::StringData> doc;  // 'documentation'
  nonstd::optional<value::StringData>
      comment;  // 'comment'  (String only metadata value)
  nonstd::optional<APISchemas> apiSchemas;  // 'apiSchemas'
  nonstd::optional<Dictionary>
      sdrMetadata;  // 'sdrMetadata' (usdShade Prim only?)

  nonstd::optional<bool> instanceable; // 'instanceable'
  nonstd::optional<Dictionary> clips; // 'clips'

  // String representation of Kind.
  // For user-defined Kind, it returns `_kind_str`
  const std::string get_kind() const;

  //
  // AssetInfo utility function
  //
  // Convert CustomDataType to AssetInfo
  AssetInfo get_assetInfo(bool *authored = nullptr) const;

  //
  // Compositions
  //
  nonstd::optional<std::pair<ListEditQual, std::vector<Reference>>> references;
  nonstd::optional<std::pair<ListEditQual, std::vector<Payload>>>
      payload;  // NOTE: not `payloads`
  nonstd::optional<std::pair<ListEditQual, std::vector<Path>>>
      inherits;  // 'inherits'
  nonstd::optional<std::pair<ListEditQual, std::vector<std::string>>>
      variantSets;  // 'variantSets'. Could be `token` but treat as
                    // `string`(Crate format uses `string`)

  nonstd::optional<VariantSelectionMap> variants;  // `variants`

  nonstd::optional<std::pair<ListEditQual, std::vector<Path>>>
      specializes;  // 'specializes'

  // USDZ extensions
  nonstd::optional<std::string> sceneName;  // 'sceneName'

  // Omniverse extensions(TODO: Use UTF8 string type?)
  // https://github.com/PixarAnimationStudios/USD/pull/2055
  nonstd::optional<std::string> displayName;  // 'displayName'

  // Unregistered metadatum. value is represented as string.
  std::map<std::string, std::string> unregisteredMetas;

  Dictionary meta;  // other non-buitin meta values. TODO: remove this variable
                    // and use `customData` instead, since pxrUSD does not allow
                    // non-builtin Prim metadatum

  ///
  /// Update metadatum with rhs(authored metadataum only)
  ///
  /// @param[in] override_authored true: override this.metadataum(authored or not-authored) when rhs.metadatum is authoerd, false override only when this.metadatum is not authored and rhs.metadataum is authored.
  ///
  void update_from(const PrimMetas &rhs, bool override_authored = true);


#if 0
  // String only metadataum.
  // TODO: Represent as `MetaVariable`?
  std::vector<value::StringData> stringData;
#endif

  // FIXME: Find a better way to detect Prim meta is authored...
  bool authored() const {
    return (active || hidden || kind || customData || references || payload ||
            inherits || variants || variantSets || specializes || displayName ||
            sceneName || doc || comment || unregisteredMetas.size() || meta.size() || apiSchemas ||
            sdrMetadata || assetInfo || instanceable || clips);
  }

  //
  // Infos used indirectly.
  //

  // Used to display/traverse Prim items based on this array
  // USDA: By appearance. USDC: "primChildren" TokenVector field
  std::vector<value::token> primChildren;

  // Used to display/traverse Property items based on this array
  // USDA: By appearance. USDC: "properties" TokenVector field
  std::vector<value::token> properties;

  nonstd::optional<std::pair<ListEditQual, std::vector<Path>>> inheritPaths;

  nonstd::optional<std::vector<value::token>> variantChildren;
  nonstd::optional<std::vector<value::token>> variantSetChildren;
};

// For backward compatibility
using PrimMeta = PrimMetas;

// Metadata for Property(Relationship and Attribute)
// TODO: Rename to PropMetas
struct AttrMetas {
  // frequently used items
  // nullopt = not specified in USD data
  nonstd::optional<Interpolation> interpolation;  // 'interpolation'
  nonstd::optional<uint32_t> elementSize;         // usdSkel 'elementSize'
  nonstd::optional<bool> hidden;                  // 'hidden'
  nonstd::optional<value::StringData> comment;    // `comment`
  nonstd::optional<Dictionary> customData;        // `customData`

  nonstd::optional<double> weight;  // usdSkel inbetween BlendShape weight.

  // usdShade
  nonstd::optional<value::token> connectability; // NOTE: applies to attr
  nonstd::optional<value::token> outputName; // NOTE: applies to rel
  nonstd::optional<value::token> renderType; // NOTE: applies to prop
  nonstd::optional<Dictionary> sdrMetadata; // NOTE: applies to attr(also seen in prim meta)

  nonstd::optional<std::string> displayName;  // 'displayName'
  nonstd::optional<std::string> displayGroup;  // 'displayGroup'


  //
  // MaterialBinding
  //
  // Could be arbitrary token value so use `token[]` type.
  // For now, either `weakerThanDescendants` or `strongerThanDescendants` are
  // valid token.
  nonstd::optional<value::token> bindMaterialAs;  // 'bindMaterialAs' NOTE: applies to rel.

  std::map<std::string, MetaVariable> meta;  // other meta values

  // String only metadataum.
  // TODO: Represent as `MetaVariable`?
  std::vector<value::StringData> stringData;


  //
  // Some handy methods for non-frequently used metadatum.
  //
  bool has_colorSpace() const;
  value::token get_colorSpace() const; // return empty when not authored or 'colorSpace' metadataum is not token type.

  bool has_unauthoredValuesIndex() const;
  int get_unauthoredValuesIndex() const; // return -1 when not authored or 'unauthoredValuesIndex' metadataum is not int type.

  bool authored() const {
    return (interpolation || elementSize || hidden || customData || weight ||
            connectability || outputName || renderType || sdrMetadata || displayName || displayGroup || bindMaterialAs || meta.size() || stringData.size());
  }
};

// For backward compatibility
using AttrMeta = AttrMetas;

using PropMetas = AttrMetas;

// TODO: Move to value-types.hh?
//
// Typed TimeSamples value
//
// double radius.timeSamples = { 0: 1.0, 1: None, 2: 3.0 }
//
// in .usd, are represented as
//
// 0: (1.0, false)
// 1: (2.0, true)
// 2: (3.0, false)
//

template <typename T>
struct TypedTimeSamples {
 public:
  struct Sample {
    double t;
    T value;
    bool blocked{false};
  };

  bool empty() const { return _samples.empty(); }

  void update() const {
    std::sort(_samples.begin(), _samples.end(),
              [](const Sample &a, const Sample &b) { return a.t < b.t; });

    _dirty = false;

    return;
  }

  // Get value at specified time.
  // For non-interpolatable types(includes enums and unknown types)
  //
  // Return `Held` value even when TimeSampleInterpolationType is
  // Linear. Returns nullopt when specified time is out-of-range.
  template<typename V = T, std::enable_if_t<!value::LerpTraits<V>::supported(), std::nullptr_t> = nullptr>
  bool get(T *dst, double t = value::TimeCode::Default(),
           value::TimeSampleInterpolationType interp =
               value::TimeSampleInterpolationType::Linear) const {

    (void)interp;

    if (!dst) {
      return false;
    }

    if (empty()) {
      return false;
    }

    if (_dirty) {
      update();
    }

    if (value::TimeCode(t).is_default()) {
      // FIXME: Use the first item for now.
      // TODO: Handle bloked
      (*dst) = _samples[0].value;
      return true;
    } else {

      if (_samples.size() == 1) {
        (*dst) = _samples[0].value;
        return true;
      }

      // Held = nerarest preceding value for a gien time.
      // example:
      // input = 0.0: 100, 1.0: 200
      //
      // t -1.0 => 100(time 0.0)
      // t 0.0 => 100(time 0.0)
      // t 0.1 => 100(time 0.0)
      // t 0.9 => 100(time 0.0)
      // t 1.0 => 200(time 1.0)
      //
      // This can be achieved by using upper_bound, and subtract 1 from the found position.
      auto it = std::upper_bound(
        _samples.begin(), _samples.end(), t,
        [](double tval, const Sample &a) { return tval < a.t; });

      const auto it_minus_1 = (it == _samples.begin()) ? _samples.begin() : (it - 1);

      (*dst) = it_minus_1->value;
      return true;
    }

  }

  // TODO: Move to .cc to save compile time.
  // Get value at specified time.
  // Return linearly interpolated value when TimeSampleInterpolationType is
  // Linear. Returns nullopt when specified time is out-of-range.
  template<typename V = T, std::enable_if_t<value::LerpTraits<V>::supported(), std::nullptr_t> = nullptr>
  bool get(T *dst, double t = value::TimeCode::Default(),
           value::TimeSampleInterpolationType interp =
               value::TimeSampleInterpolationType::Linear) const {
    if (!dst) {
      return false;
    }

    if (empty()) {
      return false;
    }

    if (_dirty) {
      update();
    }

    if (value::TimeCode(t).is_default()) {
      // FIXME: Use the first item for now.
      // TODO: Handle bloked
      (*dst) = _samples[0].value;
      return true;
    } else {

      if (_samples.size() == 1) {
        (*dst) = _samples[0].value;
        return true;
      }

      auto it = std::lower_bound(
        _samples.begin(), _samples.end(), t,
        [](const Sample &a, double tval) { return a.t < tval; });

      if (interp == value::TimeSampleInterpolationType::Linear) {

        // MS STL does not allow seek vector iterator before begin
        // Issue #110
        const auto it_minus_1 = (it == _samples.begin()) ? _samples.begin() : (it - 1);

        size_t idx0 = size_t((std::max)(
            int64_t(0),
            (std::min)(int64_t(_samples.size() - 1),
                     int64_t(std::distance(_samples.begin(), it_minus_1)))));
        size_t idx1 =
            size_t((std::max)(int64_t(0), (std::min)(int64_t(_samples.size() - 1),
                                                 int64_t(idx0) + 1)));

        double tl = _samples[idx0].t;
        double tu = _samples[idx1].t;

        double dt = (t - tl);
        if (std::fabs(tu - tl) < std::numeric_limits<double>::epsilon()) {
          // slope is zero.
          dt = 0.0;
        } else {
          dt /= (tu - tl);
        }

        // Just in case.
        dt = (std::max)(0.0, (std::min)(1.0, dt));

        const value::Value &pv0 = _samples[idx0].value;
        const value::Value &pv1 = _samples[idx1].value;

        if (pv0.type_id() != pv1.type_id()) {
          // Type mismatch.
          return false;
        }

        // To concrete type
        const T *p0 = pv0.as<T>();
        const T *p1 = pv1.as<T>();

        if (!p0 || !p1) {
          return false;
        }

        const T p = lerp(*p0, *p1, dt);

        (*dst) = std::move(p);
        return true;
      } else {
        if (it == _samples.end()) {
          // ???
          return false;
        }

        (*dst) = it->value;
        return true;
      }
    }

    return false;
  }

  void add_sample(const Sample &s) {
    _samples.push_back(s);
    _dirty = true;
  }

  void add_sample(const double t, const T &v) {
    Sample s;
    s.t = t;
    s.value = v;
    _samples.emplace_back(s);
    _dirty = true;
  }

  void add_blocked_sample(const double t) {
    Sample s;
    s.t = t;
    s.blocked = true;
    _samples.emplace_back(s);
    _dirty = true;
  }

  bool has_sample_at(const double t) const {
    if (_dirty) {
      update();
    }

    const auto it = std::find_if(_samples.begin(), _samples.end(), [&t](const Sample &s) {
      return tinyusdz::math::is_close(t, s.t);
    });

    return (it != _samples.end());
  }

  bool get_sample_at(const double t, Sample **dst) {
    if (!dst) {
      return false;
    }

    if (_dirty) {
      update();
    }

    const auto it = std::find_if(_samples.begin(), _samples.end(), [&t](const Sample &sample) {
      return math::is_close(t, sample.t);
    });

    if (it != _samples.end()) {
      (*dst) = &(*it); 
    }
    return false;
  }

  const std::vector<Sample> &get_samples() const {
    if (_dirty) {
      update();
    }

    return _samples;
  }

  std::vector<Sample> &samples() {
    if (_dirty) {
      update();
    }

    return _samples;
  }

  // From typeless timesamples.
  bool from_timesamples(const value::TimeSamples &ts) {
    std::vector<Sample> buf;
    for (size_t i = 0; i < ts.size(); i++) {
      if (ts.get_samples()[i].value.type_id() != value::TypeTraits<T>::type_id()) {
        return false;
      }
      Sample s;
      s.t = ts.get_samples()[i].t;
      s.blocked = ts.get_samples()[i].blocked;
      if (const auto pv = ts.get_samples()[i].value.as<T>()) {
        s.value = (*pv);
      } else {
        return false;
      }

      buf.push_back(s);
    }


    _samples = std::move(buf);
    _dirty = true;

    return true;
  }

  size_t size() const {
    if (_dirty) {
      update();
    }
    return _samples.size();
  }

 private:
  // Need to be sorted when looking up the value.
  mutable std::vector<Sample> _samples;
  mutable bool _dirty{false};
};

//
// Scalar(default) and/or TimeSamples
//
template <typename T>
struct Animatable {
 public:
  bool is_blocked() const { return _blocked; }

  bool is_timesamples() const {
    if (is_blocked()) {
      return false;
    }

    if (_has_value) {
      return false;
    }

    return !_ts.empty();
  }

  bool is_scalar() const {
    if (is_blocked()) {
      return false;
    }
    
    return _ts.empty();
  }

  ///
  /// Get value at specific time.
  ///
  bool get(double t, T *v,
           const value::TimeSampleInterpolationType tinerp =
               value::TimeSampleInterpolationType::Linear) const {
    if (!v) {
      return false;
    }

    if (is_blocked()) {
      return false;
    }

    if (value::TimeCode(t).is_default()) {
      if (has_value()) {
        (*v) = _value;
        return true;
      }
    }

    if (has_timesamples()) {
      return _ts.get(v, t, tinerp);
    }
    
    if (has_default()) {
      return get_scalar(v);
    }

    return false;
  }

  ///
  /// Get scalar(default) value.
  ///
  bool get_scalar(T *v) const {
    if (!v) {
      return false;
    }

    if (is_blocked()) {
      return false;
    } else if (has_value()) {
      (*v) = _value;
      return true;
    }

    // timesamples
    return false;
  }

  bool get_default(T *v) const {
    return get_scalar(v);
  }

  // TimeSamples
  // void set(double t, const T &v);

  void add_sample(const double t, const T &v) { _ts.add_sample(t, v); }

  // Add None(ValueBlock) sample to timesamples
  void add_blocked_sample(const double t) { _ts.add_blocked_sample(t); }

  // Scalar
  void set(const T &v) {
    _value = v;
    _blocked = false;
    _has_value = true;
  }

  void set_default(const T &v) {
    set(v);
  }

  void set(const TypedTimeSamples<T> &ts) {
    _ts = ts;
  }

  void set(TypedTimeSamples<T> &&ts) {
    _ts = std::move(ts);
  }

  void set_timesamples(const TypedTimeSamples<T> &ts) {
    return set(ts);
  }

  void set_timesamples(TypedTimeSamples<T> &&ts) {
    return set(ts);
  }

  void clear_scalar() {
    _has_value = false;
  }

  void clear_timesamples() {
    _ts.samples().clear();
  }

  bool has_value() const {
    return _has_value;
  }

  bool has_default() const {
    return has_value();
  }

  bool has_timesamples() const {
    return _ts.size();
  }

  const TypedTimeSamples<T> &get_timesamples() const { return _ts; }

  Animatable() {}

  Animatable(const T &v) {
    set(v);
  }

  // TODO: Init with timesamples

 private:
  // scalar
  T _value{};
  bool _has_value{false};
  bool _blocked{false};

  // timesamples
  TypedTimeSamples<T> _ts;
};

///
/// Tyeped Attribute without fallback(default) value.
/// For attribute with `uniform` qualifier or TimeSamples, or have
/// `.connect`(Connection)
///
/// To support multiple definition of attribute(up to 2), we support both having
/// Connection and values.
///
/// e.g.  float var = 1.0
///       float var.connect = </path/to/value>
///       (metadata is shared)
///
/// - `authored() = true` : Attribute value is authored(attribute is
/// described in USDA/USDC)
/// - `authored() = false` : Attribute value is not authored(not described
/// in USD). If you call `get()`, fallback value is returned.
///
template <typename T>
class TypedAttribute {
 public:
  static std::string type_name() { return value::TypeTraits<T>::type_name(); }

  static uint32_t type_id() { return value::TypeTraits<T>::type_id(); }

  TypedAttribute() = default;

  TypedAttribute &operator=(const T &value) {
    _attrib = value;

    return (*this);
  }

  // 'default' value or timeSampled value(when T = Animatable)
  void set_value(const T &v) { _attrib = v; }
  bool has_value() const { return _attrib.has_value(); }

  const nonstd::optional<T> get_value() const {
    return _attrib;
  }

  bool get_value(T *dst) const {
    if (!dst) return false;

    if (_attrib) {
      (*dst) = _attrib.value();
      return true;
    }
    return false;
  }

  bool is_blocked() const { return _blocked; }

  // for `uniform` attribute only
  void set_blocked(bool onoff) { _blocked = onoff; }

  bool is_connection() const { return _paths.size() && !has_value(); }

  void set_connection(const Path &path) {
    _paths.clear();
    _paths.push_back(path);
  }

  void set_connections(const std::vector<Path> &paths) { _paths = paths; }

  const std::vector<Path> &get_connections() const { return _paths; }
  const std::vector<Path> &connections() const { return _paths; }

  const nonstd::optional<Path> get_connection() const {
    if (_paths.size()) {
      return _paths[0];
    }

    return nonstd::nullopt;
  }

  bool has_connections() const {
    return _paths.size();
  }

  void clear_connections() {
    _paths.clear();
  }

  // TODO: Supply set_connection_empty()?

  void set_value_empty() { _value_empty = true; }

  //
  // Check if the attribute is authored, but no value(including ValueBlock) assigned.
  // e.g.
  //
  // float myval;
  //
  bool is_value_empty() const {
    if (has_connections()) {
      return false;
    }

    if (_attrib.has_value()) {
      return false;
    }

    if (_blocked) {
      return false;
    }

    return _value_empty;
  }

  // The attribute authroed?
  bool authored() const {
    if (_attrib) {
      return true;
    }

    if (has_connections()) {
      return true;
    }

    if (_value_empty) {
      // Declare only.
      return true;
    }

    if (_blocked) {
      return true;
    }

    return false;
  }

  void clear_value() {
    _attrib.reset();
    _value_empty = true;
  }

  const AttrMeta &metas() const { return _metas; }
  AttrMeta &metas() { return _metas; }

 private:
  AttrMeta _metas;
  bool _value_empty{false};  // applies `_attrib`
  std::vector<Path> _paths;
  nonstd::optional<T> _attrib;
  bool _blocked{false};
};

///
/// Tyeped Terminal(Output) Attribute(No value assign, no fallback(default)
/// value, no connection)
///
/// - `authored() = true` : Attribute value is authored(attribute is
/// described in USDA/USDC)
/// - `authored() = false` : Attribute value is not authored(not described
/// in USD).
///
template <typename T>
class TypedTerminalAttribute {
 public:
  void set_authored(bool onoff) { _authored = onoff; }

  // value set?
  bool authored() const { return _authored; }

  static std::string type_name() { return value::TypeTraits<T>::type_name(); }
  static uint32_t type_id() { return value::TypeTraits<T>::type_id(); }

  // Actual type is a typeName in USDA or USDC
  // for example, we accect float3 type for TypedTerminalAttribute<color3f> and
  // print/serialize this attribute value with actual type.
  //
  void set_actual_type_name(const std::string &type_name) {
    _actual_type_name = type_name;
  }

  bool has_actual_type() const { return _actual_type_name.size(); }

  const std::string &get_actual_type_name() const { return _actual_type_name; }

  const AttrMeta &metas() const { return _metas; }
  AttrMeta &metas() { return _metas; }

 private:
  AttrMeta _metas;
  bool _authored{false};
  std::string _actual_type_name;
};

template <typename T>
class TypedAttributeWithFallback;

///
/// Attribute with fallback(default) value.
/// For attribute with `uniform` qualifier or TimeSamples, but don't have
/// `.connect`(Connection)
///
/// - `authored() = true` : Attribute value is authored(attribute is
/// described in USDA/USDC)
/// - `authored() = false` : Attribute value is not authored(not described
/// in USD). If you call `get()`, fallback value is returned.
///
template <typename T>
class TypedAttributeWithFallback {
 public:
  static std::string type_name() { return value::TypeTraits<T>::type_name(); }
  static uint32_t type_id() { return value::TypeTraits<T>::type_id(); }

  TypedAttributeWithFallback() = delete;

  ///
  /// Init with fallback value;
  ///
  TypedAttributeWithFallback(const T &fallback) : _fallback(fallback) {}

  TypedAttributeWithFallback &operator=(const T &value) {
    _attrib = value;

    // fallback Value should be already set with `AttribWithFallback(const T&
    // fallback)` constructor.

    return (*this);
  }

  //
  // FIXME: Defininig copy constructor, move constructor and  move assignment
  // operator Gives compilation error :-(. so do not define it.
  //

  // AttribWithFallback(const AttribWithFallback &rhs) {
  //   attrib = rhs.attrib;
  //   fallback = rhs.fallback;
  // }

  // AttribWithFallback &operator=(T&& value) noexcept {
  //   if (this != &value) {
  //       attrib = std::move(value.attrib);
  //       fallback = std::move(value.fallback);
  //   }
  //   return (*this);
  // }

  // AttribWithFallback(AttribWithFallback &&rhs) noexcept {
  //   if (this != &rhs) {
  //       attrib = std::move(rhs.attrib);
  //       fallback = std::move(rhs.fallback);
  //   }
  // }

  void set_value(const T &v) { _attrib = v; }

  void set_value_empty() { _empty = true; }

  bool has_connections() const { return _paths.size(); }

  //
  // Check if the attribute is authored, but no value(including ValueBlock) assigned.
  // e.g.
  //
  // float myval;
  //
  bool is_value_empty() const {
    if (has_connections()) {
      return false;
    }

    if (_empty) {
      return true;
    }

    if (_attrib) {
      return false;
    }

    return true;
  }

  bool has_value() const {
    if (_empty) {
      return false;
    }

    return true;
  }

  const T &get_value() const {
    if (_attrib) {
      return _attrib.value();
    }
    return _fallback;
  }

  bool is_blocked() const { return _blocked; }

  // for `uniform` attribute only
  void set_blocked(bool onoff) { _blocked = onoff; }

  bool is_connection() const { return _paths.size() && !has_value() ; }

  void set_connection(const Path &path) {
    _paths.clear();
    _paths.push_back(path);
  }

  void set_connections(const std::vector<Path> &paths) { _paths = paths; }

  const std::vector<Path> &get_connections() const { return _paths; }
  const std::vector<Path> &connections() const { return _paths; }

  const nonstd::optional<Path> get_connection() const {
    if (_paths.size()) {
      return _paths[0];
    }

    return nonstd::nullopt;
  }

  void clear_connections() { _paths.clear(); }

  // value set?
  bool authored() const {
    if (_empty) {  // authored with empty value.
      return true;
    }
    if (_attrib) {
      return true;
    }
    if (_paths.size()) {
      return true;
    }
    if (_blocked) {
      return true;
    }
    return false;
  }

  const AttrMeta &metas() const { return _metas; }
  AttrMeta &metas() { return _metas; }

 private:
  AttrMeta _metas;
  std::vector<Path> _paths;
  nonstd::optional<T> _attrib;
  bool _empty{false};
  T _fallback;
  bool _blocked{false};  // for `uniform` attribute.
};

template <typename T>
using TypedAnimatableAttributeWithFallback =
    TypedAttributeWithFallback<Animatable<T>>;
 
bool ConvertTokenAttributeToStringAttribute(
      const TypedAttribute<Animatable<value::token>> &inp,
      TypedAttribute<Animatable<std::string>> &out);


///
/// Similar to pxrUSD's PrimIndex
///
class PrimNode;

#if 0  // TODO
class PrimRange
{
 public:
  class iterator;

  iterator begin() const {
  }
  iterator end() const {
  }

 private:
  const PrimNode *begin_;
  const PrimNode *end_;
  size_t depth_{0};
};
#endif

template <typename T>
class ListOp {
 public:
  ListOp() : is_explicit(false) {}

  void ClearAndMakeExplicit() {
    explicit_items.clear();
    added_items.clear();
    prepended_items.clear();
    appended_items.clear();
    deleted_items.clear();
    ordered_items.clear();

    is_explicit = true;
  }

  bool IsExplicit() const { return is_explicit; }
  bool HasExplicitItems() const { return explicit_items.size(); }

  bool HasAddedItems() const { return added_items.size(); }

  bool HasPrependedItems() const { return prepended_items.size(); }

  bool HasAppendedItems() const { return appended_items.size(); }

  bool HasDeletedItems() const { return deleted_items.size(); }

  bool HasOrderedItems() const { return ordered_items.size(); }

  const std::vector<T> &GetExplicitItems() const { return explicit_items; }

  const std::vector<T> &GetAddedItems() const { return added_items; }

  const std::vector<T> &GetPrependedItems() const { return prepended_items; }

  const std::vector<T> &GetAppendedItems() const { return appended_items; }

  const std::vector<T> &GetDeletedItems() const { return deleted_items; }

  const std::vector<T> &GetOrderedItems() const { return ordered_items; }

  void SetExplicitItems(const std::vector<T> &v) { explicit_items = v; }

  void SetAddedItems(const std::vector<T> &v) { added_items = v; }

  void SetPrependedItems(const std::vector<T> &v) { prepended_items = v; }

  void SetAppendedItems(const std::vector<T> &v) { appended_items = v; }

  void SetDeletedItems(const std::vector<T> &v) { deleted_items = v; }

  void SetOrderedItems(const std::vector<T> &v) { ordered_items = v; }

 private:
  bool is_explicit{false};
  std::vector<T> explicit_items;
  std::vector<T> added_items;
  std::vector<T> prepended_items;
  std::vector<T> appended_items;
  std::vector<T> deleted_items;
  std::vector<T> ordered_items;
};

struct ListOpHeader {
  enum Bits {
    IsExplicitBit = 1 << 0,
    HasExplicitItemsBit = 1 << 1,
    HasAddedItemsBit = 1 << 2,
    HasDeletedItemsBit = 1 << 3,
    HasOrderedItemsBit = 1 << 4,
    HasPrependedItemsBit = 1 << 5,
    HasAppendedItemsBit = 1 << 6
  };

  ListOpHeader() : bits(0) {}

  explicit ListOpHeader(uint8_t b) : bits(b) {}

  explicit ListOpHeader(ListOpHeader const &op) : bits(0) {
    bits |= op.IsExplicit() ? IsExplicitBit : 0;
    bits |= op.HasExplicitItems() ? HasExplicitItemsBit : 0;
    bits |= op.HasAddedItems() ? HasAddedItemsBit : 0;
    bits |= op.HasPrependedItems() ? HasPrependedItemsBit : 0;
    bits |= op.HasAppendedItems() ? HasAppendedItemsBit : 0;
    bits |= op.HasDeletedItems() ? HasDeletedItemsBit : 0;
    bits |= op.HasOrderedItems() ? HasOrderedItemsBit : 0;
  }

  bool IsExplicit() const { return bits & IsExplicitBit; }

  bool HasExplicitItems() const { return bits & HasExplicitItemsBit; }
  bool HasAddedItems() const { return bits & HasAddedItemsBit; }
  bool HasPrependedItems() const { return bits & HasPrependedItemsBit; }
  bool HasAppendedItems() const { return bits & HasAppendedItemsBit; }
  bool HasDeletedItems() const { return bits & HasDeletedItemsBit; }
  bool HasOrderedItems() const { return bits & HasOrderedItemsBit; }

  uint8_t bits;
};

//
// Colum-major order(e.g. employed in OpenGL).
// For example, 12th([3][0]), 13th([3][1]), 14th([3][2]) element corresponds to
// the translation.
//
// template <typename T, size_t N>
// struct Matrix {
//  T m[N][N];
//  constexpr static uint32_t n = N;
//};

inline void Identity(value::matrix2d *mat) {
  memset(mat->m, 0, sizeof(value::matrix2d));
  for (size_t i = 0; i < 2; i++) {
    mat->m[i][i] = static_cast<double>(1);
  }
}

inline void Identity(value::matrix3d *mat) {
  memset(mat->m, 0, sizeof(value::matrix3d));
  for (size_t i = 0; i < 3; i++) {
    mat->m[i][i] = static_cast<double>(1);
  }
}

inline void Identity(value::matrix4d *mat) {
  memset(mat->m, 0, sizeof(value::matrix4d));
  for (size_t i = 0; i < 4; i++) {
    mat->m[i][i] = static_cast<double>(1);
  }
}

struct Extent {
  value::float3 lower{{std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity()}};

  value::float3 upper{{-std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity()}};

  Extent() = default;

  Extent(const value::float3 &l, const value::float3 &u) : lower(l), upper(u) {}

  bool is_valid() const {
    if (lower[0] > upper[0]) return false;
    if (lower[1] > upper[1]) return false;
    if (lower[2] > upper[2]) return false;

    return std::isfinite(lower[0]) && std::isfinite(lower[1]) &&
           std::isfinite(lower[2]) && std::isfinite(upper[0]) &&
           std::isfinite(upper[1]) && std::isfinite(upper[2]);
  }

  std::array<std::array<float, 3>, 2> to_array() const {
    std::array<std::array<float, 3>, 2> ret;
    ret[0][0] = lower[0];
    ret[0][1] = lower[1];
    ret[0][2] = lower[2];
    ret[1][0] = upper[0];
    ret[1][1] = upper[1];
    ret[1][2] = upper[2];

    return ret;
  }

  const Extent &union_with(const value::float3 &p) {
    lower[0] = (std::min)(lower[0], p[0]);
    lower[1] = (std::min)(lower[1], p[1]);
    lower[2] = (std::min)(lower[2], p[2]);

    upper[0] = (std::max)(upper[0], p[0]);
    upper[1] = (std::max)(upper[1], p[1]);
    upper[2] = (std::max)(upper[2], p[2]);

    return *this;
  }

  const Extent &union_with(const value::point3f &p) {
    union_with(value::float3{p.x, p.y, p.z});

    return *this;
  }

  const Extent &union_with(const Extent &box) {
    lower[0] = (std::min)(lower[0], box.lower[0]);
    lower[1] = (std::min)(lower[1], box.lower[1]);
    lower[2] = (std::min)(lower[2], box.lower[2]);

    upper[0] = (std::max)(upper[0], box.upper[0]);
    upper[1] = (std::max)(upper[1], box.upper[1]);
    upper[2] = (std::max)(upper[2], box.upper[2]);

    return *this;
  }
};

#if 0
struct ConnectionPath {
  bool is_input{false};  // true: Input connection. false: Output connection.

  Path path;  // original Path information in USD

  std::string token;  // token(or string) in USD
  int64_t index{-1};  // corresponding array index(e.g. the array index to
                      // `Scene.shaders`)
};

// struct Connection {
//   int64_t src_index{-1};
//   int64_t dest_index{-1};
// };
//
// using connection_id_map =
//     std::unordered_map<std::pair<std::string, std::string>, Connection>;
#endif

//
// Relationship(typeless property)
//
class Relationship {
 public:
  // NOTE: no explicit `uniform` variability for Relationship
  // Relatinship have `uniform` variability implicitly.
  // (in Crate, variability is encoded as `uniform`)

  // (varying?) rel myrel    : DefineOnly(or empty)
  // (varying?) rel myrel = </a> : Path
  // (varying?) rel myrel = [</a>, </b>, ...] : PathVector
  // (varying?) rel myrel = None : ValueBlock
  //
  enum class Type { DefineOnly, Path, PathVector, ValueBlock };

  Type type{Type::DefineOnly};
  Path targetPath;
  std::vector<Path> targetPathVector;
  ListEditQual listOpQual{ListEditQual::ResetToExplicit};

  void set_listedit_qual(ListEditQual q) { listOpQual = q; }
  ListEditQual get_listedit_qual() const { return listOpQual; }

  void set_novalue() { type = Type::DefineOnly; }

  void set(const Path &p) {
    targetPath = p;
    type = Type::Path;
  }

  void set(const std::vector<Path> &pv) {
    targetPathVector = pv;
    type = Type::PathVector;
  }

  void set(const value::ValueBlock &v) {
    (void)v;
    type = Type::ValueBlock;
  }

  void set_blocked() { type = Type::ValueBlock; }

  bool has_value() const { return type != Type::DefineOnly; }

  bool is_path() const { return type == Type::Path; }

  bool is_pathvector() const { return type == Type::PathVector; }

  bool is_blocked() const { return type == Type::ValueBlock; }

  void set_varying_authored() { _varying_authored = true; }

  bool is_varying_authored() const { return _varying_authored; }

  const AttrMeta &metas() const { return _metas; }
  AttrMeta &metas() { return _metas; }

 private:
  AttrMeta _metas;

  // `varying` keyword is explicitly specified?
  bool _varying_authored{false};
};

//
// To represent Property which is explicitly Relationship(for builtin property)
//
// - When authored()
//   - !has_value() => "rel material:binding"
//   - has_value() => targetPath or array of targetPath. "rel material:binding =
//   </rel>" or "rel material:binding = [</rel1>, </rel2>]"
//   - is_blocked() => "rel material:binding = None"
//
class RelationshipProperty {
 public:
  RelationshipProperty() = default;

  RelationshipProperty(const Relationship &rel)
      : _authored(true), _relationship(rel) {}

  RelationshipProperty(const Path &p) { set(p); }

  RelationshipProperty(const std::vector<Path> &pv) { set(pv); }

  RelationshipProperty(const value::ValueBlock &v) { set(v); }

  void set_listedit_qual(ListEditQual q) { _relationship.set_listedit_qual(q); }
  ListEditQual get_listedit_qual() const {
    return _relationship.get_listedit_qual();
  }

  void set_authored() { _authored = true; }

  bool authored() const { return _authored; }

  // Declare-only: e.g. `rel myrel`
  void set_empty() {
    _relationship.set_novalue();
    _authored = true;
  }

  void set(const Path &p) {
    _relationship.set(p);
    _authored = true;
  }

  void set(const std::vector<Path> &pv) {
    _relationship.set(pv);
    _authored = true;
  }

  void set(const value::ValueBlock &v) {
    (void)v;
    _relationship.set_blocked();
    _authored = true;
  }

  void set_blocked() {
    _relationship.set_blocked();
    _authored = true;
  }

  const std::vector<Path> get_targetPaths() const {
    std::vector<Path> paths;
    if (_relationship.is_path()) {
      paths.push_back(_relationship.targetPath);
    } else if (_relationship.is_pathvector()) {
      paths = _relationship.targetPathVector;
    }
    return paths;
  }

  // TODO: Deprecate this direct access API to Relationship value?
  const Relationship &relationship() const { return _relationship; }

  Relationship &relationship() { return _relationship; }

  bool has_value() const { return _relationship.has_value(); }

  bool is_blocked() const { return _relationship.is_blocked(); }

  const AttrMeta &metas() const { return _relationship.metas(); }
  AttrMeta &metas() { return _relationship.metas(); }

 private:
  bool _authored{false};
  Relationship _relationship;
};

//
// TypedConnection is a typed version of Relationship
// example:
//
// token varname.connect = </Material/uv.name>
// float specular.connect = </Material/uv.specular>
// float specular:collection.connect = [</Material/uv.specular>,
// </Material/uv.specular_lod0>]
//
//
template <typename T>
class TypedConnection {
 public:
  using type = typename value::TypeTraits<T>::value_type;

  static std::string type_name() { return value::TypeTraits<T>::type_name(); }

  void set_listedit_qual(ListEditQual q) { _listOpQual = q; }
  ListEditQual get_listedit_qual() const { return _listOpQual; }

  // Define-only: token output:surface
  void set_empty() { _authored = true; }

  void set(const Path &p) {
    _targetPaths.clear();
    _targetPaths.push_back(p);
    _authored = true;
  }

  void set(const std::vector<Path> &pv) {
    _targetPaths = pv;
    _authored = true;
  }

  void set(const value::ValueBlock &v) {
    (void)v;
    _blocked = true;
    _authored = true;
  }

  void set_blocked() {
    _blocked = true;
    _authored = true;
  }

  const std::vector<Path> &get_connections() const { return _targetPaths; }

  bool authored() const { return _authored; }

  bool has_value() const { return _targetPaths.size(); }

  bool is_blocked() const { return _blocked; }

  const AttrMeta &metas() const { return _metas; }
  AttrMeta &metas() { return _metas; }

 private:
  std::vector<Path> _targetPaths;
  bool _authored{false};
  bool _blocked{false};
  AttrMeta _metas;
  ListEditQual _listOpQual{ListEditQual::ResetToExplicit};
};

#if 0  // Moved to value::TimeSampleInterpolationType
// Interpolator for TimeSample data
enum class TimeSampleInterpolation {
  Nearest,  // nearest neighbor
  Linear,   // lerp
  // TODO: more to support...
};
#endif

// Attribute is a struct to hold generic attribute of a property(e.g. primvar)
// of Prim.
// It can have multiple values(default value(or ValueBlock), timeSamples and connection) at once.
//
// TODO: Refactor
class Attribute {

 public:
  Attribute() = default;

  ///
  /// Construct Attribute with typed value(`float`, `token`, ...).
  ///
  template <typename T>
  Attribute(const T &v, bool varying = true) {
    static_assert((value::TypeId::TYPE_ID_VALUE_BEGIN <=
                   value::TypeTraits<T>::type_id()) &&
                      (value::TypeId::TYPE_ID_VALUE_END >
                       value::TypeTraits<T>::type_id()),
                  "T is not a value type");
    set_value(v);
    variability() = varying ? Variability::Varying : Variability::Uniform;
  }

  ///
  /// Construct uniform attribute.
  ///
  template <typename T>
  static Attribute Uniform(const T &v) {

    static_assert((value::TypeId::TYPE_ID_VALUE_BEGIN <=
                   value::TypeTraits<T>::type_id()) &&
                      (value::TypeId::TYPE_ID_VALUE_END >
                       value::TypeTraits<T>::type_id()),
                  "T is not a value type");

    Attribute attr;
    attr.set_value(v);
    attr.variability() = Variability::Uniform;
    return attr;
  }


  ///
  /// Construct connection attribute.
  ///
  Attribute(const Path &v) {
    set_connection(v);
  }

  Attribute(const std::vector<Path> &vs) {
    set_connections(vs);
  }

  const std::string &name() const { return _name; }

  std::string &name() { return _name; }

  void set_name(const std::string &name) { _name = name; }

  void set_type_name(const std::string &tname) { _type_name = tname; }

  // `var` may be empty or ValueBlock, so store type info with set_type_name and
  // set_type_id.
  std::string type_name() const {
    if (_type_name.size()) {
      return _type_name;
    }

    if (!is_connection()) {
      // Fallback. May be unreliable(`var` could be empty).
      return _var.type_name();
    }

    return std::string();
  }

  uint32_t type_id() const {
    if (_type_name.size()) {
      return value::GetTypeId(_type_name);
    }

    if (!is_connection()) {
      // Fallback. May be unreliable(`var` could be empty).
      return _var.type_id();
    }

    return value::TYPE_ID_INVALID;
  }

  template <typename T>
  void set_value(const T &v) {
    if (_type_name.empty()) {
      _type_name = value::TypeTraits<T>::type_name();
    }
    _var.set_value(v);
  }

  void set_var(primvar::PrimVar &v) {
    if (_type_name.empty()) {
      _type_name = v.type_name();
    }

    _var = v;
  }

  void set_var(primvar::PrimVar &&v) {
    if (_type_name.empty()) {
      _type_name = v.type_name();
    }

    _var = std::move(v);
  }

  bool is_value() const {
    if (is_connection()) {
      return false;
    }

    if (is_timesamples()) {
      return false;
    }

    if (is_blocked()) {
      return false;
    }

    return true;
  }

  // check if Attribute has default value
  bool has_value() const {
    return _var.has_value(); 
  }

  /// @brief Get the value of Attribute of specified type.
  /// @tparam T value type
  /// @return The value if the underlying PrimVar is type T. Return
  /// nonstd::nullpt when type mismatch.
  template <typename T>
  nonstd::optional<T> get_value() const {
    return _var.get_value<T>();
  }

  template <typename T>
  bool get_value(T *v) const {
    if (!v) {
      return false;
    }

    nonstd::optional<T> ret = _var.get_value<T>();
    if (ret) {
      (*v) = std::move(ret.value());
      return true;
    }

    return false;
  }

  template <typename T>
  void set_timesample(const T &v, double t) {
    _var.set_timesample(t, v);
  }

  template <typename T>
  bool get(const double t, T *dst,
           value::TimeSampleInterpolationType tinterp =
           value::TimeSampleInterpolationType::Linear) const {
    if (!dst) {
      return false;
    }

    if (value::TimeCode(t).is_default()) {
      if (has_value()) {
        nonstd::optional<T> v = _var.get_value<T>();
        if (v) {
          (*dst) = v.value();
          return true;
        }
      }
    }

    if (has_timesamples()) {
      return _var.get_interpolated_value(t, tinterp, dst);
    }

    // try to get 'defaut' value
    return get_value(dst);
  }

  // TODO: Deprecate 'get_value' API
  template <typename T>
  bool get_value(const double t, T *dst,
                 value::TimeSampleInterpolationType tinterp =
                     value::TimeSampleInterpolationType::Linear) const {
    return get(t, dst, tinterp);
  }


  const AttrMeta &metas() const { return _metas; }
  AttrMeta &metas() { return _metas; }

  const primvar::PrimVar &get_var() const { return _var; }
  primvar::PrimVar &get_var() { return _var; }

  void set_blocked(bool onoff) { _var.set_blocked(onoff); }

  bool is_blocked() const {
    if (has_timesamples()) {
      return false;
    }

    return _var.is_blocked(); 
  }
  bool has_blocked() const { return _var.is_blocked(); }

  Variability &variability() { return _variability; }
  Variability variability() const { return _variability; }

  bool is_uniform() const { return _variability == Variability::Uniform; }

  void set_varying_authored() { _varying_authored = true; }

  bool is_varying_authored() const { return _varying_authored; }

  bool is_connection() const {
    if (has_timesamples()) {
      return false;
    }

    if (has_blocked()) {
      return false;
    }

    if (has_value()) {
      return false;
    }

    return _paths.size() > 0;
  }

  bool has_connections() const {
    return _paths.size() > 0;
  }


  bool has_default() const {
    return has_value();
  }

  bool is_timesamples() const {
    if (has_default()) {
      return false;
    }

    if (has_connections()) {
      return false;
    }

    return _var.is_timesamples();
  }

  bool has_timesamples() const {
    return _var.has_timesamples();
  }

  void set_connection(const Path &path) {
    _paths.clear();
    _paths.push_back(path);
  }
  void set_connections(const std::vector<Path> &paths) { _paths = paths; }

  nonstd::optional<Path> get_connection() const {
    if (_paths.size() == 1) {
      return _paths[0];
    }
    return nonstd::nullopt;
  }

  const std::vector<Path> &connections() const { return _paths; }
  std::vector<Path> &connections() { return _paths; }

 private:
  std::string _name;  // attrib name
  Variability _variability{
      Variability::Varying};  // 'uniform` qualifier is handled with
                              // `variability=uniform`

  // `varying` keyword is explicitly specified?
  bool _varying_authored{false};

  // bool _blocked{false};       // Attribute Block('None')
  std::string _type_name;
  primvar::PrimVar _var;
  std::vector<Path> _paths;
  AttrMeta _metas;
};

// Generic container for Attribute or Relation/Connection. And has this property
// is custom or not (Need to lookup schema if the property is custom or not for
// Crate data)
// TODO: Move Connection to Attribute
// TODO: Deprecate `custom` attribute:
// https://github.com/PixarAnimationStudios/USD/issues/2069
class Property {
 public:
  enum class Type {
    EmptyAttrib,        // Attrib with no data.
    Attrib,             // Attrib which contains actual data
    Relation,           // `rel` with targetPath(s).
    NoTargetsRelation,  // `rel` with no targets.
    Connection,  // Connection attribute(`.connect` suffix). TODO: Deprecate
                 // this and use Attrib.
  };

  Property() = default;

  // TODO: Deprecate this constructor.
  // Property(const std::string &type_name, bool custom = false)
  //    : _has_custom(custom) {
  //  _attrib.set_type_name(type_name);
  //  _type = Type::EmptyAttrib;
  //}

  template <typename T>
  Property(bool custom = false) : _has_custom(custom) {
    _attrib.set_type_name(value::TypeTraits<T>::type_name());
    _type = Type::EmptyAttrib;
  }

  static Property MakeEmptyAttrib(const std::string &type_name,
                                  bool custom = false) {
    Property p;
    p.set_custom(custom);
    p.set_property_type(Type::EmptyAttrib);
    p.attribute().set_type_name(type_name);
    return p;
  }

  Property(const Attribute &a, bool custom = false)
      : _attrib(a), _has_custom(custom) {
    _type = Type::Attrib;
  }

  Property(Attribute &&a, bool custom = false)
      : _attrib(std::move(a)), _has_custom(custom) {
    _type = Type::Attrib;
  }

  // Relationship(typeless)
  Property(const Relationship &r, bool custom = false)
      : _rel(r), _has_custom(custom) {
    _type = Type::Relation;
    set_listedit_qual(r.get_listedit_qual());
  }

  // Relationship(typeless)
  Property(Relationship &&r, bool custom = false)
      : _has_custom(custom) {
    _type = Type::Relation;
    set_listedit_qual(r.get_listedit_qual());
    _rel = std::move(r);
  }

  // Attribute Connection: has type
  Property(const Path &path, const std::string &prop_value_type_name,
           bool custom = false)
      : _prop_value_type_name(prop_value_type_name), _has_custom(custom) {
    _attrib.set_connection(path);
    _attrib.set_type_name(prop_value_type_name);
    _type = Type::Connection;
  }

  // Attribute Connection: has multiple targetPaths
  Property(const std::vector<Path> &paths,
           const std::string &prop_value_type_name, bool custom = false)
      : _prop_value_type_name(prop_value_type_name), _has_custom(custom) {
    _attrib.set_connections(paths);
    _attrib.set_type_name(prop_value_type_name);
    _type = Type::Connection;
  }

  bool is_attribute() const {
    return (_type == Type::EmptyAttrib) || (_type == Type::Attrib);
  }
  bool is_empty() const {
    return (_type == Type::EmptyAttrib) || (_type == Type::NoTargetsRelation);
  }
  bool is_relationship() const {
    return (_type == Type::Relation) || (_type == Type::NoTargetsRelation);
  }

  // TODO: Deprecate this and use is_attribute_connection
  //bool is_connection() const { return _type == Type::Connection; }

  bool is_attribute_connection() const {
    if (is_attribute()) {
      return _attrib.is_connection();
    }

    return false;
  }

  std::string value_type_name() const {
    if (is_relationship()) {
      // relation is typeless.
      return std::string();
    } else {
      return _attrib.type_name();
    }
  }

  bool has_custom() const { return _has_custom; }
  void set_custom(const bool onoff) { _has_custom = onoff; }

  void set_property_type(Type ty) { _type = ty; }

  Type get_property_type() const { return _type; }

  void set_listedit_qual(ListEditQual qual) { _listOpQual = qual; }

  const Attribute &get_attribute() const { return _attrib; }

  Attribute &attribute() { return _attrib; }

  void set_attribute(const Attribute &attrib) {
    _attrib = attrib;
    _type = Type::Attrib;
  }

  const Relationship &get_relationship() const { return _rel; }

  Relationship &relationship() { return _rel; }

  ///
  /// Convienient methos when Property is a Relationship
  ///

  ///
  /// Return single relationTarget path when Property is a Relationship.
  /// Return the first path when Relationship is composed of PathVector(multiple
  /// paths)
  ///
  nonstd::optional<Path> get_relationTarget() const {

    if (_rel.is_path()) {
      return _rel.targetPath;
    } else if (_rel.is_pathvector()) {
      if (_rel.targetPathVector.size() > 0) {
        return _rel.targetPathVector[0];
      }
    }

    return nonstd::nullopt;
  }

  ///
  /// Return multiple relationTarget paths when Property is a Relationship.
  /// Returns empty when Property is not a Relationship or a Relationship does
  /// not contain any target paths.
  ///
  std::vector<Path> get_relationTargets() const {
    std::vector<Path> pv;

    if (_rel.is_path()) {
      pv.push_back(_rel.targetPath);
    } else if (_rel.is_pathvector()) {
      pv = _rel.targetPathVector;
    }

    return pv;
  }

  ListEditQual get_listedit_qual() const { return _listOpQual; }

 private:
  Attribute _attrib;  // attribute(value or ".connect")

  // List Edit qualifier(Attribute can never be list editable)
  // TODO:  Store listEdit qualifier to `Relation`
  ListEditQual _listOpQual{ListEditQual::ResetToExplicit};

  Type _type{Type::EmptyAttrib};
  Relationship _rel;                  // Relation(`rel`)
  std::string _prop_value_type_name;  // for Connection.
  bool _has_custom{false};  // Qualified with 'custom' keyword? This will be
                            // deprecated though
};

struct XformOp {
  enum class OpType {
    // matrix
    Transform,

    // vector3
    Translate,
    Scale,

    // scalar
    RotateX,
    RotateY,
    RotateZ,

    // vector3
    RotateXYZ,
    RotateXZY,
    RotateYXZ,
    RotateYZX,
    RotateZXY,
    RotateZYX,

    // quaternion
    Orient,

    // Special token
    ResetXformStack,  // !resetXformStack!
  };

  // OpType op;
  OpType op_type;
  bool inverted{false};  // true when `!inverted!` prefix
  std::string
      suffix;  // may contain nested namespaces. e.g. suffix will be
               // ":blender:pivot" for "xformOp:translate:blender:pivot". Suffix
               // will be empty for "xformOp:translate"

  primvar::PrimVar _var;
  // const value::TimeSamples &get_ts() const { return _var.ts_raw(); }

  std::string get_value_type_name() const { return _var.type_name(); }

  uint32_t get_value_type_id() const { return _var.type_id(); }

  // TODO: Check if T is valid type.
  template <class T>
  void set_value(const T &v) {
    _var.set_value(v);
  }

  template <class T>
  void set_default(const T &v) {
    _var.set_value(v);
  }

  template <class T>
  void set_timesample(const float t, const T &v) {
    _var.set_timesample(t, v);
  }

  void set_timesamples(const value::TimeSamples &v) { _var.set_timesamples(v); }

  void set_timesamples(value::TimeSamples &&v) { _var.set_timesamples(v); }

  bool is_timesamples() const { return _var.is_timesamples(); }
  bool has_timesamples() const { return _var.has_timesamples(); }

  void set_blocked(bool onoff) { _is_blocked = onoff; }
  void clear_blocked() { _is_blocked = false; }

  // check if 'default' value is ValueBlock.
  bool is_blocked() const { return _is_blocked || _var.is_blocked(); }

  bool is_default() const { return _var.is_scalar(); }
  bool has_default() const { return _var.has_default(); }

  nonstd::optional<value::TimeSamples> get_timesamples() const {
    if (has_timesamples()) {
      return _var.ts_raw();
    }
    return nonstd::nullopt;
  }

  nonstd::optional<value::Value> get_scalar() const {
    if (has_default()) {
      return _var.value_raw();
    }
    return nonstd::nullopt;
  }

  nonstd::optional<value::Value> get_default() const {
    return get_scalar();
  }

  template <class T>
  nonstd::optional<T> get_value(double t = value::TimeCode::Default(), 
          value::TimeSampleInterpolationType interp =
               value::TimeSampleInterpolationType::Linear) const {
    if (is_timesamples()) {
      T value{};
      if (get_interpolated_value(&value, t, interp)) {
        return value;
      }
      return nonstd::nullopt;
    }

    return _var.get_value<T>();
  }

  template <class T>
  bool get_interpolated_value(T *dst, double t = value::TimeCode::Default(),
           value::TimeSampleInterpolationType interp =
               value::TimeSampleInterpolationType::Linear) const {
    return _var.get_interpolated_value<T>(t, interp, dst);
  }

  const primvar::PrimVar &get_var() const { return _var; }

  primvar::PrimVar &var() { return _var; }

 private:

  bool _is_blocked{false};
};

// forward decl
class MaterialBinding;
struct Model;
class Prim;
class PrimSpec;

// TODO: deprecate this and use PrimSpec for variantSet statement.
// Variant item in VariantSet.
// Variant can contain Prim metas, Prim tree and properties.
struct Variant {
  // const std::string &name() const { return _name; }
  // std::string &name() { return _name; }

  const PrimMeta &metas() const { return _metas; }
  PrimMeta &metas() { return _metas; }

  std::map<std::string, Property> &properties() { return _props; }
  const std::map<std::string, Property> &properties() const { return _props; }

  const std::vector<Prim> &primChildren() const { return _primChildren; }
  std::vector<Prim> &primChildren() { return _primChildren; }

 private:
  // std::vector<int64_t> primIndices;
  std::map<std::string, Property> _props;

  // std::string _name; // variant name
  PrimMeta _metas;

  // We represent Prim children as `Prim` for a while.
  // TODO: Use PrimNode or PrimSpec?
  std::vector<Prim> _primChildren;
};


struct VariantSet {
  // variantSet name = {
  //   "variant1" ...
  //   "variant2" ...
  //   ...
  // }

  std::string name;
  std::map<std::string, Variant> variantSet;
};

// For variantSet statement in PrimSpec(composition).
struct VariantSetSpec
{
  std::string name;
  std::map<std::string, PrimSpec> variantSet;
};

// Collection API
// https://openusd.org/release/api/class_usd_collection_a_p_i.html

constexpr auto kExpandPrims = "expandPrims";
constexpr auto kExplicitOnly = "explicitOnly";
constexpr auto kExpandPrimsAndProperties = "expandPrimsAndProperties";

struct CollectionInstance {

  enum class ExpansionRule {
    ExpandPrims, // "expandPrims" (default)
    ExplicitOnly, // "explicitOnly"
    ExpandPrimsAndProperties, // "expandPrimsAndProperties"
  };

  TypedAttributeWithFallback<ExpansionRule> expansionRule{ExpansionRule::ExpandPrims}; // uniform token collection:collectionName:expansionRule
  TypedAttributeWithFallback<Animatable<bool>> includeRoot{false}; // bool collection:<collectionName>:includeRoot
  nonstd::optional<Relationship> includes; // rel collection:<collectionName>:includes
  nonstd::optional<Relationship> excludes; // rel collection:<collectionName>:excludes

};

class Collection
{
 public:
  const ordered_dict<CollectionInstance> instances() const {
    return _instances;
  }

  bool add_instance(const std::string &name, CollectionInstance &instance) {
    if (_instances.count(name)) {
      return false;
    }

    _instances.insert(name, instance);

    return true;
  }

  bool get_instance(const std::string &name, const CollectionInstance **coll) const {
    if (!coll) {
      return false;
    }

    return _instances.at(name, coll);
  }

  CollectionInstance &get_or_add_instance(const std::string &name) {
    return _instances.get_or_add(name);
  }

  bool has_instance(const std::string &name) const {
    return _instances.count(name);
  }

  bool del_instance(const std::string &name) {
    return _instances.erase(name);
  }

 private:
  ordered_dict<CollectionInstance> _instances;
};

// for bindMaterialAs
constexpr auto kWeaderThanDescendants = "weakerThanDescendants";
constexpr auto kStrongerThanDescendants = "strongerThanDescendants";

enum class MaterialBindingStrength
{
  WeakerThanDescendants, // default
  StrongerThanDescendants
};

// TODO: Move to pprinter.hh?
std::string to_string(const MaterialBindingStrength strength);

class MaterialBinding {
 public:

  static value::token kAllPurpose() {
    return value::token("");
  }

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

  // Some frequently used materialBindings
  nonstd::optional<Relationship> materialBinding; // material:binding
  nonstd::optional<Relationship> materialBindingPreview; // material:binding:preview
  nonstd::optional<Relationship> materialBindingFull; // material:binding:full

  //nonstd::optional<Relationship> materialBindingCollection; // material:binding:collection  Deprecated. use materialBindingCollectionMap[""][""] instead.

  value::token get_materialBindingStrength(const value::token &purpose);
  value::token get_materialBindingStrengthCollection(const value::token &collection_name, const value::token &purpose);

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
    if (mat_purpose.str() == kAllPurpose().str()) {
      return has_materialBinding();
    } else if (mat_purpose.str() == "full") {
      return has_materialBindingFull();
    } else if (mat_purpose.str() == "preview") {
      return has_materialBindingPreview();
    } else {
      return _materialBindingMap.count(mat_purpose.str()) > 0;
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

    return _materialBindingCollectionMap.count(tok) > 0;
  }

  void set_materialBindingCollection(const value::token &tok, const value::token &mat_purpose, const Relationship &rel) {

    // NOTE:
    // https://openusd.org/release/wp_usdshade.html#basic-proposal-for-collection-based-assignment
    // says: material:binding:collection defines a namespace of binding relationships to be applied in namespace order, with the earliest ordered binding relationship the strongest
    //
    // so the app is better first check if `tok` element alreasy exists(using has_materialBindingCollection)

    auto &m = _materialBindingCollectionMap[tok.str()];

    m.insert(mat_purpose.str(), rel);
  }

  void clear_materialBindingCollection(const value::token &tok, const value::token &mat_purpose) {
    if (_materialBindingCollectionMap.count(tok.str())) {
      _materialBindingCollectionMap[tok.str()].erase(mat_purpose.str());
    }
  }

  void set_materialBindingCollection(const value::token &tok, const value::token &mat_purpose, const Relationship &rel, MaterialBindingStrength strength) {
    value::token strength_tok(to_string(strength));

    Relationship r = rel;
    r.metas().bindMaterialAs = strength_tok;

    _materialBindingCollectionMap[tok.str()].insert(mat_purpose.str(), r);
  }

  const std::map<std::string, Relationship> &materialBindingMap() const {
    return _materialBindingMap;
  }

  const std::map<std::string, ordered_dict<Relationship>> &materialBindingCollectionMap() const {
    return _materialBindingCollectionMap;
  }

  bool get_materialBinding(const value::token &mat_purpose, Relationship *relOut) const {
    if (!relOut) {
      return false;
    }

    if (mat_purpose.str().empty()) {
      if (materialBinding.has_value()) {
        (*relOut) = materialBinding.value();
        return true;
      } else {
        return false; // not authored
      }
    } else if (mat_purpose.str() == "full") {
      if (materialBindingFull.has_value()) {
        (*relOut) = materialBindingFull.value();
        return true;
      } else {
        return false; // not authored
      }
    } else if (mat_purpose.str() == "preview") {
      if (materialBindingPreview.has_value()) {
        (*relOut) = materialBindingPreview.value();
        return true;
      } else {
        return false; // not authored
      }
    } else {
      if (_materialBindingMap.count(mat_purpose.str())) {
        (*relOut) = _materialBindingMap.at(mat_purpose.str());
        return true;
      } else {
        return false; // not authored
      }
    }
  }

 private:

  // For material:binding(excludes frequently used `material:binding`, `material:binding:full` and `material:binding:preview`)
  // key = PURPOSE, value = rel
  std::map<std::string, Relationship> _materialBindingMap;

  // For material:binding:collection
  // Use ordered dict since the requests:
  //
  // https://openusd.org/release/wp_usdshade.html#basic-proposal-for-collection-based-assignment
  //
  // `...with the earliest ordered binding relationship the strongest`
  //
  // key = PURPOSE, value = map<NAME, Rel>
  // TODO: Use multi-index map
  std::map<std::string, ordered_dict<Relationship>> _materialBindingCollectionMap;
};

// Generic primspec container.
// Unknown or unsupported Prim type are also reprenseted as Model for now.
struct Model : public Collection, MaterialBinding {
  std::string name;

  std::string prim_type_name;  // e.g. "" for `def "bora" {}`, "UnknownPrim" for
                               // `def UnknownPrim "bora" {}`
  Specifier spec{Specifier::Def};

  int64_t parent_id{-1};  // Index to parent node

  PrimMeta meta;

  std::pair<ListEditQual, std::vector<Reference>> references;
  std::pair<ListEditQual, std::vector<Payload>> payload;

  // std::map<std::string, VariantSet> variantSets;

  std::map<std::string, Property> props;

  const std::vector<value::token> &primChildrenNames() const {
    return _primChildren;
  }
  const std::vector<value::token> &propertyNames() const { return _properties; }
  std::vector<value::token> &primChildrenNames() { return _primChildren; }
  std::vector<value::token> &propertyNames() { return _properties; }

 private:
  std::vector<value::token> _primChildren;
  std::vector<value::token> _properties;
};

#if 0  // TODO: Remove
// Generic "class" Node
// Mostly identical to GPrim
struct Klass {
  std::string name;
  int64_t parent_id{-1};  // Index to parent node

  std::vector<std::pair<ListEditQual, Reference>> references;

  std::map<std::string, Property> props;
};
#endif

//
// Predefined node classes
//

// USDZ Schemas for AR
// https://developer.apple.com/documentation/arkit/usdz_schemas_for_ar/schema_definitions_for_third-party_digital_content_creation_dcc

// UsdPhysics
struct Preliminary_PhysicsGravitationalForce {
  // physics::gravitatioalForce::acceleration
  value::double3 acceleration{{0.0, -9.81, 0.0}};  // [m/s^2]
};

struct Preliminary_PhysicsMaterialAPI {
  // preliminary:physics:material:restitution
  double restitution;  // [0.0, 1.0]

  // preliminary:physics:material:friction:static
  double friction_static;

  // preliminary:physics:material:friction:dynamic
  double friction_dynamic;
};

struct Preliminary_PhysicsRigidBodyAPI {
  // preliminary:physics:rigidBody:mass
  double mass{1.0};

  // preliminary:physics:rigidBody:initiallyActive
  bool initiallyActive{true};
};

struct Preliminary_PhysicsColliderAPI {
  // preliminary::physics::collider::convexShape
  Path convexShape;
};

struct Preliminary_InfiniteColliderPlane {
  value::double3 position{{0.0, 0.0, 0.0}};
  value::double3 normal{{0.0, 0.0, 0.0}};

  Extent extent;  // [-FLT_MAX, FLT_MAX]

  Preliminary_InfiniteColliderPlane() {
    extent.lower[0] = -(std::numeric_limits<float>::max)();
    extent.lower[1] = -(std::numeric_limits<float>::max)();
    extent.lower[2] = -(std::numeric_limits<float>::max)();
    extent.upper[0] = (std::numeric_limits<float>::max)();
    extent.upper[1] = (std::numeric_limits<float>::max)();
    extent.upper[2] = (std::numeric_limits<float>::max)();
  }
};

// UsdInteractive
struct Preliminary_AnchoringAPI {
  // preliminary:anchoring:type
  std::string type;  // "plane", "image", "face", "none";

  std::string alignment;  // "horizontal", "vertical", "any";

  Path referenceImage;
};

struct Preliminary_ReferenceImage {
  int64_t image_id{-1};  // asset image

  double physicalWidth{0.0};
};

struct Preliminary_Behavior {
  Path triggers;
  Path actions;
  bool exclusive{false};
};

struct Preliminary_Trigger {
  // uniform token info:id
  std::string info;  // Store decoded string from token id
};

struct Preliminary_Action {
  // uniform token info:id
  std::string info;  // Store decoded string from token id

  std::string multiplePerformOperation{
      "ignore"};  // ["ignore", "allow", "stop"]
};

struct Preliminary_Text {
  std::string content;
  std::vector<std::string> font;  // An array of font names

  float pointSize{144.0f};
  float width;
  float height;
  float depth{0.0f};

  std::string wrapMode{"flowing"};  // ["singleLine", "hardBreaks", "flowing"]
  std::string horizontalAlignmment{
      "center"};  // ["left", "center", "right", "justified"]
  std::string verticalAlignmment{
      "middle"};  // ["top", "middle", "lowerMiddle", "baseline", "bottom"]
};

// Simple volume class.
// Currently this is just an placeholder. Not implemented.

struct OpenVDBAsset {
  std::string fieldDataType{"float"};
  std::string fieldName{"density"};
  std::string filePath;  // asset
};

// MagicaVoxel Vox
struct VoxAsset {
  std::string fieldDataType{"float"};
  std::string fieldName{"density"};
  std::string filePath;  // asset
};

struct Volume {
  OpenVDBAsset vdb;
  VoxAsset vox;
};

// `Scope` is uncommon in graphics community, its something like `Group`.
// From USD doc: Scope is the simplest grouping primitive, and does not carry
// the baggage of transformability.
struct Scope : Collection, MaterialBinding {
  std::string name;
  Specifier spec{Specifier::Def};

  int64_t parent_id{-1};

  PrimMeta meta;

  TypedAttributeWithFallback<Animatable<Visibility>> visibility{Visibility::Inherited};
  Purpose purpose{Purpose::Default};

  std::map<std::string, VariantSet> variantSet;

  std::map<std::string, Property> props;

  const std::vector<value::token> &primChildrenNames() const {
    return _primChildren;
  }
  const std::vector<value::token> &propertyNames() const { return _properties; }
  std::vector<value::token> &primChildrenNames() { return _primChildren; }
  std::vector<value::token> &propertyNames() { return _properties; }

 private:
  std::vector<value::token> _primChildren;
  std::vector<value::token> _properties;
};

///
/// Get elementName from Prim(e.g., Xform::name, GeomMesh::name)
/// `v` must be the value of Prim class.
///
nonstd::optional<std::string> GetPrimElementName(const value::Value &v);

///
/// Set name for Prim `v`(e.g. Xform::name = elementName)
/// `v` must be the value of Prim class.
///
bool SetPrimElementName(value::Value &v, const std::string &elementName);

//
// For `Stage` scene graph.
// Its a freezed state of an element of a scene graph(so no Prim
// additin/deletion from a scene graph is considered). May be Similar to `Prim`
// in pxrUSD. If you want to manipulate scene graph, use PrimSpec instead(but
// PrimSpec is W.I.P.) This class uses tree-representation of `Prim`. Easy to
// use, but may not be performant than flattened array index representation of
// Prim tree(Index-based scene graph such like glTF).
//
class Prim {
 public:
  // elementName is read from `rhs`(if it is a class of Prim)
  Prim(const value::Value &rhs);
  Prim(value::Value &&rhs);

  Prim(const std::string &elementName, const value::Value &rhs);
  Prim(const std::string &elementName, value::Value &&rhs);

  template <typename T>
  Prim(const T &prim) {
    set_primdata(prim);
  }

  template <typename T>
  Prim(const std::string &elementName, const T &prim) {
    set_primdata(elementName, prim);
  }

  // Replace exting prim
  template <typename T>
  void set_primdata(const T &prim) {
    // Check if T is Prim class type.
    static_assert((value::TypeId::TYPE_ID_MODEL_BEGIN <=
                   value::TypeTraits<T>::type_id()) &&
                      (value::TypeId::TYPE_ID_MODEL_END >
                       value::TypeTraits<T>::type_id()),
                  "T is not a Prim class type");
    _data = prim;
    // Use prim.name for elementName
    _elementPath = Path(prim.name, "");
  }

  // Replace exting prim
  template <typename T>
  void set_primdata(const std::string &elementName, const T &prim) {
    // Check if T is Prim class type.
    static_assert((value::TypeId::TYPE_ID_MODEL_BEGIN <=
                   value::TypeTraits<T>::type_id()) &&
                      (value::TypeId::TYPE_ID_MODEL_END >
                       value::TypeTraits<T>::type_id()),
                  "T is not a Prim class type");
    _data = prim;
    SetPrimElementName(_data, elementName);
    _elementPath = Path(elementName, "");
  }

  ///
  /// Add Prim as a child.
  /// When `rename_element_name` is true, rename input Prims elementName to make
  /// it unique among children(since USD(Crate) spec doesn't allow same Prim
  /// elementName in the same Prim hierarchy.
  ///
  /// Renaming rule is Maya-like:
  /// - No elementName given: `default`
  /// - Add or increment number suffix to the elementName:
  ///    - `plane` => `plane1`
  ///    - `plane1` => `plane2`
  ///
  /// Note: This function is thread-safe.
  ///
  /// @return true Upon success. false when failed(e.g. Prim with same
  /// Prim::element_name() already exists when `rename_element_name` is false)
  /// and fill `err` with error message
  ///
  bool add_child(Prim &&prim, const bool rename_element_name = true,
                 std::string *err = nullptr);

  ///
  /// Replace existing child Prim whose elementName is `child_prim_name`.
  /// When there is no child Prim with elementName `child_prim_name` exists,
  /// `prim` is added and rename is elementName to `child_prim_name`.
  ///
  /// @return true Upon success. false when failed(e.g. `child_prim_name` is
  /// empty string or invalid Prim name) and fill `err` with error message.
  ///
  bool replace_child(const std::string &child_prim_name, Prim &&prim,
                     std::string *err = nullptr);

#if 0
  ///
  /// Add Prim as a child.
  ///
  ///
  /// @return true Upon success. false when failed(e.g. Prim with same Prim::element_name() already exists) and fill `err` with error message
  ///
  /// Note: This function is thread-safe.
  ///
  bool add_child(Prim &&prim, const std::string &basename, std::string *err = nullptr);
#endif

  //{
  //
  //  _children.emplace_back(std::move(prim));
  //  _child_dirty = true;
  //}

  // TODO: Deprecate this API to disallow direct modification of children.
  std::vector<Prim> &children() { return _children; }

  const std::vector<Prim> &children() const { return _children; }

  const value::Value &data() const { return _data; }
  value::Value &get_data() { return _data; }

  Specifier &specifier() { return _specifier; }

  Specifier specifier() const { return _specifier; }

  // local_path is reserved for Prim composition.
  // for a while please use absolute_path(full Prim absolute path) or
  // element_name(leaf Prim name).
  Path &local_path() { return _path; }
  const Path &local_path() const { return _path; }

  ///
  /// Absolute Prim Path(e.g. "/xform/mesh0") is available after
  /// Stage::compute_absolute_path() or assign it manually by an app.
  ///
  Path &absolute_path() { return _abs_path; }
  const Path &absolute_path() const { return _abs_path; }

  Path &element_path() { return _elementPath; }
  const Path &element_path() const { return _elementPath; }

  // elementName = element_path's prim part
  const std::string &element_name() const { return _elementPath.prim_part(); }

  const std::string type_name() const { return _data.type_name(); }

  uint32_t type_id() const { return _data.type_id(); }

  std::string &prim_type_name() { return _prim_type_name; }
  const std::string &prim_type_name() const { return _prim_type_name; }

  template <typename T>
  bool is() const {
    return (_data.type_id() == value::TypeTraits<T>::type_id());
  }

  // Return a pointer of a concrete Prim class(Xform, Material, ...)
  // Return nullptr when failed to cast or T is not a Prim type.
  template <typename T>
  const T *as() const {
    // Check if T is Prim type. e.g. Xform, Material, ...
    if ((value::TypeId::TYPE_ID_MODEL_BEGIN <=
         value::TypeTraits<T>::type_id()) &&
        (value::TypeId::TYPE_ID_MODEL_END > value::TypeTraits<T>::type_id())) {
      return _data.as<T>();
    }

    return nullptr;
  }

#if 0
  // Compute or update world matrix of this Prim.
  // Will traverse child Prims.
  void update_world_matrix(const value::matrix4d &parent_mat);

  const value::matrix4d &get_local_matrix() const;
  const value::matrix4d &get_world_matrix() const;
#endif

  const PrimMeta &metas() const;
  PrimMeta &metas();

  int64_t prim_id() const { return _prim_id; }

  int64_t &prim_id() { return _prim_id; }

  const std::map<std::string, VariantSet> &variantSets() const {
    return _variantSets;
  }

  std::map<std::string, VariantSet> &variantSets() { return _variantSets; }

  ///
  /// Get indices for children().
  ///
  /// This is an utility API to traverse child Prims according to `primChildren`
  /// Prim metadata. If you want to traverse child Prims as done in pxrUSD(which
  /// used `primChildren` to determine the order of traversal), use this
  /// function.
  ///
  /// If no `primChildren` Prim metadata, it will simply returns [0,
  /// children().size()) sequence.
  ///
  /// index may have -1, which means invalid(child Prim not found described in
  /// by primChildren) Also, app should extra check of the value of index if
  /// `indices_is_valid` is set to false(index may be duplicated(Duplicated Prim
  /// name exits in `primChildren`)  and not in range `[0, children() -1`)
  ///
  /// NOTE: This function build a cache.
  ///
  /// @param[in] force_update Always rebuild child_indices. false = use cache if
  /// exits.
  /// @param[out] indices_is_valid Optional. Set true when returned indices are
  /// valid.
  ///
  const std::vector<int64_t> &get_child_indices_from_primChildren(
      bool force_update = true, bool *indices_is_valid = nullptr) const;

  // TODO: Add API to get parent Prim directly?
  // (Currently we need to traverse parent Prim using Stage)

 private:
  Path _abs_path;  // Absolute Prim path in a freezed(after composition state).
                   // Usually set by Stage::compute_absolute_path()
  Path _path;  // Prim's local path name. May contain Property, Relationship and
               // other infos, but do not include parent's path. To get fully
               // absolute path of a Prim(e.g. "/xform0/mymesh0", You need to
               // traverse Prim tree and concatename `elementPath` or use
               // ***(T.B.D>) method in `Stage` class
  Path _elementPath;  // leaf("terminal") Prim name.(e.g. "myxform" for `def
                      // Xform "myform"`). For root node, elementPath name is
                      // empty string("").

  std::string _prim_type_name;  // Prim's type name. e.g. "Xform", "Mesh",
                                // "UnknownPrim", ... Could be empty for `def
                                // "myprim" {}`

  Specifier _specifier{
      Specifier::Invalid};  // `def`, `over` or `class`. Usually `def`
  value::Value
      _data;  // Generic container for concrete Prim object. GPrim, Xform, ...

  std::vector<Prim> _children;  // child Prim nodes
  // std::set<std::string> _childrenNames; // child Prim name(elementName).
  std::multiset<std::string>
      _childrenNameSet;  // Stores input child Prim's elementName to assign
                         // unique elementName in `add_child`

  mutable bool _child_dirty{false};
  mutable bool _primChildrenIndicesIsValid{
      false};  // true when indices in _primChildrenIndices are not -1, unique,
               // and index value are within [0, children().size()), and also
               // _primChildrenIndices.size() == children().size()
  mutable std::vector<int64_t>
      _primChildrenIndices;  // Get corresponding array index in _children,
                             // based on `metas().primChildren` token[] info. -1
                             // = invalid.

  int64_t _prim_id{
      -1};  // Unique Prim id when positive(starts with 1). Id is assigned by
            // Stage::compute_absolute_prim_path_and_assign_prim_id. Usually [1,
            // NumPrimsInStage)

  std::map<std::string, VariantSet> _variantSets;

#if defined(TINYUSDZ_ENABLE_THREAD)
  mutable std::mutex _mutex;
#endif
};

bool IsXformablePrim(const Prim &prim);

// forward decl(xform.hh)
struct Xformable;
bool CastToXformable(const Prim &prim, const Xformable **xformable);

///
/// Get Prim's local transform(xformOps) at specified time.
/// For non-Xformable Prim it returns identity matrix.
///
/// @param[in] prim Prim
/// @param[out] resetXformStack Whether Prim's xformOps contains
/// `!resetXformStack!` or not
/// @param[in] t time
/// @param[in] tinterp Interpolation type(Linear or Held)
///
value::matrix4d GetLocalTransform(const Prim &prim, bool *resetXformStak,
                                  double t = value::TimeCode::Default(),
                                  value::TimeSampleInterpolationType tinterp =
                                      value::TimeSampleInterpolationType::Linear);

///
/// TODO: Deprecate this class and use PrimPec
/// NOTE PrimNode is designed for Stage(freezed)
///
/// Contains concrete Prim object and composition elements.
///
/// PrimNode is near to the final state of `Prim`.
/// Doing one further step(Composition, Flatten, select Variant) to get `Prim`.
///
/// Similar to `PrimIndex` in pxrUSD
///

class PrimNode {
  Path path;
  Path elementPath;

  PrimNode(const value::Value &rhs);

  PrimNode(value::Value &&rhs);

  value::Value prim;  // GPrim, Xform, ...

  std::vector<PrimNode> children;  // child nodes

  ///
  /// Select variant.
  ///
  bool select_variant(const std::string &target_name,
                      const std::string &variant_name) {
    const auto m = _vsmap.find(target_name);
    if (m != _vsmap.end()) {
      _current_vsmap[target_name] = variant_name;
      return true;
    } else {
      return false;
    }
  }

  ///
  /// Get current variant selection.
  ///
  bool current_variant_selection(const std::string &target_name,
                      std::string *selected_variant_name) {

    if (!selected_variant_name) {
      return false;
    }

    const auto m = _vsmap.find(target_name);
    if (m != _vsmap.end()) {
      const auto sm = _current_vsmap.find(target_name);
      if (sm != _current_vsmap.end()) {
        (*selected_variant_name) = sm->second;
      } else {
        (*selected_variant_name) = m->second;
      }
      return true;
    } else {
      return false;
    }
  }

  ///
  /// List variants in this Prim
  ///
  /// key = variant prim name
  /// value = variants
  ///
  const VariantSelectionMap &get_variant_selection_map() const { return _vsmap; }

  ///
  /// Variants
  ///
  /// VariantSet = Prim metas + Properties and/or child Prims
  ///            = repsetent as PrimNode for a while.
  ///
  ///
  /// key = variant name
  using VariantSet = std::map<std::string, PrimNode>;
  std::map<std::string, VariantSet> varitnSetList;  // key = variant

  VariantSelectionMap _vsmap;          // Original variant selections
  VariantSelectionMap _current_vsmap;  // Currently selected variants

  std::vector<value::token> primChildren;  // List of child Prim nodes
  std::vector<value::token> properties;    // List of property names
  std::vector<value::token> variantChildren; // List of child VariantSet nodes.
};

/// Similar to PrimSpec
/// PrimSpec is a Prim object state just after reading it from USDA and USDC.
/// The state before compositions and Prim reconstruction by applying
/// schema(ReconstructPrim in prim-reconstruct.hh) happens.
///
/// Its composed primarily of name, specifier, PrimMeta and
/// Properties(Relationships and Attributes)
class PrimSpec {
 public:
  PrimSpec() = default;

  PrimSpec(const Specifier &spec, const std::string &name)
      : _specifier(spec), _name(name) {}
  PrimSpec(const Specifier &spec, const std::string &typeName,
           const std::string &name)
      : _specifier(spec), _typeName(typeName), _name(name) {}

  PrimSpec(const PrimSpec &rhs) {
    if (this != &rhs) {
      CopyFrom(rhs);
    }
  }

  PrimSpec &operator=(const PrimSpec &rhs) {
    if (this != &rhs) {
      CopyFrom(rhs);
    }

    return *this;
  }

  PrimSpec &operator=(PrimSpec &&rhs) noexcept {
    if (this != &rhs) {
      MoveFrom(rhs);
    }

    return *this;
  }

  const std::string &name() const { return _name; }
  std::string &name() { return _name; }

  const std::string &typeName() const { return _typeName; }
  // Can change type name
  std::string &typeName() { return _typeName; }

  const Specifier &specifier() const { return _specifier; }
  Specifier &specifier() { return _specifier; }

  const std::vector<PrimSpec> &children() const { return _children; }
  std::vector<PrimSpec> &children() { return _children; }

  ///
  /// Select variant.
  ///
  bool select_variant(const std::string &target_name,
                      const std::string &variant_name) {
    if (metas().variants.has_value()) {
      const auto m = metas().variants.value().find(target_name);
      if (m != metas().variants.value().end()) {
        _current_vsmap[target_name] = variant_name;
        return true;
      } else {
        return false;
      }
    }
    return false;
  }

  bool current_variant_selection(const std::string &target_name,
                      std::string *selected_variant_name) {

    if (!selected_variant_name) {
      return false;
    }

    if (!metas().variants.has_value()) {
      return false;
    }

    const auto &vsmap = metas().variants.value();

    const auto m = vsmap.find(target_name);
    if (m != vsmap.end()) {
      const auto sm = _current_vsmap.find(target_name);
      if (sm != _current_vsmap.end()) {
        (*selected_variant_name) = sm->second;
      } else {
        (*selected_variant_name) = m->second;
      }
      return true;
    } else {
      return false;
    }
  }

  ///
  /// List variants in this PrimSpec
  /// key = variant name
  /// value = variats
  ///
  const VariantSelectionMap get_variant_selection_map() const {
    VariantSelectionMap vsmap;
    if (metas().variants.has_value()) {
      vsmap = metas().variants.value();
    }
    return vsmap;
  }

  ///
  /// Variants
  ///
  /// VariantSet = Prim metas + Properties and/or child Prims
  ///            = repsetent as PrimNode for a while.
  ///
  ///
  /// key = variant name
  std::map<std::string, VariantSetSpec> &variantSets() { return _variantSets; }
  const std::map<std::string, VariantSetSpec> &variantSets() const { return _variantSets; }

  const PrimMeta &metas() const { return _metas; }

  PrimMeta &metas() { return _metas; }

  using PropertyMap = std::map<std::string, Property>;

  const PropertyMap &props() const { return _props; }
  PropertyMap &props() { return _props; }

  const std::vector<Reference> &get_references();
  const ListEditQual &get_references_listedit_qualifier();

  const std::vector<Payload> &get_payloads();
  const ListEditQual &get_payloads_listedit_qualifier();

  const std::vector<value::token> &primChildren() const {
    return _primChildren;
  }

  const std::vector<value::token> &propertyNames() const {
    return _properties;
  }

  const std::string &get_current_working_path() const {
    return _current_working_path;
  }

  const std::vector<std::string> &get_asset_search_paths() const {
    return _asset_search_paths;
  }

  void set_current_working_path(const std::string &s) {
    _current_working_path = s;
  }

  void set_asset_search_paths(const std::vector<std::string> &search_paths) {
    _asset_search_paths = search_paths;
  }

  void set_asset_resolution_state(
    const std::string &cwp, const std::vector<std::string> &search_paths) {
    _current_working_path = cwp;
    _asset_search_paths = search_paths;
  }

 private:
  void CopyFrom(const PrimSpec &rhs) {
    _specifier = rhs._specifier;
    _typeName = rhs._typeName;
    _name = rhs._name;

    _children = rhs._children;

    _props = rhs._props;

    //_vsmap = rhs._vsmap;
    _current_vsmap = rhs._current_vsmap;

    _variantSets = rhs._variantSets;

    _primChildren = rhs._primChildren;
    _properties = rhs._properties;
    _variantChildren = rhs._variantChildren;

    _metas = rhs._metas;

    _current_working_path = rhs._current_working_path;
    _asset_search_paths = rhs._asset_search_paths;
  }

  void MoveFrom(PrimSpec &rhs) {
    _specifier = std::move(rhs._specifier);
    _typeName = std::move(rhs._typeName);
    _name = std::move(rhs._name);

    _children = std::move(rhs._children);

    _props = std::move(rhs._props);

    //_vsmap = std::move(rhs._vsmap);
    _current_vsmap = std::move(rhs._current_vsmap);

    _variantSets = std::move(rhs._variantSets);

    _primChildren = std::move(rhs._primChildren);
    _properties = std::move(rhs._properties);
    _variantChildren = std::move(rhs._variantChildren);

    _metas = std::move(rhs._metas);

    _current_working_path = rhs._current_working_path;
    _asset_search_paths = std::move(rhs._asset_search_paths);
  }

  Specifier _specifier{Specifier::Def};
  std::string _typeName;  // prim's typeName(e.g. "Xform", "Material") This is
                          // identitical to `typeName` in Crate format)
  std::string _name;      // elementName. Should not be empty.

  std::vector<PrimSpec> _children;  // child nodes

  PropertyMap _props;

  ///
  /// Variants
  ///
  /// variant element = Property or Prim
  ///
  using PrimSpecMap = std::map<std::string, PrimSpec>;

  //VariantSelectionMap _vsmap;  // Original variant selections
  VariantSelectionMap _current_vsmap;  // Currently selected variants

  std::map<std::string, VariantSetSpec> _variantSets;

  std::vector<value::token> _primChildren;  // List of child PrimSpec nodes
  std::vector<value::token> _properties;    // List of property names
  std::vector<value::token> _variantChildren;

  PrimMeta _metas;

  ///
  /// For solving asset path in nested composition.
  /// Keep asset resolution state.
  /// TODO: Use struct. Store userdata pointer.
  ///
  std::string _current_working_path;
  std::vector<std::string> _asset_search_paths;

};

struct SubLayer
{
  value::AssetPath assetPath;
  LayerOffset layerOffset;
};


struct LayerMetas {
  enum class PlaybackMode {
    PlaybackModeNone,
    PlaybackModeLoop,
  };

  // TODO: Support more predefined properties: reference =
  // <pxrUSD>/pxr/usd/sdf/wrapLayer.cpp Scene global setting
  TypedAttributeWithFallback<Axis> upAxis{
      Axis::
          Y};  // This can be changed by plugInfo.json in USD:
               // https://graphics.pixar.com/usd/dev/api/group___usd_geom_up_axis__group.html#gaf16b05f297f696c58a086dacc1e288b5
  value::token defaultPrim;                               // prim node name
  TypedAttributeWithFallback<double> metersPerUnit{1.0};  // default [m]
  TypedAttributeWithFallback<double> timeCodesPerSecond{
      24.0};  // default 24 fps
  TypedAttributeWithFallback<double> framesPerSecond{24.0};
  TypedAttributeWithFallback<double> startTimeCode{
      0.0};  // FIXME: default = -inf?
  TypedAttributeWithFallback<double> endTimeCode{
      std::numeric_limits<double>::infinity()};
  std::vector<SubLayer> subLayers;  // `subLayers`
  value::StringData comment;  // 'comment' In Stage meta, comment must be string
                              // only(`comment = "..."` is not allowed)
  value::StringData doc;      // `documentation`

  // UsdPhysics
  TypedAttributeWithFallback<double> kilogramsPerUnit{1.0};

  CustomDataType customLayerData;  // customLayerData

  // USDZ extension
  TypedAttributeWithFallback<bool> autoPlay{
      true};  // default(or not authored) = auto play
  TypedAttributeWithFallback<PlaybackMode> playbackMode{
      PlaybackMode::PlaybackModeLoop};

  // Indirectly used.
  std::vector<value::token> primChildren;
};


// Similar to SdfLayer or Stage
// It is basically hold the list of PrimSpec and Layer metadatum.
class Layer {
 public:
  const std::string name() const { return _name; }

  void set_name(const std::string name) { _name = name; }

  void clear_primspecs() { _prim_specs.clear(); }

  // Check if `primname` exists in root Prims?
  bool has_primspec(const std::string &primname) const {
    return _prim_specs.count(primname) > 0;
  }

  ///
  /// Add PrimSpec(copy PrimSpec instance).
  ///
  /// @return false when `name` already exists in `primspecs`, `name` is empty
  /// string or `name` contains invalid character to be used in Prim
  /// element_name.
  ///
  bool add_primspec(const std::string &name, const PrimSpec &ps) {
    if (name.empty()) {
      return false;
    }

    if (!ValidatePrimElementName(name)) {
      return false;
    }

    if (has_primspec(name)) {
      return false;
    }

    _prim_specs.emplace(name, ps);

    return true;
  }

  ///
  /// Add PrimSpec.
  ///
  /// @return false when `name` already exists in `primspecs`, `name` is empty
  /// string or `name` contains invalid character to be used in Prim
  /// element_name.
  ///
  bool emplace_primspec(const std::string &name, PrimSpec &&ps) {
    if (name.empty()) {
      return false;
    }

    if (!ValidatePrimElementName(name)) {
      return false;
    }

    if (has_primspec(name)) {
      return false;
    }

    _prim_specs.emplace(name, std::move(ps));

    return true;
  }

  ///
  /// Replace PrimSpec(copy PrimSpec instance)
  ///
  /// @return false when `name` does not exist in `primspecs`, `name` is empty
  /// string or `name` contains invalid character to be used in Prim
  /// element_name.
  ///
  bool replace_primspec(const std::string &name, const PrimSpec &ps) {
    if (name.empty()) {
      return false;
    }

    if (!ValidatePrimElementName(name)) {
      return false;
    }

    if (!has_primspec(name)) {
      return false;
    }

    _prim_specs.at(name) = ps;

    return true;
  }

  ///
  /// Replace PrimSpec
  ///
  /// @return false when `name` does not exist in `primspecs`, `name` is empty
  /// string or `name` contains invalid character to be used in Prim
  /// element_name.
  ///
  bool replace_primspec(const std::string &name, PrimSpec &&ps) {
    if (name.empty()) {
      return false;
    }

    if (!ValidatePrimElementName(name)) {
      return false;
    }

    if (!has_primspec(name)) {
      return false;
    }

    _prim_specs.at(name) = std::move(ps);

    return true;
  }

  const std::unordered_map<std::string, PrimSpec> &primspecs() const {
    return _prim_specs;
  }

  std::unordered_map<std::string, PrimSpec> &primspecs() { return _prim_specs; }

  const LayerMetas &metas() const { return _metas; }
  LayerMetas &metas() { return _metas; }

  bool has_unresolved_references() const {
    return _has_unresolved_references;
  }

  bool has_unresolved_payload() const {
    return _has_unresolved_payload;
  }

  bool has_unresolved_variant() const {
    return _has_unresolved_variant;
  }

  bool has_over_primspec() const {
    return _has_over_primspec;
  }

  bool has_class_primspec() const {
    return _has_class_primspec;
  }

  bool has_unresolved_inherits() const {
    return _has_unresolved_inherits;
  }

  bool has_unresolved_specializes() const {
    return _has_unresolved_specializes;
  }

  ///
  /// Check if PrimSpec tree contains any `references` and cache the result.
  ///
  /// @param[in] max_depth Maximum PrimSpec traversal depth.
  /// @returns true if PrimSpec tree contains any (unresolved) `references`. false if not.
  ///
  bool check_unresolved_references(const uint32_t max_depth = 1024 * 1024) const;

  ///
  /// Check if PrimSpec tree contains any `payload` and cache the result.
  ///
  /// @param[in] max_depth Maximum PrimSpec traversal depth.
  /// @returns true if PrimSpec tree contains any (unresolved) `payload`. false if not.
  ///
  bool check_unresolved_payload(const uint32_t max_depth = 1024 * 1024) const;

  ///
  /// Check if PrimSpec tree contains any `variant` and cache the result.
  ///
  /// @param[in] max_depth Maximum PrimSpec traversal depth.
  /// @returns true if PrimSpec tree contains any (unresolved) `variant`. false if not.
  ///
  bool check_unresolved_variant(const uint32_t max_depth = 1024 * 1024) const;

  ///
  /// Check if PrimSpec tree contains any `specializes` and cache the result.
  ///
  /// @param[in] max_depth Maximum PrimSpec traversal depth.
  /// @returns true if PrimSpec tree contains any (unresolved) `specializes`. false if not.
  ///
  bool check_unresolved_specializes(const uint32_t max_depth = 1024 * 1024) const;

  ///
  /// Check if PrimSpec tree contains any `inherits` and cache the result.
  ///
  /// @param[in] max_depth Maximum PrimSpec traversal depth.
  /// @returns true if PrimSpec tree contains any (unresolved) `inherits`. false if not.
  ///
  bool check_unresolved_inherits(const uint32_t max_depth = 1024 * 1024) const;

  ///
  /// Check if PrimSpec tree contains any Prim with `over` specifier and cache the result.
  ///
  /// @param[in] max_depth Maximum PrimSpec traversal depth.
  /// @returns true if PrimSpec tree contains any Prim with `over` specifier. false if not.
  ///
  bool check_over_primspec(const uint32_t max_depth = 1024 * 1024) const;

  ///
  /// Find a PrimSpec at `path` and returns it if found.
  ///
  /// @param[in] path PrimSpec path to find.
  /// @param[out] ps Pointer to PrimSpec pointer
  /// @param[out] err Error message
  ///
  bool find_primspec_at(const Path &path, const PrimSpec **ps, std::string *err) const;


  ///
  /// Set state for AssetResolution in the subsequent composition operation.
  ///
  void set_asset_resolution_state(
    const std::string &cwp, const std::vector<std::string> &search_paths, void *userdata=nullptr) {
    _current_working_path = cwp;
    _asset_search_paths = search_paths;
    _asset_resolution_userdata = userdata;
  }

  void get_asset_resolution_state(
    std::string &cwp, std::vector<std::string> &search_paths, void *&userdata) {
    cwp = _current_working_path;
    search_paths = _asset_search_paths;
    userdata = _asset_resolution_userdata;
  }

  const std::string get_current_working_path() const {
    return _current_working_path;
  }

  const std::vector<std::string> get_asset_search_paths() const {
    return _asset_search_paths;
  }

 private:
  std::string _name;  // layer name ~= USD filename

  // key = prim name
  std::unordered_map<std::string, PrimSpec> _prim_specs;
  LayerMetas _metas;

#if defined(TINYUSDZ_ENABLE_THREAD)
  mutable std::mutex _mutex;
#endif

  // Cached primspec path.
  // key : prim_part string (e.g. "/path/bora")
  mutable std::map<std::string, const PrimSpec *> _primspec_path_cache;
  mutable bool _dirty{true};

  // Cached flags for composition.
  // true by default even PrimSpec tree does not contain any `references`, `payload`, etc.
  mutable bool _has_unresolved_references{true};
  mutable bool _has_unresolved_payload{true};
  mutable bool _has_unresolved_variant{true};
  mutable bool _has_unresolved_inherits{true};
  mutable bool _has_unresolved_specializes{true};
  mutable bool _has_over_primspec{true};
  mutable bool _has_class_primspec{true};

  //
  // Record AssetResolution state(search paths, current working directory)
  // when this layer is opened by compostion(`references`, `payload`, `subLayers`)
  //
  mutable std::string _current_working_path;
  mutable std::vector<std::string> _asset_search_paths;
  mutable void *_asset_resolution_userdata{nullptr};

};


nonstd::optional<Interpolation> InterpolationFromString(const std::string &v);
nonstd::optional<Orientation> OrientationFromString(const std::string &v);
nonstd::optional<Kind> KindFromString(const std::string &v);

namespace value {

#include "define-type-trait.inc"

DEFINE_TYPE_TRAIT(Reference, "ref", TYPE_ID_REFERENCE, 1);
DEFINE_TYPE_TRAIT(Specifier, "specifier", TYPE_ID_SPECIFIER, 1);
DEFINE_TYPE_TRAIT(Permission, "permission", TYPE_ID_PERMISSION, 1);
DEFINE_TYPE_TRAIT(Variability, "variability", TYPE_ID_VARIABILITY, 1);

DEFINE_TYPE_TRAIT(VariantSelectionMap, "variants", TYPE_ID_VARIANT_SELECION_MAP,
                  0);

DEFINE_TYPE_TRAIT(Payload, "payload", TYPE_ID_PAYLOAD, 1);
DEFINE_TYPE_TRAIT(LayerOffset, "LayerOffset", TYPE_ID_LAYER_OFFSET, 1);

DEFINE_TYPE_TRAIT(ListOp<value::token>, "ListOpToken", TYPE_ID_LIST_OP_TOKEN,
                  1);
DEFINE_TYPE_TRAIT(ListOp<std::string>, "ListOpString", TYPE_ID_LIST_OP_STRING,
                  1);
DEFINE_TYPE_TRAIT(ListOp<Path>, "ListOpPath", TYPE_ID_LIST_OP_PATH, 1);
DEFINE_TYPE_TRAIT(ListOp<Reference>, "ListOpReference",
                  TYPE_ID_LIST_OP_REFERENCE, 1);
DEFINE_TYPE_TRAIT(ListOp<int32_t>, "ListOpInt", TYPE_ID_LIST_OP_INT, 1);
DEFINE_TYPE_TRAIT(ListOp<uint32_t>, "ListOpUInt", TYPE_ID_LIST_OP_UINT, 1);
DEFINE_TYPE_TRAIT(ListOp<int64_t>, "ListOpInt64", TYPE_ID_LIST_OP_INT64, 1);
DEFINE_TYPE_TRAIT(ListOp<uint64_t>, "ListOpUInt64", TYPE_ID_LIST_OP_UINT64, 1);
DEFINE_TYPE_TRAIT(ListOp<Payload>, "ListOpPayload", TYPE_ID_LIST_OP_PAYLOAD, 1);

DEFINE_TYPE_TRAIT(Path, "Path", TYPE_ID_PATH, 1);
DEFINE_TYPE_TRAIT(Relationship, "Relationship", TYPE_ID_RELATIONSHIP, 1);
// TODO(syoyo): Define as 1D array?
DEFINE_TYPE_TRAIT(std::vector<Path>, "PathVector", TYPE_ID_PATH_VECTOR, 1);

DEFINE_TYPE_TRAIT(std::vector<value::token>, "token[]", TYPE_ID_TOKEN_VECTOR,
                  1);

DEFINE_TYPE_TRAIT(value::TimeSamples, "TimeSamples", TYPE_ID_TIMESAMPLES, 1);

DEFINE_TYPE_TRAIT(Collection, "Collection", TYPE_ID_COLLECTION, 1);
DEFINE_TYPE_TRAIT(CollectionInstance, "CollectionInstance", TYPE_ID_COLLECTION_INSTANCE, 1);

DEFINE_TYPE_TRAIT(Model, "Model", TYPE_ID_MODEL, 1);
DEFINE_TYPE_TRAIT(Scope, "Scope", TYPE_ID_SCOPE, 1);

DEFINE_TYPE_TRAIT(CustomDataType, "customData", TYPE_ID_CUSTOMDATA,
                  1);  // TODO: Unify with `dict`?

DEFINE_TYPE_TRAIT(Extent, "float3[]", TYPE_ID_EXTENT, 2);  // float3[2]

#undef DEFINE_TYPE_TRAIT
#undef DEFINE_ROLE_TYPE_TRAIT

}  // namespace value

namespace prim {

using PropertyMap = std::map<std::string, Property>;
using ReferenceList = std::pair<ListEditQual, std::vector<Reference>>;
using PayloadList = std::pair<ListEditQual, std::vector<Payload>>;

}  // namespace prim


// TODO(syoyo): Range, Interval, Rect2i, Frustum, MultiInterval
// and Quaternion?

/*
#define VT_GFRANGE_VALUE_TYPES                 \
((      GfRange3f,           Range3f        )) \
((      GfRange3d,           Range3d        )) \
((      GfRange2f,           Range2f        )) \
((      GfRange2d,           Range2d        )) \
((      GfRange1f,           Range1f        )) \
((      GfRange1d,           Range1d        ))

#define VT_RANGE_VALUE_TYPES                   \
    VT_GFRANGE_VALUE_TYPES                     \
((      GfInterval,          Interval       )) \
((      GfRect2i,            Rect2i         ))

#define VT_QUATERNION_VALUE_TYPES           \
((      GfQuaternion,        Quaternion ))

#define VT_NONARRAY_VALUE_TYPES                 \
((      GfFrustum,           Frustum))          \
((      GfMultiInterval,     MultiInterval))

*/

}  // namespace tinyusdz

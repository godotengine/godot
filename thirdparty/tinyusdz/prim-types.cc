// SPDX-License-Identifier: MIT
// Copyright 2021 - Present, Syoyo Fujita.
#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
//
#include "prim-types.hh"
#include "str-util.hh"
#include "tiny-format.hh"
//
#include "usdGeom.hh"
#include "usdLux.hh"
#include "usdShade.hh"
#include "usdSkel.hh"
//
#include "common-macros.inc"
#include "pprinter.hh"
#include "value-pprint.hh"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

//#include "external/pystring.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#define PushError(msg) \
  do {                 \
    if (err) {         \
      (*err) += msg;   \
    }                  \
  } while (0)

namespace tinyusdz {

template<class InputIt1, class InputIt2>
bool lexicographical_compare(InputIt1 first1, InputIt1 last1,
                             InputIt2 first2, InputIt2 last2)
{
    for (; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2)
    {
        if (*first1 < *first2)
            return true;
        if (*first2 < *first1)
            return false;
    }
 
    return (first1 == last1) && (first2 != last2);
}

nonstd::optional<Interpolation> InterpolationFromString(const std::string &v) {
  if ("faceVarying" == v) {
    return Interpolation::FaceVarying;
  } else if ("constant" == v) {
    return Interpolation::Constant;
  } else if ("uniform" == v) {
    return Interpolation::Uniform;
  } else if ("vertex" == v) {
    return Interpolation::Vertex;
  } else if ("varying" == v) {
    return Interpolation::Varying;
  }
  return nonstd::nullopt;
}

nonstd::optional<Orientation> OrientationFromString(const std::string &v) {
  if ("rightHanded" == v) {
    return Orientation::RightHanded;
  } else if ("leftHanded" == v) {
    return Orientation::LeftHanded;
  }
  return nonstd::nullopt;
}

bool operator==(const Path &lhs, const Path &rhs) {
  if (!lhs.is_valid()) {
    return false;
  }

  if (!rhs.is_valid()) {
    return false;
  }

  // Currently simply compare string.
  // FIXME: Better Path identity check.
  return (lhs.full_path_name() == rhs.full_path_name());
}

bool ConvertTokenAttributeToStringAttribute(
    const TypedAttribute<Animatable<value::token>> &inp,
    TypedAttribute<Animatable<std::string>> &out) {
  
    out.metas() = inp.metas();
  
    if (inp.is_blocked()) {
      out.set_blocked(true);
    } else if (inp.is_value_empty()) {
      out.set_value_empty();
    } else if (inp.is_connection()) {
      out.set_connections(inp.get_connections());
    } else {
      Animatable<value::token> toks;
      Animatable<std::string> strs;
      if (inp.get_value(&toks)) {
        if (toks.is_scalar()) {
          value::token tok;
          toks.get_scalar(&tok);
          strs.set(tok.str());
        } else if (toks.is_timesamples()) {
          auto tok_ts = toks.get_timesamples();
  
          for (auto &item : tok_ts.get_samples()) {
            strs.add_sample(item.t, item.value.str());
          }
        } else if (toks.is_blocked()) {
          // TODO
          return false;
        }
      }
      out.set_value(strs);
    }
  
    return true;
  }
  


//
// -- Path
//

void Path::_update(const std::string &p, const std::string &prop) {
  //
  // For absolute path, starts with '/' and no other '/' exists.
  // For property part, '.' exists only once.
  //

  if (p.empty() && prop.empty()) {
    _valid = false;
    return;
  }

  auto slash_fun = [](const char c) { return c == '/'; };
  auto dot_fun = [](const char c) { return c == '.'; };

  std::vector<std::string> prims = split(p, "/");

  // TODO: More checks('{', '[', ...)

  if (prop.size()) {
    // prop should not contain slashes
    auto nslashes = std::count_if(prop.begin(), prop.end(), slash_fun);
    if (nslashes) {
      _valid = false;
      return;
    }

    // prop does not start with '.'
    if (startsWith(prop, ".")) {
      _valid = false;
      return;
    }
  }

  if (p[0] == '/') {
    // absolute path

    auto ndots = std::count_if(p.begin(), p.end(), dot_fun);

    if (ndots == 0) {
      // absolute prim.
      _prim_part = p;

      if (prop.size()) {
        _prop_part = prop;
        _element = prop;
      } else {
        if (prims.size()) {
          _element = prims[prims.size() - 1];
        } else {
          _element = p;
        }
      }
      _valid = true;
    } else if (ndots == 1) {
      // prim_part contains property name.
      if (prop.size()) {
        // prop must be empty.
        _valid = false;
        return;
      }

      if (p.size() < 3) {
        // "/."
        _valid = false;
        return;
      }

      auto loc = p.find_first_of('.');
      if (loc == std::string::npos) {
        // ?
        _valid = false;
        return;
      }

      if (loc <= 0) {
        // this should not happen though.
        _valid = false;
      }

      // split
      std::string prop_name = p.substr(size_t(loc));

      _prop_part = prop_name.erase(0, 1);  // remove '.'
      _prim_part = p.substr(0, size_t(loc));
      _element = _prop_part;  // elementName is property path

      _valid = true;

    } else {
      _valid = false;
      return;
    }

  } else if (p[0] == '.') {
    // maybe relative(e.g. "./xform", "../xform")
    // FIXME: Support relative path fully

#if 0
    auto nslashes = std::count_if(p.begin(), p.end(), slash_fun);
    if (nslashes > 0) {
      _valid = false;
      return;
    }

    _prop_part = p;
    _prop_part = _prop_part.erase(0, 1);
    _valid = true;
#else
    _prim_part = p;
    if (prop.size()) {
      _prop_part = prop;
      _element = prop;
    } else {
      if (prims.size()) {
        _element = prims[prims.size() - 1];
      } else {
        _element = p;
      }
    }
    _valid = true;

#endif

  } else {
    // prim.prop

    auto ndots = std::count_if(p.begin(), p.end(), dot_fun);
    if (ndots == 0) {
      // relative prim.
      _prim_part = p;
      if (prop.size()) {
        _prop_part = prop;
      }
      _valid = true;
    } else if (ndots == 1) {
      if (p.size() < 3) {
        // "/."
        _valid = false;
        return;
      }

      auto loc = p.find_first_of('.');
      if (loc == std::string::npos) {
        // ?
        _valid = false;
        return;
      }

      if (loc <= 0) {
        // this should not happen though.
        _valid = false;
      }

      // split
      std::string prop_name = p.substr(size_t(loc));

      // Check if No '/' in prop_part
      if (std::count_if(prop_name.begin(), prop_name.end(), slash_fun) > 0) {
        _valid = false;
        return;
      }

      _prim_part = p.substr(0, size_t(loc));
      _prop_part = prop_name.erase(0, 1);  // remove '.'

      _valid = true;

    } else {
      _valid = false;
      return;
    }
  }
}

Path::Path(const std::string &p, const std::string &prop) {
  _update(p, prop); 
}

Path Path::append_property(const std::string &elem) {
  Path &p = (*this);

  if (elem.empty()) {
    p._valid = false;
    return p;
  }

  if (is_variantElementName(elem)) {
    // variant chars are not supported yet.
    p._valid = false;
    return p;
  }

  if (elem[0] == '[') {
    // relational attrib are not supported
    p._valid = false;
    return p;
  } else if (elem[0] == '.') {
    // Relative
    // std::cerr << "???. elem[0] is '.'\n";
    // For a while, make this valid.
    p._valid = false;
    return p;
  } else {
    // TODO: Validate property path.
    p._prop_part = elem;
    p._element = elem;

    return p;
  }
}

const Path Path::AppendPrim(const std::string &elem) const {
  Path p = (*this);  // copies

  p.append_prim(elem);

  return p;
}

const Path Path::AppendElement(const std::string &elem) const {
  Path p = (*this);  // copies

  p.append_element(elem);

  return p;
}

const Path Path::AppendProperty(const std::string &elem) const {
  Path p = (*this);  // copies

  p.append_property(elem);

  return p;
}

bool Path::replace_prefix(const Path &srcPrefix, const Path &dstPrefix) {
  const std::string &srcPrefixStr = srcPrefix.prim_part();
  const std::string &dstPrefixStr = dstPrefix.prim_part();

  std::string pathStr = prim_part();
  if (startsWith(pathStr, srcPrefixStr)) {
    pathStr = dstPrefixStr + removePrefix(pathStr, srcPrefixStr);

    _update(pathStr, prop_part());

    return true;
  }

  return false;
}

// TODO: Do test more.
// Current implementation may not behave as in pxrUSD's SdfPath's
// _LessThanInternal implementation
bool Path::LessThan(const Path &lhs, const Path &rhs) {
  // DCOUT("LessThan");
  if (lhs.is_valid() && rhs.is_valid()) {
    // ok
  } else {
    // Even though this should not happen,
    // valid paths is less than invalid paths
    return lhs.is_valid();
  }

  // TODO: handle relative path correctly.
  if (lhs.is_absolute_path() && rhs.is_absolute_path()) {
    // ok
  } else {
    // Absolute paths are less than relative paths
    return lhs.is_absolute_path();
  }

  if (lhs.prim_part() == rhs.prim_part()) {
    // compare property
    const std::string &lhs_prop_part = lhs.prop_part();
    const std::string &rhs_prop_part = rhs.prop_part();

    if (lhs_prop_part.empty() || rhs_prop_part.empty()) {
      return lhs_prop_part.empty();
    }

    return ::tinyusdz::lexicographical_compare(
        lhs_prop_part.begin(), lhs_prop_part.end(), rhs_prop_part.begin(),
        rhs_prop_part.end());

  } else {
    const std::vector<std::string> lhs_prim_names = split(lhs.prim_part(), "/");
    const std::vector<std::string> rhs_prim_names = split(rhs.prim_part(), "/");
    // DCOUT("lhs_names = " << to_string(lhs_prim_names));
    // DCOUT("rhs_names = " << to_string(rhs_prim_names));

    if (lhs_prim_names.empty() || rhs_prim_names.empty()) {
      return lhs_prim_names.empty() && rhs_prim_names.size();
    }

    // common shortest depth.
    size_t didx = (std::min)(lhs_prim_names.size(), rhs_prim_names.size());

    bool same_until_common_depth = true;
    for (size_t i = 0; i < didx; i++) {
      if (lhs_prim_names[i] != rhs_prim_names[i]) {
        same_until_common_depth = false;
        break;
      }
    }

    if (same_until_common_depth) {
      // tail differs. compare by depth count.
      return lhs_prim_names.size() < rhs_prim_names.size();
    }

    // Walk until common ancestor is found
    size_t child_idx = didx - 1;
    // DCOUT("common_depth_idx = " << didx << ", lcount = " <<
    // lhs_prim_names.size() << ", rcount = " << rhs_prim_names.size());
    if (didx > 1) {
      for (size_t parent_idx = didx - 2; parent_idx > 0; parent_idx--) {
        // DCOUT("parent_idx = " << parent_idx);
        if (lhs_prim_names[parent_idx] != rhs_prim_names[parent_idx]) {
          child_idx--;
        }
      }
    }
    // DCOUT("child_idx = " << child_idx);

    // compare child node
    return ::tinyusdz::lexicographical_compare(
        lhs_prim_names[child_idx].begin(), lhs_prim_names[child_idx].end(),
        rhs_prim_names[child_idx].begin(), rhs_prim_names[child_idx].end());
  }
}

std::pair<Path, Path> Path::split_at_root() const {
  if (is_absolute_path()) {
    if (is_root_path()) {
      return std::make_pair(Path("/", ""), Path());
    }

    std::string p = full_path_name();

    if (p.size() < 2) {
      // Never should reach here. just in case
      return std::make_pair(*this, Path());
    }

    // Fine 2nd '/'
    auto ret =
        std::find_if(p.begin() + 1, p.end(), [](char c) { return c == '/'; });

    if (ret != p.end()) {
      auto ndist = std::distance(p.begin(), ret);  // distance from str[0]
      if (ndist < 1) {
        // This should not happen though.
        return std::make_pair(*this, Path());
      }
      size_t n = size_t(ndist);
      std::string root = p.substr(0, n);
      std::string siblings = p.substr(n);

      Path rP(root, "");
      Path sP(siblings, "");

      return std::make_pair(rP, sP);
    }

    return std::make_pair(*this, Path());
  } else {
    return std::make_pair(Path(), *this);
  }
}

bool Path::has_prefix(const Path &prefix) const {
  if (!is_valid() || !prefix.is_valid()) {
    return false;
  }

  if (prefix.is_prim_property_path()) {
    // No hierarchy in Prim's property path, so use ==.
    return full_path_name() == prefix.full_path_name();
  } else if (prefix.is_prim_path()) {
    // '/', prefix = '/'
    if (is_root_path() && prefix.is_root_path()) {
      // DCOUT("both are root path");
      return true;
    }

    // example:
    // - '/bora', prefix = '/'
    // - '/bora/dora', prefix = '/'
    if (is_absolute_path() && prefix.is_root_path()) {
      // DCOUT("prefix is root path");
      return true;
    }

    const std::vector<std::string> prim_names = split(prim_part(), "/");
    const std::vector<std::string> prefix_prim_names =
        split(prefix.prim_part(), "/");
    // DCOUT("prim_names = " << to_string(prim_names));
    // DCOUT("prefix.prim_names = " << to_string(prefix_prim_names));

    if (prim_names.empty() || prefix_prim_names.empty()) {
      return false;
    }

    if (prim_names.size() < prefix_prim_names.size()) {
      return false;
    }

    size_t depth = prefix_prim_names.size();
    if (depth < 1) {  // just in case
      return false;
    }

    // Move to prefix's path depth and compare each elementName of Prim tree
    // towards the root. comapre from tail would find a difference earlier.
    while (depth > 0) {
      if (prim_names[depth - 1] != prefix_prim_names[depth - 1]) {
        return false;
      }
      depth--;
    }

    // DCOUT("has_prefix");
    return true;

  } else {
    // TODO: property-only path.
    DCOUT("TODO: Unsupported Path type in has_prefix()");
    return false;
  }
}

Path Path::append_element(const std::string &elem) {
  Path &p = (*this);

  if (elem.empty()) {
    p._valid = false;
    return p;
  }

  // {variant=value}
  if (is_variantElementName(elem)) {
    std::array<std::string, 2> variant;
    if (tokenize_variantElement(elem, &variant)) {
      _variant_part = variant[0];
      _variant_selection_part = variant[0];
      _prim_part += elem;
      _element = elem;
      return p;
    } else {
      p._valid = false;
    }
  }

  if (elem[0] == '[') {
    // relational attrib are not supported
    p._valid = false;
    return p;
  } else if (elem[0] == '.') {
    // Relative path
    // For a while, make this valid.
    p._valid = false;
    return p;
  } else {
    // std::cout << "elem " << elem << "\n";
    if ((p._prim_part.size() == 1) && (p._prim_part[0] == '/')) {
      p._prim_part += elem;
    } else {
      // TODO: Validate element name.
      p._prim_part += '/' + elem;
    }

    // Also store raw element name
    p._element = elem;

    return p;
  }
}

Path Path::get_parent_path() const {
  if (!_valid) {
    return Path();
  }

  if (is_root_path()) {
    Path p("", "");
    return p;
  }

  if (is_prim_property_path()) {
    // return prim part
    return Path(prim_part(), "");
  }

  size_t n = _prim_part.find_last_of('/');
  if (n == std::string::npos) {
    // relative path(e.g. "bora") or propery only path(e.g. ".myval").
    return Path();
  }

  if (n == 0) {
    // return root
    return Path("/", "");
  }

  return Path(_prim_part.substr(0, n), "");
}

Path Path::get_parent_prim_path() const {
  if (!_valid) {
    return Path();
  }

  if (is_root_prim()) {
    return *this;
  }

  if (is_prim_property_path()) {
    // return prim part
    return Path(prim_part(), "");
  }

  size_t n = _prim_part.find_last_of('/');
  if (n == std::string::npos) {
    // this should never happen though.
    return Path();
  }

  if (n == 0) {
    // return root
    return Path("/", "");
  }

  return Path(_prim_part.substr(0, n), "");
}

const std::string &Path::element_name() const {
  if (_element.empty()) {
    // Get last item.
    std::vector<std::string> tokenized_prim_names = split(prim_part(), "/");
    if (tokenized_prim_names.size()) {
      _element = tokenized_prim_names[size_t(tokenized_prim_names.size() - 1)];
    }
  }

  return _element;
}

nonstd::optional<Kind> KindFromString(const std::string &str) {
  if (str == "model") {
    return Kind::Model;
  } else if (str == "group") {
    return Kind::Group;
  } else if (str == "assembly") {
    return Kind::Assembly;
  } else if (str == "component") {
    return Kind::Component;
  } else if (str == "subcomponent") {
    return Kind::Subcomponent;
  } else if (str == "sceneLibrary") {
    // https://developer.apple.com/documentation/arkit/usdz_schemas_for_ar/scenelibrary
    return Kind::SceneLibrary;
  } else if (str.empty()) {
    return nonstd::nullopt;
  } else {
    return Kind::UserDef;
  }
}

bool ValidatePrimElementName(const std::string &name) {
  if (name.empty()) {
    return false;
  }

  // alphanum + '_'
  // first char must not be number.

  if (std::isdigit(int(name[0]))) {
    return false;
  } else if (std::isalpha(int(name[0]))) {
    // ok
  } else if (name[0] == '_') {
    // ok
  } else {
    return false;
  }

  for (size_t i = 1; i < name.size(); i++) {
    if (std::isalnum(int(name[i])) || (name[i] == '_')) {
      // ok
    } else {
      return false;
    }
  }

  return true;
}

//
// -- Prim
//

namespace {

const PrimMeta *GetPrimMeta(const value::Value &v) {
  // Lookup PrimMeta variable in Prim class

#define GET_PRIM_META(__ty)       \
  if (v.as<__ty>()) {             \
    return &(v.as<__ty>()->meta); \
  }

  GET_PRIM_META(Model)
  GET_PRIM_META(Scope)
  GET_PRIM_META(Xform)
  GET_PRIM_META(GPrim)
  GET_PRIM_META(GeomMesh)
  GET_PRIM_META(GeomPoints)
  GET_PRIM_META(GeomCube)
  GET_PRIM_META(GeomCapsule)
  GET_PRIM_META(GeomCylinder)
  GET_PRIM_META(GeomSphere)
  GET_PRIM_META(GeomCone)
  GET_PRIM_META(GeomSubset)
  GET_PRIM_META(GeomCamera)
  GET_PRIM_META(GeomBasisCurves)
  GET_PRIM_META(DomeLight)
  GET_PRIM_META(SphereLight)
  GET_PRIM_META(CylinderLight)
  GET_PRIM_META(DiskLight)
  GET_PRIM_META(RectLight)
  GET_PRIM_META(Material)
  GET_PRIM_META(Shader)
  // GET_PRIM_META(UsdPreviewSurface)
  // GET_PRIM_META(UsdUVTexture)
  // GET_PRIM_META(UsdPrimvarReader_int)
  // GET_PRIM_META(UsdPrimvarReader_float)
  // GET_PRIM_META(UsdPrimvarReader_float2)
  // GET_PRIM_META(UsdPrimvarReader_float3)
  // GET_PRIM_META(UsdPrimvarReader_float4)
  GET_PRIM_META(SkelRoot)
  GET_PRIM_META(Skeleton)
  GET_PRIM_META(SkelAnimation)
  GET_PRIM_META(BlendShape)

#undef GET_PRIM_META

  return nullptr;
}

PrimMeta *GetPrimMeta(value::Value &v) {
  // Lookup PrimMeta variable in Prim class

#define GET_PRIM_META(__ty)       \
  if (v.as<__ty>()) {             \
    return &(v.as<__ty>()->meta); \
  }

  GET_PRIM_META(Model)
  GET_PRIM_META(Scope)
  GET_PRIM_META(Xform)
  GET_PRIM_META(GPrim)
  GET_PRIM_META(GeomMesh)
  GET_PRIM_META(GeomPoints)
  GET_PRIM_META(GeomCube)
  GET_PRIM_META(GeomCapsule)
  GET_PRIM_META(GeomCylinder)
  GET_PRIM_META(GeomSphere)
  GET_PRIM_META(GeomCone)
  GET_PRIM_META(GeomSubset)
  GET_PRIM_META(GeomCamera)
  GET_PRIM_META(GeomBasisCurves)
  GET_PRIM_META(DomeLight)
  GET_PRIM_META(SphereLight)
  GET_PRIM_META(CylinderLight)
  GET_PRIM_META(DiskLight)
  GET_PRIM_META(RectLight)
  GET_PRIM_META(Material)
  GET_PRIM_META(Shader)
  // GET_PRIM_META(UsdPreviewSurface)
  // GET_PRIM_META(UsdUVTexture)
  // GET_PRIM_META(UsdPrimvarReader_int)
  // GET_PRIM_META(UsdPrimvarReader_float)
  // GET_PRIM_META(UsdPrimvarReader_float2)
  // GET_PRIM_META(UsdPrimvarReader_float3)
  // GET_PRIM_META(UsdPrimvarReader_float4)
  GET_PRIM_META(SkelRoot)
  GET_PRIM_META(Skeleton)
  GET_PRIM_META(SkelAnimation)
  GET_PRIM_META(BlendShape)

#undef GET_PRIM_META

  return nullptr;
}

}  // namespace

///
/// Stage
///

//
// TODO: Move to prim-types.cc
//

nonstd::optional<std::string> GetPrimElementName(const value::Value &v) {
  // Since multiple get_value() call consumes lots of stack size(depends on
  // sizeof(T)?), Following code would produce 100KB of stack in debug build. So
  // use as() instead(as() => roughly 2000 bytes for stack size).
#if 0
  //
  // TODO: Find a better C++ way... use a std::function?
  //
  if (auto pv = v.get_value<Model>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<Scope>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<Xform>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GPrim>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomMesh>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomBasisCurves>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomSphere>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomCube>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomCylinder>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomCapsule>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomCone>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomSubset>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<GeomCamera>()) {
    return Path(pv.value().name, "");
  }

  if (auto pv = v.get_value<DomeLight>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<SphereLight>()) {
    return Path(pv.value().name, "");
  }
  // if (auto pv = v.get_value<CylinderLight>()) { return
  // Path(pv.value().name); } if (auto pv = v.get_value<DiskLight>()) {
  // return Path(pv.value().name); }

  if (auto pv = v.get_value<Material>()) {
    return Path(pv.value().name, "");
  }
  if (auto pv = v.get_value<Shader>()) {
    return Path(pv.value().name, "");
  }
  // if (auto pv = v.get_value<UVTexture>()) { return Path(pv.value().name); }
  // if (auto pv = v.get_value<PrimvarReader()) { return Path(pv.value().name);
  // }

  return nonstd::nullopt;
#else

  // Lookup name field of Prim class

#define EXTRACT_NAME_AND_RETURN_PATH(__ty) \
  if (v.as<__ty>()) {                      \
    return v.as<__ty>()->name;             \
  } else

  EXTRACT_NAME_AND_RETURN_PATH(Model)
  EXTRACT_NAME_AND_RETURN_PATH(Scope)
  EXTRACT_NAME_AND_RETURN_PATH(Xform)
  EXTRACT_NAME_AND_RETURN_PATH(GPrim)
  EXTRACT_NAME_AND_RETURN_PATH(GeomMesh)
  EXTRACT_NAME_AND_RETURN_PATH(GeomPoints)
  EXTRACT_NAME_AND_RETURN_PATH(GeomCube)
  EXTRACT_NAME_AND_RETURN_PATH(GeomCapsule)
  EXTRACT_NAME_AND_RETURN_PATH(GeomCylinder)
  EXTRACT_NAME_AND_RETURN_PATH(GeomSphere)
  EXTRACT_NAME_AND_RETURN_PATH(GeomCone)
  EXTRACT_NAME_AND_RETURN_PATH(GeomSubset)
  EXTRACT_NAME_AND_RETURN_PATH(GeomCamera)
  EXTRACT_NAME_AND_RETURN_PATH(GeomBasisCurves)
  EXTRACT_NAME_AND_RETURN_PATH(DomeLight)
  EXTRACT_NAME_AND_RETURN_PATH(SphereLight)
  EXTRACT_NAME_AND_RETURN_PATH(CylinderLight)
  EXTRACT_NAME_AND_RETURN_PATH(DiskLight)
  EXTRACT_NAME_AND_RETURN_PATH(RectLight)
  EXTRACT_NAME_AND_RETURN_PATH(Material)
  EXTRACT_NAME_AND_RETURN_PATH(Shader)
  // TODO: extract name must be handled in Shader class
  EXTRACT_NAME_AND_RETURN_PATH(UsdPreviewSurface)
  EXTRACT_NAME_AND_RETURN_PATH(UsdUVTexture)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_int)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_float)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_float2)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_float3)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_float4)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_string)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_normal)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_vector)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_point)
  EXTRACT_NAME_AND_RETURN_PATH(UsdPrimvarReader_matrix)
  //
  EXTRACT_NAME_AND_RETURN_PATH(SkelRoot)
  EXTRACT_NAME_AND_RETURN_PATH(Skeleton)
  EXTRACT_NAME_AND_RETURN_PATH(SkelAnimation)
  EXTRACT_NAME_AND_RETURN_PATH(BlendShape) { return nonstd::nullopt; }

#undef EXTRACT_NAME_AND_RETURN_PATH

#endif
}

bool SetPrimElementName(value::Value &v, const std::string &elementName) {
  // Lookup name field of Prim class
  bool ok{false};

#define SET_ELEMENT_NAME(__name, __ty) \
  if (v.as<__ty>()) {                  \
    v.as<__ty>()->name = __name;       \
    ok = true;                         \
  } else

  SET_ELEMENT_NAME(elementName, Model)
  SET_ELEMENT_NAME(elementName, Scope)
  SET_ELEMENT_NAME(elementName, Xform)
  SET_ELEMENT_NAME(elementName, GPrim)
  SET_ELEMENT_NAME(elementName, GeomMesh)
  SET_ELEMENT_NAME(elementName, GeomPoints)
  SET_ELEMENT_NAME(elementName, GeomCube)
  SET_ELEMENT_NAME(elementName, GeomCapsule)
  SET_ELEMENT_NAME(elementName, GeomCylinder)
  SET_ELEMENT_NAME(elementName, GeomSphere)
  SET_ELEMENT_NAME(elementName, GeomCone)
  SET_ELEMENT_NAME(elementName, GeomSubset)
  SET_ELEMENT_NAME(elementName, GeomCamera)
  SET_ELEMENT_NAME(elementName, GeomBasisCurves)
  SET_ELEMENT_NAME(elementName, DomeLight)
  SET_ELEMENT_NAME(elementName, SphereLight)
  SET_ELEMENT_NAME(elementName, CylinderLight)
  SET_ELEMENT_NAME(elementName, DiskLight)
  SET_ELEMENT_NAME(elementName, RectLight)
  SET_ELEMENT_NAME(elementName, Material)
  SET_ELEMENT_NAME(elementName, Shader)
  // TODO: set element name must be handled in Shader class
  SET_ELEMENT_NAME(elementName, UsdPreviewSurface)
  SET_ELEMENT_NAME(elementName, UsdUVTexture)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_int)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_float)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_float2)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_float3)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_float4)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_string)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_normal)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_vector)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_point)
  SET_ELEMENT_NAME(elementName, UsdPrimvarReader_matrix)
  //
  SET_ELEMENT_NAME(elementName, SkelRoot)
  SET_ELEMENT_NAME(elementName, Skeleton)
  SET_ELEMENT_NAME(elementName, SkelAnimation)
  SET_ELEMENT_NAME(elementName, BlendShape) { return false; }

#undef SET_ELEMENT_NAME

  return ok;
}

Prim::Prim(const value::Value &rhs) {
  // Check if type is Prim(Model(GPrim), usdShade, usdLux, etc.)
  if ((value::TypeId::TYPE_ID_MODEL_BEGIN <= rhs.type_id()) &&
      (value::TypeId::TYPE_ID_MODEL_END > rhs.type_id())) {
    if (auto pv = GetPrimElementName(rhs)) {
      _path = Path(pv.value(), /* prop part*/ "");
      _elementPath = Path(pv.value(), /* prop part */ "");
    }

    _data = rhs;
  } else {
    // TODO: Raise an error if rhs is not an Prim
  }
}

Prim::Prim(value::Value &&rhs) {
  // Check if type is Prim(Model(GPrim), usdShade, usdLux, etc.)
  if ((value::TypeId::TYPE_ID_MODEL_BEGIN <= rhs.type_id()) &&
      (value::TypeId::TYPE_ID_MODEL_END > rhs.type_id())) {
    _data = std::move(rhs);

    if (auto pv = GetPrimElementName(_data)) {
      _path = Path(pv.value(), "");
      _elementPath = Path(pv.value(), "");
    }

  } else {
    // TODO: Raise an error if rhs is not an Prim
  }
}

Prim::Prim(const std::string &elementPath, const value::Value &rhs) {
  // Check if type is Prim(Model(GPrim), usdShade, usdLux, etc.)
  if ((value::TypeId::TYPE_ID_MODEL_BEGIN <= rhs.type_id()) &&
      (value::TypeId::TYPE_ID_MODEL_END > rhs.type_id())) {
    _path = Path(elementPath, /* prop part*/ "");
    _elementPath = Path(elementPath, /* prop part */ "");

    _data = rhs;
    SetPrimElementName(_data, elementPath);
  } else {
    // TODO: Raise an error if rhs is not an Prim
  }
}

Prim::Prim(const std::string &elementPath, value::Value &&rhs) {
  // Check if type is Prim(Model(GPrim), usdShade, usdLux, etc.)
  if ((value::TypeId::TYPE_ID_MODEL_BEGIN <= rhs.type_id()) &&
      (value::TypeId::TYPE_ID_MODEL_END > rhs.type_id())) {
    _path = Path(elementPath, /* prop part */ "");
    _elementPath = Path(elementPath, /* prop part */ "");

    _data = std::move(rhs);
    SetPrimElementName(_data, elementPath);
  } else {
    // TODO: Raise an error if rhs is not an Prim
  }
}

bool Prim::add_child(Prim &&rhs, const bool rename_prim_name,
                     std::string *err) {
#if defined(TINYUSDZ_ENABLE_THREAD)
  // TODO: Only take a lock when dirty.
  std::lock_guard<std::mutex> lock(_mutex);
#endif

  std::string elementName = rhs.element_name();

  if (elementName.empty()) {
    if (rename_prim_name) {
      // assign default name `default`
      elementName = "default";

      if (!SetPrimElementName(rhs.get_data(), elementName)) {
        if (err) {
          (*err) = fmt::format(
              "Internal error. cannot modify Prim's elementName.\n");
        }
        return false;
      }
      rhs.element_path() = Path(elementName, /* prop_part */ "");
    } else {
      if (err) {
        (*err) = "Prim has empty elementName.\n";
      }
      return false;
    }
  }

  if (_children.size() != _childrenNameSet.size()) {
    // Rebuild _childrenNames
    _childrenNameSet.clear();
    for (size_t i = 0; i < _children.size(); i++) {
      if (_children[i].element_name().empty()) {
        if (err) {
          (*err) =
              "Internal error: Existing child Prim's elementName is empty.\n";
        }
        return false;
      }

      if (_childrenNameSet.count(_children[i].element_name())) {
        if (err) {
          (*err) =
              "Internal error: _children contains Prim with same "
              "elementName.\n";
        }
        return false;
      }

      _childrenNameSet.insert(_children[i].element_name());
    }
  }

  DCOUT("elementName = " << elementName);

  if (_childrenNameSet.count(elementName)) {
    if (rename_prim_name) {
      std::string unique_name;
      if (!makeUniqueName(_childrenNameSet, elementName, &unique_name)) {
        if (err) {
          (*err) = fmt::format(
              "Internal error. cannot assign unique name for `{}`.\n",
              elementName);
        }
        return false;
      }

      // Ensure valid Prim name
      if (!ValidatePrimElementName(unique_name)) {
        if (err) {
          (*err) = fmt::format(
              "Internally generated Prim name `{}` is invalid as a Prim "
              "name.\n",
              unique_name);
        }
        return false;
      }

      elementName = unique_name;

      // Need to modify both Prim::data::name and Prim::elementPath
      DCOUT("elementName = " << elementName);
      if (!SetPrimElementName(rhs.get_data(), elementName)) {
        if (err) {
          (*err) = fmt::format(
              "Internal error. cannot modify Prim's elementName.\n");
        }
        return false;
      }
      rhs.element_path() = Path(elementName, /* prop_part */ "");
    } else {
      if (err) {
        (*err) = fmt::format(
            "Prim name(elementName) {} already exists in children.\n",
            rhs.element_name());
      }
      return false;
    }
  }

  DCOUT("rhs.elementName = " << rhs.element_name());

  _childrenNameSet.insert(elementName);
  _children.emplace_back(std::move(rhs));
  _child_dirty = true;

  return true;
}

bool Prim::replace_child(const std::string &child_prim_name, Prim &&rhs,
                         std::string *err) {
#if defined(TINYUSDZ_ENABLE_THREAD)
  // TODO: Only take a lock when dirty.
  std::lock_guard<std::mutex> lock(_mutex);
#endif

  if (child_prim_name.empty()) {
    if (err) {
      (*err) += "child_prim_name is empty.\n";
    }
  }

  if (!ValidatePrimElementName(child_prim_name)) {
    if (err) {
      (*err) +=
          fmt::format("`{}` is not a valid Prim name.\n", child_prim_name);
    }
  }

  if (_children.size() != _childrenNameSet.size()) {
    // Rebuild _childrenNames
    _childrenNameSet.clear();
    for (size_t i = 0; i < _children.size(); i++) {
      if (_children[i].element_name().empty()) {
        if (err) {
          (*err) =
              "Internal error: Existing child Prim's elementName is empty.\n";
        }
        return false;
      }

      if (_childrenNameSet.count(_children[i].element_name())) {
        if (err) {
          (*err) =
              "Internal error: _children contains Prim with same "
              "elementName.\n";
        }
        return false;
      }

      _childrenNameSet.insert(_children[i].element_name());
    }
  }

  // Simple linear scan
  auto result = std::find_if(_children.begin(), _children.end(),
                             [child_prim_name](const Prim &p) {
                               return (p.element_name() == child_prim_name);
                             });

  if (result != _children.end()) {
    // Need to modify both Prim::data::name and Prim::elementPath
    if (!SetPrimElementName(rhs.get_data(), child_prim_name)) {
      if (err) {
        (*err) =
            fmt::format("Internal error. cannot modify Prim's elementName.\n");
      }
      return false;
    }
    rhs.element_path() = Path(child_prim_name, /* prop_part */ "");

    (*result) = std::move(rhs);  // replace

  } else {
    // Need to modify both Prim::data::name and Prim::elementPath
    if (!SetPrimElementName(rhs.get_data(), child_prim_name)) {
      if (err) {
        (*err) =
            fmt::format("Internal error. cannot modify Prim's elementName.\n");
      }
      return false;
    }
    rhs.element_path() = Path(child_prim_name, /* prop_part */ "");

    _childrenNameSet.insert(child_prim_name);
    _children.emplace_back(std::move(rhs));  // add
  }

  _child_dirty = true;

  return true;
}

const std::vector<int64_t> &Prim::get_child_indices_from_primChildren(
    bool force_update, bool *indices_is_valid) const {
#if defined(TINYUSDZ_ENABLE_THREAD)
  // TODO: Only take a lock when dirty.
  std::lock_guard<std::mutex> lock(_mutex);
#endif

  if (!force_update && (_primChildrenIndices.size() == _children.size()) &&
      !_child_dirty) {
    // got cache.
    if (indices_is_valid) {
      (*indices_is_valid) = _primChildrenIndicesIsValid;
    }
    return _primChildrenIndices;
  }

  if (!force_update) {
    _child_dirty = false;
  }

  if (metas().primChildren.empty()) {
    _primChildrenIndices.resize(_children.size());
    std::iota(_primChildrenIndices.begin(), _primChildrenIndices.end(), 0);
    _primChildrenIndicesIsValid = true;
    if (indices_is_valid) {
      (*indices_is_valid) = _primChildrenIndicesIsValid;
    }
    return _primChildrenIndices;
  }

  std::map<std::string, size_t> m;  // name -> children() index map
  for (size_t i = 0; i < _children.size(); i++) {
    m.emplace(_children[i].element_name(), i);
  }
  std::set<size_t> table;  // to check uniqueness

  // Use the length of primChildren.
  _primChildrenIndices.resize(metas().primChildren.size());

  bool valid = true;

  for (size_t i = 0; i < _primChildrenIndices.size(); i++) {
    std::string tok = metas().primChildren[i].str();
    const auto it = m.find(tok);
    if (it != m.end()) {
      _primChildrenIndices[i] = int64_t(it->second);

      table.insert(it->second);
    } else {
      // Prim name not found.
      _primChildrenIndices[i] = -1;
      valid = false;
    }
  }

  if (table.size() != _primChildrenIndices.size()) {
    // duplicated index exists.
    valid = false;
  }

  _primChildrenIndicesIsValid = valid;
  if (indices_is_valid) {
    (*indices_is_valid) = _primChildrenIndicesIsValid;
  }

  return _primChildrenIndices;
}

//
// To deal with clang's -Wexit-time-destructors, dynamically allocate buffer for
// PrimMeta.
//
// NOTE: not thread-safe.
//
class EmptyStaticMeta {
 private:
  EmptyStaticMeta() = default;

 public:
  static PrimMeta &GetEmptyStaticMeta() {
    if (!s_meta) {
      s_meta = new PrimMeta();
    }

    return *s_meta;
  }

  ~EmptyStaticMeta() {
    delete s_meta;
    s_meta = nullptr;
  }

 private:
  static PrimMeta *s_meta;
};

PrimMeta *EmptyStaticMeta::s_meta = nullptr;

PrimMeta &Prim::metas() {
  PrimMeta *p = GetPrimMeta(_data);
  if (p) {
    return *p;
  }

  // TODO: This should not happen. report an error.
  return EmptyStaticMeta::GetEmptyStaticMeta();
}

const PrimMeta &Prim::metas() const {
  const PrimMeta *p = GetPrimMeta(_data);
  if (p) {
    return *p;
  }

  // TODO: This should not happen. report an error.
  return EmptyStaticMeta::GetEmptyStaticMeta();
}


bool SetCustomDataByKey(const std::string &key, const MetaVariable &var,
                        CustomDataType &custom) {
  // split by namespace
  std::vector<std::string> names = split(key, ":");
  DCOUT("names = " << to_string(names));

  if (names.empty()) {
    DCOUT("names is empty");
    return false;
  }

  if (names.size() > 1024) {
    // too deep
    DCOUT("too deep");
    return false;
  }

  CustomDataType *curr = &custom;

  for (size_t i = 0; i < names.size(); i++) {
    const std::string &elemkey = names[i];
    DCOUT("elemkey = " << elemkey);

    if (i == (names.size() - 1)) {
      DCOUT("leaf");
      // leaf
      (*curr)[elemkey] = var;
    } else {
      auto it = curr->find(elemkey);
      if (it != curr->end()) {
        // must be CustomData type
        value::Value &data = it->second.get_raw_value();
        CustomDataType *p = data.as<CustomDataType>();
        if (p) {
          curr = p;
        } else {
          DCOUT("value is not dictionary");
          return false;
        }
      } else {
        // Add empty dictionary.
        CustomDataType customData;
        curr->emplace(elemkey, customData);
        DCOUT("add dict " << elemkey);

        MetaVariable &child = curr->at(elemkey);
        value::Value &data = child.get_raw_value();
        CustomDataType *childp = data.as<CustomDataType>();
        if (!childp) {
          DCOUT("childp is null");
          return false;
        }

        DCOUT("child = " << print_customData(*childp, "child", uint32_t(i)));

        // renew curr
        curr = childp;
      }
    }
  }

  DCOUT("dict = " << print_customData(custom, "custom", 0));

  return true;
}

bool HasCustomDataKey(const CustomDataType &custom, const std::string &key) {
  // split by namespace
  std::vector<std::string> names = split(key, ":");

  DCOUT(print_customData(custom, "customData", 0));

  if (names.empty()) {
    DCOUT("empty");
    return false;
  }

  if (names.size() > 1024) {
    DCOUT("too deep");
    // too deep
    return false;
  }

  const CustomDataType *curr = &custom;

  for (size_t i = 0; i < names.size(); i++) {
    const std::string &elemkey = names[i];
    DCOUT("elemkey = " << elemkey);

    DCOUT("dict = " << print_customData(*curr, "dict", uint32_t(i)));

    auto it = curr->find(elemkey);
    if (it == curr->end()) {
      DCOUT("key not found");
      return false;
    }

    if (i == (names.size() - 1)) {
      // leaf .ok
    } else {
      // must be CustomData type
      const value::Value &data = it->second.get_raw_value();
      const CustomDataType *p = data.as<CustomDataType>();
      if (p) {
        curr = p;
      } else {
        DCOUT("value is not dictionary type.");
        return false;
      }
    }
  }

  return true;
}

bool GetCustomDataByKey(const CustomDataType &custom, const std::string &key,
                        MetaVariable *var) {
  if (!var) {
    return false;
  }

  DCOUT(print_customData(custom, "customData", 0));

  // split by namespace
  std::vector<std::string> names = split(key, ":");

  if (names.empty()) {
    return false;
  }

  if (names.size() > 1024) {
    // too deep
    return false;
  }

  const CustomDataType *curr = &custom;

  for (size_t i = 0; i < names.size(); i++) {
    const std::string &elemkey = names[i];

    auto it = curr->find(elemkey);
    if (it == curr->end()) {
      return false;
    }

    if (i == (names.size() - 1)) {
      // leaf
      (*var) = it->second;
    } else {
      // must be CustomData type
      const value::Value &data = it->second.get_raw_value();
      const CustomDataType *p = data.as<CustomDataType>();
      if (p) {
        curr = p;
      } else {
        return false;
      }
    }
  }

  return true;
}

namespace {

bool OverrideCustomDataRec(uint32_t depth, CustomDataType &dst,
                           const CustomDataType &src, const bool override_existing) {
  if (depth > (1024 * 1024 * 128)) {
    // too deep
    return false;
  }

  for (const auto &item : src) {
    if (dst.count(item.first)) {
      if (override_existing) {
        CustomDataType *dst_dict =
            dst.at(item.first).get_raw_value().as<CustomDataType>();

        const value::Value &src_data = item.second.get_raw_value();
        const CustomDataType *src_dict = src_data.as<CustomDataType>();

        //
        // Recursively apply override op both types are dict.
        //
        if (src_dict && dst_dict) {
          // recursively override dict
          if (!OverrideCustomDataRec(depth + 1, (*dst_dict), (*src_dict), override_existing)) {
            return false;
          }

        } else {
          dst[item.first] = item.second;
        }
      }
    } else {
      // add dict value
      dst.emplace(item.first, item.second);
    }
  }

  return true;
}

}  // namespace

void OverrideDictionary(CustomDataType &dst, const CustomDataType &src, const bool override_existing) {
  OverrideCustomDataRec(0, dst, src, override_existing);
}

AssetInfo PrimMeta::get_assetInfo(bool *is_authored) const {
  AssetInfo ainfo;

  if (is_authored) {
    (*is_authored) = authored();
  }

  if (authored()) {
    ainfo._fields = meta;

    {
      MetaVariable identifier_var;
      if (GetCustomDataByKey(meta, "identifier", &identifier_var)) {
        std::string identifier;
        if (identifier_var.get_value<std::string>(&identifier)) {
          ainfo.identifier = identifier;
          ainfo._fields.erase("identifier");
        }
      }
    }

    {
      MetaVariable name_var;
      if (GetCustomDataByKey(meta, "name", &name_var)) {
        std::string name;
        if (name_var.get_value<std::string>(&name)) {
          ainfo.name = name;
          ainfo._fields.erase("name");
        }
      }
    }

    {
      MetaVariable payloadDeps_var;
      if (GetCustomDataByKey(meta, "payloadAssetDependencies",
                             &payloadDeps_var)) {
        std::vector<value::AssetPath> assets;
        if (payloadDeps_var.get_value<std::vector<value::AssetPath>>(&assets)) {
          ainfo.payloadAssetDependencies = assets;
          ainfo._fields.erase("payloadAssetDependencies");
        }
      }
    }

    {
      MetaVariable version_var;
      if (GetCustomDataByKey(meta, "version", &version_var)) {
        std::string version;
        if (version_var.get_value<std::string>(&version)) {
          ainfo.version = version;
          ainfo._fields.erase("version");
        }
      }
    }
  }

  return ainfo;
}

const std::string PrimMeta::get_kind() const {

  if (kind.has_value()) {
    if (kind.value() == Kind::UserDef) {
      return _kind_str;
    } else {
      return to_string(kind.value());
    }
  }

  return "";
}

bool IsXformablePrim(const Prim &prim) {
  uint32_t tyid = prim.type_id();

  // GeomSubset is not xformable

  switch (tyid) {
    case value::TYPE_ID_GPRIM: {
      return true;
    }
    case value::TYPE_ID_GEOM_XFORM: {
      return true;
    }
    case value::TYPE_ID_GEOM_MESH: {
      return true;
    }
    case value::TYPE_ID_GEOM_BASIS_CURVES: {
      return true;
    }
    case value::TYPE_ID_GEOM_SPHERE: {
      return true;
    }
    case value::TYPE_ID_GEOM_CUBE: {
      return true;
    }
    case value::TYPE_ID_GEOM_CYLINDER: {
      return true;
    }
    case value::TYPE_ID_GEOM_CONE: {
      return true;
    }
    case value::TYPE_ID_GEOM_CAPSULE: {
      return true;
    }
    case value::TYPE_ID_GEOM_POINTS: {
      return true;
    }
    // value::TYPE_ID_GEOM_GEOMSUBSET
    case value::TYPE_ID_GEOM_POINT_INSTANCER: {
      return true;
    }
    case value::TYPE_ID_GEOM_CAMERA: {
      return true;
    }
    case value::TYPE_ID_LUX_DOME: {
      return true;
    }
    case value::TYPE_ID_LUX_CYLINDER: {
      return true;
    }
    case value::TYPE_ID_LUX_SPHERE: {
      return true;
    }
    case value::TYPE_ID_LUX_DISK: {
      return true;
    }
    case value::TYPE_ID_LUX_DISTANT: {
      return true;
    }
    case value::TYPE_ID_LUX_RECT: {
      return true;
    }
    case value::TYPE_ID_LUX_GEOMETRY: {
      return true;
    }
    case value::TYPE_ID_LUX_PORTAL: {
      return true;
    }
    case value::TYPE_ID_LUX_PLUGIN: {
      return true;
    }
    case value::TYPE_ID_SKEL_ROOT: {
      return true;
    }
    case value::TYPE_ID_SKELETON: {
      return true;
    }
    default:
      return false;
  }
}

bool CastToXformable(const Prim &prim, const Xformable **xformable) {
  if (!xformable) {
    return false;
  }

  // __ty = class derived from Xformable.
#define TRY_CAST(__ty)             \
  if (auto pv = prim.as<__ty>()) { \
    (*xformable) = pv;             \
    return true;                   \
  }

  // TODO: Use tydra::ApplyToXformable
  TRY_CAST(GPrim)
  TRY_CAST(Xform)
  TRY_CAST(GeomMesh)
  TRY_CAST(GeomBasisCurves)
  TRY_CAST(GeomCube)
  TRY_CAST(GeomSphere)
  TRY_CAST(GeomCylinder)
  TRY_CAST(GeomCone)
  TRY_CAST(GeomCapsule)
  TRY_CAST(GeomPoints)
  // TRY_CAST(GeomPointInstancer)
  TRY_CAST(GeomCamera)
  TRY_CAST(SkelRoot)
  TRY_CAST(Skeleton)
  TRY_CAST(RectLight)
  TRY_CAST(DomeLight)
  TRY_CAST(CylinderLight)
  TRY_CAST(SphereLight)
  TRY_CAST(DiskLight)
  TRY_CAST(DistantLight)
  TRY_CAST(RectLight)
  TRY_CAST(GeometryLight)
  TRY_CAST(PortalLight)
  TRY_CAST(PluginLight)
  TRY_CAST(SkelRoot)
  TRY_CAST(Skeleton)

  return false;
}

value::matrix4d GetLocalTransform(const Prim &prim, bool *resetXformStack,
                                  double t,
                                  value::TimeSampleInterpolationType tinterp) {
  if (!IsXformablePrim(prim)) {
    if (resetXformStack) {
      (*resetXformStack) = false;
    }
    return value::matrix4d::identity();
  }

  // default false
  if (resetXformStack) {
    (*resetXformStack) = false;
  }

  const Xformable *xformable{nullptr};
  if (CastToXformable(prim, &xformable)) {
    if (!xformable) {
      return value::matrix4d::identity();
    }

    value::matrix4d m;
    bool rxs{false};
    nonstd::expected<value::matrix4d, std::string> ret =
        xformable->GetLocalMatrix(t, tinterp, &rxs);
    if (ret) {
      if (resetXformStack) {
        (*resetXformStack) = rxs;
      }
      return ret.value();
    }
  }

  return value::matrix4d::identity();
}

void PrimMetas::update_from(const PrimMetas &rhs, const bool override_authored) {
  if (rhs.active.has_value()) {
    if (override_authored || !active.has_value()) {
      active = rhs.active;
    }
  }

  if (rhs.hidden.has_value()) {
    if (override_authored || !hidden.has_value()) {
      hidden = rhs.hidden;
    }
  }

  if (rhs.kind.has_value()) {
    if (override_authored || !kind.has_value()) {
      kind = rhs.kind;
    }
  }

  if (rhs.instanceable.has_value()) {
    if (override_authored || !instanceable.has_value()) {
      instanceable = rhs.instanceable;
    }
  }

  if (rhs.assetInfo) {
    if (assetInfo) {
      OverrideDictionary(assetInfo.value(), rhs.assetInfo.value(), override_authored);
    } else if (override_authored) {
      assetInfo = rhs.assetInfo;
    }
  }

  if (rhs.clips) {
    if (clips) {
      OverrideDictionary(clips.value(), rhs.clips.value(), override_authored);
    } else if (override_authored) {
      clips = rhs.clips;
    }
  }

  if (rhs.customData) {
    if (customData) {
      OverrideDictionary(customData.value(), rhs.customData.value(), override_authored);
    } else if (override_authored) {
      customData = rhs.customData;
    }
  }

  if (rhs.doc) {
    if (override_authored || !doc.has_value()) {
      doc = rhs.doc;
    }
  }

  if (rhs.comment) {
    if (override_authored || !comment.has_value()) {
      comment = rhs.comment;
    }
  }

  if (rhs.apiSchemas) {
    if (override_authored || !apiSchemas.has_value()) {
      apiSchemas = rhs.apiSchemas;
    }
  }

  if (rhs.sdrMetadata) {
    if (sdrMetadata) {
      OverrideDictionary(sdrMetadata.value(), rhs.sdrMetadata.value(), override_authored);
    } else if (override_authored) {
      sdrMetadata = rhs.sdrMetadata;
    }
  }

  if (rhs.sceneName) {
    if (override_authored || !sceneName.has_value()) {
      sceneName = rhs.sceneName;
    }
  }

  if (rhs.displayName) {
    if (override_authored || !displayName.has_value()) {
      displayName = rhs.displayName;
    }
  }

  if (rhs.references) {
    if (override_authored || !references.has_value()) {
      references = rhs.references;
    }
  }
  if (rhs.payload) {
    if (override_authored || !payload.has_value()) {
      payload = rhs.payload;
    }
  }
  if (rhs.inherits) {
    if (override_authored || !inherits.has_value()) {
      inherits = rhs.inherits;
    }
  }
  if (rhs.variantSets) {
    if (override_authored || !variantSets.has_value()) {
      variantSets = rhs.variantSets;
    }
  }
  if (rhs.variants) {
    if (override_authored || !variants.has_value()) {
      variants = rhs.variants;
    }
  }
  if (rhs.specializes) {
    if (override_authored || !specializes.has_value()) {
      specializes = rhs.specializes;
    }
  }

  if (rhs.unregisteredMetas.size()) {
    for (const auto &item : rhs.unregisteredMetas) {
      if (unregisteredMetas.count(item.first)) {
        if (override_authored) {
          unregisteredMetas[item.first] = item.second;
        } 
      } else {
        unregisteredMetas[item.first] = item.second;
      }
    }
  }

  OverrideDictionary(meta, rhs.meta, override_authored);
}

bool AttrMetas::has_colorSpace() const {
  return meta.count("colorSpace");
}

value::token AttrMetas::get_colorSpace() const {
  if (!has_colorSpace()) {
    return value::token();
  }

  const MetaVariable &mv = meta.at("colorSpace");
  value::token tok;
  if (mv.get_value<value::token>(&tok)) {
    return tok;
  }

  return value::token();
}

bool AttrMetas::has_unauthoredValuesIndex() const {
  return meta.count("unauthoredValuesIndex");
}

int AttrMetas::get_unauthoredValuesIndex() const {
  if (!has_unauthoredValuesIndex()) {
    return -1;
  }

  const MetaVariable &mv = meta.at("unauthoredValuesIndex");
  int v;
  if (mv.get_value<int>(&v)) {
    return v;
  }

  return -1;
}

namespace {

nonstd::optional<const PrimSpec *> GetPrimSpecAtPathRec(
    const PrimSpec *parent, const std::string &parent_path, const Path &path,
    uint32_t depth) {
  if (depth > (1024 * 1024 * 128)) {
    // Too deep.
    return nonstd::nullopt;
  }

  if (!parent) {
    return nonstd::nullopt;
  }

  std::string abs_path;
  {
    std::string elementName = parent->name();

    abs_path = parent_path + "/" + elementName;

    if (abs_path == path.full_path_name()) {
      return parent;
    }
  }

  for (const auto &child : parent->children()) {
    if (auto pv = GetPrimSpecAtPathRec(&child, abs_path, path, depth + 1)) {
      return pv.value();
    }
  }

  // not found
  return nonstd::nullopt;
}

bool HasReferencesRec(uint32_t depth, const PrimSpec &primspec,
                      const uint32_t max_depth = 1024 * 128) {
  if (depth > max_depth) {
    // too deep
    return false;
  }

  if (primspec.metas().references) {
    return true;
  }

  for (auto &child : primspec.children()) {
    if (HasReferencesRec(depth + 1, child, max_depth)) {
      return true;
    }
  }

  return false;
}

bool HasPayloadRec(uint32_t depth, const PrimSpec &primspec,
                   const uint32_t max_depth = 1024 * 128) {
  if (depth > max_depth) {
    // too deep
    return false;
  }

  if (primspec.metas().payload) {
    return true;
  }

  for (auto &child : primspec.children()) {
    if (HasPayloadRec(depth + 1, child, max_depth)) {
      return true;
    }
  }

  return false;
}

bool HasVariantRec(uint32_t depth, const PrimSpec &primspec,
                   const uint32_t max_depth = 1024 * 128) {
  if (depth > max_depth) {
    // too deep
    return false;
  }

  // TODO: Also check if PrimSpec::variantSets is empty?
  if (primspec.metas().variants && primspec.metas().variantSets) {
    return true;
  }

  for (auto &child : primspec.children()) {
    if (HasVariantRec(depth + 1, child, max_depth)) {
      return true;
    }
  }

  return false;
}

bool HasInheritsRec(uint32_t depth, const PrimSpec &primspec,
                    const uint32_t max_depth = 1024 * 128) {
  if (depth > max_depth) {
    // too deep
    return false;
  }

  if (primspec.metas().inherits) {
    return true;
  }

  for (auto &child : primspec.children()) {
    if (HasInheritsRec(depth + 1, child, max_depth)) {
      return true;
    }
  }

  return false;
}

bool HasSpecializesRec(uint32_t depth, const PrimSpec &primspec,
                    const uint32_t max_depth = 1024 * 128) {
  if (depth > max_depth) {
    // too deep
    return false;
  }

  if (primspec.metas().specializes) {
    return true;
  }

  for (auto &child : primspec.children()) {
    if (HasSpecializesRec(depth + 1, child, max_depth)) {
      return true;
    }
  }

  return false;
}

bool HasOverRec(uint32_t depth, const PrimSpec &primspec,
                       const uint32_t max_depth = 1024 * 128) {
  if (depth > max_depth) {
    // too deep
    return false;
  }

  if (primspec.specifier() == Specifier::Over) {
    return true;
  }

  for (auto &child : primspec.children()) {
    if (HasOverRec(depth + 1, child, max_depth)) {
      return true;
    }
  }

  return false;
}

}  // namespace

bool Layer::find_primspec_at(const Path &path, const PrimSpec **ps,
                             std::string *err) const {
  if (!ps) {
    PUSH_ERROR_AND_RETURN("Invalid PrimSpec dst argument");
  }

  if (!path.is_valid()) {
    DCOUT("Invalid path.");
    PUSH_ERROR_AND_RETURN("Invalid path");
  }

  if (path.is_relative_path()) {
    // TODO
    PUSH_ERROR_AND_RETURN(fmt::format("TODO: Relative path: {}", path.full_path_name()));
  }

  if (!path.is_absolute_path()) {
    PUSH_ERROR_AND_RETURN(fmt::format("Path is not absolute path: {}", path.full_path_name()));
  }

#if defined(TINYUSDZ_ENABLE_THREAD)
  // TODO: Only take a lock when dirty.
  std::lock_guard<std::mutex> lock(_mutex);
#endif

  if (_dirty) {
    DCOUT("clear cache.");
    // Clear cache.
    _primspec_path_cache.clear();

    _dirty = false;
  } else {
    // First find from a cache.
    auto ret = _primspec_path_cache.find(path.prim_part());
    if (ret != _primspec_path_cache.end()) {
      DCOUT("Found cache.");
      (*ps) = ret->second;
      return true;
    }
  }

  // Brute-force search.
  for (const auto &parent : _prim_specs) {
    if (auto pv = GetPrimSpecAtPathRec(&parent.second, /* parent_path */ "",
                                       path, /* depth */ 0)) {
      (*ps) = pv.value();

      // Add to cache.
      // Assume pointer address does not change unless dirty state changes.
      _primspec_path_cache[path.prim_part()] = pv.value();
      return true;
    }
  }

  return false;
}

bool Layer::check_unresolved_references(const uint32_t max_depth) const {
  bool ret = false;

  for (const auto &item : _prim_specs) {
    if (HasReferencesRec(/* depth */ 0, item.second, max_depth)) {
      ret = true;
      break;
    }
  }

  _has_unresolved_references = ret;
  return _has_unresolved_references;
}

bool Layer::check_unresolved_payload(const uint32_t max_depth) const {
  bool ret = false;

  for (const auto &item : _prim_specs) {
    if (HasPayloadRec(/* depth */ 0, item.second, max_depth)) {
      ret = true;
      break;
    }
  }

  _has_unresolved_payload = ret;
  return _has_unresolved_payload;
}

bool Layer::check_unresolved_variant(const uint32_t max_depth) const {
  bool ret = false;

  for (const auto &item : _prim_specs) {
    if (HasVariantRec(/* depth */ 0, item.second, max_depth)) {
      ret = true;
      break;
    }
  }

  _has_unresolved_variant = ret;
  return _has_unresolved_variant;
}

bool Layer::check_unresolved_inherits(const uint32_t max_depth) const {
  bool ret = false;

  for (const auto &item : _prim_specs) {
    if (HasInheritsRec(/* depth */ 0, item.second, max_depth)) {
      ret = true;
      break;
    }
  }

  _has_unresolved_inherits = ret;
  return _has_unresolved_inherits;
}

bool Layer::check_unresolved_specializes(const uint32_t max_depth) const {
  bool ret = false;

  for (const auto &item : _prim_specs) {
    if (HasSpecializesRec(/* depth */ 0, item.second, max_depth)) {
      ret = true;
      break;
    }
  }

  _has_unresolved_specializes = ret;
  return _has_unresolved_specializes;
}

bool Layer::check_over_primspec(const uint32_t max_depth) const {
  bool ret = false;

  for (const auto &item : _prim_specs) {
    if (HasOverRec(/* depth */ 0, item.second, max_depth)) {
      ret = true;
      break;
    }
  }

  _has_over_primspec = ret;
  return _has_over_primspec;
}

}  // namespace tinyusdz

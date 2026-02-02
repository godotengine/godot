// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// UsdGeom API implementations


#include <sstream>

#include "pprinter.hh"
#include "value-types.hh"
#include "prim-types.hh"
#include "str-util.hh"
#include "tiny-format.hh"
#include "xform.hh"
#include "usdGeom.hh"
//
//#include "external/simple_match/include/simple_match/simple_match.hpp"
//
#include "common-macros.inc"
#include "math-util.inc"
#include "str-util.hh"
#include "value-pprint.hh"

#define SET_ERROR_AND_RETURN(msg) \
  if (err) {                      \
    (*err) = (msg);               \
  }                               \
  return false

namespace tinyusdz {

namespace {

constexpr auto kPrimvars = "primvars:";
constexpr auto kIndices = ":indices";

///
/// Computes
///
///  for i in len(indices):
///    for k in elementSize:
///      dest[i*elementSize + k] = values[indices[i]*elementSize + k]
///
/// `dest` is set to `values` when `indices` is empty
///
template <typename T>
nonstd::expected<bool, std::string> ExpandWithIndices(
    const std::vector<T> &values, uint32_t elementSize, const std::vector<int32_t> &indices,
    std::vector<T> *dest) {
  if (!dest) {
    return nonstd::make_unexpected("`dest` is nullptr.");
  }

  if (indices.empty()) {
    (*dest) = values;
    return true;
  }

  if (elementSize == 0) {
    return false;
  }

  if ((values.size() % elementSize) != 0) {
    return false;
  }

  dest->resize(indices.size() * elementSize);

  std::vector<size_t> invalidIndices;

  bool valid = true;
  for (size_t i = 0; i < indices.size(); i++) {
    int32_t idx = indices[i];
    if ((idx >= 0) && ((size_t(idx+1) * size_t(elementSize)) <= values.size())) {
      for (size_t k = 0; k < elementSize; k++) {
        (*dest)[i*elementSize + k] = values[size_t(idx)*elementSize + k];
      }
    } else {
      invalidIndices.push_back(i);
      valid = false;
    }
  }

  if (invalidIndices.size()) {
    return nonstd::make_unexpected(
        "Invalid indices found: " +
        value::print_array_snipped(invalidIndices,
                                   /* N to display */ 5));
  }

  return valid;
}

}  // namespace

bool IsSupportedGeomPrimvarType(uint32_t tyid) {
  //
  // scalar and 1D
  //
#define SUPPORTED_TYPE_FUN(__ty)                                           \
  case value::TypeTraits<__ty>::type_id(): {                                 \
    return true;                                                           \
  }                                                                        \
  case (value::TypeTraits<__ty>::type_id() | value::TYPE_ID_1D_ARRAY_BIT): { \
    return true;                                                           \
  }

  switch (tyid) {
    APPLY_GEOMPRIVAR_TYPE(SUPPORTED_TYPE_FUN)
    default:
      return false;
  }

#undef SUPPORTED_TYPE_FUN
}

bool IsSupportedGeomPrimvarType(const std::string &type_name) {
  return IsSupportedGeomPrimvarType(value::GetTypeId(type_name));
}

bool GeomPrimvar::has_elementSize() const {
  return _elementSize.has_value();
}

uint32_t GeomPrimvar::get_elementSize() const {
  if (_elementSize.has_value()) {
    return _elementSize.value();
  }
  return 1;
}

bool GeomPrimvar::has_interpolation() const {
  return _interpolation.has_value();
}

Interpolation GeomPrimvar::get_interpolation() const {
  if (_interpolation.has_value()) {
    return _interpolation.value();
  }

  return Interpolation::Constant;  // unauthored
}

bool GPrim::has_primvar(const std::string &varname) const {
  std::string primvar_name = kPrimvars + varname;
  return props.count(primvar_name);
}

bool GPrim::get_primvar(const std::string &varname, GeomPrimvar *out_primvar,
                        std::string *err) const {
  if (!out_primvar) {
    SET_ERROR_AND_RETURN("Output GeomPrimvar is nullptr.");
  }

  GeomPrimvar primvar;

  std::string primvar_name = kPrimvars + varname;

  const auto it = props.find(primvar_name);
  if (it == props.end()) {
    return false;
  }

  if (it->second.is_attribute()) {
    const Attribute &attr = it->second.get_attribute();

    primvar.set_value(attr);
    primvar.set_name(varname);
    if (attr.metas().interpolation.has_value()) {
      primvar.set_interpolation(attr.metas().interpolation.value());
    }
    if (attr.metas().elementSize.has_value()) {
      primvar.set_elementSize(attr.metas().elementSize.value());
    }
    if (attr.metas().has_unauthoredValuesIndex()) {
      primvar.set_unauthoredValuesIndex(attr.metas().get_unauthoredValuesIndex());
    }

  } else {
    SET_ERROR_AND_RETURN(fmt::format("{} is not Attribute. Maybe Relationship?", primvar_name));
  }

  // has indices?
  std::string index_name = primvar_name + kIndices;
  const auto indexIt = props.find(index_name);

  if (indexIt != props.end()) {
    if (indexIt->second.is_attribute()) {

      if (!(primvar.get_attribute().type_id() & value::TYPE_ID_1D_ARRAY_BIT)) {
        SET_ERROR_AND_RETURN(
            fmt::format("Indexed GeomPrimVar with scalar PrimVar Attribute is not supported. PrimVar name: {}", primvar_name));
      }

      const Attribute &indexAttr = indexIt->second.get_attribute();

      if (indexAttr.is_connection()) {
        SET_ERROR_AND_RETURN(
            "Attribute Connetion is not supported for index Attribute, since we need Stage info to find Prim referred by targetPath. Use Tydra API tydra::GetGeomPrimvar.");
      }

      if (indexAttr.is_blocked()) {
        // ignore Index attribute.
      } else {

        if (indexAttr.has_timesamples()) {
          const auto &ts = indexAttr.get_var().ts_raw();
          TypedTimeSamples<std::vector<int32_t>> tss;
          if (!tss.from_timesamples(ts)) {
            SET_ERROR_AND_RETURN(fmt::format("Index Attribute seems not an timesamples with int[] type: {}", index_name));
          }

          primvar.set_timesampled_indices(tss);
        }

        if (indexAttr.has_value()) {
          // Check if int[] type.
          // TODO: Support uint[]?
          std::vector<int32_t> indices;
          if (!indexAttr.get_value(&indices)) {
            SET_ERROR_AND_RETURN(
                fmt::format("Index Attribute is not int[] type. Got {}",
                            indexAttr.type_name()));
          }

          primvar.set_default_indices(indices);
        }
      }
    } else {
      // indices are optional, so ok to skip it.
    }
  }

  (*out_primvar) = primvar;

  return true;
}


template <typename T>
bool GeomPrimvar::flatten_with_indices(const double t, std::vector<T> *dest, const value::TimeSampleInterpolationType tinterp, std::string *err) const {
  if (!dest) {
    if (err) {
      (*err) += "Output value is nullptr.";
    }
    return false;
  }

  if (_attr.has_timesamples() || _attr.has_value()) {

    if (!IsSupportedGeomPrimvarType(_attr.type_id())) {
      if (err) {
        (*err) += fmt::format("Unsupported type for GeomPrimvar. type = `{}`",
                              _attr.type_name());
      }
      return false;
    }

    std::vector<T> value;
    if (_attr.get_value<std::vector<T>>(t, &value, tinterp)) {

      uint32_t elementSize = _attr.metas().elementSize.value_or(1);

      // Get indices at specified time
      std::vector<int32_t> indices;
      if (value::TimeCode(t).is_default()) {
        if (has_default_indices()) {
          indices = _indices;
        } else if (has_timesampled_indices()) {
          _ts_indices.get(&indices, t, tinterp);
        }
      } else {
        _ts_indices.get(&indices, t, tinterp);
      }

      std::vector<T> expanded_val;
      auto ret = ExpandWithIndices(value, elementSize, indices, &expanded_val);
      if (ret) {
        (*dest) = expanded_val;
        // Currently we ignore ret.value()
        return true;
      } else {
        const std::string &err_msg = ret.error();
        if (err) {
          (*err) += fmt::format(
              "[Internal Error] Failed to expand for GeomPrimvar type = `{}`",
              _attr.type_name());
          if (err_msg.size()) {
            (*err) += "\n" + err_msg;
          }
        }
      }
    } else {
      if (err) {
        (*err) += fmt::format(
            "`{}[]` type requested, but Attribute is type `{}`",
            value::TypeTraits<T>::type_name(), _attr.type_name());
      }
    }
  } else {
    // TODO: Report error?
  }

  return false;
}

std::vector<int32_t> GeomPrimvar::get_indices(const double t) const {
  if (value::TimeCode(t).is_default()) {
    if (has_default_indices()) {
      return get_default_indices();
    }
  }
  
  if (has_timesampled_indices()) {
    std::vector<int32_t> indices;
    if (get_timesampled_indices().get(&indices, t)) {
      return indices;
    }
  }

  if (has_default_indices()) {
    return get_default_indices();
  }

  return std::vector<int32_t>();
}

void GeomPrimvar::set_indices(const std::vector<int32_t> &indices, const double t) {
  if (value::TimeCode(t).is_default()) {
    _indices = indices;
  } else {
    TypedTimeSamples<std::vector<int32_t>>::Sample *psample{nullptr};
    if (_ts_indices.get_sample_at(t, &psample)) {
      // overwrite content
      psample->value = indices;
    } else {
      _ts_indices.add_sample(t, indices);
    }
  }
}

template <typename T>
bool GeomPrimvar::flatten_with_indices(std::vector<T> *dest, std::string *err) const {
  return flatten_with_indices(value::TimeCode::Default(), dest, value::TimeSampleInterpolationType::Linear, err);
}

// instanciation
#define INSTANCIATE_FLATTEN_WITH_INDICES(__ty) \
  template bool GeomPrimvar::flatten_with_indices(std::vector<__ty> *dest, std::string *err) const; \
  template bool GeomPrimvar::flatten_with_indices(const double t, std::vector<__ty> *dest, const value::TimeSampleInterpolationType tinterp, std::string *err) const;

APPLY_GEOMPRIVAR_TYPE(INSTANCIATE_FLATTEN_WITH_INDICES)

#undef INSTANCIATE_FLATTEN_WITH_INDICES

bool GeomPrimvar::flatten_with_indices(const double t, value::Value *dest, const value::TimeSampleInterpolationType tinterp, std::string *err) const {

  if (!dest) {
    if (err) {
      (*err) += "Output value is nullptr.";
    }
    return false;
  }

  bool processed = false;

  if (_attr.has_value() || _attr.has_timesamples()) {
    if (!IsSupportedGeomPrimvarType(_attr.type_id())) {
      if (err) {
        (*err) += fmt::format("Unsupported type for GeomPrimvar. type = `{}`",
                              _attr.type_name());
      }
      return false;
    }

    value::Value val;

    if (!(_attr.type_id() & value::TYPE_ID_1D_ARRAY_BIT)) {

      // evaluate value at specified time and return it for scalar type.
      value::Value v;
      if (!_attr.get_var().get_interpolated_value(t, tinterp, &v)) {
        if (err) {
          (*err) += fmt::format("Failed to evaluate Attribute value.");
        }
        return false;
      }
      (*dest) = v;
    } else {
      std::string err_msg;

      uint32_t elementSize = _attr.metas().elementSize.value_or(1);

      std::vector<int32_t> indices;
      // Get indices at specified time
      if (value::TimeCode(t).is_default()) {
        indices = _indices;
      } else {
        _ts_indices.get(&indices, t, tinterp);
      }

#define APPLY_FUN(__ty)                                                  \
  case value::TypeTraits<__ty>::type_id() | value::TYPE_ID_1D_ARRAY_BIT: { \
    std::vector<__ty> value; \
    std::vector<__ty> expanded_val;                                      \
    if (_attr.get_value(t, &value, tinterp)) {                \
      auto ret = ExpandWithIndices(value, elementSize, indices, &expanded_val); \
      if (ret) {                                                         \
        processed = ret.value();                                         \
        if (processed) {                                                 \
          val = expanded_val;                                            \
        }                                                                \
      } else {                                                           \
        err_msg = ret.error();                                           \
      }                                                                  \
    }                                                                    \
    break;                                                               \
  }

      switch (_attr.type_id()) {
        APPLY_GEOMPRIVAR_TYPE(APPLY_FUN)
        default: {
          processed = false;
        }
      }

#undef APPLY_FUN

      if (processed) {
        (*dest) = std::move(val);
      } else {
        if (err) {
          (*err) += fmt::format(
              "[Internal Error] Failed to expand for GeomPrimvar type = `{}`",
              _attr.type_name());
          if (err_msg.size()) {
            (*err) += "\n" + err_msg;
          }
        }
      }
    }
  }

  return processed;
}

bool GeomPrimvar::flatten_with_indices(value::Value *dest, std::string *err) const {
  return flatten_with_indices(value::TimeCode::Default(), dest, value::TimeSampleInterpolationType::Linear, err);
}

template <typename T>
bool GeomPrimvar::get_value(T *dest, std::string *err) const {
  static_assert(tinyusdz::value::TypeTraits<T>::type_id() != value::TypeTraits<value::token>::type_id(), "`token` type is not supported as a GeomPrimvar");
  static_assert(tinyusdz::value::TypeTraits<T>::type_id() != value::TypeTraits<std::vector<value::token>>::type_id(), "`token[]` type is not supported as a GeomPrimvar");

  if (!dest) {
    if (err) {
      (*err) += "Output value is nullptr.";
    }
    return false;
  }

  if (_attr.is_blocked()) {
    if (err) {
      (*err) += "Attribute is blocked.";
    }
    return false;
  }

  if (_attr.has_value()) {
    if (!IsSupportedGeomPrimvarType(_attr.type_id())) {
      if (err) {
        (*err) += fmt::format("Unsupported type for GeomPrimvar. type = `{}`",
                              _attr.type_name());
      }
      return false;
    }

    if (auto pv = _attr.get_value<T>()) {

      // copy
      (*dest) = pv.value();
      return true;

    } else {
      if (err) {
        (*err) += fmt::format("Attribute value type mismatch. Requested type `{}` but Attribute has type `{}`", value::TypeTraits<T>::type_id(), _attr.type_name());
      }
      return false;
    }
  }

  if (_attr.has_timesamples()) {
    // Return the first sample.
    const auto &ts = _attr.get_var().ts_raw();
    if (ts.empty()) {
      if (err) {
        (*err) += "No TimeSample value in Attribute..";
      }
      return false;
    }
    
    if (auto pv =ts.get_samples().at(0).value.as<T>()) {
      (*dest) = (*pv);
      return true;
    }
  }

  return false;
}

template <typename T>
bool GeomPrimvar::get_value(double timecode, T *dest, value::TimeSampleInterpolationType interp, std::string *err) const {
  static_assert(tinyusdz::value::TypeTraits<T>::type_id() != value::TypeTraits<value::token>::type_id(), "`token` type is not supported as a GeomPrimvar");
  static_assert(tinyusdz::value::TypeTraits<T>::type_id() != value::TypeTraits<std::vector<value::token>>::type_id(), "`token[]` type is not supported as a GeomPrimvar");

  if (!dest) {
    if (err) {
      (*err) += "Output value is nullptr.";
    }
    return false;
  }

  if (_attr.is_blocked()) {
    if (err) {
      (*err) += "Attribute is blocked.";
    }
    return false;
  }

  if (!IsSupportedGeomPrimvarType(_attr.type_id())) {
    if (err) {
      (*err) += fmt::format("Unsupported type for GeomPrimvar. type = `{}`",
                            _attr.type_name());
    }
    return false;
  }

#if 0
  if (value::TimeCode(timecode).is_default()) {

    if (_attr.has_value()) {
      if (auto pv = _attr.get_value<T>()) {

        // copy
        (*dest) = pv.value();
        return true;

      } else {
        if (err) {
          (*err) += fmt::format("Attribute value type mismatch. Requested type `{}` but Attribute has type `{}`", value::TypeTraits<T>::type_id(), _attr.type_name());
        }
        return false;
      }
    }

  }

  if (_attr.has_timesamples()) {
    T value;

    if (!_attr.get_value(timecode, &value, interp)) {
      if (err) {
        (*err) += fmt::format("Get Attribute value at time {} failed. Maybe type mismatch?. Requested type `{}` but Attribute has type `{}`", timecode, value::TypeTraits<T>::type_id(), _attr.type_name());
      }
      return false;
    }

    // copy
    (*dest) = value;
    return true;
  }

  return false;
#else
  T value{};

  if (!_attr.get_value(timecode, &value, interp)) {
    if (err) {
      (*err) += fmt::format("Get Attribute value at time {} failed. Maybe type mismatch?. Requested type `{}` but Attribute has type `{}`", timecode, value::TypeTraits<T>::type_id(), _attr.type_name());
    }
    return false;
  }

  // copy
  (*dest) = value;
  return true;
#endif

}

// instanciation
#define INSTANCIATE_GET_VALUE(__ty) \
  template bool GeomPrimvar::get_value(__ty *dest, std::string *err) const; \
  template bool GeomPrimvar::get_value(double, __ty *dest, value::TimeSampleInterpolationType, std::string *err) const; \
  template bool GeomPrimvar::get_value(std::vector<__ty> *dest, std::string *err) const; \
  template bool GeomPrimvar::get_value(double, std::vector<__ty> *dest, value::TimeSampleInterpolationType, std::string *err) const;

APPLY_GEOMPRIVAR_TYPE(INSTANCIATE_GET_VALUE)

#undef INSTANCIATE_GET_VALUE

std::vector<GeomPrimvar> GPrim::get_primvars() const {
  std::vector<GeomPrimvar> gpvars;

  for (const auto &prop : props) {
    if (startsWith(prop.first, kPrimvars)) {
      // skip `:indices`. Attribute with `:indices` suffix is handled in
      // `get_primvar`
      // Also skips primvar attribute with `:indices` suffix only.
      if (endsWith(prop.first, kIndices)) {
        continue;
      }

      GeomPrimvar gprimvar;
      if (get_primvar(removePrefix(prop.first, kPrimvars), &gprimvar)) {
        gpvars.emplace_back(std::move(gprimvar));
      }
    }
  }

  return gpvars;
}

bool GPrim::set_primvar(const GeomPrimvar &primvar,
                        std::string *err) {
  if (primvar.name().empty()) {
    if (err) {
      (*err) += "GeomPrimvar.name is empty.";
    }
    return false;
  }

  if (startsWith(primvar.name(), "primvars:")) {
    if (err) {
      (*err) += "GeomPrimvar.name must not start with `primvars:` namespace. name = " + primvar.name();
    }
    return false;
  }

  std::string primvar_name = kPrimvars + primvar.name();

  // Overwrite existing primvar prop.
  // TODO: Report warn when primvar name already exists.

  Attribute attr = primvar.get_attribute();

  if (primvar.has_interpolation()) {
    attr.metas().interpolation = primvar.get_interpolation();
  }

  if (primvar.has_elementSize()) {
    attr.metas().elementSize = primvar.get_elementSize();
  }

  props[primvar_name] = attr;

  {
    primvar::PrimVar var;

    if (primvar.has_default_indices()) {
      var.set_value(primvar.get_default_indices());
    }

    if (primvar.has_timesampled_indices()) {
      for (const auto &sample : primvar.get_timesampled_indices().get_samples()) {
        var.set_timesample(sample.t, sample.value);
      }
    }

    if (primvar.has_default_indices() || primvar.has_timesampled_indices()) {
      Attribute indices;
      indices.set_var(var);
      std::string index_name = primvar_name + kIndices;
      props[index_name] = indices;
    }

  }

  return true;
}

bool GPrim::get_displayColor(value::color3f *dst, double t, const value::TimeSampleInterpolationType tinterp) const
{
  // TODO: timeSamples
  (void)t;
  (void)tinterp;

  GeomPrimvar primvar;
  std::string err;
  if (!get_primvar("displayColor", &primvar, &err)) {
    // TODO: report err
    return false;
  }

  return primvar.get_value(dst);
}

bool GPrim::get_displayOpacity(float *dst, double t, const value::TimeSampleInterpolationType tinterp) const
{
  // TODO: timeSamples
  (void)t;
  (void)tinterp;

  GeomPrimvar primvar;
  std::string err;
  if (!get_primvar("displayOpacity", &primvar, &err)) {
    // TODO: report err
    return false;
  }

  return primvar.get_value(dst);
}

const std::vector<value::point3f> GeomMesh::get_points(
    double time, value::TimeSampleInterpolationType interp) const {
  std::vector<value::point3f> dst;

  if (!points.authored() || points.is_blocked()) {
    return dst;
  }

  if (points.is_connection()) {
    // TODO: connection
    return dst;
  }

  if (auto pv = points.get_value()) {
    std::vector<value::point3f> val;
    if (pv.value().get(time, &val, interp)) {
      dst = std::move(val);
    }
  }

  return dst;
}

const std::vector<value::normal3f> GeomMesh::get_normals(
    double time, value::TimeSampleInterpolationType interp) const {
  std::vector<value::normal3f> dst;

  std::string err;
  if (has_primvar("normals")) {
    GeomPrimvar primvar;
    if (!get_primvar("normals", &primvar, &err)) {
      return dst;
    }

    primvar.flatten_with_indices(time, &dst, interp);
    return dst;
  } else if (normals.authored()) {
    if (normals.is_connection()) {
      // Not supported
      return dst;
    } else if (normals.is_blocked()) {
      return dst;
    }

    std::vector<int> indices;
    if (props.count("normals:indices")) {
      Attribute indexAttr = props.at("normals:indices").get_attribute();

      if (indexAttr.is_connection()) {
        // not supported.
        return dst;
      }

      if (!indexAttr.get_value(time, &indices, interp)) {
        // err
        return dst;
      }

    }

    std::vector<value::normal3f> value;
    if (!normals.get_value().value().get(time, &value, interp)) {
      return dst;
    }

    if (indices.size()) {
      uint32_t elementSize = normals.metas().elementSize.value_or(1);

      std::vector<value::normal3f> expanded_normals;
      auto ret = ExpandWithIndices(value, elementSize, indices, &expanded_normals);

      if (!ret) {
        return dst;
      }

      dst = expanded_normals;
    } else {
      dst = value;
    }
  }

  return dst;
}

const std::vector<value::color3f> GPrim::get_displayColors(
    double time, value::TimeSampleInterpolationType interp) const {
  std::vector<value::color3f> dst;

  std::string err;
  if (has_primvar("displayColor")) {
    GeomPrimvar primvar;
    if (!get_primvar("displayColor", &primvar, &err)) {
      return dst;
    }

    primvar.flatten_with_indices(time, &dst, interp);
  }

  return dst;
}

Interpolation GeomMesh::get_normalsInterpolation() const {
  if (props.count("primvars:normals")) {
    const auto &prop = props.at("primvars:normals");
    if (prop.get_attribute().type_name() == "normal3f[]") {
      if (prop.get_attribute().metas().interpolation) {
        return prop.get_attribute().metas().interpolation.value();
      }
    }
  } else if (normals.metas().interpolation) {
    return normals.metas().interpolation.value();
  }

  return Interpolation::Vertex;  // default 'vertex'
}

Interpolation GPrim::get_displayColorsInterpolation() const {
  if (props.count("primvars:displayColor")) {
    const auto &prop = props.at("primvars:displayColor");
    if (prop.get_attribute().type_name() == "color3f[]") {
      if (prop.get_attribute().metas().interpolation) {
        return prop.get_attribute().metas().interpolation.value();
      }
    }
  }

  return Interpolation::Vertex;  // default 'vertex'
}

const std::vector<int32_t> GeomMesh::get_faceVertexCounts(double time) const {
  std::vector<int32_t> dst;

  if (!faceVertexCounts.authored() || faceVertexCounts.is_blocked()) {
    return dst;
  }

  if (faceVertexCounts.is_connection()) {
    // TODO: connection
    return dst;
  }

  if (auto pv = faceVertexCounts.get_value()) {
    std::vector<int32_t> val;
    if (pv.value().get(time, &val, value::TimeSampleInterpolationType::Held)) {
      dst = std::move(val);
    }
  }
  return dst;
}

const std::vector<int32_t> GeomMesh::get_faceVertexIndices(double time) const {
  std::vector<int32_t> dst;

  if (!faceVertexIndices.authored() || faceVertexIndices.is_blocked()) {
    return dst;
  }

  if (faceVertexIndices.is_connection()) {
    // TODO: connection
    return dst;
  }

  if (auto pv = faceVertexIndices.get_value()) {
    std::vector<int32_t> val;
    if (pv.value().get(time, &val, value::TimeSampleInterpolationType::Held)) {
      dst = std::move(val);
    }
  }
  return dst;
}

std::vector<value::token> GeomMesh::get_joints() const {
  constexpr auto kSkelJoints = "skel:joints";
#if 0
  if (has_primvar(kSkelJoints)) {
    // 'primvars:skel:joints'
    std::string err;
    GeomPrimvar primvar;
    if (!get_primvar(kSkelJoints, &primvar, &err)) {
      DCOUT("Invalid `skel:joints` primvar. err = " << err);
      return {};
    }

    if (primvar.has_indices()) {
      // indexed primvar for skel:joint is not supported 
      DCOUT("Indexed primvar is not supported for `skel:joints`");
      return {};
    }

    const Attribute &attr = primvar.get_attribute();
    if (!attr.is_uniform()) {
      DCOUT("`skel:joints` must be uniform attribute");
      return {};
    }

    std::vector<value::token> dst;
    if (!primvar.get_value(&dst)) {
      DCOUT("`skel:joints` must be token[] type, but got " << primvar.type_name());
    }
  } else {
#endif
  {
    // lookup `skel:joints` prop
    if (!props.count(kSkelJoints)) {
      return {};
    }

    const auto &prop = props.at(kSkelJoints);
    if (prop.get_attribute().is_uniform() && prop.get_attribute().type_name() == "token[]") {
      
      std::vector<value::token> dst;
      if (!prop.get_attribute().get_value(&dst)) {
        return {};
      }

      return dst;
      
    } 
    DCOUT("`skel:joints` must be uniform token[] attribute, but got " << prop.value_type_name() << " (or Relationship))");
  }
  return {};
}

// static
bool GeomSubset::ValidateSubsets(
    const std::vector<const GeomSubset *> &subsets,
    const size_t elementCount,
    const FamilyType &familyType, std::string *err) {

  if (subsets.empty()) {
    return true;
  }

  // All subsets must have the same elementType.
  GeomSubset::ElementType elementType = subsets[0]->elementType.get_value();
  for (const auto psubset : subsets) {
    if (psubset->elementType.get_value() != elementType) {
      if (err) {
        (*err) = fmt::format("GeomSubset {}'s elementType must be `{}`, but got `{}`.\n",
          psubset->name, to_string(elementType), to_string(psubset->elementType.get_value()));
      }

      return false;
    }
  }

  std::set<int32_t> indicesInFamily;

  bool valid = true;
  std::stringstream ss;

  // TODO: TimeSampled indices
  for (const auto psubset : subsets) {
    Animatable<std::vector<int32_t>> indices;
    if (!psubset->indices.get_value(&indices)) {
      ss << fmt::format("GeomSubset {}'s indices is not value Attribute. Connection or ValueBlock?\n",
          psubset->name);

      valid = false;
    }

    if (indices.is_blocked()) {
      ss << fmt::format("GeomSubset {}'s indices is Value Blocked.\n", psubset->name);
      valid = false;
    }

    if (indices.is_timesamples() || !indices.has_value()) {
      ss << fmt::format("ValidateSubsets: TimeSampled GeomSubset.indices is not yet supported.\n");
      valid = false;
    }

    std::vector<int32_t> subsetIndices;
    if (!indices.get_scalar(&subsetIndices)) {
      ss << fmt::format("ValidateSubsets: Internal error. Failed to get GeomSubset.indices.\n");
      valid = false;
    }

    for (const int32_t index : subsetIndices) {
      if (!indicesInFamily.insert(index).second && (familyType != FamilyType::Unrestricted)) {
        ss << fmt::format("Found overlapping index {} in GeomSubset `{}`\n", index, psubset->name);
        valid = false;
      }
    }
  }


  // Make sure every index appears exactly once if it's a partition.
  if ((familyType == FamilyType::Partition) && (indicesInFamily.size() != elementCount)) {
    ss << fmt::format("ValidateSubsets: The number of unique indices {} must be equal to input elementCount {}\n", indicesInFamily.size(), elementCount);
    valid = false;
  }

  // Ensure that the indices are in the range [0, faceCount)
  size_t maxIndex = static_cast<size_t>(*indicesInFamily.rbegin());
  int minIndex = *indicesInFamily.begin();

  if (maxIndex >= elementCount) {
    ss << fmt::format("ValidateSubsets: All indices must be in range [0, elementSize {}), but one or more indices are greater than elementSize. Maximum = {}\n", elementCount, maxIndex);

    valid = false;
  }

  if (minIndex < 0) {
    ss << fmt::format("ValidateSubsets: Found one or more indices that are less than 0. Minumum = {}\n", minIndex);

    valid = false;
  }

  if (!valid) {
    if (err) {
      (*err) += ss.str();
    }
  }

  return true;

}

}  // namespace tinyusdz

// SPDX-License-Identifier: Apache 2.0
// Copyright 2024-Present Light Transport Entertainment, Inc.
//
// Evaluate Attribute API
//
// TODO:
// - [ ] Reduce template code to speed-up compilation
//
#pragma once

#include <map>

#include "prim-types.hh"
#include "stage.hh"
#include "usdGeom.hh"
#include "usdShade.hh"
#include "usdSkel.hh"
#include "usdLux.hh"
#include "value-types.hh"
#include "value-type-macros.inc"
#include "prim-type-macros.inc"
#include "tiny-format.hh"

namespace tinyusdz {
namespace tydra {

///
/// Terminal Attribute value at specified timecode.
///
/// - No `None`(Value Blocked)
/// - No connection(connection target is followed and resolved(fetch 'value producing attribute' in pxrUSD terminology))
/// - No timeSampled value
///
class TerminalAttributeValue {
 public:
  TerminalAttributeValue() = default;

  TerminalAttributeValue(const value::Value &v) : _empty{false}, _value(v) {}
  TerminalAttributeValue(value::Value &&v)
      : _empty{false}, _value(std::move(v)) {}

  // "empty" attribute(type info only)
  void set_empty_attribute(const std::string &type_name) {
    _empty = true;
    _type_name = type_name;
  }

  TerminalAttributeValue(const std::string &type_name) {
    set_empty_attribute(type_name);
  }

  bool is_empty() const { return _empty; }

  template <typename T>
  const T *as() const {
    if (_empty) {
      return nullptr;
    }
    return _value.as<T>();
  }

  template <typename T>
  bool is() const {
    if (_empty) {
      return false;
    }

    if (_value.as<T>()) {
      return true;
    }
    return false;
  }

  void set_value(const value::Value &v) {
    _value = v;
    _empty = false;
  }

  void set_value(value::Value &&v) {
    _value = std::move(v);
    _empty = false;
  }

  const std::string type_name() const {
    if (_empty) {
      return _type_name;
    }

    return _value.type_name();
  }

  uint32_t type_id() const {
    if (_empty) {
      return value::GetTypeId(_type_name);
    }

    return _value.type_id();
  }

  Variability variability() const { return _variability; }
  Variability &variability() { return _variability; }

  const AttrMeta &meta() const { return _meta; }
  AttrMeta &meta() { return _meta; }

 private:
  bool _empty{true};
  std::string _type_name;
  Variability _variability{Variability::Varying};
  value::Value _value{nullptr};
  AttrMeta _meta;
};

///
/// Evaluate Attribute of the specied Prim and retrieve terminal Attribute
/// value.
///
/// - If the attribute is empty(e.g. `float outputs:r`), return "empty"
/// Attribute
/// - If the attribute is scalar value, simply returns it.
/// - If the attribute is timeSamples value, evaluate the value at specified
/// time.
/// - If the attribute is connection, follow the connection target
///
/// @param[in] stage Stage
/// @param[in] prim Prim
/// @param[in] attr_name Attribute name
/// @param[out] value Evaluated terminal attribute value.
/// @param[out] err Error message(filled when false returned). Set nullptr if you don't need error message.
/// @param[in] t (optional) TimeCode(for timeSamples Attribute)
/// @param[in] tinterp (optional) Interpolation type for timeSamples value
///
/// Return false when:
///
/// - If the attribute is None(ValueBlock)
/// - Requested attribute not found in a Prim.
/// - Invalid connection(e.g. type mismatch, circular referencing, targetPath
/// points non-existing path, etc),
/// - Other error happens.
///
bool EvaluateAttribute(
    const tinyusdz::Stage &stage, const tinyusdz::Prim &prim,
    const std::string &attr_name, TerminalAttributeValue *value,
    std::string *err, const double t = tinyusdz::value::TimeCode::Default(),
    const tinyusdz::value::TimeSampleInterpolationType tinterp =
        tinyusdz::value::TimeSampleInterpolationType::Linear);

///
/// Evaluate Attribute and retrieve terminal Attribute value.
///
/// - If the attribute is empty(e.g. `float outputs:r`), return "empty"
/// Attribute
/// - If the attribute is scalar value, simply returns it.
/// - If the attribute is timeSamples value, evaluate the value at specified
/// time.
/// - If the attribute is connection, follow the connection target
///
/// @param[in] stage Stage
/// @param[in] attr Attribute
/// @param[in] attr_name Attribute name. This is only used in error message, so it can be empty.
/// @param[out] value Evaluated terminal attribute value.
/// @param[out] err Error message(filled when false returned). Set nullptr if you don't need error message.
/// @param[in] t (optional) TimeCode(for timeSamples Attribute)
/// @param[in] tinterp (optional) Interpolation type for timeSamples value
///
bool EvaluateAttribute(
    const tinyusdz::Stage &stage, const Attribute &attr,
    const std::string &attr_name, TerminalAttributeValue *value,
    std::string *err, const double t = tinyusdz::value::TimeCode::Default(),
    const tinyusdz::value::TimeSampleInterpolationType tinterp =
        tinyusdz::value::TimeSampleInterpolationType::Linear);


//
// Typed version
//
template<typename T>
bool EvaluateTypedAttribute(
    const tinyusdz::Stage &stage,
    const TypedAttribute<T> &attr,
    const std::string &attr_name,
    T *value,
    std::string *err);


// NOTE: std::string uses the specialization, so no extern template.
#define EXTERN_EVALUATE_TYPED_ATTRIBUTE(__ty) \
extern template bool EvaluateTypedAttribute(const tinyusdz::Stage &stage, const TypedAttribute<__ty> &attr, const std::string &attr_name, __ty *value, std::string *err);

APPLY_FUNC_TO_VALUE_TYPES_NO_STRING(EXTERN_EVALUATE_TYPED_ATTRIBUTE)
template<> bool EvaluateTypedAttribute(const tinyusdz::Stage &stage, const TypedAttribute<std::string> &attr, const std::string &attr_name, std::string *value, std::string *err);

#undef EXTERN_EVALUATE_TYPED_ATTRIBUTE

template<typename T>
bool EvaluateTypedAnimatableAttribute(
    const tinyusdz::Stage &stage,
    const TypedAttribute<Animatable<T>> &attr,
    const std::string &attr_name,
    T *value,
    std::string *err, const double t = tinyusdz::value::TimeCode::Default(),
    const tinyusdz::value::TimeSampleInterpolationType tinterp =
        tinyusdz::value::TimeSampleInterpolationType::Linear);

#define EXTERN_EVALUATE_TYPED_ATTRIBUTE(__ty) \
extern template bool EvaluateTypedAnimatableAttribute(const tinyusdz::Stage &stage, const TypedAttribute<Animatable<__ty>> &attr, const std::string &attr_name, __ty *value, std::string *err, const double t, const value::TimeSampleInterpolationType tinter);

APPLY_FUNC_TO_VALUE_TYPES_NO_STRING(EXTERN_EVALUATE_TYPED_ATTRIBUTE)
template<> bool EvaluateTypedAnimatableAttribute(const tinyusdz::Stage &stage, const TypedAttribute<Animatable<std::string>> &attr, const std::string &attr_name, std::string *value, std::string *err, const double t, const value::TimeSampleInterpolationType tinter);

#undef EXTERN_EVALUATE_TYPED_ATTRIBUTE

template<typename T>
bool EvaluateTypedAttribute(
    const tinyusdz::Stage &stage,
    const TypedAttributeWithFallback<T> &attr,
    const std::string &attr_name,
    T *value,
    std::string *err);

#define EXTERN_EVALUATE_TYPED_ATTRIBUTE(__ty) \
extern template bool EvaluateTypedAttribute(const tinyusdz::Stage &stage, const TypedAttributeWithFallback<__ty> &attr, const std::string &attr_name, __ty *value, std::string *err);

APPLY_FUNC_TO_VALUE_TYPES_NO_STRING(EXTERN_EVALUATE_TYPED_ATTRIBUTE)
template<> bool EvaluateTypedAttribute(
    const tinyusdz::Stage &stage,
    const TypedAttributeWithFallback<std::string> &attr,
    const std::string &attr_name,
    std::string *value,
    std::string *err);

#undef EXTERN_EVALUATE_TYPED_ATTRIBUTE

template<typename T>
bool EvaluateTypedAnimatableAttribute(
    const tinyusdz::Stage &stage,
    const TypedAttributeWithFallback<Animatable<T>> &attr,
    const std::string &attr_name,
    T *value,
    std::string *err, const double t = tinyusdz::value::TimeCode::Default(),
    const tinyusdz::value::TimeSampleInterpolationType tinterp =
        tinyusdz::value::TimeSampleInterpolationType::Linear);

#define EXTERN_EVALUATE_TYPED_ATTRIBUTE(__ty) \
extern template bool EvaluateTypedAnimatableAttribute(const tinyusdz::Stage &stage, const TypedAttributeWithFallback<Animatable<__ty>> &attr, const std::string &attr_name, __ty *value, std::string *err, const double t, const value::TimeSampleInterpolationType tinter);

APPLY_FUNC_TO_VALUE_TYPES_NO_STRING(EXTERN_EVALUATE_TYPED_ATTRIBUTE)
template<> bool EvaluateTypedAnimatableAttribute(
    const tinyusdz::Stage &stage,
    const TypedAttributeWithFallback<Animatable<std::string>> &attr,
    const std::string &attr_name,
    std::string *value,
    std::string *err, const double t, const tinyusdz::value::TimeSampleInterpolationType tinterp);

#undef EXTERN_EVALUATE_TYPED_ATTRIBUTE

}  // namespace tydra
}  // namespace tinyusdz

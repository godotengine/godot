// SPDX-License-Identifier: Apache 2.0
// Copyright 2022-Present Light Transport Entertainment, Inc.
//
#include "attribute-eval.hh"
#include "scene-access.hh"

#include "common-macros.inc"
#include "pprinter.hh"
#include "tiny-format.hh"
#include "value-pprint.hh"

namespace tinyusdz {
namespace tydra {

// For PUSH_ERROR_AND_RETURN
#define PushError(msg) \
  if (err) {           \
    (*err) +=  msg;     \
  }


template<typename T>
bool EvaluateTypedAttributeImpl(
    const tinyusdz::Stage &stage, const TypedAttribute<Animatable<T>> &attr,
    const std::string &attr_name,
    T *value,
    std::string *err,
    const double t, const value::TimeSampleInterpolationType tinterp)
{

  if (attr.has_value()) {

    return attr.get_value(value);

  } else if (attr.has_connection()) {
    // Follow connection target Path(singple targetPath only).
    std::vector<Path> pv = attr.connections();
    if (pv.empty()) {
      PUSH_ERROR_AND_RETURN(fmt::format("Connection targetPath is empty for Attribute {}.", attr_name));
    }

    if (pv.size() > 1) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Multiple targetPaths assigned to .connection for Attribute {}.", attr_name));
    }

    auto target = pv[0];

    std::string targetPrimPath = target.prim_part();
    std::string targetPrimPropName = target.prop_part();
    DCOUT("connection targetPath : " << target << "(Prim: " << targetPrimPath
                                     << ", Prop: " << targetPrimPropName
                                     << ")");

    auto targetPrimRet =
        stage.GetPrimAtPath(Path(targetPrimPath, /* prop */ ""));
    if (targetPrimRet) {
      // Follow the connetion
      const Prim *targetPrim = targetPrimRet.value();

      std::string abs_path = target.full_path_name();

      TerminalAttributeValue attr_value;

      bool ret = EvaluateAttribute(stage, *targetPrim, targetPrimPropName,
                                   &attr_value, err, t, tinterp);

      if (!ret) {
        return false;
      }

      if (const auto pav = attr_value.as<T>()) {
        (*value) = (*pav);
        return true;
      } else {
        PUSH_ERROR_AND_RETURN(
            fmt::format("Attribute of Connection targetPath has different type `{}. Expected `{}`. Attribute `{}`.", attr_value.type_name(), value::TypeTraits<T>::type_name(), attr_name));
      }


    } else {
      PUSH_ERROR_AND_RETURN(targetPrimRet.error());
    }
  } else if (attr.is_blocked()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Attribute `{}` is ValueBlocked(None).", attr_name));
  } else {

    PUSH_ERROR_AND_RETURN("Internal error. Invalid TypedAttribute<Animatable<T>> value.");

  }

  return false;
}


namespace {

// Convert TypedAttribute Connection to Attribute Connection.
// If TypedAttribute has value, return Attribute with empty value.
// TODO: make error when Attribute is not 'connection'.
template<typename T>
Attribute ToAttributeConnection(
  const TypedAttribute<Animatable<T>> &input)
{
  Attribute attr;
  if (input.is_blocked()) {
    attr.set_blocked(true);
    attr.variability() = Variability::Varying;
  } else if (input.is_value_empty()) {
    // empty = set type info only
    attr.set_type_name(value::TypeTraits<T>::type_name());
    attr.variability() = Variability::Varying;

  } else if (input.is_connection()) {

    attr.set_connections(input.connections());

  } else{
    attr.set_type_name(value::TypeTraits<T>::type_name());
    attr.variability() = Variability::Varying;
  }

  // Copy metadata
  attr.metas() =  input.metas();

  return attr;
}

} // namespace

template<> bool EvaluateTypedAnimatableAttribute<std::string>(
    const tinyusdz::Stage &stage, const TypedAttribute<Animatable<std::string>> &tattr,
    const std::string &attr_name,
    std::string *value_out,
    std::string *err,
    const double t,
    const value::TimeSampleInterpolationType tinterp) {

  if (!value_out) {
    PUSH_ERROR_AND_RETURN("`value_out` param is nullptr.");
  }

  if (tattr.is_blocked()) {
    if (err) {
      (*err) += "Attribute is Blocked.\n";
    }
    return false;
  } else if (tattr.is_value_empty()) {
    if (err) {
      (*err) += "Attribute value is empty.\n";
    }
    return false;
  } else if (tattr.is_connection()) {

    // Follow targetPath
    Attribute attr = ToAttributeConnection(tattr);

    TerminalAttributeValue value;
    bool ret = EvaluateAttribute(stage, attr, attr_name, &value, err,
                                 t, tinterp);

    if (!ret) {
      return false;
    }

    if (auto pv = value.as<std::string>()) {
      (*value_out) = *pv;
      return true;
    }

    // Allow `token` typed value in the attribute of targetPath.
    if (auto pv = value.as<value::token>()) {
      (*value_out) = pv->str();
      return true;
    }

    if (err) {
      (*err) += fmt::format("Type mismatch. Value producing attribute has type {}, but requested type is {}[]. Attribute: {}", value.type_name(), value::TypeTraits<std::string>::type_name(), attr_name);
    }

  } else {
    Animatable<std::string> value;
    if (tattr.get_value(&value)) {
      if (value.get(t, value_out, tinterp)) {
        return true;
      } else {
        if (err) {
          (*err) += fmt::format("Failed to get TypedAnimatableAttribute value: {} \n", attr_name);
        }
        return false;
      }
    }

    if (err) {
      (*err) += fmt::format("[Internal error] Invalid TypedAttribute? : {} \n", attr_name);
    }
  }
  return false;
}

template<typename T>
bool EvaluateTypedAnimatableAttribute(
    const tinyusdz::Stage &stage, const TypedAttribute<Animatable<T>> &tattr,
    const std::string &attr_name,
    T *value_out,
    std::string *err,
    const double t,
    const value::TimeSampleInterpolationType tinterp) {

  if (!value_out) {
    PUSH_ERROR_AND_RETURN("`value_out` param is nullptr.");
  }

  // Eval order:
  // - ValueBlocked?
  // - has value?(default value or timesamped value)
  // - has connection?

  if (tattr.is_blocked()) {
    if (err) {
      (*err) += "Attribute is Blocked.\n";
    }
    return false;
  } else if (tattr.has_value()) {
    Animatable<T> value;
    if (tattr.get_value(&value)) {
      if (value.get(t, value_out, tinterp)) {
        return true;
      } else {
        if (err) {
          (*err) += fmt::format("Failed to get TypedAnimatableAttribute value: {} \n", attr_name);
        }
        return false;
      }
    }
  } else if (tattr.has_connections()) {

    // Follow targetPath
    Attribute attr = ToAttributeConnection(tattr);

    TerminalAttributeValue value;
    bool ret = EvaluateAttribute(stage, attr, attr_name, &value, err,
                                 t, tinterp);

    if (!ret) {
      return false;
    }

    if (auto pv = value.as<T>()) {
      (*value_out) = *pv;
      return true;
    }

    if (err) {
      (*err) += fmt::format("Type mismatch. Value producing attribute has type {}, but requested type is {}[]. Attribute: {}", value.type_name(), value::TypeTraits<T>::type_name(), attr_name);
    }

  } else if (tattr.is_value_empty()) {
    if (err) {
      (*err) += "Attribute value is empty.\n";
    }
    return false;
  } else {
    if (err) {
      (*err) += fmt::format("[Internal error] Invalid TypedAttribute? : {} \n", attr_name);
    }
  }
  return false;
}


// template instanciations
#define EVALUATE_TYPED_ATTRIBUTE_INSTANCIATE(__ty) \
template bool EvaluateTypedAnimatableAttribute(const tinyusdz::Stage &stage, const TypedAttribute<Animatable<__ty>> &attr, const std::string &attr_name, __ty *value, std::string *err, const double t, const value::TimeSampleInterpolationType tinterp);

APPLY_FUNC_TO_VALUE_TYPES_NO_STRING(EVALUATE_TYPED_ATTRIBUTE_INSTANCIATE)

#undef EVALUATE_TYPED_ATTRIBUTE_INSTANCIATE



}  // namespace tydra
}  // namespace tinyusdz

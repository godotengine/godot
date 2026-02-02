// SPDX-License-Identifier: Apache 2.0
// Copyright 2021 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// Reconstruct concrete Prim from PropertyMap or PrimSpec.
//
// TODO:
//   - [ ] Refactor code
//
#include "prim-reconstruct.hh"

#include "prim-types.hh"
#include "str-util.hh"
#include "io-util.hh"
#include "tiny-format.hh"

#include "usdGeom.hh"
#include "usdSkel.hh"
#include "usdLux.hh"
#include "usdShade.hh"

#include "common-macros.inc"
#include "value-types.hh"

// For PUSH_ERROR_AND_RETURN
#define PushError(s) if (err) { (*err) = s + (*err); }
#define PushWarn(s) if (warn) { (*warn) = s + (*err); }

// __VA_ARGS__ does not allow empty, thus # of args must be 2+
#define PUSH_WARN_F(s, ...) PUSH_WARN(fmt::format(s, __VA_ARGS__))
#define PUSH_ERROR_AND_RETURN_F(s, ...) PUSH_ERROR_AND_RETURN(fmt::format(s, __VA_ARGS__))

//
// NOTE:
//
// There are mainly 5 variant of Primtive property(relationship/attribute)
//
// - TypedAttribute<T> : Uniform only. `uniform T` or `uniform T var.connect`
// - TypedAttribute<Animatable<T>> : Varying. `T var`, `T var = val`, `T var.connect` or `T value.timeSamples`
// - optional<T> : For output attribute(Just author it. e.g. `float outputs:rgb`)
// - Relationship : Typeless relation(e.g. `rel material:binding`)
// - TypedConnection : Typed relation(e.g. `token outputs:result = </material/diffuse.rgb>`)

namespace tinyusdz {
namespace prim {

//constexpr auto kTag = "[PrimReconstruct]";

constexpr auto kProxyPrim = "proxyPrim";
constexpr auto kVisibility = "visibility";
constexpr auto kExtent = "extent";
constexpr auto kPurpose = "purpose";
constexpr auto kMaterialBinding = "material:binding";
constexpr auto kMaterialBindingCollection = "material:binding:collection";
constexpr auto kMaterialBindingPreview = "material:binding:preview";
constexpr auto kSkelSkeleton = "skel:skeleton";
constexpr auto kSkelAnimationSource = "skel:animationSource";
constexpr auto kSkelBlendShapes = "skel:blendShapes";
constexpr auto kSkelBlendShapeTargets = "skel:blendShapeTargets";
constexpr auto kInputsVarname = "inputs:varname";

///
/// TinyUSDZ reconstruct some frequently used shaders(e.g. UsdPreviewSurface)
/// here, not in Tydra
///
template <typename T>
bool ReconstructShader(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    T *out,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options);

namespace {


struct ParseResult
{
  enum class ResultCode
  {
    Success,
    Unmatched,
    AlreadyProcessed,
    TypeMismatch,
    VariabilityMismatch,
    ConnectionNotAllowed,
    InvalidConnection,
    PropertyTypeMismatch,
    InternalError,
  };

  ResultCode code;
  std::string err;
};

#if 0
inline std::string to_string(ParseResult::ResultCode rescode) {
  switch (rescode) {
    case ParseResult::ResultCode::Success: return "success";
    case ParseResult::ResultCode::Unmatched: return "unmatched";
    case ParseResult::ResultCode::AlreadyProcessed: return "alreadyProcessed";
    case ParseResult::ResultCode::TypeMismatch: return "typeMismatch";
    case ParseResult::ResultCode::PropertyTypeMismatch: return "propertyTypeMismatch";
    case ParseResult::ResultCode::VariabilityMismatch: return "variabilityMismatch";
    case ParseResult::ResultCode::ConnectionNotAllowed: return "connectionNotAllowed";
    case ParseResult::ResultCode::InvalidConnection: return "invalidConnection";
    case ParseResult::ResultCode::InternalError: return "internalError";
  } 
  return "[[???ResultCode]]";
}
#endif

template<typename T>
static nonstd::optional<Animatable<T>> ConvertToAnimatable(const primvar::PrimVar &var)
{
  Animatable<T> dst;

  if (!var.is_valid()) {
    DCOUT("is_valid failed");
    DCOUT("has_value " << var.has_value());
    DCOUT("has_timesamples " << var.has_timesamples());
    return nonstd::nullopt;
  }

  bool ok = false;

  if (var.has_value()) {

    if (auto pv = var.get_value<T>()) {
      dst.set_default(pv.value());

      ok = true;
      //return std::move(dst);
    }
  }

  if (var.has_timesamples()) {
    for (size_t i = 0; i < var.ts_raw().size(); i++) {
      const value::TimeSamples::Sample &s = var.ts_raw().get_samples()[i];

      // Attribute Block?
      if (s.blocked || s.value.is_none()) {
        dst.add_blocked_sample(s.t);
      } else if (auto pv = s.value.get_value<T>()) {
        dst.add_sample(s.t, pv.value());
      } else {
        // Type mismatch
        DCOUT(i << "/" << var.ts_raw().size() << " type mismatch. expected " << value::TypeTraits<T>::type_name() << ", but got " << s.value.type_name());
        return nonstd::nullopt;
      }
    }

    ok = true;
  }

  if (ok) {
    return std::move(dst);
  }

  DCOUT("???");
  return nonstd::nullopt;
}

// Require special treatment for Extent(float3[2])
template<>
nonstd::optional<Animatable<Extent>> ConvertToAnimatable(const primvar::PrimVar &var)
{
  Animatable<Extent> dst;

  if (!var.is_valid()) {
    DCOUT("is_valid failed");
    return nonstd::nullopt;
  }

  bool value_ok = false;

  if (var.has_default()) {

    if (auto pv = var.get_value<std::vector<value::float3>>()) {
      if (pv.value().size() == 2) {
        Extent ext;
        ext.lower = pv.value()[0];
        ext.upper = pv.value()[1];

        dst.set_default(ext);

      } else {
        return nonstd::nullopt;
      }

      //return std::move(dst);
    }
    value_ok = true;
  }

  if (var.has_timesamples()) {
    for (size_t i = 0; i < var.ts_raw().size(); i++) {
      const value::TimeSamples::Sample &s = var.ts_raw().get_samples()[i];

      // Attribute Block?
      if (s.blocked || s.value.is_none()) {
        dst.add_blocked_sample(s.t);
      } else if (auto pv = s.value.get_value<std::vector<value::float3>>()) {
        if (pv.value().size() == 2) {
          Extent ext;
          ext.lower = pv.value()[0];
          ext.upper = pv.value()[1];
          dst.add_sample(s.t, ext);
        } else {
          DCOUT(i << "/" << var.ts_raw().size() << " array size mismatch.");
          return nonstd::nullopt;
        }
      } else {
        // Type mismatch
        DCOUT(i << "/" << var.ts_raw().size() << " type mismatch.");
        return nonstd::nullopt;
      }
    }

    value_ok = true;
    //return std::move(dst);
  }

  if (value_ok) {
    return std::move(dst);
  }

  DCOUT("???");
  return nonstd::nullopt;
}

#if 0 // TODO: remove. moved to prim-types.cc
static bool ConvertTokenAttributeToStringAttribute(
  const TypedAttribute<Animatable<value::token>> &inp,
  TypedAttribute<Animatable<std::string>> &out) {

  out.metas() = inp.metas();

  if (inp.is_blocked()) {
    out.set_blocked(true);
  } else if (inp.is_value_empty()) {
    out.set_value_empty();
  }

  if (inp.has_connections()) {
    out.set_connections(inp.get_connections());
  }

  if (inp.has_value()) {
    Animatable<value::token> toks;
    Animatable<std::string> strs;
    if (inp.get_value(&toks)) {
      if (toks.is_blocked()) {
        // TODO
      }

      if (toks.has_default()) {
        value::token tok;
        toks.get_scalar(&tok);
        strs.set(tok.str());
      }

      
      if (toks.has_timesamples()) {
        auto tok_ts = toks.get_timesamples();

        for (auto &item : tok_ts.get_samples()) {
          strs.add_sample(item.t, item.value.str());
        }
      }
    }
    out.set_value(strs);
  }

  return true;
}
#endif

#if 0 // not used anymore. TODO: remove
static bool ConvertStringDataAttributeToStringAttribute(
  const TypedAttribute<Animatable<value::StringData>> &inp,
  TypedAttribute<Animatable<std::string>> &out) {

  out.metas() = inp.metas();

  if (inp.is_blocked()) {
    out.set_blocked(true);
  } else if (inp.is_value_empty()) {
    out.set_value_empty();
  }


  if (inp.has_connections()) {
    out.set_connections(inp.get_connections());
  }
  
  if (inp.has_value()) {
    Animatable<value::StringData> toks;
    Animatable<std::string> strs;
    if (inp.get_value(&toks)) {
      if (toks.is_blocked()) {
        // TODO
      }

      if (toks.has_default()) {
        value::StringData tok;
        toks.get_scalar(&tok);
        strs.set(tok.value);
      }

      if (toks.has_timesamples()) {
        auto tok_ts = toks.get_timesamples();

        for (auto &item : tok_ts.get_samples()) {
          strs.add_sample(item.t, item.value.value);
        }
      }
    }
    out.set_value(strs);
  }

  return true;
}
#endif

// For animatable attribute(`varying`)
template<typename T>
static ParseResult ParseTypedAttribute(std::set<std::string> &table, /* inout */
  const std::string prop_name,
  const Property &prop,
  const std::string &name,
  TypedAttributeWithFallback<Animatable<T>> &target)
{
  ParseResult ret;

#if 0 // deprecated. TODO: Remove
  if (prop_name.compare(name + ".connect") == 0) {
    std::string propname = removeSuffix(name, ".connect");
    if (table.count(propname)) {
      DCOUT("Already processed: " << prop_name);
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }
    if (prop.is_connection()) {
      if (auto pv = prop.get_relationTarget()) {
        target.set_connection(pv.value());
        //target.variability = prop.attrib.variability;
        target.metas() = prop.get_attribute().metas();
        table.insert(propname);
        ret.code = ParseResult::ResultCode::Success;
        DCOUT("Added as property with connection: " << propname);
        return ret;
      } else {
        ret.code = ParseResult::ResultCode::InvalidConnection;
        ret.err = "Connection target not found.";
        return ret;
      }
    } else {
      ret.code = ParseResult::ResultCode::InternalError;
      ret.err = "Internal error. Unsupported/Unimplemented property type.";
      return ret;
    }
#endif
  if (prop_name.compare(name) == 0) {
    //if (table.count(name)) {
    //  ret.code = ParseResult::ResultCode::AlreadyProcessed;
    //  return ret;
    //}

    if (prop.is_relationship()) {
      ret.code = ParseResult::ResultCode::PropertyTypeMismatch;
      ret.err = fmt::format("Property {} must be Attribute, but declared as Relationhip.", name);
      return ret;
    }

    const Attribute &attr = prop.get_attribute();


    std::string attr_type_name = attr.type_name();
    if ((value::TypeTraits<T>::type_name() == attr_type_name) || (value::TypeTraits<T>::underlying_type_name() == attr_type_name)) {

      bool has_connections{false};
      bool has_default{false};
      bool has_timesamples{false};

      if (attr.has_connections()) {
        target.set_connections(attr.connections());
        //target.metas() = attr.metas();
        //table.insert(prop_name);
        //ret.code = ParseResult::ResultCode::Success;
        has_connections = true;
      }

      if (prop.get_property_type() == Property::Type::EmptyAttrib) {
        DCOUT("Added prop with empty value: " << name);
        target.set_value_empty();
        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      } else if (prop.get_property_type() == Property::Type::Attrib) {

        DCOUT("Adding typed prop: " << name);

        if (attr.is_blocked()) {
          // e.g. "float radius = None"
          target.set_blocked(true);
        } else if (attr.variability() == Variability::Uniform) {
          DCOUT("Property is uniform: " << name);
          // e.g. "float radius = 1.2"
          if (attr.get_var().is_timesamples()) {
            ret.code = ParseResult::ResultCode::VariabilityMismatch;
            ret.err = fmt::format("TimeSample value is assigned to `uniform` property `{}", name);
            return ret;
          }

          if (auto pv = attr.get_value<T>()) {
            target.set_value(pv.value());
          } else {
            ret.code = ParseResult::ResultCode::TypeMismatch;
            ret.err = fmt::format("Fallback. Failed to retrieve value with requested type `{}`.", value::TypeTraits<T>::type_name());
            return ret;
          }

        }
      
        Animatable<T> animatable_value;

        if (attr.get_var().has_timesamples()) {
          // e.g. "float radius.timeSamples = {0: 1.2, 1: 2.3}"

          if (auto av = ConvertToAnimatable<T>(attr.get_var())) {
            animatable_value = av.value();
            //target.set_value(anim);
          } else {
            // Conversion failed.
            DCOUT("ConvertToAnimatable failed.");
            ret.code = ParseResult::ResultCode::InternalError;
            ret.err = fmt::format("Converting timeSamples Attribute data failed for `{}`. Guess TimeSamples have values with different type(expected is `{}`)?", prop_name, value::TypeTraits<T>::type_name());
            return ret;
          }

          has_timesamples = true;
        }
        
        if (attr.get_var().has_value()) {
          if (auto pv = attr.get_var().get_value<T>()) {
            //target.set_value(pv.value());
            animatable_value.set(pv.value());
          } else {
            ret.code = ParseResult::ResultCode::InternalError;
            ret.err = fmt::format("Internal error. Invalid attribute value? get_value<{}> failed. Attribute has type {}", value::TypeTraits<T>::type_name(), attr.get_var().type_name());
            return ret;
          }

          has_default = true;
        }

        if (has_timesamples || has_default) {
          target.set_value(animatable_value);
        }
      }

      // connections only?
      if (has_connections && (!has_timesamples && !has_default)) {
        target.set_value_empty();
      }

      if (has_connections || has_timesamples || has_default) {

        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;

      } else {
        DCOUT("Invalid Property.type");
        ret.err = "Invalid Property type(internal error)";
        ret.code = ParseResult::ResultCode::InternalError;
        return ret;
      }
    } else {
      DCOUT("tyname = " << value::TypeTraits<T>::type_name() << ", attr.type = " << attr_type_name);
      ret.code = ParseResult::ResultCode::TypeMismatch;
      std::stringstream ss;
      ss  << "Property type mismatch. " << name << " expects type `"
              << value::TypeTraits<T>::type_name()
              << "` but defined as type `" << attr_type_name << "`";
      ret.err = ss.str();
      return ret;
    }

  }

  ret.code = ParseResult::ResultCode::Unmatched;
  return ret;
}

// For 'uniform' attribute
template<typename T>
static ParseResult ParseTypedAttribute(std::set<std::string> &table, /* inout */
  const std::string prop_name,
  const Property &prop,
  const std::string &name,
  TypedAttributeWithFallback<T> &target) /* out */
{
  ParseResult ret;

#if 0
  if (prop_name.compare(name + ".connect") == 0) {
    std::string propname = removeSuffix(name, ".connect");
    if (table.count(propname)) {
      DCOUT("Already processed: " << prop_name);
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }
    if (prop.is_connection()) {
      const Attribute &attr = prop.get_attribute();
      if (attr.is_connection()) {
        target.set_connections(attr.connections());
        //target.variability = prop.attrib.variability;
        target.metas() = prop.get_attribute().metas();
        table.insert(propname);
        ret.code = ParseResult::ResultCode::Success;
        DCOUT("Added as property with connection: " << propname);
        return ret;
      } else {
        ret.code = ParseResult::ResultCode::InvalidConnection;
        ret.err = "Connection target not found.";
        return ret;
      }
    } else {
      ret.code = ParseResult::ResultCode::InternalError;
      ret.err = "Internal error. Unsupported/Unimplemented property type.";
      return ret;
    }
#endif
  if (prop_name.compare(name) == 0) {
    //if (table.count(name)) {
    //  ret.code = ParseResult::ResultCode::AlreadyProcessed;
    //  return ret;
    //}

    if (prop.is_relationship()) {
      ret.code = ParseResult::ResultCode::PropertyTypeMismatch;
      ret.err = fmt::format("Property `{}` must be Attribute, but declared as Relationship.", name);
      
    }
    
    const Attribute &attr = prop.get_attribute();

    std::string attr_type_name = attr.type_name();
    if ((value::TypeTraits<T>::type_name() == attr_type_name) || (value::TypeTraits<T>::underlying_type_name() == attr_type_name)) {

      if (attr.has_connections()) {
        target.set_connections(attr.connections());
        //target.variability = prop.attrib.variability;
        target.metas() = prop.get_attribute().metas();
        table.insert(prop_name);
        ret.code = ParseResult::ResultCode::Success;
      }

      if (prop.get_property_type() == Property::Type::EmptyAttrib) {
        DCOUT("Added prop with empty value: " << name);
        target.set_value_empty();
        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      } else if (prop.get_property_type() == Property::Type::Attrib) {
        DCOUT("Adding prop: " << name);

        if (prop.get_attribute().variability() != Variability::Uniform) {
          ret.code = ParseResult::ResultCode::VariabilityMismatch;
          ret.err = fmt::format("Attribute `{}` must be `uniform` variability.", name);
          return ret;
        }

        if (attr.is_blocked()) {
          target.set_blocked(true);
        } else if (attr.get_var().has_default()) {
          if (auto pv = attr.get_value<T>()) {
            target.set_value(pv.value());
          } else {
            ret.code = ParseResult::ResultCode::InternalError;
            ret.err = "Internal data corrupsed.";
            return ret;
          }
        } else {
          ret.code = ParseResult::ResultCode::VariabilityMismatch;
          ret.err = "TimeSample or corrupted value assigned to a property where `uniform` variability is set.";
          return ret;
        }

        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      } else {
        DCOUT("Invalid Property.type");
        ret.err = "Invalid Property type(internal error)";
        ret.code = ParseResult::ResultCode::InternalError;
        return ret;
      }
    } else {
      DCOUT("tyname = " << value::TypeTraits<T>::type_name() << ", attr.type = " << attr_type_name);
      ret.code = ParseResult::ResultCode::TypeMismatch;
      std::stringstream ss;
      ss  << "Property type mismatch. " << name << " expects type `"
              << value::TypeTraits<T>::type_name()
              << "` but defined as type `" << attr_type_name << "`";
      ret.err = ss.str();
      return ret;
    }

  }

  ret.code = ParseResult::ResultCode::Unmatched;
  return ret;
}

// For animatable attribute(`varying`)
template<typename T>
static ParseResult ParseTypedAttribute(std::set<std::string> &table, /* inout */
  const std::string prop_name,
  const Property &prop,
  const std::string &name,
  TypedAttribute<Animatable<T>> &target) /* out */
{
  ParseResult ret;

#if 0
  if (prop_name.compare(name + ".connect") == 0) {
    std::string propname = removeSuffix(name, ".connect");
    if (table.count(propname)) {
      DCOUT("Already processed: " << prop_name);
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }
    if (prop.is_connection()) {
      const Attribute &attr = prop.get_attribute();
      if (attr.is_connection()) {
        target.set_connections(attr.connections());
        //target.variability = prop.attrib.variability;
        target.metas() = prop.get_attribute().metas();
        table.insert(propname);
        ret.code = ParseResult::ResultCode::Success;
        DCOUT("Added as property with connection: " << propname);
        return ret;
      } else {
        ret.code = ParseResult::ResultCode::InvalidConnection;
        ret.err = "Connection target not found.";
        return ret;
      }
    } else {
      ret.code = ParseResult::ResultCode::InternalError;
      ret.err = "Internal error. Unsupported/Unimplemented property type.";
      return ret;
    }
#endif
  if (prop_name.compare(name) == 0) {
    //if (table.count(name)) {
    //  ret.code = ParseResult::ResultCode::AlreadyProcessed;
    //  return ret;
    //}
    
    if (prop.is_relationship()) {
      ret.code = ParseResult::ResultCode::PropertyTypeMismatch;
      ret.err = fmt::format("Property `{}` must be Attribute, but declared as Relationship.", name);
      
    }

    const Attribute &attr = prop.get_attribute();

    if (attr.has_connections()) {
      target.set_connections(attr.connections());
      //target.variability = prop.attrib.variability;
      //target.metas() = prop.get_attribute().metas();
      //table.insert(prop_name);
      ret.code = ParseResult::ResultCode::Success;
    }

    std::string attr_type_name = attr.type_name();
    if ((value::TypeTraits<T>::type_name() == attr_type_name) || (value::TypeTraits<T>::underlying_type_name() == attr_type_name)) {
      if (prop.get_property_type() == Property::Type::EmptyAttrib) {
        DCOUT("Added prop with empty value: " << name);
        target.set_value_empty();
        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      } else if (prop.get_property_type() == Property::Type::Attrib) {

        DCOUT("Adding typed attribute: " << name);
        DCOUT("T.tyid = " << value::TypeTraits<T>::type_id() << ", var.tyid = " << attr.get_var().type_id());

        if (attr.is_blocked()) {
          DCOUT("Attribute is blocked: " << name);
          // e.g. "uniform float radius = None"
          target.set_blocked(true);
        }

        const auto &var = attr.get_var();
        DCOUT("has_value = " << var.has_value());

        if (var.has_default() || var.has_timesamples()) {
          if (auto av = ConvertToAnimatable<T>(var)) {
            target.set_value(av.value());
          } else {
            DCOUT("ConvertToAnimatable failed.");
            ret.code = ParseResult::ResultCode::InternalError;
            ret.err = "Converting Attribute data failed. Maybe TimeSamples have values with different types?";
            return ret;
          }

          DCOUT("Added typed attribute: " << name);

          target.metas() = attr.metas();
          table.insert(name);
          ret.code = ParseResult::ResultCode::Success;
          return ret;
        }
      } else {
        DCOUT("Invalid Property.type");
        ret.err = "Invalid Property type(internal error)";
        ret.code = ParseResult::ResultCode::InternalError;
        return ret;
      }
    } else {
      DCOUT("tyname = " << value::TypeTraits<T>::type_name() << ", attr.type = " << attr_type_name);
      ret.code = ParseResult::ResultCode::TypeMismatch;
      std::stringstream ss;
      ss  << "Property type mismatch. " << name << " expects type `"
              << value::TypeTraits<T>::type_name()
              << "` but defined as type `" << attr_type_name << "`";
      ret.err = ss.str();
      return ret;
    }

    if (attr.has_connections()) { // connection only
      DCOUT("Connection only attribute.");
      target.metas() = prop.get_attribute().metas();
      table.insert(prop_name);
      ret.code = ParseResult::ResultCode::Success;
      return ret;
    } else {
      DCOUT("???.");
    }
    return ret;
  }

  ret.code = ParseResult::ResultCode::Unmatched;
  return ret;
}

// TODO: Unify code with TypedAttribute<Animatable<T>> variant
template<typename T>
static ParseResult ParseTypedAttribute(std::set<std::string> &table, /* inout */
  const std::string prop_name,
  const Property &prop,
  const std::string &name,
  TypedAttribute<T> &target) /* out */
{
  ParseResult ret;

  DCOUT(fmt::format("prop name {}", prop_name));

#if 0
  if (prop_name.compare(name + ".connect") == 0) {
    std::string propname = removeSuffix(name, ".connect");
    if (table.count(propname)) {
      DCOUT("Already processed: " << prop_name);
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }
    if (prop.is_connection()) {
      const Attribute &attr = prop.get_attribute();
      if (attr.is_connection()) {
        target.set_connections(attr.connections());
        //target.variability = prop.attrib.variability;
        target.metas() = prop.get_attribute().metas();
        table.insert(propname);
        ret.code = ParseResult::ResultCode::Success;
        DCOUT("Added as property with connection: " << propname);
        return ret;
      } else {
        ret.code = ParseResult::ResultCode::InvalidConnection;
        ret.err = "Connection target not found.";
        return ret;
      }
    } else {
      ret.code = ParseResult::ResultCode::InternalError;
      ret.err = "Internal error. Unsupported/Unimplemented property type.";
      return ret;
    }
#endif
  if (prop_name.compare(name) == 0) {
    DCOUT(fmt::format("prop name match {}", name));
    //if (table.count(name)) {
    //  ret.code = ParseResult::ResultCode::AlreadyProcessed;
    //  return ret;
    //}

    const Attribute &attr = prop.get_attribute();
    std::string attr_type_name = attr.type_name();
    DCOUT(fmt::format("prop name {}, type = {}", prop_name, attr_type_name));
    if ((value::TypeTraits<T>::type_name() == attr_type_name) || (value::TypeTraits<T>::underlying_type_name() == attr_type_name)) {

      bool has_connections{false};
      bool has_default{false};

      if (attr.has_connections()) {
        target.set_connections(attr.connections());
        //target.variability = prop.attrib.variability;
        //target.metas() = prop.get_attribute().metas();
        //table.insert(prop_name);
        //ret.code = ParseResult::ResultCode::Success;
        has_connections = true;
      }

      if (prop.get_property_type() == Property::Type::EmptyAttrib) {
        DCOUT("Added prop with empty value: " << name);
        target.set_value_empty();
        has_default = true; // has empty 'default' 
      } else if (prop.get_property_type() == Property::Type::Attrib) {

        DCOUT("Adding typed attribute: " << name);

        if (prop.get_attribute().variability() != Variability::Uniform) {
          ret.code = ParseResult::ResultCode::VariabilityMismatch;
          ret.err = fmt::format("Attribute `{}` must be `uniform` variability.", name);
          return ret;
        }

        if (attr.get_var().has_timesamples()) {
          ret.code = ParseResult::ResultCode::VariabilityMismatch;
          ret.err = "TimeSample or corrupted value assigned to a property where `uniform` variability is set.";
          return ret;
        }

        if (attr.is_blocked()) {
          target.set_blocked(true);
          has_default = true;
        } else if (attr.get_var().has_default()) {
          if (auto pv = attr.get_value<T>()) {
            target.set_value(pv.value());
            has_default = true;
          } else {
            ret.code = ParseResult::ResultCode::VariabilityMismatch;
            ret.err = "Internal data corrupsed.";
            return ret;
          }
        }

      }

      if (has_connections || has_default) {
        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      } else {
        ret.code = ParseResult::ResultCode::InternalError;
        ret.err = "Internal data corrupsed.";
        return ret;
      }
      
    } else {
      DCOUT("tyname = " << value::TypeTraits<T>::type_name() << ", attr.type = " << attr_type_name);
      ret.code = ParseResult::ResultCode::TypeMismatch;
      std::stringstream ss;
      ss  << "Property type mismatch. " << name << " expects type `"
              << value::TypeTraits<T>::type_name()
              << "` but defined as type `" << attr_type_name << "`";
      ret.err = ss.str();
      return ret;
    }

    return ret;
  }

  ret.code = ParseResult::ResultCode::Unmatched;

  return ret;
}

// Special case for Extent(float3[2]) type.
// TODO: Reuse code of ParseTypedAttribute as much as possible
static ParseResult ParseExtentAttribute(std::set<std::string> &table, /* inout */
  const std::string prop_name,
  const Property &prop,
  const std::string &name,
  TypedAttribute<Animatable<Extent>> &target) /* out */
{
  ParseResult ret;

#if 0
  if (prop_name.compare(name + ".connect") == 0) {
    std::string propname = removeSuffix(name, ".connect");
    if (table.count(propname)) {
      DCOUT("Already processed: " << prop_name);
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }
    if (prop.is_connection()) {
      const Attribute &attr = prop.get_attribute();
      if (attr.is_connection()) {
        target.set_connections(attr.connections());
        //target.variability = prop.attrib.variability;
        target.metas() = prop.get_attribute().metas();
        table.insert(propname);
        ret.code = ParseResult::ResultCode::Success;
        DCOUT("Added as property with connection: " << propname);
        return ret;
      } else {
        ret.code = ParseResult::ResultCode::InvalidConnection;
        ret.err = "Connection target not found.";
        return ret;
      }
    } else {
      ret.code = ParseResult::ResultCode::InternalError;
      ret.err = "Internal error. Unsupported/Unimplemented property type.";
      return ret;
    }
#endif
  if (prop_name.compare(name) == 0) {
    if (table.count(name)) {
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }

    const Attribute &attr = prop.get_attribute();

    std::string attr_type_name = attr.type_name();
    if (prop.get_property_type() == Property::Type::EmptyAttrib) {
      DCOUT("Added prop with empty value: " << name);
      target.set_value_empty();
      target.metas() = attr.metas();
      table.insert(name);
      ret.code = ParseResult::ResultCode::Success;
      return ret;
    } else if (prop.get_property_type() == Property::Type::Attrib) {

      //bool has_default{false};
      bool has_connections{false};

      if (attr.has_connections()) {
        target.set_connections(attr.connections());
        //target.variability = prop.attrib.variability;
        //target.metas() = prop.get_attribute().metas();
        //table.insert(prop_name);
        //ret.code = ParseResult::ResultCode::Success;
        //return ret;
        has_connections = true;
      }

      DCOUT("Adding typed extent attribute: " << name);

      if (attr.is_blocked()) {
        // e.g. "float3[] extent = None"
        target.set_blocked(true);
      }

#if 0
      } else {
        
        //
        // No variability check. allow `uniform extent`(promote to varying)
        //
        if (auto pv = attr.get_value<std::vector<value::float3>>()) {
          if (pv.value().size() != 2) {
            ret.code = ParseResult::ResultCode::TypeMismatch;
            ret.err = fmt::format("`extent` must be `float3[2]`, but got array size {}", pv.value().size());
            return ret;
          }

          Extent ext;
          ext.lower = pv.value()[0];
          ext.upper = pv.value()[1];

          //target.set_value(ext);
          animatable_value.set(ext);
        } else {
          ret.code = ParseResult::ResultCode::TypeMismatch;
          ret.err = fmt::format("`extent` must be `float3[]` type, but got `{}`", attr.type_name());
          return ret;
        }
      }

      if (attr.get_var().has_timesamples()) {
        // e.g. "float3[] extent.timeSamples = ..."

        if (auto av = ConvertToAnimatable<Extent>(attr.get_var())) {
          animatable_value.set(av.value().get_timesamples());
          //target.set_value(anim);
          
          has_timesamples = true;
        } else {
          // Conversion failed.
          DCOUT("ConvertToAnimatable failed.");
          ret.code = ParseResult::ResultCode::InternalError;
          ret.err = "Converting Attribute data failed. Maybe TimeSamples have values with different types or invalid array size?";
          return ret;
        }
      }

      if (has_default || has_timesamples) {
        DCOUT("Added Extent attribute: " << name);
        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      } else {
        DCOUT("Internal error.");
        ret.code = ParseResult::ResultCode::InternalError;
        ret.err = "Internal error. Invalid Attribute data";
        return ret;
      }
#else
      
      const auto &var = attr.get_var();

      if (var.has_default() || var.has_timesamples()) {
        if (auto av = ConvertToAnimatable<Extent>(var)) {
          target.set_value(av.value());
        } else {
          DCOUT("ConvertToAnimatable failed.");
          ret.code = ParseResult::ResultCode::InternalError;
          ret.err = "Converting Attribute data failed. Maybe TimeSamples have values with different types?";
          return ret;
        }

        DCOUT("Added typed extent attribute: " << name);

        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      }

      if (has_connections) {
        DCOUT("Added Extent connection attribute: " << name);
        target.metas() = attr.metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      }

#endif

    } else {
      DCOUT("Invalid Property.type");
      ret.err = "Invalid Property type(internal error)";
      ret.code = ParseResult::ResultCode::InternalError;
      return ret;
    }

  }

  ret.code = ParseResult::ResultCode::Unmatched;
  return ret;
}


// Empty allowedTokens = allow all
template <class E, size_t N>
static nonstd::expected<bool, std::string> CheckAllowedTokens(
    const std::array<std::pair<E, const char *>, N> &allowedTokens,
    const std::string &tok) {
  if (allowedTokens.empty()) {
    return true;
  }

  for (size_t i = 0; i < N; i++) {
    if (tok.compare(std::get<1>(allowedTokens[i])) == 0) {
      return true;
    }
  }

  std::vector<std::string> toks;
  for (size_t i = 0; i < N; i++) {
    toks.push_back(std::get<1>(allowedTokens[i]));
  }

  std::string s = join(", ", tinyusdz::quote(toks));

  return nonstd::make_unexpected("Allowed tokens are [" + s + "] but got " +
                                 quote(tok) + ".");
};

// Allowed syntax:
//   "T varname"
template<typename T>
static ParseResult ParseShaderOutputTerminalAttribute(std::set<std::string> &table, /* inout */
  const std::string prop_name,
  const Property &prop,
  const std::string &name,
  TypedTerminalAttribute<T> &target) /* out */
{
  ParseResult ret;

#if 0 // Old code: TODO: Remove
  if (prop_name.compare(name + ".connect") == 0) {
    ret.code = ParseResult::ResultCode::ConnectionNotAllowed;
    ret.err = "Connection is not allowed for output terminal attribute.";
    return ret;
#endif
  if (prop_name.compare(name) == 0) {
    if (table.count(name)) {
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }

    if (prop.is_attribute_connection()) {
      ret.code = ParseResult::ResultCode::ConnectionNotAllowed;
      ret.err = "Connection is not allowed for output terminal attribute.";
      return ret;
    } else {

      if (prop.get_property_type() != Property::Type::EmptyAttrib) {
          DCOUT("Output Invalid shader output terminal attribute");
          ret.err = "No value should be assigned for shader output terminal attribute.";
          ret.code = ParseResult::ResultCode::InvalidConnection;
          return ret;
      }

      const Attribute &attr = prop.get_attribute();

      std::string attr_type_name = attr.type_name();

      bool attr_is_role_type = value::IsRoleType(attr_type_name);

      DCOUT("attrname = " << name);
      DCOUT("value typename = " << value::TypeTraits<T>::type_name());
      DCOUT("attr-type_name = " << attr_type_name);


      // First check if both types are same, then
      // Allow either type is role-types(e.g. allow color3f attribute for TypedTerminalAttribute<float3>)
      // TODO: Allow both role-types case?(e.g. point3f attribute for TypedTerminalAttribute<vector3f>)
      if (value::TypeTraits<T>::type_name() == attr_type_name) {
        DCOUT("Author output terminal attribute: " << name);
        target.set_authored(true);
        target.metas() = prop.get_attribute().metas();
        table.insert(name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      } else if (value::TypeTraits<T>::is_role_type()) {
        if (attr_is_role_type) {
          ret.code = ParseResult::ResultCode::TypeMismatch;
          ret.err = fmt::format("Attribute type mismatch. {} expects type `{}` but defined as type `{}`.", name, value::TypeTraits<T>::type_name(), attr_type_name);
          return ret;
        } else {
          if (value::TypeTraits<T>::underlying_type_name() == attr_type_name) {
            target.set_authored(true);
            target.set_actual_type_name(attr_type_name);
            target.metas() = prop.get_attribute().metas();
            table.insert(name);
            ret.code = ParseResult::ResultCode::Success;
            return ret;
          } else {
            ret.code = ParseResult::ResultCode::TypeMismatch;
            ret.err = fmt::format("Attribute type mismatch. {} expects type `{}`(and its underlying types) but defined as type `{}`.", name, value::TypeTraits<T>::type_name(), attr_type_name);
            return ret;
          }
        }
      } else if (attr_is_role_type) {
        if (value::TypeTraits<T>::is_role_type()) {
          ret.code = ParseResult::ResultCode::TypeMismatch;
          ret.err = fmt::format("Attribute type mismatch. {} expects type `{}` but defined as type `{}`.", name, value::TypeTraits<T>::type_name(), attr_type_name);
          return ret;
        } else {
          uint32_t attr_underlying_type_id = value::GetUnderlyingTypeId(attr_type_name);
          if (value::TypeTraits<T>::type_id() == attr_underlying_type_id) {
            target.set_authored(true);
            target.set_actual_type_name(attr_type_name);
            target.metas() = prop.get_attribute().metas();
            table.insert(name);
            ret.code = ParseResult::ResultCode::Success;
            return ret;
          } else {
            ret.code = ParseResult::ResultCode::TypeMismatch;
            ret.err = fmt::format("Attribute type mismatch. {} expects type `{}` but defined as type `{}`(and its underlying types).", name, value::TypeTraits<T>::type_name(), attr_type_name);
            return ret;
          }
        }

      } else {
        DCOUT("attr.type = " << attr_type_name);
        ret.code = ParseResult::ResultCode::TypeMismatch;
        ret.err = fmt::format("Property type mismatch. {} expects type `{}` but defined as type `{}`.", name, value::TypeTraits<T>::type_name(), attr_type_name);
        return ret;
      }
    }
  }

  ret.code = ParseResult::ResultCode::Unmatched;
  return ret;
}

#if 0 // TODO: Remove since not used.
// Allowed syntax:
//   "token outputs:surface"
//   "token outputs:surface.connect = </path/to/conn/>"
static ParseResult ParseShaderOutputProperty(std::set<std::string> &table, /* inout */
  const std::string prop_name,
  const Property &prop,
  const std::string &name,
  nonstd::optional<Relationship> &target) /* out */
{
  ParseResult ret;

  if (prop_name.compare(name + ".connect") == 0) {
    std::string propname = removeSuffix(name, ".connect");
    if (table.count(propname)) {
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }
    if (auto pv = prop.get_relationTarget()) {
      Relationship rel;
      rel.set(pv.value());
      rel.metas() = prop.get_attribute().metas();
      target = rel;
      table.insert(propname);
      ret.code = ParseResult::ResultCode::Success;
      return ret;
    }
  } else if (prop_name.compare(name) == 0) {
    if (table.count(name)) {
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }

    if (prop.is_connection()) {
      const Attribute &attr = prop.get_attribute();
      if (attr.is_connection()) {
        Relationship rel;
        std::vector<Path> conns = attr.connections();

        if (conns.size() == 0) {
          ret.code = ParseResult::ResultCode::InternalError;
          ret.err = "Invalid shader output attribute with connection. connection targetPath size is zero.";
          return ret;
        }

        if (conns.size() == 1) {
          rel.set(conns[0]);
        } else if (conns.size() > 1) {
          rel.set(conns);
        }

        rel.metas() = prop.get_attribute().metas();
        target = rel;
        table.insert(prop_name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;

      } else {
        ret.code = ParseResult::ResultCode::InternalError;
        ret.err = "Invalid shader output attribute with connection.";
        return ret;
      }
    } else {

      const Attribute &attr = prop.get_attribute();

      std::string attr_type_name = attr.type_name();
      if (value::TypeTraits<value::token>::type_name() == attr_type_name) {
        if (prop.get_property_type() == Property::Type::EmptyAttrib) {
          Relationship rel;
          rel.set_novalue();
          rel.metas() = prop.get_attribute().metas();
          table.insert(name);
          target = rel;
          ret.code = ParseResult::ResultCode::Success;
          return ret;
        } else {
          DCOUT("Output Invalid Property.type");
          ret.err = "Invalid connection or value assigned for output attribute.";
          ret.code = ParseResult::ResultCode::InvalidConnection;
          return ret;
        }
      } else {
        DCOUT("attr.type = " << attr.type_name());
        ret.code = ParseResult::ResultCode::TypeMismatch;
        std::stringstream ss;
        ss  << "Property type mismatch. " << name << " expects type `token` but defined as type `" << attr.type_name() << "`";
        ret.err = ss.str();
        return ret;
      }
    }
  }

  ret.code = ParseResult::ResultCode::Unmatched;
  return ret;
}
#endif

// Allowed syntax:
//   "token outputs:surface.connect = </path/to/conn/>"
static ParseResult ParseShaderInputConnectionProperty(std::set<std::string> &table, /* inout */
  const std::string prop_name,
  const Property &prop,
  const std::string &name,
  TypedConnection<value::token> &target) /* out */
{
  ParseResult ret;
  ret.code = ParseResult::ResultCode::InternalError;

#if 0
  if (prop_name.compare(name + ".connect") == 0) {
    std::string propname = removeSuffix(name, ".connect");
    if (table.count(propname)) {
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }
    if (auto pv = prop.get_relationTarget()) {
      TypedConnection<value::token> conn;
      conn.set(pv.value());
      conn.metas() = prop.get_attribute().metas();
      target = conn;
      table.insert(propname);
      ret.code = ParseResult::ResultCode::Success;
      return ret;
    } else {
      ret.code = ParseResult::ResultCode::InternalError;
      ret.err = "Property does not contain connectionPath.";
      return ret;
    }
#endif
  if (prop_name.compare(name) == 0) {
    if (table.count(name)) {
      ret.code = ParseResult::ResultCode::AlreadyProcessed;
      return ret;
    }

    DCOUT("is_attribute = " << prop.is_attribute());
    DCOUT("is_attribute_connection = " << prop.is_attribute_connection());

    // allow empty value
    if (prop.is_empty()) {
      target.set_empty();
      target.metas() = prop.get_attribute().metas();
      table.insert(prop_name);
      ret.code = ParseResult::ResultCode::Success;
      return ret;
    } else if (prop.is_attribute_connection()) {
      const Attribute &attr = prop.get_attribute();
      if (attr.is_connection()) {
        target.set(attr.connections());
        target.metas() = prop.get_attribute().metas();

        table.insert(prop_name);
        ret.code = ParseResult::ResultCode::Success;
        return ret;
      } else {
        ret.code = ParseResult::ResultCode::InternalError;
        ret.err = "Property is invalid Attribute connection.";
        return ret;
      }
    } else {
      ret.code = ParseResult::ResultCode::InternalError;
      ret.err = fmt::format("Property `{}` must be Attribute connection.", prop_name);
      return ret;
    }
  }

  ret.code = ParseResult::ResultCode::Unmatched;
  return ret;
}

// Rel with single targetPath(or empty)
#define PARSE_SINGLE_TARGET_PATH_RELATION(__table, __prop, __propname, __target) \
  if (prop.first == __propname) { \
    if (__table.count(__propname)) { \
       continue; \
    } \
    if (!prop.second.is_relationship()) { \
      PUSH_ERROR_AND_RETURN(fmt::format("Property `{}` must be a Relationship.", __propname)); \
    } \
    const Relationship &rel = prop.second.get_relationship(); \
    if (rel.is_path()) { \
      __target = rel; \
      table.insert(prop.first); \
      DCOUT("Added rel " << __propname); \
      continue; \
    } else if (rel.is_pathvector()) { \
      if (rel.targetPathVector.size() == 1) { \
        __target = rel; \
        table.insert(prop.first); \
        DCOUT("Added rel " << __propname); \
        continue; \
      } \
      PUSH_ERROR_AND_RETURN(fmt::format("`{}` target is empty or has mutiple Paths. Must be single Path.", __propname)); \
    } else if (!rel.has_value()) { \
      /* define-only. accept  */ \
      __target = rel; \
      table.insert(prop.first); \
      DCOUT("Added rel " << __propname); \
    } else if (rel.is_blocked()) { \
      __target = rel; \
      table.insert(prop.first); \
      DCOUT("Added ValueBlocked rel " << __propname); \
    } else { \
      PUSH_ERROR_AND_RETURN(fmt::format("Internal error. Property `{}` is not a valid Relationship.", __propname)); \
    } \
  }

// Rel with targetPaths(single path or array of Paths)
#define PARSE_TARGET_PATHS_RELATION(__table, __prop, __propname, __target) \
  if (prop.first == __propname) { \
    if (__table.count(__propname)) { \
       continue; \
    } \
    if (!prop.second.is_relationship()) { \
      PUSH_ERROR_AND_RETURN(fmt::format("`{}` must be a Relationship", __propname)); \
    } \
    const Relationship &rel = prop.second.get_relationship(); \
    __target = rel; \
    table.insert(prop.first); \
    DCOUT("Added rel " << __propname); \
    continue; \
  }


#define PARSE_SHADER_TERMINAL_ATTRIBUTE(__table, __prop, __name, __klass, __target) { \
  ParseResult ret = ParseShaderOutputTerminalAttribute(__table, __prop.first, __prop.second, __name, __target); \
  if (ret.code == ParseResult::ResultCode::Success || ret.code == ParseResult::ResultCode::AlreadyProcessed) { \
    DCOUT("Added shader terminal attribute: " << __name); \
    continue; /* got it */\
  } else if (ret.code == ParseResult::ResultCode::Unmatched) { \
    /* go next */ \
  } else { \
    PUSH_ERROR_AND_RETURN(fmt::format("Parsing shader output property `{}` failed. Error: {}", __name, ret.err)); \
  } \
}

#if 0 // TODO: Remove since not used.
#define PARSE_SHADER_OUTPUT_PROPERTY(__table, __prop, __name, __klass, __target) { \
  ParseResult ret = ParseShaderOutputProperty(__table, __prop.first, __prop.second, __name, __target); \
  if (ret.code == ParseResult::ResultCode::Success || ret.code == ParseResult::ResultCode::AlreadyProcessed) { \
    DCOUT("Added shader output property: " << __name); \
    continue; /* got it */\
  } else if (ret.code == ParseResult::ResultCode::Unmatched) { \
    /* go next */ \
  } else { \
    PUSH_ERROR_AND_RETURN(fmt::format("Parsing shader output property `{}` failed. Error: {}", __name, ret.err)); \
  } \
}
#endif

#define PARSE_SHADER_INPUT_CONNECTION_PROPERTY(__table, __prop, __name, __klass, __target) { \
  ParseResult ret = ParseShaderInputConnectionProperty(__table, __prop.first, __prop.second, __name, __target); \
  if (ret.code == ParseResult::ResultCode::Success || ret.code == ParseResult::ResultCode::AlreadyProcessed) { \
    DCOUT("Added shader input connection: " << __name); \
    continue; /* got it */\
  } else if (ret.code == ParseResult::ResultCode::Unmatched) { \
    /* go next */ \
  } else { \
    PUSH_ERROR_AND_RETURN(fmt::format("Parsing shader property `{}` failed. Error: {}", __name, ret.err)); \
  } \
}

template <class E>
static nonstd::expected<bool, std::string> CheckAllowedTokens(
    const std::vector<std::pair<E, const char *>> &allowedTokens,
    const std::string &tok) {
  if (allowedTokens.empty()) {
    return true;
  }

  for (size_t i = 0; i < allowedTokens.size(); i++) {
    if (tok.compare(std::get<1>(allowedTokens[i])) == 0) {
      return true;
    }
  }

  std::vector<std::string> toks;
  for (size_t i = 0; i < allowedTokens.size(); i++) {
    toks.push_back(std::get<1>(allowedTokens[i]));
  }

  std::string s = join(", ", tinyusdz::quote(toks));

  return nonstd::make_unexpected("Allowed tokens are [" + s + "] but got " +
                                 quote(tok) + ".");
};

template <typename T>
nonstd::expected<T, std::string> EnumHandler(
    const std::string &prop_name, const std::string &tok,
    const std::vector<std::pair<T, const char *>> &enums) {
  auto ret = CheckAllowedTokens<T>(enums, tok);
  if (!ret) {
    return nonstd::make_unexpected(ret.error());
  }

  for (auto &item : enums) {
    if (tok == item.second) {
      return item.first;
    }
  }
  // Should never reach here, though.
  return nonstd::make_unexpected(
      quote(tok) + " is an invalid token for attribute `" + prop_name + "`");
}


} // namespace

#define PARSE_TYPED_ATTRIBUTE(__table, __prop, __name, __klass, __target) { \
  ParseResult ret = ParseTypedAttribute(__table, __prop.first, __prop.second, __name, __target); \
  if (ret.code == ParseResult::ResultCode::Success || ret.code == ParseResult::ResultCode::AlreadyProcessed) { \
    continue; /* got it */\
  } else if (ret.code == ParseResult::ResultCode::Unmatched) { \
    /* go next */ \
  } else { \
    PUSH_ERROR_AND_RETURN(fmt::format("Parsing attribute `{}` failed. Error: {}", __name, ret.err)); \
  } \
}

#define PARSE_TYPED_ATTRIBUTE_NOCONTINUE(__table, __prop, __name, __klass, __target) { \
  ParseResult ret = ParseTypedAttribute(__table, __prop.first, __prop.second, __name, __target); \
  if (ret.code == ParseResult::ResultCode::Success || ret.code == ParseResult::ResultCode::AlreadyProcessed) { \
    /* do nothing */ \
  } else if (ret.code == ParseResult::ResultCode::Unmatched) { \
    /* go next */ \
  } else { \
    PUSH_ERROR_AND_RETURN(fmt::format("Parsing attribute `{}` failed. Error: {}", __name, ret.err)); \
  } \
}

#define PARSE_EXTENT_ATTRIBUTE(__table, __prop, __name, __klass, __target) { \
  ParseResult ret = ParseExtentAttribute(__table, __prop.first, __prop.second, __name, __target); \
  if (ret.code == ParseResult::ResultCode::Success || ret.code == ParseResult::ResultCode::AlreadyProcessed) { \
    continue; /* got it */\
  } else if (ret.code == ParseResult::ResultCode::Unmatched) { \
    /* go next */ \
  } else { \
    PUSH_ERROR_AND_RETURN(fmt::format("Parsing attribute `extent` failed. Error: {}", ret.err)); \
  } \
}

template <typename EnumTy>
using EnumHandlerFun = std::function<nonstd::expected<EnumTy, std::string>(
    const std::string &)>;

static nonstd::expected<Axis, std::string> AxisEnumHandler(const std::string &tok) {
  using EnumTy = std::pair<Axis, const char *>;
  const std::vector<EnumTy> enums = {
      std::make_pair(Axis::X, "X"),
      std::make_pair(Axis::Y,
                     "Y"),
      std::make_pair(Axis::Z, "Z"),
  };
  return EnumHandler<Axis>("axis", tok, enums);
};

static nonstd::expected<Visibility, std::string> VisibilityEnumHandler(const std::string &tok) {
  using EnumTy = std::pair<Visibility, const char *>;
  const std::vector<EnumTy> enums = {
      std::make_pair(Visibility::Inherited, "inherited"),
      std::make_pair(Visibility::Invisible, "invisible"),
  };
  return EnumHandler<Visibility>(kVisibility, tok, enums);
};

static nonstd::expected<Purpose, std::string> PurposeEnumHandler(const std::string &tok) {
  using EnumTy = std::pair<Purpose, const char *>;
  const std::vector<EnumTy> enums = {
      std::make_pair(Purpose::Default, "default"),
      std::make_pair(Purpose::Proxy, "proxy"),
      std::make_pair(Purpose::Render, "render"),
      std::make_pair(Purpose::Guide, "guide"),
  };
  return EnumHandler<Purpose>("purpose", tok, enums);
};

static nonstd::expected<Orientation, std::string> OrientationEnumHandler(const std::string &tok) {
  using EnumTy = std::pair<Orientation, const char *>;
  const std::vector<EnumTy> enums = {
      std::make_pair(Orientation::RightHanded, "rightHanded"),
      std::make_pair(Orientation::LeftHanded, "leftHanded"),
  };
  return EnumHandler<Orientation>("orientation", tok, enums);
};

#if 1

template<typename T, typename EnumTy>
bool ParseUniformEnumProperty(
  const std::string &prop_name,
  bool strict_allowedToken_check,
  EnumHandlerFun<EnumTy> enum_handler,
  const Attribute &attr,
  TypedAttributeWithFallback<T> *result,
  std::string *warn = nullptr,
  std::string *err = nullptr)
{

  if (!result) {
    PUSH_ERROR_AND_RETURN("[Internal error] `result` arg is nullptr.");
  }

  if (attr.is_connection()) {
    PUSH_ERROR_AND_RETURN_F("Attribute connection is not supported in TinyUSDZ for built-in 'enum' token attribute: {}", prop_name);
  }


  if (attr.variability() == Variability::Uniform) {
    // scalar

    if (attr.is_blocked()) {
      result->set_blocked(true);
      return true;
    }

    if (attr.get_var().is_timesamples()) {
      PUSH_ERROR_AND_RETURN_F("Attribute `{}` is defined as `uniform` variability but TimeSample value is assigned.", prop_name);
    }

    if (auto tok = attr.get_value<value::token>()) {
      auto e = enum_handler(tok.value().str());
      if (e) {
        (*result) = e.value();
        return true;
      } else if (strict_allowedToken_check) {
        PUSH_ERROR_AND_RETURN_F("Attribute `{}`: `{}` is not an allowed token.", prop_name, tok.value().str());
      } else {
        PUSH_WARN_F("Attribute `{}`: `{}` is not an allowed token. Ignore it.", prop_name, tok.value().str());
        result->set_value_empty();
        return true;
      }
    } else {
      PUSH_ERROR_AND_RETURN_F("Internal error. Maybe type mismatch? Attribute `{}` must be type `token`, but got type `{}`", prop_name, attr.type_name());
    }


  } else {
    // uniform or TimeSamples
    if (attr.get_var().is_scalar()) {

      if (attr.is_blocked()) {
        result->set_blocked(true);
        return true;
      }

      if (auto tok = attr.get_value<value::token>()) {
        auto e = enum_handler(tok.value().str());
        if (e) {
          (*result) = e.value();
          return true;
        } else if (strict_allowedToken_check) {
          PUSH_ERROR_AND_RETURN_F("Attribute `{}`: `{}` is not an allowed token.", prop_name, tok.value().str());
        } else {
          PUSH_WARN_F("Attribute `{}`: `{}` is not an allowed token. Ignore it.", prop_name, tok.value().str());
          result->set_value_empty();
          return true;
        }
      } else {
        PUSH_ERROR_AND_RETURN_F("Internal error. Maybe type mismatch? Attribute `{}` must be type `token`, but got type `{}`", prop_name, attr.type_name());
      }
    } else if (attr.get_var().is_timesamples()) {
      PUSH_ERROR_AND_RETURN_F("Attribute `{}` is uniform variability, but TimeSampled value is authored.",
 prop_name);

    } else {
      PUSH_ERROR_AND_RETURN_F("Internal error. Attribute `{}` is invalid", prop_name);
    }

  }

  return false;
}

// Animatable enum tokens
template<typename T, typename EnumTy>
bool ParseTimeSampledEnumProperty(
  const std::string &prop_name,
  bool strict_allowedToken_check,
  EnumHandlerFun<EnumTy> enum_handler,
  const Attribute &attr,
  TypedAttributeWithFallback<Animatable<T>> *result,
  std::string *warn = nullptr,
  std::string *err = nullptr)
{

  if (!result) {
    PUSH_ERROR_AND_RETURN("[Internal error] `result` arg is nullptr.");
  }

  if (attr.is_connection()) {
    PUSH_ERROR_AND_RETURN_F("Attribute connection is not supported in TinyUSDZ for built-in 'enum' token attribute: {}", prop_name);
  }


  if (attr.variability() == Variability::Uniform) {
    // scalar

    if (attr.is_blocked()) {
      result->set_blocked(true);
      return true;
    }

    if (attr.get_var().is_timesamples()) {
      PUSH_ERROR_AND_RETURN_F("Attribute `{}` is defined as `uniform` variability but TimeSample value is assigned.", prop_name);
    }

    if (auto tok = attr.get_value<value::token>()) {
      auto e = enum_handler(tok.value().str());
      if (e) {
        (*result) = e.value();
        return true;
      } else if (strict_allowedToken_check) {
        PUSH_ERROR_AND_RETURN_F("Attribute `{}`: `{}` is not an allowed token.", prop_name, tok.value().str());
      } else {
        PUSH_WARN_F("Attribute `{}`: `{}` is not an allowed token. Ignore it.", prop_name, tok.value().str());
        result->set_value_empty();
        return true;
      }
    } else {
      PUSH_ERROR_AND_RETURN_F("Internal error. Maybe type mismatch? Attribute `{}` must be type `token`, but got type `{}`", prop_name, attr.type_name());
    }


  } else {
    // default and/or TimeSamples
    bool has_default{false};
    bool has_timesamples{false};

    Animatable<T> animatable_value;

    if (attr.is_blocked()) {
      result->set_blocked(true);
      has_default = true;
      //return true;
    }

    if (attr.get_var().has_default()) {
      DCOUT("has default.");

      if (auto tok = attr.get_value<value::token>()) {
        auto e = enum_handler(tok.value().str());
        if (e) {
          animatable_value.set_default(e.value());
          has_default = true;
          //return true;

        } else if (strict_allowedToken_check) {
          PUSH_ERROR_AND_RETURN_F("Attribute `{}`: `{}` is not an allowed token.", prop_name, tok.value().str());
        } else {
          PUSH_WARN_F("Attribute `{}`: `{}` is not an allowed token. Ignore it.", prop_name, tok.value().str());
          //result->set_value_empty();
          //return true;
        }
      } else {
        PUSH_ERROR_AND_RETURN_F("Internal error. Maybe type mismatch? Attribute `{}` must be type `token`, but got type `{}`", prop_name, attr.type_name());
      }
    }

    if (attr.get_var().has_timesamples()) {
      DCOUT("has timesamples.");
      size_t n = attr.get_var().num_timesamples();

      for (size_t i = 0; i < n; i++) {

        double sample_time{value::TimeCode::Default()};

        if (auto pv = attr.get_var().get_ts_time(i)) {
          sample_time = pv.value();
        } else {
          // This should not happen.
          PUSH_ERROR_AND_RETURN_F("Internal error. Failed to get timecode for `{}`", prop_name);
        }

        if (auto pv = attr.get_var().is_ts_value_blocked(i)) {
          if (pv.value() == true) {
            animatable_value.add_blocked_sample(sample_time);
            continue;
          }
        } else {
          // This should not happen.
          PUSH_ERROR_AND_RETURN_F("Internal error. Failed to get valueblock info for `{}`", prop_name);
        }

        if (auto tok = attr.get_var().get_ts_value<value::token>(i)) {
          auto e = enum_handler(tok.value().str());
          if (e) {
            animatable_value.add_sample(sample_time, e.value());
          } else if (strict_allowedToken_check) {
            PUSH_ERROR_AND_RETURN_F("Attribute `{}`: `{}` is not an allowed token.", prop_name, tok.value().str());
          } else {
            PUSH_WARN_F("Attribute `{}`: `{}` at {}'th timesample is not an allowed token. Ignore it.", prop_name, i, tok.value().str());
            continue;
          }
        } else {
          PUSH_ERROR_AND_RETURN_F("Internal error. Maybe type mismatch? Attribute `{}`'s {}'th timesample must be type `token`, but got type `{}`", prop_name, i, attr.type_name());
        }
      }

      has_timesamples = true;
      //return true;

    }

    if (has_default || has_timesamples) {
      result->set_value(animatable_value);
    }

    return true;

  }

  return false;
}
#endif


#if 0
// TODO: TimeSamples
#define PARSE_ENUM_PROPETY(__table, __prop, __name, __enum_handler, __klass, \
                           __target, __strict_check) {                          \
  if (__prop.first == __name) {                                              \
    if (__table.count(__name)) { continue; } \
    if ((__prop.second.value_type_name() == value::TypeTraits<value::token>::type_name()) && __prop.second.is_attribute() && __prop.second.is_empty()) { \
      PUSH_WARN("No value assigned to `" << __name << "` token attribute. Set default token value."); \
      /* TODO: attr meta __target.meta = attr.meta;  */                    \
      __table.insert(__name);                                              \
    } else { \
      const Attribute &attr = __prop.second.get_attribute();                           \
      if (auto tok = attr.get_value<value::token>()) {                     \
        auto e = __enum_handler(tok.value().str());                            \
        if (e) {                                                               \
          __target = e.value();                                                \
          /* TODO: attr meta __target.meta = attr.meta;  */                    \
          __table.insert(__name);                                              \
        } else if (__strict_check) {                                            \
          PUSH_ERROR_AND_RETURN("(" << value::TypeTraits<__klass>::type_name()  \
                                    << ") " << e.error());                     \
        } else { \
          PUSH_WARN("`" << tok.value().str() << "` is not allowed token for `" << __name << "`. Set to default token value."); \
          /* TODO: attr meta __target.meta = attr.meta;  */                    \
          __table.insert(__name);                                              \
        } \
      } else {                                                                 \
        PUSH_ERROR_AND_RETURN("(" << value::TypeTraits<__klass>::type_name()    \
                                  << ") Property type mismatch. " << __name    \
                                  << " must be type `token`, but got `"        \
                                  << attr.type_name() << "`.");            \
      }                                                                        \
    } \
  } \
}
#else
#define PARSE_UNIFORM_ENUM_PROPERTY(__table, __prop, __name, __enum_ty, __enum_handler, __klass, \
                           __target, __strict_check) {                          \
  if (__prop.first == __name) {                                              \
    if (__table.count(__name)) { continue; } \
    if ((__prop.second.value_type_name() == value::TypeTraits<value::token>::type_name()) && __prop.second.is_attribute() && __prop.second.is_empty()) { \
      PUSH_WARN("No value assigned to `" << __name << "` token attribute. Set default token value."); \
      __target.metas() = __prop.second.get_attribute().metas();                    \
      __table.insert(__name);                                              \
      continue; \
    } else { \
      const Attribute &attr = __prop.second.get_attribute();                           \
      std::function<nonstd::expected<__enum_ty, std::string>(const std::string &)> fun = __enum_handler; \
      if (!ParseUniformEnumProperty(__name, __strict_check, fun, attr, &__target, warn, err)) { \
        return false; \
      } \
      __target.metas() = attr.metas(); \
      __table.insert(__name);                                              \
      continue; \
    } \
  } \
}

#define PARSE_TIMESAMPLED_ENUM_PROPERTY(__table, __prop, __name, __enum_ty, __enum_handler, __klass, \
                           __target, __strict_check) {                          \
  if (__prop.first == __name) {                                              \
    if (__table.count(__name)) { continue; } \
    if ((__prop.second.value_type_name() == value::TypeTraits<value::token>::type_name()) && __prop.second.is_attribute() && __prop.second.is_empty()) { \
      PUSH_WARN("No value assigned to `" << __name << "` token attribute. Set default token value."); \
      const Attribute &attr = __prop.second.get_attribute();                           \
      __target.metas() = attr.metas();                    \
      __table.insert(__name);                                              \
      continue; \
    } else { \
      const Attribute &attr = __prop.second.get_attribute();                           \
      std::function<nonstd::expected<__enum_ty, std::string>(const std::string &)> fun = __enum_handler; \
      if (!ParseTimeSampledEnumProperty(__name, __strict_check, fun, attr, &__target, warn, err)) { \
        return false; \
      } \
      __target.metas() = attr.metas(); \
     __table.insert(__name);                                              \
     continue; \
    } \
  } \
}
#endif


// Add custom property(including property with "primvars" prefix)
// Please call this macro after listing up all predefined property with
// `PARSE_PROPERTY` and `PARSE_***_ENUM_PROPERTY`
#define ADD_PROPERTY(__table, __prop, __klass, __dst) {        \
  /* Check if the property name is a predefined property */  \
  if (!__table.count(__prop.first)) {                        \
    DCOUT("custom property added: name = " << __prop.first); \
    __dst[__prop.first] = __prop.second;                     \
    __table.insert(__prop.first);                            \
  } \
 }

// This code path should not be reached though.
#define PARSE_PROPERTY_END_MAKE_ERROR(__table, __prop) {                     \
  if (!__table.count(__prop.first)) {                              \
    PUSH_ERROR_AND_RETURN("Unsupported/unimplemented property: " + \
                          __prop.first);                           \
  } \
 }

// This code path should not be reached though.
#define PARSE_PROPERTY_END_MAKE_WARN(__table, __prop) { \
  if (!__table.count(__prop.first)) { \
    PUSH_WARN("Unsupported/unimplemented property: " + __prop.first); \
   } \
 }

bool ReconstructXformOpsFromProperties(
  const Specifier &spec,
  std::set<std::string> &table, /* inout */
  const std::map<std::string, Property> &properties,
  std::vector<XformOp> *xformOps,
  std::string *err)
{

  if (spec == Specifier::Class) {
    // Do not materialize xformOps here.
    return true;
  }


  constexpr auto kTranslate = "xformOp:translate";
  constexpr auto kTransform = "xformOp:transform";
  constexpr auto kScale = "xformOp:scale";
  constexpr auto kRotateX = "xformOp:rotateX";
  constexpr auto kRotateY = "xformOp:rotateY";
  constexpr auto kRotateZ = "xformOp:rotateZ";
  constexpr auto kRotateXYZ = "xformOp:rotateXYZ";
  constexpr auto kRotateXZY = "xformOp:rotateXZY";
  constexpr auto kRotateYXZ = "xformOp:rotateYXZ";
  constexpr auto kRotateYZX = "xformOp:rotateYZX";
  constexpr auto kRotateZXY = "xformOp:rotateZXY";
  constexpr auto kRotateZYX = "xformOp:rotateZYX";
  constexpr auto kOrient = "xformOp:orient";

  // false : no prefix found.
  // true : return suffix(first namespace ':' is ommited.).
  // - "" for prefix only "xformOp:translate"
  // - "blender:pivot" for "xformOp:translate:blender:pivot"
  auto SplitXformOpToken =
      [](const std::string &s,
         const std::string &prefix) -> nonstd::optional<std::string> {
    if (startsWith(s, prefix)) {
      if (s.compare(prefix) == 0) {
        // prefix only.
        return std::string();  // empty suffix
      } else {
        std::string suffix = removePrefix(s, prefix);
        DCOUT("suffix = " << suffix);
        if (suffix.length() == 1) {  // maybe namespace only.
          return nonstd::nullopt;
        }

        // remove namespace ':'
        if (suffix[0] == ':') {
          // ok
          suffix.erase(0, 1);
        } else {
          return nonstd::nullopt;
        }

        return std::move(suffix);
      }
    }

    return nonstd::nullopt;
  };

  // Lookup xform values from `xformOpOrder`
  // TODO: TimeSamples, Connection
  if (properties.count("xformOpOrder")) {
    // array of string
    auto prop = properties.at("xformOpOrder");

    if (prop.is_relationship()) {
      PUSH_ERROR_AND_RETURN("Relationship for `xformOpOrder` is not supported.");
    } else if (auto pv =
                   prop.get_attribute().get_value<std::vector<value::token>>()) {

      // 'uniform' check
      if (prop.get_attribute().variability() != Variability::Uniform) {
        PUSH_ERROR_AND_RETURN("`xformOpOrder` must have `uniform` variability.");
      }

      for (size_t i = 0; i < pv.value().size(); i++) {
        const auto &item = pv.value()[i];

        XformOp op;

        std::string tok = item.str();
        DCOUT("xformOp token = " << tok);

        if (startsWith(tok, "!resetXformStack!")) {
          if (tok.compare("!resetXformStack!") != 0) {
            PUSH_ERROR_AND_RETURN(
                "`!resetXformStack!` must be defined solely(not to be a prefix "
                "to \"xformOp:*\")");
          }

          if (i != 0) {
            PUSH_ERROR_AND_RETURN(
                "`!resetXformStack!` must appear at the first element of "
                "xformOpOrder list.");
          }

          op.op_type = XformOp::OpType::ResetXformStack;
          xformOps->emplace_back(op);

          // skip looking up property
          continue;
        }

        if (startsWith(tok, "!invert!")) {
          DCOUT("invert!");
          op.inverted = true;
          tok = removePrefix(tok, "!invert!");
          DCOUT("tok = " << tok);
        }

        auto it = properties.find(tok);
        if (it == properties.end()) {
          PUSH_ERROR_AND_RETURN("Property `" + tok + "` not found.");
        }
        if (it->second.is_attribute_connection()) {
          PUSH_ERROR_AND_RETURN(
              "Connection(.connect) for xformOp attribute is not yet supported: "
              "`" +
              tok + "`");
        }
        const Attribute &attr = it->second.get_attribute();

        // Check `xformOp` namespace
        if (auto xfm = SplitXformOpToken(tok, kTransform)) {
          op.op_type = XformOp::OpType::Transform;
          op.suffix = xfm.value();  // may contain nested namespaces

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::matrix4d>::type_id()) {
                value::matrix4d dummy{value::matrix4d::identity()};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:transform` must be type `matrix4d`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::matrix4d>()) {
              op.set_value(pvd.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:transform` must be type `matrix4d`, but got type `" +
                  attr.type_name() + "`.");
            }
          }

        } else if (auto tx = SplitXformOpToken(tok, kTranslate)) {
          op.op_type = XformOp::OpType::Translate;
          op.suffix = tx.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::double3>::type_id()) {
                value::double3 dummy{0.0, 0.0, 0.0};
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<value::float3>::type_id()) {
                value::float3 dummy{0.0f, 0.0f, 0.0f};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:translate` must be type `double3` or `float3`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::double3>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::float3>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:translate` must be type `double3` or `float3`, but "
                  "got type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto scale = SplitXformOpToken(tok, kScale)) {
          op.op_type = XformOp::OpType::Scale;
          op.suffix = scale.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::double3>::type_id()) {
                value::double3 dummy{0.0, 0.0, 0.0};
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<value::float3>::type_id()) {
                value::float3 dummy{0.0f, 0.0f, 0.0f};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:scale` must be type `double3` or `float3`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::double3>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::float3>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:scale` must be type `double3` or `float3`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotX = SplitXformOpToken(tok, kRotateX)) {
          op.op_type = XformOp::OpType::RotateX;
          op.suffix = rotX.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<double>::type_id()) {
                double dummy(0.0);
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<float>::type_id()) {
                float dummy(0.0f);
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateX` must be type `double` or `float`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<double>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<float>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateX` must be type `double` or `float`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotY = SplitXformOpToken(tok, kRotateY)) {
          op.op_type = XformOp::OpType::RotateY;
          op.suffix = rotY.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<double>::type_id()) {
                double dummy(0.0);
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<float>::type_id()) {
                float dummy(0.0f);
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateY` must be type `double` or `float`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<double>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<float>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateY` must be type `double` or `float`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotZ = SplitXformOpToken(tok, kRotateZ)) {
          op.op_type = XformOp::OpType::RotateZ;
          op.suffix = rotZ.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<double>::type_id()) {
                double dummy(0.0);
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<float>::type_id()) {
                float dummy(0.0f);
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateZ` must be type `double` or `float`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<double>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<float>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateZ` must be type `double` or `float`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotateXYZ = SplitXformOpToken(tok, kRotateXYZ)) {
          op.op_type = XformOp::OpType::RotateXYZ;
          op.suffix = rotateXYZ.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::double3>::type_id()) {
                value::double3 dummy{0.0, 0.0, 0.0};
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<value::float3>::type_id()) {
                value::float3 dummy{0.0f, 0.0f, 0.0f};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateXYZ` must be type `double3` or `float3`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::double3>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::float3>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateXYZ` must be type `double3` or `float3`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotateXZY = SplitXformOpToken(tok, kRotateXZY)) {
          op.op_type = XformOp::OpType::RotateXZY;
          op.suffix = rotateXZY.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::double3>::type_id()) {
                value::double3 dummy{0.0, 0.0, 0.0};
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<value::float3>::type_id()) {
                value::float3 dummy{0.0f, 0.0f, 0.0f};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateXZY` must be type `double3` or `float3`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::double3>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::float3>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateXZY` must be type `double3` or `float3`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotateYXZ = SplitXformOpToken(tok, kRotateYXZ)) {
          op.op_type = XformOp::OpType::RotateYXZ;
          op.suffix = rotateYXZ.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::double3>::type_id()) {
                value::double3 dummy{0.0, 0.0, 0.0};
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<value::float3>::type_id()) {
                value::float3 dummy{0.0f, 0.0f, 0.0f};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateYXZ` must be type `double3` or `float3`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::double3>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::float3>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateYXZ` must be type `double3` or `float3`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotateYZX = SplitXformOpToken(tok, kRotateYZX)) {
          op.op_type = XformOp::OpType::RotateYZX;
          op.suffix = rotateYZX.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::double3>::type_id()) {
                value::double3 dummy{0.0, 0.0, 0.0};
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<value::float3>::type_id()) {
                value::float3 dummy{0.0f, 0.0f, 0.0f};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateYZX` must be type `double3` or `float3`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::double3>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::float3>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateYZX` must be type `double3` or `float3`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotateZXY = SplitXformOpToken(tok, kRotateZXY)) {
          op.op_type = XformOp::OpType::RotateZXY;
          op.suffix = rotateZXY.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::double3>::type_id()) {
                value::double3 dummy{0.0, 0.0, 0.0};
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<value::float3>::type_id()) {
                value::float3 dummy{0.0f, 0.0f, 0.0f};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateZXY` must be type `double3` or `float3`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::double3>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::float3>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateZXY` must be type `double3` or `float3`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto rotateZYX = SplitXformOpToken(tok, kRotateZYX)) {
          op.op_type = XformOp::OpType::RotateZYX;
          op.suffix = rotateZYX.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::double3>::type_id()) {
                value::double3 dummy{0.0, 0.0, 0.0};
                op.set_value(dummy);
              } else if (attr.type_id() == value::TypeTraits<value::float3>::type_id()) {
                value::float3 dummy{0.0f, 0.0f, 0.0f};
                op.set_value(dummy);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:rotateZYX` must be type `double3` or `float3`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::double3>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::float3>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:rotateZYX` must be type `double3` or `float3`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else if (auto orient = SplitXformOpToken(tok, kOrient)) {
          op.op_type = XformOp::OpType::Orient;
          op.suffix = orient.value();

          if (attr.get_var().has_timesamples()) {
            op.set_timesamples(attr.get_var().ts_raw());
          }

          if (attr.get_var().has_default()) {
            if (attr.has_blocked()) {
              // Set dummy value for `op.get_value_type_id/op.get_value_type_name'
              if (attr.type_id() == value::TypeTraits<value::quatf>::type_id()) {
                value::quatf q;
                q.real = 1.0f;
                q.imag = {0.0f, 0.0f, 0.0f};
                op.set_value(q);
              } else if (attr.type_id() == value::TypeTraits<value::quatd>::type_id()) {
                value::quatd q;
                q.real = 1.0;
                q.imag = {0.0, 0.0, 0.0};
                op.set_value(q);
              } else {
                PUSH_ERROR_AND_RETURN(
                    "`xformOp:orient` must be type `quatf` or `quatd`, but got "
                    "type `" +
                    attr.type_name() + "`.");
              }
              op.set_blocked(true);
            } else if (auto pvd = attr.get_value<value::quatf>()) {
              op.set_value(pvd.value());
            } else if (auto pvf = attr.get_value<value::quatd>()) {
              op.set_value(pvf.value());
            } else {
              PUSH_ERROR_AND_RETURN(
                  "`xformOp:orient` must be type `quatf` or `quatd`, but got "
                  "type `" +
                  attr.type_name() + "`.");
            }
          }
        } else {
          PUSH_ERROR_AND_RETURN(
              "token for xformOpOrder must have namespace `xformOp:***`, or .");
        }

        xformOps->emplace_back(op);
        table.insert(tok);
      }

    } else {
      PUSH_ERROR_AND_RETURN(
          "`xformOpOrder` must be type `token[]` but got type `"
          << prop.get_attribute().type_name() << "`.");
    }
  }

  table.insert("xformOpOrder");
  return true;
}

namespace {

bool ReconstructMaterialBindingProperties(
  std::set<std::string> &table, /* inout */
  const std::map<std::string, Property> &properties,
  MaterialBinding *mb, /* inout */
  std::string *err)
{

  if (!mb) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_SINGLE_TARGET_PATH_RELATION(table, prop, kMaterialBinding, mb->materialBinding)
    PARSE_SINGLE_TARGET_PATH_RELATION(table, prop, kMaterialBindingPreview, mb->materialBindingPreview)
    PARSE_SINGLE_TARGET_PATH_RELATION(table, prop, kMaterialBindingPreview, mb->materialBindingFull)
    // material:binding:collection
    if (prop.first == kMaterialBindingCollection) {

      if (table.count(prop.first)) {
         continue;
      }

      if (!prop.second.is_relationship()) {
        PUSH_ERROR_AND_RETURN(fmt::format("`{}` must be a Relationship", prop.first));
      }

      const Relationship &rel = prop.second.get_relationship();

      mb->set_materialBindingCollection(value::token(""), value::token(""), rel);

      table.insert(prop.first);
      continue;
    }
    // material:binding:collection[:PURPOSE]:NAME
    if (startsWith(prop.first, kMaterialBindingCollection + std::string(":"))) {

      if (table.count(prop.first)) {
         continue;
      }

      if (!prop.second.is_relationship()) {
        PUSH_ERROR_AND_RETURN(fmt::format("`{}` must be a Relationship", prop.first));
      }

      std::string collection_name = removePrefix(prop.first, kMaterialBindingCollection + std::string(":"));
      if (collection_name.empty()) {
        PUSH_ERROR_AND_RETURN("empty NAME is not allowed for 'mateirial:binding:collection'");
      }
      std::vector<std::string> names = split(collection_name, ":");
      if (names.size() > 2) {
        PUSH_ERROR_AND_RETURN("3 or more namespaces is not allowed for 'mateirial:binding:collection'");
      }
      value::token mat_purpose; // empty = all-purpose
      if (names.size() == 1) {
        collection_name = names[0];
      } else {
        mat_purpose = value::token(names[0]);
        collection_name = names[1];
      }

      const Relationship &rel = prop.second.get_relationship();

      mb->set_materialBindingCollection(value::token(collection_name), mat_purpose, rel);

      table.insert(prop.first);
      continue;
    }
    // material:binding:PURPOSE
    if (startsWith(prop.first, kMaterialBinding + std::string(":"))) {

      if (table.count(prop.first)) {
         continue;
      }

      if (!prop.second.is_relationship()) {
        PUSH_ERROR_AND_RETURN(fmt::format("`{}` must be a Relationship", prop.first));
      }

      std::string purpose_name = removePrefix(prop.first, kMaterialBinding + std::string(":"));
      if (purpose_name.empty()) {
        PUSH_ERROR_AND_RETURN("empty PURPOSE is not allowed for 'mateirial:binding:'");
      }
      std::vector<std::string> names = split(purpose_name, ":");
      if (names.size() > 1) {
        PUSH_ERROR_AND_RETURN(fmt::format("PURPOSE `{}` must not have nested namespaces for 'mateirial:binding'", purpose_name));
      }
      value::token mat_purpose = value::token(names[0]);

      const Relationship &rel = prop.second.get_relationship();

      mb->set_materialBinding(rel, mat_purpose);

      table.insert(prop.first);
      continue;
    }
  }

  return true;
}

bool ReconstructCollectionProperties(
  std::set<std::string> &table, /* inout */
  const std::map<std::string, Property> &properties,
  Collection *coll, /* inout */
  std::string *warn,
  std::string *err,
  bool strict_allowedToken_check)
{
  constexpr auto kCollectionPrefix = "collection:";

  std::function<nonstd::expected<CollectionInstance::ExpansionRule, std::string>(const std::string &)> ExpansionRuleEnumHandler = [](const std::string &tok) {
  //auto ExpansionRuleEnumHandler = [](const std::string &tok) {
    using EnumTy = std::pair<CollectionInstance::ExpansionRule, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(CollectionInstance::ExpansionRule::ExplicitOnly, kExplicitOnly),
        std::make_pair(CollectionInstance::ExpansionRule::ExpandPrims, kExpandPrims),
        std::make_pair(CollectionInstance::ExpansionRule::ExpandPrimsAndProperties, kExpandPrimsAndProperties),
    };
    return EnumHandler<CollectionInstance::ExpansionRule>("expansionRule", tok, enums);
  };

  if (!coll) {
    return false;
  }

  for (const auto &prop : properties) {
    if (startsWith(prop.first, kCollectionPrefix)) {
      if (table.count(prop.first)) {
         continue;
      }

      std::string suffix = removePrefix(prop.first, kCollectionPrefix);
      std::vector<std::string> names = split(suffix, ":");
      if (names.size() != 2) {
        PUSH_ERROR_AND_RETURN(fmt::format("Invalid collection property name. Must be 'collection:INSTANCE_NAME:<prop_name>' but got '{}'",  prop.first));
      }
      if (names[0].empty()) {
        PUSH_ERROR_AND_RETURN("INSTANCE_NAME is empty for collection property name");
      }
      if (names[1].empty()) {
        PUSH_ERROR_AND_RETURN("Collection property name is empty");
      }

      std::string instance_name = names[0];

      if (names[1] == "includes") {

        if (!prop.second.is_relationship()) {
          PUSH_ERROR_AND_RETURN(fmt::format("`{}` must be a Relationship", prop.first));
        }

        CollectionInstance &coll_instance = coll->get_or_add_instance(instance_name);
        coll_instance.includes = prop.second.get_relationship();
        table.insert(prop.first);

      } else if (names[1] == "expansionRule") {

        TypedAttributeWithFallback<CollectionInstance::ExpansionRule> r{CollectionInstance::ExpansionRule::ExpandPrims};

        PARSE_UNIFORM_ENUM_PROPERTY(table, prop, prop.first, CollectionInstance::ExpansionRule, ExpansionRuleEnumHandler, CollectionInstance,
                       r, strict_allowedToken_check)

        if (table.count(prop.first)) {
          CollectionInstance &coll_instance = coll->get_or_add_instance(instance_name);
          coll_instance.expansionRule = r.get_value();
        }
      } else if (names[1] == "includeRoot") {

        TypedAttributeWithFallback<Animatable<bool>> includeRoot{false};
        PARSE_TYPED_ATTRIBUTE_NOCONTINUE(table, prop, prop.first, CollectionInstance, includeRoot)

        if (table.count(prop.first)) {
          CollectionInstance &coll_instance = coll->get_or_add_instance(instance_name);
          coll_instance.includeRoot = includeRoot;
        }
      } else if (names[1] == "excludes") {

        if (!prop.second.is_relationship()) {
          PUSH_ERROR_AND_RETURN(fmt::format("`{}` must be a Relationship", prop.first));
        }

        CollectionInstance &coll_instance = coll->get_or_add_instance(instance_name);
        coll_instance.excludes = prop.second.get_relationship();
        table.insert(prop.first);

      }
    }
  }

  return true;
}
// xformOps and built-in props
bool ReconstructGPrimProperties(
  const Specifier &spec,
  std::set<std::string> &table, /* inout */
  const std::map<std::string, Property> &properties,
  GPrim *gprim, /* inout */
  std::string *warn,
  std::string *err,
  bool strict_allowedToken_check)
{

  (void)warn;
  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &gprim->xformOps, err)) {
    return false;
  }

  if (!prim::ReconstructMaterialBindingProperties(table, properties, gprim, err)) {
    return false;
  }

  if (!prim::ReconstructCollectionProperties(
    table, properties, gprim, warn, err, strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_SINGLE_TARGET_PATH_RELATION(table, prop, kProxyPrim, gprim->proxyPrim)
    PARSE_TYPED_ATTRIBUTE(table, prop, "doubleSided", GPrim, gprim->doubleSided)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, GPrim,
                   gprim->visibility, strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "purpose", Purpose, PurposeEnumHandler, GPrim,
                       gprim->purpose, strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "orientation", Orientation, OrientationEnumHandler, GPrim,
                       gprim->orientation, strict_allowedToken_check)
    PARSE_EXTENT_ATTRIBUTE(table, prop, "extent", GPrim, gprim->extent)
  }

  return true;
}

} // namespace local


template <>
bool ReconstructPrim<Xform>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    Xform *xform,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)options;
  (void)references;

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, xform, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    ADD_PROPERTY(table, prop, Xform, xform->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<Model>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    Model *model,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {
  DCOUT("Model ");
  (void)spec;
  (void)references;
  (void)model;
  (void)err;
  (void)options;

  std::set<std::string> table;
  for (const auto &prop : properties) {
    ADD_PROPERTY(table, prop, Model, model->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<Scope>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    Scope *scope,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {
  // `Scope` is just a namespace in scene graph(no node xform)

  (void)spec;
  (void)references;
  (void)scope;
  (void)err;
  (void)options;

  DCOUT("Scope");
  std::set<std::string> table;
  for (const auto &prop : properties) {
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, Scope,
                   scope->visibility, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, Scope, scope->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<SkelRoot>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    SkelRoot *root,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)references;
  (void)options;

  std::set<std::string> table;
  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &root->xformOps, err)) {
    return false;
  }

  // SkelRoot is something like a grouping node, having 1 Skeleton and possibly?
  // multiple Prim hierarchy containing GeomMesh.
  // No specific properties for SkelRoot(AFAIK)

  // custom props only
  for (const auto &prop : properties) {
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, SkelRoot,
                   root->visibility, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, kPurpose, Purpose, PurposeEnumHandler, SkelRoot,
                       root->purpose, options.strict_allowedToken_check)
    PARSE_EXTENT_ATTRIBUTE(table, prop, kExtent, SkelRoot, root->extent)
    ADD_PROPERTY(table, prop, SkelRoot, root->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<Skeleton>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    Skeleton *skel,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)warn;
  (void)references;
  (void)options;

  std::set<std::string> table;
  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &skel->xformOps, err)) {
    return false;
  }

  for (auto &prop : properties) {

    // SkelBindingAPI
    if (prop.first == kSkelAnimationSource) {

      // Must be relation of type Path.
      if (prop.second.is_relationship() && prop.second.get_relationship().is_path()) {
        {
          const Relationship &rel = prop.second.get_relationship();
          if (rel.is_path()) {
            skel->animationSource = rel;
            table.insert(kSkelAnimationSource);
          } else {
            PUSH_ERROR_AND_RETURN("`" << kSkelAnimationSource << "` target must be Path.");
          }
        }
      } else {
        PUSH_ERROR_AND_RETURN(
            "`" << kSkelAnimationSource << "` must be a Relationship with Path target.");
      }
    }

    //

    PARSE_TYPED_ATTRIBUTE(table, prop, "bindTransforms", Skeleton, skel->bindTransforms)
    PARSE_TYPED_ATTRIBUTE(table, prop, "joints", Skeleton, skel->joints)
    PARSE_TYPED_ATTRIBUTE(table, prop, "jointNames", Skeleton, skel->jointNames)
    PARSE_TYPED_ATTRIBUTE(table, prop, "restTransforms", Skeleton, skel->restTransforms)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, Skeleton,
                   skel->visibility, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "purpose", Purpose, PurposeEnumHandler, Skeleton,
                       skel->purpose, options.strict_allowedToken_check)
    PARSE_EXTENT_ATTRIBUTE(table, prop, "extent", Skeleton, skel->extent)
    ADD_PROPERTY(table, prop, Skeleton, skel->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

#if 0 // TODO: bindTransforms & restTransforms check somewhere.
  // usdview and Houdini USD importer expects both `bindTransforms` and `restTransforms` are authored in USD
  if (!table.count("bindTransforms")) {
    // usdview and Houdini allow `bindTransforms` is not authord in USD, but it cannot compute skinning correctly without it,
    // so report an error in TinyUSDZ for a while.
    PUSH_ERROR_AND_RETURN_TAG(kTag, "`bindTransforms` is missing in Skeleton. Currently TinyUSDZ expects `bindTransforms` must exist in Skeleton.");
  }

  if (!table.count("restTransforms")) {
    // usdview and Houdini allow `restTransforms` is not authord in USD(usdview warns it), but it cannot compute skinning correctly without it,
    // (even SkelAnimation supplies trasnforms for all joints)
    // so report an error in TinyUSDZ for a while.
    PUSH_ERROR_AND_RETURN_TAG(kTag, "`restTransforms`(local joint matrices at rest state) is missing in Skeleton. Currently TinyUSDZ expects `restTransforms` must exist in Skeleton.");
  }

  // len(bindTransforms) must be equal to len(restTransforms)
  // TODO: Support connection
  {
    bool valid = false;
    if (auto bt = skel->bindTransforms.get_value()) {
      if (auto rt = skel->restTransforms.get_value()) {
        if (bt.value().size() == rt.value().size()) {
          // ok
          valid = true;
        }
      }
    }

    if (!valid) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Array length must be same for `bindTransforms` and `restTransforms`.");
    }
  }
#endif

  return true;
}

template <>
bool ReconstructPrim<SkelAnimation>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    SkelAnimation *skelanim,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)spec;
  (void)warn;
  (void)references;
  (void)options;
  std::set<std::string> table;
  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "joints", SkelAnimation, skelanim->joints)
    PARSE_TYPED_ATTRIBUTE(table, prop, "translations", SkelAnimation, skelanim->translations)
    PARSE_TYPED_ATTRIBUTE(table, prop, "rotations", SkelAnimation, skelanim->rotations)
    PARSE_TYPED_ATTRIBUTE(table, prop, "scales", SkelAnimation, skelanim->scales)
    PARSE_TYPED_ATTRIBUTE(table, prop, "blendShapes", SkelAnimation, skelanim->blendShapes)
    PARSE_TYPED_ATTRIBUTE(table, prop, "blendShapeWeights", SkelAnimation, skelanim->blendShapeWeights)
    ADD_PROPERTY(table, prop, Skeleton, skelanim->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<BlendShape>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    BlendShape *bs,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {
  (void)spec;
  (void)warn;
  (void)references;
  (void)options;

  DCOUT("Reconstruct BlendShape");

  constexpr auto kOffsets = "offsets";
  constexpr auto kNormalOffsets = "normalOffsets";
  constexpr auto kPointIndices = "pointIndices";

  std::set<std::string> table;
  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, kOffsets, BlendShape, bs->offsets)
    PARSE_TYPED_ATTRIBUTE(table, prop, kNormalOffsets, BlendShape, bs->normalOffsets)
    PARSE_TYPED_ATTRIBUTE(table, prop, kPointIndices, BlendShape, bs->pointIndices)
    ADD_PROPERTY(table, prop, Skeleton, bs->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

#if 0 // TODO: Check required properties exist in strict mode.
  // `offsets` and `normalOffsets` are required property
  if (!table.count(kOffsets)) {
    PUSH_ERROR_AND_RETURN("`offsets` property is missing. `uniform vector3f[] offsets` is a required property.");
  }
  if (!table.count(kNormalOffsets)) {
    PUSH_ERROR_AND_RETURN("`normalOffsets` property is missing. `uniform vector3f[] normalOffsets` is a required property.");
  }
#endif

  return true;
}

template <>
bool ReconstructPrim(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GPrim *gprim,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {
  (void)gprim;
  (void)err;

  (void)references;
  (void)properties;

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, gprim, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  return true;
}

template <>
bool ReconstructPrim(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomBasisCurves *curves,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {
  (void)references;
  (void)options;

  DCOUT("GeomBasisCurves");

  auto BasisHandler = [](const std::string &tok)
      -> nonstd::expected<GeomBasisCurves::Basis, std::string> {
    using EnumTy = std::pair<GeomBasisCurves::Basis, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(GeomBasisCurves::Basis::Bezier, "bezier"),
        std::make_pair(GeomBasisCurves::Basis::Bspline, "bspline"),
        std::make_pair(GeomBasisCurves::Basis::CatmullRom, "catmullRom"),
    };

    return EnumHandler<GeomBasisCurves::Basis>("basis", tok, enums);
  };

  auto TypeHandler = [](const std::string &tok)
      -> nonstd::expected<GeomBasisCurves::Type, std::string> {
    using EnumTy = std::pair<GeomBasisCurves::Type, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(GeomBasisCurves::Type::Cubic, "cubic"),
        std::make_pair(GeomBasisCurves::Type::Linear, "linear"),
    };

    return EnumHandler<GeomBasisCurves::Type>("type", tok, enums);
  };

  auto WrapHandler = [](const std::string &tok)
      -> nonstd::expected<GeomBasisCurves::Wrap, std::string> {
    using EnumTy = std::pair<GeomBasisCurves::Wrap, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(GeomBasisCurves::Wrap::Nonperiodic, "nonperiodic"),
        std::make_pair(GeomBasisCurves::Wrap::Periodic, "periodic"),
        std::make_pair(GeomBasisCurves::Wrap::Pinned, "periodic"),
    };

    return EnumHandler<GeomBasisCurves::Wrap>("wrap", tok, enums);
  };

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, curves, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "curveVertexCounts", GeomBasisCurves,
                         curves->curveVertexCounts)
    PARSE_TYPED_ATTRIBUTE(table, prop, "points", GeomBasisCurves, curves->points)
    PARSE_TYPED_ATTRIBUTE(table, prop, "velocities", GeomBasisCurves,
                          curves->velocities)
    PARSE_TYPED_ATTRIBUTE(table, prop, "normals", GeomBasisCurves,
                  curves->normals)
    PARSE_TYPED_ATTRIBUTE(table, prop, "accelerations", GeomBasisCurves,
                 curves->accelerations)
    PARSE_TYPED_ATTRIBUTE(table, prop, "widths", GeomBasisCurves, curves->widths)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "type", GeomBasisCurves::Type, TypeHandler, GeomBasisCurves,
                       curves->type, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "basis", GeomBasisCurves::Basis, BasisHandler, GeomBasisCurves,
                       curves->basis, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "wrap", GeomBasisCurves::Wrap, WrapHandler, GeomBasisCurves,
                       curves->wrap, options.strict_allowedToken_check)

    ADD_PROPERTY(table, prop, GeomBasisCurves, curves->props)

    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomNurbsCurves *curves,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {
  (void)references;
  (void)options;

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, curves, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "curveVertexCounts", GeomNurbsCurves,
                         curves->curveVertexCounts)
    PARSE_TYPED_ATTRIBUTE(table, prop, "points", GeomNurbsCurves, curves->points)
    PARSE_TYPED_ATTRIBUTE(table, prop, "velocities", GeomNurbsCurves,
                          curves->velocities)
    PARSE_TYPED_ATTRIBUTE(table, prop, "normals", GeomNurbsCurves,
                  curves->normals)
    PARSE_TYPED_ATTRIBUTE(table, prop, "accelerations", GeomNurbsCurves,
                 curves->accelerations)
    PARSE_TYPED_ATTRIBUTE(table, prop, "widths", GeomNurbsCurves, curves->widths)

    //
    PARSE_TYPED_ATTRIBUTE(table, prop, "order", GeomNurbsCurves, curves->order)
    PARSE_TYPED_ATTRIBUTE(table, prop, "knots", GeomNurbsCurves, curves->knots)
    PARSE_TYPED_ATTRIBUTE(table, prop, "ranges", GeomNurbsCurves, curves->ranges)
    PARSE_TYPED_ATTRIBUTE(table, prop, "pointWeights", GeomNurbsCurves, curves->pointWeights)

    ADD_PROPERTY(table, prop, GeomBasisCurves, curves->props)

    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<SphereLight>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    SphereLight *light,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)references;

  (void)options;
  std::set<std::string> table;

  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &light->xformOps, err)) {
    return false;
  }

  for (const auto &prop : properties) {
    // PARSE_PROPERTY(prop, "inputs:colorTemperature", light->colorTemperature)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:color", SphereLight, light->color)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:radius", SphereLight, light->radius)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:intensity", SphereLight,
                   light->intensity)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, SphereLight,
                   light->visibility, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, kPurpose, Purpose, PurposeEnumHandler, SphereLight,
                       light->purpose, options.strict_allowedToken_check)
    PARSE_EXTENT_ATTRIBUTE(table, prop, kExtent, SphereLight, light->extent)
    ADD_PROPERTY(table, prop, SphereLight, light->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<RectLight>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    RectLight *light,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)references;
  (void)options;

  std::set<std::string> table;

  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &light->xformOps, err)) {
    return false;
  }

  for (const auto &prop : properties) {
    // PARSE_PROPERTY(prop, "inputs:colorTemperature", light->colorTemperature)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:texture:file", UsdUVTexture, light->file)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:color", RectLight, light->color)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:height", RectLight, light->height)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:width", RectLight, light->width)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:intensity", RectLight,
                   light->intensity)
    PARSE_EXTENT_ATTRIBUTE(table, prop, kExtent, RectLight, light->extent)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, RectLight,
                   light->visibility, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, kPurpose, Purpose, PurposeEnumHandler, RectLight,
                       light->purpose, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, SphereLight, light->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<DiskLight>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    DiskLight *light,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)references;
  (void)options;

  std::set<std::string> table;

  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &light->xformOps, err)) {
    return false;
  }

  for (const auto &prop : properties) {
    // PARSE_PROPERTY(prop, "inputs:colorTemperature", light->colorTemperature)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:radius", DiskLight, light->radius)
    PARSE_EXTENT_ATTRIBUTE(table, prop, kExtent, DiskLight, light->extent)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, DiskLight,
                       light->visibility, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, kPurpose, Purpose, PurposeEnumHandler, DiskLight,
                       light->purpose, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, DiskLight, light->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<CylinderLight>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    CylinderLight *light,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)references;
  (void)options;

  std::set<std::string> table;

  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &light->xformOps, err)) {
    return false;
  }

  for (const auto &prop : properties) {
    // PARSE_PROPERTY(prop, "inputs:colorTemperature", light->colorTemperature)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:length", CylinderLight, light->length)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:radius", CylinderLight, light->radius)
    PARSE_EXTENT_ATTRIBUTE(table, prop, kExtent, CylinderLight, light->extent)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, CylindrLight,
                   light->visibility, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, kPurpose, Purpose, PurposeEnumHandler, CylinderLight,
                       light->purpose, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, SphereLight, light->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<DistantLight>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    DistantLight *light,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)references;
  (void)options;

  std::set<std::string> table;

  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &light->xformOps, err)) {
    return false;
  }

  for (const auto &prop : properties) {
    // PARSE_PROPERTY(prop, "inputs:colorTemperature", light->colorTemperature)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:angle", DistantLight, light->angle)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, kPurpose, Purpose, PurposeEnumHandler, DistantLight,
                       light->purpose, options.strict_allowedToken_check)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, DistantLight,
                   light->visibility, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, SphereLight, light->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<DomeLight>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    DomeLight *light,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)references;
  (void)options;

  std::set<std::string> table;

  if (!prim::ReconstructXformOpsFromProperties(spec, table, properties, &light->xformOps, err)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "guideRadius", DomeLight, light->guideRadius)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:diffuse", DomeLight, light->diffuse)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:specular", DomeLight,
                   light->specular)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:colorTemperature", DomeLight,
                   light->colorTemperature)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:color", DomeLight, light->color)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:intensity", DomeLight,
                   light->intensity)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, kVisibility, Visibility, VisibilityEnumHandler, DomeLight,
                   light->visibility, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, kPurpose, Purpose, PurposeEnumHandler, DomeLight,
                       light->purpose, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, DomeLight, light->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  DCOUT("Implement DomeLight");
  return true;
}

template <>
bool ReconstructPrim<GeomSphere>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomSphere *sphere,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)warn;
  (void)references;
  (void)options;

  DCOUT("Reconstruct Sphere.");

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, sphere, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "radius", GeomSphere, sphere->radius)
    ADD_PROPERTY(table, prop, GeomSphere, sphere->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<GeomPoints>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomPoints *points,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)warn;
  (void)references;
  (void)options;

  DCOUT("Reconstruct Points.");

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, points, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    DCOUT("prop: " << prop.first);
    PARSE_TYPED_ATTRIBUTE(table, prop, "points", GeomPoints, points->points)
    PARSE_TYPED_ATTRIBUTE(table, prop, "normals", GeomPoints, points->normals)
    PARSE_TYPED_ATTRIBUTE(table, prop, "widths", GeomPoints, points->widths)
    PARSE_TYPED_ATTRIBUTE(table, prop, "ids", GeomPoints, points->ids)
    PARSE_TYPED_ATTRIBUTE(table, prop, "velocities", GeomPoints, points->velocities)
    PARSE_TYPED_ATTRIBUTE(table, prop, "accelerations", GeomPoints, points->accelerations)
    ADD_PROPERTY(table, prop, GeomSphere, points->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<GeomCone>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomCone *cone,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)warn;
  (void)references;
  (void)options;

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, cone, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    DCOUT("prop: " << prop.first);
    PARSE_TYPED_ATTRIBUTE(table, prop, "radius", GeomCone, cone->radius)
    PARSE_TYPED_ATTRIBUTE(table, prop, "height", GeomCone, cone->height)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "axis", Axis, AxisEnumHandler, GeomCone, cone->axis, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, GeomCone, cone->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<GeomCylinder>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomCylinder *cylinder,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)warn;
  (void)references;
  (void)options;

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, cylinder, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    DCOUT("prop: " << prop.first);
    PARSE_TYPED_ATTRIBUTE(table, prop, "radius", GeomCylinder,
                         cylinder->radius)
    PARSE_TYPED_ATTRIBUTE(table, prop, "height", GeomCylinder,
                         cylinder->height)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "axis", Axis, AxisEnumHandler, GeomCylinder, cylinder->axis, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, GeomCylinder, cylinder->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<GeomCapsule>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomCapsule *capsule,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)warn;
  (void)references;
  (void)options;

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, capsule, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "radius", GeomCapsule, capsule->radius)
    PARSE_TYPED_ATTRIBUTE(table, prop, "height", GeomCapsule, capsule->height)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "axis", Axis, AxisEnumHandler, GeomCapsule, capsule->axis, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, GeomCapsule, capsule->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<GeomCube>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomCube *cube,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)warn;
  (void)references;
  (void)options;

  //
  // pxrUSD says... "If you author size you must also author extent."
  //
  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, cube, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    DCOUT("prop: " << prop.first);
    PARSE_TYPED_ATTRIBUTE(table, prop, "size", GeomCube, cube->size)
    ADD_PROPERTY(table, prop, GeomCube, cube->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<GeomMesh>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomMesh *mesh,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)references;
  (void)options;

  DCOUT("GeomMesh");

  auto SubdivisionSchemeHandler = [](const std::string &tok)
      -> nonstd::expected<GeomMesh::SubdivisionScheme, std::string> {
    using EnumTy = std::pair<GeomMesh::SubdivisionScheme, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(GeomMesh::SubdivisionScheme::SubdivisionSchemeNone, "none"),
        std::make_pair(GeomMesh::SubdivisionScheme::CatmullClark,
                       "catmullClark"),
        std::make_pair(GeomMesh::SubdivisionScheme::Loop, "loop"),
        std::make_pair(GeomMesh::SubdivisionScheme::Bilinear, "bilinear"),
    };
    return EnumHandler<GeomMesh::SubdivisionScheme>("subdivisionScheme", tok,
                                                    enums);
  };

  auto InterpolateBoundaryHandler = [](const std::string &tok)
      -> nonstd::expected<GeomMesh::InterpolateBoundary, std::string> {
    using EnumTy = std::pair<GeomMesh::InterpolateBoundary, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(GeomMesh::InterpolateBoundary::InterpolateBoundaryNone, "none"),
        std::make_pair(GeomMesh::InterpolateBoundary::EdgeAndCorner,
                       "edgeAndCorner"),
        std::make_pair(GeomMesh::InterpolateBoundary::EdgeOnly, "edgeOnly"),
    };
    return EnumHandler<GeomMesh::InterpolateBoundary>("interpolateBoundary",
                                                      tok, enums);
  };

  auto FaceVaryingLinearInterpolationHandler = [](const std::string &tok)
      -> nonstd::expected<GeomMesh::FaceVaryingLinearInterpolation,
                          std::string> {
    using EnumTy =
        std::pair<GeomMesh::FaceVaryingLinearInterpolation, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(GeomMesh::FaceVaryingLinearInterpolation::CornersPlus1,
                       "cornersPlus1"),
        std::make_pair(GeomMesh::FaceVaryingLinearInterpolation::CornersPlus2,
                       "cornersPlus2"),
        std::make_pair(GeomMesh::FaceVaryingLinearInterpolation::CornersOnly,
                       "cornersOnly"),
        std::make_pair(GeomMesh::FaceVaryingLinearInterpolation::Boundaries,
                       "boundaries"),
        std::make_pair(GeomMesh::FaceVaryingLinearInterpolation::FaceVaryingLinearInterpolationNone, "none"),
        std::make_pair(GeomMesh::FaceVaryingLinearInterpolation::All, "all"),
    };
    return EnumHandler<GeomMesh::FaceVaryingLinearInterpolation>(
        "facevaryingLinearInterpolation", tok, enums);
  };

  auto FamilyTypeHandler = [](const std::string &tok)
      -> nonstd::expected<GeomSubset::FamilyType, std::string> {
    using EnumTy = std::pair<GeomSubset::FamilyType, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(GeomSubset::FamilyType::Partition, "partition"),
        std::make_pair(GeomSubset::FamilyType::NonOverlapping, "nonOverlapping"),
        std::make_pair(GeomSubset::FamilyType::Unrestricted, "unrestricted"),
    };
    return EnumHandler<GeomSubset::FamilyType>("familyType", tok,
                                                    enums);
  };

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, mesh, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    DCOUT("GeomMesh prop: " << prop.first);
    PARSE_SINGLE_TARGET_PATH_RELATION(table, prop, kSkelSkeleton, mesh->skeleton)
    PARSE_TARGET_PATHS_RELATION(table, prop, kSkelBlendShapeTargets, mesh->blendShapeTargets)
    PARSE_TYPED_ATTRIBUTE(table, prop, "points", GeomMesh, mesh->points)
    PARSE_TYPED_ATTRIBUTE(table, prop, "normals", GeomMesh, mesh->normals)
    PARSE_TYPED_ATTRIBUTE(table, prop, "faceVertexCounts", GeomMesh,
                         mesh->faceVertexCounts)
    PARSE_TYPED_ATTRIBUTE(table, prop, "faceVertexIndices", GeomMesh,
                         mesh->faceVertexIndices)
    // Subd
    PARSE_TYPED_ATTRIBUTE(table, prop, "cornerIndices", GeomMesh,
                         mesh->cornerIndices)
    PARSE_TYPED_ATTRIBUTE(table, prop, "cornerSharpnesses", GeomMesh,
                         mesh->cornerSharpnesses)
    PARSE_TYPED_ATTRIBUTE(table, prop, "creaseIndices", GeomMesh,
                         mesh->creaseIndices)
    PARSE_TYPED_ATTRIBUTE(table, prop, "creaseLengths", GeomMesh,
                         mesh->creaseLengths)
    PARSE_TYPED_ATTRIBUTE(table, prop, "creaseSharpnesses", GeomMesh,
                         mesh->creaseSharpnesses)
    PARSE_TYPED_ATTRIBUTE(table, prop, "holeIndices", GeomMesh,
                         mesh->holeIndices)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "subdivisionScheme", GeomMesh::SubdivisionScheme,
                       SubdivisionSchemeHandler, GeomMesh,
                       mesh->subdivisionScheme, options.strict_allowedToken_check)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, "interpolateBoundary",
                       GeomMesh::InterpolateBoundary, InterpolateBoundaryHandler, GeomMesh,
                       mesh->interpolateBoundary, options.strict_allowedToken_check)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, "facevaryingLinearInterpolation",
                       GeomMesh::FaceVaryingLinearInterpolation, FaceVaryingLinearInterpolationHandler, GeomMesh,
                       mesh->faceVaryingLinearInterpolation, options.strict_allowedToken_check)
    // blendShape names
    PARSE_TYPED_ATTRIBUTE(table, prop, kSkelBlendShapes, GeomMesh, mesh->blendShapes)

    // subsetFamily for GeomSubset
    if (startsWith(prop.first, "subsetFamily")) {
      // uniform subsetFamily::<FAMILYNAME>:familyType = ...
      std::vector<std::string> names = split(prop.first, ":");

      if ((names.size() == 3) &&
          (names[0] == "subsetFamily") &&
          (names[2] == "familyType")) {

        DCOUT("subsetFamily" << prop.first);
        TypedAttributeWithFallback<GeomSubset::FamilyType> familyType{GeomSubset::FamilyType::Unrestricted};

        PARSE_UNIFORM_ENUM_PROPERTY(table, prop, prop.first,
                           GeomSubset::FamilyType, FamilyTypeHandler, GeomMesh,
                           familyType, options.strict_allowedToken_check)

        // NOTE: Ignore metadataum of familyType.
        
        // TODO: Validate familyName
        mesh->subsetFamilyTypeMap[value::token(names[1])] = familyType.get_value();

      }
    }

    // generic
    ADD_PROPERTY(table, prop, GeomMesh, mesh->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }


  return true;
}


template <>
bool ReconstructPrim<GeomCamera>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomCamera *camera,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {
  (void)references;
  (void)warn;
  (void)options;

  auto ProjectionHandler = [](const std::string &tok)
      -> nonstd::expected<GeomCamera::Projection, std::string> {
    using EnumTy = std::pair<GeomCamera::Projection, const char *>;
    constexpr std::array<EnumTy, 2> enums = {
        std::make_pair(GeomCamera::Projection::Perspective, "perspective"),
        std::make_pair(GeomCamera::Projection::Orthographic, "orthographic"),
    };

    auto ret =
        CheckAllowedTokens<GeomCamera::Projection, enums.size()>(enums, tok);
    if (!ret) {
      return nonstd::make_unexpected(ret.error());
    }

    for (auto &item : enums) {
      if (tok == item.second) {
        return item.first;
      }
    }

    // Should never reach here, though.
    return nonstd::make_unexpected(
        quote(tok) + " is invalid token for `projection` propety");
  };

  auto StereoRoleHandler = [](const std::string &tok)
      -> nonstd::expected<GeomCamera::StereoRole, std::string> {
    using EnumTy = std::pair<GeomCamera::StereoRole, const char *>;
    constexpr std::array<EnumTy, 3> enums = {
        std::make_pair(GeomCamera::StereoRole::Mono, "mono"),
        std::make_pair(GeomCamera::StereoRole::Left, "left"),
        std::make_pair(GeomCamera::StereoRole::Right, "right"),
    };

    auto ret =
        CheckAllowedTokens<GeomCamera::StereoRole, enums.size()>(enums, tok);
    if (!ret) {
      return nonstd::make_unexpected(ret.error());
    }

    for (auto &item : enums) {
      if (tok == item.second) {
        return item.first;
      }
    }

    // Should never reach here, though.
    return nonstd::make_unexpected(
        quote(tok) + " is invalid token for `stereoRole` propety");
  };

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, camera, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "focalLength", GeomCamera, camera->focalLength)
    PARSE_TYPED_ATTRIBUTE(table, prop, "focusDistance", GeomCamera,
                   camera->focusDistance)
    PARSE_TYPED_ATTRIBUTE(table, prop, "exposure", GeomCamera, camera->exposure)
    PARSE_TYPED_ATTRIBUTE(table, prop, "fStop", GeomCamera, camera->fStop)
    PARSE_TYPED_ATTRIBUTE(table, prop, "horizontalAperture", GeomCamera,
                   camera->horizontalAperture)
    PARSE_TYPED_ATTRIBUTE(table, prop, "horizontalApertureOffset", GeomCamera,
                   camera->horizontalApertureOffset)
    PARSE_TYPED_ATTRIBUTE(table, prop, "verticalAperture", GeomCamera,
                   camera->verticalAperture)
    PARSE_TYPED_ATTRIBUTE(table, prop, "verticalApertureOffset", GeomCamera,
                   camera->verticalApertureOffset)
    PARSE_TYPED_ATTRIBUTE(table, prop, "clippingRange", GeomCamera,
                   camera->clippingRange)
    PARSE_TYPED_ATTRIBUTE(table, prop, "clippingPlanes", GeomCamera,
                   camera->clippingPlanes)
    PARSE_TYPED_ATTRIBUTE(table, prop, "shutter:open", GeomCamera, camera->shutterOpen)
    PARSE_TYPED_ATTRIBUTE(table, prop, "shutter:close", GeomCamera,
                   camera->shutterClose)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, "projection", GeomCamera::Projection, ProjectionHandler, GeomCamera,
                       camera->projection, options.strict_allowedToken_check)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "stereoRole", GeomCamera::StereoRole, StereoRoleHandler, GeomCamera,
                       camera->stereoRole, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, GeomCamera, camera->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<GeomSubset>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    GeomSubset *subset,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)spec;
  (void)references;

  DCOUT("GeomSubset");

  // Currently schema only allows 'face'
  auto ElementTypeHandler = [](const std::string &tok)
      -> nonstd::expected<GeomSubset::ElementType, std::string> {
    using EnumTy = std::pair<GeomSubset::ElementType, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(GeomSubset::ElementType::Face, "face"),
        std::make_pair(GeomSubset::ElementType::Point, "point"),
    };
    return EnumHandler<GeomSubset::ElementType>("elementType", tok,
                                                    enums);
  };

  std::set<std::string> table;

  if (!prim::ReconstructMaterialBindingProperties(table, properties, subset, err)) {
    return false;
  }

  if (!prim::ReconstructCollectionProperties(
    table, properties, subset, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "familyName", GeomSubset, subset->familyName)
    PARSE_TYPED_ATTRIBUTE(table, prop, "indices", GeomSubset, subset->indices)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, "elementType", GeomSubset::ElementType, ElementTypeHandler, GeomSubset, subset->elementType, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, GeomSubset, subset->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<PointInstancer>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    PointInstancer *instancer,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {

  (void)warn;
  (void)references;
  (void)options;

  DCOUT("Reconstruct PointInstancer.");

  std::set<std::string> table;
  if (!ReconstructGPrimProperties(spec, table, properties, instancer, warn, err, options.strict_allowedToken_check)) {
    return false;
  }

  for (const auto &prop : properties) {
    PARSE_TARGET_PATHS_RELATION(table, prop, "prototypes", instancer->prototypes)
    PARSE_TYPED_ATTRIBUTE(table, prop, "protoIndices", PointInstancer, instancer->protoIndices)
    PARSE_TYPED_ATTRIBUTE(table, prop, "ids", PointInstancer, instancer->ids)
    PARSE_TYPED_ATTRIBUTE(table, prop, "positions", PointInstancer, instancer->positions)
    PARSE_TYPED_ATTRIBUTE(table, prop, "orientations", PointInstancer, instancer->orientations)
    PARSE_TYPED_ATTRIBUTE(table, prop, "scales", PointInstancer, instancer->scales)
    PARSE_TYPED_ATTRIBUTE(table, prop, "velocities", PointInstancer, instancer->velocities)
    PARSE_TYPED_ATTRIBUTE(table, prop, "accelerations", PointInstancer, instancer->accelerations)
    PARSE_TYPED_ATTRIBUTE(table, prop, "angularVelocities", PointInstancer, instancer->angularVelocities)
    PARSE_TYPED_ATTRIBUTE(table, prop, "invisibleIds", PointInstancer, instancer->invisibleIds)

    ADD_PROPERTY(table, prop, PointInstancer, instancer->props)
    PARSE_PROPERTY_END_MAKE_ERROR(table, prop)
  }

  return true;
}

template <>
bool ReconstructShader<ShaderNode>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    ShaderNode *node,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)options;

  if (!node) {
    return false;
  }

  // TODO: references
  (void)references;

  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>

  // Add everything to props.
  for (auto &prop : properties) {
    ADD_PROPERTY(table, prop, ShaderNode, node->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  DCOUT("ShaderNode reconstructed.");
  return true;
}

template <>
bool ReconstructShader<UsdPreviewSurface>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPreviewSurface *surface,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options) {
  (void)spec;
  (void)references;
  (void)options;

  auto OpacityModeHandler = [](const std::string &tok)
      -> nonstd::expected<UsdPreviewSurface::OpacityMode, std::string> {
    using EnumTy = std::pair<UsdPreviewSurface::OpacityMode, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(UsdPreviewSurface::OpacityMode::Transparent, "transparent"),
        std::make_pair(UsdPreviewSurface::OpacityMode::Presence, "presence"),
    };

    return EnumHandler<UsdPreviewSurface::OpacityMode>(
        "inputs:opacityMode", tok, enums);
  };

  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>
  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:diffuseColor", UsdPreviewSurface,
                         surface->diffuseColor)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:emissiveColor", UsdPreviewSurface,
                         surface->emissiveColor)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:roughness", UsdPreviewSurface,
                         surface->roughness)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:specularColor", UsdPreviewSurface,
                         surface->specularColor)  // specular workflow
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:metallic", UsdPreviewSurface,
                         surface->metallic)  // non specular workflow
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:clearcoat", UsdPreviewSurface,
                         surface->clearcoat)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:clearcoatRoughness",
                         UsdPreviewSurface, surface->clearcoatRoughness)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:opacity", UsdPreviewSurface,
                         surface->opacity)
    // From 2.6
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, "inputs:opacityMode",
                       UsdPreviewSurface::OpacityMode, OpacityModeHandler, UsdPreviewSurface,
                       surface->opacityMode, options.strict_allowedToken_check)

    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:opacityThreshold",
                         UsdPreviewSurface, surface->opacityThreshold)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:ior", UsdPreviewSurface,
                         surface->ior)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:normal", UsdPreviewSurface,
                         surface->normal)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:dispacement", UsdPreviewSurface,
                         surface->displacement)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:occlusion", UsdPreviewSurface,
                         surface->occlusion)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:useSpecularWorkflow",
                         UsdPreviewSurface, surface->useSpecularWorkflow)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:surface", UsdPreviewSurface,
                   surface->outputsSurface)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:displacement", UsdPreviewSurface,
                   surface->outputsDisplacement)
    ADD_PROPERTY(table, prop, UsdPreviewSurface, surface->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructShader<UsdUVTexture>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdUVTexture *texture,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;

  auto SourceColorSpaceHandler = [](const std::string &tok)
      -> nonstd::expected<UsdUVTexture::SourceColorSpace, std::string> {
    using EnumTy = std::pair<UsdUVTexture::SourceColorSpace, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(UsdUVTexture::SourceColorSpace::Auto, "auto"),
        std::make_pair(UsdUVTexture::SourceColorSpace::Raw, "raw"),
        std::make_pair(UsdUVTexture::SourceColorSpace::SRGB, "sRGB"),
    };

    return EnumHandler<UsdUVTexture::SourceColorSpace>(
        "inputs:sourceColorSpace", tok, enums);
  };

  auto WrapHandler = [](const std::string &tok)
      -> nonstd::expected<UsdUVTexture::Wrap, std::string> {
    using EnumTy = std::pair<UsdUVTexture::Wrap, const char *>;
    const std::vector<EnumTy> enums = {
        std::make_pair(UsdUVTexture::Wrap::UseMetadata, "useMetadata"),
        std::make_pair(UsdUVTexture::Wrap::Black, "black"),
        std::make_pair(UsdUVTexture::Wrap::Clamp, "clamp"),
        std::make_pair(UsdUVTexture::Wrap::Repeat, "repeat"),
        std::make_pair(UsdUVTexture::Wrap::Mirror, "mirror"),
    };

    return EnumHandler<UsdUVTexture::Wrap>(
        "inputs:wrap*", tok, enums);
  };

  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>

  for (auto &prop : properties) {
    DCOUT("prop.name = " << prop.first);
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:file", UsdUVTexture, texture->file)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:st", UsdUVTexture,
                          texture->st)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, "inputs:sourceColorSpace",
                       UsdUVTexture::SourceColorSpace, SourceColorSpaceHandler, UsdUVTexture,
                       texture->sourceColorSpace, options.strict_allowedToken_check)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, "inputs:wrapS",
                       UsdUVTexture::Wrap, WrapHandler, UsdUVTexture,
                       texture->wrapS, options.strict_allowedToken_check)
    PARSE_TIMESAMPLED_ENUM_PROPERTY(table, prop, "inputs:wrapT",
                       UsdUVTexture::Wrap, WrapHandler, UsdUVTexture,
                       texture->wrapT, options.strict_allowedToken_check)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:r", UsdUVTexture,
                                  texture->outputsR)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:g", UsdUVTexture,
                                  texture->outputsG)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:b", UsdUVTexture,
                                  texture->outputsB)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:a", UsdUVTexture,
                                  texture->outputsA)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:rgb", UsdUVTexture,
                                  texture->outputsRGB)
    ADD_PROPERTY(table, prop, UsdUVTexture, texture->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  DCOUT("UsdUVTexture reconstructed.");
  return true;
}

template <>
bool ReconstructShader<UsdPrimvarReader_int>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_int *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>
  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_int,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        ret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
        if (ret.code == ParseResult::ResultCode::Success) {
          // ok
          continue;
        } else {
          PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", ret.err));
        }
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_int, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_int, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return false;
}

template <>
bool ReconstructShader<UsdPrimvarReader_float>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_float *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>
  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_float,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_float, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_float, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return false;
}

template <>
bool ReconstructShader<UsdPrimvarReader_float2>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_float2 *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>
  for (auto &prop : properties) {
    DCOUT("Primreader_float2 prop = " << prop.first);
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_float2,
                   preader->fallback)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_float2, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_float2, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructShader<UsdPrimvarReader_float3>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_float3 *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>
  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_float3,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_float3, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_float3, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructShader<UsdPrimvarReader_float4>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_float4 *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>

  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_float4,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_float4, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_float4, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return true;
}

template <>
bool ReconstructShader<UsdPrimvarReader_string>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_string *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>

  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_string,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_string, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_string, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return true;
}

template <>
bool ReconstructShader<UsdPrimvarReader_vector>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_vector *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>

  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_vector,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_vector, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_vector, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return true;
}

template <>
bool ReconstructShader<UsdPrimvarReader_normal>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_normal *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>

  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_normal,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_normal, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_normal, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return true;
}

template <>
bool ReconstructShader<UsdPrimvarReader_point>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_point *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>

  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_point,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_point, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_point, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return true;
}

template <>
bool ReconstructShader<UsdPrimvarReader_matrix>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdPrimvarReader_matrix *preader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>

  for (auto &prop : properties) {
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:fallback", UsdPrimvarReader_matrix,
                   preader->fallback)
    if ((prop.first == kInputsVarname) && !table.count(kInputsVarname)) {
      // Support older spec: `token` for varname
      TypedAttribute<Animatable<value::token>> tok_attr;
      auto ret = ParseTypedAttribute(table, prop.first, prop.second, kInputsVarname, tok_attr);
      if (ret.code == ParseResult::ResultCode::Success) {
        if (!ConvertTokenAttributeToStringAttribute(tok_attr, preader->varname)) {
          PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname token type to string type.");
        }
        DCOUT("`token` attribute is converted to `string` attribute.");
        continue;
      } else if (ret.code == ParseResult::ResultCode::TypeMismatch) {
        //TypedAttribute<Animatable<value::StringData>> sdata_attr;
        //auto sdret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", sdata_attr);
        //if (sdret.code == ParseResult::ResultCode::Success) {
        //  if (!ConvertStringDataAttributeToStringAttribute(sdata_attr, preader->varname)) {
        //    PUSH_ERROR_AND_RETURN("Failed to convert inputs:varname StringData type to string type.");
        //  }
        //} else if (sdret.code == ParseResult::ResultCode::TypeMismatch) {
          auto sret = ParseTypedAttribute(table, prop.first, prop.second, "inputs:varname", preader->varname);
          if (sret.code == ParseResult::ResultCode::Success) {
            DCOUT("Parsed string typed inputs:varname.");
            // ok
            continue;
          } else {
            PUSH_ERROR_AND_RETURN(fmt::format("Faied to parse inputs:varname: {}", sret.err));
          }
        //}
      }
    }
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdPrimvarReader_matrix, preader->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_matrix, preader->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return true;
}

template <>
bool ReconstructShader<UsdTransform2d>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    UsdTransform2d *transform,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;
  table.insert("info:id"); // `info:id` is already parsed in ReconstructPrim<Shader>
  for (auto &prop : properties) {
    DCOUT("prop = " << prop.first);
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:in", UsdTransform2d,
                   transform->in)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:rotation", UsdTransform2d,
                   transform->rotation)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:scale", UsdTransform2d,
                   transform->scale)
    PARSE_TYPED_ATTRIBUTE(table, prop, "inputs:translation", UsdTransform2d,
                   transform->translation)
    PARSE_SHADER_TERMINAL_ATTRIBUTE(table, prop, "outputs:result",
                                  UsdTransform2d, transform->result)
    ADD_PROPERTY(table, prop, UsdPrimvarReader_float2, transform->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }

  return true;
}

template <>
bool ReconstructPrim<Shader>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    Shader *shader,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)properties;
  (void)options;

  bool is_generic_shader{false};
  auto info_id_prop = properties.find("info:id");
  if (info_id_prop == properties.end()) {
    // Guess MatrialX shader. info:id will be resolved by importing referenced .mtlx.
    // Treat generic Shader at the moment.
    is_generic_shader = true;
    //PUSH_ERROR_AND_RETURN("`Shader` must contain `info:id` property.");
  }

  std::string shader_type;
  if (!is_generic_shader) {
    if (info_id_prop->second.is_attribute()) {
      const Attribute &attr = info_id_prop->second.get_attribute();
      if ((attr.type_name() == value::kToken)) {
        if (auto pv = attr.get_value<value::token>()) {
          shader_type = pv.value().str();
        } else {
          PUSH_ERROR_AND_RETURN("Internal errror. `info:id` has invalid type.");
        }
      } else {
        PUSH_ERROR_AND_RETURN("`info:id` attribute must be `token` type.");
      }

      // For some corrupted? USDZ file does not have `uniform` variability.
      if (attr.variability() != Variability::Uniform) {
        PUSH_WARN("`info:id` attribute must have `uniform` variability.");
      }
    } else {
      PUSH_ERROR_AND_RETURN("Invalid type or value for `info:id` property in `Shader`.");
    }

    DCOUT("info:id = " << shader_type);
  }


  if (shader_type.compare(kUsdPreviewSurface) == 0) {
    UsdPreviewSurface surface;
    if (!ReconstructShader<UsdPreviewSurface>(spec, properties, references,
                                              &surface, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct " << kUsdPreviewSurface);
    }
    shader->info_id = kUsdPreviewSurface;
    shader->value = surface;
    DCOUT("info_id = " << shader->info_id);
  } else if (shader_type.compare(kUsdUVTexture) == 0) {
    UsdUVTexture texture;
    if (!ReconstructShader<UsdUVTexture>(spec, properties, references,
                                         &texture, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct " << kUsdUVTexture);
    }
    shader->info_id = kUsdUVTexture;
    shader->value = texture;
  } else if (shader_type.compare(kUsdPrimvarReader_int) == 0) {
    UsdPrimvarReader_int preader;
    if (!ReconstructShader<UsdPrimvarReader_int>(spec, properties, references,
                                                 &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_int);
    }
    shader->info_id = kUsdPrimvarReader_int;
    shader->value = preader;
  } else if (shader_type.compare(kUsdPrimvarReader_float) == 0) {
    UsdPrimvarReader_float preader;
    if (!ReconstructShader<UsdPrimvarReader_float>(spec, properties, references,
                                                   &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_float);
    }
    shader->info_id = kUsdPrimvarReader_float;
    shader->value = preader;
  } else if (shader_type.compare(kUsdPrimvarReader_float2) == 0) {
    UsdPrimvarReader_float2 preader;
    if (!ReconstructShader<UsdPrimvarReader_float2>(spec, properties, references,
                                                    &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_float2);
    }
    shader->info_id = kUsdPrimvarReader_float2;
    shader->value = preader;
  } else if (shader_type.compare(kUsdPrimvarReader_float3) == 0) {
    UsdPrimvarReader_float3 preader;
    if (!ReconstructShader<UsdPrimvarReader_float3>(spec,properties, references,
                                                    &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_float3);
    }
    shader->info_id = kUsdPrimvarReader_float3;
    shader->value = preader;
  } else if (shader_type.compare(kUsdPrimvarReader_float4) == 0) {
    UsdPrimvarReader_float4 preader;
    if (!ReconstructShader<UsdPrimvarReader_float4>(spec,properties, references,
                                                    &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_float4);
    }
    shader->info_id = kUsdPrimvarReader_float4;
    shader->value = preader;
  } else if (shader_type.compare(kUsdPrimvarReader_string) == 0) {
    UsdPrimvarReader_string preader;
    if (!ReconstructShader<UsdPrimvarReader_string>(spec,properties, references,
                                                    &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_string);
    }
    shader->info_id = kUsdPrimvarReader_string;
    shader->value = preader;
  } else if (shader_type.compare(kUsdPrimvarReader_vector) == 0) {
    UsdPrimvarReader_vector preader;
    if (!ReconstructShader<UsdPrimvarReader_vector>(spec,properties, references,
                                                    &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_vector);
    }
    shader->info_id = kUsdPrimvarReader_vector;
    shader->value = preader;
  } else if (shader_type.compare(kUsdPrimvarReader_normal) == 0) {
    UsdPrimvarReader_normal preader;
    if (!ReconstructShader<UsdPrimvarReader_normal>(spec,properties, references,
                                                    &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_normal);
    }
    shader->info_id = kUsdPrimvarReader_normal;
    shader->value = preader;
  } else if (shader_type.compare(kUsdPrimvarReader_point) == 0) {
    UsdPrimvarReader_point preader;
    if (!ReconstructShader<UsdPrimvarReader_point>(spec,properties, references,
                                                    &preader, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdPrimvarReader_point);
    }
    shader->info_id = kUsdPrimvarReader_point;
    shader->value = preader;
  } else if (shader_type.compare(kUsdTransform2d) == 0) {
    UsdTransform2d transform;
    if (!ReconstructShader<UsdTransform2d>(spec,properties, references,
                                                    &transform, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct "
                            << kUsdTransform2d);
    }
    shader->info_id = kUsdTransform2d;
    shader->value = transform;
  } else {
    // Reconstruct as generic ShaderNode
    ShaderNode surface;
    if (!ReconstructShader<ShaderNode>(spec,properties, references,
                                              &surface, warn, err, options)) {
      PUSH_ERROR_AND_RETURN("Failed to Reconstruct " << shader_type);
    }
    if (shader_type.size()) {
      shader->info_id = shader_type;
    }
    shader->value = surface;
  }

  DCOUT("Shader reconstructed.");

  return true;
}

template <>
bool ReconstructPrim<Material>(
    const Specifier &spec,
    const PropertyMap &properties,
    const ReferenceList &references,
    Material *material,
    std::string *warn,
    std::string *err,
    const PrimReconstructOptions &options)
{
  (void)spec;
  (void)references;
  (void)options;
  std::set<std::string> table;

  // TODO: special treatment for properties with 'inputs' and 'outputs' namespace.

  // For `Material`, `outputs` are terminal attribute and treated as input attribute with connection(Should be "token output:surface.connect = </path/to/shader>").
  for (auto &prop : properties) {
    PARSE_SHADER_INPUT_CONNECTION_PROPERTY(table, prop, "outputs:surface",
                                  Material, material->surface)
    PARSE_SHADER_INPUT_CONNECTION_PROPERTY(table, prop, "outputs:displacement",
                                  Material, material->displacement)
    PARSE_SHADER_INPUT_CONNECTION_PROPERTY(table, prop, "outputs:volume",
                                  Material, material->volume)
    PARSE_UNIFORM_ENUM_PROPERTY(table, prop, kPurpose, Purpose, PurposeEnumHandler, Material,
                       material->purpose, options.strict_allowedToken_check)
    ADD_PROPERTY(table, prop, Material, material->props)
    PARSE_PROPERTY_END_MAKE_WARN(table, prop)
  }
  return true;
}

///
/// -- PrimSpec
///

#define RECONSTRUCT_PRIM_PRIMSPEC_IMPL(__prim_ty) \
template <> \
bool ReconstructPrim<__prim_ty>( \
    const PrimSpec &primspec, \
    __prim_ty *prim, \
    std::string *warn, \
    std::string *err, \
    const PrimReconstructOptions &options) { \
 \
  ReferenceList references; /* dummy */ \
 \
  return ReconstructPrim<__prim_ty>(primspec.specifier(), primspec.props(), references, prim, warn, err, options); \
}

RECONSTRUCT_PRIM_PRIMSPEC_IMPL(Xform)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(Model)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(Scope)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomMesh)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomPoints)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomCylinder)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomCube)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomCone)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomSphere)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomCapsule)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomBasisCurves)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomCamera)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(GeomSubset)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(SphereLight)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(DomeLight)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(CylinderLight)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(DiskLight)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(DistantLight)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(SkelRoot)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(Skeleton)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(SkelAnimation)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(BlendShape)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(Shader)
RECONSTRUCT_PRIM_PRIMSPEC_IMPL(Material)


} // namespace prim

} // namespace tinyusdz

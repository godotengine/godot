// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// UsdGeom API implementations

#include <sstream>

#include "usdShade.hh"
#include "str-util.hh"

#include "common-macros.inc"

namespace tinyusdz {

std::string to_string(const MaterialBindingStrength strength) {
  switch (strength) {
    case MaterialBindingStrength::WeakerThanDescendants: {
      return kWeaderThanDescendants;
    }
    case MaterialBindingStrength::StrongerThanDescendants: {
      return kStrongerThanDescendants;
    }
  }

  return "[[Invalid MaterialBindingStrength]]";
}

bool UsdShadePrim::has_sdr_metadata(const std::string &key) {
  if (!metas().sdrMetadata.has_value()) {
    return false;
  }

  const Dictionary &dict = metas().sdrMetadata.value();

  if (!HasCustomDataKey(dict, key)) {
    return false;
  }

  // check the type of value.
  MetaVariable value;
  if (!GetCustomDataByKey(dict, key, &value)) {
    return false;
  }

  if (value.type_id() != value::TypeTraits<std::string>::type_id()) {
    return false;
  }

  return true;
}

const std::string UsdShadePrim::get_sdr_metadata(const std::string &key) {
  if (!metas().sdrMetadata.has_value()) {
    return std::string();
  }

  const Dictionary &dict = metas().sdrMetadata.value();

  if (!HasCustomDataKey(dict, key)) {
    return std::string();
  }

  // check the type of value.
  MetaVariable var;
  if (!GetCustomDataByKey(dict, key, &var)) {
    return std::string();
  }

  if (var.type_id() != value::TypeTraits<std::string>::type_id()) {
    return std::string();
  }

  std::string svalue;
  if (!var.get_value(&svalue)) {
    return std::string();
  }

  return svalue;
}

bool UsdShadePrim::set_sdr_metadata(const std::string &key, const std::string &value) {

  Dictionary &dict = metas().sdrMetadata.value();
  bool ret = SetCustomDataByKey(key, value, dict);
  return ret;
}

value::token MaterialBinding::get_materialBindingStrength(const value::token &purpose) {

  if (purpose.str() == kAllPurpose().str()) {
    if (materialBinding && materialBinding.value().metas().bindMaterialAs) {
      return materialBinding.value().metas().bindMaterialAs.value();
    }
  } else if (purpose.str() == "full") {
    if (materialBindingFull && materialBindingFull.value().metas().bindMaterialAs) {
      return materialBindingFull.value().metas().bindMaterialAs.value();
    }
  } else if (purpose.str() == "preview") {
    if (materialBindingPreview && materialBindingPreview.value().metas().bindMaterialAs) {
      return materialBindingPreview.value().metas().bindMaterialAs.value();
    }
  } else {
    if (_materialBindingMap.count(purpose.str())) {
      const auto &m = _materialBindingMap.at(purpose.str());
      if (m.metas().bindMaterialAs) {
        return m.metas().bindMaterialAs.value();
      }
    }
  }

  return value::token(kWeaderThanDescendants);
}

value::token MaterialBinding::get_materialBindingStrengthCollection(const value::token &coll_name, const value::token &purpose) {

  if (coll_name.str().empty()) {
    return get_materialBindingStrength(purpose);
  }

  if (_materialBindingCollectionMap.count(coll_name.str())) {
    const auto &coll_mb = _materialBindingCollectionMap.at(coll_name.str());

    if (coll_mb.count(purpose.str())) {
      const Relationship *prel{nullptr};
      if (coll_mb.at(purpose.str(), &prel)) {
        if (prel->metas().bindMaterialAs) {
          return prel->metas().bindMaterialAs.value();
        }
      }
    }
  }

  return value::token(kWeaderThanDescendants);
}

namespace {

//constexpr auto kPrimvarPrefix = "primvar::";

} // namespace


} // namespace tinyusdz



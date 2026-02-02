// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
#if defined(__wasi__)
#else
#include <thread>
#endif

#include "common-macros.inc"
#include "crate-format.hh"
#include "external/mapbox/eternal/include/mapbox/eternal.hpp"
#include "pprinter.hh"
#include "value-types.hh"

namespace tinyusdz {
namespace crate {

#if 0

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif

nonstd::expected<CrateDataType, std::string> GetCrateDataType(int32_t type_id) {


  // TODO: Compile-time map using maxbox/eternal
  static std::map<uint32_t, CrateDataType> table;
  DCOUT("type_id = " << type_id);

  if (table.size() == 0) {
    // Register data types

    // See <pxrUSD>/pxr/usd/usd/crateDataTypes.h

#define ADD_VALUE_TYPE(NAME_STR, TYPE_ID, SUPPORTS_ARRAY)     \
  {                                                           \
    assert(table.count(static_cast<uint32_t>(TYPE_ID)) == 0); \
    table[static_cast<uint32_t>(TYPE_ID)] =                   \
        CrateDataType(NAME_STR, TYPE_ID, SUPPORTS_ARRAY);     \
  }

    // (num_string, type_id(in crateData), supports_array)

    // 0 is reserved as `Invalid` type.
    ADD_VALUE_TYPE("Invald", CrateDataTypeId::CRATE_DATA_TYPE_INVALID, false)

    // Array types.
    ADD_VALUE_TYPE("Bool", CrateDataTypeId::CRATE_DATA_TYPE_BOOL, true)

    ADD_VALUE_TYPE("UChar", CrateDataTypeId::CRATE_DATA_TYPE_UCHAR, true)
    ADD_VALUE_TYPE("Int", CrateDataTypeId::CRATE_DATA_TYPE_INT, true)
    ADD_VALUE_TYPE("UInt", CrateDataTypeId::CRATE_DATA_TYPE_UINT, true)
    ADD_VALUE_TYPE("Int64", CrateDataTypeId::CRATE_DATA_TYPE_INT64, true)
    ADD_VALUE_TYPE("UInt64", CrateDataTypeId::CRATE_DATA_TYPE_UINT64, true)

    ADD_VALUE_TYPE("Half", CrateDataTypeId::CRATE_DATA_TYPE_HALF, true)
    ADD_VALUE_TYPE("Float", CrateDataTypeId::CRATE_DATA_TYPE_FLOAT, true)
    ADD_VALUE_TYPE("Double", CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE, true)

    ADD_VALUE_TYPE("String", CrateDataTypeId::CRATE_DATA_TYPE_STRING, true)
    ADD_VALUE_TYPE("Token", CrateDataTypeId::CRATE_DATA_TYPE_TOKEN, true)
    ADD_VALUE_TYPE("AssetPath", CrateDataTypeId::CRATE_DATA_TYPE_ASSET_PATH,
                   true)

    ADD_VALUE_TYPE("Matrix2d", CrateDataTypeId::CRATE_DATA_TYPE_MATRIX2D, true)
    ADD_VALUE_TYPE("Matrix3d", CrateDataTypeId::CRATE_DATA_TYPE_MATRIX3D, true)
    ADD_VALUE_TYPE("Matrix4d", CrateDataTypeId::CRATE_DATA_TYPE_MATRIX4D, true)

    ADD_VALUE_TYPE("Quatd", CrateDataTypeId::CRATE_DATA_TYPE_QUATD, true)
    ADD_VALUE_TYPE("Quatf", CrateDataTypeId::CRATE_DATA_TYPE_QUATF, true)
    ADD_VALUE_TYPE("Quath", CrateDataTypeId::CRATE_DATA_TYPE_QUATH, true)

    ADD_VALUE_TYPE("Vec2d", CrateDataTypeId::CRATE_DATA_TYPE_VEC2D, true)
    ADD_VALUE_TYPE("Vec2f", CrateDataTypeId::CRATE_DATA_TYPE_VEC2F, true)
    ADD_VALUE_TYPE("Vec2h", CrateDataTypeId::CRATE_DATA_TYPE_VEC2H, true)
    ADD_VALUE_TYPE("Vec2i", CrateDataTypeId::CRATE_DATA_TYPE_VEC2I, true)

    ADD_VALUE_TYPE("Vec3d", CrateDataTypeId::CRATE_DATA_TYPE_VEC3D, true)
    ADD_VALUE_TYPE("Vec3f", CrateDataTypeId::CRATE_DATA_TYPE_VEC3F, true)
    ADD_VALUE_TYPE("Vec3h", CrateDataTypeId::CRATE_DATA_TYPE_VEC3H, true)
    ADD_VALUE_TYPE("Vec3i", CrateDataTypeId::CRATE_DATA_TYPE_VEC3I, true)

    ADD_VALUE_TYPE("Vec4d", CrateDataTypeId::CRATE_DATA_TYPE_VEC4D, true)
    ADD_VALUE_TYPE("Vec4f", CrateDataTypeId::CRATE_DATA_TYPE_VEC4F, true)
    ADD_VALUE_TYPE("Vec4h", CrateDataTypeId::CRATE_DATA_TYPE_VEC4H, true)
    ADD_VALUE_TYPE("Vec4i", CrateDataTypeId::CRATE_DATA_TYPE_VEC4I, true)

    // Non-array types.

    //
    // commented = TODO
    //
    ADD_VALUE_TYPE("Dictionary", CrateDataTypeId::CRATE_DATA_TYPE_DICTIONARY,
                   false)

    ADD_VALUE_TYPE("TokenListOp",
                   CrateDataTypeId::CRATE_DATA_TYPE_TOKEN_LIST_OP, false)
    ADD_VALUE_TYPE("StringListOp",
                   CrateDataTypeId::CRATE_DATA_TYPE_STRING_LIST_OP, false)
    ADD_VALUE_TYPE("PathListOp", CrateDataTypeId::CRATE_DATA_TYPE_PATH_LIST_OP,
                   false)
    ADD_VALUE_TYPE("ReferenceListOp",
                   CrateDataTypeId::CRATE_DATA_TYPE_REFERENCE_LIST_OP, false)
    ADD_VALUE_TYPE("IntListOp", CrateDataTypeId::CRATE_DATA_TYPE_INT_LIST_OP,
                   false)
    ADD_VALUE_TYPE("Int64ListOp",
                   CrateDataTypeId::CRATE_DATA_TYPE_INT64_LIST_OP, false)
    ADD_VALUE_TYPE("UIntListOp", CrateDataTypeId::CRATE_DATA_TYPE_UINT_LIST_OP,
                   false)
    ADD_VALUE_TYPE("UInt64ListOp",
                   CrateDataTypeId::CRATE_DATA_TYPE_UINT64_LIST_OP, false)

    ADD_VALUE_TYPE("PathVector", CrateDataTypeId::CRATE_DATA_TYPE_PATH_VECTOR,
                   false)
    ADD_VALUE_TYPE("TokenVector", CrateDataTypeId::CRATE_DATA_TYPE_TOKEN_VECTOR,
                   false)

    ADD_VALUE_TYPE("Specifier", CrateDataTypeId::CRATE_DATA_TYPE_SPECIFIER,
                   false)
    ADD_VALUE_TYPE("Permission", CrateDataTypeId::CRATE_DATA_TYPE_PERMISSION,
                   false)
    ADD_VALUE_TYPE("Variability", CrateDataTypeId::CRATE_DATA_TYPE_VARIABILITY,
                   false)

    ADD_VALUE_TYPE("VariantSelectionMap",
                   CrateDataTypeId::CRATE_DATA_TYPE_VARIANT_SELECTION_MAP,
                   false)
    ADD_VALUE_TYPE("TimeSamples", CrateDataTypeId::CRATE_DATA_TYPE_TIME_SAMPLES,
                   false)
    ADD_VALUE_TYPE("Payload", CrateDataTypeId::CRATE_DATA_TYPE_PAYLOAD, false)
    ADD_VALUE_TYPE("DoubleVector",
                   CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE_VECTOR, false)
    ADD_VALUE_TYPE("LayerOffsetVector",
                   CrateDataTypeId::CRATE_DATA_TYPE_LAYER_OFFSET_VECTOR, false)
    ADD_VALUE_TYPE("StringVector",
                   CrateDataTypeId::CRATE_DATA_TYPE_STRING_VECTOR, false)
    ADD_VALUE_TYPE("ValueBlock", CrateDataTypeId::CRATE_DATA_TYPE_VALUE_BLOCK,
                   false)
    ADD_VALUE_TYPE("Value", CrateDataTypeId::CRATE_DATA_TYPE_VALUE, false)
    ADD_VALUE_TYPE("UnregisteredValue",
                   CrateDataTypeId::CRATE_DATA_TYPE_UNREGISTERED_VALUE, false)
    ADD_VALUE_TYPE("UnregisteredValueListOp",
                   CrateDataTypeId::CRATE_DATA_TYPE_UNREGISTERED_VALUE_LIST_OP,
                   false)
    ADD_VALUE_TYPE("PayloadListOp",
                   CrateDataTypeId::CRATE_DATA_TYPE_PAYLOAD_LIST_OP, false)
    ADD_VALUE_TYPE("TimeCode", CrateDataTypeId::CRATE_DATA_TYPE_TIME_CODE, true)
  }
#undef ADD_VALUE_TYPE

  if (type_id < 0) {
    return nonstd::make_unexpected("Unknown type id: " +
                                   std::to_string(type_id));
  }

  if (!table.count(static_cast<uint32_t>(type_id))) {
    // Invalid or unsupported.
    return nonstd::make_unexpected("Unknown or unspported type id: " +
                                   std::to_string(type_id));
  }

  return table.at(static_cast<uint32_t>(type_id));
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#else

nonstd::expected<CrateDataType, std::string> GetCrateDataType(int32_t type_id) {
  // See <pxrUSD>/pxr/usd/usd/crateDataTypes.h

  // TODO: Use type name in value-types.hh and prim-types.hh?
  MAPBOX_ETERNAL_CONSTEXPR const auto tymap =
      mapbox::eternal::map<CrateDataTypeId, mapbox::eternal::string>({
          {CrateDataTypeId::CRATE_DATA_TYPE_INVALID, "Invalid"},  // 0
          {CrateDataTypeId::CRATE_DATA_TYPE_BOOL, "Bool"},
          {CrateDataTypeId::CRATE_DATA_TYPE_UCHAR, "UChar"},
          {CrateDataTypeId::CRATE_DATA_TYPE_INT, "Int"},
          {CrateDataTypeId::CRATE_DATA_TYPE_UINT, "UInt"},
          {CrateDataTypeId::CRATE_DATA_TYPE_INT64, "Int64"},
          {CrateDataTypeId::CRATE_DATA_TYPE_UINT64, "UInt64"},

          {CrateDataTypeId::CRATE_DATA_TYPE_HALF, "Half"},
          {CrateDataTypeId::CRATE_DATA_TYPE_FLOAT, "Float"},
          {CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE, "Double"},

          {CrateDataTypeId::CRATE_DATA_TYPE_STRING, "String"},
          {CrateDataTypeId::CRATE_DATA_TYPE_TOKEN, "Token"},
          {CrateDataTypeId::CRATE_DATA_TYPE_ASSET_PATH, "AssetPath"},

          {CrateDataTypeId::CRATE_DATA_TYPE_MATRIX2D, "Matrix2d"},
          {CrateDataTypeId::CRATE_DATA_TYPE_MATRIX3D, "Matrix3d"},
          {CrateDataTypeId::CRATE_DATA_TYPE_MATRIX4D, "Matrix4d"},

          {CrateDataTypeId::CRATE_DATA_TYPE_QUATD, "Quatd"},
          {CrateDataTypeId::CRATE_DATA_TYPE_QUATF, "Quatf"},
          {CrateDataTypeId::CRATE_DATA_TYPE_QUATH, "Quath"},

          {CrateDataTypeId::CRATE_DATA_TYPE_VEC2D, "Vec2d"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC2F, "Vec2f"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC2H, "Vec2h"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC2I, "Vec2i"},

          {CrateDataTypeId::CRATE_DATA_TYPE_VEC3D, "Vec3d"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC3F, "Vec3f"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC3H, "Vec3h"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC3I, "Vec3i"},

          {CrateDataTypeId::CRATE_DATA_TYPE_VEC4D, "Vec4d"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC4F, "Vec4f"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC4H, "Vec4h"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC4I, "Vec4i"},

          // Non-array types.
          {CrateDataTypeId::CRATE_DATA_TYPE_DICTIONARY, "Dictionary"},
          {CrateDataTypeId::CRATE_DATA_TYPE_TOKEN_LIST_OP, "TokenListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_STRING_LIST_OP, "StringListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_PATH_LIST_OP, "PathListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_REFERENCE_LIST_OP,
           "ReferenceListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_INT_LIST_OP, "IntListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_INT64_LIST_OP, "Int64ListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_UINT_LIST_OP, "UIntListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_UINT64_LIST_OP, "UInt64ListOp"},

          {CrateDataTypeId::CRATE_DATA_TYPE_PATH_VECTOR, "PathVector"},
          {CrateDataTypeId::CRATE_DATA_TYPE_TOKEN_VECTOR, "TokenVector"},
          {CrateDataTypeId::CRATE_DATA_TYPE_SPECIFIER, "Specifier"},
          {CrateDataTypeId::CRATE_DATA_TYPE_PERMISSION, "Permission"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VARIABILITY, "Variability"},

          {CrateDataTypeId::CRATE_DATA_TYPE_VARIANT_SELECTION_MAP,
           "VariantSelectionMap"},
          {CrateDataTypeId::CRATE_DATA_TYPE_TIME_SAMPLES, "TimeSamples"},
          {CrateDataTypeId::CRATE_DATA_TYPE_PAYLOAD, "Payload"},
          {CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE_VECTOR, "DoubleVector"},
          {CrateDataTypeId::CRATE_DATA_TYPE_LAYER_OFFSET_VECTOR,
           "LayerOffsetVector"},
          {CrateDataTypeId::CRATE_DATA_TYPE_STRING_VECTOR, "StringVector"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VALUE_BLOCK, "ValueBlock"},
          {CrateDataTypeId::CRATE_DATA_TYPE_VALUE, "Value"},
          {CrateDataTypeId::CRATE_DATA_TYPE_UNREGISTERED_VALUE,
           "UnregisteredValue"},
          {CrateDataTypeId::CRATE_DATA_TYPE_UNREGISTERED_VALUE_LIST_OP,
           "UnregisteredValueListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_PAYLOAD_LIST_OP, "PayloadListOp"},
          {CrateDataTypeId::CRATE_DATA_TYPE_TIME_CODE, "TimeCode"},
      });

  // List up `supports array` type.
  // TODO: Use compile-time `set`
  MAPBOX_ETERNAL_CONSTEXPR const auto arrmap =
      mapbox::eternal::map<CrateDataTypeId, bool>({
          {CrateDataTypeId::CRATE_DATA_TYPE_BOOL, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_UCHAR, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_INT, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_UINT, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_INT64, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_UINT64, true},

          {CrateDataTypeId::CRATE_DATA_TYPE_HALF, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_FLOAT, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE, true},

          {CrateDataTypeId::CRATE_DATA_TYPE_STRING, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_TOKEN, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_ASSET_PATH, true},

          {CrateDataTypeId::CRATE_DATA_TYPE_MATRIX2D, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_MATRIX3D, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_MATRIX4D, true},

          {CrateDataTypeId::CRATE_DATA_TYPE_QUATD, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_QUATF, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_QUATH, true},

          {CrateDataTypeId::CRATE_DATA_TYPE_VEC2D, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC2F, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC2H, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC2I, true},

          {CrateDataTypeId::CRATE_DATA_TYPE_VEC3D, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC3F, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC3H, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC3I, true},

          {CrateDataTypeId::CRATE_DATA_TYPE_VEC4D, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC4F, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC4H, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_VEC4I, true},
          {CrateDataTypeId::CRATE_DATA_TYPE_TIME_CODE, true},
      });

  if (type_id < 0) {
    return nonstd::make_unexpected("Unknown type id: " +
                                   std::to_string(type_id));
  }

  auto tyret = tymap.find(static_cast<CrateDataTypeId>(type_id));

  if (tyret == tymap.end()) {
    // Invalid or unsupported.
    return nonstd::make_unexpected("Unknown or unspported type id: " +
                                   std::to_string(type_id));
  }

  bool supports_array = arrmap.count(static_cast<CrateDataTypeId>(type_id));

  CrateDataType dst(tyret->second.data(), static_cast<CrateDataTypeId>(type_id),
                    supports_array);

  return std::move(dst);
}
#endif

std::string GetCrateDataTypeRepr(CrateDataType dty) {
  auto tyRet = GetCrateDataType(static_cast<int32_t>(dty.dtype_id));
  if (!tyRet) {
    return "[Invalid]";
  }

  const CrateDataType ty = tyRet.value();

  std::stringstream ss;
  ss << "CrateDataType: " << ty.name << "("
     << static_cast<uint32_t>(ty.dtype_id)
     << "), supports_array = " << ty.supports_array;
  return ss.str();
}

std::string GetCrateDataTypeName(int32_t type_id) {
  auto tyRet = GetCrateDataType(type_id);
  if (!tyRet) {
    return "[Invalid]";
  }

  const CrateDataType dty = tyRet.value();
  return dty.name;
}

std::string GetCrateDataTypeName(CrateDataTypeId did) {
  return GetCrateDataTypeName(static_cast<int32_t>(did));
}

// std::string CrateValue::GetTypeName() const { return value_.type_name(); }
// uint32_t CrateValue::GetTypeId() const { return value_.type_id(); }

}  // namespace crate
}  // namespace tinyusdz

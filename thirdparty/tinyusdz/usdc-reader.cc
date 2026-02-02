// SPDX-License-Identifier: Apache 2.0
// Copyright 2020 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// USDC(Crate) reader
//
// TODO:
//
// - [ ] Validate the existence of connection Paths(Connection) and target
// Paths(Relation)
// - [ ] GeomSubset
// - [ ] Refactor Variant related code.
//

#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "usdc-reader.hh"

#if !defined(TINYUSDZ_DISABLE_MODULE_USDC_READER)

#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "prim-types.hh"
#include "tinyusdz.hh"
#include "value-types.hh"
#if defined(__wasi__)
#else
#include <thread>
#endif

#include "crate-format.hh"
#include "crate-pprint.hh"
#include "crate-reader.hh"
#include "integerCoding.h"
#include "lz4-compression.hh"
#include "path-util.hh"
#include "pprinter.hh"
#include "prim-reconstruct.hh"
#include "str-util.hh"
#include "stream-reader.hh"
#include "tiny-format.hh"
#include "value-pprint.hh"
#include "usdShade.hh"
#include "ascii-parser.hh"

//
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "nonstd/expected.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

//
#include "common-macros.inc"

namespace tinyusdz {

namespace prim {

// template specialization forward decls.
// implimentations will be located in prim-reconstruct.cc
#define RECONSTRUCT_PRIM_DECL(__ty)                                      \
  template <>                                                            \
  bool ReconstructPrim<__ty>(const Specifier &spec, const PropertyMap &, const ReferenceList &, \
                             __ty *, std::string *, std::string *, const PrimReconstructOptions &)

RECONSTRUCT_PRIM_DECL(Xform);
RECONSTRUCT_PRIM_DECL(Model);
RECONSTRUCT_PRIM_DECL(Scope);
RECONSTRUCT_PRIM_DECL(GeomPoints);
RECONSTRUCT_PRIM_DECL(GeomMesh);
RECONSTRUCT_PRIM_DECL(GeomCapsule);
RECONSTRUCT_PRIM_DECL(GeomCube);
RECONSTRUCT_PRIM_DECL(GeomCone);
RECONSTRUCT_PRIM_DECL(GeomCylinder);
RECONSTRUCT_PRIM_DECL(GeomSphere);
RECONSTRUCT_PRIM_DECL(GeomSubset);
RECONSTRUCT_PRIM_DECL(GeomBasisCurves);
RECONSTRUCT_PRIM_DECL(GeomNurbsCurves);
RECONSTRUCT_PRIM_DECL(GeomCamera);
RECONSTRUCT_PRIM_DECL(PointInstancer);
RECONSTRUCT_PRIM_DECL(SphereLight);
RECONSTRUCT_PRIM_DECL(DomeLight);
RECONSTRUCT_PRIM_DECL(DiskLight);
RECONSTRUCT_PRIM_DECL(DistantLight);
RECONSTRUCT_PRIM_DECL(CylinderLight);
RECONSTRUCT_PRIM_DECL(SkelRoot);
RECONSTRUCT_PRIM_DECL(SkelAnimation);
RECONSTRUCT_PRIM_DECL(Skeleton);
RECONSTRUCT_PRIM_DECL(BlendShape);
RECONSTRUCT_PRIM_DECL(Material);
RECONSTRUCT_PRIM_DECL(Shader);

#undef RECONSTRUCT_PRIM_DECL

}  // namespace prim

namespace usdc {

constexpr auto kTag = "[USDC]";

// TODO: Unify with ascii-parser.cc
static void RegisterPrimAttrTypes(std::set<std::string> &d) {
  d.clear();

  d.insert(value::kBool);

  d.insert(value::kInt64);

  d.insert(value::kInt);
  d.insert(value::kInt2);
  d.insert(value::kInt3);
  d.insert(value::kInt4);

  d.insert(value::kUInt64);

  d.insert(value::kUInt);
  d.insert(value::kUInt2);
  d.insert(value::kUInt3);
  d.insert(value::kUInt4);

  d.insert(value::kFloat);
  d.insert(value::kFloat2);
  d.insert(value::kFloat3);
  d.insert(value::kFloat4);

  d.insert(value::kDouble);
  d.insert(value::kDouble2);
  d.insert(value::kDouble3);
  d.insert(value::kDouble4);

  d.insert(value::kHalf);
  d.insert(value::kHalf2);
  d.insert(value::kHalf3);
  d.insert(value::kHalf4);

  d.insert(value::kQuath);
  d.insert(value::kQuatf);
  d.insert(value::kQuatd);

  d.insert(value::kNormal3f);
  d.insert(value::kPoint3f);
  d.insert(value::kTexCoord2h);
  d.insert(value::kTexCoord3h);
  d.insert(value::kTexCoord4h);
  d.insert(value::kTexCoord2f);
  d.insert(value::kTexCoord3f);
  d.insert(value::kTexCoord4f);
  d.insert(value::kTexCoord2d);
  d.insert(value::kTexCoord3d);
  d.insert(value::kTexCoord4d);
  d.insert(value::kVector3f);
  d.insert(value::kVector4f);
  d.insert(value::kColor3h);
  d.insert(value::kColor3f);
  d.insert(value::kColor3d);
  d.insert(value::kColor4h);
  d.insert(value::kColor4f);
  d.insert(value::kColor4d);

  // Allow `matrixNf` type for USDC
  d.insert(value::kMatrix2f);
  d.insert(value::kMatrix3f);
  d.insert(value::kMatrix4f);

  d.insert(value::kMatrix2d);
  d.insert(value::kMatrix3d);
  d.insert(value::kMatrix4d);

  d.insert(value::kToken);
  d.insert(value::kString);

  d.insert(value::kRelationship);
  d.insert(value::kAssetPath);

  d.insert(value::kDictionary);

  // variantSet. Require special treatment.
  // d.insert("variantSet");

  // TODO: Add more types...
}


static bool IsUnregisteredValueType(const std::string &typeName)
{
  std::string tyname = typeName;

  //bool array_type = false;
  if (endsWith(typeName, "[]")) {
    tyname = removeSuffix(typeName, "[]");
    //array_type = true;
  }

  // TODO: Define in crate-format?
  if (tyname == value::TypeTraits<value::uint2>::type_name()) {
    return true;
  }
  if (tyname == value::TypeTraits<value::uint3>::type_name()) {
    return true;
  }
  if (tyname == value::TypeTraits<value::uint4>::type_name()) {
    return true;
  }

  return false;

}

class USDCReader::Impl {
 public:
  Impl(StreamReader *sr, const USDCReaderConfig &config) : _sr(sr) {
    set_reader_config(config);
    RegisterPrimAttrTypes(_supported_prim_attr_types);
  }

  ~Impl() {
    delete crate_reader;
    crate_reader = nullptr;
  }

  void set_reader_config(const USDCReaderConfig &config) {
    _config = config;

#if defined(__wasi__)
    _config.numThreads = 1;
#else
    if (_config.numThreads == -1) {
      _config.numThreads =
          (std::max)(1, int(std::thread::hardware_concurrency()));
    }
    // Limit to 1024 threads.
    _config.numThreads = (std::min)(1024, _config.numThreads);
#endif
  }

  const USDCReaderConfig get_reader_config() const {
    return _config;
  }

  bool ReadUSDC();

  using PathIndexToSpecIndexMap = std::unordered_map<uint32_t, uint32_t>;

  ///
  /// Construct Property(Attribute, Relationship/Connection) from
  /// FieldValuePairs
  ///
  bool ParseProperty(const SpecType specType,
                     const crate::FieldValuePairVector &fvs, Property *prop);

  ///
  /// Parse Prim spec from FieldValuePairs
  ///
  bool ParsePrimSpec(const crate::FieldValuePairVector &fvs,
                     nonstd::optional<std::string> &typeName, /* out */
                     nonstd::optional<Specifier> &specifier,  /* out */
                     std::vector<value::token> &primChildren,   /* out */
                     std::vector<value::token> &properties,   /* out */
                     PrimMeta &primMeta);                     /* out */

  bool ParseVariantSetFields(
      const crate::FieldValuePairVector &fvs,
      std::vector<value::token> &variantChildren); /* out */

  template <typename T>
  bool ReconstructPrim(const Specifier &spec, const crate::CrateReader::Node &node,
                       const PathIndexToSpecIndexMap &psmap, T *prim);

  ///
  /// Reconstrcut Prim node.
  /// Returns reconstruct Prim to `primOut`
  /// When `current` is 0(StageMeta), `primOut` is not set.
  /// `is_parent_variant` : True when parent path is Variant
  ///
  bool ReconstructPrimNode(int parent, int current, int level,
                           bool is_parent_variant,
                           const PathIndexToSpecIndexMap &psmap, Stage *stage,
                           nonstd::optional<Prim> *primOut);

  ///
  /// Reconstrcut PrimSpec node.
  /// Returns reconstruct PrimSpec to `primOut`
  /// When `current` is 0(StageMeta), `primOut` is not set.
  /// `is_parent_variant` : True when parent path is Variant
  ///
  /// TODO: Unify code with ReconstructPrimNode.
  ///
  bool ReconstructPrimSpecNode(int parent, int current, int level,
                           bool is_parent_variant,
                           const PathIndexToSpecIndexMap &psmap, Layer *layer,
                           nonstd::optional<PrimSpec> *primOut);

  ///
  /// Reconstruct Prim from given `typeName` string(e.g. "Xform")
  ///
  /// @param[out] is_unsupported_prim true when encounter Unsupported Prim type(and returns nullopt)
  ///
  nonstd::optional<Prim> ReconstructPrimFromTypeName(
      const std::string &typeName, const std::string &primTypeName, const std::string &prim_name,
      const crate::CrateReader::Node &node, const Specifier spec,
      const std::vector<value::token> &primChildren,
      const std::vector<value::token> &properties,
      const PathIndexToSpecIndexMap &psmap, const PrimMeta &meta, bool *is_unsupported_prim = nullptr);

  bool ReconstructPrimRecursively(int parent_id, int current_id, Prim *rootPrim,
                                  int level,
                                  const PathIndexToSpecIndexMap &psmap,
                                  Stage *stage);

  //bool ReconstructPrimTree(Prim *rootPrim, const PathIndexToSpecIndexMap &psmap,
  //                         Stage *stage);

  bool ReconstructStage(Stage *stage);

  ///
  /// For Layer
  ///

  bool ReconstructPrimSpecRecursively(int parent_id, int current_id, PrimSpec *rootPrim,
                                  int level,
                                  const PathIndexToSpecIndexMap &psmap,
                                  Layer *stage);

  //bool ReconstructPrimSpecTree(PrimSpec *rootPrim, const PathIndexToSpecIndexMap &psmap,
  //                         Layer *layer);

  bool ToLayer(Layer *layer);

  ///
  /// --------------------------------------------------
  ///

  void PushError(const std::string &s) { _err = s + _err; }

  void PushWarn(const std::string &s) { _warn = s + _warn; }

  std::string GetError() { return _err; }

  std::string GetWarning() { return _warn; }

  // Approximated memory usage in [mb]
  size_t GetMemoryUsage() const { return memory_used / (1024 * 1024); }

 private:
  nonstd::expected<APISchemas, std::string> ToAPISchemas(
      const ListOp<value::token> &, bool ignore_unknown, std::string &warn);

  // ListOp<T> -> (ListEditOp, [T])
  template <typename T>
  std::vector<std::pair<ListEditQual, std::vector<T>>> DecodeListOp(
      const ListOp<T> &);

  ///
  /// Builds std::map<std::string, Property> from the list of Path(Spec)
  /// indices.
  ///
  bool BuildPropertyMap(const std::vector<size_t> &pathIndices,
                        const PathIndexToSpecIndexMap &psmap,
                        prim::PropertyMap *props);

  bool ReconstrcutStageMeta(const crate::FieldValuePairVector &fvs,
                            StageMetas *out);

  bool AddVariantChildrenToPrimNode(
      int32_t prim_idx, const std::vector<value::token> &variantChildren) {
    if (prim_idx < 0) {
      return false;
    }

    if (_variantChildren.count(uint32_t(prim_idx))) {
      PUSH_WARN("Multiple Field with VariantSet SpecType detected.");
    }

    _variantChildren[uint32_t(prim_idx)] = variantChildren;

    return true;
  }

  bool AddVariantToPrimNode(int32_t prim_idx, const value::Value &variant);

  crate::CrateReader *crate_reader{nullptr};

  StreamReader *_sr = nullptr;
  std::string _err;
  std::string _warn;

  USDCReaderConfig _config;

  // Tracks the memory used(In advisorily manner since counting memory usage is
  // done by manually, so not all memory consumption could be tracked)
  size_t memory_used{0};  // in bytes.

  nonstd::optional<Path> GetPath(crate::Index index) const {
    if (index.value < _paths.size()) {
      return _paths[index.value];
    }

    return nonstd::nullopt;
  }

  nonstd::optional<Path> GetElemPath(crate::Index index) const {
    if (index.value < _elemPaths.size()) {
      return _elemPaths[index.value];
    }

    return nonstd::nullopt;
  }

  // TODO: Do not copy data from crate_reader.
  std::vector<crate::CrateReader::Node> _nodes;
  std::vector<crate::Spec> _specs;
  std::vector<crate::Field> _fields;
  std::vector<crate::Index> _fieldset_indices;
  std::vector<crate::Index> _string_indices;
  std::vector<Path> _paths;
  std::vector<Path> _elemPaths;

  std::map<crate::Index, crate::FieldValuePairVector>
      _live_fieldsets;  // <fieldset index, List of field with unpacked Values>

  // std::vector<PrimNode> _prim_nodes;

  // VariantSet Spec. variantChildren
  std::map<uint32_t, std::vector<value::token>> _variantChildren;

  // For Prim/Props defined as Variant(SpecType::VariantSet)
  // key = path index.
  std::map<int32_t, Prim> _variantPrims; // For Stage
  std::map<int32_t, PrimSpec> _variantPrimSpecs; // For Layer
  std::map<int32_t, std::pair<Path, Property>> _variantProps;
  std::map<int32_t, Variant> _variants;

  // key = parent path index, values = key to `_variantPrims`, `_variantProps`
  std::map<int32_t, std::vector<int32_t>> _variantPrimChildren;
  std::map<int32_t, std::vector<int32_t>> _variantPropChildren;

  // Check if given node_id is a prim node.
  std::set<int32_t> _prim_table;

  std::set<std::string> _supported_prim_attr_types;
};

//
// -- Impl
//

#if 0

bool USDCReader::Impl::ReconstructGeomSubset(
    const Node &node, const FieldValuePairVector &fields,
    const std::unordered_map<uint32_t, uint32_t> &path_index_to_spec_index_map,
    GeomSubset *geom_subset) {

  DCOUT("Reconstruct GeomSubset");

  for (const auto &fv : fields) {
    if (fv.first == "properties") {
      FIELDVALUE_DATATYPE_CHECK(fv, "properties", crate::kTokenVector)

      // for (size_t i = 0; i < fv.second.GetStringArray().size(); i++) {
      //   // if (fv.second.GetStringArray()[i] == "points") {
      //   // }
      // }
    }
  }

  for (size_t i = 0; i < node.GetChildren().size(); i++) {
    int child_index = int(node.GetChildren()[i]);
    if ((child_index < 0) || (child_index >= int(_nodes.size()))) {
      PUSH_ERROR("Invalid child node id: " + std::to_string(child_index) +
                 ". Must be in range [0, " + std::to_string(_nodes.size()) +
                 ")");
      return false;
    }

    // const Node &child_node = _nodes[size_t(child_index)];

    if (!path_index_to_spec_index_map.count(uint32_t(child_index))) {
      // No specifier assigned to this child node.
      // TODO: Should we report an error?
      continue;
    }

    uint32_t spec_index =
        path_index_to_spec_index_map.at(uint32_t(child_index));
    if (spec_index >= _specs.size()) {
      PUSH_ERROR("Invalid specifier id: " + std::to_string(spec_index) +
                 ". Must be in range [0, " + std::to_string(_specs.size()) +
                 ")");
      return false;
    }

    const crate::Spec &spec = _specs[spec_index];

    Path path = GetPath(spec.path_index);
    DCOUT("Path prim part: " << path.prim_part()
                             << ", prop part: " << path.prop_part()
                             << ", spec_index = " << spec_index);

    if (!_live_fieldsets.count(spec.fieldset_index)) {
      _err += "FieldSet id: " + std::to_string(spec.fieldset_index.value) +
              " must exist in live fieldsets.\n";
      return false;
    }

    const FieldValuePairVector &child_fields =
        _live_fieldsets.at(spec.fieldset_index);

    {
      std::string prop_name = path.prop_part();

      Attribute attr;
      bool ret = ParseAttribute(child_fields, &attr, prop_name);
      DCOUT("prop: " << prop_name << ", ret = " << ret);

      if (ret) {
        // TODO(syoyo): Support more prop names
        if (prop_name == "elementType") {
          auto p = attr.var.get_value<tinyusdz::value::token>();
          if (p) {
            std::string str = p->str();
            if (str == "face") {
              geom_subset->elementType = GeomSubset::ElementType::Face;
            } else {
              PUSH_ERROR("`elementType` must be `face`, but got `" + str + "`");
              return false;
            }
          } else {
            PUSH_ERROR("`elementType` must be token type, but got " +
                       value::GetTypeName(attr.var.type_id()));
            return false;
          }
        } else if (prop_name == "faces") {
          auto p = attr.var.get_value<std::vector<int>>();
          if (p) {
            geom_subset->faces = (*p);
          }

          DCOUT("faces.num = " << geom_subset->faces.size());

        } else {
          // Assume Primvar.
          if (geom_subset->attribs.count(prop_name)) {
            _err += "Duplicated property name found: " + prop_name + "\n";
            return false;
          }

#ifdef TINYUSDZ_LOCAL_DEBUG_PRINT
          std::cout << "add [" << prop_name << "] to generic attrs\n";
#endif

          geom_subset->attribs[prop_name] = std::move(attr);
        }
      }
    }
  }

  return true;
}

#endif

namespace {}

nonstd::expected<APISchemas, std::string> USDCReader::Impl::ToAPISchemas(
    const ListOp<value::token> &arg, bool ignore_unknown, std::string &warn) {
  APISchemas schemas;

  auto SchemaHandler =
      [](const value::token &tok) -> nonstd::optional<APISchemas::APIName> {
    if (tok.str() == "MaterialBindingAPI") {
      return APISchemas::APIName::MaterialBindingAPI;
    } else if (tok.str() == "NodeDefAPI") {
      return APISchemas::APIName::NodeDefAPI;
    } else if (tok.str() == "CoordSysAPI") {
      return APISchemas::APIName::CoordSysAPI;
    } else if (tok.str() == "ConnectableAPI") {
      return APISchemas::APIName::ConnectableAPI;
    } else if (tok.str() == "CollectionAPI") {
      return APISchemas::APIName::CollectionAPI;
    } else if (tok.str() == "SkelBindingAPI") {
      return APISchemas::APIName::SkelBindingAPI;
    } else if (tok.str() == "VisibilityAPI") {
      return APISchemas::APIName::VisibilityAPI;
    } else if (tok.str() == "GeomModelAPI") {
      return APISchemas::APIName::GeomModelAPI;
    } else if (tok.str() == "MotionAPI") {
      return APISchemas::APIName::MotionAPI;
    } else if (tok.str() == "PrimvarsAPI") {
      return APISchemas::APIName::PrimvarsAPI;
    } else if (tok.str() == "XformCommonAPI") {
      return APISchemas::APIName::XformCommonAPI;
    } else if (tok.str() == "ListAPI") {
      return APISchemas::APIName::ListAPI;
    } else if (tok.str() == "LightListAPI") {
      return APISchemas::APIName::LightListAPI;
    } else if (tok.str() == "LightAPI") {
      return APISchemas::APIName::LightAPI;
    } else if (tok.str() == "MeshLightAPI") {
      return APISchemas::APIName::MeshLightAPI;
    } else if (tok.str() == "VolumeLightAPI") {
      return APISchemas::APIName::VolumeLightAPI;
    } else if (tok.str() == "ConnectableAPI") {
      return APISchemas::APIName::ConnectableAPI;
    } else if (tok.str() == "ShadowAPI") {
      return APISchemas::APIName::ShadowAPI;
    } else if (tok.str() == "ShapingAPI") {
      return APISchemas::APIName::ShapingAPI;
    } else if (tok.str() == "Preliminary_AnchoringAPI") {
      return APISchemas::APIName::Preliminary_AnchoringAPI;
    } else if (tok.str() == "Preliminary_PhysicsColliderAPI") {
      return APISchemas::APIName::Preliminary_PhysicsColliderAPI;
    } else if (tok.str() == "Preliminary_PhysicsMaterialAPI") {
      return APISchemas::APIName::Preliminary_PhysicsMaterialAPI;
    } else if (tok.str() == "Preliminary_PhysicsRigidBodyAPI") {
      return APISchemas::APIName::Preliminary_PhysicsRigidBodyAPI;
    } else {
      return nonstd::nullopt;
    }
  };

  if (arg.IsExplicit()) {  // fast path
    for (auto &item : arg.GetExplicitItems()) {
      if (auto pv = SchemaHandler(item)) {
        std::string instanceName = "";  // TODO
        schemas.names.push_back({pv.value(), instanceName});
      } else if (ignore_unknown) {
        warn += "Ignored unknown or unsupported API schema: " +
                                       item.str() + "\n";
      } else {
        return nonstd::make_unexpected("Invalid or Unsupported API schema: " +
                                       item.str());
      }
    }
    schemas.listOpQual = ListEditQual::ResetToExplicit;

  } else {
    // Assume all items have same ListEdit qualifier.
    if (arg.GetExplicitItems().size()) {
      if (arg.GetAddedItems().size() || arg.GetAppendedItems().size() ||
          arg.GetDeletedItems().size() || arg.GetPrependedItems().size() ||
          arg.GetOrderedItems().size()) {
        return nonstd::make_unexpected(
            "Currently TinyUSDZ does not support ListOp with different "
            "ListEdit qualifiers.");
      }
      for (auto &&item : arg.GetExplicitItems()) {
        if (auto pv = SchemaHandler(item)) {
          std::string instanceName = "";  // TODO
          schemas.names.push_back({pv.value(), instanceName});
        } else if (ignore_unknown) {
          warn += "Ignored unknown or unsupported API schema: " +
                                         item.str() + "\n";
        } else {
          return nonstd::make_unexpected("Invalid or Unsupported API schema: " +
                                         item.str());
        }
      }
      schemas.listOpQual = ListEditQual::ResetToExplicit;

    } else if (arg.GetAddedItems().size()) {
      if (arg.GetExplicitItems().size() || arg.GetAppendedItems().size() ||
          arg.GetDeletedItems().size() || arg.GetPrependedItems().size() ||
          arg.GetOrderedItems().size()) {
        return nonstd::make_unexpected(
            "Currently TinyUSDZ does not support ListOp with different "
            "ListEdit qualifiers.");
      }
      for (auto &item : arg.GetAddedItems()) {
        if (auto pv = SchemaHandler(item)) {
          std::string instanceName = "";  // TODO
          schemas.names.push_back({pv.value(), instanceName});
        } else if (ignore_unknown) {
          warn += "Ignored unknown or unsupported API schema: " +
                                         item.str() + "\n";
        } else {
          return nonstd::make_unexpected("Invalid or Unsupported API schema: " +
                                         item.str());
        }
      }
      schemas.listOpQual = ListEditQual::Add;
    } else if (arg.GetAppendedItems().size()) {
      if (arg.GetExplicitItems().size() || arg.GetAddedItems().size() ||
          arg.GetDeletedItems().size() || arg.GetPrependedItems().size() ||
          arg.GetOrderedItems().size()) {
        return nonstd::make_unexpected(
            "Currently TinyUSDZ does not support ListOp with different "
            "ListEdit qualifiers.");
      }
      for (auto &&item : arg.GetAppendedItems()) {
        if (auto pv = SchemaHandler(item)) {
          std::string instanceName = "";  // TODO
          schemas.names.push_back({pv.value(), instanceName});
        } else if (ignore_unknown) {
          warn += "Ignored unknown or unsupported API schema: " +
                                         item.str() + "\n";
        } else {
          return nonstd::make_unexpected("Invalid or Unsupported API schema: " +
                                         item.str());
        }
      }
      schemas.listOpQual = ListEditQual::Append;
    } else if (arg.GetDeletedItems().size()) {
      if (arg.GetExplicitItems().size() || arg.GetAddedItems().size() ||
          arg.GetAppendedItems().size() || arg.GetPrependedItems().size() ||
          arg.GetOrderedItems().size()) {
        return nonstd::make_unexpected(
            "Currently TinyUSDZ does not support ListOp with different "
            "ListEdit qualifiers.");
      }
      for (auto &&item : arg.GetDeletedItems()) {
        if (auto pv = SchemaHandler(item)) {
          std::string instanceName = "";  // TODO
          schemas.names.push_back({pv.value(), instanceName});
        } else if (ignore_unknown) {
          warn += "Ignored unknown or unsupported API schema: " +
                                         item.str() + "\n";
        } else {
          return nonstd::make_unexpected("Invalid or Unsupported API schema: " +
                                         item.str());
        }
      }
      schemas.listOpQual = ListEditQual::Delete;
    } else if (arg.GetPrependedItems().size()) {
      if (arg.GetExplicitItems().size() || arg.GetAddedItems().size() ||
          arg.GetAppendedItems().size() || arg.GetDeletedItems().size() ||
          arg.GetOrderedItems().size()) {
        return nonstd::make_unexpected(
            "Currently TinyUSDZ does not support ListOp with different "
            "ListEdit qualifiers.");
      }
      for (auto &&item : arg.GetPrependedItems()) {
        if (auto pv = SchemaHandler(item)) {
          std::string instanceName = "";  // TODO
          schemas.names.push_back({pv.value(), instanceName});
        } else if (ignore_unknown) {
          warn += "Ignored unknown or unsupported API schema: " +
                                         item.str() + "\n";
        } else {
          return nonstd::make_unexpected("Invalid or Unsupported API schema: " +
                                         item.str());
        }
      }
      schemas.listOpQual = ListEditQual::Prepend;
    } else if (arg.GetOrderedItems().size()) {
      if (arg.GetExplicitItems().size() || arg.GetAddedItems().size() ||
          arg.GetAppendedItems().size() || arg.GetDeletedItems().size() ||
          arg.GetPrependedItems().size()) {
        return nonstd::make_unexpected(
            "Currently TinyUSDZ does not support ListOp with different "
            "ListEdit qualifiers.");
      }

      // schemas.qual = ListEditQual::Order;
      return nonstd::make_unexpected("TODO: Ordered ListOp items.");
    } else {
      // ??? This should not happend.
      return nonstd::make_unexpected("Internal error: ListOp conversion.");
    }
  }

  return std::move(schemas);
}

template <typename T>
std::vector<std::pair<ListEditQual, std::vector<T>>>
USDCReader::Impl::DecodeListOp(const ListOp<T> &arg) {
  std::vector<std::pair<ListEditQual, std::vector<T>>> dst;

  if (arg.IsExplicit()) {  // fast path
    dst.push_back({ListEditQual::ResetToExplicit, arg.GetExplicitItems()});
  } else {
    // Assume all items have same ListEdit qualifier.
    if (arg.GetExplicitItems().size()) {
      dst.push_back({ListEditQual::ResetToExplicit, arg.GetExplicitItems()});
    }
    if (arg.GetAddedItems().size()) {
      dst.push_back({ListEditQual::Add, arg.GetAddedItems()});
    }
    if (arg.GetAppendedItems().size()) {
      dst.push_back({ListEditQual::Append, arg.GetAppendedItems()});
    }
    if (arg.GetDeletedItems().size()) {
      dst.push_back({ListEditQual::Delete, arg.GetDeletedItems()});
    }
    if (arg.GetPrependedItems().size()) {
      dst.push_back({ListEditQual::Prepend, arg.GetPrependedItems()});
    }
    if (arg.GetOrderedItems().size()) {
      dst.push_back({ListEditQual::Order, arg.GetOrderedItems()});
    }
  }

  return dst;
}

bool USDCReader::Impl::BuildPropertyMap(const std::vector<size_t> &pathIndices,
                                        const PathIndexToSpecIndexMap &psmap,
                                        prim::PropertyMap *props) {
  for (size_t i = 0; i < pathIndices.size(); i++) {
    int child_index = int(pathIndices[i]);
    if ((child_index < 0) || (child_index >= int(_nodes.size()))) {
      PUSH_ERROR("Invalid child node id: " + std::to_string(child_index) +
                 ". Must be in range [0, " + std::to_string(_nodes.size()) +
                 ")");
      return false;
    }

    if (!psmap.count(uint32_t(child_index))) {
      // No specifier assigned to this child node.
      // Should we report an error?
      continue;
    }

    uint32_t spec_index = psmap.at(uint32_t(child_index));
    if (spec_index >= _specs.size()) {
      PUSH_ERROR("Invalid specifier id: " + std::to_string(spec_index) +
                 ". Must be in range [0, " + std::to_string(_specs.size()) +
                 ")");
      return false;
    }

    const crate::Spec &spec = _specs[spec_index];

    // Property must be Attribute or Relationship
    if ((spec.spec_type == SpecType::Attribute) ||
        (spec.spec_type == SpecType::Relationship)) {
      // OK
    } else {
      continue;
    }

    nonstd::optional<Path> path = GetPath(spec.path_index);

    if (!path) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid PathIndex.");
    }

    DCOUT("Path prim part: " << path.value().prim_part()
                             << ", prop part: " << path.value().prop_part()
                             << ", spec_index = " << spec_index);

    if (!_live_fieldsets.count(spec.fieldset_index)) {
      PUSH_ERROR("FieldSet id: " + std::to_string(spec.fieldset_index.value) +
                 " must exist in live fieldsets.");
      return false;
    }

    const crate::FieldValuePairVector &child_fvs =
        _live_fieldsets.at(spec.fieldset_index);

    {
      std::string prop_name = path.value().prop_part();
      if (prop_name.empty()) {
        DCOUT("path = " << dump_path(path.value()));
        // ???
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Property Prop.PropPart is empty");
      }

      std::string prop_err;
      if (!pathutil::ValidatePropPath(Path("", prop_name), &prop_err)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Invalid Property name `{}`: {}", prop_name, prop_err));
      }

      Property prop;
      if (!ParseProperty(spec.spec_type, child_fvs, &prop)) {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag,
            fmt::format(
                "Failed to construct Property `{}` from FieldValuePairVector.",
                prop_name));
      }

      (*props)[prop_name] = prop;
      DCOUT("Add property : " << prop_name);
    }
  }

  return true;
}


/// Property fieldSet example
///
///   specTyppe = SpecTypeAttribute
///
///     - typeName(token) : type name of Attribute(e.g. `float`)
///     - custom(bool) : `custom` qualifier
///     - variability(variability) : Variability(meta?)
///     <value>
///       - default : Default(fallback) value.
///       - timeSample(TimeSamples) : `.timeSamples` data.
///       - connectionPaths(type = ListOpPath) : `.connect`
///       - (Empty) : Define only(Neiher connection nor value assigned. e.g.
///       "float outputs:rgb")
bool USDCReader::Impl::ParseProperty(const SpecType spec_type,
                                     const crate::FieldValuePairVector &fvs,
                                     Property *prop) {
  if (fvs.size() > _config.kMaxFieldValuePairs) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too much FieldValue pairs.");
  }

  if (!prop) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Internal error. prop is nullptr.");
  }

  bool custom{false};
  nonstd::optional<value::token> typeName;
  nonstd::optional<Interpolation> interpolation;
  nonstd::optional<int> elementSize;
  nonstd::optional<bool> hidden;
  nonstd::optional<CustomDataType> customData;
  nonstd::optional<double> weight;
  nonstd::optional<value::token> bindMaterialAs;
  nonstd::optional<value::token> connectability;
  nonstd::optional<value::token> renderType;
  nonstd::optional<value::token> outputName;
  nonstd::optional<CustomDataType> sdrMetadata;
  nonstd::optional<value::StringData> comment;
  nonstd::optional<Variability> variability;
  AttrMeta meta; // for other not frequently-used attribute/relationship metadata.
  //Property::Type propType{Property::Type::EmptyAttrib};
  Attribute attr;

  value::Value defaultValue;
  Relationship rel;

  // for attribute
  bool isValueBlock{false};
  bool hasDefault{false};
  bool hasTimeSamples{false};
  bool hasConnectionPaths{false};

  // for relationship
  bool hasTargetPaths{false};

  // metadata(ignore these for now)
  bool hasConnectionChildren{false};
  bool hasTargetChildren{false};

  DCOUT("== List of Fields");

  primvar::PrimVar var;

  // first detect typeName
  for (auto &fv : fvs) {
    if (fv.first == "typeName") {
      if (auto pv = fv.second.get_value<value::token>()) {
        DCOUT("  typeName = " << pv.value().str());
        typeName = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`typeName` field is not `token` type.");
      }
    }
  }

  if (typeName) { // this should be always true though.
    attr.set_type_name(typeName.value().str());
  }

  for (auto &fv : fvs) {
    DCOUT(" fv name " << fv.first << "(type = " << fv.second.type_name()
                      << ")");

    if (fv.first == "custom") {
      if (auto pv = fv.second.get_value<bool>()) {
        custom = pv.value();
        DCOUT("  custom = " << pv.value());
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "`custom` field is not `bool` type.");
      }
    } else if (fv.first == "variability") {
      if (auto pv = fv.second.get_value<Variability>()) {
        variability = pv.value();
        DCOUT("  variability = " << to_string(variability.value()));
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`variability` field is not `varibility` type.");
      }
    } else if (fv.first == "typeName") {
      // 'typeName' is already processed. nothing to do here.
      continue;
    } else if (fv.first == "default") {
      //propType = Property::Type::Attrib;

      // Set scalar(non-timesampled) value
      // TODO: Easier CrateValue to Attribute.var conversion
      defaultValue = fv.second.get_raw();
      hasDefault = true;

      // TODO: Handle UnregisteredValue in crate-reader.cc
      // UnregisteredValue is represented as string.
      if (const auto pv = defaultValue.get_value<std::string>()) {
        if (typeName && (typeName.value().str() != "string")) {
          if (IsUnregisteredValueType(typeName.value().str())) {
            DCOUT("UnregisteredValue type: " << typeName.value().str());

            std::string local_err;
            if (!ascii::ParseUnregistredValue(typeName.value().str(), pv.value(), &defaultValue, &local_err)) {
              PUSH_ERROR_AND_RETURN(fmt::format("Failed to parse UnregisteredValue string with type `{}`: {}", typeName.value().str(), local_err));
            }
          }
        }
      }

    } else if (fv.first == "timeSamples") {
      //propType = Property::Type::Attrib;
      

      hasTimeSamples = true;

      if (auto pv = fv.second.get_value<value::TimeSamples>()) {
        var.set_timesamples(pv.value());
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`timeSamples` is not TimeSamples data.");
      }
    } else if (fv.first == "interpolation") {
      //propType = Property::Type::Attrib;

      if (auto pv = fv.second.get_value<value::token>()) {
        DCOUT("  interpolation = " << pv.value().str());

        if (auto interp = InterpolationFromString(pv.value().str())) {
          interpolation = interp.value();
        } else {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid token for `interpolation`.");
        }
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`interpolation` field is not `token` type.");
      }
    } else if (fv.first == "connectionPaths") {
      // Attribute connection(.connect)
      //propType = Property::Type::Connection;
      hasConnectionPaths = true;

      if (auto pv = fv.second.get_value<ListOp<Path>>()) {
        auto p = pv.value();
        DCOUT("connectionPaths = " << to_string(p));

        if (!p.IsExplicit()) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, "`connectionPaths` must be composed of Explicit items.");
        }

        // Must be explicit_items for now.
        auto items = p.GetExplicitItems();
        if (items.size() == 0) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, "`connectionPaths` have empty Explicit items.");
        }

        attr.set_connections(items); 

      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`connectionPaths` field is not `ListOp[Path]` type.");
      }
    } else if (fv.first == "targetPaths") {
      // `rel`
      //propType = Property::Type::Relation;
      hasTargetPaths = true;

      if (auto pv = fv.second.get_value<ListOp<Path>>()) {
        const ListOp<Path> &p = pv.value();
        DCOUT("targetPaths = " << to_string(p));

        auto ps = DecodeListOp<Path>(p);

        if (ps.empty()) {
          // Empty `targetPaths`
          PUSH_ERROR_AND_RETURN_TAG(kTag, "`targetPaths` is empty.");
        }

        if (ps.size() > 1) {
          // This should not happen though.
          PUSH_WARN(
              "ListOp with multiple ListOpType is not supported for now. Use "
              "the first one: " +
              to_string(std::get<0>(ps[0])));
        }

        auto qual = std::get<0>(ps[0]);
        auto items = std::get<1>(ps[0]);

        if (items.size() == 1) {
          // Single
          const Path path = items[0];

          rel.set(path);

        } else {
          rel.set(items);  // [Path]
        }

        rel.set_listedit_qual(qual);

      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`targetPaths` field is not `ListOp[Path]` type.");
      }

    } else if (fv.first == "hidden") {
      // Attribute hidden param
      if (auto pv = fv.second.get_value<bool>()) {
        auto p = pv.value();
        DCOUT("hidden = " << to_string(p));

        hidden = p;

      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`elementSize` field is not `int` type.");
      }
    } else if (fv.first == "elementSize") {
      // Attribute Meta
      if (auto pv = fv.second.get_value<int>()) {
        auto p = pv.value();
        DCOUT("elementSize = " << to_string(p));

        if ((p < 1) || (uint32_t(p) > _config.kMaxElementSize)) {
          PUSH_WARN(
              fmt::format("`elementSize` too large. Must be within [{}, {}), but got {}",
                          1, _config.kMaxElementSize, p));
        }

        elementSize = p;

      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`elementSize` field is not `int` type.");
      }
    } else if (fv.first == "weight") {
      // pxrUSD uses float type.
      if (auto pv = fv.second.get_value<float>()) {
        auto p = pv.value();
        DCOUT("weight = " << p);

        weight = double(p);

      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`weight` field is not `float` type.");
      }
    } else if (fv.first == "bindMaterialAs") {
      // Attribute Meta
      if (auto pv = fv.second.get_value<value::token>()) {
        auto p = pv.value();
        DCOUT("bindMaterialAs = " << to_string(p));

        if ((p.str() == kWeaderThanDescendants) || (p.str() == kStrongerThanDescendants)) {
          // ok
        } else {
          // still any token is valid(for future usecase)
          PUSH_WARN("Unsupported bindMaterialAs token: " << p.str());
        }
        bindMaterialAs = p;
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`bindMaterialAs` field is not `token` type.");
      }
    } else if (fv.first == "targetChildren") {
      // `targetChildren` seems optionally exist to validate the existence of
      // target Paths when `targetPaths` field exists.
      // TODO: validate path of `targetChildren`
      hasTargetChildren = true;

      // Path vector
      if (auto pv = fv.second.get_value<std::vector<Path>>()) {
        DCOUT("targetChildren = " << pv.value());
        // PUSH_WARN("TODO: targetChildren");

      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`targetChildren` field is not `PathVector` type.");
      }
    } else if (fv.first == "connectionChildren") {
      // `connectionChildren` seems optionally exist to validate the existence
      // of connection Paths when `connectiontPaths` field exists.
      // TODO: validate path of `connetionChildren`
      hasConnectionChildren = true;

      // Path vector
      if (auto pv = fv.second.get_value<std::vector<Path>>()) {
        DCOUT("connectionChildren = " << pv.value());
        // PUSH_WARN("TODO: connectionChildren");
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`connectionChildren` field is not `PathVector` type.");
      }
    } else if (fv.first == "connectability") {
      if (auto pv = fv.second.get_value<value::token>()) {
        connectability = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`connectability` must be type `token`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "outputName") {
      if (auto pv = fv.second.get_value<value::token>()) {
        outputName = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`outputName` must be type `token`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "renderType") {
      if (auto pv = fv.second.get_value<value::token>()) {
        renderType = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`renderType` must be type `token`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "sdrMetadata") {
      if (auto pv = fv.second.get_value<CustomDataType>()) {
        sdrMetadata = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`sdrMetadata` must be type `dictionary`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "customData") {
      // CustomData(dict)
      if (auto pv = fv.second.get_value<CustomDataType>()) {
        customData = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`customData` must be type `dictionary`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "comment") {
      if (auto pv = fv.second.get_value<std::string>()) {
        value::StringData s;
        s.value = pv.value();
        s.is_triple_quoted = hasNewline(s.value);
        comment = s;
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`comment` must be type `string`, but got type `"
                      << fv.second.type_name() << "`");
      }

    } else if (fv.first == "colorSpace") {
      if (auto pv = fv.second.get_value<value::token>()) {
        
        MetaVariable mv;
        mv.set_name("colorSpace");
        mv.set_value(pv.value());

        meta.meta["colorSpace"] = mv;
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`colorSpace` must be type `token`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "displayName") {
      if (auto pv = fv.second.get_value<std::string>()) {
        meta.displayName = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`displayName` must be type `string`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "displayGroup") {
      if (auto pv = fv.second.get_value<std::string>()) {
        meta.displayGroup = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`displayGroup` must be type `string`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "unauthoredValuesIndex") {
      if (auto pv = fv.second.get_value<int>()) {
        MetaVariable mv;
        mv.set_name("unauthoredValuesIndex");
        mv.set_value(pv.value());

        meta.meta["unauthoredValuesIndex"] = mv;
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`unauthoredValuesIndex` must be type `int`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else {
      // TODO: register unkown metadataum as custom metadata?
      PUSH_WARN("TODO: " << fv.first);
      DCOUT("TODO: " << fv.first);
    }
  }
  DCOUT("== End List of Fields");

  // Post check
#if 0
  if (hasConnectionChildren) {
    // Validate existence of Path..
  }

  if (hasTargetChildren) {
    // Validate existence of Path..
  }
#else
  (void)hasTargetChildren;
  (void)hasConnectionChildren;
  (void)hasConnectionPaths;
#endif

  // Do role type cast for default value.
  // (TODO: do role type cast for timeSamples?)
  if (hasDefault) {
    if (typeName) {
      if (defaultValue.type_id() == value::TypeTraits<value::ValueBlock>::type_id()) {
        // nothing to do
      } else {
        std::string reqTy = typeName.value().str();
        std::string scalarTy = defaultValue.type_name();

        if (reqTy.compare(scalarTy) != 0) {

          // Some inlined? value uses less accuracy type(e.g. `half3`) than
          // typeName(e.g. `float3`) Use type specified in `typeName` as much as
          // possible.
          bool ret = value::UpcastType(reqTy, defaultValue);
          if (ret) {
            DCOUT(fmt::format("Upcast type from {} to {}.", scalarTy, reqTy));
          }

          // Optionally, cast to role type(in crate data, `typeName` uses role typename(e.g. `color3f`), whereas stored data uses base typename(e.g. VEC3F)
          scalarTy = defaultValue.type_name();
          if (value::RoleTypeCast(value::GetTypeId(reqTy), defaultValue)) {
            DCOUT(fmt::format("Casted to Role type {} from type {}.", reqTy, scalarTy));
          } else {
            // Its ok.
          }
        }
      }
    }
    var.set_value(defaultValue);

    if (defaultValue.type_id() == value::TypeTraits<value::ValueBlock>::type_id()) {
      isValueBlock = true;
    }
  }

  attr.set_var(std::move(var));

  if (isValueBlock) {
    // attr's type is replaced with ValueBlock type  by `set_var`, so overwrite type with typeName
    if (typeName) {
      // Use `typeName`
      attr.set_type_name(typeName.value().str());
    }
  }

  // Attribute metas
  {
    if (interpolation) {
      meta.interpolation = interpolation.value();
    }
    if (elementSize) {
      meta.elementSize = elementSize.value();
    }
    if (hidden) {
      meta.hidden = hidden.value();
    }
    if (customData) {
      meta.customData = customData.value();
    }
    if (weight) {
      meta.weight = weight.value();
    }
    if (comment) {
      meta.comment = comment.value();
    }
    if (bindMaterialAs) {
      meta.bindMaterialAs = bindMaterialAs.value();
    }
    if (outputName) {
      meta.outputName = outputName.value();
    }
    if (sdrMetadata) {
      meta.sdrMetadata = sdrMetadata.value();
    }
    if (connectability) {
      meta.connectability = connectability.value();
    }
    if (renderType) {
      meta.renderType = renderType.value();
    }
  }



  if (hasTargetPaths) {
    // Relationship

    // TODO: Report as error?
    if (hasDefault) {
      PUSH_WARN("Relationship property has `default` field. Ignore `default` field.");
    }

    if (hasTimeSamples) {
      PUSH_WARN("Relationship property has `timeSamples` field. Ignore `timeSamples` field.");
    }

    if (hasConnectionPaths) {
      PUSH_WARN("Relationship property has `connectionPaths` field. Ignore `connectionPaths` field.");
    }

    if (variability) {
      if (variability.value() == Variability::Varying) {
        rel.set_varying_authored();
      }
    }
    rel.metas() = meta;
    (*prop) = Property(rel, custom);
  } else if (hasDefault || hasTimeSamples || hasConnectionPaths) {

    // Attribute
    if (hasTargetPaths) {
      PUSH_WARN("Attribute property has `targetPaths` field. Ignore `targetPaths` field.");
    }

    if (variability) {
      attr.variability() = variability.value();
    }
    attr.metas() = meta;
    (*prop) = Property(attr, custom);

  } else {

    // FIXME: SpecType supercedes propType.

    if (typeName) {
      // declare only attribute, e.g.: float myval
      // typeName may be array type.
      std::string baseTypeName = typeName.value().str();
      if (endsWith(baseTypeName, "[]")) {
        baseTypeName = removeSuffix(baseTypeName, "[]");
      }

      // Assume Attribute
      if (!_supported_prim_attr_types.count(baseTypeName)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Invalid or unsupported `typeName` {}", typeName.value()));
      }

      Property p;
      p.set_property_type(Property::Type::EmptyAttrib);
      p.attribute().set_type_name(typeName.value().str());
      p.set_custom(custom);

      if (variability) {
        p.attribute().variability() = variability.value();
      }
      p.attribute().metas() = meta;

      (*prop) = p;

    } else {
      DCOUT("spec_type = " << to_string(spec_type));
      if (spec_type == SpecType::Relationship) {
        // `rel` with no target. e.g. `rel target`
        rel = Relationship();
        rel.set_novalue();
        if (variability == Variability::Varying) {
          rel.set_varying_authored();
        }
        rel.metas() = meta;
        (*prop) = Property(rel, custom);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "`typeName` field is missing.");
      }
    }
  }

  return true;
}

template <typename T>
bool USDCReader::Impl::ReconstructPrim(const Specifier &spec, const crate::CrateReader::Node &node,
                                       const PathIndexToSpecIndexMap &psmap,
                                       T *prim) {
  // Prim's properties are stored in its children nodes.
  prim::PropertyMap properties;
  if (!BuildPropertyMap(node.GetChildren(), psmap, &properties)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to build PropertyMap.");
  }

  prim::ReferenceList refs;  // dummy

  prim::PrimReconstructOptions reconstruct_options;
  reconstruct_options.strict_allowedToken_check = _config.strict_allowedToken_check;

  if (!prim::ReconstructPrim<T>(spec, properties, refs, prim, &_warn, &_err, reconstruct_options)) {
    return false;
  }

  return true;
}

bool USDCReader::Impl::ReconstrcutStageMeta(
    const crate::FieldValuePairVector &fvs, StageMetas *metas) {
  /// Stage(toplevel layer) Meta fieldSet example.
  ///
  ///   specTy = SpecTypePseudoRoot
  ///
  ///     - subLayers(+ subLayerOffsets)
  ///     - customLayerData(dict)
  ///     - defaultPrim(token)
  ///     - metersPerUnit(double)
  ///     - kilogramsPerUnit(double)
  ///     - timeCodesPerSecond(double)
  ///     - upAxis(token)
  ///     - documentation(string) : `doc`
  ///     - comment(string) : comment
  ///     - primChildren(token[]) : Crate only. List of root prims(Root Prim should be traversed based on this array)

  std::vector<std::string> subLayers;
  std::vector<LayerOffset> subLayerOffsets;

  for (const auto &fv : fvs) {
    if (fv.first == "upAxis") {
      auto vt = fv.second.get_value<value::token>();
      if (!vt) {
        PUSH_ERROR_AND_RETURN("`upAxis` must be `token` type.");
      }

      std::string v = vt.value().str();
      if (v == "Y") {
        metas->upAxis = Axis::Y;
      } else if (v == "Z") {
        metas->upAxis = Axis::Z;
      } else if (v == "X") {
        metas->upAxis = Axis::X;
      } else {
        PUSH_ERROR_AND_RETURN("`upAxis` must be 'X', 'Y' or 'Z' but got '" + v +
                              "'(note: Case sensitive)");
      }
      DCOUT("upAxis = " << to_string(metas->upAxis.get_value()));

    } else if (fv.first == "metersPerUnit") {
      if (auto vf = fv.second.get_value<float>()) {
        metas->metersPerUnit = double(vf.value());
      } else if (auto vd = fv.second.get_value<double>()) {
        metas->metersPerUnit = vd.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "`metersPerUnit` value must be double or float type, but got '" +
            fv.second.type_name() + "'");
      }
      DCOUT("metersPerUnit = " << metas->metersPerUnit.get_value());
    } else if (fv.first == "kilogramsPerUnit") {
      if (auto vf = fv.second.get_value<float>()) {
        metas->kilogramsPerUnit = double(vf.value());
      } else if (auto vd = fv.second.get_value<double>()) {
        metas->kilogramsPerUnit = vd.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "`kilogramsPerUnit` value must be double or float type, but got '" +
            fv.second.type_name() + "'");
      }
      DCOUT("kilogramsPerUnit = " << metas->kilogramsPerUnit.get_value());
    } else if (fv.first == "timeCodesPerSecond") {
      if (auto vf = fv.second.get_value<float>()) {
        metas->timeCodesPerSecond = double(vf.value());
      } else if (auto vd = fv.second.get_value<double>()) {
        metas->timeCodesPerSecond = vd.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "`timeCodesPerSecond` value must be double or float "
            "type, but got '" +
            fv.second.type_name() + "'");
      }
      DCOUT("timeCodesPerSecond = " << metas->timeCodesPerSecond.get_value());
    } else if (fv.first == "startTimeCode") {
      if (auto vf = fv.second.get_value<float>()) {
        metas->startTimeCode = double(vf.value());
      } else if (auto vd = fv.second.get_value<double>()) {
        metas->startTimeCode = vd.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "`startTimeCode` value must be double or float "
            "type, but got '" +
            fv.second.type_name() + "'");
      }
      DCOUT("startimeCode = " << metas->startTimeCode.get_value());
    } else if (fv.first == "subLayers") {
      if (auto vs = fv.second.get_value<std::vector<std::string>>()) {
        subLayers = vs.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "`subLayers` value must be string[] "
            "type, but got '" +
            fv.second.type_name() + "'");
      }
    } else if (fv.first == "subLayerOffsets") {
      if (auto vs = fv.second.get_value<std::vector<LayerOffset>>()) {
        subLayerOffsets = vs.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "`subLayerOffsets` value must be LayerOffset[] "
            "type, but got '" +
            fv.second.type_name() + "'");
      }
    } else if (fv.first == "endTimeCode") {
      if (auto vf = fv.second.get_value<float>()) {
        metas->endTimeCode = double(vf.value());
      } else if (auto vd = fv.second.get_value<double>()) {
        metas->endTimeCode = vd.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "`endTimeCode` value must be double or float "
            "type, but got '" +
            fv.second.type_name() + "'");
      }
      DCOUT("endTimeCode = " << metas->endTimeCode.get_value());
    } else if (fv.first == "framesPerSecond") {
      if (auto vf = fv.second.get_value<float>()) {
        metas->framesPerSecond = double(vf.value());
      } else if (auto vd = fv.second.get_value<double>()) {
        metas->framesPerSecond = vd.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "`framesPerSecond` value must be double or float "
            "type, but got '" +
            fv.second.type_name() + "'");
      }
      DCOUT("framesPerSecond = " << metas->framesPerSecond.get_value());
    } else if (fv.first == "autoPlay") {
      if (auto vf = fv.second.get_value<bool>()) {
        metas->autoPlay = vf.value();
      } else if (auto vs = fv.second.get_value<std::string>()) {
        // unregisteredvalue uses string type.
        bool autoPlay{true};
        if (vs.value() == "true") {
          autoPlay = true;
        } else if (vs.value() == "false") {
          autoPlay = false;
        } else {
          PUSH_ERROR_AND_RETURN(
              "Unsupported value for `autoPlay`: " << vs.value());
        }
        metas->autoPlay = autoPlay;
      } else {
        PUSH_ERROR_AND_RETURN(
            "`autoPlay` value must be bool "
            "type or string type, but got '" +
            fv.second.type_name() + "'");
      }
      DCOUT("autoPlay = " << metas->autoPlay.get_value());
    } else if (fv.first == "playbackMode") {
      if (auto vf = fv.second.get_value<value::token>()) {
        if (vf.value().str() == "none") {
          metas->playbackMode = StageMetas::PlaybackMode::PlaybackModeNone;
        } else if (vf.value().str() == "loop") {
          metas->playbackMode = StageMetas::PlaybackMode::PlaybackModeLoop;
        } else {
          PUSH_ERROR_AND_RETURN("Unsupported token value for `playbackMode`.");
        }
      } else if (auto vs = fv.second.get_value<std::string>()) {
        // unregisteredvalue uses string type.
        if (vs.value() == "none") {
          metas->playbackMode = StageMetas::PlaybackMode::PlaybackModeNone;
        } else if (vs.value() == "loop") {
          metas->playbackMode = StageMetas::PlaybackMode::PlaybackModeLoop;
        } else {
          PUSH_ERROR_AND_RETURN(
              "Unsupported value for `playbackMode`: " << vs.value());
        }
      } else {
        PUSH_ERROR_AND_RETURN(
            "`playbackMode` value must be token "
            "type, but got '" +
            fv.second.type_name() + "'");
      }
    } else if ((fv.first == "defaultPrim")) {
      auto v = fv.second.get_value<value::token>();
      if (!v) {
        PUSH_ERROR_AND_RETURN("`defaultPrim` must be `token` type.");
      }

      metas->defaultPrim = v.value();
      DCOUT("defaultPrim = " << metas->defaultPrim.str());
    } else if (fv.first == "customLayerData") {
      if (auto v = fv.second.get_value<CustomDataType>()) {
        metas->customLayerData = v.value();
      } else {
        PUSH_ERROR_AND_RETURN(
            "customLayerData must be `dictionary` type, but got type `" +
            fv.second.type_name());
      }
    } else if (fv.first == "primChildren") {  // only appears in USDC.
      auto v = fv.second.get_value<std::vector<value::token>>();
      if (!v) {
        PUSH_ERROR_AND_RETURN("Type must be `token[]` for `primChildren`, but got " +
                   fv.second.type_name());
      }

      metas->primChildren = v.value();
    } else if (fv.first == "documentation") {  // 'doc'
      auto v = fv.second.get_value<std::string>();
      if (!v) {
        PUSH_ERROR_AND_RETURN("Type must be `string` for `documentation`, but got " +
                   fv.second.type_name());
      }
      value::StringData sdata;
      sdata.value = v.value();
      sdata.is_triple_quoted = hasNewline(sdata.value);
      metas->doc = sdata;
      DCOUT("doc = " << metas->doc.value);
    } else if (fv.first == "comment") {  // 'comment'
      auto v = fv.second.get_value<std::string>();
      if (!v) {
        PUSH_ERROR_AND_RETURN("Type must be `string` for `comment`, but got " +
                   fv.second.type_name());
      }
      value::StringData sdata;
      sdata.value = v.value();
      sdata.is_triple_quoted = hasNewline(sdata.value);
      metas->comment = sdata;
      DCOUT("comment = " << metas->comment.value);
    } else {
      PUSH_WARN("[StageMeta] TODO: " + fv.first);
    }
  }

  if (subLayers.size()) {
    std::vector<SubLayer> dst;
    for (size_t i = 0; i < subLayers.size(); i++) {
      SubLayer s;
      s.assetPath = subLayers[i];
      dst.push_back(s);
    }

    if (subLayers.size() == subLayerOffsets.size()) {
      for (size_t i = 0; i < subLayerOffsets.size(); i++) {
        dst[i].layerOffset = subLayerOffsets[i];
      }
    }

    metas->subLayers = dst;

  } else if (subLayerOffsets.size()) {
    PUSH_WARN("Corrupted subLayer info? `subLayers` Fileld not found.");
  }

  return true;
}

nonstd::optional<Prim> USDCReader::Impl::ReconstructPrimFromTypeName(
    const std::string &typeName, // TinyUSDZ's Prim type name
    const std::string &primTypeName, // USD's Prim typeName
    const std::string &prim_name,
    const crate::CrateReader::Node &node, const Specifier spec,
    const std::vector<value::token> &primChildren,
    const std::vector<value::token> &properties,
    const PathIndexToSpecIndexMap &psmap, const PrimMeta &meta, bool *is_unsupported_prim) {

  if (is_unsupported_prim) {
    (*is_unsupported_prim) = false; // init with false
  }


#define RECONSTRUCT_PRIM(__primty, __node_ty, __prim_name, __spec) \
  if (__node_ty == value::TypeTraits<__primty>::type_name()) {     \
    __primty typed_prim;                                           \
    if (!ReconstructPrim(__spec, node, psmap, &typed_prim)) {         \
      PUSH_ERROR("Failed to reconstruct Prim " << __node_ty << " elementName: " << __prim_name);      \
      return nonstd::nullopt;                                      \
    }                                                              \
    typed_prim.meta = meta;                                        \
    typed_prim.name = __prim_name;                                 \
    typed_prim.spec = __spec;                                      \
    typed_prim.propertyNames() = properties; \
    typed_prim.primChildrenNames() = primChildren; \
    value::Value primdata = typed_prim;                            \
    Prim prim(__prim_name, primdata);                            \
    prim.prim_type_name() = primTypeName; \
    /* also add primChildren to Prim */ \
    prim.metas().primChildren = primChildren; \
    return std::move(prim); \
  } else

  if (typeName == "Model" || typeName == "__AnyType__") {
    // Code is mostly identical to RECONSTRUCT_PRIM.
    // Difference is store primTypeName to Model class itself.
    Model typed_prim;
    if (!ReconstructPrim(spec, node, psmap, &typed_prim)) {
      PUSH_ERROR("Failed to reconstruct Model");
      return nonstd::nullopt;
    }
    typed_prim.meta = meta;
    typed_prim.name = prim_name;
    if (typeName == "__AnyType__") {
      typed_prim.prim_type_name = "";
    } else {
      typed_prim.prim_type_name = primTypeName;
    }
    typed_prim.spec = spec;
    typed_prim.propertyNames() = properties;
    typed_prim.primChildrenNames() = primChildren;
    value::Value primdata = typed_prim;
    Prim prim(prim_name, primdata);
    prim.prim_type_name() = primTypeName;
    /* also add primChildren to Prim */
    prim.metas().primChildren = primChildren; \
    return std::move(prim); \
  } else

  RECONSTRUCT_PRIM(Xform, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(Model, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(Scope, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomMesh, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomPoints, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomCylinder, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomCube, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomCone, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomSphere, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomCapsule, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomBasisCurves, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomNurbsCurves, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(PointInstancer, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomCamera, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(GeomSubset, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(SphereLight, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(DomeLight, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(CylinderLight, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(DiskLight, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(DistantLight, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(SkelRoot, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(Skeleton, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(SkelAnimation, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(BlendShape, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(Shader, typeName, prim_name, spec)
  RECONSTRUCT_PRIM(Material, typeName, prim_name, spec) {
    PUSH_WARN("TODO or unsupported prim type: " << typeName);
    if (is_unsupported_prim) {
      (*is_unsupported_prim) = true;
    }
    return nonstd::nullopt;
  }

#undef RECONSTRUCT_PRIM
}

///
///
/// Prim(Model) fieldSet example.
///
///
///   specTy = SpecTypePrim
///
///     - specifier(specifier) : e.g. `def`, `over`, ...
///     - kind(token) : kind metadataum
///     - optional: typeName(token) : type name of Prim(e.g. `Xform`). No
///     typeName = `def "mynode"`
///     - primChildren(TokenVector): List of child prims.
///     - properties(TokenVector) : List of name of Prim properties.
///
///
bool USDCReader::Impl::ParsePrimSpec(const crate::FieldValuePairVector &fvs,
                                     nonstd::optional<std::string> &typeName,
                                     nonstd::optional<Specifier> &specifier,
                                     std::vector<value::token> &primChildren,
                                     std::vector<value::token> &properties,
                                     PrimMeta &primMeta) {
  // Fields for Prim and Prim metas.
  for (const auto &fv : fvs) {
    if (fv.first == "typeName") {
      if (auto pv = fv.second.as<value::token>()) {
        typeName = pv->str();
        DCOUT("typeName = " << typeName.value());
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`typeName` must be type `token`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "specifier") {
      if (auto pv = fv.second.as<Specifier>()) {
        specifier = (*pv);
        DCOUT("specifier = " << to_string(specifier.value()));
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`specifier` must be type `Specifier`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "properties") {
      if (auto pv = fv.second.as<std::vector<value::token>>()) {
        properties = (*pv);
        DCOUT("properties = " << properties);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`properties` must be type `token[]`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "primChildren") {
      // Crate only
      if (auto pv = fv.second.as<std::vector<value::token>>()) {
        primChildren = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`primChildren` must be type `token[]`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "active") {
      if (auto pv = fv.second.as<bool>()) {
        primMeta.active = (*pv);
        DCOUT("active = " << to_string(primMeta.active.value()));
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`active` must be type `bool`, but got type `"
                                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "hidden") {
      if (auto pv = fv.second.as<bool>()) {
        primMeta.hidden = (*pv);
        DCOUT("hidden = " << to_string(primMeta.hidden.value()));
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`hidden` must be type `bool`, but got type `"
                                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "instanceable") {
      if (auto pv = fv.second.as<bool>()) {
        primMeta.instanceable = (*pv);
        DCOUT("instanceable = " << to_string(primMeta.instanceable.value()));
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`instanceable` must be type `bool`, but got type `"
                                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "assetInfo") {
      // CustomData(dict)
      if (auto pv = fv.second.as<CustomDataType>()) {
        primMeta.assetInfo = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`assetInfo` must be type `dictionary`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "clips") {
      // CustomData(dict)
      if (auto pv = fv.second.as<CustomDataType>()) {
        primMeta.clips = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`clips` must be type `dictionary`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "kind") {
      if (auto pv = fv.second.as<value::token>()) {

          const value::token tok = (*pv);
          if (tok.str() == "subcomponent") {
            primMeta.kind = Kind::Subcomponent;
          } else if (tok.str() == "component") {
            primMeta.kind = Kind::Component;
          } else if (tok.str() == "model") {
            primMeta.kind = Kind::Model;
          } else if (tok.str() == "group") {
            primMeta.kind = Kind::Group;
          } else if (tok.str() == "assembly") {
            primMeta.kind = Kind::Assembly;
          } else if (tok.str() == "sceneLibrary") {
            // USDZ specific: https://developer.apple.com/documentation/arkit/usdz_schemas_for_ar/scenelibrary
            primMeta.kind = Kind::SceneLibrary;
          } else {

            primMeta.kind = Kind::UserDef;
            primMeta._kind_str = tok.str();
          }
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "`kind` must be type `token`, but got type `"
                                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "apiSchemas") {
      if (auto pv = fv.second.as<ListOp<value::token>>()) {
        auto listop = (*pv);

        std::string warn;
        auto ret = ToAPISchemas(listop, _config.allow_unknown_apiSchemas, warn);
        if (!ret) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, "Failed to validate `apiSchemas`: " + ret.error());
        } else {
          if (warn.size()) {
            PUSH_WARN(warn);
          }
          primMeta.apiSchemas = (*ret);
        }
        // DCOUT("apiSchemas = " << to_string(listop));
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`apiSchemas` must be type `ListOp[Token]`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "documentation") {
      if (auto pv = fv.second.as<std::string>()) {
        value::StringData s;
        s.value = (*pv);
        s.is_triple_quoted = hasNewline(s.value);
        primMeta.doc = s;
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`documentation` must be type `string`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "comment") {
      if (auto pv = fv.second.as<std::string>()) {
        value::StringData s;
        s.value = (*pv);
        s.is_triple_quoted = hasNewline(s.value);
        primMeta.comment = s;
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`comment` must be type `string`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "sdrMetadata") {
      // CustomData(dict)
      if (auto pv = fv.second.as<CustomDataType>()) {
        // TODO: Check if all keys are string type.
        primMeta.sdrMetadata = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`sdrMetadata` must be type `dictionary`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "customData") {
      // CustomData(dict)
      if (auto pv = fv.second.as<CustomDataType>()) {
        primMeta.customData = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`customData` must be type `dictionary`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "variantSelection") {
      if (auto pv = fv.second.as<VariantSelectionMap>()) {
        primMeta.variants = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`variantSelection` must be type `variants`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "variantChildren") {
      // Used internally
      if (auto pv = fv.second.as<std::vector<value::token>>()) {
        primMeta.variantChildren = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`variantChildren` must be type `token[]`, but got type `"
                      << fv.second.type_name() << "`");
      }

    } else if (fv.first == "variantSetChildren") {
      // Used internally
      if (auto pv = fv.second.as<std::vector<value::token>>()) {
        primMeta.variantSetChildren = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`variantSetChildren` must be type `token[]`, but got type `"
                      << fv.second.type_name() << "`");
      }

    } else if (fv.first == "variantSetNames") {
      // ListOp<string>
      if (auto pv = fv.second.as<ListOp<std::string>>()) {
        const ListOp<std::string> &p = *pv;
        DCOUT("variantSetNames = " << to_string(p));

        auto ps = DecodeListOp<std::string>(p);

        if (ps.size() > 1) {
          // This should not happen though.
          PUSH_WARN(
              "ListOp with multiple ListOpType is not supported for now. Use "
              "the first one: " +
              to_string(std::get<0>(ps[0])));
        }

        auto qual = std::get<0>(ps[0]);
        auto items = std::get<1>(ps[0]);
        auto listop = (*pv);
        primMeta.variantSets = std::make_pair(qual, items);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag,
            "`variantSetNames` must be type `ListOp[String]`, but got type `"
                << fv.second.type_name() << "`");
      }
    } else if (fv.first == "sceneName") {  // USDZ extension
      if (auto pv = fv.second.as<std::string>()) {
        primMeta.sceneName = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`sceneName` must be type `string`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "displayName") {  // USD supported since 23.xx?
      if (auto pv = fv.second.as<std::string>()) {
        primMeta.displayName = (*pv);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`displayName` must be type `string`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "inherits") {  // `inherits` composition
      if (auto pvb = fv.second.as<value::ValueBlock>()) {
        (void)pvb;
        // make empty array
        primMeta.inherits =
            std::make_pair(ListEditQual::ResetToExplicit, std::vector<Path>());
      } else if (auto pv = fv.second.as<ListOp<Path>>()) {
        const ListOp<Path> &p = *pv;
        DCOUT("inherits = " << to_string(p));

        auto ps = DecodeListOp<Path>(p);

        if (ps.size() > 1) {
          // This should not happen though.
          PUSH_WARN(
              "ListOp with multiple ListOpType is not supported for now. Use "
              "the first one: " +
              to_string(std::get<0>(ps[0])));
        }

        auto qual = std::get<0>(ps[0]);
        auto items = std::get<1>(ps[0]);
        primMeta.inherits = std::make_pair(qual, items);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`inherits` must be type `path` o `path[]`, but got type `"
                      << fv.second.type_name() << "`");
      }

    } else if (fv.first == "references") {  // `references` composition
      if (auto pvb = fv.second.as<value::ValueBlock>()) {
        (void)pvb;
        // make empty array
        primMeta.references = std::make_pair(ListEditQual::ResetToExplicit,
                                             std::vector<Reference>());
      } else if (auto pv = fv.second.as<ListOp<Reference>>()) {
        const ListOp<Reference> &p = *pv;
        DCOUT("references = " << to_string(p));

        auto ps = DecodeListOp<Reference>(p);

        if (ps.size() > 1) {
          // This should not happen though.
          PUSH_WARN(
              "ListOp with multiple ListOpType is not supported for now. Use "
              "the first one: " +
              to_string(std::get<0>(ps[0])));
        }

        auto qual = std::get<0>(ps[0]);
        auto items = std::get<1>(ps[0]);
        auto listop = (*pv);
        primMeta.references = std::make_pair(qual, items);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag,
            "`references` must be type `ListOp[Reference]`, but got type `"
                << fv.second.type_name() << "`");
      }
    } else if (fv.first == "payload") {  // `payload` composition
      if (auto pvb = fv.second.as<value::ValueBlock>()) {
        (void)pvb;
        // make empty array
        primMeta.payload = std::make_pair(ListEditQual::ResetToExplicit,
                                             std::vector<Payload>());
      } else if (auto pv = fv.second.as<Payload>()) {
        // payload can be non-listop

        std::vector<Payload> pls;
        pls.push_back(*pv);
        primMeta.payload = std::make_pair(ListEditQual::ResetToExplicit, pls);
      } else if (auto pvs = fv.second.as<ListOp<Payload>>()) {
        const ListOp<Payload> &p = *pvs;
        DCOUT("payload = " << to_string(p));

        auto ps = DecodeListOp<Payload>(p);

        if (ps.size() > 1) {
          // This should not happen though.
          PUSH_WARN(
              "ListOp with multiple ListOpType is not supported for now. Use "
              "the first one: " +
              to_string(std::get<0>(ps[0])));
        }

        auto qual = std::get<0>(ps[0]);
        auto items = std::get<1>(ps[0]);
        auto listop = (*pvs);
        primMeta.payload = std::make_pair(qual, items);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag,
            "`payload` must be type `ListOp[Payload]`, but got type `"
                << fv.second.type_name() << "`");
      }
    } else if (fv.first == "specializes") {  // `specializes` composition
      if (auto pv = fv.second.as<ListOp<Path>>()) {
        const ListOp<Path> &p = *pv;
        DCOUT("specializes = " << to_string(p));

        auto ps = DecodeListOp<Path>(p);

        if (ps.size() > 1) {
          // This should not happen though.
          PUSH_WARN(
              "ListOp with multiple ListOpType is not supported for now. Use "
              "the first one: " +
              to_string(std::get<0>(ps[0])));
        }

        auto qual = std::get<0>(ps[0]);
        auto items = std::get<1>(ps[0]);
        auto listop = (*pv);
        primMeta.specializes = std::make_pair(qual, items);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`specializes` must be type `ListOp[Path]`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else if (fv.first == "inheritPaths") {  // `specializes` composition
      if (auto pv = fv.second.as<ListOp<Path>>()) {
        const ListOp<Path> &p = *pv;
        DCOUT("inheritPaths = " << to_string(p));

        auto ps = DecodeListOp<Path>(p);

        if (ps.size() > 1) {
          // This should not happen though.
          PUSH_WARN(
              "ListOp with multiple ListOpType is not supported for now. Use "
              "the first one: " +
              to_string(std::get<0>(ps[0])));
        }

        auto qual = std::get<0>(ps[0]);
        auto items = std::get<1>(ps[0]);
        auto listop = (*pv);
        primMeta.inheritPaths = std::make_pair(qual, items);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`inheritPaths` must be type `ListOp[Path]`, but got type `"
                      << fv.second.type_name() << "`");
      }

    } else {
      // TODO: support int, int[], uint, uint[], int64, uint64, ...
      // https://github.com/syoyo/tinyusdz/issues/106
      if (auto pv = fv.second.as<std::string>()) {
        // Assume unregistered Prim metadatum
        primMeta.unregisteredMetas[fv.first] = (*pv);
      } else if (auto ptv = fv.second.as<value::token>()) {
        // store value as string type.
        primMeta.unregisteredMetas[fv.first] = quote((*ptv).str());
      } else {
        DCOUT("PrimProp TODO: " << fv.first);
        PUSH_WARN("PrimProp TODO: " << fv.first);
      }
    }
  }

  return true;
}

///
///
/// VariantSet fieldSet example.
///
///
///   specTy = SpecTypeVariantSet
///
///     - variantChildren(token[])
///
///
bool USDCReader::Impl::ParseVariantSetFields(
    const crate::FieldValuePairVector &fvs,
    std::vector<value::token> &variantChildren) {
  // Fields for Prim and Prim metas.
  for (const auto &fv : fvs) {
    if (fv.first == "variantChildren") {
      if (auto pv = fv.second.as<std::vector<value::token>>()) {
        variantChildren = (*pv);
        DCOUT("variantChildren: " << variantChildren);
      } else {
        PUSH_ERROR_AND_RETURN_TAG(
            kTag, "`variantChildren` must be type `token[]`, but got type `"
                      << fv.second.type_name() << "`");
      }
    } else {
      DCOUT("VariantSet field TODO: " << fv.first);
      PUSH_WARN("VariantSet field TODO: " << fv.first);
    }
  }

  return true;
}

bool USDCReader::Impl::ReconstructPrimNode(int parent, int current, int level,
                                           bool is_parent_variant,
                                           const PathIndexToSpecIndexMap &psmap,
                                           Stage *stage,
                                           nonstd::optional<Prim> *primOut) {
  (void)level;
  const crate::CrateReader::Node &node = _nodes[size_t(current)];

  DCOUT(fmt::format("parent = {}, curent = {}, is_parent_variant = {}", parent, current, is_parent_variant));

#ifdef TINYUSDZ_LOCAL_DEBUG_PRINT
  std::cout << pprint::Indent(uint32_t(level)) << "lv[" << level
            << "] node_index[" << current << "] " << node.GetLocalPath()
            << " ==\n";
  std::cout << pprint::Indent(uint32_t(level)) << " childs = [";
  for (size_t i = 0; i < node.GetChildren().size(); i++) {
    std::cout << node.GetChildren()[i];
    if (i != (node.GetChildren().size() - 1)) {
      std::cout << ", ";
    }
  }
  std::cout << "] (is_parent_variant = " << is_parent_variant << ")\n";
#endif

  if (!psmap.count(uint32_t(current))) {
    // No specifier assigned to this node.
    DCOUT("No specifier assigned to this node: " << current);
    return true;  // would be OK.
  }

  uint32_t spec_index = psmap.at(uint32_t(current));
  if (spec_index >= _specs.size()) {
    PUSH_ERROR("Invalid specifier id: " + std::to_string(spec_index) +
               ". Must be in range [0, " + std::to_string(_specs.size()) + ")");
    return false;
  }

  const crate::Spec &spec = _specs[spec_index];

  DCOUT(pprint::Indent(uint32_t(level))
        << "  specTy = " << to_string(spec.spec_type));
  DCOUT(pprint::Indent(uint32_t(level))
        << "  fieldSetIndex = " << spec.fieldset_index.value);

  if ((spec.spec_type == SpecType::Attribute) ||
      (spec.spec_type == SpecType::Relationship)) {
    if (_prim_table.count(parent)) {
      // This node is a Properties node. These are processed in
      // ReconstructPrim(), so nothing to do here.
      return true;
    }
  }

  if (!_live_fieldsets.count(spec.fieldset_index)) {
    PUSH_ERROR("FieldSet id: " + std::to_string(spec.fieldset_index.value) +
               " must exist in live fieldsets.");
    return false;
  }

  const crate::FieldValuePairVector &fvs =
      _live_fieldsets.at(spec.fieldset_index);

  if (fvs.size() > _config.kMaxFieldValuePairs) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too much FieldValue pairs.");
  }

#if defined(TINYUSDZ_LOCAL_DEBUG_PRINT)
  // DBG
  for (auto &fv : fvs) {
    DCOUT("parent[" << current << "] level [" << level << "] fv name "
                    << fv.first << "(type = " << fv.second.type_name() << ")");
  }
#endif

  // StageMeta = root only attributes.
  // TODO: Unify reconstrction code with USDAReder?
  if (current == 0) {
    if (const auto &pv = GetElemPath(crate::Index(uint32_t(current)))) {
      DCOUT("Root element path: " << pv.value().full_path_name());
    } else {
      PUSH_ERROR_AND_RETURN("(Internal error). Root Element Path not found.");
    }

    // Root layer(Stage) is PseudoRoot spec type.
    if (spec.spec_type != SpecType::PseudoRoot) {
      PUSH_ERROR_AND_RETURN(
          "SpecTypePseudoRoot expected for root layer(Stage) element.");
    }

    if (!ReconstrcutStageMeta(fvs, &stage->metas())) {
      PUSH_ERROR_AND_RETURN("Failed to reconstruct StageMeta.");
    }

    // TODO: Validate scene using `StageMetas::primChildren`.

    _prim_table.insert(current);

    return true;
  }

  DCOUT("spec.type = " << to_string(spec.spec_type));
  switch (spec.spec_type) {
    case SpecType::PseudoRoot: {
      PUSH_ERROR_AND_RETURN_TAG(
          kTag, "SpecType PseudoRoot in a child node is not supported(yet)");
    }
    case SpecType::Prim: {
      nonstd::optional<std::string> typeName;
      nonstd::optional<Specifier> specifier;
      std::vector<value::token> primChildren;
      std::vector<value::token> properties;

      PrimMeta primMeta;

      DCOUT("== PrimFields begin ==> ");

      if (!ParsePrimSpec(fvs, typeName, specifier, primChildren, properties, primMeta)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to parse Prim fields.");
        return false;
      }

      DCOUT("<== PrimFields end ===");

      Path elemPath;

      if (const auto &pv = GetElemPath(crate::Index(uint32_t(current)))) {
        DCOUT(fmt::format("Element path: {}", pv.value().full_path_name()));
        elemPath = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "(Internal errror) Element path not found.");
      }

      // Sanity check
      if (specifier) {
        if (specifier.value() == Specifier::Def) {
          // ok
        } else if (specifier.value() == Specifier::Class) {
          // ok
        } else if (specifier.value() == Specifier::Over) {
          // ok
        } else {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Specifier.");
        }
      } else {
        // default `over`
        specifier = Specifier::Over;
      }

      std::string pTyName;
      if (!typeName) {
        //PUSH_WARN("Treat this node as Model(`typeName` field is missing).");
        pTyName = "Model";
      } else {
        pTyName = typeName.value();
      }

      {
        DCOUT("elemPath.prim_name = " << elemPath.prim_part());
        std::string prim_name = elemPath.prim_part();
        std::string primTypeName = typeName.has_value() ? typeName.value() : "";

        // __AnyType__
        if (typeName.has_value() && typeName.value() == "__AnyType__") {
          primTypeName = "";
        }

        // Validation check should be already done in crate-reader, so no
        // further validation required.
        if (!ValidatePrimElementName(prim_name)) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Prim name.");
        }

        bool is_unsupported_prim{false};
        auto prim = ReconstructPrimFromTypeName(pTyName, primTypeName, prim_name,
                                                node, specifier.value(), primChildren, properties,
                                                psmap, primMeta, &is_unsupported_prim);

        if (prim) {
          // Prim name
          prim.value().element_path() = elemPath;
        } else {
          if (_config.allow_unknown_prims && is_unsupported_prim) {
            // Try to reconsrtuct as Model
            prim = ReconstructPrimFromTypeName("Model", primTypeName, prim_name,
                                                    node, specifier.value(), primChildren, properties,
                                                    psmap, primMeta);
            if (prim) {
              // Prim name
              prim.value().element_path() = elemPath;
            } else {
              return false;
            }
          } else {
            return false;
          }
        }

        if (primOut) {
          (*primOut) = prim;
        }
      }

      DCOUT("add prim idx " << current);
      if (_prim_table.count(current)) {
        DCOUT("??? prim idx already set " << current);
      } else {
        _prim_table.insert(current);
      }

      break;
    }
    case SpecType::VariantSet: {
      // Assume parent(Prim) already exists(parsed)
      // TODO: Confirm Crate format allow defining Prim after VariantSet
      // serialization.
      if (!_prim_table.count(parent)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "Parent Prim for this VariantSet not found.");
      }

      DCOUT(
          fmt::format("[{}] is a VariantSet node(parent = {}). prim_idx? = {}",
                      current, parent, _prim_table.count(current)));

      Path elemPath;

      if (const auto &pv = GetElemPath(crate::Index(uint32_t(current)))) {
        elemPath = pv.value();

        DCOUT(fmt::format("Element path: {}", dump_path(elemPath)));

        // Ensure ElementPath is variant
        if (!tokenize_variantElement(elemPath.full_path_name())) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, fmt::format("Invalid Variant ElementPath '{}'.", elemPath));
        }

      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "(Internal errror) Element path not found.");
      }

      std::vector<value::token> variantChildren;

      // Only contains `variantChildren` field with type `token[]`

      DCOUT("== VariantSetFields begin ==> ");

      if (!ParseVariantSetFields(fvs, variantChildren)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to parse VariantSet fields.");
        return false;
      }

      DCOUT("<== VariantSetFields end === ");

      // Add variantChildren to prim node.
      // TODO: elemPath
      if (!AddVariantChildrenToPrimNode(parent, variantChildren)) {
        return false;
      }

      break;
    }
    case SpecType::Variant: {
      // Since the Prim this Variant node belongs to is not yet reconstructed
      // during the Prim tree traversal, We manage variant node separately

      DCOUT(fmt::format("[{}] is a Variant node(parent = {}). prim_idx? = {}",
                        current, parent, _prim_table.count(current)));

      nonstd::optional<std::string> typeName;
      nonstd::optional<Specifier> specifier;
      std::vector<value::token> primChildren;
      std::vector<value::token> properties;

      PrimMeta primMeta;

      DCOUT("== VariantFields begin ==> ");

      if (!ParsePrimSpec(fvs, typeName, specifier, primChildren, properties, primMeta)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "Failed to parse Prim fields under Variant.");
        return false;
      }

      DCOUT("<== VariantFields end === ");

      Path elemPath;
      if (const auto &pv = GetElemPath(crate::Index(uint32_t(current)))) {
        elemPath = pv.value();
        DCOUT(fmt::format("Element path: {}", elemPath.full_path_name()));
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "(Internal errror) Element path not found.");
      }

      // Sanity check
      if (specifier) {
        if (specifier.value() == Specifier::Def) {
          // ok
        } else if (specifier.value() == Specifier::Class) {
          // ok
        } else if (specifier.value() == Specifier::Over) {
          // ok
        } else {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Specifier.");
        }
      } else {
        // Seems Variant is only composed of Properties.
        // Create pseudo `def` Prim
        specifier = Specifier::Def;
      }

      std::string pTyName; // TinyUSDZ' prim typename
      if (!typeName) {
        //PUSH_WARN("Treat this node as Model(where `typeName` is missing).");
        pTyName = "Model";
      } else {
        pTyName = typeName.value();
      }

      nonstd::optional<Prim> variantPrim;
      {
        std::string prim_name = elemPath.prim_part();
        DCOUT("elemPath = " << dump_path(elemPath));
        DCOUT("prim_name = " << prim_name);

        std::string primTypeName = typeName.has_value() ? typeName.value() : "";
        // __AnyType__
        if (typeName.has_value() && typeName.value() == "__AnyType__") {
          primTypeName = "";
        }

        // Something like '{shapeVariant=Capsule}'

        std::array<std::string, 2> variantPair;
        if (!tokenize_variantElement(prim_name, &variantPair)) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, fmt::format("Invalid Variant ElementPath '{}'.", elemPath));
        }

        std::string variantSetName = variantPair[0];
        std::string variantPrimName = variantPair[1];

        if (!ValidatePrimElementName(variantPrimName)) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, fmt::format("Invalid Prim name in Variant: `{}`",
                                variantPrimName));
        }

        bool is_unsupported_prim{false};
        variantPrim = ReconstructPrimFromTypeName(
            pTyName, primTypeName, variantPrimName, node, specifier.value(), primChildren, properties,
            psmap, primMeta, &is_unsupported_prim);

        if (variantPrim) {
          // Prim name
          variantPrim.value().element_path() =
              elemPath;  // FIXME: Use variantPrimName?

          // Prim Specifier
          variantPrim.value().specifier() = specifier.value();

          // Store variantPrim to temporary buffer.
          DCOUT(fmt::format("parent {} add prim idx {} as variant: ", parent, current));
          if (_variantPrims.count(current)) {
            DCOUT("??? prim idx already set " << current);
          } else {
            _variantPrims.emplace(current,  variantPrim.value());
            _variantPrimChildren[parent].push_back(current);
          }
        } else {
          if (_config.allow_unknown_prims && is_unsupported_prim) {
            // Try to reconstruct as Model
            variantPrim = ReconstructPrimFromTypeName(
                "Model", primTypeName, variantPrimName, node, specifier.value(), primChildren, properties,
                psmap, primMeta);

            if (variantPrim) {
              // Prim name
              variantPrim.value().element_path() =
                  elemPath;  // FIXME: Use variantPrimName?

              // Prim Specifier
              variantPrim.value().specifier() = specifier.value();

              // Store variantPrim to temporary buffer.
              DCOUT(fmt::format("parent {} add prim idx {} as variant: ", parent, current));
              if (_variantPrims.count(current)) {
                DCOUT("??? prim idx already set " << current);
              } else {
                _variantPrims.emplace(current, variantPrim.value());
                _variantPrimChildren[parent].push_back(current);
              }
            } else {
              return false;
            }
          } else {
            return false;
          }
        }
      }

      break;
    }
    case SpecType::Attribute: {
      if (is_parent_variant) {
        nonstd::optional<Path> path = GetPath(spec.path_index);

        if (!path) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid PathIndex.");
        }

        Property prop;
        if (!ParseProperty(spec.spec_type, fvs, &prop)) {
          PUSH_ERROR_AND_RETURN_TAG(kTag,
                                    fmt::format("Failed to parse Attribut: {}.",
                                                path.value().prop_part()));
        }

        // Parent Prim is not yet reconstructed, so store info to temporary
        // buffer _variantAttributeNodes.
        _variantProps[current] = {path.value(), prop};
        _variantPropChildren[parent].push_back(current);

        DCOUT(
            fmt::format("parent {} current [{}] Parsed Attribute {} under Variant. PathIndex {}",
                        parent, current, path.value().prop_part(), spec.path_index));

      } else {
        // Maybe parent is Class/Over, or inherited
        PUSH_WARN(
            "TODO: SpecTypeAttribute(in conjunction with Class/Over specifier, "
            "or inherited?)");
      }
      break;
    }
    case SpecType::Connection:
    case SpecType::Relationship:
    case SpecType::RelationshipTarget: {
      PUSH_ERROR_AND_RETURN_TAG(
          kTag, fmt::format("TODO: Unsupported/Unimplemented SpecType: {}.",
                            to_string(spec.spec_type)));
      break;
    }
    case SpecType::Expression:
    case SpecType::Mapper:
    case SpecType::MapperArg: {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Unsupported SpecType: {}.",
                                                  to_string(spec.spec_type)));
      break;
    }
    case SpecType::Unknown:
    case SpecType::Invalid: {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "[InternalError] Invalid SpecType.");
      break;
    }
  }

  return true;
}

bool USDCReader::Impl::ReconstructPrimSpecNode(int parent, int current, int level,
                                           bool is_parent_variant,
                                           const PathIndexToSpecIndexMap &psmap,
                                           Layer *layer,
                                           nonstd::optional<PrimSpec> *primOut) {
  (void)level;
  const crate::CrateReader::Node &node = _nodes[size_t(current)];

#ifdef TINYUSDZ_LOCAL_DEBUG_PRINT
  std::cout << pprint::Indent(uint32_t(level)) << "lv[" << level
            << "] node_index[" << current << "] " << node.GetLocalPath()
            << " ==\n";
  std::cout << pprint::Indent(uint32_t(level)) << " childs = [";
  for (size_t i = 0; i < node.GetChildren().size(); i++) {
    std::cout << node.GetChildren()[i];
    if (i != (node.GetChildren().size() - 1)) {
      std::cout << ", ";
    }
  }
  std::cout << "] (is_parent_variant = " << is_parent_variant << ")\n";
#endif

  if (!psmap.count(uint32_t(current))) {
    // No specifier assigned to this node.
    DCOUT("No specifier assigned to this node: " << current);
    return true;  // would be OK.
  }

  uint32_t spec_index = psmap.at(uint32_t(current));
  if (spec_index >= _specs.size()) {
    PUSH_ERROR("Invalid specifier id: " + std::to_string(spec_index) +
               ". Must be in range [0, " + std::to_string(_specs.size()) + ")");
    return false;
  }

  const crate::Spec &spec = _specs[spec_index];

  DCOUT(pprint::Indent(uint32_t(level))
        << "  specTy = " << to_string(spec.spec_type));
  DCOUT(pprint::Indent(uint32_t(level))
        << "  fieldSetIndex = " << spec.fieldset_index.value);

  if ((spec.spec_type == SpecType::Attribute) ||
      (spec.spec_type == SpecType::Relationship)) {
    if (_prim_table.count(parent)) {
      // This node is a Properties node. These are processed in
      // ReconstructPrim(), so nothing to do here.
      return true;
    }
  }

  if (!_live_fieldsets.count(spec.fieldset_index)) {
    PUSH_ERROR("FieldSet id: " + std::to_string(spec.fieldset_index.value) +
               " must exist in live fieldsets.");
    return false;
  }

  const crate::FieldValuePairVector &fvs =
      _live_fieldsets.at(spec.fieldset_index);

  if (fvs.size() > _config.kMaxFieldValuePairs) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too much FieldValue pairs.");
  }

#if defined(TINYUSDZ_LOCAL_DEBUG_PRINT)
  // DBG
  for (auto &fv : fvs) {
    DCOUT("parent[" << current << "] level [" << level << "] fv name "
                    << fv.first << "(type = " << fv.second.type_name() << ")");
  }
#endif

  // StageMeta = root only attributes.
  // TODO: Unify reconstrction code with USDAReder?
  if (current == 0) {
    if (const auto &pv = GetElemPath(crate::Index(uint32_t(current)))) {
      DCOUT("Root element path: " << pv.value().full_path_name());
    } else {
      PUSH_ERROR_AND_RETURN("(Internal error). Root Element Path not found.");
    }

    // Root layer(Stage) is PseudoRoot spec type.
    if (spec.spec_type != SpecType::PseudoRoot) {
      PUSH_ERROR_AND_RETURN(
          "SpecTypePseudoRoot expected for root layer(Stage) element.");
    }

    if (!ReconstrcutStageMeta(fvs, &layer->metas())) {
      PUSH_ERROR_AND_RETURN("Failed to reconstruct StageMeta.");
    }

    // TODO: Validate scene using `StageMetas::primChildren`.

    _prim_table.insert(current);

    return true;
  }

  switch (spec.spec_type) {
    case SpecType::PseudoRoot: {
      PUSH_ERROR_AND_RETURN_TAG(
          kTag, "SpecType PseudoRoot in a child node is not supported(yet)");
    }
    case SpecType::Prim: {
      nonstd::optional<std::string> typeName;
      nonstd::optional<Specifier> specifier;
      std::vector<value::token> primChildren;
      std::vector<value::token> properties;

      PrimMeta primMeta;

      DCOUT("== PrimFields begin ==> ");

      if (!ParsePrimSpec(fvs, typeName, specifier, primChildren, properties, primMeta)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to parse Prim fields.");
        return false;
      }

      DCOUT("<== PrimFields end ===");

      Path elemPath;

      if (const auto &pv = GetElemPath(crate::Index(uint32_t(current)))) {
        DCOUT(fmt::format("Element path: {}", pv.value().full_path_name()));
        elemPath = pv.value();
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "(Internal errror) Element path not found.");
      }

      // Sanity check
      if (specifier) {
        if (specifier.value() == Specifier::Def) {
          // ok
        } else if (specifier.value() == Specifier::Class) {
          // ok
        } else if (specifier.value() == Specifier::Over) {
          // ok
        } else {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Specifier.");
        }
      } else {
        // Default = Over Prim.
        specifier = Specifier::Over;
      }

      std::string pTyName;
      if (!typeName) {
        //PUSH_WARN("Treat this node as Model(`typeName` field is missing).");
        pTyName = "Model";
      } else {
        pTyName = typeName.value();
      }

      {
        DCOUT("elemPath.prim_name = " << elemPath.prim_part());
        std::string prim_name = elemPath.prim_part();
        std::string primTypeName = typeName.has_value() ? typeName.value() : "";
        // __AnyType__
        if (typeName.has_value() && typeName.value() == "__AnyType__") {
          primTypeName = "";
        }

        // Validation check should be already done in crate-reader, so no
        // further validation required.
        if (!ValidatePrimElementName(prim_name)) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Prim name.");
        }

        PrimSpec primspec;

#if 0
        bool is_unsupported_prim{false};
        auto prim = ReconstructPrimFromTypeName(pTyName, primTypeName, prim_name,
                                                node, specifier.value(), primChildren, properties,
                                                psmap, primMeta, &is_unsupported_prim);

        if (prim) {
          // Prim name
          prim.value().element_path() = elemPath;
        } else {
          if (_config.allow_unknown_prims && is_unsupported_prim) {
            // Try to reconsrtuct as Model
            prim = ReconstructPrimFromTypeName("Model", primTypeName, prim_name,
                                                    node, specifier.value(), primChildren, properties,
                                                    psmap, primMeta);
            if (prim) {
              // Prim name
              prim.value().element_path() = elemPath;
            } else {
              return false;
            }
          } else {
            return false;
          }
        }
#else
        primspec.typeName() = primTypeName;
        primspec.name() = prim_name;

        prim::PropertyMap props;
        if (!BuildPropertyMap(node.GetChildren(), psmap, &props)) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to build PropertyMap.");
        }
        primspec.props() = props;
        primspec.metas() = primMeta;
        // TODO: primChildren, properties

        if (primOut) {
          (*primOut) = primspec;
        }
#endif
      }

      DCOUT("add prim idx " << current);
      if (_prim_table.count(current)) {
        DCOUT("??? prim idx already set " << current);
      } else {
        _prim_table.insert(current);
      }

      break;
    }
    case SpecType::VariantSet: {
      // Assume parent(Prim) already exists(parsed)
      // TODO: Confirm Crate format allow defining Prim after VariantSet
      // serialization.
      if (!_prim_table.count(parent)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "Parent Prim for this VariantSet not found.");
      }

      DCOUT(
          fmt::format("[{}] is a Variantset node(parent = {}). prim_idx? = {}",
                      current, parent, _prim_table.count(current)));

      Path elemPath;

      if (const auto &pv = GetElemPath(crate::Index(uint32_t(current)))) {
        elemPath = pv.value();

        DCOUT(fmt::format("Element path: {}", dump_path(elemPath)));

        // Ensure ElementPath is variant
        if (!tokenize_variantElement(elemPath.full_path_name())) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, fmt::format("Invalid Variant ElementPath '{}'.", elemPath));
        }

      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "(Internal errror) Element path not found.");
      }

      std::vector<value::token> variantChildren;

      // Only contains `variantChildren` field with type `token[]`

      DCOUT("== VariantSetFields begin ==> ");

      if (!ParseVariantSetFields(fvs, variantChildren)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to parse VariantSet fields.");
        return false;
      }

      DCOUT("<== VariantSetFields end === ");

      // Add variantChildren to prim node.
      // TODO: elemPath
      if (!AddVariantChildrenToPrimNode(parent, variantChildren)) {
        return false;
      }

      break;
    }
    case SpecType::Variant: {
      // Since the Prim this Variant node belongs to is not yet reconstructed
      // during the Prim tree traversal, We manage variant node separately

      DCOUT(fmt::format("[{}] is a Variant node(parent = {}). prim_idx? = {}",
                        current, parent, _prim_table.count(current)));

      nonstd::optional<std::string> typeName;
      nonstd::optional<Specifier> specifier;
      std::vector<value::token> primChildren;
      std::vector<value::token> properties;

      PrimMeta primMeta;

      DCOUT("== VariantFields begin ==> ");

      if (!ParsePrimSpec(fvs, typeName, specifier, primChildren, properties, primMeta)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "Failed to parse Prim fields under Variant.");
        return false;
      }

      DCOUT("<== VariantFields end === ");

      Path elemPath;
      if (const auto &pv = GetElemPath(crate::Index(uint32_t(current)))) {
        elemPath = pv.value();
        DCOUT(fmt::format("Element path: {}", elemPath.full_path_name()));
      } else {
        PUSH_ERROR_AND_RETURN_TAG(kTag,
                                  "(Internal errror) Element path not found.");
      }

      // Sanity check
      if (specifier) {
        if (specifier.value() == Specifier::Def) {
          // ok
        } else if (specifier.value() == Specifier::Class) {
          // ok
        } else if (specifier.value() == Specifier::Over) {
          // ok
        } else {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Specifier.");
        }
      } else {
        // Seems Variant is only composed of Properties.
        // Create pseudo `def` Prim
        // FIXME: default is `Over`?
        specifier = Specifier::Def;
      }

      std::string pTyName; // TinyUSDZ' prim typename
      if (!typeName) {
        //PUSH_WARN("Treat this node as Model(where `typeName` is missing).");
        pTyName = "Model";
      } else {
        pTyName = typeName.value();
      }

      {
        std::string prim_name = elemPath.prim_part();
        DCOUT("elemPath = " << dump_path(elemPath));
        DCOUT("prim_name = " << prim_name);

        std::string primTypeName = typeName.has_value() ? typeName.value() : "";
        // __AnyType__
        if (typeName.has_value() && typeName.value() == "__AnyType__") {
          primTypeName = "";
        }

        // Something like '{shapeVariant=Capsule}'

        std::array<std::string, 2> variantPair;
        if (!tokenize_variantElement(prim_name, &variantPair)) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, fmt::format("Invalid Variant ElementPath '{}'.", elemPath));
        }

        std::string variantSetName = variantPair[0];
        std::string variantPrimName = variantPair[1];

        if (!ValidatePrimElementName(variantPrimName)) {
          PUSH_ERROR_AND_RETURN_TAG(
              kTag, fmt::format("Invalid Prim name in Variant: `{}`",
                                variantPrimName));
        }

#if 0
        nonstd::optional<PrimSpec> variantPrimSpec;
        bool is_unsupported_prim{false};
        variantPrim = ReconstructPrimFromTypeName(
            pTyName, primTypeName, variantPrimName, node, specifier.value(), primChildren, properties,
            psmap, primMeta, &is_unsupported_prim);

        if (variantPrim) {
          // Prim name
          variantPrim.value().element_path() =
              elemPath;  // FIXME: Use variantPrimName?

          // Prim Specifier
          variantPrim.value().specifier() = specifier.value();

          // Store variantPrim to temporary buffer.
          DCOUT(fmt::format("parent {} add prim idx {} as variant: ", parent, current));
          if (_variantPrims.count(current)) {
            DCOUT("??? prim idx already set " << current);
          } else {
            _variantPrims[current] =  variantPrim.value();
            _variantPrimChildren[parent].push_back(current);
          }
        } else {
          if (_config.allow_unknown_prims && is_unsupported_prim) {
            // Try to reconstruct as Model
            variantPrim = ReconstructPrimFromTypeName(
                "Model", primTypeName, variantPrimName, node, specifier.value(), primChildren, properties,
                psmap, primMeta);

            if (variantPrim) {
              // Prim name
              variantPrim.value().element_path() =
                  elemPath;  // FIXME: Use variantPrimName?

              // Prim Specifier
              variantPrim.value().specifier() = specifier.value();

              // Store variantPrim to temporary buffer.
              DCOUT(fmt::format("parent {} add prim idx {} as variant: ", parent, current));
              if (_variantPrims.count(current)) {
                DCOUT("??? prim idx already set " << current);
              } else {
                _variantPrims[current] = variantPrim.value();
                _variantPrimChildren[parent].push_back(current);
              }
            } else {
              return false;
            }
          } else {
            return false;
          }
        }
#else
        PrimSpec variantPrimSpec;
        variantPrimSpec.typeName() = primTypeName;
        variantPrimSpec.name() = prim_name;

        prim::PropertyMap props;
        if (!BuildPropertyMap(node.GetChildren(), psmap, &props)) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to build PropertyMap.");
        }
        variantPrimSpec.props() = props;
        variantPrimSpec.metas() = primMeta;

        // Store variantPrimSpec to temporary buffer.
        DCOUT(fmt::format("parent {} add primspec idx {} as variant: ", parent, current));
        if (_variantPrimSpecs.count(current)) {
          DCOUT("??? prim idx already set " << current);
        } else {
          _variantPrimSpecs[current] = variantPrimSpec;
          _variantPrimChildren[parent].push_back(current);
        }

#endif
      }

      break;
    }
    case SpecType::Attribute: {
      if (is_parent_variant) {
        nonstd::optional<Path> path = GetPath(spec.path_index);

        if (!path) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid PathIndex.");
        }

        Property prop;
        if (!ParseProperty(spec.spec_type, fvs, &prop)) {
          PUSH_ERROR_AND_RETURN_TAG(kTag,
                                    fmt::format("Failed to parse Attribut: {}.",
                                                path.value().prop_part()));
        }

        // Parent Prim is not yet reconstructed, so store info to temporary
        // buffer _variantAttributeNodes.
        _variantProps[current] = {path.value(), prop};
        _variantPropChildren[parent].push_back(current);

        DCOUT(
            fmt::format("parent {} current [{}] Parsed Attribute {} under Variant. PathIndex {}",
                        parent, current, path.value().prop_part(), spec.path_index));

      } else {
        // Maybe parent is Class/Over, or inherited
        PUSH_WARN(
            "TODO: SpecTypeAttribute(in conjunction with Class/Over specifier, "
            "or inherited?)");
      }
      break;
    }
    case SpecType::Connection:
    case SpecType::Relationship:
    case SpecType::RelationshipTarget: {
      PUSH_ERROR_AND_RETURN_TAG(
          kTag, fmt::format("TODO: Unsupported/Unimplemented SpecType: {}.",
                            to_string(spec.spec_type)));
      break;
    }
    case SpecType::Expression:
    case SpecType::Mapper:
    case SpecType::MapperArg: {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Unsupported SpecType: {}.",
                                                  to_string(spec.spec_type)));
      break;
    }
    case SpecType::Unknown:
    case SpecType::Invalid: {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "[InternalError] Invalid SpecType.");
      break;
    }
  }

  return true;
}

//
// TODO: rewrite code in bottom-up manner
//
bool USDCReader::Impl::ReconstructPrimRecursively(
    int parent, int current, Prim *parentPrim, int level,
    const PathIndexToSpecIndexMap &psmap, Stage *stage) {
  if (level > int32_t(_config.kMaxPrimNestLevel)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Prim hierarchy is too deep.");
  }

  DCOUT("ReconstructPrimRecursively: parent = "
        << std::to_string(parent) << ", current = " << current
        << ", level = " << std::to_string(level));

  if ((current < 0) || (current >= int(_nodes.size()))) {
    PUSH_ERROR("Invalid current node id: " + std::to_string(current) +
               ". Must be in range [0, " + std::to_string(_nodes.size()) + ")");
    return false;
  }

  //
  // TODO: Use bottom-up reconstruction(traverse child first)
  //

  // null : parent node is Property or other Spec type.
  // non-null : parent node is Prim
  Prim *currPrimPtr = nullptr;
  nonstd::optional<Prim> prim;

  bool is_parent_variant = _variantPrims.count(parent);

  if (!ReconstructPrimNode(parent, current, level, is_parent_variant, psmap,
                           stage, &prim)) {
    return false;
  }

  if (prim) {
    currPrimPtr = &(prim.value());
  }

  // Traverse children
  {
    const crate::CrateReader::Node &node = _nodes[size_t(current)];
    DCOUT("node.Children.size = " << node.GetChildren().size());
    for (size_t i = 0; i < node.GetChildren().size(); i++) {
      DCOUT("Reconstuct Prim children: " << i << " / "
                                         << node.GetChildren().size());
      if (!ReconstructPrimRecursively(current, int(node.GetChildren()[i]),
                                      currPrimPtr, level + 1, psmap, stage)) {
        return false;
      }
      DCOUT("DONE Reconstuct Prim children: " << i << " / "
                                              << node.GetChildren().size());
    }
  }

  //
  // Reonstruct variant
  //
  DCOUT(fmt::format("parent {}, current {}", parent, current));

  DCOUT(fmt::format("  has variant properties {}, has variant children {}",
    _variantPropChildren.count(current),
    _variantPrimChildren.count(current)));

  if (_variantPropChildren.count(current)) {

    // - parentPrim
    //   - variantPrim(SpecTypeVariant) <- current
    //     - variant property(SpecTypeAttribute)

    //
    // `current` must be VariantPrim and `parentPrim` should exist
    //
    if (!_variantPrims.count(current)) {
      PUSH_ERROR_AND_RETURN("Internal error: variant attribute is not a child of VariantPrim.");
    }

    if (!parentPrim) {
      PUSH_ERROR_AND_RETURN("Internal error: parentPrim should exist.");
    }

    const Prim &variantPrim = _variantPrims.at(current);

    DCOUT("variant prim name: " << variantPrim.element_name());


    // element_name must be variant: "{variant=value}"
    if (!is_variantElementName(variantPrim.element_name())) {
      PUSH_ERROR_AND_RETURN("Corrupted Crate. VariantAttribute is not the child of VariantPrim.");
    }

    std::array<std::string, 2> toks;
    if (!tokenize_variantElement(variantPrim.element_name(), &toks)) {
      PUSH_ERROR_AND_RETURN("Invalid variant element_name.");
    }

    std::string variantSetName = toks[0];
    std::string variantName = toks[1];

    Variant variant;

    for (const auto &item : _variantPropChildren.at(current)) {
      // item should exist in _variantProps.
      if (!_variantProps.count(item)) {
        PUSH_ERROR_AND_RETURN("Internal error: variant Property not found.");
      }
      const std::pair<Path, Property> &pp = _variantProps.at(item);

      std::string prop_name = std::get<0>(pp).prop_part();
      DCOUT(fmt::format("  node_index = {}, prop name {}", item, prop_name));

      variant.properties()[prop_name] = std::get<1>(pp);
    }

    VariantSet &vs = parentPrim->variantSets()[variantSetName];

    if (vs.name.empty()) {
      vs.name = variantSetName;
    }
    vs.variantSet[variantName] = variant;

  }

  if (_variantPrimChildren.count(current)) {

    // - currentPrim <- current
    //   - variant Prim children

    if (!prim) {
      PUSH_ERROR_AND_RETURN("Internal error: must be Prim.");
    }

    DCOUT(fmt::format("{} has variant Prim ", prim->element_name()));


    for (const auto &item : _variantPrimChildren.at(current)) {

      if (!_variantPrims.count(item)) {
        PUSH_ERROR_AND_RETURN("Internal error: variant Prim children not found.");
      }

      const Prim &vp = _variantPrims.at(item);

      DCOUT(fmt::format("  variantPrim name {}", vp.element_name()));

      // element_name must be variant: "{variant=value}"
      if (!is_variantElementName(vp.element_name())) {
        PUSH_ERROR_AND_RETURN("Corrupted Crate. Variant Prim has invalid element_name.");
      }

      std::array<std::string, 2> toks;
      if (!tokenize_variantElement(vp.element_name(), &toks)) {
        PUSH_ERROR_AND_RETURN("Invalid variant element_name.");
      }

      std::string variantSetName = toks[0];
      std::string variantName = toks[1];

      VariantSet &vs = prim->variantSets()[variantSetName];

      if (vs.name.empty()) {
        vs.name = variantSetName;
      }
      vs.variantSet[variantName].metas() = vp.metas();
      DCOUT("# of primChildren = " << vp.children().size());
      vs.variantSet[variantName].primChildren() = std::move(vp.children());

    }
  }

  if (parent == 0) {  // root prim
    if (prim) {
      stage->root_prims().emplace_back(std::move(prim.value()));
    }
  } else {
    if (_variantPrims.count(parent)) {
      // Add to variantPrim
      DCOUT("parent is variantPrim: " << parent);
      if (!prim) {
        // FIXME: Validate current should be Prim.
        PUSH_WARN("parent is variantPrim, but current is not Prim.");
      } else {
        DCOUT("Adding prim to child...");
        Prim &vp = _variantPrims.at(parent);
        vp.children().emplace_back(std::move(prim.value()));
      }
    } else if (prim && parentPrim) {
      // Add to parent prim.
      parentPrim->children().emplace_back(std::move(prim.value()));
    }
  }

  return true;
}

bool USDCReader::Impl::ReconstructStage(Stage *stage) {

  // format test
  DCOUT(fmt::format("# of Paths = {}", crate_reader->NumPaths()));

  if (crate_reader->NumNodes() == 0) {
    PUSH_WARN("Empty scene.");
    return true;
  }

  // TODO: Directly access data in crate_reader.
  _nodes = crate_reader->GetNodes();
  _specs = crate_reader->GetSpecs();
  _fields = crate_reader->GetFields();
  _fieldset_indices = crate_reader->GetFieldsetIndices();
  _paths = crate_reader->GetPaths();
  _elemPaths = crate_reader->GetElemPaths();
  _live_fieldsets = crate_reader->GetLiveFieldSets();

  PathIndexToSpecIndexMap
      path_index_to_spec_index_map;  // path_index -> spec_index

  {
    for (size_t i = 0; i < _specs.size(); i++) {
      if (_specs[i].path_index.value == ~0u) {
        continue;
      }

      // path_index should be unique.
      if (path_index_to_spec_index_map.count(_specs[i].path_index.value) != 0) {
        PUSH_ERROR_AND_RETURN("Multiple PathIndex found in Crate data.");
      }

      DCOUT(fmt::format("path index[{}] -> spec index [{}]",
                        _specs[i].path_index.value, uint32_t(i)));
      path_index_to_spec_index_map[_specs[i].path_index.value] = uint32_t(i);
    }
  }

  stage->root_prims().clear();

  int root_node_id = 0;
  bool ret = ReconstructPrimRecursively(/* no further root for root_node */ -1,
                                        root_node_id, /* root Prim */ nullptr,
                                        /* level */ 0,
                                        path_index_to_spec_index_map, stage);

  if (!ret) {
    PUSH_ERROR_AND_RETURN("Failed to reconstruct Stage(Prim hierarchy)");
  }

  stage->compute_absolute_prim_path_and_assign_prim_id();

  return true;
}

bool USDCReader::Impl::ReconstructPrimSpecRecursively(
    int parent, int current, PrimSpec *parentPrimSpec, int level,
    const PathIndexToSpecIndexMap &psmap, Layer *layer) {
  if (level > int32_t(_config.kMaxPrimNestLevel)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "PrimSpec hierarchy is too deep.");
  }

  DCOUT("ReconstructPrimRecursively: parent = "
        << std::to_string(parent) << ", current = " << current
        << ", level = " << std::to_string(level));

  if ((current < 0) || (current >= int(_nodes.size()))) {
    PUSH_ERROR("Invalid current node id: " + std::to_string(current) +
               ". Must be in range [0, " + std::to_string(_nodes.size()) + ")");
    return false;
  }

  // TODO: Refactor

  // null : parent node is Property or other Spec type.
  // non-null : parent node is PrimSpec
  PrimSpec *currPrimSpecPtr = nullptr;
  nonstd::optional<PrimSpec> primspec;

  // Assume parent node is already processed.
  bool is_parent_variant = _variantPrims.count(parent);

  if (!ReconstructPrimSpecNode(parent, current, level, is_parent_variant, psmap,
                           layer, &primspec)) {
    return false;
  }

  if (primspec) {
    currPrimSpecPtr = &(primspec.value());
  }

  {
    const crate::CrateReader::Node &node = _nodes[size_t(current)];
    DCOUT("node.Children.size = " << node.GetChildren().size());
    for (size_t i = 0; i < node.GetChildren().size(); i++) {
      DCOUT("Reconstuct Prim children: " << i << " / "
                                         << node.GetChildren().size());
      if (!ReconstructPrimSpecRecursively(current, int(node.GetChildren()[i]),
                                      currPrimSpecPtr, level + 1, psmap, layer)) {
        return false;
      }
      DCOUT("DONE Reconstuct PrimSpec children: " << i << " / "
                                              << node.GetChildren().size());
    }
  }

  //
  // Reonstruct variant
  //
  DCOUT(fmt::format("parent {}, current {}", parent, current));

  DCOUT(fmt::format("  has variant properties {}, has variant children {}",
    _variantPropChildren.count(current),
    _variantPrimChildren.count(current)));

  if (_variantPropChildren.count(current)) {

    // - parentPrim
    //   - variantPrim(SpecTypeVariant) <- current
    //     - variant property(SpecTypeAttribute)

    //
    // `current` must be VariantPrim and `parentPrim` should exist
    //
    if (!_variantPrims.count(current)) {
      PUSH_ERROR_AND_RETURN("Internal error: variant attribute is not a child of VariantPrim.");
    }

    if (!parentPrimSpec) {
      PUSH_ERROR_AND_RETURN("Internal error: parentPrimSpec should exist.");
    }

    const Prim &variantPrim = _variantPrims.at(current);

    DCOUT("variant prim name: " << variantPrim.element_name());


    // element_name must be variant: "{variant=value}"
    if (!is_variantElementName(variantPrim.element_name())) {
      PUSH_ERROR_AND_RETURN("Corrupted Crate. VariantAttribute is not the child of VariantPrim.");
    }

    std::array<std::string, 2> toks;
    if (!tokenize_variantElement(variantPrim.element_name(), &toks)) {
      PUSH_ERROR_AND_RETURN("Invalid variant element_name.");
    }

    std::string variantSetName = toks[0];
    std::string variantName = toks[1];

    PrimSpec variant;

    for (const auto &item : _variantPropChildren.at(current)) {
      // item should exist in _variantProps.
      if (!_variantProps.count(item)) {
        PUSH_ERROR_AND_RETURN("Internal error: variant Property not found.");
      }
      const std::pair<Path, Property> &pp = _variantProps.at(item);

      std::string prop_name = std::get<0>(pp).prop_part();
      DCOUT(fmt::format("  node_index = {}, prop name {}", item, prop_name));

      variant.props()[prop_name] = std::get<1>(pp);
    }

    VariantSetSpec &vs = parentPrimSpec->variantSets()[variantSetName];

    if (vs.name.empty()) {
      vs.name = variantSetName;
    }
    vs.variantSet[variantName] = variant;

  }

  if (_variantPrimChildren.count(current)) {

    // - currentPrim <- current
    //   - variant Prim children

    if (!primspec) {
      PUSH_ERROR_AND_RETURN("Internal error: must be Prim.");
    }

    DCOUT(fmt::format("{} has variant PrimSpec ", primspec->name()));


    for (const auto &item : _variantPrimChildren.at(current)) {

      if (!_variantPrimSpecs.count(item)) {
        PUSH_ERROR_AND_RETURN("Internal error: variant Prim children not found.");
      }

      const PrimSpec &vp = _variantPrimSpecs.at(item);

      DCOUT(fmt::format("  variantPrim name {}", vp.name()));

      // element_name must be variant: "{variant=value}"
      if (!is_variantElementName(vp.name())) {
        PUSH_ERROR_AND_RETURN("Corrupted Crate. Variant Prim has invalid element_name.");
      }

      std::array<std::string, 2> toks;
      if (!tokenize_variantElement(vp.name(), &toks)) {
        PUSH_ERROR_AND_RETURN("Invalid variant element_name.");
      }

      std::string variantSetName = toks[0];
      std::string variantName = toks[1];

      VariantSetSpec &vs = primspec->variantSets()[variantSetName];

      if (vs.name.empty()) {
        vs.name = variantSetName;
      }
      vs.variantSet[variantName].metas() = vp.metas();
      DCOUT("# of primChildren = " << vp.children().size());
      vs.variantSet[variantName].children() = std::move(vp.children());

    }
  }

  if (parent == 0) {  // root prim
    if (primspec) {
      layer->primspecs()[primspec.value().name()] = std::move(primspec.value());
    }
  } else {
    if (_variantPrimSpecs.count(parent)) {
      // Add to variantPrim
      DCOUT("parent is variantPrim: " << parent);
      if (!primspec) {
        // FIXME: Validate current should be Prim.
        PUSH_WARN("parent is variantPrim, but current is not Prim.");
      } else {
        DCOUT("Adding prim to child...");
        PrimSpec &vps = _variantPrimSpecs.at(parent);
        vps.children().emplace_back(std::move(primspec.value()));
      }
    } else if (primspec && parentPrimSpec) {
      // Add to parent prim.
      parentPrimSpec->children().emplace_back(std::move(primspec.value()));
    }
  }

  return true;
}

bool USDCReader::Impl::ToLayer(Layer *layer) {

  if (!layer) {
    PUSH_ERROR_AND_RETURN("`layer` argument is nullptr.");
  }

  // format test
  DCOUT(fmt::format("# of Paths = {}", crate_reader->NumPaths()));

  if (crate_reader->NumNodes() == 0) {
    PUSH_WARN("Empty scene.");
    return true;
  }

  // TODO: Directly access data in crate_reader.
  _nodes = crate_reader->GetNodes();
  _specs = crate_reader->GetSpecs();
  _fields = crate_reader->GetFields();
  _fieldset_indices = crate_reader->GetFieldsetIndices();
  _paths = crate_reader->GetPaths();
  _elemPaths = crate_reader->GetElemPaths();
  _live_fieldsets = crate_reader->GetLiveFieldSets();

  PathIndexToSpecIndexMap
      path_index_to_spec_index_map;  // path_index -> spec_index

  {
    for (size_t i = 0; i < _specs.size(); i++) {
      if (_specs[i].path_index.value == ~0u) {
        continue;
      }

      // path_index should be unique.
      if (path_index_to_spec_index_map.count(_specs[i].path_index.value) != 0) {
        PUSH_ERROR_AND_RETURN("Multiple PathIndex found in Crate data.");
      }

      DCOUT(fmt::format("path index[{}] -> spec index [{}]",
                        _specs[i].path_index.value, uint32_t(i)));
      path_index_to_spec_index_map[_specs[i].path_index.value] = uint32_t(i);
    }
  }

  layer->primspecs().clear();

  int root_node_id = 0;
  bool ret = ReconstructPrimSpecRecursively(/* no further root for root_node */ -1,
                                        root_node_id, /* root Prim */ nullptr,
                                        /* level */ 0,
                                        path_index_to_spec_index_map, layer);

  if (!ret) {
    PUSH_ERROR_AND_RETURN("Failed to reconstruct Layer(PrimSpec hierarchy)");
  }

  //stage->compute_absolute_prim_path_and_assign_prim_id();

  return true;
}

bool USDCReader::Impl::ReadUSDC() {
  if (crate_reader) {
    delete crate_reader;
  }

  // TODO: Setup CrateReaderConfig.
  crate::CrateReaderConfig config;

  // Transfer settings
  config.numThreads = _config.numThreads;

  size_t sz_mb = _config.kMaxAllowedMemoryInMB;
  if (sizeof(size_t) == 4) {
    // 32bit
    // cap to 2GB
    sz_mb = (std::min)(size_t(1024 * 2), sz_mb);

    config.maxMemoryBudget = sz_mb * 1024 * 1024;
  } else {
    config.maxMemoryBudget = _config.kMaxAllowedMemoryInMB * 1024ull * 1024ull;
  }

  crate_reader = new crate::CrateReader(_sr, config);

  _warn.clear();
  _err.clear();

  if (!crate_reader->ReadBootStrap()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();
    return false;
  }

  if (!crate_reader->ReadTOC()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();
    return false;
  }

  // Read known sections

  if (!crate_reader->ReadTokens()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();
    return false;
  }

  if (!crate_reader->ReadStrings()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();
    return false;
  }

  if (!crate_reader->ReadFields()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();
    return false;
  }

  if (!crate_reader->ReadFieldSets()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();
    return false;
  }

  if (!crate_reader->ReadPaths()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();
    return false;
  }

  if (!crate_reader->ReadSpecs()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();
    return false;
  }

  // TODO(syoyo): Read unknown sections

  ///
  /// Reconstruct C++ representation of USD scene graph.
  ///
  DCOUT("BuildLiveFieldSets");
  if (!crate_reader->BuildLiveFieldSets()) {
    _warn = crate_reader->GetWarning();
    _err = crate_reader->GetError();

    return false;
  }

  _warn += crate_reader->GetWarning();
  _err += crate_reader->GetError();

  DCOUT("Read Crate.");

  return true;
}

//
// -- Interface --
//
USDCReader::USDCReader(StreamReader *sr, const USDCReaderConfig &config) {
  impl_ = new USDCReader::Impl(sr, config);
}

USDCReader::~USDCReader() {
  delete impl_;
  impl_ = nullptr;
}

void USDCReader::set_reader_config(const USDCReaderConfig &config) {
  impl_->set_reader_config(config);
}

const USDCReaderConfig USDCReader::get_reader_config() const {
  return impl_->get_reader_config();
}

bool USDCReader::ReconstructStage(Stage *stage) {
  DCOUT("Reconstruct Stage.");
  return impl_->ReconstructStage(stage);
}

bool USDCReader::get_as_layer(Layer *layer) {
  return impl_->ToLayer(layer);
}

std::string USDCReader::GetError() { return impl_->GetError(); }

std::string USDCReader::GetWarning() { return impl_->GetWarning(); }

bool USDCReader::ReadUSDC() { return impl_->ReadUSDC(); }

}  // namespace usdc
}  // namespace tinyusdz

#else  // TINYUSDZ_DISABLE_MODULE_USDC_READER

namespace tinyusdz {
namespace usdc {

//
// -- Interface --
//
USDCReader::USDCReader(StreamReader *sr, USDCReaderConfig &config) {
  (void)sr;
  (void)config;
}

USDCReader::~USDCReader() {}

void USDCReader::set_reader_config(const USDCReaderConfig &config) {
  (void)config;
}

const USDCReaderConfig USDCReader::get_reader_config() const {
  return USDCReaderConfig();
}

bool USDCReader::ReconstructStage(Stage *stage) {
  (void)scene;
  DCOUT("Reconstruct Stage.");
  return false;
}

bool USDCReader::get_as_layer(Layer *layer) {
  (void)layer;
  return false;
}

std::string USDCReader::GetError() {
  return "USDC reader feature is disabled in this build.\n";
}

std::string USDCReader::GetWarning() { return ""; }

}  // namespace usdc
}  // namespace tinyusdz

#endif  // TINYUSDZ_DISABLE_MODULE_USDC_READER

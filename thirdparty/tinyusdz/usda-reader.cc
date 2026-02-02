// SPDX-License-Identifier: Apache 2.0
// Copyright 2021 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// USDA reader
// TODO:
//   - [ ] Refactor and unify Prim and PrimSpec related code.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stack>

#include "ascii-parser.hh"
//#include "asset-resolution.hh"
#include "usdGeom.hh"
#include "usdSkel.hh"
#if defined(__wasi__)
#else
#include <mutex>
#include <thread>
#endif
#include <vector>

#include "usda-reader.hh"

//
#if !defined(TINYUSDZ_DISABLE_MODULE_USDA_READER)

//

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// external

#include "nonstd/expected.hpp"
#include "nonstd/optional.hpp"

//

#ifdef __clang__
#pragma clang diagnostic pop
#endif

//

// Tentative
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#include "io-util.hh"
#include "math-util.inc"
#include "pprinter.hh"
#include "prim-types.hh"
#include "prim-reconstruct.hh"
#include "primvar.hh"
#include "str-util.hh"
#include "stream-reader.hh"
#include "tinyusdz.hh"
#include "usdShade.hh"
#include "value-pprint.hh"
#include "value-types.hh"
#include "tiny-format.hh"

#include "common-macros.inc"

namespace tinyusdz {

namespace prim {

// template specialization forward decls.
// implimentations will be located in prim-reconstruct.cc
#define RECONSTRUCT_PRIM_DECL(__ty) template<> bool ReconstructPrim<__ty>(const Specifier &spec, const PropertyMap &, const ReferenceList &, __ty *, std::string *, std::string *, const PrimReconstructOptions &)

RECONSTRUCT_PRIM_DECL(Xform);
RECONSTRUCT_PRIM_DECL(Model);
RECONSTRUCT_PRIM_DECL(Scope);
RECONSTRUCT_PRIM_DECL(Skeleton);
RECONSTRUCT_PRIM_DECL(SkelRoot);
RECONSTRUCT_PRIM_DECL(SkelAnimation);
RECONSTRUCT_PRIM_DECL(BlendShape);
RECONSTRUCT_PRIM_DECL(DomeLight);
RECONSTRUCT_PRIM_DECL(SphereLight);
RECONSTRUCT_PRIM_DECL(CylinderLight);
RECONSTRUCT_PRIM_DECL(DiskLight);
RECONSTRUCT_PRIM_DECL(DistantLight);
RECONSTRUCT_PRIM_DECL(GPrim);
RECONSTRUCT_PRIM_DECL(GeomMesh);
RECONSTRUCT_PRIM_DECL(GeomSubset);
RECONSTRUCT_PRIM_DECL(GeomSphere);
RECONSTRUCT_PRIM_DECL(GeomPoints);
RECONSTRUCT_PRIM_DECL(GeomCone);
RECONSTRUCT_PRIM_DECL(GeomCube);
RECONSTRUCT_PRIM_DECL(GeomCylinder);
RECONSTRUCT_PRIM_DECL(GeomCapsule);
RECONSTRUCT_PRIM_DECL(GeomBasisCurves);
RECONSTRUCT_PRIM_DECL(GeomNurbsCurves);
RECONSTRUCT_PRIM_DECL(GeomCamera);
RECONSTRUCT_PRIM_DECL(PointInstancer);
RECONSTRUCT_PRIM_DECL(Material);
RECONSTRUCT_PRIM_DECL(Shader);
RECONSTRUCT_PRIM_DECL(NodeGraph);

#undef RECONSTRUCT_PRIM_DECL

} // namespace prim

namespace usda {

constexpr auto kTag = "[USDA]";

namespace {

// intermediate data structure for VariantSet stmt
struct VariantNode {
  PrimMeta metas;
  std::map<std::string, Property> props;
  std::vector<int64_t> primChildren;
};

struct PrimNode {
  value::Value prim; // stores typed Prim value. Xform, GeomMesh, ...
  std::string elementName;
  std::string typeName; // Prim's typeName

  int64_t parent{-1};            // -1 = root node
  //bool parent_is_variant{false}; // True when this Prim is defined under variantSet stmt.
  std::vector<size_t> children;  // index to USDAReader._prims[] of childPrims. it contains variant's primChildren also.

  std::map<std::string, std::map<std::string, VariantNode>> variantNodeMap;
};

// For USD scene read for composition(read by references, subLayers, payloads)
struct PrimSpecNode {
  PrimSpec primSpec;

  int64_t parent{-1};            // -1 = root node
  //bool parent_is_variant{false}; // True when this Prim is defined under variantSet stmt.
  std::vector<size_t> children;  // index to USDAReader._primspecs[]

  std::map<std::string, std::map<std::string, VariantNode>> variantNodeMap;
};

// TODO: Move to prim-types.hh?

template <typename T>
struct PrimTypeTraits;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-const-variable"
#endif

#define DEFINE_PRIM_TYPE(__dty, __name, __tyid)    \
  template <>                                      \
  struct PrimTypeTraits<__dty> {                    \
    using primt_type = __dty;                      \
    static constexpr uint32_t type_id = __tyid;    \
    static constexpr auto prim_type_name = __name; \
  }

DEFINE_PRIM_TYPE(Model, "Model", value::TYPE_ID_MODEL);

DEFINE_PRIM_TYPE(Xform, kGeomXform, value::TYPE_ID_GEOM_XFORM);
DEFINE_PRIM_TYPE(GeomMesh, kGeomMesh, value::TYPE_ID_GEOM_MESH);
DEFINE_PRIM_TYPE(GeomPoints, kGeomPoints, value::TYPE_ID_GEOM_POINTS);
DEFINE_PRIM_TYPE(GeomSphere, kGeomSphere, value::TYPE_ID_GEOM_SPHERE);
DEFINE_PRIM_TYPE(GeomCube, kGeomCube, value::TYPE_ID_GEOM_CUBE);
DEFINE_PRIM_TYPE(GeomCone, kGeomCone, value::TYPE_ID_GEOM_CONE);
DEFINE_PRIM_TYPE(GeomCapsule, kGeomCapsule, value::TYPE_ID_GEOM_CAPSULE);
DEFINE_PRIM_TYPE(GeomCylinder, kGeomCylinder, value::TYPE_ID_GEOM_CYLINDER);
DEFINE_PRIM_TYPE(GeomBasisCurves, kGeomBasisCurves,
                 value::TYPE_ID_GEOM_BASIS_CURVES);
DEFINE_PRIM_TYPE(GeomNurbsCurves, kGeomNurbsCurves,
                 value::TYPE_ID_GEOM_NURBS_CURVES);
DEFINE_PRIM_TYPE(GeomSubset, kGeomSubset, value::TYPE_ID_GEOM_GEOMSUBSET);
DEFINE_PRIM_TYPE(SphereLight, kSphereLight, value::TYPE_ID_LUX_SPHERE);
DEFINE_PRIM_TYPE(DomeLight, kDomeLight, value::TYPE_ID_LUX_DOME);
DEFINE_PRIM_TYPE(DiskLight, kDiskLight, value::TYPE_ID_LUX_DISK);
DEFINE_PRIM_TYPE(DistantLight, kDistantLight, value::TYPE_ID_LUX_DISTANT);
DEFINE_PRIM_TYPE(CylinderLight,  kCylinderLight, value::TYPE_ID_LUX_CYLINDER);
DEFINE_PRIM_TYPE(Material, kMaterial, value::TYPE_ID_MATERIAL);
DEFINE_PRIM_TYPE(Shader, kShader, value::TYPE_ID_SHADER);
DEFINE_PRIM_TYPE(NodeGraph, kNodeGraph, value::TYPE_ID_NODEGRAPH);
DEFINE_PRIM_TYPE(SkelRoot, kSkelRoot, value::TYPE_ID_SKEL_ROOT);
DEFINE_PRIM_TYPE(Skeleton, kSkeleton, value::TYPE_ID_SKELETON);
DEFINE_PRIM_TYPE(SkelAnimation, kSkelAnimation, value::TYPE_ID_SKELANIMATION);
DEFINE_PRIM_TYPE(BlendShape, kBlendShape, value::TYPE_ID_BLENDSHAPE);
DEFINE_PRIM_TYPE(GeomCamera, kGeomCamera, value::TYPE_ID_GEOM_CAMERA);
DEFINE_PRIM_TYPE(PointInstancer, kPointInstancer, value::TYPE_ID_GEOM_POINT_INSTANCER);
DEFINE_PRIM_TYPE(Scope, "Scope", value::TYPE_ID_SCOPE);

DEFINE_PRIM_TYPE(GPrim, "GPrim", value::TYPE_ID_GPRIM);

#ifdef __clang__
#pragma clang diagnostic pop
#endif

}  // namespace

class VariableDef {
 public:
  std::string type;
  std::string name;

  VariableDef() = default;

  VariableDef(const std::string &t, const std::string &n) : type(t), name(n) {}

  VariableDef(const VariableDef &rhs) = default;

  VariableDef &operator=(const VariableDef &rhs) {
    type = rhs.type;
    name = rhs.name;

    return *this;
  }
};

inline bool hasConnect(const std::string &str) {
  return endsWith(str, ".connect");
}

inline bool hasInputs(const std::string &str) {
  return startsWith(str, "inputs:");
}

inline bool hasOutputs(const std::string &str) {
  return startsWith(str, "outputs:");
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

class USDAReader::Impl {
 private:
  Stage _stage;

 public:
  Impl(StreamReader *sr) { _parser.SetStream(sr); }

#if 0 // TODO: Remove
  // Return the flag if the .usda is read from `references`
  bool IsReferenced() { return _referenced; }

  // Return the flag if the .usda is read from `subLayers`
  bool IsSubLayered() { return _sub_layered; }

  // Return the flag if the .usda is read from `payload`
  bool IsPayloaded() { return _payloaded; }

  // Return true if the .udsa is read in the top layer(stage)
  bool IsToplevel() {
    return !IsReferenced() && !IsSubLayered() && !IsPayloaded();
  }
#endif

  void SetBaseDir(const std::string &str) { _base_dir = str; }

#if 0
  ///
  /// True: create PrimSpec instead of typed Prim.
  /// Set true if you do USD composition.
  ///
  void set_primspec_mode(bool onoff) { _primspec_mode = onoff; }
#endif

  void set_reader_config(const USDAReaderConfig &config) {
    _config = config;
  }

  const USDAReaderConfig get_reader_config() const {
    return _config;
  }

  std::string GetCurrentPath() {
    if (_path_stack.empty()) {
      return "/";
    }

    return _path_stack.top();
  }

  bool PathStackDepth() { return _path_stack.size(); }

  void PushPath(const std::string &p) { _path_stack.push(p); }

  void PopPath() {
    if (!_path_stack.empty()) {
      _path_stack.pop();
    }
  }

  void PushError(const std::string &s) {
    _err += s;
  }

  void PushWarn(const std::string &s) {
    _warn += s;
  }

  template <typename T>
  bool ReconstructPrim(
      const Specifier &spec,
      const prim::PropertyMap &properties,
      const prim::ReferenceList &references,
      T *out);


  template <typename T>
  bool RegisterReconstructCallback() {
    _parser.RegisterPrimConstructFunction(
        PrimTypeTraits<T>::prim_type_name,
        [&](const Path &full_path, const Specifier spec, const std::string &_primTypeName, const Path &prim_name, const int64_t primIdx,
            const int64_t parentPrimIdx,
            const prim::PropertyMap &properties,
            const ascii::AsciiParser::PrimMetaMap &in_meta,
            const ascii::AsciiParser::VariantSetList &in_variants)
            -> nonstd::expected<bool, std::string> {

          std::string primTypeName = _primTypeName;
          if (primTypeName == "__AnyType__") {
            primTypeName = ""; // Make empty
          }

          if (!prim_name.is_valid()) {
            return nonstd::make_unexpected("Invalid Prim name: " +
                                           prim_name.full_path_name());
          }
          if (prim_name.is_absolute_path() || prim_name.is_root_path()) {
            return nonstd::make_unexpected(
                "Prim name should not starts with '/' or contain `/`: Prim "
                "name = " +
                prim_name.full_path_name());
          }

          if (!prim_name.prop_part().empty()) {
            return nonstd::make_unexpected(
                "Prim path should not contain property part(`.`): Prim name "
                "= " +
                prim_name.full_path_name());
          }

          if (primIdx < 0) {
            return nonstd::make_unexpected(
                "Unexpected primIdx value. primIdx must be positive.");
          }

          T prim;

          if (!ReconstructPrimMeta(in_meta, &prim.meta)) {
            return nonstd::make_unexpected(
                "Failed to process Prim metadataum.");
          }

          DCOUT("primType = " << value::TypeTraits<T>::type_name()
                              << ", node.size "
                              << std::to_string(_prim_nodes.size())
                              << ", primIdx = " << primIdx
                              << ", parentPrimIdx = " << parentPrimIdx);

          DCOUT("full_path = " << full_path.full_path_name());
          DCOUT("primName = " << prim_name.full_path_name());

          prim::ReferenceList references;
          if (prim.meta.references) {
            references = prim.meta.references.value();
          }

          bool ret = ReconstructPrim<T>(spec, properties, references, &prim);

          if (!ret) {
            return nonstd::make_unexpected("Failed to reconstruct Prim: " +
                                           prim_name.full_path_name());
          }

          prim.spec = spec;
          prim.name = prim_name.prim_part();

          //
          // variants
          // NOTE: variantChildren setup is delayed. It will be processed in ConstructPrimSpecTreeRec
          //
          std::map<std::string, std::map<std::string, VariantNode>> variantSets;
          for (const auto &variantContext : in_variants) {
            const std::string variant_name = variantContext.first;

            // Convert VariantContent -> VariantNode
            std::map<std::string, VariantNode> variantNodes;
            for (const auto &item : variantContext.second) {
              VariantNode variant;
              if (!ReconstructPrimMeta(item.second.metas, &variant.metas)) {
                return nonstd::make_unexpected(fmt::format("Failed to process Prim metadataum in variantSet {} item {} ", variant_name, item.first));
              }
              variant.props = item.second.props;

              // child Prim should be already reconstructed.
              for (const auto &childPrimIdx : item.second.primIndices) {
                if (childPrimIdx < 0) {
                  return nonstd::make_unexpected(fmt::format("[InternalError] Invalid primIndex found within VariantSet."));
                }

                if (size_t(childPrimIdx) >= _prim_nodes.size()) {
                  return nonstd::make_unexpected(fmt::format("[InternalError] Invalid primIndex found within VariantSet. variantChildPrimIdsx {} Exceeds _prim_nodes.size() {}", childPrimIdx, _prim_nodes.size()));
                }

                variant.primChildren.push_back(childPrimIdx);

                //_prim_nodes[size_t(childPrimIdx)].parent_is_variant = true;
              }
              DCOUT("Add variant: " << item.first);
              variantNodes.emplace(item.first, std::move(variant));
            }

            DCOUT("Add variantSet: " << variant_name);
            variantSets.emplace(variant_name, std::move(variantNodes));
          }

          // Add to scene graph.
          // NOTE: Scene graph is constructed from bottom up manner(Children
          // first), so add this primIdx to parent's children.
          if (size_t(primIdx) >= _prim_nodes.size()) {
            _prim_nodes.resize(size_t(primIdx) + 1);
          }
          DCOUT("sz " << std::to_string(_prim_nodes.size())
                      << ", primIdx = " << primIdx);

          _prim_nodes[size_t(primIdx)].prim = std::move(prim);
          _prim_nodes[size_t(primIdx)].typeName = primTypeName;
          _prim_nodes[size_t(primIdx)].variantNodeMap = variantSets;


          // Store actual Prim typeName also for Model Prim type.
          // TODO: Find more better way.
          {
            value::Value *p = &(_prim_nodes[size_t(primIdx)].prim);
            Model *model = p->as<Model>();
            if (model) {
              DCOUT("Set prim typeName " << primTypeName << " to Model Prim[" << primIdx << "]");
              model->prim_type_name = primTypeName;
            }
          }

          DCOUT("prim[" << primIdx << "].ty = "
                        << _prim_nodes[size_t(primIdx)].prim.type_name());
          _prim_nodes[size_t(primIdx)].parent = parentPrimIdx;

          if (parentPrimIdx == -1) {
            _toplevel_prims.push_back(size_t(primIdx));
          } else {
            _prim_nodes[size_t(parentPrimIdx)].children.push_back(
                  size_t(primIdx));
          }

          return true;
        });

    return true;
  }

  void RegisterPrimSpecHandler() {
    _parser.RegisterPrimSpecFunction(
         [&](const Path &full_path, const Specifier spec, const std::string &typeName, const Path &prim_name, const int64_t primIdx,
            const int64_t parentPrimIdx,
            const prim::PropertyMap &properties,
            const ascii::AsciiParser::PrimMetaMap &in_meta,
            const ascii::AsciiParser::VariantSetList &in_variants)
            -> nonstd::expected<bool, std::string> {

          if (!prim_name.is_valid()) {
            return nonstd::make_unexpected("Invalid Prim name: " +
                                           prim_name.full_path_name());
          }
          if (prim_name.is_absolute_path() || prim_name.is_root_path()) {
            return nonstd::make_unexpected(
                "Prim name should not starts with '/' or contain `/`: Prim "
                "name = " +
                prim_name.full_path_name());
          }

          if (!prim_name.prop_part().empty()) {
            return nonstd::make_unexpected(
                "Prim path should not contain property part(`.`): Prim name "
                "= " +
                prim_name.full_path_name());
          }

          if (primIdx < 0) {
            return nonstd::make_unexpected(
                "Unexpected primIdx value. primIdx must be positive.");
          }

          if (prim_name.prim_part().empty()) {
            return nonstd::make_unexpected("Prim's name should not be empty ");
          }

          PrimSpec primspec;
          primspec.name() = prim_name.prim_part();
          primspec.specifier() = spec;
          primspec.typeName() = typeName;

          DCOUT("primspec name, primType = " << prim_name.prim_part() << ", " << typeName);

          if (!ReconstructPrimMeta(in_meta, &primspec.metas())) {
            return nonstd::make_unexpected(
                "Failed to process Prim metadataum.");
          }

          primspec.props() = properties;

          //
          // variants
          // NOTE: variantChildren setup is delayed. It will be processed ConstructPrimTreeRec()
          //
          std::map<std::string, std::map<std::string, VariantNode>> variantSets;
          for (const auto &variantContext : in_variants) {
            const std::string variant_name = variantContext.first;

            // Convert VariantContent -> VariantNode
            std::map<std::string, VariantNode> variantNodes;
            for (const auto &item : variantContext.second) {
              VariantNode variant;
              if (!ReconstructPrimMeta(item.second.metas, &variant.metas)) {
                return nonstd::make_unexpected(fmt::format("Failed to process Prim metadataum in variantSet {} item {} ", variant_name, item.first));
              }
              variant.props = item.second.props;

              // child Prim should be already reconstructed.
              for (const auto &childPrimIdx : item.second.primIndices) {
                if (childPrimIdx < 0) {
                  return nonstd::make_unexpected(fmt::format("[InternalError] Invalid primIndex found within VariantSet."));
                }

                if (size_t(childPrimIdx) >= _primspec_nodes.size()) {
                  return nonstd::make_unexpected(fmt::format("[InternalError] Invalid primIndex found within VariantSet. variantChildPrimIdsx {} Exceeds _prim_nodes.size() {}", childPrimIdx, _primspec_nodes.size()));
                }

                variant.primChildren.push_back(childPrimIdx);

                //_primspec_nodes[size_t(childPrimIdx)].parent_is_variant = true;
              }
              DCOUT("Add variant: " << item.first);
              variantNodes.emplace(item.first, std::move(variant));
            }

            DCOUT("Add variantSet: " << variant_name);
            variantSets.emplace(variant_name, std::move(variantNodes));
          }


          // Assign index for PrimSpec
          // TODO: Use sample id table(= _prim_nodes)

          if (size_t(primIdx) >= _primspec_nodes.size()) {
            _primspec_nodes.resize(size_t(primIdx) + 1);
          }
          DCOUT("sz " << std::to_string(_primspec_nodes.size())
                      << ", primIdx = " << primIdx);

          _primspec_nodes[size_t(primIdx)].primSpec = std::move(primspec);
          DCOUT("primspec[" << primIdx << "].ty = "
                        << _primspec_nodes[size_t(primIdx)].primSpec.typeName());
          _primspec_nodes[size_t(primIdx)].parent = parentPrimIdx;
          _primspec_nodes[size_t(primIdx)].variantNodeMap = variantSets;

          if (parentPrimIdx == -1) {
            _toplevel_primspecs.push_back(size_t(primIdx));
          } else {
            _primspec_nodes[size_t(parentPrimIdx)].children.push_back(
                size_t(primIdx));
            return true;
          }

          return true;
      }
    );

  }

  void StageMetaProcessor() {
    _parser.RegisterStageMetaProcessFunction(
        [&](const ascii::AsciiParser::StageMetas &metas) {
          DCOUT("StageMeta CB:");

          _stage.metas().doc = metas.doc;
          if (metas.upAxis) {
            _stage.metas().upAxis = metas.upAxis.value();
          }

          _stage.metas().comment = metas.comment;

          if (metas.subLayers.size()) {
            // TODO subLayer offset.
            std::vector<SubLayer> sublayers;
            for (size_t i = 0; i < metas.subLayers.size(); i++) {
              SubLayer sublayer;
              sublayer.assetPath = metas.subLayers[i];
              sublayers.push_back(sublayer);
            }
            _stage.metas().subLayers = sublayers;
          }

          _stage.metas().defaultPrim = metas.defaultPrim;
          if (metas.metersPerUnit) {
            _stage.metas().metersPerUnit = metas.metersPerUnit.value();
          }

          if (metas.kilogramsPerUnit) {
            _stage.metas().kilogramsPerUnit = metas.kilogramsPerUnit.value();
          }

          if (metas.timeCodesPerSecond) {
            _stage.metas().timeCodesPerSecond =
                metas.timeCodesPerSecond.value();
          }

          if (metas.startTimeCode) {
            _stage.metas().startTimeCode = metas.startTimeCode.value();
          }

          if (metas.endTimeCode) {
            _stage.metas().endTimeCode = metas.endTimeCode.value();
          }

          if (metas.framesPerSecond) {
            _stage.metas().framesPerSecond = metas.framesPerSecond.value();
          }

          if (metas.autoPlay) {
            _stage.metas().autoPlay = metas.autoPlay.value();
          }

          if (metas.playbackMode) {
            value::token tok = metas.playbackMode.value();
            if (tok.str() == "none") {
              _stage.metas().playbackMode = StageMetas::PlaybackMode::PlaybackModeNone;
            } else if (tok.str() == "loop") {
              _stage.metas().playbackMode = StageMetas::PlaybackMode::PlaybackModeLoop;
            } else {
              PUSH_ERROR_AND_RETURN("Unsupported playbackMode: " + tok.str());
            }
          }

          _stage.metas().customLayerData = metas.customLayerData;


          return true;  // ok
        });
  }

  void RegisterPrimIdxAssignCallback() {
    _parser.RegisterPrimIdxAssignFunction([&](const int64_t parentPrimIdx) {
      size_t idx = _prim_nodes.size();

      DCOUT("parentPrimIdx: " << parentPrimIdx << ", idx = " << idx);

      _prim_nodes.resize(idx + 1);

      // if (parentPrimIdx < 0) { // root
      //   // allocate empty prim to reserve _prim_nodes[idx]
      //   _prim_nodes.resize(idx + 1);
      //   DCOUT("resize to : " << (idx + 1));
      // }

      return idx;
    });
  }

  bool ReconstructPrimMeta(const ascii::AsciiParser::PrimMetaMap &in_meta,
                           PrimMeta *out) {

    auto ApiSchemaHandler = [](const std::string &tok)
        -> nonstd::expected<APISchemas::APIName, std::string> {
      using EnumTy = std::pair<APISchemas::APIName, const char *>;
      const std::vector<EnumTy> enums = {
          std::make_pair(APISchemas::APIName::SkelBindingAPI, "SkelBindingAPI"),
          std::make_pair(APISchemas::APIName::CollectionAPI, "CollectionAPI"),
          std::make_pair(APISchemas::APIName::MaterialBindingAPI,
                         "MaterialBindingAPI"),
          std::make_pair(APISchemas::APIName::ShapingAPI,
                         "ShapingAPI"),
          std::make_pair(APISchemas::APIName::ShadowAPI,
                         "ShadowAPI"),
          std::make_pair(APISchemas::APIName::VolumeLightAPI,
                         "VolumeLightAPI"),
          std::make_pair(APISchemas::APIName::Preliminary_PhysicsMaterialAPI,
                         "Preliminary_PhysicsMaterialAPI"),
          std::make_pair(APISchemas::APIName::Preliminary_PhysicsRigidBodyAPI,
                         "Preliminary_PhysicsRigidBodyAPI"),
          std::make_pair(APISchemas::APIName::Preliminary_PhysicsColliderAPI,
                         "Preliminary_PhysicsColliderAPI"),
          std::make_pair(APISchemas::APIName::Preliminary_AnchoringAPI,
                         "Preliminary_AnchoringAPI"),
          std::make_pair(APISchemas::APIName::LightAPI,
                         "LightAPI"),
          std::make_pair(APISchemas::APIName::MeshLightAPI,
                         "MeshLightAPI"),
          std::make_pair(APISchemas::APIName::LightListAPI,
                         "LightListAPI"),
          std::make_pair(APISchemas::APIName::ListAPI,
                         "ListAPI"),
          std::make_pair(APISchemas::APIName::MotionAPI,
                         "MotionAPI"),
          std::make_pair(APISchemas::APIName::PrimvarsAPI,
                         "PrimvarsAPI"),
          std::make_pair(APISchemas::APIName::GeomModelAPI,
                         "GeomModelAPI"),
          std::make_pair(APISchemas::APIName::VisibilityAPI,
                         "VisibilityAPI"),
          std::make_pair(APISchemas::APIName::XformCommonAPI,
                         "XformCommonAPI"),
          std::make_pair(APISchemas::APIName::NodeDefAPI,
                         "NodeDefAPI"),
          std::make_pair(APISchemas::APIName::CoordSysAPI,
                         "CoordSysAPI"),
          std::make_pair(APISchemas::APIName::ConnectableAPI,
                         "ConnectableAPI")
      };
      return EnumHandler<APISchemas::APIName>("apiSchemas", tok, enums);
    };

    auto BuildVariants = [](const Dictionary &dict) -> nonstd::expected<VariantSelectionMap, std::string> {

      // Allow empty dict.

      VariantSelectionMap m;

      for (const auto &item : dict) {
        // TODO: duplicated key check?
        if (auto pv = item.second.get_value<std::string>()) {
          m[item.first] = pv.value();
        } else if (auto pvs = item.second.get_value<value::StringData>()) {
          // TODO: store triple-quote info
          m[item.first] = pvs.value().value;
        } else {
          return nonstd::make_unexpected(fmt::format("TinyUSDZ only accepts `string` value for `variants` element, but got type `{}`(type_id {}).", item.second.type_name(), item.second.type_id()));
        }
      }

      return std::move(m);

    };

    DCOUT("ReconstructPrimMeta");
    for (const auto &meta : in_meta) {
      DCOUT("meta.name = " << meta.first);

      const auto &listEditQual = std::get<0>(meta.second);
      const MetaVariable &var = std::get<1>(meta.second);

      if (meta.first == "active") {
        DCOUT("active. type = " << var.type_name());
        if (var.type_name() == "bool") {
          if (auto pv = var.get_value<bool>()) {
            out->active = pv.value();
          } else {
            PUSH_ERROR_AND_RETURN(
                "(Internal error?) `active` metadataum is not type `bool`.");
          }
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `active` metadataum is not type `bool`. got `"
              << var.type_name() << "`.");
        }
      } else if (meta.first == "hidden") {
        DCOUT("hidden. type = " << var.type_name());
        if (var.type_name() == "bool") {
          if (auto pv = var.get_value<bool>()) {
            out->hidden = pv.value();
          } else {
            PUSH_ERROR_AND_RETURN(
                "(Internal error?) `hidden` metadataum is not type `bool`.");
          }
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `hidden` metadataum is not type `bool`. got `"
              << var.type_name() << "`.");
        }

      } else if (meta.first == "instanceable") {
        DCOUT("instanceable. type = " << var.type_name());
        if (var.type_name() == "bool") {
          if (auto pv = var.get_value<bool>()) {
            out->instanceable = pv.value();
          } else {
            PUSH_ERROR_AND_RETURN(
                "(Internal error?) `instanceable` metadataum is not type `bool`.");
          }
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `instanceable` metadataum is not type `bool`. got `"
              << var.type_name() << "`.");
        }

      } else if (meta.first == "sceneName") {
        DCOUT("sceneName. type = " << var.type_name());
        if (var.type_name() == value::kString) {
          if (auto pv = var.get_value<std::string>()) {
            out->sceneName = pv.value();
          } else {
            PUSH_ERROR_AND_RETURN(
                "(Internal error?) `sceneName` metadataum is not type `string`.");
          }
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `sceneName` metadataum is not type `string`. got `"
              << var.type_name() << "`.");
        }
      } else if (meta.first == "displayName") {
        DCOUT("displayName. type = " << var.type_name());
        if (var.type_name() == value::kString) {
          if (auto pv = var.get_value<std::string>()) {
            out->displayName = pv.value();
          } else {
            PUSH_ERROR_AND_RETURN(
                "(Internal error?) `displayName` metadataum is not type `string`.");
          }
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `displayName` metadataum is not type `string`. got `"
              << var.type_name() << "`.");
        }
      } else if (meta.first == "kind") {
        // std::tuple<ListEditQual, MetaVariable>
        // TODO: list-edit qual
        DCOUT("kind. type = " << var.type_name());
        if (var.type_name() == "token") {
          if (auto pv = var.get_value<value::token>()) {
            const value::token tok = pv.value();
            if (tok.str() == "subcomponent") {
              out->kind = Kind::Subcomponent;
            } else if (tok.str() == "component") {
              out->kind = Kind::Component;
            } else if (tok.str() == "model") {
              out->kind = Kind::Model;
            } else if (tok.str() == "group") {
              out->kind = Kind::Group;
            } else if (tok.str() == "assembly") {
              out->kind = Kind::Assembly;
            } else if (tok.str() == "sceneLibrary") {
              // USDZ specific: https://developer.apple.com/documentation/arkit/usdz_schemas_for_ar/scenelibrary
              out->kind = Kind::SceneLibrary;
            } else {
              // NOTE: empty token allowed.

              out->kind = Kind::UserDef;
              out->_kind_str = tok.str();
            }
            DCOUT("Added kind: " << to_string(out->kind.value()));
          } else {
            PUSH_ERROR_AND_RETURN(
                "(Internal error?) `kind` metadataum is not type `token`.");
          }
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `kind` metadataum is not type `token`. got `"
              << var.type_name() << "`.");
        }
      } else if (meta.first == "sdrMetadata") {
        DCOUT("sdrMetadata. type = " << var.type_name());
        if (var.type_id() == value::TypeTraits<Dictionary>::type_id()) {
          if (auto pv = var.get_value<Dictionary>()) {
            // TODO: Check if all items are string type.
            out->sdrMetadata = pv.value();
          } else {
            PUSH_ERROR_AND_RETURN_TAG(kTag,
                "(Internal error?) `sdrMetadata` metadataum is not type "
                "`dictionary`. got type `"
                << var.type_name() << "`");
          }

        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `sdrMetadata` metadataum is not type "
              "`dictionary`. got type `"
              << var.type_name() << "`");
        }
      } else if (meta.first == "customData") {
        DCOUT("customData. type = " << var.type_name());
        if (var.type_id() == value::TypeTraits<Dictionary>::type_id()) {
          if (auto pv = var.get_value<Dictionary>()) {
            out->customData = pv.value();
          } else {
            PUSH_ERROR_AND_RETURN_TAG(kTag,
                "(Internal error?) `customData` metadataum is not type "
                "`dictionary`. got type `"
                << var.type_name() << "`");
          }

        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `customData` metadataum is not type "
              "`dictionary`. got type `"
              << var.type_name() << "`");
        }
      } else if (meta.first == "clips") {
        DCOUT("clips. type = " << var.type_name());
        if (var.type_id() == value::TypeTraits<Dictionary>::type_id()) {
          if (auto pv = var.get_value<Dictionary>()) {
            out->clips = pv.value();
          } else {
            PUSH_ERROR_AND_RETURN_TAG(kTag,
                "(Internal error?) `clips` metadataum is not type "
                "`dictionary`. got type `"
                << var.type_name() << "`");
          }

        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `clips` metadataum is not type "
              "`dictionary`. got type `"
              << var.type_name() << "`");
        }
      } else if (meta.first == "assetInfo") {
        DCOUT("assetInfo. type = " << var.type_name());
        if (auto pv = var.get_value<Dictionary>()) {
          out->assetInfo = pv.value();
        } else {
          PUSH_ERROR_AND_RETURN_TAG(kTag,
              "(Internal error?) `assetInfo` metadataum is not type "
              "`dictionary`. got type `"
              << var.type_name() << "`");
        }
      } else if (meta.first == "variants") {
        if (auto pv = var.get_value<Dictionary>()) {
          auto pm = BuildVariants(pv.value());
          if (!pm) {
            PUSH_ERROR_AND_RETURN(pm.error());
          }
          out->variants = (*pm);
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `variants` metadataum is not type "
              "`dictionary`. got type `"
              << var.type_name() << "`");
        }
      } else if (meta.first == "inherits") {
        if (auto pvb = var.get_value<value::ValueBlock>()) {
          if (listEditQual != ListEditQual::ResetToExplicit) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          out->inherits = std::make_pair(listEditQual, std::vector<Path>());
        } else if (auto pv = var.get_value<std::vector<Path>>()) {
          if (pv.value().empty() && (listEditQual != ListEditQual::ResetToExplicit)) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          out->inherits = std::make_pair(listEditQual, pv.value());
        } else if (auto pvp = var.get_value<Path>()) {
          std::vector<Path> vs;
          vs.push_back(pvp.value());
          out->inherits = std::make_pair(listEditQual, vs);
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `inherits` metadataum should be either `path` or `path[]`. "
              "got type `"
              << var.type_name() << "`");
        }

      } else if (meta.first == "specializes") {
        if (auto pvb = var.get_value<value::ValueBlock>()) {
          if (listEditQual != ListEditQual::ResetToExplicit) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          out->specializes = std::make_pair(listEditQual, std::vector<Path>());
        } else if (auto pv = var.get_value<std::vector<Path>>()) {
          if (pv.value().empty() && (listEditQual != ListEditQual::ResetToExplicit)) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          out->specializes = std::make_pair(listEditQual, pv.value());
        } else if (auto pvp = var.get_value<Path>()) {
          std::vector<Path> vs;
          vs.push_back(pvp.value());
          out->specializes = std::make_pair(listEditQual, vs);
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `specializes` metadataum should be either `path` or `path[]`. "
              "got type `"
              << var.type_name() << "`");
        }

      } else if (meta.first == "variantSets") {
        // treat as `string`
        if (auto pvb = var.get_value<value::ValueBlock>()) {
          if (listEditQual != ListEditQual::ResetToExplicit) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          out->variantSets = std::make_pair(listEditQual, std::vector<std::string>());
        } else if (auto pv = var.get_value<value::StringData>()) {
          std::vector<std::string> vs;
          vs.push_back(pv.value().value);
          out->variantSets = std::make_pair(listEditQual, vs);
        } else if (auto pvs = var.get_value<std::string>()) {
          std::vector<std::string> vs;
          vs.push_back(pvs.value());
          out->variantSets = std::make_pair(listEditQual, vs);
        } else if (auto pva = var.get_value<std::vector<std::string>>()) {
          if (pva.value().empty() && (listEditQual != ListEditQual::ResetToExplicit)) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          out->variantSets = std::make_pair(listEditQual, pva.value());
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `variantSets` metadataum is not type "
              "`string` or `string[]`. got type `"
              << var.type_name() << "`");
        }
      } else if (meta.first == "apiSchemas") {
        DCOUT("apiSchemas. type = " << var.type_name());
        if (var.type_name() == "token[]") {
          APISchemas apiSchemas;
          if ((listEditQual != ListEditQual::Prepend) && (listEditQual != ListEditQual::ResetToExplicit)) {
            PUSH_ERROR_AND_RETURN("(PrimMeta) " << "ListEdit op for `apiSchemas` must be empty or `prepend` in TinyUSDZ, but got `" << to_string(listEditQual) << "`");
          }
          apiSchemas.listOpQual = listEditQual;

          if (auto pv = var.get_value<std::vector<value::token>>()) {

            for (const auto &item : pv.value()) {
              // TODO: Multi-apply schema(instance name)
              auto ret = ApiSchemaHandler(item.str());
              if (ret) {
                apiSchemas.names.push_back({ret.value(), /* instanceName */""});
              } else if (_config.allow_unknown_apiSchema) {
                PUSH_WARN("(PrimMeta) " << ret.error());
              } else {
                PUSH_ERROR_AND_RETURN("Unknown or invalid apiSchema: " + ret.error());
              }
            }
          } else {
            PUSH_ERROR_AND_RETURN_TAG(kTag, "(Internal error?) `apiSchemas` metadataum is not type "
            "`token[]`. got type `"
            << var.type_name() << "`");
          }

          out->apiSchemas = std::move(apiSchemas);
        } else {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "(Internal error?) `apiSchemas` metadataum is not type "
          "`token[]`. got type `"
          << var.type_name() << "`");
        }
      } else if (meta.first == "references") {

        if (var.is_blocked()) {
          // Treat as empty list
          // empty list must be qualified as 'explicit' 
          if (listEditQual != ListEditQual::ResetToExplicit) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          std::vector<Reference> refs;
          out->references = std::make_pair(listEditQual, refs);
        } else if (auto pv = var.get_value<Reference>()) {
          // To Reference
          std::vector<Reference> refs;
          refs.emplace_back(pv.value());
          out->references = std::make_pair(listEditQual, refs);
        } else if (auto pva = var.get_value<std::vector<Reference>>()) {
          if (pva.value().empty() && (listEditQual != ListEditQual::ResetToExplicit)) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          out->references = std::make_pair(listEditQual, pva.value());
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error?) `references` metadataum is not type "
              "`Reference`. got type `"
              << var.type_name() << "`");
        }
      } else if (meta.first == "payload") {

        if (var.is_blocked()) {
          if (listEditQual != ListEditQual::ResetToExplicit) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          // make empty
          std::vector<Payload> refs;
          out->payload = std::make_pair(listEditQual, refs);
        } else if (auto pv = var.get_value<Payload>()) {
          // To Payload
          std::vector<Payload> pls;
          pls.emplace_back(pv.value());
          out->payload = std::make_pair(listEditQual, pls);
        } else if (auto pva = var.get_value<std::vector<Payload>>()) {
          if (pva.value().empty() && (listEditQual != ListEditQual::ResetToExplicit)) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("None or Empty list must be `explicit`(no qualifier), but has qualifier `{}`", to_string(listEditQual)));
          }
          out->payload = std::make_pair(listEditQual, pva.value());
        } else {
          PUSH_ERROR_AND_RETURN(
              "(Internal error) `payload` metadataum is not type "
              "Payload. got type `"
              << var.type_name() << "`");
        }
      } else if (meta.first == "comment") {
        if (auto pv = var.get_value<value::StringData>()) {
          out->comment = pv.value().value;
        } else if (auto spv = var.get_value<std::string>()) {
          out->comment = spv.value();
        }
      } else {
        // Must be string value for unregisteredMeta for now.
        // TODO: infer int, string, token, int[], string[] and token[] type from the value for custom(unregisteredMeta) metadata.
        if (auto spv = var.get_value<std::string>()) {
          out->unregisteredMetas[meta.first] = spv.value();
        } else {
          PUSH_WARN("(Internal) unregistered Metadata must be type string for now, but got type " + var.type_name());
        }

      }
    }

    return true;
  }

  ///
  /// Reader entry point
  /// TODO: Use callback function(visitor) so that Reconstruct**** function is
  /// invoked in the Parser context.
  ///
  bool Read(const uint32_t state_flags, bool as_primspec);

  // std::vector<GPrim> GetGPrims() { return _gprims; }

  std::string GetDefaultPrimName() const { return _defaultPrim; }

  std::string GetError() { return _err; }

  std::string GetWarning() { return _warn; }

  ///
  /// Valid after `Read`.
  ///
  bool GetAsLayer(Layer *layer);

  ///
  /// Valid after `Read`.
  ///
  bool ReconstructStage();

  ///
  /// Valid after `ReconstructStage`.
  ///
  const Stage &GetStage() const { return _stage; }


 private:
  //bool stage_reconstructed_{false};

#if 0
  ///
  /// -- Iterators --
  ///
  class PrimIterator {
   public:
    PrimIterator(const std::vector<size_t> &indices,
                 const std::vector<value::Value> &values, size_t idx = 0)
        : _indices(indices), _values(values), _idx(idx) {}

    const value::Value &operator*() const { return _values[_indices[_idx]]; }

    PrimIterator &operator++() {
      _idx++;
      return *this;
    }
    bool operator!=(const PrimIterator &rhs) { return _idx != rhs._idx; }

   private:
    const std::vector<size_t> &_indices;
    const std::vector<value::Value> &_values;
    size_t _idx{0};
  };
  friend class PrimIterator;

  // currently const only
  using const_prim_iterator = const PrimIterator;

  // Iterate over toplevel prims
  const_prim_iterator PrimBegin() {
    return PrimIterator(_toplevel_prims, _prims);
  }
  const_prim_iterator PrimEnd() {
    return PrimIterator(_toplevel_prims, _prims, _toplevel_prims.size());
  }
  size_t PrimSize() { return _toplevel_prims.size(); }
#endif

  ///
  /// -- Members --
  ///

  // TODO: Remove
  // std::set<std::string> _node_types;

  std::stack<ParseState> parse_stack;

  std::string _base_dir;  // Used for importing another USD file
  //AssetResolutionResolver _arr;

#if 0 // TODO: Remove since not used.
  nonstd::optional<tinyusdz::Stage> _imported_scene;  // Imported scene.
#endif

  // "class" defs
  //std::map<std::string, Klass> _klasses;

  std::stack<std::string> _path_stack;

  std::string _err;
  std::string _warn;

  // Cache of loaded `references`
  // <filename, {defaultPrim index, Layer(PrimSpec data of usd file)}>
  std::map<std::string, std::pair<uint32_t, Layer>>
      _reference_cache;

  // toplevel prims
  std::vector<size_t> _toplevel_prims;  // index to _prim_nodes

  // 1D Linearized array of prim nodes.
  std::vector<PrimNode> _prim_nodes;

  // Path(prim part only) -> index to _prim_nodes[]
  std::map<std::string, size_t> _primpath_to_prim_idx_map;


  // toplevel primspecs
  std::vector<size_t> _toplevel_primspecs;  // index to _prim_nodes

  // Flattened array of primspec nodes.
  std::vector<PrimSpecNode> _primspec_nodes;
  // Path(prim part only) -> index to _primspec_nodes[]
  std::map<std::string, size_t> _primpath_to_primspec_idx_map;
  bool _primspec_invalidated{false};

  std::string _defaultPrim;

  // Used for Ascii parser option
  USDAReaderConfig _config;

  ascii::AsciiParser _parser;

};  // namespace usda

namespace {

// bottom up conversion.
bool ToPrimSpecRec(const size_t primSpecIdx,
                        std::vector<PrimSpecNode> &primspec_nodes, PrimSpec &parent, std::string *err) {

  if (primSpecIdx >= primspec_nodes.size()) {
    if (err) {
      (*err) += "Internal error; primSpecIdx exceeds primspec_nodes.size.";
    }
    return false;
  }

  const PrimSpecNode &node = primspec_nodes[primSpecIdx];

  PrimSpec primspec = node.primSpec;

  // Firstly process variants.
  std::set<int64_t> variantChildrenIndices; // record variantChildren indices
  {

    std::map<std::string, VariantSetSpec> variantSets;
    for (const auto &variantNodes : node.variantNodeMap) {
      DCOUT("variantSet " << variantNodes.first);
      VariantSetSpec variantSet;
      for (const auto &item : variantNodes.second) {
        DCOUT("variant " << item.first);
        PrimSpec variant; // variantNode can be represented as PrimSpec.
        for (const int64_t vidx : item.second.primChildren) {
          if (variantChildrenIndices.count(vidx)) {
            // Duplicated variant childrenIndices
            if (err) {
              (*err) = fmt::format("variant primIdx {} is referenced multiple times.\n", vidx);
            }
            return false;
          } else {
            // Add prim to variants
            if ((vidx >= 0) && (size_t(vidx) <= primspec_nodes.size())) {

              PrimSpec variantChildPrim; // dummy
              if (!ToPrimSpecRec(size_t(vidx), primspec_nodes, variantChildPrim, err)) {
                return false;
              }

              DCOUT(fmt::format("Added prim {} to variantSet {} : variant {}", variantChildPrim.name(), variantNodes.first, item.first));
              variant.children().emplace_back(variantChildPrim);
            } else {
              if (err) {
                (*err) = "primIndex exceeds prim_nodes.size()\n";
              }
              return false;
            }

            variantChildrenIndices.insert(vidx);
          }
        }

        variant.metas() = std::move(item.second.metas);
        variant.props() = std::move(item.second.props);

        variantSet.name = variantNodes.first;
        variantSet.variantSet.emplace(item.first, std::move(variant));
      }
      DCOUT(fmt::format("Add {} to variantSet", variantNodes.first));
      variantSets.emplace(variantNodes.first, std::move(variantSet));
    }
    primspec.variantSets() = std::move(variantSets);
  }

  for (const auto &cidx : node.children) {

    if (variantChildrenIndices.count(int64_t(cidx))) {
      // PrimSpec is already processed
      continue;
    }

    PrimSpec childPrimSpec;
    if (!ToPrimSpecRec(cidx, primspec_nodes, childPrimSpec, err)) {
      return false;
    }
    primspec.children().emplace_back(std::move(childPrimSpec));
  }

  parent = std::move(primspec);

  return true;
}

}  // namespace

bool USDAReader::Impl::GetAsLayer(Layer *layer) {

  if (!layer) {
    PUSH_ERROR_AND_RETURN("layer arg is nullptr.");
  }

  if (_primspec_invalidated) {
    PUSH_ERROR_AND_RETURN("PrimSpec data is invalid. USD data is not loaded or there was an error in earlier GetAsLayer call, or GetAsLayer was invoked multiple times.");
  }

  layer->clear_primspecs();
  DCOUT("# of subLayers = " << _stage.metas().subLayers.size());
  layer->metas() = _stage.metas();

  for (const auto &idx : _toplevel_primspecs) {
    DCOUT("Toplevel primspec idx: " << std::to_string(idx));

    if (idx >= _primspec_nodes.size()) {
      PUSH_ERROR_AND_RETURN("[Internal Error] out-of-bounds access.");
    }

    auto &node = _primspec_nodes[idx];
    PrimSpec &primSpec = node.primSpec;

    DCOUT("primspec[" << idx << "].typeName = " << primSpec.typeName());
    DCOUT("primspec[" << idx << "].name = " << primSpec.name());
    DCOUT("root prim[" << idx << "].num_children = " << primSpec.children().size());

    if (!ToPrimSpecRec(idx, _primspec_nodes, /* inout */primSpec, &_err)) {
      _primspec_invalidated = true;
      PUSH_ERROR_AND_RETURN("Construct PrimSpec tree failed.");
    }

    if (!layer->emplace_primspec(primSpec.name(), std::move(_primspec_nodes[idx].primSpec))) {
      PUSH_ERROR_AND_RETURN(fmt::format("Construct PrimSpec tree failed: PrimSpec.name = {}", primSpec.name()));
    }
  }

  // NOTE: _toplevel_primspecs are destroyed(std::move'ed)
  _primspec_invalidated = true;

  return true;
}

///
/// -- Impl reconstruct
//

namespace {

//
// Construct Prim from PrimNode with botom-up approach
//
bool ConstructPrimTreeRec(const size_t primIdx,
                        const std::vector<PrimNode> &prim_nodes,
                        Prim *destPrim,
                        std::string *err) {

  if (!destPrim) {
    if (err) {
      (*err) = "`destPrim` is nullptr.\n";
    }
    return false;
  }

  if (primIdx >= prim_nodes.size()) {
    if (err) {
      (*err) = "primIndex exceeds prim_nodes.size()\n";
    }
    return false;
  }

  const auto &node = prim_nodes[primIdx];

  Prim prim(node.prim);
  prim.prim_type_name() = node.typeName;

  DCOUT("prim[" << primIdx << "].type = " << node.prim.type_name());
  DCOUT("prim[" << primIdx << "].variantNodeMap.size = " << node.variantNodeMap.size());
  //prim.prim_id() = int64_t(idx);

  // Firstly process variants.
  std::set<int64_t> variantChildrenIndices; // record variantChildren indices

  std::map<std::string, VariantSet> variantSets;
  for (const auto &variantNodes : node.variantNodeMap) {
    DCOUT("variantSet " << variantNodes.first);
    VariantSet variantSet;
    for (const auto &item : variantNodes.second) {
      DCOUT("variant " << item.first);
      Variant variant;
      for (const int64_t vidx : item.second.primChildren) {
        if (variantChildrenIndices.count(vidx)) {
          // Duplicated variant childrenIndices
          if (err) {
            (*err) = fmt::format("variant primIdx {} is referenced multiple times.\n", vidx);
          }
          return false;
        } else {
          // Add prim to variants
          if ((vidx >= 0) && (size_t(vidx) <= prim_nodes.size())) {

            Prim variantChildPrim(value::Value(nullptr)); // dummy
            if (!ConstructPrimTreeRec(size_t(vidx), prim_nodes, &variantChildPrim, err)) {
              return false;
            }

            DCOUT(fmt::format("Added prim {} to variantSet {} : variant {}", variantChildPrim.element_name(), variantNodes.first, item.first));
            variant.primChildren().emplace_back(variantChildPrim);
          } else {
            if (err) {
              (*err) = "primIndex exceeds prim_nodes.size()\n";
            }
            return false;
          }

          variantChildrenIndices.insert(vidx);
        }
      }
      variant.metas() = std::move(item.second.metas);
      variant.properties() = std::move(item.second.props);

      variantSet.name = variantNodes.first;
      variantSet.variantSet.emplace(item.first, std::move(variant));
    }
    variantSets.emplace(variantNodes.first, std::move(variantSet));
  }
  prim.variantSets() = std::move(variantSets);

  for (const auto &cidx : node.children) {
    if (variantChildrenIndices.count(int64_t(cidx))) {
      // Prim is processed
      continue;
    }

    Prim childPrim(value::Value(nullptr)); // dummy
    if (!ConstructPrimTreeRec(cidx, prim_nodes, &childPrim, err)) {
      return false;
    }

    prim.children().emplace_back(std::move(childPrim));
  }

  (*destPrim) = std::move(prim);
  return true;
}

}  // namespace



bool USDAReader::Impl::ReconstructStage() {
  _stage.root_prims().clear();

  for (const auto &idx : _toplevel_prims) {
    DCOUT("Toplevel prim idx: " << std::to_string(idx));

    Prim prim(value::Value(nullptr)); // init with dummy Prim
    if (!ConstructPrimTreeRec(idx, _prim_nodes, &prim, &_err)) {
      return false;
    }

    _stage.root_prims().emplace_back(std::move(prim));

    DCOUT("num_children = " << _stage.root_prims()[size_t(_stage.root_prims().size() - 1)].children().size());
  }

  // Compute Abs Path from built Prim tree and Assign prim id.
  _stage.compute_absolute_prim_path_and_assign_prim_id();

  return true;
}

template <>
bool USDAReader::Impl::ReconstructPrim(
    const Specifier &spec,
    const prim::PropertyMap &properties,
    const prim::ReferenceList &references,
    Xform *xform) {

  std::string err;
  if (!prim::ReconstructPrim(spec, properties, references, xform, &_warn, &err)) {
    PUSH_ERROR_AND_RETURN("Failed to reconstruct Xform Prim: " << err);
  }
  return true;
}

#if 0
///
/// -- RegisterReconstructCallback specializations
///

template <>
bool USDAReader::Impl::ReconstructPrim(
    const Specifier &spec,
    const prim::PropertyMap &properties,
    const prim::ReferenceList &references,
    GPrim *gprim) {
  (void)spec;
  (void)gprim;

  DCOUT("TODO: Reconstruct GPrim.");

  PUSH_WARN("TODO: Reconstruct GPrim.");

  return true;
}


template <>
bool USDAReader::Impl::ReconstructPrim<NodeGraph>(
    const Specifier &spec,
    const prim::PropertyMap &properties,
    const prim::ReferenceList &references,
    NodeGraph *graph) {
  (void)properties;
  (void)references;
  (void)graph;

  PUSH_WARN("TODO: reconstruct NodeGrah.");

  return true;
}
#endif

// Generic Prim handler. T = Xform, GeomMesh, ...
template <typename T>
bool USDAReader::Impl::ReconstructPrim(
    const Specifier &spec,
    const prim::PropertyMap &properties,
    const prim::ReferenceList &references,
    T *prim) {

  prim::PrimReconstructOptions options;
  options.strict_allowedToken_check = _config.strict_allowedToken_check;
  DCOUT("strict_allowedToken_check " << options.strict_allowedToken_check);

  std::string err;
  if (!prim::ReconstructPrim(spec, properties, references, prim, &_warn, &err, options)) {
    PUSH_ERROR_AND_RETURN(fmt::format("Failed to reconstruct {} Prim: {}", value::TypeTraits<T>::type_name(), err));
  }
  return true;
}

///
/// -- Impl callback specializations
///

///
/// -- Impl Read
///

bool USDAReader::Impl::Read(const uint32_t state_flags, bool as_primspec) {

  ///
  /// Convert parser option.
  ///
  ascii::AsciiParserOption ascii_parser_option;
  ascii_parser_option.allow_unknown_prim = _config.allow_unknown_prims;
  ascii_parser_option.allow_unknown_apiSchema = _config.allow_unknown_apiSchema;
  ascii_parser_option.strict_allowedToken_check = _config.strict_allowedToken_check;

  ///
  /// Setup callbacks.
  ///
  StageMetaProcessor();

  RegisterPrimIdxAssignCallback();

  // For composition(as_primspec == true)
  RegisterPrimSpecHandler();

  // For direct Prim reconstruction(load state = Toplevel)
  RegisterReconstructCallback<Model>();  // Generic prim.

  RegisterReconstructCallback<GPrim>(); // Geometric prim

  RegisterReconstructCallback<Xform>();
  RegisterReconstructCallback<GeomCube>();
  RegisterReconstructCallback<GeomSphere>();
  RegisterReconstructCallback<GeomCone>();
  RegisterReconstructCallback<GeomPoints>();
  RegisterReconstructCallback<GeomCylinder>();
  RegisterReconstructCallback<GeomCapsule>();
  RegisterReconstructCallback<GeomMesh>();
  RegisterReconstructCallback<GeomSubset>();
  RegisterReconstructCallback<GeomBasisCurves>();
  RegisterReconstructCallback<GeomNurbsCurves>();
  RegisterReconstructCallback<GeomCamera>();

  RegisterReconstructCallback<Material>();
  RegisterReconstructCallback<Shader>();

  RegisterReconstructCallback<Scope>();

  RegisterReconstructCallback<SphereLight>();
  RegisterReconstructCallback<DomeLight>();
  RegisterReconstructCallback<DiskLight>();
  RegisterReconstructCallback<DistantLight>();
  RegisterReconstructCallback<CylinderLight>();

  RegisterReconstructCallback<SkelRoot>();
  RegisterReconstructCallback<Skeleton>();
  RegisterReconstructCallback<SkelAnimation>();
  RegisterReconstructCallback<BlendShape>();

  _parser.set_primspec_mode(as_primspec);

  bool ret = _parser.Parse(state_flags, ascii_parser_option);

  std::string warn = _parser.GetWarning();
  if (!warn.empty()) {
    PUSH_WARN("<USDAParser> " + warn);
  }

  if (!ret) {
    PUSH_ERROR_AND_RETURN("Parse failed:\n" + _parser.GetError());
  }


  return true;
}

//
// --
//

bool IsUSDA(const std::string &filename, size_t max_filesize) {
  // TODO: Read only first N bytes
  std::vector<uint8_t> data;
  std::string err;

  if (!io::ReadWholeFile(&data, &err, filename, max_filesize)) {
    return false;
  }

  tinyusdz::StreamReader sr(data.data(), data.size(), /* swap endian */ false);
  tinyusdz::ascii::AsciiParser parser(&sr);

  return parser.CheckHeader();
}

///
/// -- USDAReader
///
USDAReader::USDAReader(StreamReader *sr) { _impl = new Impl(sr); }

USDAReader::~USDAReader() { delete _impl; }

bool USDAReader::read(const uint32_t state_flags, bool as_primspec) {
  return _impl->Read(state_flags, as_primspec);
}

void USDAReader::set_base_dir(const std::string &dir) {
  return _impl->SetBaseDir(dir);
}

// std::vector<GPrim> USDAReader::GetGPrims() { return _impl->GetGPrims(); }

//std::string USDAReader::GetDefaultPrimName() const {
//  return _impl->GetDefaultPrimName();
//}

std::string USDAReader::get_error() { return _impl->GetError(); }
std::string USDAReader::get_warning() { return _impl->GetWarning(); }

bool USDAReader::get_as_layer(Layer *layer) { return _impl->GetAsLayer(layer); }

bool USDAReader::reconstruct_stage() { return _impl->ReconstructStage(); }

const Stage &USDAReader::get_stage() const { return _impl->GetStage(); }

void USDAReader::set_reader_config(const USDAReaderConfig &config) {
  return _impl->set_reader_config(config);
}

const USDAReaderConfig USDAReader::get_reader_config() const {
  return _impl->get_reader_config();
}

}  // namespace usda
}  // namespace tinyusdz

#else

namespace tinyusdz {
namespace usda {

USDAReader::USDAReader(StreamReader *sr) {
  _empty_stage = new Stage();
  (void)sr;
}

USDAReader::~USDAReader() {
  delete _empty_stage;
  _empty_stage = nullptr;
}

bool USDAReader::check_header() { return false; }

bool USDAReader::read(const LoadState state, bool as_primspec) {
  (void)state;
  (void)as_primspec;
  return false;
}

void USDAReader::set_base_dir(const std::string &dir) { (void)dir; }

//std::vector<GPrim> USDAReader::GetGPrims() { return {}; }

//std::string USDAReader::GetDefaultPrimName() const { return std::string{}; }

std::string USDAReader::get_error() {
  return "USDA parser feature is disabled in this build.\n";
}
std::string USDAReader::get_warning() { return std::string{}; }
bool USDAReader::reconstruct_stage() { return false; }

bool USDAReader::get_as_layer(Layer *layer) { return false; }

const Stage &USDAReader::get_stage() const {
  return *_empty_stage;
}

void USDAReader::set_reader_config(const USDAReaderConfig &config) {
  (void)config;
}

USDAReaderConfig USDAReader::get_reader_config() const {
  return USDAReaderConfig();
}

}  // namespace usda
}  // namespace tinyusdz

#endif

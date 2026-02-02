// SPDX-License-Identifier: Apache 2.0
// Copyright 2022-Present Light Transport Entertainment Inc.
//
// Layer and Prim composition features.
//
#pragma once

#include "asset-resolution.hh"
#include "prim-types.hh"

// TODO
// - [x] Compose `references`
// - [x] Compose `payloads`
// - [ ] Compose `specializes`
// - [x] Compose `inherits`
// - [ ] Compose `variantSets`
// - [x] Compose `over`
// - [ ] Consider `active` Prim metadatum

namespace tinyusdz {

// Forward decl.
class Stage;

// USD asset loading state.
enum class LoadState : uint32_t {
  Toplevel = 1,        // load initial .usd(default)
  Sublayer = 1 << 1,   // load USD from Stage meta sublayer
  Reference = 1 << 2,  // load USD from Prim meta reference
  Payload = 1 << 3     // load USD from Prim meta payload
};

struct SublayersCompositionOptions {
  // The maximum depth for nested `subLayers`
  uint32_t max_depth = 1024u;

  // Make an error when referenced asset does not contain prims.
  bool error_when_no_prims_in_sublayer{false};

  // Make an error when referenced asset is not found
  bool error_when_asset_not_found{false};

  // Make an error when referenced asset is unsupported(e.g. unknown file extension)
  bool error_when_unsupported_fileformat{false};

  // File formats
  std::map<std::string, FileFormatHandler> fileformats;
};

struct ReferencesCompositionOptions {
  // The maximum depth for nested `references`
  uint32_t max_depth = 1024u;

  // Make an error when referenced asset is not found
  bool error_when_asset_not_found{false};

  // Make an error when referenced asset is unsupported(e.g. unknown file extension)
  bool error_when_unsupported_fileformat{false};

  // File formats
  std::map<std::string, FileFormatHandler> fileformats;
};

struct PayloadCompositionOptions {
  // The maximum depth for nested `payload`
  uint32_t max_depth = 1024u;

  // Make an error when referenced asset is not found
  bool error_when_asset_not_found{false};

  // Make an error when referenced asset is unsupported(e.g. unknown file extension)
  bool error_when_unsupported_fileformat{false};

  // File formats
  std::map<std::string, FileFormatHandler> fileformats;
};


///
/// Extract subLayers asset paths
///
std::vector<std::string> ExtractSublayerAssetPaths(const Layer &layer);

///
/// Extract references asset paths
///
std::vector<std::string> ExtractReferencesAssetPaths(const Layer &layer);

///
/// Extract payload asset paths
///
std::vector<std::string> ExtractPayloadAssetPaths(const Layer &layer);

///
/// Return true when any PrimSpec in the Layer contains `references` Prim metadataum
///
/// Layer has cached flag for quicky detecting whether Layer has unresolved `references` or not.
///
/// @param[in] layer Layer
/// @param[in] force_check When true, traverse PrimSpec hierarchy and find `references` metadatum. Use cached flag in `layer` when false.
///
bool HasReferences(const Layer &layer, const bool force_check = false, const ReferencesCompositionOptions options = ReferencesCompositionOptions());

///
/// Return true when any PrimSpec in the Layer contains `payload` Prim metadataum
///
/// Layer has cached flag for quicky detecting whether Layer has unresolved `payload` or not.
///
/// @param[in] layer Layer
/// @param[in] force_check When true, traverse PrimSpec hierarchy and find `payload` metadatum. Use cached flag in `layer` when false.
///
bool HasPayload(const Layer &layer, const bool force_check = false, const PayloadCompositionOptions options = PayloadCompositionOptions());

///
/// Return true when any PrimSpec in the Layer contains `specializes` Prim metadataum.
/// We think specializers are not intensively used, so no caching.
///
/// @param[in] layer Layer
///
bool HasSpecializes(const Layer &layer);

///
/// Return true when any PrimSpec in the Layer contains `inherits` Prim metadataum.
///
/// @param[in] layer Layer
///
bool HasInherits(const Layer &layer);

///
/// Return true when any PrimSpec in the Layer contains `variants` Prim metadataum.
///
/// @param[in] layer Layer
///
bool HasVariants(const Layer &layer);

///
/// Return true when any PrimSpec in the Layer contains `over` Prim.
///
/// @param[in] layer Layer
///
bool HasOver(const Layer &layer);

#if 0 // deprecate it.
///
/// Load subLayer USD files in `layer`, and return composited(flattened) Layer
/// to `composited_layer` Supply search_path with `base_dir`
///
bool CompositeSublayers(
    const std::string &base_dir, const Layer &layer, Layer *composited_layer,
    std::string *warn, std::string *err,
    const SublayersCompositionOptions options = SublayersCompositionOptions());
#endif

///
/// Load subLayer USD files in `layer`, and return composited(flattened) Layer
/// to `composited_layer` Supply AssetResolutionResolver
///
bool CompositeSublayers(
    AssetResolutionResolver &resolver /* inout */, const Layer &layer,
    Layer *composited_layer, std::string *warn, std::string *err,
    const SublayersCompositionOptions options = SublayersCompositionOptions());

///
/// Resolve `references` for each PrimSpe, and return composited(flattened)
/// Layer to `composited_layer` in `layer`.
///
bool CompositeReferences(AssetResolutionResolver &resolver /* inout */,
                         const Layer &layer, Layer *composited_layer,
                         std::string *warn, std::string *err,
                         const ReferencesCompositionOptions options =
                             ReferencesCompositionOptions());

///
/// Resolve `payload` for each PrimSpec, and return composited(flattened) Layer
/// to `composited_layer` in `layer`.
///
bool CompositePayload(
    AssetResolutionResolver &resolver /* inout */, const Layer &layer,
    Layer *composited_layer, std::string *warn, std::string *err,
    const PayloadCompositionOptions options = PayloadCompositionOptions());

///
/// Resolve `variantSet` for each PrimSpec, and return composited(flattened) Layer
/// to `composited_layer` in `layer`.
/// Use variant selection info in each PrimSpec.
/// To externally specify variants to select, Use `ApplyVariantSelector`.
///
bool CompositeVariant(
    const Layer &layer,
    Layer *composited_layer, std::string *warn, std::string *err);

///
/// Resolve `specializes` for each PrimSpec, and return composited(flattened) Layer
/// to `composited_layer` in `layer`.
///
bool CompositeSpecializes(const Layer &layer,
    Layer *composited_layer, std::string *warn, std::string *err);

///
/// Resolve `inherits` for each PrimSpec, and return composited(flattened) Layer
/// to `composited_layer` in `layer`.
///
bool CompositeInherits(const Layer &layer,
    Layer *composited_layer, std::string *warn, std::string *err);

///
/// Override a PrimSpec with another PrimSpec.
///
/// @param[inout] dst PrimSpec to be override(must be `def` or `class` spec)
/// @param[in] src PrimSpec for override(must be `over` spec)
///
/// @return true upon success. false when error.
///
bool OverridePrimSpec(PrimSpec &dst, const PrimSpec &src, std::string *warn,
                      std::string *err);

///
/// Inherit PrimSpec. All PrimSpec tree in `src` PrimSpec will be inheritated to
/// `dst` PrimSpec.
///
/// @param[inout] dst PrimSpec to be inheritated
/// @param[in] src Source PrimSpec. Source PrimSpec can be any specifier(i.e,
/// `class`, `def` or `over`), but `class` recommended.
///
/// @return true upon success. false when error.
///
bool InheritPrimSpec(PrimSpec &dst, const PrimSpec &src, std::string *warn,
                     std::string *err);

///
/// Build USD Stage from Layer
///
bool LayerToStage(const Layer &layer, Stage *stage, std::string *warn,
                  std::string *err);

///
/// Build USD Stage from Layer
///
/// `layer` object will be destroyed after `stage` is being build.
///
bool LayerToStage(Layer &&layer, Stage *stage, std::string *warn,
                  std::string *err);

struct VariantSelector {
  std::string selection;  // current selection
  VariantSelectionMap vsmap;
};

using VariantSelectorMap = std::map<Path, VariantSelector>;

///
/// Recursively traverse PrimSpec tree and collect variantSelection information.
///
/// key : PrimSpec path(e.g. "/root/xform0")
/// value : VariantSelectionInfo
///
/// TODO: Move to Tydra API?
///
bool ListVariantSelectionMaps(const Layer &layer, VariantSelectorMap &m);

///
/// Select variant(PrimSpec subtree) `variant_name` from `src` PrimSpec and
/// write it to `dst` PrimSpec.
///
/// @param[inout] dst PrimSpec where selected variant are written.
/// @param[in] src Source PrimSpec. Source PrimSpec.
/// @param[in] variant_selection Variant Selection list. key = variantSet name, value = variant name. Can be empty(when empty, use PrimSpec's variants information)
///
/// @return true upon success. false when error. No error when any of variant info in `variant_selection` does not exist in `src` PrimSpec.
///
bool VariantSelectPrimSpec(PrimSpec &dst, const PrimSpec &src,
                           const std::map<std::string, std::string> &variant_selection, std::string *warn,
                           std::string *err);

///
/// Resolve variant in PrimSpec tree and write result to `dst`.
/// `dst` does not contain any variant info.
///
bool ApplyVariantSelector(const Layer &layer, const VariantSelectorMap &vsmap,
                          Layer *dst, std::string *warn, std::string *err);

///
/// Handy version of ApplyVariantSelector.
/// Use same variant name for all variantSets in Prim tree.
///
bool ApplyVariantSelector(const Layer &layer, const std::string &variant_name,
                          Layer *dst, std::string *warn, std::string *err);

///
/// Implementation of `references`
///
/// Import `layer` to this PrimSpec.
///
/// @param[inout] dst PrimSpec to be referenced.
/// @param[in] layer Layer(PrimSpec tree) to reference.
/// @param[in] primPath root Prim path in `layer`. Default = invalid Path =
/// defaultPrim in `layer`.
/// @param[in] layerOffset Layer(PrimSpec tree) to reference.
///
/// Use `defaultPrim` in `layer` as the root PrimSpec to import
///
///
bool ReferenceLayerToPrimSpec(PrimSpec &dst, const Layer &layer,
                              const Path primPath = Path(),
                              const LayerOffset layerOffset = LayerOffset());

///
/// Extract Variant information from Layer.
///
/// Example:
/// 
/// { "/cube0" : { "variantSets" : ["colorVariant"], "variants" : { "colorVariant" : "green" } } }
///
bool ExtractVariants(const Layer &layer, Dictionary *dict, std::string *err);

///
/// Extract Variant information from Stage.
///
bool ExtractVariants(const Stage &stage, Dictionary *dict, std::string *err);

#if 0  // TODO
///
/// Implementation of `references`
///
bool ReferenceLayersToPrimSpec(PrimSpec &dst, const std::vector<Layer> &layers
#endif

}  // namespace tinyusdz

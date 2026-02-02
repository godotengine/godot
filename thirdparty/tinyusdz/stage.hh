// SPDX-License-Identifier: Apache 2.
// Copyright 2022 - Present, Light Transport Entertainment, Inc.
//
// Stage: Similar to Scene or Scene graph
#pragma once

#include "composition.hh"
#include "prim-types.hh"

#if defined(TINYUSDZ_ENABLE_THREAD)
#include <mutex>
#endif

namespace tinyusdz {

// TODO: Use LayerMetas?
using StageMetas = LayerMetas;

class PrimRange;

// Similar to UsdStage, but much more something like a Scene(scene graph)
class Stage {
 public:
  // pxrUSD compat API ----------------------------------------
  static Stage CreateInMemory() { return Stage(); }

  ///
  /// Traverse by depth-first order.
  /// NOTE: Not yet implementd. Use tydra::VisitPrims() for a while.
  ///
  // PrimRange Traverse();

  ///
  /// Get Prim at a Path.
  /// Path must be absolute Path.
  ///
  /// @returns Const pointer to Prim(to avoid a copy). Never returns nullptr
  /// upon success.
  ///
  nonstd::expected<const Prim *, std::string> GetPrimAtPath(
      const Path &path) const;

  ///
  /// pxrUSD Compat API
  ///
  bool Flatten(bool addSourceFileComment = true) const {
    return compose(addSourceFileComment);
  }

  ///
  /// Dump Stage as ASCII(USDA) representation.
  /// @param[in] relative_path (optional) Print Path as relative Path.
  ///
  std::string ExportToString(bool relative_path = false) const;

  // pxrUSD compat API end -------------------------------------

  ///
  /// Get Prim from a children of given root Prim.
  /// Path must be relative Path.
  ///
  /// @returns pointer to Prim(to avoid a copy). Never return nullptr upon
  /// success.
  ///
  nonstd::expected<const Prim *, std::string> GetPrimFromRelativePath(
      const Prim &root, const Path &path) const;

  /// Find(Get) Prim at a Path.
  /// Path must be absolute Path.
  ///
  /// @param[in] path Absolute path(e.g. `/bora/dora`)
  /// @param[out] prim const reference to Prim(if found)
  /// @param[out] err Error message(filled when false is returned)
  ///
  /// @returns true if found a Prim.
  bool find_prim_at_path(const Path &path, const Prim *&prim,
                         std::string *err = nullptr) const;

  /// Find(Get) Prim at a Path and returns its Prim id.
  /// Path must be absolute Path.
  ///
  /// @param[in] path Absolute path(e.g. `/bora/dora`)
  /// @param[out] prim_id Prim's id(should be '1 or greater' upon success)
  /// @param[out] err Error message(filled when false is returned)
  ///
  /// @returns true if found a Prim.
  bool find_prim_at_path(const Path &path, int64_t *prim_id,
                         std::string *err = nullptr) const;

  /// Find(Get) Prim from a relative Path.
  /// Path must be relative Path.
  ///
  /// @param[in] root Find from this Prim
  /// @param[in] relative_path relative path(e.g. `dora/muda`)
  /// @param[out] prim const reference to the pointer to Prim(if found)
  /// @param[out] err Error message(filled when false is returned)
  ///
  /// @returns true if found a Prim.
  bool find_prim_from_relative_path(const Prim &root, const Path &relative_path,
                                    const Prim *&prim, std::string *err) const;

  ///
  /// Find(Get) Prim from Prim ID. Prim with no Prim ID assigned(-1 or 0) are
  /// ignored.
  ///
  /// @param[in] prim_id Prim ID(1 or greater)
  /// @param[out] prim const reference to the pointer to Prim(if found)
  /// @param[out] err Error message(filled when false is returned)
  ///
  /// @returns true if found a Prim.
  bool find_prim_by_prim_id(const uint64_t prim_id, const Prim *&prim,
                            std::string *err = nullptr) const;

  // non-const version
  bool find_prim_by_prim_id(const uint64_t prim_id, Prim *&prim,
                            std::string *err = nullptr);

  ///
  /// @brief Get Root Prims
  ///
  /// @return Const array of Root Prims.
  ///
  const std::vector<Prim> &root_prims() const { return _root_nodes; }

  ///
  /// @brief Reference to Root Prims array
  ///
  /// @return Array of Root Prims.
  /// TODO: Deprecate non-const `root_prims()` API and use `add_root_prim()` instead.
  ///
  std::vector<Prim> &root_prims() { return _root_nodes; }

  ///
  /// Add Prim to root.
  ///
  /// @param[in] prim Prim
  /// @param[in] rename_prim_name Rename Prim's elementName if required(to be unique among root Prims)
  ///
  /// @return true Upon success. false when failed to add Prim to root(e.g. same name exists in the root)
  /// (error message can be retrieved using `get_error()`)
  ///
  bool add_root_prim(Prim &&prim, bool rename_prim_name = true);

  ///
  /// Replace root Prim of elementName `prim_name` with `prim` 
  ///
  /// `prim`'s elementName will be modified to `prim_name`.
  ///
  /// If no root prim with `prim_name` exists, `prim` is added to root Prim and rename `prim`'s elementName to `prim_name`.
  ///
  /// @return true Upon succes. false when failed to replace Prim at root(e.g. `prim_name` is empty).
  ///
  bool replace_root_prim(const std::string &prim_name, Prim &&prim);

  ///
  /// @brief Get Stage metadatum
  ///
  /// @return Stage metadatum struct.
  ///
  const StageMetas &metas() const { return stage_metas; }

  StageMetas &metas() { return stage_metas; }

  ///
  /// @brief Assign unique Prim id inside this Stage.
  ///
  /// @param[out] prim_id Allocated Primitive ID.
  ///
  /// @return true upon success.
  ///
  bool allocate_prim_id(uint64_t *prim_id) const;

  ///
  /// @brief Release Prim id inside this Stage.
  ///
  /// @param[prim_id] prim_id Primitive ID to release(allocated by
  /// `allocate_prim_id`)
  ///
  /// @return true upon success. false when given `prim_id` is an invalid id.
  ///
  bool release_prim_id(const uint64_t prim_id) const;

  ///
  /// @brief Check if given prim_id exists in this Stage.
  ///
  /// @param[prim_id] prim_id Primitive ID to check.
  ///
  /// @return true if `prim_id` exists in this Stage.
  ///
  bool has_prim_id(const uint64_t prim_id) const;

  ///
  /// @brief Commit Stage state.
  ///
  /// Call this function after you finished adding Prims manually(through
  /// `root_prims()`) to Stage.
  ///
  /// (No need to call this if you just use ether USDA/USDC/USDZ reader).
  ///
  /// - Compute absolute path and set it to Prim::abs_path for each Prim
  /// currently added to this Stage.
  /// - Assign unique ID to Prim
  ///
  /// @param[in] force_assign_prim_id true Overwrite `prim_id` of each Prim.
  /// false only assign Prim id when `prim_id` is -1(preserve user-assgiend
  /// prim_id). Setting `false` is not recommended since prim_id may not be
  /// unique over Prims in Stage.
  /// @return false when the Stage contains any invalid Prim
  ///
  /// TODO: Deprecate this API an use `commit()`
  bool compute_absolute_prim_path_and_assign_prim_id(
      bool force_assign_prim_id = true);

  ///
  /// @brief Commit Stage state.
  ///
  bool commit() {
    // Currently we always allocate Prim ID.
    return compute_absolute_prim_path_and_assign_prim_id(true);
  }

  ///
  /// Compute absolute Prim path for Prims in this Stage.
  ///
  bool compute_absolute_prim_path();

  ///
  /// Dump Prim tree info(mainly for debugging).
  ///
  std::string dump_prim_tree() const;

  ///
  /// Compose scene(Not implemented yet).
  ///
  bool compose(bool addSourceFileComment = true) const;

  const std::string &get_warning() const {
    return _warn;
  }

  const std::string &get_error() const {
    return _err;
  }

 private:

#if defined(TINYUSDZ_ENABLE_THREAD)
  mutable std::mutex _mutex;
#endif

#if 0 // Deprecated. remove.
  ///
  /// Loads USD from and return it as Layer
  ///
  /// @param[in] filename USD filename
  /// @param[in] resolver AssetResolutionResolver
  /// @param[out] layer Layer representation of USD data.
  /// @param[in] load_states Bitmask of LoadState(optional)
  ///
  bool LoadLayerFromFile(const std::string &filename, const AssetResolutionResolver &resolver, Layer *layer, const uint32_t load_states = static_cast<uint32_t>(LoadState::Toplevel));

  ///
  /// Loads USD asset from memory and return it as Layer
  ///
  /// @param[in] addr Memory address
  /// @param[in] nbytes Num bytes
  /// @param[in] asset_name Asset name(usually filename)
  /// @param[out] layer Layer representation of USD data.
  /// @param[in] load_states Bitmask of LoadState(optional)
  ///
  bool LoadLayerFromMemory(const uint8_t *addr, const size_t nbytes, const std::string &asset_name, Layer *layer, const uint32_t load_states = static_cast<uint32_t>(LoadState::Toplevel));
#endif

#if 0 // Deprecated. moved to composition.hh
  ///
  /// Loads `reference` USD asset and return it as Layer
  ///
  bool LoadReference(const Reference &reference, Layer *dest);

  ///
  /// Loads USD assets described in `subLayers` Stage/Layer meta and return it as Layers
  ///
  bool LoadSubLayers(std::vector<Layer> *dest_sublayers);
#endif

  // Root nodes
  std::vector<Prim> _root_nodes;
  std::multiset<std::string> _root_node_nameSet;

  std::string name;       // Scene name
  int64_t default_root_node{-1};  // index to default root node

  StageMetas stage_metas;

  mutable std::string _err;
  mutable std::string _warn;

  // Cached prim path.
  // key : prim_part string (e.g. "/path/bora")
  mutable std::map<std::string, const Prim *> _prim_path_cache;

  // Cached prim_id -> Prim lookup
  // key : prim_id
  mutable std::map<uint64_t, const Prim *> _prim_id_cache;

  mutable bool _dirty{true}; // True when Stage content changes(addition, deletion, composition/flatten, etc.)

  mutable bool _prim_id_dirty{true}; // True when Prim Id assignent changed(TODO: Unify with `_dirty` flag)

  mutable HandleAllocator<uint64_t> _prim_id_allocator;
};

inline std::string to_string(const Stage &stage, bool relative_path = false) {
  return stage.ExportToString(relative_path);
}

}  // namespace tinyusdz

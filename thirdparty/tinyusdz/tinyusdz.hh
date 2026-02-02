// SPDX-License-Identifier: Apache 2.0
// Copyright (c) 2019 - 2023, Syoyo Fujita.
// Copyright (c) 2023 - Present, Light Transport Entertainment Inc.

#ifndef TINYUSDZ_HH_
#define TINYUSDZ_HH_

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef TINYUSDZ_LOCAL_DEBUG_PRINT
#include <iostream>  // dbg
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// TODO: Use std:: version for C++17
#include "nonstd/expected.hpp"
#include "nonstd/optional.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "image-types.hh"
#include "prim-types.hh"
#include "texture-types.hh"
#include "usdGeom.hh"
#include "usdLux.hh"
#include "usdShade.hh"
#include "usdSkel.hh"
//#include "usdVox.hh"
#include "stage.hh"
#include "asset-resolution.hh"


namespace tinyusdz {

constexpr int version_major = 0;
constexpr int version_minor = 9;
constexpr int version_micro = 1;
constexpr auto version_rev = "";  // extra revision suffix(e.g. "rc.1")

struct USDLoadOptions {
  ///
  /// Set the number of threads to use when parsing USD scene.
  /// -1 = use # of system threads(CPU cores/threads).
  ///
  int num_threads{-1};

  // Set the maximum memory limit advisorily(including image data).
  // This feature would be helpful if you want to load USDZ model in mobile
  // device.
  int32_t max_memory_limit_in_mb{16384};  // in [mb] Default 16GB

  ///
  /// TODO: Deprecate
  /// Loads asset data(e.g. texture image, audio). Default is true.
  /// If you want to load asset data in your own way or don't need asset data to
  /// be loaded, Set this false.
  ///
  bool load_assets{true};

  ///
  /// (Deprecated. to be removed in the future release)
  /// Do composition on load(Load sublayers, references, etc)
  /// For USDZ model, this should be false.
  ///
  bool do_composition{false};

  ///
  /// (Deprecated. to be removed in the future release)
  /// Following load flags are valid when `do_composition` is set `true`.
  ///
  bool load_sublayers{false}; // true: Load `subLayers`
  bool load_references{false}; // true: Load `references`
  bool load_payloads{false}; // true: Load `paylod` at top USD loading(no lazy loading).

  ///
  /// Max MBs allowed for each asset file(e.g. jpeg)
  ///
  uint32_t max_allowed_asset_size_in_mb{1024};  // [mb] default 1GB.

  ///
  /// For texture size
  ///
  uint32_t max_image_width = 2048;
  uint32_t max_image_height = 2048;
  uint32_t max_image_channels = 4;

  ///
  /// For usdSkel
  ///
  bool strict_usdSkel_check{false}; // Strict usdSkel parsing check when true.

  ///
  /// allowedToken
  ///
  bool strict_allowedToken_check{false}; // Make parse error when token value is not in allowedToken list(when the schema defines allowedToken list)

  ///
  /// apiSchema
  ///
  bool strict_apiSchema_check{false}; // Make parse error when unknown apiSchema
  
  ///
  /// User-defined fileformat hander.
  /// key = file(asset) extension(`.` excluded. example: 'mtlx', 'obj').
  ///
  std::map<std::string, FileFormatHandler> fileformats;

  Axis upAxis{Axis::Y};
};


// TODO: Provide profiles for loader option.
// e.g.
// - Embedded(e.g. web, tigh resource size limit for security)
// - Realtime(moderate resource size limit)
// - DCC(for data conversion. Unlimited resource size)

#if 0  // TODO
//struct USDWriteOptions
//{
//
//
//};
#endif

//

///
/// Load USD(USDA/USDC/USDZ) from a file.
/// Automatically detect file format.
///
/// @param[in] filename USD filename(UTF-8)
/// @param[out] stage USD stage(scene graph).
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadUSDFromFile(const std::string &filename, Stage *stage,
                     std::string *warn, std::string *err,
                     const USDLoadOptions &options = USDLoadOptions());

///
/// Load USD(USDA/USDC/USDZ) from memory.
/// Automatically detect file format.
///
/// @param[in] addr Memory address of USD data
/// @param[in] length Byte length of USD data
/// @param[in] filename Filename(can be empty).
/// @param[out] stage USD stage(scene graph).
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadUSDFromMemory(const uint8_t *addr, const size_t length,
                       const std::string &filename, Stage *stage,
                       std::string *warn, std::string *err,
                       const USDLoadOptions &options = USDLoadOptions());

///
/// Load USDZ(zip) from a file.
/// It will load first USD file in USDZ container.
///
/// @param[in] filename USDZ filename(UTF-8)
/// @param[out] stage USD stage(scene graph).
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadUSDZFromFile(const std::string &filename, Stage *stage,
                      std::string *warn, std::string *err,
                      const USDLoadOptions &options = USDLoadOptions());

#ifdef _WIN32
// WideChar(Unicode filename) version
bool LoadUSDZFromFile(const std::wstring &filename, Stage *stage,
                      std::string *warn, std::string *err,
                      const USDLoadOptions &options = USDLoadOptions());
#endif


///
/// Load USDZ(zip) from memory.
/// It will load first USD file in USDZ container.
///
/// @param[in] addr Memory address of USDZ data
/// @param[in] length Byte length of USDZ data
/// @param[in] filename Filename(can be empty).
/// @param[out] stage USD stage(scene graph).
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadUSDZFromMemory(const uint8_t *addr, const size_t length,
                        const std::string &filename, Stage *stage,
                        std::string *warn, std::string *err,
                        const USDLoadOptions &options = USDLoadOptions());

struct USDZAsset
{
  // key: asset name(USD, Image, Audio, ...), value = byte begin/end in USDZ data.
  std::map<std::string, std::pair<size_t, size_t>> asset_map;

  // When mmapped, `data` is empty, and `addr`(Usually pointer to mmaped address) and `size`  are set.
  // When non-mmapped, `data` holds the copy of whole USDZ data.
  std::vector<uint8_t> data; // USDZ itself
  const uint8_t *addr{nullptr};
  size_t size{0}; // in bytes.
  
  bool is_mmaped() const {
    return !data.empty();
  }
};

///
/// Read USDZ(zip) asset info from a file.
///
/// Whole file content(USDZ) is copied into USDZAsset::data.
/// If you want to save memory to load USDZ with assets, first read USDZ conent into memory(or Use io-util.hh::MMapFile() to mmap file), then use `ReadUSDZAssetInfoFromMemory with `assert_on_memory` true.
///
/// @param[in] filename USDZ filename(UTF-8)
/// @param[out] asset USDZ asset info.
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] max_file_size_in_mb Maximum file size
///
/// @return true upon success
///
bool ReadUSDZAssetInfoFromFile(const std::string &filename, USDZAsset *asset,
  std::string *warn, std::string *err, size_t max_file_size_in_mb = 16384ull);

///
/// Read USDZ(zip) asset info from memory.
///
/// @param[in] addr Memory address
/// @param[in] asset_on_memory When true, do not copy USDZ data(`length` bytes from `addr` address) to USDZAsset. Instead just retain `addr` and `length` in USDZAsset. Memory address `addr` must be retained during any asset data in USDZAsset is accessed. When false, USDZ data is copied into USDZAsset.
/// 
/// @param[out] asset USDZ asset info.
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
///
/// @return true upon success
///
bool ReadUSDZAssetInfoFromMemory(const uint8_t *addr, const size_t length, const bool asset_on_memory, USDZAsset *asset,
  std::string *warn, std::string *err);

///
/// Handy utility API to setup AssetResolutionResolver to load asset data from USDZ data.
///
/// @param[inout] resolver Add asset resolution to the resolver. The resolver retains the pointer to USDZAsset.
/// @param[in] pusdzAsset Pointer to data struct(USDZAsset struct). Must be retained until there is (potential) access to any asset, since AssetResolutionResolver and FileSystemHandler loads an asset from this struct.
///
/// @return upon success and setup `resolver` and `fsHandler`.
///
bool SetupUSDZAssetResolution(
  AssetResolutionResolver &resolver,
  const USDZAsset *pusdzAsset);

///
/// Default AssetResolution handler for USDZ(read asset from USDZ container)
///
int USDZResolveAsset(const char *asset_name, const std::vector<std::string> &search_paths, std::string *resolved_asset_name, std::string *err, void *userdata);
int USDZSizeAsset(const char *resolved_asset_name, uint64_t *nbytes, std::string *err, void *userdata);
int USDZReadAsset(const char *resolved_asset_name, uint64_t req_bytes, uint8_t *out_buf, uint64_t *nbytes, std::string *err, void *userdata);

///
/// Load USDC(binary) from a file.
///
/// @param[in] filename USDC filename(UTF-8)
/// @param[out] stage USD stage(scene graph).
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadUSDCFromFile(const std::string &filename, Stage *stage,
                      std::string *warn, std::string *err,
                      const USDLoadOptions &options = USDLoadOptions());

///
/// Load USDC(binary) from a memory.
///
/// @param[in] addr Memory address of USDC data
/// @param[in] length Byte length of USDC data
/// @param[in] filename Filename(can be empty).
/// @param[out] stage USD stage.
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadUSDCFromMemory(const uint8_t *addr, const size_t length,
                        const std::string &filename, Stage *stage,
                        std::string *warn, std::string *err,
                        const USDLoadOptions &options = USDLoadOptions());

///
/// Load USDA(ascii) from a file.
///
/// @param[in] filename USDA filename(UTF-8)
/// @param[out] stage USD stage.
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadUSDAFromFile(const std::string &filename, Stage *stage,
                      std::string *warn, std::string *err,
                      const USDLoadOptions &options = USDLoadOptions());

///
/// Load USDA(ascii) from a memory.
///
/// @param[in] addr Memory address of USDA data
/// @param[in] length Byte length of USDA data
/// @param[in[ base_dir Base directory(can be empty)
/// @param[out] stage USD stage.
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadUSDAFromMemory(const uint8_t *addr, const size_t length,
                        const std::string &base_dir, Stage *stage,
                        std::string *warn, std::string *err,
                        const USDLoadOptions &options = USDLoadOptions());

///
/// For composition
///

///
/// Load USD(USDA/USDC/USDZ) from a file and return it as Layer.
/// Automatically detect file format.
///
/// @param[in] filename USD filename(UTF-8)
/// @param[out] layer USD layer(scene graph).
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadLayerFromFile(const std::string &filename, Layer *stage,
                     std::string *warn, std::string *err,
                     const USDLoadOptions &options = USDLoadOptions());

///
/// Load USD(USDA/USDC/USDZ) from memory and return it as Layer.
/// Automatically detect file format.
///
/// @param[in] addr Memory address of USD data
/// @param[in] length Byte length of USD data
/// @param[in] filename Corresponding Filename(can be empty).
/// @param[out] layer USD layer(scene graph).
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadLayerFromMemory(const uint8_t *addr, const size_t length,
                       const std::string &filename, Layer *layer,
                       std::string *warn, std::string *err,
                       const USDLoadOptions &options = USDLoadOptions());


bool LoadUSDALayerFromMemory(const uint8_t *addr, const size_t length,
                       const std::string &filename, Layer *layer,
                       std::string *warn, std::string *err,
                       const USDLoadOptions &options = USDLoadOptions());

bool LoadUSDCLayerFromMemory(const uint8_t *addr, const size_t length,
                       const std::string &filename, Layer *layer,
                       std::string *warn, std::string *err,
                       const USDLoadOptions &options = USDLoadOptions());

bool LoadUSDZLayerFromMemory(const uint8_t *addr, const size_t length,
                       const std::string &filename, Layer *layer,
                       std::string *warn, std::string *err,
                       const USDLoadOptions &options = USDLoadOptions());

///
/// Load USD(USDA/USDC/USDZ) layer using AssetResolution resolver.
/// This API would be useful if you want to load USD from custom storage(e.g, on Android), URI(web), DB, etc.
/// Automatically detect file format.
///
/// resolved_asset_name must be resolved asset name using AssetResolutionResolver::resolve()
///
/// @param[in] resolver AssetResolution resolver.
/// @param[in] resolved_asset_name Resolved asset name.
/// @param[out] layer USD layer(scene graph).
/// @param[out] warn Warning message.
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Load options(optional)
///
/// @return true upon success
///
bool LoadLayerFromAsset(AssetResolutionResolver &resolver,
                       const std::string &resolved_asset_name, Layer *layer,
                       std::string *warn, std::string *err,
                       const USDLoadOptions &options = USDLoadOptions());


#if 0  // TODO
///
/// Write stage as USDC to a file.
///
/// @param[in] filename USDC filename
/// @param[out] err Error message(filled when the function returns false)
/// @param[in] options Write options(optional)
///
/// @return true upon success
///
bool WriteAsUSDCToFile(const std::string &filename, std::string *err, const USDCWriteOptions &options = USDCWriteOptions());

#endif

// Test if input is any of USDA/USDC/USDZ format.
// Optionally returns detected format("usda", "usdc", or "usdz") to
// `detected_format` when a given file/binary is a USD format.
bool IsUSD(const std::string &filename, std::string *detected_format = nullptr);
bool IsUSD(const uint8_t *addr, const size_t length,
            std::string *detected_format = nullptr);

// Test if input is USDA format.
bool IsUSDA(const std::string &filename);
bool IsUSDA(const uint8_t *addr, const size_t length);

// Test if input is USDC(Crate binary) format.
bool IsUSDC(const std::string &filename);
bool IsUSDC(const uint8_t *addr, const size_t length);

// Test if input is USDZ(Uncompressed ZIP) format.
bool IsUSDZ(const std::string &filename);
bool IsUSDZ(const uint8_t *addr, const size_t length);

}  // namespace tinyusdz

#endif  // TINYUSDZ_HH_

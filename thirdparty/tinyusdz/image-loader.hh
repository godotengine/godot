// SPDX-License-Identifier: Apache 2.0

// Simple image loader
// supported file format: PNG(use fpng), BMP/JPEG(use stb_image), OpenEXR(use tinyexr), TIFF(use tinydng)  
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "image-types.hh"

#include "nonstd/expected.hpp"

namespace tinyusdz {
namespace image {

struct ImageResult {
  Image image;
  std::string warning;
};

struct ImageInfoResult {
  uint32_t width;
  uint32_t height;
  uint32_t channels;
  std::string warning;
};

///
/// User-defined Image asset loader
///
/// TOOD: Use FileFormat API?
///

///
/// Callback function to load an image from memory.
///
/// @param[in] addr Image data byte address.
/// @param[in] datasize Image data size in bytes.
/// @param[in] asset_name Corresponding asset/file name.
/// @param[inout] user_data User data pointer. Can be nullptr. 
/// @param[out] warn Warning message. Can be nullptr.
/// @param[out] err Error message. Can be nullptr.
///
/// @return true upon success.

typedef bool (*LoadImageDataFunction)(ImageResult *image, const uint8_t *addr, const size_t datasize, const std::string &asset_name, void *user_data, std::string *warn, std::string *err);

///
/// Callback function to get info of an image from memory.
///
/// @param[in] addr Image data byte address.
/// @param[in] datasize Image data size in bytes.
/// @param[in] asset_name Corresponding asset/file name.
/// @param[inout] user_data User data pointer. Can be nullptr. 
/// @param[out] warn Warning message. Can be nullptr.
/// @param[out] err Error message. Can be nullptr.
///
/// @return true upon success.

typedef bool (*GetImageInfoFunction)(ImageInfoResult *image, const uint8_t *addr, const size_t datasize, const std::string &asset_name, void *user_data);


///
/// Load image from a file.
/// 
/// @param[in] filename Input filename(or URI)
/// @param[in] max_memory_limit_in_mb Optional. Maximum image file size in [MB]. Default = 1 TB.
/// @return ImageResult or error message(std::string)
///
nonstd::expected<ImageResult, std::string> LoadImageFromFile(const std::string &filename, const size_t max_memory_limit_in_mb = 1024*1024);

///
/// Get Image info from file.
/// 
/// @param[in] filename Input filename(or URI)
/// @return ImageInfoResult or error message(std::string)
///
nonstd::expected<ImageInfoResult, std::string> GetImageInfoFromFile(const std::string &filename);

///
/// Load image from memory
///
/// @param[in] addr Memory address
/// @param[in] datasize Data size(in bytes)
/// @param[in] uri Input URI(or filename) as a hint. This is used only in error message.
/// @return ImageResult or error message(std::string)
///
nonstd::expected<ImageResult, std::string> LoadImageFromMemory(const uint8_t *addr, const size_t datasize, const std::string &uri);

///
/// Get Image info from a file.
///
/// @param[in] addr Memory address
/// @param[in] datasize Data size(in bytes)
/// @param[in] uri Input URI(or filename) as a hint. This is used only in error message.
/// @return ImageResult or error message(std::string)
///
nonstd::expected<ImageInfoResult, std::string> GetImageInfoFromMemory(const uint8_t *addr, const size_t datasize, const std::string &uri);

} // namespace image
} // namespace tinyusdz

// Simple image writer
// supported file format: PNG(use fpng), JPEG(use stb_image), OpenEXR(use
// tinyexr), TIFF/DNG(use tinydng)
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "nonstd/expected.hpp"
#include "tinyusdz.hh"

namespace tinyusdz {
namespace image {

//
// Autodetect = determine file format using filename or Image's pixelformat +
// bpp.
//
enum class WriteImageFormat { Autodetect, BMP, PNG, JPEG, EXR, TIFF, DNG };

struct WriteOption {
  WriteImageFormat format{WriteImageFormat::Autodetect};
  bool half{false};  // Use half float for EXR

  // When non-zero value is set, prefer this bitdepth than Image's bpp.
  // Can specify 10, 12 and 14 for DNG when writing 16bit input image as 10, 12 and 14bit respectively.
  int bitdepth{
      0};  
};

///
/// @param[in] filename Output filename
/// @param[in] image Image data
/// @param[in] option Image write option(optional)
///
/// @return true upon success. or error message(std::string) when failed.
///
nonstd::expected<bool, std::string> WriteImageToFile(
    const std::string &filename, const Image &image,
    WriteOption option = WriteOption());

///
/// @param[in] image Image data
/// @param[in] option Image write option(optional)
/// @return Serialized image data
///
nonstd::expected<std::vector<uint8_t>, std::string> WriteImageToMemory(
    const Image &image, const WriteOption option = WriteOption());

}  // namespace image
}  // namespace tinyusdz

// SPDX-License-Identifier: Apache 2.0
#if defined(TINYUSDZ_WITH_EXR)
#include "external/tinyexr.h"
#endif

#if defined(TINYUSDZ_WITH_TIFF)
#include "external/tiny_dng_writer.h"
#endif

#ifndef TINYUSDZ_NO_STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "external/stb_image_write.h"
#include "external/fpng.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "image-writer.hh"
#include "io-util.hh"
#include "str-util.hh"

namespace tinyusdz {
namespace image {

namespace {

bool DetectFileFormatFromExtension(const std::string &_ext, tinyusdz::image::WriteImageFormat &format) {

  std::string ext = to_lower(_ext);

  if (ext == "bmp") {
    format = tinyusdz::image::WriteImageFormat::BMP;
  } else if (ext == "png") {
    format = tinyusdz::image::WriteImageFormat::PNG;
  } else if ((ext == "jpg") || (ext == "jpeg")) {
    format = tinyusdz::image::WriteImageFormat::JPEG;
  } else if ((ext == "tiff") || (ext == "tif")) {
    format = tinyusdz::image::WriteImageFormat::TIFF;
  } else if (ext == "dng") {
    format = tinyusdz::image::WriteImageFormat::DNG;
  } else if (ext == "exr") {
    format = tinyusdz::image::WriteImageFormat::EXR;
  }

  return false;
}

} // namespace

nonstd::expected<bool, std::string> WriteImageToFile(
    const std::string &filename, const Image &image,
    WriteOption option) {


  tinyusdz::image::WriteImageFormat format = option.format;

  if (format == tinyusdz::image::WriteImageFormat::Autodetect) {
    if (!DetectFileFormatFromExtension(io::GetFileExtension(filename), format)) {
      return nonstd::make_unexpected("Failed to determine image file format from extension: " + filename);
    }
  }

  if ((format == tinyusdz::image::WriteImageFormat::BMP) ||
      (format == tinyusdz::image::WriteImageFormat::JPEG)) {
    // Currently LDR only
    if (image.bpp != 8) {
      return nonstd::make_unexpected("8bit only for BMP/JPEG output.");
    }

  } else if (format == tinyusdz::image::WriteImageFormat::EXR) {

    if (image.bpp == 8) {
      return nonstd::make_unexpected("Invalid bit per pixel(8) for EXR output.");
    }

  } else if (format == tinyusdz::image::WriteImageFormat::TIFF) {

    if (image.bpp == 8) {

    } else if (image.bpp == 16) {

    } else if (image.bpp == 32) {

    } else {
      return nonstd::make_unexpected("Invalid bit per pixel for EXR output.");
    }

  } else if (format == tinyusdz::image::WriteImageFormat::DNG) {
    // 16bit only for DNG
    if (image.bpp != 16) {
      return nonstd::make_unexpected("Bit per pixel must be 16 for DNG output.");
    }
  } else {
    // ???
    return nonstd::make_unexpected("Internal error in WriteImageToFile.");
  }

  (void)image;

  return nonstd::make_unexpected("TODO: Implement WriteImageToFile");
}

///
/// @param[in] image Image data
/// @param[in] option Image write option(optional)
/// @return Serialized image data
///
nonstd::expected<std::vector<uint8_t>, std::string> WriteImageToMemory(
    const Image &image, const WriteOption option)
{
  (void)image;
  (void)option;

  // TODO: Autodetect format
  if (option.format == tinyusdz::image::WriteImageFormat::Autodetect) {
    return nonstd::make_unexpected("TODO: Autodetect image format.");
  }

  return nonstd::make_unexpected("TODO: Implement WriteImageToFile");
}

} // namespace image
} // namespace tinyusdz

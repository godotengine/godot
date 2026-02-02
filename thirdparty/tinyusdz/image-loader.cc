// Support files
//
// - OpenEXR(through TinyEXR). 16bit and 32bit
// - TIFF/DNG(through TinyDNG). 8bit, 16bit and 32bit
// - PNG(8bit, 16bit), Jpeg, bmp, tga, ...(through stb_image or wuffs).
//
// TODO:
//
// - [ ] Use fpng for 8bit PNG when `stb_image` is used
// - [ ] 10bit, 12bit and 14bit DNG image
// - [ ] Support LoD tile, multi-channel for TIFF image
//

#if defined(TINYUSDZ_WITH_EXR)
#include "external/tinyexr.h"
#endif

#if defined(TINYUSDZ_USE_WUFFS_IMAGE_LOADER)

#ifndef TINYUSDZ_NO_WUFFS_IMPLEMENTATION
#define WUFFS_IMPLEMENTATION

#define WUFFS_CONFIG__MODULES
#define WUFFS_CONFIG__MODULE__BASE
#define WUFFS_CONFIG__MODULE__BMP
//#define WUFFS_CONFIG__MODULE__GIF
#define WUFFS_CONFIG__MODULE__PNG
#define WUFFS_CONFIG__MODULE__JPEG
//#define WUFFS_CONFIG__MODULE__WBMP
#endif

#else

#if !defined( TINYUSDZ_NO_BUILTIN_IMAGE_LOADER)

// stb_image
#ifndef TINYUSDZ_NO_STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif

#endif

#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif


#if defined(TINYUSDZ_USE_WUFFS_IMAGE_LOADER)

#include "external/wuffs-unsupported-snapshot.c"

#else

#if !defined( TINYUSDZ_NO_BUILTIN_IMAGE_LOADER)

// fpng, stb_image
#include "external/fpng.h"

// avoid duplicated symbols when tinyusdz is linked to an app/library whose also use stb_image.
#define STB_IMAGE_STATIC
#include "external/stb_image.h"

#endif

#endif

#if defined(TINYUSDZ_WITH_TIFF)
#ifndef TINYUSDZ_NO_TINY_DNG_LOADER_IMPLEMENTATION
#define TINY_DNG_LOADER_IMPLEMENTATION
#endif

#ifndef TINY_DNG_NO_EXCEPTION
#define TINY_DNG_NO_EXCEPTION
#endif

#ifndef TINY_DNG_LOADER_NO_STDIO
#define TINY_DNG_LOADER_NO_STDIO
#endif

// Prevent including `stb_image.h` inside of tiny_dng_loader.h
#ifndef TINY_DNG_LOADER_NO_STB_IMAGE_INCLUDE
#define TINY_DNG_LOADER_NO_STB_IMAGE_INCLUDE
#endif

#include "external/tiny_dng_loader.h"
#endif


#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "image-loader.hh"
#include "io-util.hh"

namespace tinyusdz {
namespace image {

namespace {

#if defined(TINYUSDZ_USE_WUFFS_IMAGE_LOADER)

bool DecodeImageWUFF(const uint8_t *bytes, const size_t size,
                    const std::string &uri, Image *image, std::string *warn,
                    std::string *err) {

  if (err) {
    (*err) = "TODO: WUFF image loader.\n";
  }

  return false;

}

bool GetImageInfoWUFF(const uint8_t *bytes, const size_t size,
                    const std::string &uri, uint32_t *width, uint32_t *height, uint32_t *channels, std::string *warn,
                    std::string *err) {
  if (err) {
    (*err) = "TODO: WUFF image loader.\n";
  }

  return false;
}


#else

#if !defined( TINYUSDZ_NO_BUILTIN_IMAGE_LOADER)

// Decode image(png, jpg, ...) using STB
// 16bit PNG is supported.
bool DecodeImageSTB(const uint8_t *bytes, const size_t size,
                    const std::string &uri, Image *image, std::string *warn,
                    std::string *err) {
  (void)warn;

  int w = 0, h = 0, comp = 0, req_comp = 0;

  unsigned char *data = nullptr;

  // force 32-bit textures for common Vulkan compatibility. It appears that
  // some GPU drivers do not support 24-bit images for Vulkan
  req_comp = 4;
  int bits = 8;

  // It is possible that the image we want to load is a 16bit per channel image
  // We are going to attempt to load it as 16bit per channel, and if it worked,
  // set the image data accodingly. We are casting the returned pointer into
  // unsigned char, because we are representing "bytes". But we are updating
  // the Image metadata to signal that this image uses 2 bytes (16bits) per
  // channel:
  if (stbi_is_16_bit_from_memory(bytes, int(size))) {
    data = reinterpret_cast<unsigned char *>(
        stbi_load_16_from_memory(bytes, int(size), &w, &h, &comp, req_comp));
    if (data) {
      bits = 16;
    }
  }

  // at this point, if data is still NULL, it means that the image wasn't
  // 16bit per channel, we are going to load it as a normal 8bit per channel
  // mage as we used to do:
  // if image cannot be decoded, ignore parsing and keep it by its path
  // don't break in this case
  // FIXME we should only enter this function if the image is embedded. If
  // `uri` references an image file, it should be left as it is. Image loading
  // should not be mandatory (to support other formats)
  if (!data) {
    data = stbi_load_from_memory(bytes, int(size), &w, &h, &comp, req_comp);
  }

  if (!data) {
    // NOTE: you can use `warn` instead of `err`
    if (err) {
      (*err) +=
          "Unknown image format. STB cannot decode image data for image: " +
          uri + "\".\n";
    }
    return false;
  }

  if ((w < 1) || (h < 1)) {
    stbi_image_free(data);
    if (err) {
      (*err) += "Invalid image data for image: " + uri + "\"\n";
    }
    return false;
  }

  image->width = w;
  image->height = h;
  image->channels = req_comp;
  image->bpp = bits;
  image->format = Image::PixelFormat::UInt;
  image->data.resize(size_t(w) * size_t(h) * size_t(req_comp) * size_t(bits / 8));
  std::copy(data, data + w * h * req_comp * (bits / 8), image->data.begin());
  stbi_image_free(data);

  return true;
}

bool GetImageInfoSTB(const uint8_t *bytes, const size_t size,
                    const std::string &uri, uint32_t *width, uint32_t *height, uint32_t *channels, std::string *warn,
                    std::string *err) {
  (void)warn;
  (void)uri;
  (void)err; // TODO

  int w = 0, h = 0, comp = 0;

  int ret = stbi_info_from_memory(bytes, int(size), &w, &h, &comp);

  if (w < 0) w = 0;
  if (h < 0) h = 0;
  if (comp < 0) comp = 0;

  if (ret == 1) {
    if (width) { (*width) = uint32_t(w); }
    if (height) { (*height) = uint32_t(h); }
    if (channels) { (*channels) = uint32_t(comp); }
    return true;
  }

  return false;
}
#endif
#endif

#if defined(TINYUSDZ_WITH_EXR)

bool DecodeImageEXR(const uint8_t *bytes, const size_t size,
                    const std::string &uri, Image *image,
                    std::string *err) {
  // TODO(syoyo):
  // - [ ] Read fp16 image as fp16
  // - [ ] Read int16 image as int16
  // - [ ] Read int32 image as int32
  // - [ ] Multi-channel EXR

  float *rgba = nullptr;
  int width;
  int height;
  const char *exrerr = nullptr;
  // LoadEXRFromMemory always load EXR image as fp32 x RGBA
  int ret = LoadEXRFromMemory(&rgba, &width, &height, bytes, size, &exrerr);

  if (exrerr) {
    (*err) += std::string(exrerr);

    FreeEXRErrorMessage(exrerr);
  }

  if (!ret) {
    (*err) += "Failed to load EXR image: " + uri + "\n";
    return false;
  }

  image->width = width;
  image->height = height;
  image->channels = 4;  // RGBA
  image->bpp = 32;      // fp32
  image->format = Image::PixelFormat::Float;
  image->data.resize(size_t(width) * size_t(height) * 4 * sizeof(float));
  memcpy(image->data.data(), rgba, sizeof(float) * size_t(width) * size_t(height) * 4);

  free(rgba);

  return true;
}

#endif

#if defined(TINYUSDZ_WITH_TIFF)

bool DecodeImageTIFF(const uint8_t *bytes, const size_t size,
                    const std::string &uri, Image *image,
                    std::string *err) {


  std::vector<tinydng::FieldInfo> custom_fields; // no custom fields
  std::vector<tinydng::DNGImage> images;

  std::string warn;
  std::string dngerr;

  bool ret = tinydng::LoadDNGFromMemory(reinterpret_cast<const char *>(bytes), uint32_t(size), custom_fields, &images, &warn, &dngerr);

  if (!dngerr.empty()) {
    (*err) += dngerr;
  }

  if (!ret) {
    (*err) += "Failed to load TIFF/DNG image: " + uri + "\n";
    return false;
  }

  // TODO(syoyo): Multi-layer TIFF
  // Use the largest image(based on width pixels).
  size_t largest = 0;
  int largest_width = images[0].width;
  for (size_t i = 1; i < images.size(); i++) {
    if (largest_width < images[i].width) {
      largest = i;
      largest_width = images[i].width;
     }
  }

  size_t spp = size_t(images[largest].samples_per_pixel);
  size_t bps = size_t(images[largest].bits_per_sample);

  if (spp > 4) {
    (*err) += "Samples per pixel must be 0 ~ 4, but got " + std::to_string(spp) + " for image: " + uri + "\n";
    return false;
  }

  // TODO: Support 10, 12 and 14bit Image(e.g. Apple ProRAW 12bit)
  if ((bps == 8) || (bps == 16) || (bps == 32)) {
    // ok
  } else {
    (*err) += "Invalid or unsupported bits per sample " + std::to_string(bps) + " for image: " + uri + "\n";
    return false;
  }

  auto sample_format = images[largest].sample_format;
  if (sample_format == tinydng::SAMPLEFORMAT_UINT) {
    image->format = Image::PixelFormat::UInt;
  } else if (sample_format == tinydng::SAMPLEFORMAT_INT) {
    image->format = Image::PixelFormat::Int;
  } else if (sample_format == tinydng::SAMPLEFORMAT_IEEEFP) {
    image->format = Image::PixelFormat::Float;
  } else {
    (*err) += "Invalid Sample format for image: " + uri + "\n";
    return false;
  }

  image->width = images[largest].width;
  image->height = images[largest].height;
  image->channels = int(spp);
  image->bpp = int(bps);

  image->data.swap(images[largest].data);

  return true;
}

#endif

}  // namespace

nonstd::expected<image::ImageResult, std::string> LoadImageFromMemory(
    const uint8_t *addr, size_t sz, const std::string &uri) {
  image::ImageResult ret;
  std::string err;

#if defined(TINYUSDZ_WITH_EXR)
  if (TINYEXR_SUCCESS == IsEXRFromMemory(addr, sz)) {

    bool ok = DecodeImageEXR(addr, sz, uri, &ret.image, &err);

    if (!ok) {
      return nonstd::make_unexpected(err);
    }

    return std::move(ret);
  }
#endif

#if defined(TINYUSDZ_WITH_TIFF)
  {
    std::string msg;
    if (tinydng::IsDNGFromMemory(reinterpret_cast<const char *>(addr), uint32_t(sz), &msg)) {

      bool ok = DecodeImageTIFF(addr, sz, uri, &ret.image, &err);

      if (!ok) {
        return nonstd::make_unexpected(err);
      }

      return std::move(ret);
    }
  }
#endif

#if defined(TINYUSDZ_USE_WUFFS_IMAGE_LOADER)
  bool ok = DecodeImageWUFF(addr, sz, uri, &ret.image, &ret.warning, &err);
#elif !defined(TINYUSDZ_NO_BUILTIN_IMAGE_LOADER)
  bool ok = DecodeImageSTB(addr, sz, uri, &ret.image, &ret.warning, &err);
#else
  // TODO: Use user-supplied image loader
  (void)addr;
  (void)sz;
  (void)uri;
  bool ok = false;
  err = "Image loading feature is disabled in this build. TODO: use user-supplied image loader\n";
#endif
  if (!ok) {
    return nonstd::make_unexpected(err);
  }

  return std::move(ret);
}

nonstd::expected<image::ImageInfoResult, std::string> GetImageInfoFromMemory(
    const uint8_t *addr, size_t sz, const std::string &uri) {
  image::ImageInfoResult ret;
  std::string err;

#if defined(TINYUSDZ_WITH_EXR)
  if (TINYEXR_SUCCESS == IsEXRFromMemory(addr, sz)) {

    return nonstd::make_unexpected("TODO: EXR format");
  }
#endif

#if defined(TINYUSDZ_WITH_TIFF)
  if (tinydng::IsDNGFromMemory(reinterpret_cast<const char *>(addr), uint32_t(sz), &err)) {

      return nonstd::make_unexpected("TODO: TIFF/DNG format");

  }
#endif

#if defined(TINYUSDZ_USE_WUFFS_IMAGE_LOADER)
  bool ok = GetImageInfoWUFF(addr, sz, uri, &ret.width, &ret.height, &ret.channels, &ret.warning, &err);
#elif !defined(TINYUSDZ_NO_BUILTIN_IMAGE_LOADER)
  bool ok = GetImageInfoSTB(addr, sz, uri, &ret.width, &ret.height, &ret.channels, &ret.warning, &err);
#else
  (void)addr;
  (void)sz;
  (void)uri;
  bool ok = false;
  err = "Image loading feature is disabled in this build. TODO: use user-supplied image info function\n";
#endif
  if (!ok) {
    return nonstd::make_unexpected(err);
  }

  return std::move(ret);
}

nonstd::expected<image::ImageResult, std::string> LoadImageFromFile(
    const std::string &filename, const size_t max_memory_limit_in_mb) {

  // Assume filename is already resolved.
  std::string filepath = filename;

  std::vector<uint8_t> data;
  size_t max_bytes = size_t(1024 * 1024 * max_memory_limit_in_mb);
  std::string err;
  if (!io::ReadWholeFile(&data, &err, filepath, max_bytes,
                         /* userdata */ nullptr)) {
    return nonstd::make_unexpected("File not found or failed to read : \"" + filepath + "\"\n");
  }

  if (data.size() < 4) {
    return nonstd::make_unexpected("File size too short. Looks like this file is not an image file : \"" +
                filepath + "\"\n");
  }

  return LoadImageFromMemory(data.data(), data.size(), filename);
}

}  // namespace image
}  // namespace tinyusdz

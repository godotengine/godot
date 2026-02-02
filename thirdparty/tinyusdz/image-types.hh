// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2024 - Present, Light Transport Entertainment Inc.
#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace tinyusdz {

// Simple image class.
// No colorspace conversion will be applied when decoding image data
// (e.g. from .jpg, .png).
struct Image {
  // NOTE: Actual pixel value format is determined with combination of PixelFormat x bpp
  // e.g. Float + 16 bpp = fp16
  enum class PixelFormat {
    UInt, // LDR and HDR image
    Int, // For ao/normal/displacement map, DNG photo
    Float, // HDR image
  };
   
  std::string uri;  // filename or uri;

  int width{-1};     // -1 = invalid
  int height{-1};    // -1 = invalid
  int channels{-1};  // Image channels. 3=RGB, 4=RGBA. -1 = invalid
  int bpp{-1};       // bits per pixel. 8=LDR, 16,32=HDR
  PixelFormat format{PixelFormat::UInt};
  
  std::vector<uint8_t> data; // Raw data.

  std::string colorspace; // Colorspace metadata in the image. Optional.
};

inline std::string to_string(Image::PixelFormat fmt) {
  std::string s{"[[InvalidPixelFormat]]"};
  // work around for false-positive behavior of `error: 'switch' missing 'default' label [-Werror,-Wswitch-default]`
  // happens in NDK 27 clang
  if (Image::PixelFormat::UInt == fmt) { s =  "uint"; }
  else if (Image::PixelFormat::Int == fmt) { s =  "int"; }
  else if (Image::PixelFormat::Float == fmt) { s =  "float"; }

  return s;
}

} // namespace tinyusdz

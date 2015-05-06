// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Coding tools configuration
//
// Author: Skal (pascal.massimino@gmail.com)

#include "../webp/encode.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// WebPConfig
//------------------------------------------------------------------------------

int WebPConfigInitInternal(WebPConfig* config,
                           WebPPreset preset, float quality, int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_ENCODER_ABI_VERSION)) {
    return 0;   // caller/system version mismatch!
  }
  if (config == NULL) return 0;

  config->quality = quality;
  config->target_size = 0;
  config->target_PSNR = 0.;
  config->method = 4;
  config->sns_strength = 50;
  config->filter_strength = 20;   // default: light filtering
  config->filter_sharpness = 0;
  config->filter_type = 0;        // default: simple
  config->partitions = 0;
  config->segments = 4;
  config->pass = 1;
  config->show_compressed = 0;
  config->preprocessing = 0;
  config->autofilter = 0;
  config->partition_limit = 0;
  config->alpha_compression = 1;
  config->alpha_filtering = 1;
  config->alpha_quality = 100;
  config->lossless = 0;
  config->image_hint = WEBP_HINT_DEFAULT;

  // TODO(skal): tune.
  switch (preset) {
    case WEBP_PRESET_PICTURE:
      config->sns_strength = 80;
      config->filter_sharpness = 4;
      config->filter_strength = 35;
      break;
    case WEBP_PRESET_PHOTO:
      config->sns_strength = 80;
      config->filter_sharpness = 3;
      config->filter_strength = 30;
      break;
    case WEBP_PRESET_DRAWING:
      config->sns_strength = 25;
      config->filter_sharpness = 6;
      config->filter_strength = 10;
      break;
    case WEBP_PRESET_ICON:
      config->sns_strength = 0;
      config->filter_strength = 0;   // disable filtering to retain sharpness
      break;
    case WEBP_PRESET_TEXT:
      config->sns_strength = 0;
      config->filter_strength = 0;   // disable filtering to retain sharpness
      config->segments = 2;
      break;
    case WEBP_PRESET_DEFAULT:
    default:
      break;
  }
  return WebPValidateConfig(config);
}

int WebPValidateConfig(const WebPConfig* config) {
  if (config == NULL) return 0;
  if (config->quality < 0 || config->quality > 100)
    return 0;
  if (config->target_size < 0)
    return 0;
  if (config->target_PSNR < 0)
    return 0;
  if (config->method < 0 || config->method > 6)
    return 0;
  if (config->segments < 1 || config->segments > 4)
    return 0;
  if (config->sns_strength < 0 || config->sns_strength > 100)
    return 0;
  if (config->filter_strength < 0 || config->filter_strength > 100)
    return 0;
  if (config->filter_sharpness < 0 || config->filter_sharpness > 7)
    return 0;
  if (config->filter_type < 0 || config->filter_type > 1)
    return 0;
  if (config->autofilter < 0 || config->autofilter > 1)
    return 0;
  if (config->pass < 1 || config->pass > 10)
    return 0;
  if (config->show_compressed < 0 || config->show_compressed > 1)
    return 0;
  if (config->preprocessing < 0 || config->preprocessing > 1)
    return 0;
  if (config->partitions < 0 || config->partitions > 3)
    return 0;
  if (config->partition_limit < 0 || config->partition_limit > 100)
    return 0;
  if (config->alpha_compression < 0)
    return 0;
  if (config->alpha_filtering < 0)
    return 0;
  if (config->alpha_quality < 0 || config->alpha_quality > 100)
    return 0;
  if (config->lossless < 0 || config->lossless > 1)
    return 0;
  if (config->image_hint >= WEBP_HINT_LAST)
    return 0;
  return 1;
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

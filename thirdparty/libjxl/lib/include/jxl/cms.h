// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef JXL_CMS_H_
#define JXL_CMS_H_

// ICC profiles and color space conversions.

#include <jxl/cms_interface.h>
#include <jxl/jxl_cms_export.h>

#ifdef __cplusplus
extern "C" {
#endif

JXL_CMS_EXPORT const JxlCmsInterface* JxlGetDefaultCms();

#ifdef __cplusplus
}
#endif

#endif  // JXL_CMS_H_

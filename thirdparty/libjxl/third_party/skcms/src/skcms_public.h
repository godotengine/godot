/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#pragma once

// skcms_public.h contains the entire public API for skcms.

#ifndef SKCMS_API
    #define SKCMS_API
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// A row-major 3x3 matrix (ie vals[row][col])
typedef struct skcms_Matrix3x3 {
    float vals[3][3];
} skcms_Matrix3x3;

// It is _not_ safe to alias the pointers to invert in-place.
SKCMS_API bool            skcms_Matrix3x3_invert(const skcms_Matrix3x3*, skcms_Matrix3x3*);
SKCMS_API skcms_Matrix3x3 skcms_Matrix3x3_concat(const skcms_Matrix3x3*, const skcms_Matrix3x3*);

// A row-major 3x4 matrix (ie vals[row][col])
typedef struct skcms_Matrix3x4 {
    float vals[3][4];
} skcms_Matrix3x4;

// A transfer function mapping encoded values to linear values,
// represented by this 7-parameter piecewise function:
//
//   linear = sign(encoded) *  (c*|encoded| + f)       , 0 <= |encoded| < d
//          = sign(encoded) * ((a*|encoded| + b)^g + e), d <= |encoded|
//
// (A simple gamma transfer function sets g to gamma and a to 1.)
typedef struct skcms_TransferFunction {
    float g, a,b,c,d,e,f;
} skcms_TransferFunction;

SKCMS_API float skcms_TransferFunction_eval  (const skcms_TransferFunction*, float);
SKCMS_API bool  skcms_TransferFunction_invert(const skcms_TransferFunction*,
                                              skcms_TransferFunction*);

typedef enum skcms_TFType {
    skcms_TFType_Invalid,
    skcms_TFType_sRGBish,
    skcms_TFType_PQish,
    skcms_TFType_HLGish,
    skcms_TFType_HLGinvish,
    skcms_TFType_PQ,
    skcms_TFType_HLG,
} skcms_TFType;

// Identify which kind of transfer function is encoded in an skcms_TransferFunction
SKCMS_API skcms_TFType skcms_TransferFunction_getType(const skcms_TransferFunction*);

// We can jam a couple alternate transfer function forms into skcms_TransferFunction,
// including those matching the general forms of the SMPTE ST 2084 PQ function or HLG.
//
// PQish:
//                              max(A + B|encoded|^C, 0)
//    linear = sign(encoded) * (------------------------) ^ F
//                                  D + E|encoded|^C
SKCMS_API bool skcms_TransferFunction_makePQish(skcms_TransferFunction*,
                                                float A, float B, float C,
                                                float D, float E, float F);
// HLGish:
//            { K * sign(encoded) * ( (R|encoded|)^G )          when 0   <= |encoded| <= 1/R
//   linear = { K * sign(encoded) * ( e^(a(|encoded|-c)) + b )  when 1/R <  |encoded|
SKCMS_API bool skcms_TransferFunction_makeScaledHLGish(skcms_TransferFunction*,
                                                       float K, float R, float G,
                                                       float a, float b, float c);

// Compatibility shim with K=1 for old callers.
static inline bool skcms_TransferFunction_makeHLGish(skcms_TransferFunction* fn,
                                                     float R, float G,
                                                     float a, float b, float c) {
    return skcms_TransferFunction_makeScaledHLGish(fn, 1.0f, R,G, a,b,c);
}

// The PQ transfer function. The function skcms_TransferFunction_eval will always evaluate to the
// unit PQ EOTF, which maps [0, 1] to [0, 1], regardless of the other parameters.
// This is stored differently from PQish transfer functions. In particular:
//   - the constant -5 is stored in g
//   - the hdr_reference_white_luminance parameter is stored in a
//   - all other parameters are set to 0
// When this is used as an SkColorSpace, the transformation to XYZD50 will be as follows:
//   1. Apply the unit PQ EOTF to each channel
//   2. Multiply by 10,000 nits
//   3. Divide by hdr_reference_white_luminance nits (default is 203)
//   4. Transform primaries to XYZD50
SKCMS_API void skcms_TransferFunction_makePQ(
    skcms_TransferFunction*,
    float hdr_reference_white_luminance);

// The HLG transfer function. The function skcms_TransferFunction_eval will always evaluate to the
// HLG inverse OETF, which maps [0, 1] to [0, 1], regardless of the other parameters.
// This is stored differently from PQish transfer functions. In particular:
//   - the constant -6 is stored in g
//   - the hdr_reference_white_luminance parameter is stored in a
//   - the peak_white_luminance parameter is stored in b
//   - the system_gamma parameter is stored in c
//   - all other parameters are set to 0
// When this is used as an SkColorSpace, the transformation to XYZD50 will be as follows:
//   1. Apply the HLG inverse OETF to each channel
//   2. Transform primaries to Rec2020
//   3. Apply the channel-mixing HLG OOTF using system_gamma (default is 1.2)
//   4. Multiply by peak_luminance nits (default is 1,000)
//   5. Divide by hdr_reference_white nits (default is 203)
//   6. Transform primaries to XYZD50
SKCMS_API void skcms_TransferFunction_makeHLG(
    skcms_TransferFunction*,
    float hdr_reference_white_luminance,
    float peak_luminance,
    float system_gamma);

// Is this an ordinary sRGB-ish transfer function, or one of the HDR forms we support?
SKCMS_API bool skcms_TransferFunction_isSRGBish(const skcms_TransferFunction*);
SKCMS_API bool skcms_TransferFunction_isPQish  (const skcms_TransferFunction*);
SKCMS_API bool skcms_TransferFunction_isHLGish (const skcms_TransferFunction*);
SKCMS_API bool skcms_TransferFunction_isPQ     (const skcms_TransferFunction*);
SKCMS_API bool skcms_TransferFunction_isHLG    (const skcms_TransferFunction*);

// Unified representation of 'curv' or 'para' tag data, or a 1D table from 'mft1' or 'mft2'
typedef union skcms_Curve {
    struct {
        // this needs to line up with alias_of_table_entries so we can tell if there are or
        // are not table entries. If this is 0, this struct is a parametric function,
        // otherwise it's a table entry.
        uint32_t alias_of_table_entries;
        skcms_TransferFunction parametric;
    };
    struct {
        uint32_t table_entries;
        const uint8_t* table_8;
        const uint8_t* table_16;
    };
} skcms_Curve;

// Complex transforms between device space (A) and profile connection space (B):
//   A2B:  device -> [ "A" curves -> CLUT ] -> [ "M" curves -> matrix ] -> "B" curves -> PCS
//   B2A:  device <- [ "A" curves <- CLUT ] <- [ "M" curves <- matrix ] <- "B" curves <- PCS

typedef struct skcms_A2B {
    // Optional: N 1D "A" curves, followed by an N-dimensional CLUT.
    // If input_channels == 0, these curves and CLUT are skipped,
    // Otherwise, input_channels must be in [1, 4].
    skcms_Curve     input_curves[4];
    const uint8_t*  grid_8;
    const uint8_t*  grid_16;
    uint32_t        input_channels;
    uint8_t         grid_points[4];

    // Optional: 3 1D "M" curves, followed by a color matrix.
    // If matrix_channels == 0, these curves and matrix are skipped,
    // Otherwise, matrix_channels must be 3.
    skcms_Curve     matrix_curves[3];
    skcms_Matrix3x4 matrix;
    uint32_t        matrix_channels;

    // Required: 3 1D "B" curves. Always present, and output_channels must be 3.
    uint32_t        output_channels; // list first to pack with matrix_channels
    skcms_Curve     output_curves[3];
} skcms_A2B;

typedef struct skcms_B2A {
    // Required: 3 1D "B" curves. Always present, and input_channels must be 3.
    skcms_Curve     input_curves[3];
    uint32_t        input_channels;

    // Optional: a color matrix, followed by 3 1D "M" curves.
    // If matrix_channels == 0, this matrix and these curves are skipped,
    // Otherwise, matrix_channels must be 3.
    uint32_t        matrix_channels; // list first to pack with input_channels
    skcms_Curve     matrix_curves[3];
    skcms_Matrix3x4 matrix;

    // Optional: an N-dimensional CLUT, followed by N 1D "A" curves.
    // If output_channels == 0, this CLUT and these curves are skipped,
    // Otherwise, output_channels must be in [1, 4].
    skcms_Curve     output_curves[4];
    const uint8_t*  grid_8;
    const uint8_t*  grid_16;
    uint8_t         grid_points[4];
    uint32_t        output_channels;
} skcms_B2A;

typedef struct skcms_CICP {
    uint8_t color_primaries;
    uint8_t transfer_characteristics;
    uint8_t matrix_coefficients;
    uint8_t video_full_range_flag;
} skcms_CICP;

typedef struct skcms_ICCProfile {
    const uint8_t* buffer;

    uint32_t size;
    uint32_t data_color_space;
    uint32_t pcs;
    uint32_t tag_count;

    // skcms_Parse() will set commonly-used fields for you when possible:

    // If we can parse red, green and blue transfer curves from the profile,
    // trc will be set to those three curves, and has_trc will be true.
    skcms_Curve            trc[3];

    // If this profile's gamut can be represented by a 3x3 transform to XYZD50,
    // skcms_Parse() sets toXYZD50 to that transform and has_toXYZD50 to true.
    skcms_Matrix3x3        toXYZD50;

    // If the profile has a valid A2B0 or A2B1 tag, skcms_Parse() sets A2B to
    // that data, and has_A2B to true.  skcms_ParseWithA2BPriority() does the
    // same following any user-provided prioritization of A2B0, A2B1, or A2B2.
    skcms_A2B              A2B;

    // If the profile has a valid B2A0 or B2A1 tag, skcms_Parse() sets B2A to
    // that data, and has_B2A to true.  skcms_ParseWithA2BPriority() does the
    // same following any user-provided prioritization of B2A0, B2A1, or B2A2.
    skcms_B2A              B2A;

    // If the profile has a valid CICP tag, skcms_Parse() sets CICP to that data,
    // and has_CICP to true.
    skcms_CICP             CICP;

    bool                   has_trc;
    bool                   has_toXYZD50;
    bool                   has_A2B;
    bool                   has_B2A;
    bool                   has_CICP;
} skcms_ICCProfile;

// The sRGB color profile is so commonly used that we offer a canonical skcms_ICCProfile for it.
SKCMS_API const skcms_ICCProfile* skcms_sRGB_profile(void);
// Ditto for XYZD50, the most common profile connection space.
SKCMS_API const skcms_ICCProfile* skcms_XYZD50_profile(void);

SKCMS_API const skcms_TransferFunction* skcms_sRGB_TransferFunction(void);
SKCMS_API const skcms_TransferFunction* skcms_sRGB_Inverse_TransferFunction(void);
SKCMS_API const skcms_TransferFunction* skcms_Identity_TransferFunction(void);

// Practical equality test for two skcms_ICCProfiles.
// The implementation is subject to change, but it will always try to answer
// "can I substitute A for B?" and "can I skip transforming from A to B?".
SKCMS_API bool skcms_ApproximatelyEqualProfiles(const skcms_ICCProfile* A,
                                                const skcms_ICCProfile* B);

// Practical test that answers: Is curve roughly the inverse of inv_tf? Typically used by passing
// the inverse of a known parametric transfer function (like sRGB), to determine if a particular
// curve is very close to sRGB.
SKCMS_API bool skcms_AreApproximateInverses(const skcms_Curve* curve,
                                            const skcms_TransferFunction* inv_tf);

// Similar to above, answering the question for all three TRC curves of the given profile. Again,
// passing skcms_sRGB_InverseTransferFunction as inv_tf will answer the question:
// "Does this profile have a transfer function that is very close to sRGB?"
SKCMS_API bool skcms_TRCs_AreApproximateInverse(const skcms_ICCProfile* profile,
                                                const skcms_TransferFunction* inv_tf);

// Parse an ICC profile and return true if possible, otherwise return false.
// Selects an A2B profile (if present) according to priority list (each entry 0-2).
// The buffer is not copied; it must remain valid as long as the skcms_ICCProfile will be used.
SKCMS_API bool skcms_ParseWithA2BPriority(const void*, size_t,
                                          const int priority[], int priorities,
                                          skcms_ICCProfile*);

static inline bool skcms_Parse(const void* buf, size_t len, skcms_ICCProfile* profile) {
    // For continuity of existing user expectations,
    // prefer A2B0 (perceptual) over A2B1 (relative colormetric), and ignore A2B2 (saturation).
    const int priority[] = {0,1};
    return skcms_ParseWithA2BPriority(buf, len,
                                      priority, sizeof(priority)/sizeof(*priority),
                                      profile);
}

SKCMS_API bool skcms_ApproximateCurve(const skcms_Curve* curve,
                                      skcms_TransferFunction* approx,
                                      float* max_error);

SKCMS_API bool skcms_GetCHAD(const skcms_ICCProfile*, skcms_Matrix3x3*);
SKCMS_API bool skcms_GetWTPT(const skcms_ICCProfile*, float xyz[3]);

// Returns the number of channels of input data that are expected on the "A" side of the profile.
// This is useful for image codecs, where the image data and the accompanying profile might have
// conflicting data shapes. In some cases, the result is unclear or invalid. In that case, the
// function will return a negative value to signal an error.
SKCMS_API int skcms_GetInputChannelCount(const skcms_ICCProfile*);

// These are common ICC signature values
enum {
    // common data_color_space values
    skcms_Signature_CMYK = 0x434D594B,
    skcms_Signature_Gray = 0x47524159,
    skcms_Signature_RGB  = 0x52474220,

    // pcs (or data_color_space)
    skcms_Signature_Lab  = 0x4C616220,
    skcms_Signature_XYZ  = 0x58595A20,

    // other, less common data_color_space values
    skcms_Signature_CIELUV = 0x4C757620,
    skcms_Signature_YCbCr  = 0x59436272,
    skcms_Signature_CIEYxy = 0x59787920,
    skcms_Signature_HSV    = 0x48535620,
    skcms_Signature_HLS    = 0x484C5320,
    skcms_Signature_CMY    = 0x434D5920,
    skcms_Signature_2CLR   = 0x32434C52,
    skcms_Signature_3CLR   = 0x33434C52,
    skcms_Signature_4CLR   = 0x34434C52,
    skcms_Signature_5CLR   = 0x35434C52,
    skcms_Signature_6CLR   = 0x36434C52,
    skcms_Signature_7CLR   = 0x37434C52,
    skcms_Signature_8CLR   = 0x38434C52,
    skcms_Signature_9CLR   = 0x39434C52,
    skcms_Signature_10CLR  = 0x41434C52,
    skcms_Signature_11CLR  = 0x42434C52,
    skcms_Signature_12CLR  = 0x43434C52,
    skcms_Signature_13CLR  = 0x44434C52,
    skcms_Signature_14CLR  = 0x45434C52,
    skcms_Signature_15CLR  = 0x46434C52,
};

typedef enum skcms_PixelFormat {
    skcms_PixelFormat_A_8,
    skcms_PixelFormat_A_8_,
    skcms_PixelFormat_G_8,
    skcms_PixelFormat_G_8_,
    skcms_PixelFormat_GA_88,  // Grayscale with alpha.
    skcms_PixelFormat_GA_88_,

    skcms_PixelFormat_RGB_565,
    skcms_PixelFormat_BGR_565,

    skcms_PixelFormat_ABGR_4444,
    skcms_PixelFormat_ARGB_4444,

    skcms_PixelFormat_RGB_888,
    skcms_PixelFormat_BGR_888,
    skcms_PixelFormat_RGBA_8888,
    skcms_PixelFormat_BGRA_8888,
    skcms_PixelFormat_RGBA_8888_sRGB,  // Automatic sRGB encoding / decoding.
    skcms_PixelFormat_BGRA_8888_sRGB,  // (Generally used with linear transfer functions.)

    skcms_PixelFormat_RGBA_1010102,
    skcms_PixelFormat_BGRA_1010102,

    skcms_PixelFormat_RGB_161616LE,  // Little-endian.  Pointers must be 16-bit aligned.
    skcms_PixelFormat_BGR_161616LE,
    skcms_PixelFormat_RGBA_16161616LE,
    skcms_PixelFormat_BGRA_16161616LE,

    skcms_PixelFormat_RGB_161616BE,  // Big-endian.  Pointers must be 16-bit aligned.
    skcms_PixelFormat_BGR_161616BE,
    skcms_PixelFormat_RGBA_16161616BE,
    skcms_PixelFormat_BGRA_16161616BE,

    skcms_PixelFormat_RGB_hhh_Norm,  // 1-5-10 half-precision float in [0,1]
    skcms_PixelFormat_BGR_hhh_Norm,  // Pointers must be 16-bit aligned.
    skcms_PixelFormat_RGBA_hhhh_Norm,
    skcms_PixelFormat_BGRA_hhhh_Norm,

    skcms_PixelFormat_RGB_hhh,  // 1-5-10 half-precision float.
    skcms_PixelFormat_BGR_hhh,  // Pointers must be 16-bit aligned.
    skcms_PixelFormat_RGBA_hhhh,
    skcms_PixelFormat_BGRA_hhhh,

    skcms_PixelFormat_RGB_fff,  // 1-8-23 single-precision float (the normal kind).
    skcms_PixelFormat_BGR_fff,  // Pointers must be 32-bit aligned.
    skcms_PixelFormat_RGBA_ffff,
    skcms_PixelFormat_BGRA_ffff,

    skcms_PixelFormat_RGB_101010x_XR,    // Note: This is located here to signal no clamping.
    skcms_PixelFormat_BGR_101010x_XR,    // Compatible with MTLPixelFormatBGR10_XR.
    skcms_PixelFormat_RGBA_10101010_XR,  // Note: This is located here to signal no clamping.
    skcms_PixelFormat_BGRA_10101010_XR,  // Compatible with MTLPixelFormatBGRA10_XR.
} skcms_PixelFormat;

// We always store any alpha channel linearly.  In the chart below, tf-1() is the inverse
// transfer function for the given color profile (applying the transfer function linearizes).

// We treat opaque as a strong requirement, not just a performance hint: we will ignore
// any source alpha and treat it as 1.0, and will make sure that any destination alpha
// channel is filled with the equivalent of 1.0.

// We used to offer multiple types of premultiplication, but now just one, PremulAsEncoded.
// This is the premul you're probably used to working with.

typedef enum skcms_AlphaFormat {
    skcms_AlphaFormat_Opaque,          // alpha is always opaque
                                       //   tf-1(r),   tf-1(g),   tf-1(b),   1.0
    skcms_AlphaFormat_Unpremul,        // alpha and color are unassociated
                                       //   tf-1(r),   tf-1(g),   tf-1(b),   a
    skcms_AlphaFormat_PremulAsEncoded, // premultiplied while encoded
                                       //   tf-1(r)*a, tf-1(g)*a, tf-1(b)*a, a
} skcms_AlphaFormat;

// Convert npixels pixels from src format and color profile to dst format and color profile
// and return true, otherwise return false.  It is safe to alias dst == src if dstFmt == srcFmt.
SKCMS_API bool skcms_Transform(const void*             src,
                               skcms_PixelFormat       srcFmt,
                               skcms_AlphaFormat       srcAlpha,
                               const skcms_ICCProfile* srcProfile,
                               void*                   dst,
                               skcms_PixelFormat       dstFmt,
                               skcms_AlphaFormat       dstAlpha,
                               const skcms_ICCProfile* dstProfile,
                               size_t                  npixels);

// If profile can be used as a destination in skcms_Transform, return true. Otherwise, attempt to
// rewrite it with approximations where reasonable. If successful, return true. If no reasonable
// approximation exists, leave the profile unchanged and return false.
SKCMS_API bool skcms_MakeUsableAsDestination(skcms_ICCProfile* profile);

// If profile can be used as a destination with a single parametric transfer function (ie for
// rasterization), return true. Otherwise, attempt to rewrite it with approximations where
// reasonable. If successful, return true. If no reasonable approximation exists, leave the
// profile unchanged and return false.
SKCMS_API bool skcms_MakeUsableAsDestinationWithSingleCurve(skcms_ICCProfile* profile);

// Returns a matrix to adapt XYZ color from given the whitepoint to D50.
SKCMS_API bool skcms_AdaptToXYZD50(float wx, float wy,
                                   skcms_Matrix3x3* toXYZD50);

// Returns a matrix to convert RGB color into XYZ adapted to D50, given the
// primaries and whitepoint of the RGB model.
SKCMS_API bool skcms_PrimariesToXYZD50(float rx, float ry,
                                       float gx, float gy,
                                       float bx, float by,
                                       float wx, float wy,
                                       skcms_Matrix3x3* toXYZD50);

// Call before your first call to skcms_Transform() to skip runtime CPU detection.
SKCMS_API void skcms_DisableRuntimeCPUDetection(void);

// Utilities for programmatically constructing profiles
static inline void skcms_Init(skcms_ICCProfile* p) {
    memset(p, 0, sizeof(*p));
    p->data_color_space = skcms_Signature_RGB;
    p->pcs = skcms_Signature_XYZ;
}

static inline void skcms_SetTransferFunction(skcms_ICCProfile* p,
                                             const skcms_TransferFunction* tf) {
    p->has_trc = true;
    for (int i = 0; i < 3; ++i) {
        p->trc[i].table_entries = 0;
        p->trc[i].parametric = *tf;
    }
}

static inline void skcms_SetXYZD50(skcms_ICCProfile* p, const skcms_Matrix3x3* m) {
    p->has_toXYZD50 = true;
    p->toXYZD50 = *m;
}

#ifdef __cplusplus
}
#endif

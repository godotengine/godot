/**
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef ZSTD_ERRORS_H_398273423
#define ZSTD_ERRORS_H_398273423

#if defined (__cplusplus)
extern "C" {
#endif

/*===== dependency =====*/
#include <stddef.h>   /* size_t */


/* =====   ZSTDERRORLIB_API : control library symbols visibility   ===== */
#ifndef ZSTDERRORLIB_VISIBILITY
#  if defined(__GNUC__) && (__GNUC__ >= 4)
#    define ZSTDERRORLIB_VISIBILITY __attribute__ ((visibility ("default")))
#  else
#    define ZSTDERRORLIB_VISIBILITY
#  endif
#endif
#if defined(ZSTD_DLL_EXPORT) && (ZSTD_DLL_EXPORT==1)
#  define ZSTDERRORLIB_API __declspec(dllexport) ZSTDERRORLIB_VISIBILITY
#elif defined(ZSTD_DLL_IMPORT) && (ZSTD_DLL_IMPORT==1)
#  define ZSTDERRORLIB_API __declspec(dllimport) ZSTDERRORLIB_VISIBILITY /* It isn't required but allows to generate better code, saving a function pointer load from the IAT and an indirect jump.*/
#else
#  define ZSTDERRORLIB_API ZSTDERRORLIB_VISIBILITY
#endif

/*-****************************************
 *  error codes list
 *  note : this API is still considered unstable
 *         it should not be used with a dynamic library
 *         only static linking is allowed
 ******************************************/
typedef enum {
  ZSTD_error_no_error,
  ZSTD_error_GENERIC,
  ZSTD_error_prefix_unknown,
  ZSTD_error_version_unsupported,
  ZSTD_error_parameter_unknown,
  ZSTD_error_frameParameter_unsupported,
  ZSTD_error_frameParameter_unsupportedBy32bits,
  ZSTD_error_frameParameter_windowTooLarge,
  ZSTD_error_compressionParameter_unsupported,
  ZSTD_error_compressionParameter_outOfBound,
  ZSTD_error_init_missing,
  ZSTD_error_memory_allocation,
  ZSTD_error_stage_wrong,
  ZSTD_error_dstSize_tooSmall,
  ZSTD_error_srcSize_wrong,
  ZSTD_error_corruption_detected,
  ZSTD_error_checksum_wrong,
  ZSTD_error_tableLog_tooLarge,
  ZSTD_error_maxSymbolValue_tooLarge,
  ZSTD_error_maxSymbolValue_tooSmall,
  ZSTD_error_dictionary_corrupted,
  ZSTD_error_dictionary_wrong,
  ZSTD_error_dictionaryCreation_failed,
  ZSTD_error_frameIndex_tooLarge,
  ZSTD_error_seekableIO,
  ZSTD_error_maxCode
} ZSTD_ErrorCode;

/*! ZSTD_getErrorCode() :
    convert a `size_t` function result into a `ZSTD_ErrorCode` enum type,
    which can be used to compare with enum list published above */
ZSTDERRORLIB_API ZSTD_ErrorCode ZSTD_getErrorCode(size_t functionResult);
ZSTDERRORLIB_API const char* ZSTD_getErrorString(ZSTD_ErrorCode code);


#if defined (__cplusplus)
}
#endif

#endif /* ZSTD_ERRORS_H_398273423 */

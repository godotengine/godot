/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_error.h
 * @brief Error codes and error-name lookup for the ZXC library.
 *
 * Every public function that can fail returns a value from @ref zxc_error_t.
 * A return value < 0 indicates an error; use zxc_error_name() to convert
 * any code to a human-readable string.
 */

#ifndef ZXC_ERROR_H
#define ZXC_ERROR_H

#include "zxc_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup error Error Handling
 * @brief Error codes returned by ZXC library functions.
 * @{
 */

/**
 * @brief Error codes returned by ZXC library functions.
 *
 * All error codes are negative integers. Functions that return int or int64_t
 * will return these codes on failure. Check with `result < 0` for errors.
 *
 * Use zxc_error_name() to get a human-readable string for any error code.
 */
typedef enum {
    ZXC_OK = 0, /**< Success (no error). */

    /* Memory errors */
    ZXC_ERROR_MEMORY = -1, /**< Memory allocation failure. */

    /* Buffer/capacity errors */
    ZXC_ERROR_DST_TOO_SMALL = -2, /**< Destination buffer too small. */
    ZXC_ERROR_SRC_TOO_SMALL = -3, /**< Source buffer too small or truncated input. */

    /* Format/header errors */
    ZXC_ERROR_BAD_MAGIC = -4,    /**< Invalid magic word in file header. */
    ZXC_ERROR_BAD_VERSION = -5,  /**< Unsupported file format version. */
    ZXC_ERROR_BAD_HEADER = -6,   /**< Corrupted or invalid header (CRC mismatch). */
    ZXC_ERROR_BAD_CHECKSUM = -7, /**< Block or global checksum verification failed. */

    /* Data integrity errors */
    ZXC_ERROR_CORRUPT_DATA = -8, /**< Corrupted compressed data. */
    ZXC_ERROR_BAD_OFFSET = -9,   /**< Invalid match offset during decompression. */
    ZXC_ERROR_OVERFLOW = -10,    /**< Buffer overflow detected during processing. */

    /* I/O errors */
    ZXC_ERROR_IO = -11,         /**< Read/write/seek failure on file. */
    ZXC_ERROR_NULL_INPUT = -12, /**< Required input pointer is NULL. */

    /* Block errors */
    ZXC_ERROR_BAD_BLOCK_TYPE = -13, /**< Unknown or unexpected block type. */
    ZXC_ERROR_BAD_BLOCK_SIZE = -14, /**< Invalid block size. */

    /* Dictionary errors */
    ZXC_ERROR_DICT_REQUIRED = -15,  /**< File requires a dictionary but none was provided. */
    ZXC_ERROR_DICT_MISMATCH = -16,  /**< Provided dictionary ID does not match the file header. */
    ZXC_ERROR_DICT_TOO_LARGE = -17, /**< Dictionary exceeds maximum allowed size. */

} zxc_error_t;

/**
 * @brief Returns a human-readable name for the given error code.
 *
 * @param[in] code An error code from zxc_error_t (or any integer).
 * @return A constant string such as "ZXC_OK" or "ZXC_ERROR_MEMORY".
 *         Returns "ZXC_UNKNOWN_ERROR" for unrecognized codes.
 */
ZXC_EXPORT const char* zxc_error_name(const int code);

/** @} */ /* end of error */

#ifdef __cplusplus
}
#endif

#endif /* ZXC_ERROR_H */

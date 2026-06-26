/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_export.h
 * @brief Platform-specific symbol visibility macros.
 *
 * This header defines the `ZXC_EXPORT`, `ZXC_NO_EXPORT`, and `ZXC_DEPRECATED`
 * macros that control which symbols are exported from the shared library.
 *
 * - Define @c ZXC_STATIC_DEFINE when building or consuming ZXC as a **static**
 *   library to disable import/export annotations.
 * - When building the shared library the CMake target defines
 *   @c zxc_lib_EXPORTS automatically, selecting `dllexport` / `visibility("default")`.
 * - When consuming the shared library neither macro is defined, so the header
 *   selects `dllimport` / `visibility("default")`.
 */

#ifndef ZXC_EXPORT_H
#define ZXC_EXPORT_H

/**
 * @defgroup export Symbol Visibility
 * @brief Macros controlling DLL export/import and deprecation attributes.
 * @{
 */

#ifdef ZXC_STATIC_DEFINE

/**
 * @def ZXC_EXPORT
 * @brief Marks a symbol as part of the public shared-library API.
 *
 * Expands to nothing when building a static library (@c ZXC_STATIC_DEFINE),
 * to `__declspec(dllexport)` or `__declspec(dllimport)` on Windows, or
 * to `__attribute__((visibility("default")))` on GCC/Clang.
 */
#define ZXC_EXPORT

/**
 * @def ZXC_NO_EXPORT
 * @brief Marks a symbol as hidden (not exported from the shared library).
 *
 * Expands to nothing for static builds or Windows, and to
 * `__attribute__((visibility("hidden")))` on GCC/Clang.
 */
#define ZXC_NO_EXPORT

#else /* shared library */

#ifndef ZXC_EXPORT
#ifdef zxc_lib_EXPORTS
/* Building the library */
#ifdef _WIN32
#define ZXC_EXPORT __declspec(dllexport)
#else
#define ZXC_EXPORT __attribute__((visibility("default")))
#endif
#else
/* Consuming the library */
#ifdef _WIN32
#define ZXC_EXPORT __declspec(dllimport)
#else
#define ZXC_EXPORT __attribute__((visibility("default")))
#endif
#endif
#endif

#ifndef ZXC_NO_EXPORT
#ifdef _WIN32
#define ZXC_NO_EXPORT
#else
#define ZXC_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif

#endif /* ZXC_STATIC_DEFINE */

#ifndef ZXC_DEPRECATED
/**
 * @def ZXC_DEPRECATED
 * @brief Marks a symbol as deprecated.
 *
 * The compiler will emit a warning when a deprecated symbol is referenced.
 * Expands to `__declspec(deprecated)` on MSVC or
 * `__attribute__((__deprecated__))` on GCC/Clang.
 */
#ifdef _WIN32
#define ZXC_DEPRECATED __declspec(deprecated)
#else
#define ZXC_DEPRECATED __attribute__((__deprecated__))
#endif
#endif

/**
 * @def ZXC_DEPRECATED_EXPORT
 * @brief Combines `ZXC_EXPORT` with `ZXC_DEPRECATED`.
 */
#ifndef ZXC_DEPRECATED_EXPORT
#define ZXC_DEPRECATED_EXPORT ZXC_EXPORT ZXC_DEPRECATED
#endif

/**
 * @def ZXC_DEPRECATED_NO_EXPORT
 * @brief Combines `ZXC_NO_EXPORT` with `ZXC_DEPRECATED`.
 */
#ifndef ZXC_DEPRECATED_NO_EXPORT
#define ZXC_DEPRECATED_NO_EXPORT ZXC_NO_EXPORT ZXC_DEPRECATED
#endif

/** @} */ /* end of export */

#endif /* ZXC_EXPORT_H */

/* filter_neon.S - placeholder file
 *
 * Copyright (c) 2024 Cosmin Truta
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

/* IMPORTANT NOTE:
 *
 * Historically, the hand-coded assembler implementation of Neon optimizations
 * in this module had not been in sync with the intrinsics-based implementation
 * in filter_neon_intrinsics.c and palette_neon_intrinsics.c, at least since
 * the introduction of riffled palette optimizations. Moreover, the assembler
 * code used to work on 32-bit ARM only, and it caused problems, even if empty,
 * on 64-bit ARM.
 *
 * All references to this module from our internal build scripts and projects
 * have been removed.
 *
 * For the external projects that might still expect this module to be present,
 * we leave this stub in place, for the remaining lifetime of libpng-1.6.x.
 * Everything should continue to function normally, as long as there are no
 * deliberate attempts to use the old hand-made assembler code. A build error
 * will be raised otherwise.
 */

/* This is required to get the symbol renames, which are #defines, and the
 * definitions (or not) of PNG_ARM_NEON_OPT and PNG_ARM_NEON_IMPLEMENTATION.
 */
#define PNG_VERSION_INFO_ONLY
#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED
#if PNG_ARM_NEON_IMPLEMENTATION == 2 /* hand-coded assembler */
#if PNG_ARM_NEON_OPT > 0

#if defined(__clang__)
#define GNUC_VERSION 0 /* not gcc, although it might pretend to be */
#elif defined(__GNUC__)
#define GNUC_MAJOR (__GNUC__ + 0)
#define GNUC_MINOR (__GNUC_MINOR__ + 0)
#define GNUC_PATCHLEVEL (__GNUC_PATCHLEVEL__ + 0)
#define GNUC_VERSION (GNUC_MAJOR * 10000 + GNUC_MINOR * 100 + GNUC_PATCHLEVEL)
#else
#define GNUC_VERSION 0 /* not gcc */
#endif

#if (GNUC_VERSION > 0) && (GNUC_VERSION < 40300)
#error "PNG_ARM_NEON is not supported with gcc versions earlier than 4.3.0"
#elif GNUC_VERSION == 40504
#error "PNG_ARM_NEON is not supported with gcc version 4.5.4"
#else
#error "Please use 'arm/*_neon_intrinsics.c' for PNG_ARM_NEON support"
#endif

#endif /* PNG_ARM_NEON_OPT > 0 */
#endif /* PNG_ARM_NEON_IMPLEMENTATION == 2 */
#endif /* READ */

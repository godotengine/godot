/*
 * Copyright © 2022 Red Hat, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#ifndef HB_FEATURES_H
#define HB_FEATURES_H

HB_BEGIN_DECLS

/**
 * SECTION: hb-features
 * @title: hb-features
 * @short_description: Feature detection
 * @include: hb-features.h
 *
 * Macros for detecting optional HarfBuzz features at build time.
 **/

/**
 * HB_HAS_CAIRO:
 *
 * Defined if Harfbuzz has been built with cairo support.
 */
#

/**
 * HB_HAS_CORETEXT:
 *
 * Defined if Harfbuzz has been built with CoreText support.
 */
#undef HB_HAS_CORETEXT

/**
 * HB_HAS_DIRECTWRITE:
 *
 * Defined if Harfbuzz has been built with DirectWrite support.
 */
#undef HB_HAS_DIRECTWRITE

/**
 * HB_HAS_FREETYPE:
 *
 * Defined if Harfbuzz has been built with Freetype support.
 */
#define HB_HAS_FREETYPE 1

/**
 * HB_HAS_GDI:
 *
 * Defined if Harfbuzz has been built with GDI support.
 */
#undef HB_HAS_GDI

/**
 * HB_HAS_GLIB:
 *
 * Defined if Harfbuzz has been built with GLib support.
 */
#define HB_HAS_GLIB 1

/**
 * HB_HAS_GOBJECT:
 *
 * Defined if Harfbuzz has been built with GObject support.
 */
#undef HB_HAS_GOBJECT

/**
 * HB_HAS_GRAPHITE:
 *
 * Defined if Harfbuzz has been built with Graphite support.
 */
#undef HB_HAS_GRAPHITE

/**
 * HB_HAS_ICU:
 *
 * Defined if Harfbuzz has been built with ICU support.
 */
#undef HB_HAS_ICU

/**
 * HB_HAS_UNISCRIBE:
 *
 * Defined if Harfbuzz has been built with Uniscribe support.
 */
#undef HB_HAS_UNISCRIBE

/**
 * HB_HAS_WASM:
 *
 * Defined if Harfbuzz has been built with WebAssembly support.
 */
#undef HB_HAS_WASM


HB_END_DECLS

#endif /* HB_FEATURES_H */

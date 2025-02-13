/*
 * Copyright © 2013 Ran Benita
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _XKBCOMMON_COMPOSE_H
#define _XKBCOMMON_COMPOSE_H

#include <xkbcommon/xkbcommon.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file
 * libxkbcommon Compose API - support for Compose and dead-keys.
 */

/**
 * @defgroup compose Compose and dead-keys support
 * Support for Compose and dead-keys.
 * @since 0.5.0
 *
 * @{
 */

/**
 * @page compose-overview Overview
 * @parblock
 *
 * Compose and dead-keys are a common feature of many keyboard input
 * systems.  They extend the range of the keysysm that can be produced
 * directly from a keyboard by using a sequence of key strokes, instead
 * of just one.
 *
 * Here are some example sequences, in the libX11 Compose file format:
 *
 *     <dead_acute> <a>         : "á"   aacute # LATIN SMALL LETTER A WITH ACUTE
 *     <Multi_key> <A> <T>      : "@"   at     # COMMERCIAL AT
 *
 * When the user presses a key which produces the `<dead_acute>` keysym,
 * nothing initially happens (thus the key is dubbed a "dead-key").  But
 * when the user enters `<a>`, "á" is "composed", in place of "a".  If
 * instead the user had entered a keysym which does not follow
 * `<dead_acute>` in any compose sequence, the sequence is said to be
 * "cancelled".
 *
 * Compose files define many such sequences.  For a description of the
 * common file format for Compose files, see the Compose(5) man page.
 *
 * A successfuly-composed sequence has two results: a keysym and a UTF-8
 * string.  At least one of the two is defined for each sequence.  If only
 * a keysym is given, the keysym's string representation is used for the
 * result string (using xkb_keysym_to_utf8()).
 *
 * This library provides low-level support for Compose file parsing and
 * processing.  Higher-level APIs (such as libX11's `Xutf8LookupString`(3))
 * may be built upon it, or it can be used directly.
 *
 * @endparblock
 */

/**
 * @page compose-conflicting Conflicting Sequences
 * @parblock
 *
 * To avoid ambiguity, a sequence is not allowed to be a prefix of another.
 * In such a case, the conflict is resolved thus:
 *
 * 1. A longer sequence overrides a shorter one.
 * 2. An equal sequence overrides an existing one.
 * 3. A shorter sequence does not override a longer one.
 *
 * Sequences of length 1 are allowed.
 *
 * @endparblock
 */

/**
 * @page compose-cancellation Cancellation Behavior
 * @parblock
 *
 * What should happen when a sequence is cancelled?  For example, consider
 * there are only the above sequences, and the input keysyms are
 * `<dead_acute> <b>`.  There are a few approaches:
 *
 * 1. Swallow the cancelling keysym; that is, no keysym is produced.
 *    This is the approach taken by libX11.
 * 2. Let the cancelling keysym through; that is, `<b>` is produced.
 * 3. Replay the entire sequence; that is, `<dead_acute> <b>` is produced.
 *    This is the approach taken by Microsoft Windows (approximately;
 *    instead of `<dead_acute>`, the underlying key is used.  This is
 *    difficult to simulate with XKB keymaps).
 *
 * You can program whichever approach best fits users' expectations.
 *
 * @endparblock
 */

/**
 * @struct xkb_compose_table
 * Opaque Compose table object.
 *
 * The compose table holds the definitions of the Compose sequences, as
 * gathered from Compose files.  It is immutable.
 */
struct xkb_compose_table;

/**
 * @struct xkb_compose_state
 * Opaque Compose state object.
 *
 * The compose state maintains state for compose sequence matching, such
 * as which possible sequences are being matched, and the position within
 * these sequences.  It acts as a simple state machine wherein keysyms are
 * the input, and composed keysyms and strings are the output.
 *
 * The compose state is usually associated with a keyboard device.
 */
struct xkb_compose_state;

/** Flags affecting Compose file compilation. */
enum xkb_compose_compile_flags {
    /** Do not apply any flags. */
    XKB_COMPOSE_COMPILE_NO_FLAGS = 0
};

/** The recognized Compose file formats. */
enum xkb_compose_format {
    /** The classic libX11 Compose text format, described in Compose(5). */
    XKB_COMPOSE_FORMAT_TEXT_V1 = 1
};

/**
 * @page compose-locale Compose Locale
 * @parblock
 *
 * Compose files are locale dependent:
 * - Compose files are written for a locale, and the locale is used when
 *   searching for the appropriate file to use.
 * - Compose files may reference the locale internally, with directives
 *   such as \%L.
 *
 * As such, functions like xkb_compose_table_new_from_locale() require
 * a `locale` parameter.  This will usually be the current locale (see
 * locale(7) for more details).  You may also want to allow the user to
 * explicitly configure it, so he can use the Compose file of a given
 * locale, but not use that locale for other things.
 *
 * You may query the current locale as follows:
 * @code
 *     const char *locale;
 *     locale = setlocale(LC_CTYPE, NULL);
 * @endcode
 *
 * This will only give useful results if the program had previously set
 * the current locale using setlocale(3), with `LC_CTYPE` or `LC_ALL`
 * and a non-NULL argument.
 *
 * If you prefer not to use the locale system of the C runtime library,
 * you may nevertheless obtain the user's locale directly using
 * environment variables, as described in locale(7).  For example,
 * @code
 *     const char *locale;
 *     locale = getenv("LC_ALL");
 *     if (!locale || !*locale)
 *         locale = getenv("LC_CTYPE");
 *     if (!locale || !*locale)
 *         locale = getenv("LANG");
 *     if (!locale || !*locale)
 *         locale = "C";
 * @endcode
 *
 * Note that some locales supported by the C standard library may not
 * have a Compose file assigned.
 *
 * @endparblock
 */

/**
 * Create a compose table for a given locale.
 *
 * The locale is used for searching the file-system for an appropriate
 * Compose file.  The search order is described in Compose(5).  It is
 * affected by the following environment variables:
 *
 * 1. `XCOMPOSEFILE` - see Compose(5).
 * 2. `XDG_CONFIG_HOME` - before `$HOME/.XCompose` is checked,
 *    `$XDG_CONFIG_HOME/XCompose` is checked (with a fall back to
 *    `$HOME/.config/XCompose` if `XDG_CONFIG_HOME` is not defined).
 *    This is a libxkbcommon extension to the search procedure in
 *    Compose(5) (since libxkbcommon 1.0.0). Note that other
 *    implementations, such as libX11, might not find a Compose file in
 *    this path.
 * 3. `HOME` - see Compose(5).
 * 4. `XLOCALEDIR` - if set, used as the base directory for the system's
 *    X locale files, e.g. `/usr/share/X11/locale`, instead of the
 *    preconfigured directory.
 *
 * @param context
 *     The library context in which to create the compose table.
 * @param locale
 *     The current locale.  See @ref compose-locale.
 *     \n
 *     The value is copied, so it is safe to pass the result of getenv(3)
 *     (or similar) without fear of it being invalidated by a subsequent
 *     setenv(3) (or similar).
 * @param flags
 *     Optional flags for the compose table, or 0.
 *
 * @returns A compose table for the given locale, or NULL if the
 * compilation failed or a Compose file was not found.
 *
 * @memberof xkb_compose_table
 */
struct xkb_compose_table *
xkb_compose_table_new_from_locale(struct xkb_context *context,
                                  const char *locale,
                                  enum xkb_compose_compile_flags flags);

/**
 * Create a new compose table from a Compose file.
 *
 * @param context
 *     The library context in which to create the compose table.
 * @param file
 *     The Compose file to compile.
 * @param locale
 *     The current locale.  See @ref compose-locale.
 * @param format
 *     The text format of the Compose file to compile.
 * @param flags
 *     Optional flags for the compose table, or 0.
 *
 * @returns A compose table compiled from the given file, or NULL if
 * the compilation failed.
 *
 * @memberof xkb_compose_table
 */
struct xkb_compose_table *
xkb_compose_table_new_from_file(struct xkb_context *context,
                                FILE *file,
                                const char *locale,
                                enum xkb_compose_format format,
                                enum xkb_compose_compile_flags flags);

/**
 * Create a new compose table from a memory buffer.
 *
 * This is just like xkb_compose_table_new_from_file(), but instead of
 * a file, gets the table as one enormous string.
 *
 * @see xkb_compose_table_new_from_file()
 * @memberof xkb_compose_table
 */
struct xkb_compose_table *
xkb_compose_table_new_from_buffer(struct xkb_context *context,
                                  const char *buffer, size_t length,
                                  const char *locale,
                                  enum xkb_compose_format format,
                                  enum xkb_compose_compile_flags flags);

/**
 * Take a new reference on a compose table.
 *
 * @returns The passed in object.
 *
 * @memberof xkb_compose_table
 */
struct xkb_compose_table *
xkb_compose_table_ref(struct xkb_compose_table *table);

/**
 * Release a reference on a compose table, and possibly free it.
 *
 * @param table The object.  If it is NULL, this function does nothing.
 *
 * @memberof xkb_compose_table
 */
void
xkb_compose_table_unref(struct xkb_compose_table *table);

/** Flags for compose state creation. */
enum xkb_compose_state_flags {
    /** Do not apply any flags. */
    XKB_COMPOSE_STATE_NO_FLAGS = 0
};

/**
 * Create a new compose state object.
 *
 * @param table
 *     The compose table the state will use.
 * @param flags
 *     Optional flags for the compose state, or 0.
 *
 * @returns A new compose state, or NULL on failure.
 *
 * @memberof xkb_compose_state
 */
struct xkb_compose_state *
xkb_compose_state_new(struct xkb_compose_table *table,
                      enum xkb_compose_state_flags flags);

/**
 * Take a new reference on a compose state object.
 *
 * @returns The passed in object.
 *
 * @memberof xkb_compose_state
 */
struct xkb_compose_state *
xkb_compose_state_ref(struct xkb_compose_state *state);

/**
 * Release a reference on a compose state object, and possibly free it.
 *
 * @param state The object.  If NULL, do nothing.
 *
 * @memberof xkb_compose_state
 */
void
xkb_compose_state_unref(struct xkb_compose_state *state);

/**
 * Get the compose table which a compose state object is using.
 *
 * @returns The compose table which was passed to xkb_compose_state_new()
 * when creating this state object.
 *
 * This function does not take a new reference on the compose table; you
 * must explicitly reference it yourself if you plan to use it beyond the
 * lifetime of the state.
 *
 * @memberof xkb_compose_state
 */
struct xkb_compose_table *
xkb_compose_state_get_compose_table(struct xkb_compose_state *state);

/** Status of the Compose sequence state machine. */
enum xkb_compose_status {
    /** The initial state; no sequence has started yet. */
    XKB_COMPOSE_NOTHING,
    /** In the middle of a sequence. */
    XKB_COMPOSE_COMPOSING,
    /** A complete sequence has been matched. */
    XKB_COMPOSE_COMPOSED,
    /** The last sequence was cancelled due to an unmatched keysym. */
    XKB_COMPOSE_CANCELLED
};

/** The effect of a keysym fed to xkb_compose_state_feed(). */
enum xkb_compose_feed_result {
    /** The keysym had no effect - it did not affect the status. */
    XKB_COMPOSE_FEED_IGNORED,
    /** The keysym started, advanced or cancelled a sequence. */
    XKB_COMPOSE_FEED_ACCEPTED
};

/**
 * Feed one keysym to the Compose sequence state machine.
 *
 * This function can advance into a compose sequence, cancel a sequence,
 * start a new sequence, or do nothing in particular.  The resulting
 * status may be observed with xkb_compose_state_get_status().
 *
 * Some keysyms, such as keysyms for modifier keys, are ignored - they
 * have no effect on the status or otherwise.
 *
 * The following is a description of the possible status transitions, in
 * the format CURRENT STATUS => NEXT STATUS, given a non-ignored input
 * keysym `keysym`:
 *
   @verbatim
   NOTHING or CANCELLED or COMPOSED =>
      NOTHING   if keysym does not start a sequence.
      COMPOSING if keysym starts a sequence.
      COMPOSED  if keysym starts and terminates a single-keysym sequence.

   COMPOSING =>
      COMPOSING if keysym advances any of the currently possible
                sequences but does not terminate any of them.
      COMPOSED  if keysym terminates one of the currently possible
                sequences.
      CANCELLED if keysym does not advance any of the currently
                possible sequences.
   @endverbatim
 *
 * The current Compose formats do not support multiple-keysyms.
 * Therefore, if you are using a function such as xkb_state_key_get_syms()
 * and it returns more than one keysym, consider feeding XKB_KEY_NoSymbol
 * instead.
 *
 * @param state
 *     The compose state object.
 * @param keysym
 *     A keysym, usually obtained after a key-press event, with a
 *     function such as xkb_state_key_get_one_sym().
 *
 * @returns Whether the keysym was ignored.  This is useful, for example,
 * if you want to keep a record of the sequence matched thus far.
 *
 * @memberof xkb_compose_state
 */
enum xkb_compose_feed_result
xkb_compose_state_feed(struct xkb_compose_state *state,
                       xkb_keysym_t keysym);

/**
 * Reset the Compose sequence state machine.
 *
 * The status is set to XKB_COMPOSE_NOTHING, and the current sequence
 * is discarded.
 *
 * @memberof xkb_compose_state
 */
void
xkb_compose_state_reset(struct xkb_compose_state *state);

/**
 * Get the current status of the compose state machine.
 *
 * @see xkb_compose_status
 * @memberof xkb_compose_state
 **/
enum xkb_compose_status
xkb_compose_state_get_status(struct xkb_compose_state *state);

/**
 * Get the result Unicode/UTF-8 string for a composed sequence.
 *
 * See @ref compose-overview for more details.  This function is only
 * useful when the status is XKB_COMPOSE_COMPOSED.
 *
 * @param[in] state
 *     The compose state.
 * @param[out] buffer
 *     A buffer to write the string into.
 * @param[in] size
 *     Size of the buffer.
 *
 * @warning If the buffer passed is too small, the string is truncated
 * (though still NUL-terminated).
 *
 * @returns
 *   The number of bytes required for the string, excluding the NUL byte.
 *   If the sequence is not complete, or does not have a viable result
 *   string, returns 0, and sets `buffer` to the empty string (if possible).
 * @returns
 *   You may check if truncation has occurred by comparing the return value
 *   with the size of `buffer`, similarly to the `snprintf`(3) function.
 *   You may safely pass NULL and 0 to `buffer` and `size` to find the
 *   required size (without the NUL-byte).
 *
 * @memberof xkb_compose_state
 **/
int
xkb_compose_state_get_utf8(struct xkb_compose_state *state,
                           char *buffer, size_t size);

/**
 * Get the result keysym for a composed sequence.
 *
 * See @ref compose-overview for more details.  This function is only
 * useful when the status is XKB_COMPOSE_COMPOSED.
 *
 * @returns The result keysym.  If the sequence is not complete, or does
 * not specify a result keysym, returns XKB_KEY_NoSymbol.
 *
 * @memberof xkb_compose_state
 **/
xkb_keysym_t
xkb_compose_state_get_one_sym(struct xkb_compose_state *state);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _XKBCOMMON_COMPOSE_H */

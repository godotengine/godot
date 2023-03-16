#ifndef foovolumehfoo
#define foovolumehfoo

/***
  This file is part of PulseAudio.

  Copyright 2004-2006 Lennart Poettering
  Copyright 2006 Pierre Ossman <ossman@cendio.se> for Cendio AB

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 2.1 of the License,
  or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#include <inttypes.h>
#include <limits.h>

#include <pulse/cdecl.h>
#include <pulse/gccmacro.h>
#include <pulse/sample.h>
#include <pulse/channelmap.h>
#include <pulse/version.h>

/** \page volume Volume Control
 *
 * \section overv_sec Overview
 *
 * Sinks, sources, sink inputs and samples can all have their own volumes.
 * To deal with these, The PulseAudio library contains a number of functions
 * that ease handling.
 *
 * The basic volume type in PulseAudio is the \ref pa_volume_t type. Most of
 * the time, applications will use the aggregated pa_cvolume structure that
 * can store the volume of all channels at once.
 *
 * Volumes commonly span between muted (0%), and normal (100%). It is possible
 * to set volumes to higher than 100%, but clipping might occur.
 *
 * There is no single well-defined meaning attached to the 100% volume for a
 * sink input. In fact, it depends on the server configuration. With flat
 * volumes enabled (the default in most Linux distributions), it means the
 * maximum volume that the sound hardware is capable of, which is usually so
 * high that you absolutely must not set sink input volume to 100% unless the
 * the user explicitly requests that (note that usually you shouldn't set the
 * volume anyway if the user doesn't explicitly request it, instead, let
 * PulseAudio decide the volume for the sink input). With flat volumes disabled
 * (the default in Ubuntu), the sink input volume is relative to the sink
 * volume, so 100% sink input volume means that the sink input is played at the
 * current sink volume level. In this case 100% is often a good default volume
 * for a sink input, although you still should let PulseAudio decide the
 * default volume. It is possible to figure out whether flat volume mode is in
 * effect for a given sink by calling pa_context_get_sink_info_by_name().
 *
 * \section calc_sec Calculations
 *
 * The volumes in PulseAudio are logarithmic in nature and applications
 * shouldn't perform calculations with them directly. Instead, they should
 * be converted to and from either dB or a linear scale:
 *
 * \li dB - pa_sw_volume_from_dB() / pa_sw_volume_to_dB()
 * \li Linear - pa_sw_volume_from_linear() / pa_sw_volume_to_linear()
 *
 * For simple multiplication, pa_sw_volume_multiply() and
 * pa_sw_cvolume_multiply() can be used.
 *
 * Calculations can only be reliably performed on software volumes
 * as it is commonly unknown what scale hardware volumes relate to.
 *
 * The functions described above are only valid when used with
 * software volumes. Hence it is usually a better idea to treat all
 * volume values as opaque with a range from PA_VOLUME_MUTED (0%) to
 * PA_VOLUME_NORM (100%) and to refrain from any calculations with
 * them.
 *
 * \section conv_sec Convenience Functions
 *
 * To handle the pa_cvolume structure, the PulseAudio library provides a
 * number of convenience functions:
 *
 * \li pa_cvolume_valid() - Tests if a pa_cvolume structure is valid.
 * \li pa_cvolume_equal() - Tests if two pa_cvolume structures are identical.
 * \li pa_cvolume_channels_equal_to() - Tests if all channels of a pa_cvolume
 *                             structure have a given volume.
 * \li pa_cvolume_is_muted() - Tests if all channels of a pa_cvolume
 *                             structure are muted.
 * \li pa_cvolume_is_norm() - Tests if all channels of a pa_cvolume structure
 *                            are at a normal volume.
 * \li pa_cvolume_set() - Set the first n channels of a pa_cvolume structure to
 *                        a certain volume.
 * \li pa_cvolume_reset() - Set the first n channels of a pa_cvolume structure
 *                          to a normal volume.
 * \li pa_cvolume_mute() - Set the first n channels of a pa_cvolume structure
 *                         to a muted volume.
 * \li pa_cvolume_avg() - Return the average volume of all channels.
 * \li pa_cvolume_snprint() - Pretty print a pa_cvolume structure.
 */

/** \file
 * Constants and routines for volume handling
 *
 * See also \subpage volume
 */

PA_C_DECL_BEGIN

/** Volume specification:
 *  PA_VOLUME_MUTED: silence;
 * < PA_VOLUME_NORM: decreased volume;
 *   PA_VOLUME_NORM: normal volume;
 * > PA_VOLUME_NORM: increased volume */
typedef uint32_t pa_volume_t;

/** Normal volume (100%, 0 dB) */
#define PA_VOLUME_NORM ((pa_volume_t) 0x10000U)

/** Muted (minimal valid) volume (0%, -inf dB) */
#define PA_VOLUME_MUTED ((pa_volume_t) 0U)

/** Maximum valid volume we can store. \since 0.9.15 */
#define PA_VOLUME_MAX ((pa_volume_t) UINT32_MAX/2)

/** Recommended maximum volume to show in user facing UIs.
 * Note: UIs should deal gracefully with volumes greater than this value
 * and not cause feedback loops etc. - i.e. if the volume is more than
 * this, the UI should not limit it and push the limited value back to
 * the server. \since 0.9.23 */
#define PA_VOLUME_UI_MAX (pa_sw_volume_from_dB(+11.0))

/** Special 'invalid' volume. \since 0.9.16 */
#define PA_VOLUME_INVALID ((pa_volume_t) UINT32_MAX)

/** Check if volume is valid. \since 1.0 */
#define PA_VOLUME_IS_VALID(v) ((v) <= PA_VOLUME_MAX)

/** Clamp volume to the permitted range. \since 1.0 */
#define PA_CLAMP_VOLUME(v) (PA_CLAMP_UNLIKELY((v), PA_VOLUME_MUTED, PA_VOLUME_MAX))

/** A structure encapsulating a per-channel volume */
typedef struct pa_cvolume {
    uint8_t channels;                     /**< Number of channels */
    pa_volume_t values[PA_CHANNELS_MAX];  /**< Per-channel volume */
} pa_cvolume;

/** Return non-zero when *a == *b */
int pa_cvolume_equal(const pa_cvolume *a, const pa_cvolume *b) PA_GCC_PURE;

/** Initialize the specified volume and return a pointer to
 * it. The sample spec will have a defined state but
 * pa_cvolume_valid() will fail for it. \since 0.9.13 */
pa_cvolume* pa_cvolume_init(pa_cvolume *a);

/** Set the volume of the first n channels to PA_VOLUME_NORM */
#define pa_cvolume_reset(a, n) pa_cvolume_set((a), (n), PA_VOLUME_NORM)

/** Set the volume of the first n channels to PA_VOLUME_MUTED */
#define pa_cvolume_mute(a, n) pa_cvolume_set((a), (n), PA_VOLUME_MUTED)

/** Set the volume of the specified number of channels to the volume v */
pa_cvolume* pa_cvolume_set(pa_cvolume *a, unsigned channels, pa_volume_t v);

/** Maximum length of the strings returned by
 * pa_cvolume_snprint(). Please note that this value can change with
 * any release without warning and without being considered API or ABI
 * breakage. You should not use this definition anywhere where it
 * might become part of an ABI.*/
#define PA_CVOLUME_SNPRINT_MAX 320

/** Pretty print a volume structure */
char *pa_cvolume_snprint(char *s, size_t l, const pa_cvolume *c);

/** Maximum length of the strings returned by
 * pa_sw_cvolume_snprint_dB(). Please note that this value can change with
 * any release without warning and without being considered API or ABI
 * breakage. You should not use this definition anywhere where it
 * might become part of an ABI. \since 0.9.13 */
#define PA_SW_CVOLUME_SNPRINT_DB_MAX 448

/** Pretty print a volume structure but show dB values. \since 0.9.13 */
char *pa_sw_cvolume_snprint_dB(char *s, size_t l, const pa_cvolume *c);

/** Maximum length of the strings returned by pa_cvolume_snprint_verbose().
 * Please note that this value can change with any release without warning and
 * without being considered API or ABI breakage. You should not use this
 * definition anywhere where it might become part of an ABI. \since 5.0 */
#define PA_CVOLUME_SNPRINT_VERBOSE_MAX 1984

/** Pretty print a volume structure in a verbose way. The volume for each
 * channel is printed in several formats: the raw pa_volume_t value,
 * percentage, and if print_dB is non-zero, also the dB value. If map is not
 * NULL, the channel names will be printed. \since 5.0 */
char *pa_cvolume_snprint_verbose(char *s, size_t l, const pa_cvolume *c, const pa_channel_map *map, int print_dB);

/** Maximum length of the strings returned by
 * pa_volume_snprint(). Please note that this value can change with
 * any release without warning and without being considered API or ABI
 * breakage. You should not use this definition anywhere where it
 * might become part of an ABI. \since 0.9.15 */
#define PA_VOLUME_SNPRINT_MAX 10

/** Pretty print a volume \since 0.9.15 */
char *pa_volume_snprint(char *s, size_t l, pa_volume_t v);

/** Maximum length of the strings returned by
 * pa_sw_volume_snprint_dB(). Please note that this value can change with
 * any release without warning and without being considered API or ABI
 * breakage. You should not use this definition anywhere where it
 * might become part of an ABI. \since 0.9.15 */
#define PA_SW_VOLUME_SNPRINT_DB_MAX 11

/** Pretty print a volume but show dB values. \since 0.9.15 */
char *pa_sw_volume_snprint_dB(char *s, size_t l, pa_volume_t v);

/** Maximum length of the strings returned by pa_volume_snprint_verbose().
 * Please note that this value can change with any release without warning and
 * withou being considered API or ABI breakage. You should not use this
 * definition anywhere where it might become part of an ABI. \since 5.0 */
#define PA_VOLUME_SNPRINT_VERBOSE_MAX 35

/** Pretty print a volume in a verbose way. The volume is printed in several
 * formats: the raw pa_volume_t value, percentage, and if print_dB is non-zero,
 * also the dB value. \since 5.0 */
char *pa_volume_snprint_verbose(char *s, size_t l, pa_volume_t v, int print_dB);

/** Return the average volume of all channels */
pa_volume_t pa_cvolume_avg(const pa_cvolume *a) PA_GCC_PURE;

/** Return the average volume of all channels that are included in the
 * specified channel map with the specified channel position mask. If
 * cm is NULL this call is identical to pa_cvolume_avg(). If no
 * channel is selected the returned value will be
 * PA_VOLUME_MUTED. \since 0.9.16 */
pa_volume_t pa_cvolume_avg_mask(const pa_cvolume *a, const pa_channel_map *cm, pa_channel_position_mask_t mask) PA_GCC_PURE;

/** Return the maximum volume of all channels. \since 0.9.12 */
pa_volume_t pa_cvolume_max(const pa_cvolume *a) PA_GCC_PURE;

/** Return the maximum volume of all channels that are included in the
 * specified channel map with the specified channel position mask. If
 * cm is NULL this call is identical to pa_cvolume_max(). If no
 * channel is selected the returned value will be PA_VOLUME_MUTED.
 * \since 0.9.16 */
pa_volume_t pa_cvolume_max_mask(const pa_cvolume *a, const pa_channel_map *cm, pa_channel_position_mask_t mask) PA_GCC_PURE;

/** Return the minimum volume of all channels. \since 0.9.16 */
pa_volume_t pa_cvolume_min(const pa_cvolume *a) PA_GCC_PURE;

/** Return the minimum volume of all channels that are included in the
 * specified channel map with the specified channel position mask. If
 * cm is NULL this call is identical to pa_cvolume_min(). If no
 * channel is selected the returned value will be PA_VOLUME_MUTED.
 * \since 0.9.16 */
pa_volume_t pa_cvolume_min_mask(const pa_cvolume *a, const pa_channel_map *cm, pa_channel_position_mask_t mask) PA_GCC_PURE;

/** Return non-zero when the passed cvolume structure is valid */
int pa_cvolume_valid(const pa_cvolume *v) PA_GCC_PURE;

/** Return non-zero if the volume of all channels is equal to the specified value */
int pa_cvolume_channels_equal_to(const pa_cvolume *a, pa_volume_t v) PA_GCC_PURE;

/** Return 1 if the specified volume has all channels muted */
#define pa_cvolume_is_muted(a) pa_cvolume_channels_equal_to((a), PA_VOLUME_MUTED)

/** Return 1 if the specified volume has all channels on normal level */
#define pa_cvolume_is_norm(a) pa_cvolume_channels_equal_to((a), PA_VOLUME_NORM)

/** Multiply two volume specifications, return the result. This uses
 * PA_VOLUME_NORM as neutral element of multiplication. This is only
 * valid for software volumes! */
pa_volume_t pa_sw_volume_multiply(pa_volume_t a, pa_volume_t b) PA_GCC_CONST;

/** Multiply two per-channel volumes and return the result in
 * *dest. This is only valid for software volumes! a, b and dest may
 * point to the same structure. */
pa_cvolume *pa_sw_cvolume_multiply(pa_cvolume *dest, const pa_cvolume *a, const pa_cvolume *b);

/** Multiply a per-channel volume with a scalar volume and return the
 * result in *dest. This is only valid for software volumes! a
 * and dest may point to the same structure. \since
 * 0.9.16 */
pa_cvolume *pa_sw_cvolume_multiply_scalar(pa_cvolume *dest, const pa_cvolume *a, pa_volume_t b);

/** Divide two volume specifications, return the result. This uses
 * PA_VOLUME_NORM as neutral element of division. This is only valid
 * for software volumes! If a division by zero is tried the result
 * will be 0. \since 0.9.13 */
pa_volume_t pa_sw_volume_divide(pa_volume_t a, pa_volume_t b) PA_GCC_CONST;

/** Divide two per-channel volumes and return the result in
 * *dest. This is only valid for software volumes! a, b
 * and dest may point to the same structure. \since 0.9.13 */
pa_cvolume *pa_sw_cvolume_divide(pa_cvolume *dest, const pa_cvolume *a, const pa_cvolume *b);

/** Divide a per-channel volume by a scalar volume and return the
 * result in *dest. This is only valid for software volumes! a
 * and dest may point to the same structure. \since
 * 0.9.16 */
pa_cvolume *pa_sw_cvolume_divide_scalar(pa_cvolume *dest, const pa_cvolume *a, pa_volume_t b);

/** Convert a decibel value to a volume (amplitude, not power). This is only valid for software volumes! */
pa_volume_t pa_sw_volume_from_dB(double f) PA_GCC_CONST;

/** Convert a volume to a decibel value (amplitude, not power). This is only valid for software volumes! */
double pa_sw_volume_to_dB(pa_volume_t v) PA_GCC_CONST;

/** Convert a linear factor to a volume.  0.0 and less is muted while
 * 1.0 is PA_VOLUME_NORM.  This is only valid for software volumes! */
pa_volume_t pa_sw_volume_from_linear(double v) PA_GCC_CONST;

/** Convert a volume to a linear factor. This is only valid for software volumes! */
double pa_sw_volume_to_linear(pa_volume_t v) PA_GCC_CONST;

#ifdef INFINITY
#define PA_DECIBEL_MININFTY ((double) -INFINITY)
#else
/** This floor value is used as minus infinity when using pa_sw_volume_to_dB() / pa_sw_volume_from_dB(). */
#define PA_DECIBEL_MININFTY ((double) -200.0)
#endif

/** Remap a volume from one channel mapping to a different channel mapping. \since 0.9.12 */
pa_cvolume *pa_cvolume_remap(pa_cvolume *v, const pa_channel_map *from, const pa_channel_map *to);

/** Return non-zero if the specified volume is compatible with the
 * specified sample spec. \since 0.9.13 */
int pa_cvolume_compatible(const pa_cvolume *v, const pa_sample_spec *ss) PA_GCC_PURE;

/** Return non-zero if the specified volume is compatible with the
 * specified sample spec. \since 0.9.15 */
int pa_cvolume_compatible_with_channel_map(const pa_cvolume *v, const pa_channel_map *cm) PA_GCC_PURE;

/** Calculate a 'balance' value for the specified volume with the
 * specified channel map. The return value will range from -1.0f
 * (left) to +1.0f (right). If no balance value is applicable to this
 * channel map the return value will always be 0.0f. See
 * pa_channel_map_can_balance(). \since 0.9.15 */
float pa_cvolume_get_balance(const pa_cvolume *v, const pa_channel_map *map) PA_GCC_PURE;

/** Adjust the 'balance' value for the specified volume with the
 * specified channel map. v will be modified in place and
 * returned. The balance is a value between -1.0f and +1.0f. This
 * operation might not be reversible! Also, after this call
 * pa_cvolume_get_balance() is not guaranteed to actually return the
 * requested balance value (e.g. when the input volume was zero anyway for
 * all channels). If no balance value is applicable to
 * this channel map the volume will not be modified. See
 * pa_channel_map_can_balance(). \since 0.9.15 */
pa_cvolume* pa_cvolume_set_balance(pa_cvolume *v, const pa_channel_map *map, float new_balance);

/** Calculate a 'fade' value (i.e.\ 'balance' between front and rear)
 * for the specified volume with the specified channel map. The return
 * value will range from -1.0f (rear) to +1.0f (left). If no fade
 * value is applicable to this channel map the return value will
 * always be 0.0f. See pa_channel_map_can_fade(). \since 0.9.15 */
float pa_cvolume_get_fade(const pa_cvolume *v, const pa_channel_map *map) PA_GCC_PURE;

/** Adjust the 'fade' value (i.e.\ 'balance' between front and rear)
 * for the specified volume with the specified channel map. v will be
 * modified in place and returned. The balance is a value between
 * -1.0f and +1.0f. This operation might not be reversible! Also,
 * after this call pa_cvolume_get_fade() is not guaranteed to actually
 * return the requested fade value (e.g. when the input volume was
 * zero anyway for all channels). If no fade value is applicable to
 * this channel map the volume will not be modified. See
 * pa_channel_map_can_fade(). \since 0.9.15 */
pa_cvolume* pa_cvolume_set_fade(pa_cvolume *v, const pa_channel_map *map, float new_fade);

/** Calculate a 'lfe balance' value for the specified volume with
 * the specified channel map. The return value will range from
 * -1.0f (no lfe) to +1.0f (only lfe), where 0.0f is balanced.
 * If no value is applicable to this channel map the return value
 * will always be 0.0f. See pa_channel_map_can_lfe_balance(). \since 8.0 */
float pa_cvolume_get_lfe_balance(const pa_cvolume *v, const pa_channel_map *map) PA_GCC_PURE;

/** Adjust the 'lfe balance' value for the specified volume with
 * the specified channel map. v will be modified in place and returned.
 * The balance is a value between -1.0f (no lfe) and +1.0f (only lfe).
 * This operation might not be reversible! Also, after this call
 * pa_cvolume_get_lfe_balance() is not guaranteed to actually
 * return the requested value (e.g. when the input volume was
 * zero anyway for all channels). If no lfe balance value is applicable to
 * this channel map the volume will not be modified. See
 * pa_channel_map_can_lfe_balance(). \since 8.0 */
pa_cvolume* pa_cvolume_set_lfe_balance(pa_cvolume *v, const pa_channel_map *map, float new_balance);

/** Scale the passed pa_cvolume structure so that the maximum volume
 * of all channels equals max. The proportions between the channel
 * volumes are kept. \since 0.9.15 */
pa_cvolume* pa_cvolume_scale(pa_cvolume *v, pa_volume_t max);

/** Scale the passed pa_cvolume structure so that the maximum volume
 * of all channels selected via cm/mask equals max. This also modifies
 * the volume of those channels that are unmasked. The proportions
 * between the channel volumes are kept. \since 0.9.16 */
pa_cvolume* pa_cvolume_scale_mask(pa_cvolume *v, pa_volume_t max, pa_channel_map *cm, pa_channel_position_mask_t mask);

/** Set the passed volume to all channels at the specified channel
 * position. Will return the updated volume struct, or NULL if there
 * is no channel at the position specified. You can check if a channel
 * map includes a specific position by calling
 * pa_channel_map_has_position(). \since 0.9.16 */
pa_cvolume* pa_cvolume_set_position(pa_cvolume *cv, const pa_channel_map *map, pa_channel_position_t t, pa_volume_t v);

/** Get the maximum volume of all channels at the specified channel
 * position. Will return 0 if there is no channel at the position
 * specified. You can check if a channel map includes a specific
 * position by calling pa_channel_map_has_position(). \since 0.9.16 */
pa_volume_t pa_cvolume_get_position(pa_cvolume *cv, const pa_channel_map *map, pa_channel_position_t t) PA_GCC_PURE;

/** This goes through all channels in a and b and sets the
 * corresponding channel in dest to the greater volume of both. a, b
 * and dest may point to the same structure. \since 0.9.16 */
pa_cvolume* pa_cvolume_merge(pa_cvolume *dest, const pa_cvolume *a, const pa_cvolume *b);

/** Increase the volume passed in by 'inc', but not exceeding 'limit'.
 * The proportions between the channels are kept. \since 0.9.19 */
pa_cvolume* pa_cvolume_inc_clamp(pa_cvolume *v, pa_volume_t inc, pa_volume_t limit);

/** Increase the volume passed in by 'inc'. The proportions between
 * the channels are kept. \since 0.9.16 */
pa_cvolume* pa_cvolume_inc(pa_cvolume *v, pa_volume_t inc);

/** Decrease the volume passed in by 'dec'. The proportions between
 * the channels are kept. \since 0.9.16 */
pa_cvolume* pa_cvolume_dec(pa_cvolume *v, pa_volume_t dec);

PA_C_DECL_END

#endif

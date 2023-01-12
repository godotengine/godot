#ifndef foochannelmaphfoo
#define foochannelmaphfoo

/***
  This file is part of PulseAudio.

  Copyright 2005-2006 Lennart Poettering
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

#include <pulse/sample.h>
#include <pulse/cdecl.h>
#include <pulse/gccmacro.h>
#include <pulse/version.h>

/** \page channelmap Channel Maps
 *
 * \section overv_sec Overview
 *
 * Channel maps provide a way to associate channels in a stream with a
 * specific speaker position. This relieves applications of having to
 * make sure their channel order is identical to the final output.
 *
 * \section init_sec Initialisation
 *
 * A channel map consists of an array of \ref pa_channel_position values,
 * one for each channel. This array is stored together with a channel count
 * in a pa_channel_map structure.
 *
 * Before filling the structure, the application must initialise it using
 * pa_channel_map_init(). There are also a number of convenience functions
 * for standard channel mappings:
 *
 * \li pa_channel_map_init_mono() - Create a channel map with only mono audio.
 * \li pa_channel_map_init_stereo() - Create a standard stereo mapping.
 * \li pa_channel_map_init_auto() - Create a standard channel map for a specific number of channels
 * \li pa_channel_map_init_extend() - Similar to
 * pa_channel_map_init_auto() but synthesize a channel map if no
 * predefined one is known for the specified number of channels.
 *
 * \section conv_sec Convenience Functions
 *
 * The library contains a number of convenience functions for dealing with
 * channel maps:
 *
 * \li pa_channel_map_valid() - Tests if a channel map is valid.
 * \li pa_channel_map_equal() - Tests if two channel maps are identical.
 * \li pa_channel_map_snprint() - Creates a textual description of a channel
 *                                map.
 */

/** \file
 * Constants and routines for channel mapping handling
 *
 * See also \subpage channelmap
 */

PA_C_DECL_BEGIN

/** A list of channel labels */
typedef enum pa_channel_position {
    PA_CHANNEL_POSITION_INVALID = -1,
    PA_CHANNEL_POSITION_MONO = 0,

    PA_CHANNEL_POSITION_FRONT_LEFT,               /**< Apple, Dolby call this 'Left' */
    PA_CHANNEL_POSITION_FRONT_RIGHT,              /**< Apple, Dolby call this 'Right' */
    PA_CHANNEL_POSITION_FRONT_CENTER,             /**< Apple, Dolby call this 'Center' */

/** \cond fulldocs */
    PA_CHANNEL_POSITION_LEFT = PA_CHANNEL_POSITION_FRONT_LEFT,
    PA_CHANNEL_POSITION_RIGHT = PA_CHANNEL_POSITION_FRONT_RIGHT,
    PA_CHANNEL_POSITION_CENTER = PA_CHANNEL_POSITION_FRONT_CENTER,
/** \endcond */

    PA_CHANNEL_POSITION_REAR_CENTER,              /**< Microsoft calls this 'Back Center', Apple calls this 'Center Surround', Dolby calls this 'Surround Rear Center' */
    PA_CHANNEL_POSITION_REAR_LEFT,                /**< Microsoft calls this 'Back Left', Apple calls this 'Left Surround' (!), Dolby calls this 'Surround Rear Left'  */
    PA_CHANNEL_POSITION_REAR_RIGHT,               /**< Microsoft calls this 'Back Right', Apple calls this 'Right Surround' (!), Dolby calls this 'Surround Rear Right'  */

    PA_CHANNEL_POSITION_LFE,                      /**< Microsoft calls this 'Low Frequency', Apple calls this 'LFEScreen' */
/** \cond fulldocs */
    PA_CHANNEL_POSITION_SUBWOOFER = PA_CHANNEL_POSITION_LFE,
/** \endcond */

    PA_CHANNEL_POSITION_FRONT_LEFT_OF_CENTER,     /**< Apple, Dolby call this 'Left Center' */
    PA_CHANNEL_POSITION_FRONT_RIGHT_OF_CENTER,    /**< Apple, Dolby call this 'Right Center */

    PA_CHANNEL_POSITION_SIDE_LEFT,                /**< Apple calls this 'Left Surround Direct', Dolby calls this 'Surround Left' (!) */
    PA_CHANNEL_POSITION_SIDE_RIGHT,               /**< Apple calls this 'Right Surround Direct', Dolby calls this 'Surround Right' (!) */

    PA_CHANNEL_POSITION_AUX0,
    PA_CHANNEL_POSITION_AUX1,
    PA_CHANNEL_POSITION_AUX2,
    PA_CHANNEL_POSITION_AUX3,
    PA_CHANNEL_POSITION_AUX4,
    PA_CHANNEL_POSITION_AUX5,
    PA_CHANNEL_POSITION_AUX6,
    PA_CHANNEL_POSITION_AUX7,
    PA_CHANNEL_POSITION_AUX8,
    PA_CHANNEL_POSITION_AUX9,
    PA_CHANNEL_POSITION_AUX10,
    PA_CHANNEL_POSITION_AUX11,
    PA_CHANNEL_POSITION_AUX12,
    PA_CHANNEL_POSITION_AUX13,
    PA_CHANNEL_POSITION_AUX14,
    PA_CHANNEL_POSITION_AUX15,
    PA_CHANNEL_POSITION_AUX16,
    PA_CHANNEL_POSITION_AUX17,
    PA_CHANNEL_POSITION_AUX18,
    PA_CHANNEL_POSITION_AUX19,
    PA_CHANNEL_POSITION_AUX20,
    PA_CHANNEL_POSITION_AUX21,
    PA_CHANNEL_POSITION_AUX22,
    PA_CHANNEL_POSITION_AUX23,
    PA_CHANNEL_POSITION_AUX24,
    PA_CHANNEL_POSITION_AUX25,
    PA_CHANNEL_POSITION_AUX26,
    PA_CHANNEL_POSITION_AUX27,
    PA_CHANNEL_POSITION_AUX28,
    PA_CHANNEL_POSITION_AUX29,
    PA_CHANNEL_POSITION_AUX30,
    PA_CHANNEL_POSITION_AUX31,

    PA_CHANNEL_POSITION_TOP_CENTER,               /**< Apple calls this 'Top Center Surround' */

    PA_CHANNEL_POSITION_TOP_FRONT_LEFT,           /**< Apple calls this 'Vertical Height Left' */
    PA_CHANNEL_POSITION_TOP_FRONT_RIGHT,          /**< Apple calls this 'Vertical Height Right' */
    PA_CHANNEL_POSITION_TOP_FRONT_CENTER,         /**< Apple calls this 'Vertical Height Center' */

    PA_CHANNEL_POSITION_TOP_REAR_LEFT,            /**< Microsoft and Apple call this 'Top Back Left' */
    PA_CHANNEL_POSITION_TOP_REAR_RIGHT,           /**< Microsoft and Apple call this 'Top Back Right' */
    PA_CHANNEL_POSITION_TOP_REAR_CENTER,          /**< Microsoft and Apple call this 'Top Back Center' */

    PA_CHANNEL_POSITION_MAX
} pa_channel_position_t;

/** \cond fulldocs */
#define PA_CHANNEL_POSITION_INVALID PA_CHANNEL_POSITION_INVALID
#define PA_CHANNEL_POSITION_MONO PA_CHANNEL_POSITION_MONO
#define PA_CHANNEL_POSITION_LEFT PA_CHANNEL_POSITION_LEFT
#define PA_CHANNEL_POSITION_RIGHT PA_CHANNEL_POSITION_RIGHT
#define PA_CHANNEL_POSITION_CENTER PA_CHANNEL_POSITION_CENTER
#define PA_CHANNEL_POSITION_FRONT_LEFT PA_CHANNEL_POSITION_FRONT_LEFT
#define PA_CHANNEL_POSITION_FRONT_RIGHT PA_CHANNEL_POSITION_FRONT_RIGHT
#define PA_CHANNEL_POSITION_FRONT_CENTER PA_CHANNEL_POSITION_FRONT_CENTER
#define PA_CHANNEL_POSITION_REAR_CENTER PA_CHANNEL_POSITION_REAR_CENTER
#define PA_CHANNEL_POSITION_REAR_LEFT PA_CHANNEL_POSITION_REAR_LEFT
#define PA_CHANNEL_POSITION_REAR_RIGHT PA_CHANNEL_POSITION_REAR_RIGHT
#define PA_CHANNEL_POSITION_LFE PA_CHANNEL_POSITION_LFE
#define PA_CHANNEL_POSITION_SUBWOOFER PA_CHANNEL_POSITION_SUBWOOFER
#define PA_CHANNEL_POSITION_FRONT_LEFT_OF_CENTER PA_CHANNEL_POSITION_FRONT_LEFT_OF_CENTER
#define PA_CHANNEL_POSITION_FRONT_RIGHT_OF_CENTER PA_CHANNEL_POSITION_FRONT_RIGHT_OF_CENTER
#define PA_CHANNEL_POSITION_SIDE_LEFT PA_CHANNEL_POSITION_SIDE_LEFT
#define PA_CHANNEL_POSITION_SIDE_RIGHT PA_CHANNEL_POSITION_SIDE_RIGHT
#define PA_CHANNEL_POSITION_AUX0 PA_CHANNEL_POSITION_AUX0
#define PA_CHANNEL_POSITION_AUX1 PA_CHANNEL_POSITION_AUX1
#define PA_CHANNEL_POSITION_AUX2 PA_CHANNEL_POSITION_AUX2
#define PA_CHANNEL_POSITION_AUX3 PA_CHANNEL_POSITION_AUX3
#define PA_CHANNEL_POSITION_AUX4 PA_CHANNEL_POSITION_AUX4
#define PA_CHANNEL_POSITION_AUX5 PA_CHANNEL_POSITION_AUX5
#define PA_CHANNEL_POSITION_AUX6 PA_CHANNEL_POSITION_AUX6
#define PA_CHANNEL_POSITION_AUX7 PA_CHANNEL_POSITION_AUX7
#define PA_CHANNEL_POSITION_AUX8 PA_CHANNEL_POSITION_AUX8
#define PA_CHANNEL_POSITION_AUX9 PA_CHANNEL_POSITION_AUX9
#define PA_CHANNEL_POSITION_AUX10 PA_CHANNEL_POSITION_AUX10
#define PA_CHANNEL_POSITION_AUX11 PA_CHANNEL_POSITION_AUX11
#define PA_CHANNEL_POSITION_AUX12 PA_CHANNEL_POSITION_AUX12
#define PA_CHANNEL_POSITION_AUX13 PA_CHANNEL_POSITION_AUX13
#define PA_CHANNEL_POSITION_AUX14 PA_CHANNEL_POSITION_AUX14
#define PA_CHANNEL_POSITION_AUX15 PA_CHANNEL_POSITION_AUX15
#define PA_CHANNEL_POSITION_AUX16 PA_CHANNEL_POSITION_AUX16
#define PA_CHANNEL_POSITION_AUX17 PA_CHANNEL_POSITION_AUX17
#define PA_CHANNEL_POSITION_AUX18 PA_CHANNEL_POSITION_AUX18
#define PA_CHANNEL_POSITION_AUX19 PA_CHANNEL_POSITION_AUX19
#define PA_CHANNEL_POSITION_AUX20 PA_CHANNEL_POSITION_AUX20
#define PA_CHANNEL_POSITION_AUX21 PA_CHANNEL_POSITION_AUX21
#define PA_CHANNEL_POSITION_AUX22 PA_CHANNEL_POSITION_AUX22
#define PA_CHANNEL_POSITION_AUX23 PA_CHANNEL_POSITION_AUX23
#define PA_CHANNEL_POSITION_AUX24 PA_CHANNEL_POSITION_AUX24
#define PA_CHANNEL_POSITION_AUX25 PA_CHANNEL_POSITION_AUX25
#define PA_CHANNEL_POSITION_AUX26 PA_CHANNEL_POSITION_AUX26
#define PA_CHANNEL_POSITION_AUX27 PA_CHANNEL_POSITION_AUX27
#define PA_CHANNEL_POSITION_AUX28 PA_CHANNEL_POSITION_AUX28
#define PA_CHANNEL_POSITION_AUX29 PA_CHANNEL_POSITION_AUX29
#define PA_CHANNEL_POSITION_AUX30 PA_CHANNEL_POSITION_AUX30
#define PA_CHANNEL_POSITION_AUX31 PA_CHANNEL_POSITION_AUX31
#define PA_CHANNEL_POSITION_TOP_CENTER PA_CHANNEL_POSITION_TOP_CENTER
#define PA_CHANNEL_POSITION_TOP_FRONT_LEFT PA_CHANNEL_POSITION_TOP_FRONT_LEFT
#define PA_CHANNEL_POSITION_TOP_FRONT_RIGHT PA_CHANNEL_POSITION_TOP_FRONT_RIGHT
#define PA_CHANNEL_POSITION_TOP_FRONT_CENTER PA_CHANNEL_POSITION_TOP_FRONT_CENTER
#define PA_CHANNEL_POSITION_TOP_REAR_LEFT PA_CHANNEL_POSITION_TOP_REAR_LEFT
#define PA_CHANNEL_POSITION_TOP_REAR_RIGHT PA_CHANNEL_POSITION_TOP_REAR_RIGHT
#define PA_CHANNEL_POSITION_TOP_REAR_CENTER PA_CHANNEL_POSITION_TOP_REAR_CENTER
#define PA_CHANNEL_POSITION_MAX PA_CHANNEL_POSITION_MAX
/** \endcond */

/** A mask of channel positions. \since 0.9.16 */
typedef uint64_t pa_channel_position_mask_t;

/** Makes a bit mask from a channel position. \since 0.9.16 */
#define PA_CHANNEL_POSITION_MASK(f) ((pa_channel_position_mask_t) (1ULL << (f)))

/** A list of channel mapping definitions for pa_channel_map_init_auto() */
typedef enum pa_channel_map_def {
    PA_CHANNEL_MAP_AIFF,
    /**< The mapping from RFC3551, which is based on AIFF-C */

/** \cond fulldocs */
    PA_CHANNEL_MAP_ALSA,
    /**< The default mapping used by ALSA. This mapping is probably
     * not too useful since ALSA's default channel mapping depends on
     * the device string used. */
/** \endcond */

    PA_CHANNEL_MAP_AUX,
    /**< Only aux channels */

    PA_CHANNEL_MAP_WAVEEX,
    /**< Microsoft's WAVEFORMATEXTENSIBLE mapping. This mapping works
     * as if all LSBs of dwChannelMask are set.  */

/** \cond fulldocs */
    PA_CHANNEL_MAP_OSS,
    /**< The default channel mapping used by OSS as defined in the OSS
     * 4.0 API specs. This mapping is probably not too useful since
     * the OSS API has changed in this respect and no longer knows a
     * default channel mapping based on the number of channels. */
/** \endcond */

    /**< Upper limit of valid channel mapping definitions */
    PA_CHANNEL_MAP_DEF_MAX,

    PA_CHANNEL_MAP_DEFAULT = PA_CHANNEL_MAP_AIFF
    /**< The default channel map */
} pa_channel_map_def_t;

/** \cond fulldocs */
#define PA_CHANNEL_MAP_AIFF PA_CHANNEL_MAP_AIFF
#define PA_CHANNEL_MAP_ALSA PA_CHANNEL_MAP_ALSA
#define PA_CHANNEL_MAP_AUX PA_CHANNEL_MAP_AUX
#define PA_CHANNEL_MAP_WAVEEX PA_CHANNEL_MAP_WAVEEX
#define PA_CHANNEL_MAP_OSS PA_CHANNEL_MAP_OSS
#define PA_CHANNEL_MAP_DEF_MAX PA_CHANNEL_MAP_DEF_MAX
#define PA_CHANNEL_MAP_DEFAULT PA_CHANNEL_MAP_DEFAULT
/** \endcond */

/** A channel map which can be used to attach labels to specific
 * channels of a stream. These values are relevant for conversion and
 * mixing of streams */
typedef struct pa_channel_map {
    uint8_t channels;
    /**< Number of channels */

    pa_channel_position_t map[PA_CHANNELS_MAX];
    /**< Channel labels */
} pa_channel_map;

/** Initialize the specified channel map and return a pointer to
 * it. The channel map will have a defined state but
 * pa_channel_map_valid() will fail for it. */
pa_channel_map* pa_channel_map_init(pa_channel_map *m);

/** Initialize the specified channel map for monaural audio and return a pointer to it */
pa_channel_map* pa_channel_map_init_mono(pa_channel_map *m);

/** Initialize the specified channel map for stereophonic audio and return a pointer to it */
pa_channel_map* pa_channel_map_init_stereo(pa_channel_map *m);

/** Initialize the specified channel map for the specified number of
 * channels using default labels and return a pointer to it. This call
 * will fail (return NULL) if there is no default channel map known for this
 * specific number of channels and mapping. */
pa_channel_map* pa_channel_map_init_auto(pa_channel_map *m, unsigned channels, pa_channel_map_def_t def);

/** Similar to pa_channel_map_init_auto() but instead of failing if no
 * default mapping is known with the specified parameters it will
 * synthesize a mapping based on a known mapping with fewer channels
 * and fill up the rest with AUX0...AUX31 channels  \since 0.9.11 */
pa_channel_map* pa_channel_map_init_extend(pa_channel_map *m, unsigned channels, pa_channel_map_def_t def);

/** Return a text label for the specified channel position */
const char* pa_channel_position_to_string(pa_channel_position_t pos) PA_GCC_PURE;

/** The inverse of pa_channel_position_to_string(). \since 0.9.16 */
pa_channel_position_t pa_channel_position_from_string(const char *s) PA_GCC_PURE;

/** Return a human readable text label for the specified channel position. \since 0.9.7 */
const char* pa_channel_position_to_pretty_string(pa_channel_position_t pos);

/** The maximum length of strings returned by
 * pa_channel_map_snprint(). Please note that this value can change
 * with any release without warning and without being considered API
 * or ABI breakage. You should not use this definition anywhere where
 * it might become part of an ABI. */
#define PA_CHANNEL_MAP_SNPRINT_MAX 336

/** Make a human readable string from the specified channel map */
char* pa_channel_map_snprint(char *s, size_t l, const pa_channel_map *map);

/** Parse a channel position list or well-known mapping name into a
 * channel map structure. This turns the output of
 * pa_channel_map_snprint() and pa_channel_map_to_name() back into a
 * pa_channel_map */
pa_channel_map *pa_channel_map_parse(pa_channel_map *map, const char *s);

/** Compare two channel maps. Return 1 if both match. */
int pa_channel_map_equal(const pa_channel_map *a, const pa_channel_map *b) PA_GCC_PURE;

/** Return non-zero if the specified channel map is considered valid */
int pa_channel_map_valid(const pa_channel_map *map) PA_GCC_PURE;

/** Return non-zero if the specified channel map is compatible with
 * the specified sample spec. \since 0.9.12 */
int pa_channel_map_compatible(const pa_channel_map *map, const pa_sample_spec *ss) PA_GCC_PURE;

/** Returns non-zero if every channel defined in b is also defined in a. \since 0.9.15 */
int pa_channel_map_superset(const pa_channel_map *a, const pa_channel_map *b) PA_GCC_PURE;

/** Returns non-zero if it makes sense to apply a volume 'balance'
 * with this mapping, i.e.\ if there are left/right channels
 * available. \since 0.9.15 */
int pa_channel_map_can_balance(const pa_channel_map *map) PA_GCC_PURE;

/** Returns non-zero if it makes sense to apply a volume 'fade'
 * (i.e.\ 'balance' between front and rear) with this mapping, i.e.\ if
 * there are front/rear channels available. \since 0.9.15 */
int pa_channel_map_can_fade(const pa_channel_map *map) PA_GCC_PURE;

/** Returns non-zero if it makes sense to apply a volume 'lfe balance'
 * (i.e.\ 'balance' between LFE and non-LFE channels) with this mapping,
 *  i.e.\ if there are LFE and non-LFE channels available. \since 8.0 */
int pa_channel_map_can_lfe_balance(const pa_channel_map *map) PA_GCC_PURE;

/** Tries to find a well-known channel mapping name for this channel
 * mapping, i.e.\ "stereo", "surround-71" and so on. If the channel
 * mapping is unknown NULL will be returned. This name can be parsed
 * with pa_channel_map_parse() \since 0.9.15 */
const char* pa_channel_map_to_name(const pa_channel_map *map) PA_GCC_PURE;

/** Tries to find a human readable text label for this channel
mapping, i.e.\ "Stereo", "Surround 7.1" and so on. If the channel
mapping is unknown NULL will be returned. \since 0.9.15 */
const char* pa_channel_map_to_pretty_name(const pa_channel_map *map) PA_GCC_PURE;

/** Returns non-zero if the specified channel position is available at
 * least once in the channel map. \since 0.9.16 */
int pa_channel_map_has_position(const pa_channel_map *map, pa_channel_position_t p) PA_GCC_PURE;

/** Generates a bit mask from a channel map. \since 0.9.16 */
pa_channel_position_mask_t pa_channel_map_mask(const pa_channel_map *map) PA_GCC_PURE;

PA_C_DECL_END

#endif

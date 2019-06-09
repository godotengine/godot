// jar_xm.h - v0.01 - public domain - Joshua Reisenauer, MAR 2016
//
// HISTORY:
//
//   v0.01  2016-02-22  Setup
//
//
// USAGE:
//
// In ONE source file, put:
//
//    #define JAR_XM_IMPLEMENTATION
//    #include "jar_xm.h"
//
// Other source files should just include jar_xm.h
//
// SAMPLE CODE:
//
// jar_xm_context_t *musicptr;
// float musicBuffer[48000 / 60];
// int intro_load(void)
// {
//     jar_xm_create_context_from_file(&musicptr, 48000, "Song.XM");
//     return 1;
// }
// int intro_unload(void)
// {
//     jar_xm_free_context(musicptr);
//     return 1;
// }
// int intro_tick(long counter)
// {
//     jar_xm_generate_samples(musicptr, musicBuffer, (48000 / 60) / 2);
//     if(IsKeyDown(KEY_ENTER))
//         return 1;
//     return 0;
// }
//
//
// LISCENSE - FOR LIBXM:
//
// Author: Romain "Artefact2" Dalmaso <artefact2@gmail.com>
// Contributor: Dan Spencer <dan@atomicpotato.net>
// Repackaged into jar_xm.h By: Joshua Adam Reisenauer <kd7tck@gmail.com>
// This program is free software. It comes without any warranty, to the
// extent permitted by applicable law. You can redistribute it and/or
// modify it under the terms of the Do What The Fuck You Want To Public
// License, Version 2, as published by Sam Hocevar. See
// http://sam.zoy.org/wtfpl/COPYING for more details.

#ifndef INCLUDE_JAR_XM_H
#define INCLUDE_JAR_XM_H

#define JAR_XM_DEBUG 0
#define JAR_XM_LINEAR_INTERPOLATION 1 // speed increase with decrease in quality
#define JAR_XM_DEFENSIVE 1
#define JAR_XM_RAMPING 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>



//-------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

struct jar_xm_context_s;
typedef struct jar_xm_context_s jar_xm_context_t;

/** Create a XM context.
 *
 * @param moddata the contents of the module
 * @param rate play rate in Hz, recommended value of 48000
 *
 * @returns 0 on success
 * @returns 1 if module data is not sane
 * @returns 2 if memory allocation failed
 * @returns 3 unable to open input file
 * @returns 4 fseek() failed
 * @returns 5 fread() failed
 * @returns 6 unkown error
 *
 * @deprecated This function is unsafe!
 * @see jar_xm_create_context_safe()
 */
int jar_xm_create_context_from_file(jar_xm_context_t** ctx, uint32_t rate, const char* filename);

/** Create a XM context.
 *
 * @param moddata the contents of the module
 * @param rate play rate in Hz, recommended value of 48000
 *
 * @returns 0 on success
 * @returns 1 if module data is not sane
 * @returns 2 if memory allocation failed
 *
 * @deprecated This function is unsafe!
 * @see jar_xm_create_context_safe()
 */
int jar_xm_create_context(jar_xm_context_t** ctx, const char* moddata, uint32_t rate);

/** Create a XM context.
 *
 * @param moddata the contents of the module
 * @param moddata_length the length of the contents of the module, in bytes
 * @param rate play rate in Hz, recommended value of 48000
 *
 * @returns 0 on success
 * @returns 1 if module data is not sane
 * @returns 2 if memory allocation failed
 */
int jar_xm_create_context_safe(jar_xm_context_t** ctx, const char* moddata, size_t moddata_length, uint32_t rate);

/** Free a XM context created by jar_xm_create_context(). */
void jar_xm_free_context(jar_xm_context_t* ctx);

/** Play the module and put the sound samples in an output buffer.
 *
 * @param output buffer of 2*numsamples elements (A left and right value for each sample)
 * @param numsamples number of samples to generate
 */
void jar_xm_generate_samples(jar_xm_context_t* ctx, float* output, size_t numsamples);

/** Play the module, resample from 32 bit to 16 bit, and put the sound samples in an output buffer.
 *
 * @param output buffer of 2*numsamples elements (A left and right value for each sample)
 * @param numsamples number of samples to generate
 */
void jar_xm_generate_samples_16bit(jar_xm_context_t* ctx, short* output, size_t numsamples);


/** Play the module, resample from 32 bit to 8 bit, and put the sound samples in an output buffer.
 *
 * @param output buffer of 2*numsamples elements (A left and right value for each sample)
 * @param numsamples number of samples to generate
 */
void jar_xm_generate_samples_8bit(jar_xm_context_t* ctx, char* output, size_t numsamples);



/** Set the maximum number of times a module can loop. After the
 * specified number of loops, calls to jar_xm_generate_samples will only
 * generate silence. You can control the current number of loops with
 * jar_xm_get_loop_count().
 *
 * @param loopcnt maximum number of loops. Use 0 to loop
 * indefinitely. */
void jar_xm_set_max_loop_count(jar_xm_context_t* ctx, uint8_t loopcnt);

/** Get the loop count of the currently playing module. This value is
 * 0 when the module is still playing, 1 when the module has looped
 * once, etc. */
uint8_t jar_xm_get_loop_count(jar_xm_context_t* ctx);

/** Seek to a specific position in a module.
 *
 * WARNING, WITH BIG LETTERS: seeking modules is broken by design,
 * don't expect miracles.
 */
void jar_xm_seek(jar_xm_context_t*, uint8_t pot, uint8_t row, uint16_t tick);

/** Mute or unmute a channel.
 *
 * @note Channel numbers go from 1 to jar_xm_get_number_of_channels(...).
 *
 * @return whether the channel was muted.
 */
bool jar_xm_mute_channel(jar_xm_context_t* ctx, uint16_t, bool);

/** Mute or unmute an instrument.
 *
 * @note Instrument numbers go from 1 to
 * jar_xm_get_number_of_instruments(...).
 *
 * @return whether the instrument was muted.
 */
bool jar_xm_mute_instrument(jar_xm_context_t* ctx, uint16_t, bool);



/** Get the module name as a NUL-terminated string. */
const char* jar_xm_get_module_name(jar_xm_context_t* ctx);

/** Get the tracker name as a NUL-terminated string. */
const char* jar_xm_get_tracker_name(jar_xm_context_t* ctx);



/** Get the number of channels. */
uint16_t jar_xm_get_number_of_channels(jar_xm_context_t* ctx);

/** Get the module length (in patterns). */
uint16_t jar_xm_get_module_length(jar_xm_context_t*);

/** Get the number of patterns. */
uint16_t jar_xm_get_number_of_patterns(jar_xm_context_t* ctx);

/** Get the number of rows of a pattern.
 *
 * @note Pattern numbers go from 0 to
 * jar_xm_get_number_of_patterns(...)-1.
 */
uint16_t jar_xm_get_number_of_rows(jar_xm_context_t* ctx, uint16_t);

/** Get the number of instruments. */
uint16_t jar_xm_get_number_of_instruments(jar_xm_context_t* ctx);

/** Get the number of samples of an instrument.
 *
 * @note Instrument numbers go from 1 to
 * jar_xm_get_number_of_instruments(...).
 */
uint16_t jar_xm_get_number_of_samples(jar_xm_context_t* ctx, uint16_t);



/** Get the current module speed.
 *
 * @param bpm will receive the current BPM
 * @param tempo will receive the current tempo (ticks per line)
 */
void jar_xm_get_playing_speed(jar_xm_context_t* ctx, uint16_t* bpm, uint16_t* tempo);

/** Get the current position in the module being played.
 *
 * @param pattern_index if not NULL, will receive the current pattern
 * index in the POT (pattern order table)
 *
 * @param pattern if not NULL, will receive the current pattern number
 *
 * @param row if not NULL, will receive the current row
 *
 * @param samples if not NULL, will receive the total number of
 * generated samples (divide by sample rate to get seconds of
 * generated audio)
 */
void jar_xm_get_position(jar_xm_context_t* ctx, uint8_t* pattern_index, uint8_t* pattern, uint8_t* row, uint64_t* samples);

/** Get the latest time (in number of generated samples) when a
 * particular instrument was triggered in any channel.
 *
 * @note Instrument numbers go from 1 to
 * jar_xm_get_number_of_instruments(...).
 */
uint64_t jar_xm_get_latest_trigger_of_instrument(jar_xm_context_t* ctx, uint16_t);

/** Get the latest time (in number of generated samples) when a
 * particular sample was triggered in any channel.
 *
 * @note Instrument numbers go from 1 to
 * jar_xm_get_number_of_instruments(...).
 *
 * @note Sample numbers go from 0 to
 * jar_xm_get_nubmer_of_samples(...,instr)-1.
 */
uint64_t jar_xm_get_latest_trigger_of_sample(jar_xm_context_t* ctx, uint16_t instr, uint16_t sample);

/** Get the latest time (in number of generated samples) when any
 * instrument was triggered in a given channel.
 *
 * @note Channel numbers go from 1 to jar_xm_get_number_of_channels(...).
 */
uint64_t jar_xm_get_latest_trigger_of_channel(jar_xm_context_t* ctx, uint16_t);

/** Get the number of remaining samples. Divide by 2 to get the number of individual LR data samples.
 *
 * @note This is the remaining number of samples before the loop starts module again, or halts if on last pass.
 * @note This function is very slow and should only be run once, if at all.
 */
uint64_t jar_xm_get_remaining_samples(jar_xm_context_t* ctx);

// not part of the original library
void jar_xm_set_playback_rate(jar_xm_context_t* ctx, float rate);
jar_xm_context_t* jar_xm_context_copy(jar_xm_context_t *ctx);
void jar_xm_reset(jar_xm_context_t *ctx);

#ifdef __cplusplus
}
#endif
//-------------------------------------------------------------------------------


#endif//end of INCLUDE_JAR_XM_H

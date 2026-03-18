/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/**
 * # CategoryAudio
 *
 * Audio functionality for the SDL library.
 *
 * All audio in SDL3 revolves around SDL_AudioStream. Whether you want to play
 * or record audio, convert it, stream it, buffer it, or mix it, you're going
 * to be passing it through an audio stream.
 *
 * Audio streams are quite flexible; they can accept any amount of data at a
 * time, in any supported format, and output it as needed in any other format,
 * even if the data format changes on either side halfway through.
 *
 * An app opens an audio device and binds any number of audio streams to it,
 * feeding more data to the streams as available. When the device needs more
 * data, it will pull it from all bound streams and mix them together for
 * playback.
 *
 * Audio streams can also use an app-provided callback to supply data
 * on-demand, which maps pretty closely to the SDL2 audio model.
 *
 * SDL also provides a simple .WAV loader in SDL_LoadWAV (and SDL_LoadWAV_IO
 * if you aren't reading from a file) as a basic means to load sound data into
 * your program.
 *
 * ## Logical audio devices
 *
 * In SDL3, opening a physical device (like a SoundBlaster 16 Pro) gives you a
 * logical device ID that you can bind audio streams to. In almost all cases,
 * logical devices can be used anywhere in the API that a physical device is
 * normally used. However, since each device opening generates a new logical
 * device, different parts of the program (say, a VoIP library, or
 * text-to-speech framework, or maybe some other sort of mixer on top of SDL)
 * can have their own device opens that do not interfere with each other; each
 * logical device will mix its separate audio down to a single buffer, fed to
 * the physical device, behind the scenes. As many logical devices as you like
 * can come and go; SDL will only have to open the physical device at the OS
 * level once, and will manage all the logical devices on top of it
 * internally.
 *
 * One other benefit of logical devices: if you don't open a specific physical
 * device, instead opting for the default, SDL can automatically migrate those
 * logical devices to different hardware as circumstances change: a user
 * plugged in headphones? The system default changed? SDL can transparently
 * migrate the logical devices to the correct physical device seamlessly and
 * keep playing; the app doesn't even have to know it happened if it doesn't
 * want to.
 *
 * ## Simplified audio
 *
 * As a simplified model for when a single source of audio is all that's
 * needed, an app can use SDL_OpenAudioDeviceStream, which is a single
 * function to open an audio device, create an audio stream, bind that stream
 * to the newly-opened device, and (optionally) provide a callback for
 * obtaining audio data. When using this function, the primary interface is
 * the SDL_AudioStream and the device handle is mostly hidden away; destroying
 * a stream created through this function will also close the device, stream
 * bindings cannot be changed, etc. One other quirk of this is that the device
 * is started in a _paused_ state and must be explicitly resumed; this is
 * partially to offer a clean migration for SDL2 apps and partially because
 * the app might have to do more setup before playback begins; in the
 * non-simplified form, nothing will play until a stream is bound to a device,
 * so they start _unpaused_.
 *
 * ## Channel layouts
 *
 * Audio data passing through SDL is uncompressed PCM data, interleaved. One
 * can provide their own decompression through an MP3, etc, decoder, but SDL
 * does not provide this directly. Each interleaved channel of data is meant
 * to be in a specific order.
 *
 * Abbreviations:
 *
 * - FRONT = single mono speaker
 * - FL = front left speaker
 * - FR = front right speaker
 * - FC = front center speaker
 * - BL = back left speaker
 * - BR = back right speaker
 * - SR = surround right speaker
 * - SL = surround left speaker
 * - BC = back center speaker
 * - LFE = low-frequency speaker
 *
 * These are listed in the order they are laid out in memory, so "FL, FR"
 * means "the front left speaker is laid out in memory first, then the front
 * right, then it repeats for the next audio frame".
 *
 * - 1 channel (mono) layout: FRONT
 * - 2 channels (stereo) layout: FL, FR
 * - 3 channels (2.1) layout: FL, FR, LFE
 * - 4 channels (quad) layout: FL, FR, BL, BR
 * - 5 channels (4.1) layout: FL, FR, LFE, BL, BR
 * - 6 channels (5.1) layout: FL, FR, FC, LFE, BL, BR (last two can also be
 *   SL, SR)
 * - 7 channels (6.1) layout: FL, FR, FC, LFE, BC, SL, SR
 * - 8 channels (7.1) layout: FL, FR, FC, LFE, BL, BR, SL, SR
 *
 * This is the same order as DirectSound expects, but applied to all
 * platforms; SDL will swizzle the channels as necessary if a platform expects
 * something different.
 *
 * SDL_AudioStream can also be provided channel maps to change this ordering
 * to whatever is necessary, in other audio processing scenarios.
 */

#ifndef SDL_audio_h_
#define SDL_audio_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_endian.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_mutex.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_iostream.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Mask of bits in an SDL_AudioFormat that contains the format bit size.
 *
 * Generally one should use SDL_AUDIO_BITSIZE instead of this macro directly.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_MASK_BITSIZE       (0xFFu)

/**
 * Mask of bits in an SDL_AudioFormat that contain the floating point flag.
 *
 * Generally one should use SDL_AUDIO_ISFLOAT instead of this macro directly.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_MASK_FLOAT         (1u<<8)

/**
 * Mask of bits in an SDL_AudioFormat that contain the bigendian flag.
 *
 * Generally one should use SDL_AUDIO_ISBIGENDIAN or SDL_AUDIO_ISLITTLEENDIAN
 * instead of this macro directly.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_MASK_BIG_ENDIAN    (1u<<12)

/**
 * Mask of bits in an SDL_AudioFormat that contain the signed data flag.
 *
 * Generally one should use SDL_AUDIO_ISSIGNED instead of this macro directly.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_MASK_SIGNED        (1u<<15)

/**
 * Define an SDL_AudioFormat value.
 *
 * SDL does not support custom audio formats, so this macro is not of much use
 * externally, but it can be illustrative as to what the various bits of an
 * SDL_AudioFormat mean.
 *
 * For example, SDL_AUDIO_S32LE looks like this:
 *
 * ```c
 * SDL_DEFINE_AUDIO_FORMAT(1, 0, 0, 32)
 * ```
 *
 * \param signed 1 for signed data, 0 for unsigned data.
 * \param bigendian 1 for bigendian data, 0 for littleendian data.
 * \param flt 1 for floating point data, 0 for integer data.
 * \param size number of bits per sample.
 * \returns a format value in the style of SDL_AudioFormat.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_DEFINE_AUDIO_FORMAT(signed, bigendian, flt, size) \
    (((Uint16)(signed) << 15) | ((Uint16)(bigendian) << 12) | ((Uint16)(flt) << 8) | ((size) & SDL_AUDIO_MASK_BITSIZE))

/**
 * Audio format.
 *
 * \since This enum is available since SDL 3.2.0.
 *
 * \sa SDL_AUDIO_BITSIZE
 * \sa SDL_AUDIO_BYTESIZE
 * \sa SDL_AUDIO_ISINT
 * \sa SDL_AUDIO_ISFLOAT
 * \sa SDL_AUDIO_ISBIGENDIAN
 * \sa SDL_AUDIO_ISLITTLEENDIAN
 * \sa SDL_AUDIO_ISSIGNED
 * \sa SDL_AUDIO_ISUNSIGNED
 */
typedef enum SDL_AudioFormat
{
    SDL_AUDIO_UNKNOWN   = 0x0000u,  /**< Unspecified audio format */
    SDL_AUDIO_U8        = 0x0008u,  /**< Unsigned 8-bit samples */
        /* SDL_DEFINE_AUDIO_FORMAT(0, 0, 0, 8), */
    SDL_AUDIO_S8        = 0x8008u,  /**< Signed 8-bit samples */
        /* SDL_DEFINE_AUDIO_FORMAT(1, 0, 0, 8), */
    SDL_AUDIO_S16LE     = 0x8010u,  /**< Signed 16-bit samples */
        /* SDL_DEFINE_AUDIO_FORMAT(1, 0, 0, 16), */
    SDL_AUDIO_S16BE     = 0x9010u,  /**< As above, but big-endian byte order */
        /* SDL_DEFINE_AUDIO_FORMAT(1, 1, 0, 16), */
    SDL_AUDIO_S32LE     = 0x8020u,  /**< 32-bit integer samples */
        /* SDL_DEFINE_AUDIO_FORMAT(1, 0, 0, 32), */
    SDL_AUDIO_S32BE     = 0x9020u,  /**< As above, but big-endian byte order */
        /* SDL_DEFINE_AUDIO_FORMAT(1, 1, 0, 32), */
    SDL_AUDIO_F32LE     = 0x8120u,  /**< 32-bit floating point samples */
        /* SDL_DEFINE_AUDIO_FORMAT(1, 0, 1, 32), */
    SDL_AUDIO_F32BE     = 0x9120u,  /**< As above, but big-endian byte order */
        /* SDL_DEFINE_AUDIO_FORMAT(1, 1, 1, 32), */

    /* These represent the current system's byteorder. */
    #if SDL_BYTEORDER == SDL_LIL_ENDIAN
    SDL_AUDIO_S16 = SDL_AUDIO_S16LE,
    SDL_AUDIO_S32 = SDL_AUDIO_S32LE,
    SDL_AUDIO_F32 = SDL_AUDIO_F32LE
    #else
    SDL_AUDIO_S16 = SDL_AUDIO_S16BE,
    SDL_AUDIO_S32 = SDL_AUDIO_S32BE,
    SDL_AUDIO_F32 = SDL_AUDIO_F32BE
    #endif
} SDL_AudioFormat;


/**
 * Retrieve the size, in bits, from an SDL_AudioFormat.
 *
 * For example, `SDL_AUDIO_BITSIZE(SDL_AUDIO_S16)` returns 16.
 *
 * \param x an SDL_AudioFormat value.
 * \returns data size in bits.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_BITSIZE(x)         ((x) & SDL_AUDIO_MASK_BITSIZE)

/**
 * Retrieve the size, in bytes, from an SDL_AudioFormat.
 *
 * For example, `SDL_AUDIO_BYTESIZE(SDL_AUDIO_S16)` returns 2.
 *
 * \param x an SDL_AudioFormat value.
 * \returns data size in bytes.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_BYTESIZE(x)        (SDL_AUDIO_BITSIZE(x) / 8)

/**
 * Determine if an SDL_AudioFormat represents floating point data.
 *
 * For example, `SDL_AUDIO_ISFLOAT(SDL_AUDIO_S16)` returns 0.
 *
 * \param x an SDL_AudioFormat value.
 * \returns non-zero if format is floating point, zero otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_ISFLOAT(x)         ((x) & SDL_AUDIO_MASK_FLOAT)

/**
 * Determine if an SDL_AudioFormat represents bigendian data.
 *
 * For example, `SDL_AUDIO_ISBIGENDIAN(SDL_AUDIO_S16LE)` returns 0.
 *
 * \param x an SDL_AudioFormat value.
 * \returns non-zero if format is bigendian, zero otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_ISBIGENDIAN(x)     ((x) & SDL_AUDIO_MASK_BIG_ENDIAN)

/**
 * Determine if an SDL_AudioFormat represents littleendian data.
 *
 * For example, `SDL_AUDIO_ISLITTLEENDIAN(SDL_AUDIO_S16BE)` returns 0.
 *
 * \param x an SDL_AudioFormat value.
 * \returns non-zero if format is littleendian, zero otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_ISLITTLEENDIAN(x)  (!SDL_AUDIO_ISBIGENDIAN(x))

/**
 * Determine if an SDL_AudioFormat represents signed data.
 *
 * For example, `SDL_AUDIO_ISSIGNED(SDL_AUDIO_U8)` returns 0.
 *
 * \param x an SDL_AudioFormat value.
 * \returns non-zero if format is signed, zero otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_ISSIGNED(x)        ((x) & SDL_AUDIO_MASK_SIGNED)

/**
 * Determine if an SDL_AudioFormat represents integer data.
 *
 * For example, `SDL_AUDIO_ISINT(SDL_AUDIO_F32)` returns 0.
 *
 * \param x an SDL_AudioFormat value.
 * \returns non-zero if format is integer, zero otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_ISINT(x)           (!SDL_AUDIO_ISFLOAT(x))

/**
 * Determine if an SDL_AudioFormat represents unsigned data.
 *
 * For example, `SDL_AUDIO_ISUNSIGNED(SDL_AUDIO_S16)` returns 0.
 *
 * \param x an SDL_AudioFormat value.
 * \returns non-zero if format is unsigned, zero otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_ISUNSIGNED(x)      (!SDL_AUDIO_ISSIGNED(x))


/**
 * SDL Audio Device instance IDs.
 *
 * Zero is used to signify an invalid/null device.
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef Uint32 SDL_AudioDeviceID;

/**
 * A value used to request a default playback audio device.
 *
 * Several functions that require an SDL_AudioDeviceID will accept this value
 * to signify the app just wants the system to choose a default device instead
 * of the app providing a specific one.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK ((SDL_AudioDeviceID) 0xFFFFFFFFu)

/**
 * A value used to request a default recording audio device.
 *
 * Several functions that require an SDL_AudioDeviceID will accept this value
 * to signify the app just wants the system to choose a default device instead
 * of the app providing a specific one.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_DEVICE_DEFAULT_RECORDING ((SDL_AudioDeviceID) 0xFFFFFFFEu)

/**
 * Format specifier for audio data.
 *
 * \since This struct is available since SDL 3.2.0.
 *
 * \sa SDL_AudioFormat
 */
typedef struct SDL_AudioSpec
{
    SDL_AudioFormat format;     /**< Audio data format */
    int channels;               /**< Number of channels: 1 mono, 2 stereo, etc */
    int freq;                   /**< sample rate: sample frames per second */
} SDL_AudioSpec;

/**
 * Calculate the size of each audio frame (in bytes) from an SDL_AudioSpec.
 *
 * This reports on the size of an audio sample frame: stereo Sint16 data (2
 * channels of 2 bytes each) would be 4 bytes per frame, for example.
 *
 * \param x an SDL_AudioSpec to query.
 * \returns the number of bytes used per sample frame.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_AUDIO_FRAMESIZE(x) (SDL_AUDIO_BYTESIZE((x).format) * (x).channels)

/**
 * The opaque handle that represents an audio stream.
 *
 * SDL_AudioStream is an audio conversion interface.
 *
 * - It can handle resampling data in chunks without generating artifacts,
 *   when it doesn't have the complete buffer available.
 * - It can handle incoming data in any variable size.
 * - It can handle input/output format changes on the fly.
 * - It can remap audio channels between inputs and outputs.
 * - You push data as you have it, and pull it when you need it
 * - It can also function as a basic audio data queue even if you just have
 *   sound that needs to pass from one place to another.
 * - You can hook callbacks up to them when more data is added or requested,
 *   to manage data on-the-fly.
 *
 * Audio streams are the core of the SDL3 audio interface. You create one or
 * more of them, bind them to an opened audio device, and feed data to them
 * (or for recording, consume data from them).
 *
 * \since This struct is available since SDL 3.2.0.
 *
 * \sa SDL_CreateAudioStream
 */
typedef struct SDL_AudioStream SDL_AudioStream;


/* Function prototypes */

/**
 * Use this function to get the number of built-in audio drivers.
 *
 * This function returns a hardcoded number. This never returns a negative
 * value; if there are no drivers compiled into this build of SDL, this
 * function returns zero. The presence of a driver in this list does not mean
 * it will function, it just means SDL is capable of interacting with that
 * interface. For example, a build of SDL might have esound support, but if
 * there's no esound server available, SDL's esound driver would fail if used.
 *
 * By default, SDL tries all drivers, in its preferred order, until one is
 * found to be usable.
 *
 * \returns the number of built-in audio drivers.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioDriver
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetNumAudioDrivers(void);

/**
 * Use this function to get the name of a built in audio driver.
 *
 * The list of audio drivers is given in the order that they are normally
 * initialized by default; the drivers that seem more reasonable to choose
 * first (as far as the SDL developers believe) are earlier in the list.
 *
 * The names of drivers are all simple, low-ASCII identifiers, like "alsa",
 * "coreaudio" or "wasapi". These never have Unicode characters, and are not
 * meant to be proper names.
 *
 * \param index the index of the audio driver; the value ranges from 0 to
 *              SDL_GetNumAudioDrivers() - 1.
 * \returns the name of the audio driver at the requested index, or NULL if an
 *          invalid index was specified.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetNumAudioDrivers
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetAudioDriver(int index);

/**
 * Get the name of the current audio driver.
 *
 * The names of drivers are all simple, low-ASCII identifiers, like "alsa",
 * "coreaudio" or "wasapi". These never have Unicode characters, and are not
 * meant to be proper names.
 *
 * \returns the name of the current audio driver or NULL if no driver has been
 *          initialized.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetCurrentAudioDriver(void);

/**
 * Get a list of currently-connected audio playback devices.
 *
 * This returns of list of available devices that play sound, perhaps to
 * speakers or headphones ("playback" devices). If you want devices that
 * record audio, like a microphone ("recording" devices), use
 * SDL_GetAudioRecordingDevices() instead.
 *
 * This only returns a list of physical devices; it will not have any device
 * IDs returned by SDL_OpenAudioDevice().
 *
 * If this function returns NULL, to signify an error, `*count` will be set to
 * zero.
 *
 * \param count a pointer filled in with the number of devices returned, may
 *              be NULL.
 * \returns a 0 terminated array of device instance IDs or NULL on error; call
 *          SDL_GetError() for more information. This should be freed with
 *          SDL_free() when it is no longer needed.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_OpenAudioDevice
 * \sa SDL_GetAudioRecordingDevices
 */
extern SDL_DECLSPEC SDL_AudioDeviceID * SDLCALL SDL_GetAudioPlaybackDevices(int *count);

/**
 * Get a list of currently-connected audio recording devices.
 *
 * This returns of list of available devices that record audio, like a
 * microphone ("recording" devices). If you want devices that play sound,
 * perhaps to speakers or headphones ("playback" devices), use
 * SDL_GetAudioPlaybackDevices() instead.
 *
 * This only returns a list of physical devices; it will not have any device
 * IDs returned by SDL_OpenAudioDevice().
 *
 * If this function returns NULL, to signify an error, `*count` will be set to
 * zero.
 *
 * \param count a pointer filled in with the number of devices returned, may
 *              be NULL.
 * \returns a 0 terminated array of device instance IDs, or NULL on failure;
 *          call SDL_GetError() for more information. This should be freed
 *          with SDL_free() when it is no longer needed.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_OpenAudioDevice
 * \sa SDL_GetAudioPlaybackDevices
 */
extern SDL_DECLSPEC SDL_AudioDeviceID * SDLCALL SDL_GetAudioRecordingDevices(int *count);

/**
 * Get the human-readable name of a specific audio device.
 *
 * \param devid the instance ID of the device to query.
 * \returns the name of the audio device, or NULL on failure; call
 *          SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioPlaybackDevices
 * \sa SDL_GetAudioRecordingDevices
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetAudioDeviceName(SDL_AudioDeviceID devid);

/**
 * Get the current audio format of a specific audio device.
 *
 * For an opened device, this will report the format the device is currently
 * using. If the device isn't yet opened, this will report the device's
 * preferred format (or a reasonable default if this can't be determined).
 *
 * You may also specify SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK or
 * SDL_AUDIO_DEVICE_DEFAULT_RECORDING here, which is useful for getting a
 * reasonable recommendation before opening the system-recommended default
 * device.
 *
 * You can also use this to request the current device buffer size. This is
 * specified in sample frames and represents the amount of data SDL will feed
 * to the physical hardware in each chunk. This can be converted to
 * milliseconds of audio with the following equation:
 *
 * `ms = (int) ((((Sint64) frames) * 1000) / spec.freq);`
 *
 * Buffer size is only important if you need low-level control over the audio
 * playback timing. Most apps do not need this.
 *
 * \param devid the instance ID of the device to query.
 * \param spec on return, will be filled with device details.
 * \param sample_frames pointer to store device buffer size, in sample frames.
 *                      Can be NULL.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetAudioDeviceFormat(SDL_AudioDeviceID devid, SDL_AudioSpec *spec, int *sample_frames);

/**
 * Get the current channel map of an audio device.
 *
 * Channel maps are optional; most things do not need them, instead passing
 * data in the [order that SDL expects](CategoryAudio#channel-layouts).
 *
 * Audio devices usually have no remapping applied. This is represented by
 * returning NULL, and does not signify an error.
 *
 * \param devid the instance ID of the device to query.
 * \param count On output, set to number of channels in the map. Can be NULL.
 * \returns an array of the current channel mapping, with as many elements as
 *          the current output spec's channels, or NULL if default. This
 *          should be freed with SDL_free() when it is no longer needed.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamInputChannelMap
 */
extern SDL_DECLSPEC int * SDLCALL SDL_GetAudioDeviceChannelMap(SDL_AudioDeviceID devid, int *count);

/**
 * Open a specific audio device.
 *
 * You can open both playback and recording devices through this function.
 * Playback devices will take data from bound audio streams, mix it, and send
 * it to the hardware. Recording devices will feed any bound audio streams
 * with a copy of any incoming data.
 *
 * An opened audio device starts out with no audio streams bound. To start
 * audio playing, bind a stream and supply audio data to it. Unlike SDL2,
 * there is no audio callback; you only bind audio streams and make sure they
 * have data flowing into them (however, you can simulate SDL2's semantics
 * fairly closely by using SDL_OpenAudioDeviceStream instead of this
 * function).
 *
 * If you don't care about opening a specific device, pass a `devid` of either
 * `SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK` or
 * `SDL_AUDIO_DEVICE_DEFAULT_RECORDING`. In this case, SDL will try to pick
 * the most reasonable default, and may also switch between physical devices
 * seamlessly later, if the most reasonable default changes during the
 * lifetime of this opened device (user changed the default in the OS's system
 * preferences, the default got unplugged so the system jumped to a new
 * default, the user plugged in headphones on a mobile device, etc). Unless
 * you have a good reason to choose a specific device, this is probably what
 * you want.
 *
 * You may request a specific format for the audio device, but there is no
 * promise the device will honor that request for several reasons. As such,
 * it's only meant to be a hint as to what data your app will provide. Audio
 * streams will accept data in whatever format you specify and manage
 * conversion for you as appropriate. SDL_GetAudioDeviceFormat can tell you
 * the preferred format for the device before opening and the actual format
 * the device is using after opening.
 *
 * It's legal to open the same device ID more than once; each successful open
 * will generate a new logical SDL_AudioDeviceID that is managed separately
 * from others on the same physical device. This allows libraries to open a
 * device separately from the main app and bind its own streams without
 * conflicting.
 *
 * It is also legal to open a device ID returned by a previous call to this
 * function; doing so just creates another logical device on the same physical
 * device. This may be useful for making logical groupings of audio streams.
 *
 * This function returns the opened device ID on success. This is a new,
 * unique SDL_AudioDeviceID that represents a logical device.
 *
 * Some backends might offer arbitrary devices (for example, a networked audio
 * protocol that can connect to an arbitrary server). For these, as a change
 * from SDL2, you should open a default device ID and use an SDL hint to
 * specify the target if you care, or otherwise let the backend figure out a
 * reasonable default. Most backends don't offer anything like this, and often
 * this would be an end user setting an environment variable for their custom
 * need, and not something an application should specifically manage.
 *
 * When done with an audio device, possibly at the end of the app's life, one
 * should call SDL_CloseAudioDevice() on the returned device id.
 *
 * \param devid the device instance id to open, or
 *              SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK or
 *              SDL_AUDIO_DEVICE_DEFAULT_RECORDING for the most reasonable
 *              default device.
 * \param spec the requested device configuration. Can be NULL to use
 *             reasonable defaults.
 * \returns the device ID on success or 0 on failure; call SDL_GetError() for
 *          more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CloseAudioDevice
 * \sa SDL_GetAudioDeviceFormat
 */
extern SDL_DECLSPEC SDL_AudioDeviceID SDLCALL SDL_OpenAudioDevice(SDL_AudioDeviceID devid, const SDL_AudioSpec *spec);

/**
 * Determine if an audio device is physical (instead of logical).
 *
 * An SDL_AudioDeviceID that represents physical hardware is a physical
 * device; there is one for each piece of hardware that SDL can see. Logical
 * devices are created by calling SDL_OpenAudioDevice or
 * SDL_OpenAudioDeviceStream, and while each is associated with a physical
 * device, there can be any number of logical devices on one physical device.
 *
 * For the most part, logical and physical IDs are interchangeable--if you try
 * to open a logical device, SDL understands to assign that effort to the
 * underlying physical device, etc. However, it might be useful to know if an
 * arbitrary device ID is physical or logical. This function reports which.
 *
 * This function may return either true or false for invalid device IDs.
 *
 * \param devid the device ID to query.
 * \returns true if devid is a physical device, false if it is logical.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_IsAudioDevicePhysical(SDL_AudioDeviceID devid);

/**
 * Determine if an audio device is a playback device (instead of recording).
 *
 * This function may return either true or false for invalid device IDs.
 *
 * \param devid the device ID to query.
 * \returns true if devid is a playback device, false if it is recording.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_IsAudioDevicePlayback(SDL_AudioDeviceID devid);

/**
 * Use this function to pause audio playback on a specified device.
 *
 * This function pauses audio processing for a given device. Any bound audio
 * streams will not progress, and no audio will be generated. Pausing one
 * device does not prevent other unpaused devices from running.
 *
 * Unlike in SDL2, audio devices start in an _unpaused_ state, since an app
 * has to bind a stream before any audio will flow. Pausing a paused device is
 * a legal no-op.
 *
 * Pausing a device can be useful to halt all audio without unbinding all the
 * audio streams. This might be useful while a game is paused, or a level is
 * loading, etc.
 *
 * Physical devices can not be paused or unpaused, only logical devices
 * created through SDL_OpenAudioDevice() can be.
 *
 * \param devid a device opened by SDL_OpenAudioDevice().
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_ResumeAudioDevice
 * \sa SDL_AudioDevicePaused
 */
extern SDL_DECLSPEC bool SDLCALL SDL_PauseAudioDevice(SDL_AudioDeviceID devid);

/**
 * Use this function to unpause audio playback on a specified device.
 *
 * This function unpauses audio processing for a given device that has
 * previously been paused with SDL_PauseAudioDevice(). Once unpaused, any
 * bound audio streams will begin to progress again, and audio can be
 * generated.
 *
 * Unlike in SDL2, audio devices start in an _unpaused_ state, since an app
 * has to bind a stream before any audio will flow. Unpausing an unpaused
 * device is a legal no-op.
 *
 * Physical devices can not be paused or unpaused, only logical devices
 * created through SDL_OpenAudioDevice() can be.
 *
 * \param devid a device opened by SDL_OpenAudioDevice().
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_AudioDevicePaused
 * \sa SDL_PauseAudioDevice
 */
extern SDL_DECLSPEC bool SDLCALL SDL_ResumeAudioDevice(SDL_AudioDeviceID devid);

/**
 * Use this function to query if an audio device is paused.
 *
 * Unlike in SDL2, audio devices start in an _unpaused_ state, since an app
 * has to bind a stream before any audio will flow.
 *
 * Physical devices can not be paused or unpaused, only logical devices
 * created through SDL_OpenAudioDevice() can be. Physical and invalid device
 * IDs will report themselves as unpaused here.
 *
 * \param devid a device opened by SDL_OpenAudioDevice().
 * \returns true if device is valid and paused, false otherwise.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_PauseAudioDevice
 * \sa SDL_ResumeAudioDevice
 */
extern SDL_DECLSPEC bool SDLCALL SDL_AudioDevicePaused(SDL_AudioDeviceID devid);

/**
 * Get the gain of an audio device.
 *
 * The gain of a device is its volume; a larger gain means a louder output,
 * with a gain of zero being silence.
 *
 * Audio devices default to a gain of 1.0f (no change in output).
 *
 * Physical devices may not have their gain changed, only logical devices, and
 * this function will always return -1.0f when used on physical devices.
 *
 * \param devid the audio device to query.
 * \returns the gain of the device or -1.0f on failure; call SDL_GetError()
 *          for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioDeviceGain
 */
extern SDL_DECLSPEC float SDLCALL SDL_GetAudioDeviceGain(SDL_AudioDeviceID devid);

/**
 * Change the gain of an audio device.
 *
 * The gain of a device is its volume; a larger gain means a louder output,
 * with a gain of zero being silence.
 *
 * Audio devices default to a gain of 1.0f (no change in output).
 *
 * Physical devices may not have their gain changed, only logical devices, and
 * this function will always return false when used on physical devices. While
 * it might seem attractive to adjust several logical devices at once in this
 * way, it would allow an app or library to interfere with another portion of
 * the program's otherwise-isolated devices.
 *
 * This is applied, along with any per-audiostream gain, during playback to
 * the hardware, and can be continuously changed to create various effects. On
 * recording devices, this will adjust the gain before passing the data into
 * an audiostream; that recording audiostream can then adjust its gain further
 * when outputting the data elsewhere, if it likes, but that second gain is
 * not applied until the data leaves the audiostream again.
 *
 * \param devid the audio device on which to change gain.
 * \param gain the gain. 1.0f is no change, 0.0f is silence.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioDeviceGain
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioDeviceGain(SDL_AudioDeviceID devid, float gain);

/**
 * Close a previously-opened audio device.
 *
 * The application should close open audio devices once they are no longer
 * needed.
 *
 * This function may block briefly while pending audio data is played by the
 * hardware, so that applications don't drop the last buffer of data they
 * supplied if terminating immediately afterwards.
 *
 * \param devid an audio device id previously returned by
 *              SDL_OpenAudioDevice().
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_OpenAudioDevice
 */
extern SDL_DECLSPEC void SDLCALL SDL_CloseAudioDevice(SDL_AudioDeviceID devid);

/**
 * Bind a list of audio streams to an audio device.
 *
 * Audio data will flow through any bound streams. For a playback device, data
 * for all bound streams will be mixed together and fed to the device. For a
 * recording device, a copy of recorded data will be provided to each bound
 * stream.
 *
 * Audio streams can only be bound to an open device. This operation is
 * atomic--all streams bound in the same call will start processing at the
 * same time, so they can stay in sync. Also: either all streams will be bound
 * or none of them will be.
 *
 * It is an error to bind an already-bound stream; it must be explicitly
 * unbound first.
 *
 * Binding a stream to a device will set its output format for playback
 * devices, and its input format for recording devices, so they match the
 * device's settings. The caller is welcome to change the other end of the
 * stream's format at any time with SDL_SetAudioStreamFormat().
 *
 * \param devid an audio device to bind a stream to.
 * \param streams an array of audio streams to bind.
 * \param num_streams number streams listed in the `streams` array.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_BindAudioStreams
 * \sa SDL_UnbindAudioStream
 * \sa SDL_GetAudioStreamDevice
 */
extern SDL_DECLSPEC bool SDLCALL SDL_BindAudioStreams(SDL_AudioDeviceID devid, SDL_AudioStream * const *streams, int num_streams);

/**
 * Bind a single audio stream to an audio device.
 *
 * This is a convenience function, equivalent to calling
 * `SDL_BindAudioStreams(devid, &stream, 1)`.
 *
 * \param devid an audio device to bind a stream to.
 * \param stream an audio stream to bind to a device.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_BindAudioStreams
 * \sa SDL_UnbindAudioStream
 * \sa SDL_GetAudioStreamDevice
 */
extern SDL_DECLSPEC bool SDLCALL SDL_BindAudioStream(SDL_AudioDeviceID devid, SDL_AudioStream *stream);

/**
 * Unbind a list of audio streams from their audio devices.
 *
 * The streams being unbound do not all have to be on the same device. All
 * streams on the same device will be unbound atomically (data will stop
 * flowing through all unbound streams on the same device at the same time).
 *
 * Unbinding a stream that isn't bound to a device is a legal no-op.
 *
 * \param streams an array of audio streams to unbind. Can be NULL or contain
 *                NULL.
 * \param num_streams number streams listed in the `streams` array.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_BindAudioStreams
 */
extern SDL_DECLSPEC void SDLCALL SDL_UnbindAudioStreams(SDL_AudioStream * const *streams, int num_streams);

/**
 * Unbind a single audio stream from its audio device.
 *
 * This is a convenience function, equivalent to calling
 * `SDL_UnbindAudioStreams(&stream, 1)`.
 *
 * \param stream an audio stream to unbind from a device. Can be NULL.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_BindAudioStream
 */
extern SDL_DECLSPEC void SDLCALL SDL_UnbindAudioStream(SDL_AudioStream *stream);

/**
 * Query an audio stream for its currently-bound device.
 *
 * This reports the audio device that an audio stream is currently bound to.
 *
 * If not bound, or invalid, this returns zero, which is not a valid device
 * ID.
 *
 * \param stream the audio stream to query.
 * \returns the bound audio device, or 0 if not bound or invalid.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_BindAudioStream
 * \sa SDL_BindAudioStreams
 */
extern SDL_DECLSPEC SDL_AudioDeviceID SDLCALL SDL_GetAudioStreamDevice(SDL_AudioStream *stream);

/**
 * Create a new audio stream.
 *
 * \param src_spec the format details of the input audio.
 * \param dst_spec the format details of the output audio.
 * \returns a new audio stream on success or NULL on failure; call
 *          SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_PutAudioStreamData
 * \sa SDL_GetAudioStreamData
 * \sa SDL_GetAudioStreamAvailable
 * \sa SDL_FlushAudioStream
 * \sa SDL_ClearAudioStream
 * \sa SDL_SetAudioStreamFormat
 * \sa SDL_DestroyAudioStream
 */
extern SDL_DECLSPEC SDL_AudioStream * SDLCALL SDL_CreateAudioStream(const SDL_AudioSpec *src_spec, const SDL_AudioSpec *dst_spec);

/**
 * Get the properties associated with an audio stream.
 *
 * \param stream the SDL_AudioStream to query.
 * \returns a valid property ID on success or 0 on failure; call
 *          SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC SDL_PropertiesID SDLCALL SDL_GetAudioStreamProperties(SDL_AudioStream *stream);

/**
 * Query the current format of an audio stream.
 *
 * \param stream the SDL_AudioStream to query.
 * \param src_spec where to store the input audio format; ignored if NULL.
 * \param dst_spec where to store the output audio format; ignored if NULL.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamFormat
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetAudioStreamFormat(SDL_AudioStream *stream, SDL_AudioSpec *src_spec, SDL_AudioSpec *dst_spec);

/**
 * Change the input and output formats of an audio stream.
 *
 * Future calls to and SDL_GetAudioStreamAvailable and SDL_GetAudioStreamData
 * will reflect the new format, and future calls to SDL_PutAudioStreamData
 * must provide data in the new input formats.
 *
 * Data that was previously queued in the stream will still be operated on in
 * the format that was current when it was added, which is to say you can put
 * the end of a sound file in one format to a stream, change formats for the
 * next sound file, and start putting that new data while the previous sound
 * file is still queued, and everything will still play back correctly.
 *
 * If a stream is bound to a device, then the format of the side of the stream
 * bound to a device cannot be changed (src_spec for recording devices,
 * dst_spec for playback devices). Attempts to make a change to this side will
 * be ignored, but this will not report an error. The other side's format can
 * be changed.
 *
 * \param stream the stream the format is being changed.
 * \param src_spec the new format of the audio input; if NULL, it is not
 *                 changed.
 * \param dst_spec the new format of the audio output; if NULL, it is not
 *                 changed.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioStreamFormat
 * \sa SDL_SetAudioStreamFrequencyRatio
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioStreamFormat(SDL_AudioStream *stream, const SDL_AudioSpec *src_spec, const SDL_AudioSpec *dst_spec);

/**
 * Get the frequency ratio of an audio stream.
 *
 * \param stream the SDL_AudioStream to query.
 * \returns the frequency ratio of the stream or 0.0 on failure; call
 *          SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamFrequencyRatio
 */
extern SDL_DECLSPEC float SDLCALL SDL_GetAudioStreamFrequencyRatio(SDL_AudioStream *stream);

/**
 * Change the frequency ratio of an audio stream.
 *
 * The frequency ratio is used to adjust the rate at which input data is
 * consumed. Changing this effectively modifies the speed and pitch of the
 * audio. A value greater than 1.0 will play the audio faster, and at a higher
 * pitch. A value less than 1.0 will play the audio slower, and at a lower
 * pitch.
 *
 * This is applied during SDL_GetAudioStreamData, and can be continuously
 * changed to create various effects.
 *
 * \param stream the stream the frequency ratio is being changed.
 * \param ratio the frequency ratio. 1.0 is normal speed. Must be between 0.01
 *              and 100.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioStreamFrequencyRatio
 * \sa SDL_SetAudioStreamFormat
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioStreamFrequencyRatio(SDL_AudioStream *stream, float ratio);

/**
 * Get the gain of an audio stream.
 *
 * The gain of a stream is its volume; a larger gain means a louder output,
 * with a gain of zero being silence.
 *
 * Audio streams default to a gain of 1.0f (no change in output).
 *
 * \param stream the SDL_AudioStream to query.
 * \returns the gain of the stream or -1.0f on failure; call SDL_GetError()
 *          for more information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamGain
 */
extern SDL_DECLSPEC float SDLCALL SDL_GetAudioStreamGain(SDL_AudioStream *stream);

/**
 * Change the gain of an audio stream.
 *
 * The gain of a stream is its volume; a larger gain means a louder output,
 * with a gain of zero being silence.
 *
 * Audio streams default to a gain of 1.0f (no change in output).
 *
 * This is applied during SDL_GetAudioStreamData, and can be continuously
 * changed to create various effects.
 *
 * \param stream the stream on which the gain is being changed.
 * \param gain the gain. 1.0f is no change, 0.0f is silence.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioStreamGain
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioStreamGain(SDL_AudioStream *stream, float gain);

/**
 * Get the current input channel map of an audio stream.
 *
 * Channel maps are optional; most things do not need them, instead passing
 * data in the [order that SDL expects](CategoryAudio#channel-layouts).
 *
 * Audio streams default to no remapping applied. This is represented by
 * returning NULL, and does not signify an error.
 *
 * \param stream the SDL_AudioStream to query.
 * \param count On output, set to number of channels in the map. Can be NULL.
 * \returns an array of the current channel mapping, with as many elements as
 *          the current output spec's channels, or NULL if default. This
 *          should be freed with SDL_free() when it is no longer needed.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamInputChannelMap
 */
extern SDL_DECLSPEC int * SDLCALL SDL_GetAudioStreamInputChannelMap(SDL_AudioStream *stream, int *count);

/**
 * Get the current output channel map of an audio stream.
 *
 * Channel maps are optional; most things do not need them, instead passing
 * data in the [order that SDL expects](CategoryAudio#channel-layouts).
 *
 * Audio streams default to no remapping applied. This is represented by
 * returning NULL, and does not signify an error.
 *
 * \param stream the SDL_AudioStream to query.
 * \param count On output, set to number of channels in the map. Can be NULL.
 * \returns an array of the current channel mapping, with as many elements as
 *          the current output spec's channels, or NULL if default. This
 *          should be freed with SDL_free() when it is no longer needed.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamInputChannelMap
 */
extern SDL_DECLSPEC int * SDLCALL SDL_GetAudioStreamOutputChannelMap(SDL_AudioStream *stream, int *count);

/**
 * Set the current input channel map of an audio stream.
 *
 * Channel maps are optional; most things do not need them, instead passing
 * data in the [order that SDL expects](CategoryAudio#channel-layouts).
 *
 * The input channel map reorders data that is added to a stream via
 * SDL_PutAudioStreamData. Future calls to SDL_PutAudioStreamData must provide
 * data in the new channel order.
 *
 * Each item in the array represents an input channel, and its value is the
 * channel that it should be remapped to. To reverse a stereo signal's left
 * and right values, you'd have an array of `{ 1, 0 }`. It is legal to remap
 * multiple channels to the same thing, so `{ 1, 1 }` would duplicate the
 * right channel to both channels of a stereo signal. An element in the
 * channel map set to -1 instead of a valid channel will mute that channel,
 * setting it to a silence value.
 *
 * You cannot change the number of channels through a channel map, just
 * reorder/mute them.
 *
 * Data that was previously queued in the stream will still be operated on in
 * the order that was current when it was added, which is to say you can put
 * the end of a sound file in one order to a stream, change orders for the
 * next sound file, and start putting that new data while the previous sound
 * file is still queued, and everything will still play back correctly.
 *
 * Audio streams default to no remapping applied. Passing a NULL channel map
 * is legal, and turns off remapping.
 *
 * SDL will copy the channel map; the caller does not have to save this array
 * after this call.
 *
 * If `count` is not equal to the current number of channels in the audio
 * stream's format, this will fail. This is a safety measure to make sure a
 * race condition hasn't changed the format while this call is setting the
 * channel map.
 *
 * Unlike attempting to change the stream's format, the input channel map on a
 * stream bound to a recording device is permitted to change at any time; any
 * data added to the stream from the device after this call will have the new
 * mapping, but previously-added data will still have the prior mapping.
 *
 * \param stream the SDL_AudioStream to change.
 * \param chmap the new channel map, NULL to reset to default.
 * \param count The number of channels in the map.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running. Don't change the
 *               stream's format to have a different number of channels from a
 *               a different thread at the same time, though!
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamInputChannelMap
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioStreamInputChannelMap(SDL_AudioStream *stream, const int *chmap, int count);

/**
 * Set the current output channel map of an audio stream.
 *
 * Channel maps are optional; most things do not need them, instead passing
 * data in the [order that SDL expects](CategoryAudio#channel-layouts).
 *
 * The output channel map reorders data that leaving a stream via
 * SDL_GetAudioStreamData.
 *
 * Each item in the array represents an input channel, and its value is the
 * channel that it should be remapped to. To reverse a stereo signal's left
 * and right values, you'd have an array of `{ 1, 0 }`. It is legal to remap
 * multiple channels to the same thing, so `{ 1, 1 }` would duplicate the
 * right channel to both channels of a stereo signal. An element in the
 * channel map set to -1 instead of a valid channel will mute that channel,
 * setting it to a silence value.
 *
 * You cannot change the number of channels through a channel map, just
 * reorder/mute them.
 *
 * The output channel map can be changed at any time, as output remapping is
 * applied during SDL_GetAudioStreamData.
 *
 * Audio streams default to no remapping applied. Passing a NULL channel map
 * is legal, and turns off remapping.
 *
 * SDL will copy the channel map; the caller does not have to save this array
 * after this call.
 *
 * If `count` is not equal to the current number of channels in the audio
 * stream's format, this will fail. This is a safety measure to make sure a
 * race condition hasn't changed the format while this call is setting the
 * channel map.
 *
 * Unlike attempting to change the stream's format, the output channel map on
 * a stream bound to a recording device is permitted to change at any time;
 * any data added to the stream after this call will have the new mapping, but
 * previously-added data will still have the prior mapping. When the channel
 * map doesn't match the hardware's channel layout, SDL will convert the data
 * before feeding it to the device for playback.
 *
 * \param stream the SDL_AudioStream to change.
 * \param chmap the new channel map, NULL to reset to default.
 * \param count The number of channels in the map.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, as it holds
 *               a stream-specific mutex while running. Don't change the
 *               stream's format to have a different number of channels from a
 *               a different thread at the same time, though!
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamInputChannelMap
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioStreamOutputChannelMap(SDL_AudioStream *stream, const int *chmap, int count);

/**
 * Add data to the stream.
 *
 * This data must match the format/channels/samplerate specified in the latest
 * call to SDL_SetAudioStreamFormat, or the format specified when creating the
 * stream if it hasn't been changed.
 *
 * Note that this call simply copies the unconverted data for later. This is
 * different than SDL2, where data was converted during the Put call and the
 * Get call would just dequeue the previously-converted data.
 *
 * \param stream the stream the audio data is being added to.
 * \param buf a pointer to the audio data to add.
 * \param len the number of bytes to write to the stream.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, but if the
 *               stream has a callback set, the caller might need to manage
 *               extra locking.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_ClearAudioStream
 * \sa SDL_FlushAudioStream
 * \sa SDL_GetAudioStreamData
 * \sa SDL_GetAudioStreamQueued
 */
extern SDL_DECLSPEC bool SDLCALL SDL_PutAudioStreamData(SDL_AudioStream *stream, const void *buf, int len);

/**
 * Get converted/resampled data from the stream.
 *
 * The input/output data format/channels/samplerate is specified when creating
 * the stream, and can be changed after creation by calling
 * SDL_SetAudioStreamFormat.
 *
 * Note that any conversion and resampling necessary is done during this call,
 * and SDL_PutAudioStreamData simply queues unconverted data for later. This
 * is different than SDL2, where that work was done while inputting new data
 * to the stream and requesting the output just copied the converted data.
 *
 * \param stream the stream the audio is being requested from.
 * \param buf a buffer to fill with audio data.
 * \param len the maximum number of bytes to fill.
 * \returns the number of bytes read from the stream or -1 on failure; call
 *          SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread, but if the
 *               stream has a callback set, the caller might need to manage
 *               extra locking.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_ClearAudioStream
 * \sa SDL_GetAudioStreamAvailable
 * \sa SDL_PutAudioStreamData
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetAudioStreamData(SDL_AudioStream *stream, void *buf, int len);

/**
 * Get the number of converted/resampled bytes available.
 *
 * The stream may be buffering data behind the scenes until it has enough to
 * resample correctly, so this number might be lower than what you expect, or
 * even be zero. Add more data or flush the stream if you need the data now.
 *
 * If the stream has so much data that it would overflow an int, the return
 * value is clamped to a maximum value, but no queued data is lost; if there
 * are gigabytes of data queued, the app might need to read some of it with
 * SDL_GetAudioStreamData before this function's return value is no longer
 * clamped.
 *
 * \param stream the audio stream to query.
 * \returns the number of converted/resampled bytes available or -1 on
 *          failure; call SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioStreamData
 * \sa SDL_PutAudioStreamData
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetAudioStreamAvailable(SDL_AudioStream *stream);


/**
 * Get the number of bytes currently queued.
 *
 * This is the number of bytes put into a stream as input, not the number that
 * can be retrieved as output. Because of several details, it's not possible
 * to calculate one number directly from the other. If you need to know how
 * much usable data can be retrieved right now, you should use
 * SDL_GetAudioStreamAvailable() and not this function.
 *
 * Note that audio streams can change their input format at any time, even if
 * there is still data queued in a different format, so the returned byte
 * count will not necessarily match the number of _sample frames_ available.
 * Users of this API should be aware of format changes they make when feeding
 * a stream and plan accordingly.
 *
 * Queued data is not converted until it is consumed by
 * SDL_GetAudioStreamData, so this value should be representative of the exact
 * data that was put into the stream.
 *
 * If the stream has so much data that it would overflow an int, the return
 * value is clamped to a maximum value, but no queued data is lost; if there
 * are gigabytes of data queued, the app might need to read some of it with
 * SDL_GetAudioStreamData before this function's return value is no longer
 * clamped.
 *
 * \param stream the audio stream to query.
 * \returns the number of bytes queued or -1 on failure; call SDL_GetError()
 *          for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_PutAudioStreamData
 * \sa SDL_ClearAudioStream
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetAudioStreamQueued(SDL_AudioStream *stream);


/**
 * Tell the stream that you're done sending data, and anything being buffered
 * should be converted/resampled and made available immediately.
 *
 * It is legal to add more data to a stream after flushing, but there may be
 * audio gaps in the output. Generally this is intended to signal the end of
 * input, so the complete output becomes available.
 *
 * \param stream the audio stream to flush.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_PutAudioStreamData
 */
extern SDL_DECLSPEC bool SDLCALL SDL_FlushAudioStream(SDL_AudioStream *stream);

/**
 * Clear any pending data in the stream.
 *
 * This drops any queued data, so there will be nothing to read from the
 * stream until more is added.
 *
 * \param stream the audio stream to clear.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioStreamAvailable
 * \sa SDL_GetAudioStreamData
 * \sa SDL_GetAudioStreamQueued
 * \sa SDL_PutAudioStreamData
 */
extern SDL_DECLSPEC bool SDLCALL SDL_ClearAudioStream(SDL_AudioStream *stream);

/**
 * Use this function to pause audio playback on the audio device associated
 * with an audio stream.
 *
 * This function pauses audio processing for a given device. Any bound audio
 * streams will not progress, and no audio will be generated. Pausing one
 * device does not prevent other unpaused devices from running.
 *
 * Pausing a device can be useful to halt all audio without unbinding all the
 * audio streams. This might be useful while a game is paused, or a level is
 * loading, etc.
 *
 * \param stream the audio stream associated with the audio device to pause.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_ResumeAudioStreamDevice
 */
extern SDL_DECLSPEC bool SDLCALL SDL_PauseAudioStreamDevice(SDL_AudioStream *stream);

/**
 * Use this function to unpause audio playback on the audio device associated
 * with an audio stream.
 *
 * This function unpauses audio processing for a given device that has
 * previously been paused. Once unpaused, any bound audio streams will begin
 * to progress again, and audio can be generated.
 *
 * Remember, SDL_OpenAudioDeviceStream opens device in a paused state, so this
 * function call is required for audio playback to begin on such device.
 *
 * \param stream the audio stream associated with the audio device to resume.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_PauseAudioStreamDevice
 */
extern SDL_DECLSPEC bool SDLCALL SDL_ResumeAudioStreamDevice(SDL_AudioStream *stream);

/**
 * Use this function to query if an audio device associated with a stream is
 * paused.
 *
 * Unlike in SDL2, audio devices start in an _unpaused_ state, since an app
 * has to bind a stream before any audio will flow.
 *
 * \param stream the audio stream associated with the audio device to query.
 * \returns true if device is valid and paused, false otherwise.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_PauseAudioStreamDevice
 * \sa SDL_ResumeAudioStreamDevice
 */
extern SDL_DECLSPEC bool SDLCALL SDL_AudioStreamDevicePaused(SDL_AudioStream *stream);


/**
 * Lock an audio stream for serialized access.
 *
 * Each SDL_AudioStream has an internal mutex it uses to protect its data
 * structures from threading conflicts. This function allows an app to lock
 * that mutex, which could be useful if registering callbacks on this stream.
 *
 * One does not need to lock a stream to use in it most cases, as the stream
 * manages this lock internally. However, this lock is held during callbacks,
 * which may run from arbitrary threads at any time, so if an app needs to
 * protect shared data during those callbacks, locking the stream guarantees
 * that the callback is not running while the lock is held.
 *
 * As this is just a wrapper over SDL_LockMutex for an internal lock; it has
 * all the same attributes (recursive locks are allowed, etc).
 *
 * \param stream the audio stream to lock.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_UnlockAudioStream
 */
extern SDL_DECLSPEC bool SDLCALL SDL_LockAudioStream(SDL_AudioStream *stream);


/**
 * Unlock an audio stream for serialized access.
 *
 * This unlocks an audio stream after a call to SDL_LockAudioStream.
 *
 * \param stream the audio stream to unlock.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety You should only call this from the same thread that
 *               previously called SDL_LockAudioStream.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_LockAudioStream
 */
extern SDL_DECLSPEC bool SDLCALL SDL_UnlockAudioStream(SDL_AudioStream *stream);

/**
 * A callback that fires when data passes through an SDL_AudioStream.
 *
 * Apps can (optionally) register a callback with an audio stream that is
 * called when data is added with SDL_PutAudioStreamData, or requested with
 * SDL_GetAudioStreamData.
 *
 * Two values are offered here: one is the amount of additional data needed to
 * satisfy the immediate request (which might be zero if the stream already
 * has enough data queued) and the other is the total amount being requested.
 * In a Get call triggering a Put callback, these values can be different. In
 * a Put call triggering a Get callback, these values are always the same.
 *
 * Byte counts might be slightly overestimated due to buffering or resampling,
 * and may change from call to call.
 *
 * This callback is not required to do anything. Generally this is useful for
 * adding/reading data on demand, and the app will often put/get data as
 * appropriate, but the system goes on with the data currently available to it
 * if this callback does nothing.
 *
 * \param stream the SDL audio stream associated with this callback.
 * \param additional_amount the amount of data, in bytes, that is needed right
 *                          now.
 * \param total_amount the total amount of data requested, in bytes, that is
 *                     requested or available.
 * \param userdata an opaque pointer provided by the app for their personal
 *                 use.
 *
 * \threadsafety This callbacks may run from any thread, so if you need to
 *               protect shared data, you should use SDL_LockAudioStream to
 *               serialize access; this lock will be held before your callback
 *               is called, so your callback does not need to manage the lock
 *               explicitly.
 *
 * \since This datatype is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamGetCallback
 * \sa SDL_SetAudioStreamPutCallback
 */
typedef void (SDLCALL *SDL_AudioStreamCallback)(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount);

/**
 * Set a callback that runs when data is requested from an audio stream.
 *
 * This callback is called _before_ data is obtained from the stream, giving
 * the callback the chance to add more on-demand.
 *
 * The callback can (optionally) call SDL_PutAudioStreamData() to add more
 * audio to the stream during this call; if needed, the request that triggered
 * this callback will obtain the new data immediately.
 *
 * The callback's `additional_amount` argument is roughly how many bytes of
 * _unconverted_ data (in the stream's input format) is needed by the caller,
 * although this may overestimate a little for safety. This takes into account
 * how much is already in the stream and only asks for any extra necessary to
 * resolve the request, which means the callback may be asked for zero bytes,
 * and a different amount on each call.
 *
 * The callback is not required to supply exact amounts; it is allowed to
 * supply too much or too little or none at all. The caller will get what's
 * available, up to the amount they requested, regardless of this callback's
 * outcome.
 *
 * Clearing or flushing an audio stream does not call this callback.
 *
 * This function obtains the stream's lock, which means any existing callback
 * (get or put) in progress will finish running before setting the new
 * callback.
 *
 * Setting a NULL function turns off the callback.
 *
 * \param stream the audio stream to set the new callback on.
 * \param callback the new callback function to call when data is requested
 *                 from the stream.
 * \param userdata an opaque pointer provided to the callback for its own
 *                 personal use.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information. This only fails if `stream` is NULL.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamPutCallback
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioStreamGetCallback(SDL_AudioStream *stream, SDL_AudioStreamCallback callback, void *userdata);

/**
 * Set a callback that runs when data is added to an audio stream.
 *
 * This callback is called _after_ the data is added to the stream, giving the
 * callback the chance to obtain it immediately.
 *
 * The callback can (optionally) call SDL_GetAudioStreamData() to obtain audio
 * from the stream during this call.
 *
 * The callback's `additional_amount` argument is how many bytes of
 * _converted_ data (in the stream's output format) was provided by the
 * caller, although this may underestimate a little for safety. This value
 * might be less than what is currently available in the stream, if data was
 * already there, and might be less than the caller provided if the stream
 * needs to keep a buffer to aid in resampling. Which means the callback may
 * be provided with zero bytes, and a different amount on each call.
 *
 * The callback may call SDL_GetAudioStreamAvailable to see the total amount
 * currently available to read from the stream, instead of the total provided
 * by the current call.
 *
 * The callback is not required to obtain all data. It is allowed to read less
 * or none at all. Anything not read now simply remains in the stream for
 * later access.
 *
 * Clearing or flushing an audio stream does not call this callback.
 *
 * This function obtains the stream's lock, which means any existing callback
 * (get or put) in progress will finish running before setting the new
 * callback.
 *
 * Setting a NULL function turns off the callback.
 *
 * \param stream the audio stream to set the new callback on.
 * \param callback the new callback function to call when data is added to the
 *                 stream.
 * \param userdata an opaque pointer provided to the callback for its own
 *                 personal use.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information. This only fails if `stream` is NULL.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioStreamGetCallback
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioStreamPutCallback(SDL_AudioStream *stream, SDL_AudioStreamCallback callback, void *userdata);


/**
 * Free an audio stream.
 *
 * This will release all allocated data, including any audio that is still
 * queued. You do not need to manually clear the stream first.
 *
 * If this stream was bound to an audio device, it is unbound during this
 * call. If this stream was created with SDL_OpenAudioDeviceStream, the audio
 * device that was opened alongside this stream's creation will be closed,
 * too.
 *
 * \param stream the audio stream to destroy.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreateAudioStream
 */
extern SDL_DECLSPEC void SDLCALL SDL_DestroyAudioStream(SDL_AudioStream *stream);


/**
 * Convenience function for straightforward audio init for the common case.
 *
 * If all your app intends to do is provide a single source of PCM audio, this
 * function allows you to do all your audio setup in a single call.
 *
 * This is also intended to be a clean means to migrate apps from SDL2.
 *
 * This function will open an audio device, create a stream and bind it.
 * Unlike other methods of setup, the audio device will be closed when this
 * stream is destroyed, so the app can treat the returned SDL_AudioStream as
 * the only object needed to manage audio playback.
 *
 * Also unlike other functions, the audio device begins paused. This is to map
 * more closely to SDL2-style behavior, since there is no extra step here to
 * bind a stream to begin audio flowing. The audio device should be resumed
 * with `SDL_ResumeAudioStreamDevice(stream);`
 *
 * This function works with both playback and recording devices.
 *
 * The `spec` parameter represents the app's side of the audio stream. That
 * is, for recording audio, this will be the output format, and for playing
 * audio, this will be the input format. If spec is NULL, the system will
 * choose the format, and the app can use SDL_GetAudioStreamFormat() to obtain
 * this information later.
 *
 * If you don't care about opening a specific audio device, you can (and
 * probably _should_), use SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK for playback and
 * SDL_AUDIO_DEVICE_DEFAULT_RECORDING for recording.
 *
 * One can optionally provide a callback function; if NULL, the app is
 * expected to queue audio data for playback (or unqueue audio data if
 * capturing). Otherwise, the callback will begin to fire once the device is
 * unpaused.
 *
 * Destroying the returned stream with SDL_DestroyAudioStream will also close
 * the audio device associated with this stream.
 *
 * \param devid an audio device to open, or SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK
 *              or SDL_AUDIO_DEVICE_DEFAULT_RECORDING.
 * \param spec the audio stream's data format. Can be NULL.
 * \param callback a callback where the app will provide new data for
 *                 playback, or receive new data for recording. Can be NULL,
 *                 in which case the app will need to call
 *                 SDL_PutAudioStreamData or SDL_GetAudioStreamData as
 *                 necessary.
 * \param userdata app-controlled pointer passed to callback. Can be NULL.
 *                 Ignored if callback is NULL.
 * \returns an audio stream on success, ready to use, or NULL on failure; call
 *          SDL_GetError() for more information. When done with this stream,
 *          call SDL_DestroyAudioStream to free resources and close the
 *          device.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAudioStreamDevice
 * \sa SDL_ResumeAudioStreamDevice
 */
extern SDL_DECLSPEC SDL_AudioStream * SDLCALL SDL_OpenAudioDeviceStream(SDL_AudioDeviceID devid, const SDL_AudioSpec *spec, SDL_AudioStreamCallback callback, void *userdata);

/**
 * A callback that fires when data is about to be fed to an audio device.
 *
 * This is useful for accessing the final mix, perhaps for writing a
 * visualizer or applying a final effect to the audio data before playback.
 *
 * This callback should run as quickly as possible and not block for any
 * significant time, as this callback delays submission of data to the audio
 * device, which can cause audio playback problems.
 *
 * The postmix callback _must_ be able to handle any audio data format
 * specified in `spec`, which can change between callbacks if the audio device
 * changed. However, this only covers frequency and channel count; data is
 * always provided here in SDL_AUDIO_F32 format.
 *
 * The postmix callback runs _after_ logical device gain and audiostream gain
 * have been applied, which is to say you can make the output data louder at
 * this point than the gain settings would suggest.
 *
 * \param userdata a pointer provided by the app through
 *                 SDL_SetAudioPostmixCallback, for its own use.
 * \param spec the current format of audio that is to be submitted to the
 *             audio device.
 * \param buffer the buffer of audio samples to be submitted. The callback can
 *               inspect and/or modify this data.
 * \param buflen the size of `buffer` in bytes.
 *
 * \threadsafety This will run from a background thread owned by SDL. The
 *               application is responsible for locking resources the callback
 *               touches that need to be protected.
 *
 * \since This datatype is available since SDL 3.2.0.
 *
 * \sa SDL_SetAudioPostmixCallback
 */
typedef void (SDLCALL *SDL_AudioPostmixCallback)(void *userdata, const SDL_AudioSpec *spec, float *buffer, int buflen);

/**
 * Set a callback that fires when data is about to be fed to an audio device.
 *
 * This is useful for accessing the final mix, perhaps for writing a
 * visualizer or applying a final effect to the audio data before playback.
 *
 * The buffer is the final mix of all bound audio streams on an opened device;
 * this callback will fire regularly for any device that is both opened and
 * unpaused. If there is no new data to mix, either because no streams are
 * bound to the device or all the streams are empty, this callback will still
 * fire with the entire buffer set to silence.
 *
 * This callback is allowed to make changes to the data; the contents of the
 * buffer after this call is what is ultimately passed along to the hardware.
 *
 * The callback is always provided the data in float format (values from -1.0f
 * to 1.0f), but the number of channels or sample rate may be different than
 * the format the app requested when opening the device; SDL might have had to
 * manage a conversion behind the scenes, or the playback might have jumped to
 * new physical hardware when a system default changed, etc. These details may
 * change between calls. Accordingly, the size of the buffer might change
 * between calls as well.
 *
 * This callback can run at any time, and from any thread; if you need to
 * serialize access to your app's data, you should provide and use a mutex or
 * other synchronization device.
 *
 * All of this to say: there are specific needs this callback can fulfill, but
 * it is not the simplest interface. Apps should generally provide audio in
 * their preferred format through an SDL_AudioStream and let SDL handle the
 * difference.
 *
 * This function is extremely time-sensitive; the callback should do the least
 * amount of work possible and return as quickly as it can. The longer the
 * callback runs, the higher the risk of audio dropouts or other problems.
 *
 * This function will block until the audio device is in between iterations,
 * so any existing callback that might be running will finish before this
 * function sets the new callback and returns.
 *
 * Setting a NULL callback function disables any previously-set callback.
 *
 * \param devid the ID of an opened audio device.
 * \param callback a callback function to be called. Can be NULL.
 * \param userdata app-controlled pointer passed to callback. Can be NULL.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetAudioPostmixCallback(SDL_AudioDeviceID devid, SDL_AudioPostmixCallback callback, void *userdata);


/**
 * Load the audio data of a WAVE file into memory.
 *
 * Loading a WAVE file requires `src`, `spec`, `audio_buf` and `audio_len` to
 * be valid pointers. The entire data portion of the file is then loaded into
 * memory and decoded if necessary.
 *
 * Supported formats are RIFF WAVE files with the formats PCM (8, 16, 24, and
 * 32 bits), IEEE Float (32 bits), Microsoft ADPCM and IMA ADPCM (4 bits), and
 * A-law and mu-law (8 bits). Other formats are currently unsupported and
 * cause an error.
 *
 * If this function succeeds, the return value is zero and the pointer to the
 * audio data allocated by the function is written to `audio_buf` and its
 * length in bytes to `audio_len`. The SDL_AudioSpec members `freq`,
 * `channels`, and `format` are set to the values of the audio data in the
 * buffer.
 *
 * It's necessary to use SDL_free() to free the audio data returned in
 * `audio_buf` when it is no longer used.
 *
 * Because of the underspecification of the .WAV format, there are many
 * problematic files in the wild that cause issues with strict decoders. To
 * provide compatibility with these files, this decoder is lenient in regards
 * to the truncation of the file, the fact chunk, and the size of the RIFF
 * chunk. The hints `SDL_HINT_WAVE_RIFF_CHUNK_SIZE`,
 * `SDL_HINT_WAVE_TRUNCATION`, and `SDL_HINT_WAVE_FACT_CHUNK` can be used to
 * tune the behavior of the loading process.
 *
 * Any file that is invalid (due to truncation, corruption, or wrong values in
 * the headers), too big, or unsupported causes an error. Additionally, any
 * critical I/O error from the data source will terminate the loading process
 * with an error. The function returns NULL on error and in all cases (with
 * the exception of `src` being NULL), an appropriate error message will be
 * set.
 *
 * It is required that the data source supports seeking.
 *
 * Example:
 *
 * ```c
 * SDL_LoadWAV_IO(SDL_IOFromFile("sample.wav", "rb"), true, &spec, &buf, &len);
 * ```
 *
 * Note that the SDL_LoadWAV function does this same thing for you, but in a
 * less messy way:
 *
 * ```c
 * SDL_LoadWAV("sample.wav", &spec, &buf, &len);
 * ```
 *
 * \param src the data source for the WAVE data.
 * \param closeio if true, calls SDL_CloseIO() on `src` before returning, even
 *                in the case of an error.
 * \param spec a pointer to an SDL_AudioSpec that will be set to the WAVE
 *             data's format details on successful return.
 * \param audio_buf a pointer filled with the audio data, allocated by the
 *                  function.
 * \param audio_len a pointer filled with the length of the audio data buffer
 *                  in bytes.
 * \returns true on success. `audio_buf` will be filled with a pointer to an
 *          allocated buffer containing the audio data, and `audio_len` is
 *          filled with the length of that audio buffer in bytes.
 *
 *          This function returns false if the .WAV file cannot be opened,
 *          uses an unknown data format, or is corrupt; call SDL_GetError()
 *          for more information.
 *
 *          When the application is done with the data returned in
 *          `audio_buf`, it should call SDL_free() to dispose of it.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_free
 * \sa SDL_LoadWAV
 */
extern SDL_DECLSPEC bool SDLCALL SDL_LoadWAV_IO(SDL_IOStream *src, bool closeio, SDL_AudioSpec *spec, Uint8 **audio_buf, Uint32 *audio_len);

/**
 * Loads a WAV from a file path.
 *
 * This is a convenience function that is effectively the same as:
 *
 * ```c
 * SDL_LoadWAV_IO(SDL_IOFromFile(path, "rb"), true, spec, audio_buf, audio_len);
 * ```
 *
 * \param path the file path of the WAV file to open.
 * \param spec a pointer to an SDL_AudioSpec that will be set to the WAVE
 *             data's format details on successful return.
 * \param audio_buf a pointer filled with the audio data, allocated by the
 *                  function.
 * \param audio_len a pointer filled with the length of the audio data buffer
 *                  in bytes.
 * \returns true on success. `audio_buf` will be filled with a pointer to an
 *          allocated buffer containing the audio data, and `audio_len` is
 *          filled with the length of that audio buffer in bytes.
 *
 *          This function returns false if the .WAV file cannot be opened,
 *          uses an unknown data format, or is corrupt; call SDL_GetError()
 *          for more information.
 *
 *          When the application is done with the data returned in
 *          `audio_buf`, it should call SDL_free() to dispose of it.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_free
 * \sa SDL_LoadWAV_IO
 */
extern SDL_DECLSPEC bool SDLCALL SDL_LoadWAV(const char *path, SDL_AudioSpec *spec, Uint8 **audio_buf, Uint32 *audio_len);

/**
 * Mix audio data in a specified format.
 *
 * This takes an audio buffer `src` of `len` bytes of `format` data and mixes
 * it into `dst`, performing addition, volume adjustment, and overflow
 * clipping. The buffer pointed to by `dst` must also be `len` bytes of
 * `format` data.
 *
 * This is provided for convenience -- you can mix your own audio data.
 *
 * Do not use this function for mixing together more than two streams of
 * sample data. The output from repeated application of this function may be
 * distorted by clipping, because there is no accumulator with greater range
 * than the input (not to mention this being an inefficient way of doing it).
 *
 * It is a common misconception that this function is required to write audio
 * data to an output stream in an audio callback. While you can do that,
 * SDL_MixAudio() is really only needed when you're mixing a single audio
 * stream with a volume adjustment.
 *
 * \param dst the destination for the mixed audio.
 * \param src the source audio buffer to be mixed.
 * \param format the SDL_AudioFormat structure representing the desired audio
 *               format.
 * \param len the length of the audio buffer in bytes.
 * \param volume ranges from 0.0 - 1.0, and should be set to 1.0 for full
 *               audio volume.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_MixAudio(Uint8 *dst, const Uint8 *src, SDL_AudioFormat format, Uint32 len, float volume);

/**
 * Convert some audio data of one format to another format.
 *
 * Please note that this function is for convenience, but should not be used
 * to resample audio in blocks, as it will introduce audio artifacts on the
 * boundaries. You should only use this function if you are converting audio
 * data in its entirety in one call. If you want to convert audio in smaller
 * chunks, use an SDL_AudioStream, which is designed for this situation.
 *
 * Internally, this function creates and destroys an SDL_AudioStream on each
 * use, so it's also less efficient than using one directly, if you need to
 * convert multiple times.
 *
 * \param src_spec the format details of the input audio.
 * \param src_data the audio data to be converted.
 * \param src_len the len of src_data.
 * \param dst_spec the format details of the output audio.
 * \param dst_data will be filled with a pointer to converted audio data,
 *                 which should be freed with SDL_free(). On error, it will be
 *                 NULL.
 * \param dst_len will be filled with the len of dst_data.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_ConvertAudioSamples(const SDL_AudioSpec *src_spec, const Uint8 *src_data, int src_len, const SDL_AudioSpec *dst_spec, Uint8 **dst_data, int *dst_len);

/**
 * Get the human readable name of an audio format.
 *
 * \param format the audio format to query.
 * \returns the human readable name of the specified audio format or
 *          "SDL_AUDIO_UNKNOWN" if the format isn't recognized.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetAudioFormatName(SDL_AudioFormat format);

/**
 * Get the appropriate memset value for silencing an audio format.
 *
 * The value returned by this function can be used as the second argument to
 * memset (or SDL_memset) to set an audio buffer in a specific format to
 * silence.
 *
 * \param format the audio data format to query.
 * \returns a byte value that can be passed to memset.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetSilenceValueForFormat(SDL_AudioFormat format);


/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_audio_h_ */

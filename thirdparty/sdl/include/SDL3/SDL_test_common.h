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
 *  Common functions of SDL test framework.
 *
 *  This code is a part of the SDL test library, not the main SDL library.
 */

/* Ported from original test/common.h file. */

#ifndef SDL_test_common_h_
#define SDL_test_common_h_

#include <SDL3/SDL.h>

#ifdef SDL_PLATFORM_PSP
#define DEFAULT_WINDOW_WIDTH  480
#define DEFAULT_WINDOW_HEIGHT 272
#elif defined(SDL_PLATFORM_VITA)
#define DEFAULT_WINDOW_WIDTH  960
#define DEFAULT_WINDOW_HEIGHT 544
#else
#define DEFAULT_WINDOW_WIDTH  640
#define DEFAULT_WINDOW_HEIGHT 480
#endif

typedef Uint32 SDLTest_VerboseFlags;
#define VERBOSE_VIDEO   0x00000001
#define VERBOSE_MODES   0x00000002
#define VERBOSE_RENDER  0x00000004
#define VERBOSE_EVENT   0x00000008
#define VERBOSE_AUDIO   0x00000010
#define VERBOSE_MOTION  0x00000020

/* !< Function pointer parsing one argument at argv[index], returning the number of parsed arguments,
 *    or a negative value when the argument is invalid */
typedef int (SDLCALL *SDLTest_ParseArgumentsFp)(void *data, char **argv, int index);

/* !< Finalize the argument parser. */
typedef void (SDLCALL *SDLTest_FinalizeArgumentParserFp)(void *arg);

typedef struct SDLTest_ArgumentParser
{
    /* !< Parse an argument. */
    SDLTest_ParseArgumentsFp parse_arguments;
    /* !< Finalize this argument parser. Called once before parsing the first argument. */
    SDLTest_FinalizeArgumentParserFp finalize;
    /* !< Null-terminated array of arguments. Printed when running with --help. */
    const char **usage;
    /* !< User data, passed to all callbacks. */
    void *data;
    /* !< Next argument parser. */
    struct SDLTest_ArgumentParser *next;
} SDLTest_ArgumentParser;

typedef struct
{
    /* SDL init flags */
    char **argv;
    SDL_InitFlags flags;
    SDLTest_VerboseFlags verbose;

    /* Video info */
    const char *videodriver;
    int display_index;
    SDL_DisplayID displayID;
    const char *window_title;
    const char *window_icon;
    SDL_WindowFlags window_flags;
    bool flash_on_focus_loss;
    int window_x;
    int window_y;
    int window_w;
    int window_h;
    int window_minW;
    int window_minH;
    int window_maxW;
    int window_maxH;
    float window_min_aspect;
    float window_max_aspect;
    int logical_w;
    int logical_h;
    bool auto_scale_content;
    SDL_RendererLogicalPresentation logical_presentation;
    float scale;
    int depth;
    float refresh_rate;
    bool fill_usable_bounds;
    bool fullscreen_exclusive;
    SDL_DisplayMode fullscreen_mode;
    int num_windows;
    SDL_Window **windows;
    const char *gpudriver;

    /* Renderer info */
    const char *renderdriver;
    int render_vsync;
    bool skip_renderer;
    SDL_Renderer **renderers;
    SDL_Texture **targets;

    /* Audio info */
    const char *audiodriver;
    SDL_AudioFormat audio_format;
    int audio_channels;
    int audio_freq;
    SDL_AudioDeviceID audio_id;

    /* GL settings */
    int gl_red_size;
    int gl_green_size;
    int gl_blue_size;
    int gl_alpha_size;
    int gl_buffer_size;
    int gl_depth_size;
    int gl_stencil_size;
    int gl_double_buffer;
    int gl_accum_red_size;
    int gl_accum_green_size;
    int gl_accum_blue_size;
    int gl_accum_alpha_size;
    int gl_stereo;
    int gl_release_behavior;
    int gl_multisamplebuffers;
    int gl_multisamplesamples;
    int gl_retained_backing;
    int gl_accelerated;
    int gl_major_version;
    int gl_minor_version;
    int gl_debug;
    int gl_profile_mask;

    /* Mouse info */
    SDL_Rect confine;
    bool hide_cursor;

    /* Options info */
    SDLTest_ArgumentParser common_argparser;
    SDLTest_ArgumentParser video_argparser;
    SDLTest_ArgumentParser audio_argparser;

    SDLTest_ArgumentParser *argparser;
} SDLTest_CommonState;

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* Function prototypes */

/**
 * Parse command line parameters and create common state.
 *
 * \param argv Array of command line parameters
 * \param flags Flags indicating which subsystem to initialize (i.e. SDL_INIT_VIDEO | SDL_INIT_AUDIO)
 *
 * \returns a newly allocated common state object.
 */
SDLTest_CommonState * SDLCALL SDLTest_CommonCreateState(char **argv, SDL_InitFlags flags);

/**
 * Free the common state object.
 *
 * You should call SDL_Quit() before calling this function.
 *
 * \param state The common state object to destroy
 */
void SDLCALL SDLTest_CommonDestroyState(SDLTest_CommonState *state);

/**
 * Process one common argument.
 *
 * \param state The common state describing the test window to create.
 * \param index The index of the argument to process in argv[].
 *
 * \returns the number of arguments processed (i.e. 1 for --fullscreen, 2 for --video [videodriver], or -1 on error.
 */
int SDLCALL SDLTest_CommonArg(SDLTest_CommonState *state, int index);


/**
 * Logs command line usage info.
 *
 * This logs the appropriate command line options for the subsystems in use
 *  plus other common options, and then any application-specific options.
 *  This uses the SDL_Log() function and splits up output to be friendly to
 *  80-character-wide terminals.
 *
 * \param state The common state describing the test window for the app.
 * \param argv0 argv[0], as passed to main/SDL_main.
 * \param options an array of strings for application specific options. The last element of the array should be NULL.
 */
void SDLCALL SDLTest_CommonLogUsage(SDLTest_CommonState *state, const char *argv0, const char **options);

/**
 * Open test window.
 *
 * \param state The common state describing the test window to create.
 *
 * \returns true if initialization succeeded, false otherwise
 */
bool SDLCALL SDLTest_CommonInit(SDLTest_CommonState *state);

/**
 * Easy argument handling when test app doesn't need any custom args.
 *
 * \param state The common state describing the test window to create.
 * \param argc argc, as supplied to SDL_main
 * \param argv argv, as supplied to SDL_main
 *
 * \returns false if app should quit, true otherwise.
 */
bool SDLCALL SDLTest_CommonDefaultArgs(SDLTest_CommonState *state, int argc, char **argv);

/**
 * Print the details of an event.
 *
 * This is automatically called by SDLTest_CommonEvent() as needed.
 *
 * \param event The event to print.
 */
void SDLCALL SDLTest_PrintEvent(const SDL_Event *event);

/**
 * Common event handler for test windows if you use a standard SDL_main.
 *
 * \param state The common state used to create test window.
 * \param event The event to handle.
 * \param done Flag indicating we are done.
 */
void SDLCALL SDLTest_CommonEvent(SDLTest_CommonState *state, SDL_Event *event, int *done);

/**
 * Common event handler for test windows if you use SDL_AppEvent.
 *
 * This does _not_ free anything in `event`.
 *
 * \param state The common state used to create test window.
 * \param event The event to handle.
 * \returns Value suitable for returning from SDL_AppEvent().
 */
SDL_AppResult SDLCALL SDLTest_CommonEventMainCallbacks(SDLTest_CommonState *state, const SDL_Event *event);

/**
 * Close test window.
 *
 * \param state The common state used to create test window.
 *
 */
void SDLCALL SDLTest_CommonQuit(SDLTest_CommonState *state);

/**
 * Draws various window information (position, size, etc.) to the renderer.
 *
 * \param renderer The renderer to draw to.
 * \param window The window whose information should be displayed.
 * \param usedHeight Returns the height used, so the caller can draw more below.
 *
 */
void SDLCALL SDLTest_CommonDrawWindowInfo(SDL_Renderer *renderer, SDL_Window *window, float *usedHeight);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_test_common_h_ */

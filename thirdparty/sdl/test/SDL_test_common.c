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

/* Ported from original test/common.c file. */
#include <SDL3/SDL_test.h>

#define SDL_MAIN_NOIMPL
#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL_main.h>

static const char *common_usage[] = {
    "[-h | --help]",
    "[--trackmem]",
    "[--randmem]",
    "[--info all|video|modes|render|event|event_motion]",
    "[--log all|error|system|audio|video|render|input]",
    NULL
};

static const char *video_usage[] = {
    "[--always-on-top]",
    "[--aspect min-max]",
    "[--auto-scale-content]",
    "[--center | --position X,Y]",
    "[--confine-cursor X,Y,W,H]",
    "[--depth N]",
    "[--display N]",
    "[--flash-on-focus-loss]",
    "[--fullscreen | --fullscreen-desktop | --windows N]",
    "[--geometry WxH]",
    "[--gldebug]",
    "[--grab]",
    "[--hidden]",
    "[--hide-cursor]",
    "[--high-pixel-density]",
    "[--icon icon.bmp]",
    "[--input-focus]",
    "[--keyboard-grab]",
    "[--logical-presentation disabled|match|stretch|letterbox|overscan|integer_scale]",
    "[--logical WxH]",
    "[--max-geometry WxH]",
    "[--maximize]",
    "[--metal-window | --opengl-window | --vulkan-window]",
    "[--min-geometry WxH]",
    "[--minimize]",
    "[--mouse-focus]",
    "[--noframe]",
    "[--refresh R]",
    "[--renderer driver]",
    "[--resizable]",
    "[--scale N]",
    "[--title title]",
    "[--transparent]",
    "[--usable-bounds]",
    "[--utility]",
    "[--video driver]",
    "[--gpu driver]",
    "[--vsync]",
    NULL
};

/* !!! FIXME: Float32? Sint32? */
static const char *audio_usage[] = {
    "[--audio driver]",
    "[--rate N]",
    "[--format U8|S8|S16|S16LE|S16BE|S32|S32LE|S32BE|F32|F32LE|F32BE]",
    "[--channels N]",
    NULL
};

static void SDL_snprintfcat(SDL_OUT_Z_CAP(maxlen) char *text, size_t maxlen, SDL_PRINTF_FORMAT_STRING const char *fmt, ...)
{
    size_t length = SDL_strlen(text);
    va_list ap;

    va_start(ap, fmt);
    text += length;
    maxlen -= length;
    (void)SDL_vsnprintf(text, maxlen, fmt, ap);
    va_end(ap);
}

static void SDLCALL SDLTest_CommonArgParserFinalize(void *data)
{
    SDLTest_CommonState *state = data;

    if (!(state->flags & SDL_INIT_VIDEO)) {
        state->video_argparser.usage = NULL;
    }
    if (!(state->flags & SDL_INIT_AUDIO)) {
        state->audio_argparser.usage = NULL;
    }
}

#define SEARCHARG(dim)                  \
    while (*(dim) && *(dim) != ',') {   \
        ++(dim);                        \
    }                                   \
    if (!*(dim)) {                      \
        return -1;                      \
    }                                   \
    *(dim)++ = '\0';

static int SDLCALL SDLTest_CommonStateParseCommonArguments(void *data, char **argv, int index)
{
    SDLTest_CommonState *state = data;

    if ((SDL_strcasecmp(argv[index], "-h") == 0) || (SDL_strcasecmp(argv[index], "--help") == 0)) {
        /* Print the usage message */
        return -1;
    }
    if (SDL_strcasecmp(argv[index], "--trackmem") == 0) {
        /* Already handled in SDLTest_CommonCreateState() */
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--randmem") == 0) {
        /* Already handled in SDLTest_CommonCreateState() */
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--log") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        if (SDL_strcasecmp(argv[index], "all") == 0) {
            SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "system") == 0) {
            SDL_SetLogPriority(SDL_LOG_CATEGORY_SYSTEM, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "audio") == 0) {
            SDL_SetLogPriority(SDL_LOG_CATEGORY_AUDIO, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "video") == 0) {
            SDL_SetLogPriority(SDL_LOG_CATEGORY_VIDEO, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "render") == 0) {
            SDL_SetLogPriority(SDL_LOG_CATEGORY_RENDER, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "input") == 0) {
            SDL_SetLogPriority(SDL_LOG_CATEGORY_INPUT, SDL_LOG_PRIORITY_VERBOSE);
            return 2;
        }
        return -1;
    }

    if (SDL_strcasecmp(argv[index], "--info") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        if (SDL_strcasecmp(argv[index], "all") == 0) {
            state->verbose |=
                (VERBOSE_VIDEO | VERBOSE_MODES | VERBOSE_RENDER |
                 VERBOSE_EVENT);
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "video") == 0) {
            state->verbose |= VERBOSE_VIDEO;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "modes") == 0) {
            state->verbose |= VERBOSE_MODES;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "render") == 0) {
            state->verbose |= VERBOSE_RENDER;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "event") == 0) {
            state->verbose |= VERBOSE_EVENT;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "event_motion") == 0) {
            state->verbose |= (VERBOSE_EVENT | VERBOSE_MOTION);
            return 2;
        }
        return -1;
    }
    if (SDL_strcmp(argv[index], "-NSDocumentRevisionsDebugMode") == 0) {
        /* Debug flag sent by Xcode */
        return 2;
    }
    return 0;
}

static int SDLCALL SDLTest_CommonStateParseVideoArguments(void *data, char **argv, int index)
{
    SDLTest_CommonState *state = data;

    if (!(state->flags & SDL_INIT_VIDEO)) {
        return 0;
    }

    if (SDL_strcasecmp(argv[index], "--video") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->videodriver = argv[index];
        SDL_SetHint(SDL_HINT_VIDEO_DRIVER, state->videodriver);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--renderer") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->renderdriver = argv[index];
        SDL_SetHint(SDL_HINT_RENDER_DRIVER, state->renderdriver);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--gldebug") == 0) {
        state->gl_debug = 1;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--display") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->display_index = SDL_atoi(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--metal-window") == 0) {
        state->window_flags |= SDL_WINDOW_METAL;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--opengl-window") == 0) {
        state->window_flags |= SDL_WINDOW_OPENGL;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--vulkan-window") == 0) {
        state->window_flags |= SDL_WINDOW_VULKAN;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--fullscreen") == 0) {
        state->window_flags |= SDL_WINDOW_FULLSCREEN;
        state->fullscreen_exclusive = true;
        state->num_windows = 1;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--fullscreen-desktop") == 0) {
        state->window_flags |= SDL_WINDOW_FULLSCREEN;
        state->fullscreen_exclusive = false;
        state->num_windows = 1;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--windows") == 0) {
        ++index;
        if (!argv[index] || !SDL_isdigit((unsigned char) *argv[index])) {
            return -1;
        }
        if (!(state->window_flags & SDL_WINDOW_FULLSCREEN)) {
            state->num_windows = SDL_atoi(argv[index]);
        }
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--title") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->window_title = argv[index];
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--icon") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->window_icon = argv[index];
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--center") == 0) {
        state->window_x = SDL_WINDOWPOS_CENTERED;
        state->window_y = SDL_WINDOWPOS_CENTERED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--position") == 0) {
        char *x, *y;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        x = argv[index];
        y = argv[index];
        while (*y && *y != ',') {
            ++y;
        }
        if (!*y) {
            return -1;
        }
        *y++ = '\0';
        state->window_x = SDL_atoi(x);
        state->window_y = SDL_atoi(y);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--confine-cursor") == 0) {
        char *x, *y, *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        x = argv[index];
        y = argv[index];
        SEARCHARG(y)
        w = y;
        SEARCHARG(w)
        h = w;
        SEARCHARG(h)
        state->confine.x = SDL_atoi(x);
        state->confine.y = SDL_atoi(y);
        state->confine.w = SDL_atoi(w);
        state->confine.h = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--usable-bounds") == 0) {
        state->fill_usable_bounds = true;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--geometry") == 0) {
        char *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        w = argv[index];
        h = argv[index];
        while (*h && *h != 'x') {
            ++h;
        }
        if (!*h) {
            return -1;
        }
        *h++ = '\0';
        state->window_w = SDL_atoi(w);
        state->window_h = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--min-geometry") == 0) {
        char *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        w = argv[index];
        h = argv[index];
        while (*h && *h != 'x') {
            ++h;
        }
        if (!*h) {
            return -1;
        }
        *h++ = '\0';
        state->window_minW = SDL_atoi(w);
        state->window_minH = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--max-geometry") == 0) {
        char *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        w = argv[index];
        h = argv[index];
        while (*h && *h != 'x') {
            ++h;
        }
        if (!*h) {
            return -1;
        }
        *h++ = '\0';
        state->window_maxW = SDL_atoi(w);
        state->window_maxH = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--aspect") == 0) {
        char *min_aspect, *max_aspect;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        min_aspect = argv[index];
        max_aspect = argv[index];
        while (*max_aspect && *max_aspect != '-') {
            ++max_aspect;
        }
        if (*max_aspect) {
            *max_aspect++ = '\0';
        } else {
            max_aspect = min_aspect;
        }
        state->window_min_aspect = (float)SDL_atof(min_aspect);
        state->window_max_aspect = (float)SDL_atof(max_aspect);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--logical") == 0) {
        char *w, *h;
        ++index;
        if (!argv[index]) {
            return -1;
        }
        w = argv[index];
        h = argv[index];
        while (*h && *h != 'x') {
            ++h;
        }
        if (!*h) {
            return -1;
        }
        *h++ = '\0';
        state->logical_w = SDL_atoi(w);
        state->logical_h = SDL_atoi(h);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--high-pixel-density") == 0) {
        state->window_flags |= SDL_WINDOW_HIGH_PIXEL_DENSITY;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--auto-scale-content") == 0) {
        state->auto_scale_content = true;

        if (state->logical_presentation == SDL_LOGICAL_PRESENTATION_DISABLED) {
            state->logical_presentation = SDL_LOGICAL_PRESENTATION_STRETCH;
        }
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--logical-presentation") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        if (SDL_strcasecmp(argv[index], "disabled") == 0) {
            state->logical_presentation = SDL_LOGICAL_PRESENTATION_DISABLED;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "stretch") == 0) {
            state->logical_presentation = SDL_LOGICAL_PRESENTATION_STRETCH;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "letterbox") == 0) {
            state->logical_presentation = SDL_LOGICAL_PRESENTATION_LETTERBOX;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "overscan") == 0) {
            state->logical_presentation = SDL_LOGICAL_PRESENTATION_OVERSCAN;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "integer_scale") == 0) {
            state->logical_presentation = SDL_LOGICAL_PRESENTATION_INTEGER_SCALE;
            return 2;
        }
        return -1;
    }
    if (SDL_strcasecmp(argv[index], "--scale") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->scale = (float) SDL_atof(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--depth") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->depth = SDL_atoi(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--refresh") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->refresh_rate = (float) SDL_atof(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--vsync") == 0) {
        state->render_vsync = 1;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--noframe") == 0) {
        state->window_flags |= SDL_WINDOW_BORDERLESS;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--resizable") == 0) {
        state->window_flags |= SDL_WINDOW_RESIZABLE;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--transparent") == 0) {
        state->window_flags |= SDL_WINDOW_TRANSPARENT;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--always-on-top") == 0) {
        state->window_flags |= SDL_WINDOW_ALWAYS_ON_TOP;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--minimize") == 0) {
        state->window_flags |= SDL_WINDOW_MINIMIZED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--maximize") == 0) {
        state->window_flags |= SDL_WINDOW_MAXIMIZED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--hidden") == 0) {
        state->window_flags |= SDL_WINDOW_HIDDEN;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--input-focus") == 0) {
        state->window_flags |= SDL_WINDOW_INPUT_FOCUS;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--mouse-focus") == 0) {
        state->window_flags |= SDL_WINDOW_MOUSE_FOCUS;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--flash-on-focus-loss") == 0) {
        state->flash_on_focus_loss = true;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--grab") == 0) {
        state->window_flags |= SDL_WINDOW_MOUSE_GRABBED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--keyboard-grab") == 0) {
        state->window_flags |= SDL_WINDOW_KEYBOARD_GRABBED;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--utility") == 0) {
        state->window_flags |= SDL_WINDOW_UTILITY;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--hide-cursor") == 0) {
        state->hide_cursor = true;
        return 1;
    }
    if (SDL_strcasecmp(argv[index], "--gpu") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->gpudriver = argv[index];
        SDL_SetHint(SDL_HINT_GPU_DRIVER, state->gpudriver);
        return 2;
    }
    return 0;
}

static int SDLCALL SDLTest_CommonStateParseAudioArguments(void *data, char **argv, int index)
{
    SDLTest_CommonState *state = data;

    if (!(state->flags & SDL_INIT_AUDIO)) {
        return 0;
    }
    if (SDL_strcasecmp(argv[index], "--audio") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->audiodriver = argv[index];
        SDL_SetHint(SDL_HINT_AUDIO_DRIVER, state->audiodriver);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--rate") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->audio_freq = SDL_atoi(argv[index]);
        return 2;
    }
    if (SDL_strcasecmp(argv[index], "--format") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        if (SDL_strcasecmp(argv[index], "U8") == 0) {
            state->audio_format = SDL_AUDIO_U8;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S8") == 0) {
            state->audio_format = SDL_AUDIO_S8;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S16") == 0) {
            state->audio_format = SDL_AUDIO_S16;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S16LE") == 0) {
            state->audio_format = SDL_AUDIO_S16LE;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S16BE") == 0) {
            state->audio_format = SDL_AUDIO_S16BE;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S32") == 0) {
            state->audio_format = SDL_AUDIO_S32;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S32LE") == 0) {
            state->audio_format = SDL_AUDIO_S32LE;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "S32BE") == 0) {
            state->audio_format = SDL_AUDIO_S32BE;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "F32") == 0) {
            state->audio_format = SDL_AUDIO_F32;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "F32LE") == 0) {
            state->audio_format = SDL_AUDIO_F32LE;
            return 2;
        }
        if (SDL_strcasecmp(argv[index], "F32BE") == 0) {
            state->audio_format = SDL_AUDIO_F32BE;
            return 2;
        }
        return -1;
    }
    if (SDL_strcasecmp(argv[index], "--channels") == 0) {
        ++index;
        if (!argv[index]) {
            return -1;
        }
        state->audio_channels = (Uint8) SDL_atoi(argv[index]);
        return 2;
    }
    return 0;
}

SDLTest_CommonState *SDLTest_CommonCreateState(char **argv, SDL_InitFlags flags)
{
    int i;
    SDLTest_CommonState *state;

    /* Do this first so we catch all allocations */
    for (i = 1; argv[i]; ++i) {
        if (SDL_strcasecmp(argv[i], "--trackmem") == 0) {
            SDLTest_TrackAllocations();
        } else if (SDL_strcasecmp(argv[i], "--randmem") == 0) {
            SDLTest_RandFillAllocations();
        }
    }

    state = (SDLTest_CommonState *)SDL_calloc(1, sizeof(*state));
    if (!state) {
        return NULL;
    }

    /* Initialize some defaults */
    state->argv = argv;
    state->flags = flags;
    state->window_title = argv[0];
    state->window_flags = SDL_WINDOW_HIDDEN;
    state->window_x = SDL_WINDOWPOS_UNDEFINED;
    state->window_y = SDL_WINDOWPOS_UNDEFINED;
    state->window_w = DEFAULT_WINDOW_WIDTH;
    state->window_h = DEFAULT_WINDOW_HEIGHT;
    state->logical_presentation = SDL_LOGICAL_PRESENTATION_DISABLED;
    state->num_windows = 1;
    state->audio_freq = 22050;
    state->audio_format = SDL_AUDIO_S16;
    state->audio_channels = 2;

    /* Set some very sane GL defaults */
    state->gl_red_size = 8;
    state->gl_green_size = 8;
    state->gl_blue_size = 8;
    state->gl_alpha_size = 8;
    state->gl_buffer_size = 0;
    state->gl_depth_size = 16;
    state->gl_stencil_size = 0;
    state->gl_double_buffer = 1;
    state->gl_accum_red_size = 0;
    state->gl_accum_green_size = 0;
    state->gl_accum_blue_size = 0;
    state->gl_accum_alpha_size = 0;
    state->gl_stereo = 0;
    state->gl_multisamplebuffers = 0;
    state->gl_multisamplesamples = 0;
    state->gl_retained_backing = 1;
    state->gl_accelerated = -1;
    state->gl_debug = 0;

    state->common_argparser.parse_arguments = SDLTest_CommonStateParseCommonArguments;
    state->common_argparser.finalize = SDLTest_CommonArgParserFinalize;
    state->common_argparser.usage = common_usage;
    state->common_argparser.data = state;
    state->common_argparser.next = &state->video_argparser;

    state->video_argparser.parse_arguments = SDLTest_CommonStateParseVideoArguments;
    state->video_argparser.finalize = NULL;
    state->video_argparser.usage = video_usage;
    state->video_argparser.data = state;
    state->video_argparser.next = &state->audio_argparser;

    state->audio_argparser.parse_arguments = SDLTest_CommonStateParseAudioArguments;
    state->audio_argparser.finalize = NULL;
    state->audio_argparser.usage = audio_usage;
    state->audio_argparser.data = state;

    state->argparser = &state->common_argparser;

    return state;
}

void SDLTest_CommonDestroyState(SDLTest_CommonState *state) {
    SDL_free(state);
    SDLTest_LogAllocations();
}

int SDLTest_CommonArg(SDLTest_CommonState *state, int index)
{
    SDLTest_ArgumentParser *argparser = state->argparser;

    /* Go back and parse arguments as we go */
    while (argparser) {
        if (argparser->parse_arguments) {
            int consumed = argparser->parse_arguments(argparser->data, state->argv, index);
            if (consumed != 0) {
                return consumed;
            }
        }
        argparser = argparser->next;
    }
    return 0;
}

void SDLTest_CommonLogUsage(SDLTest_CommonState *state, const char *argv0, const char **options)
{
    SDLTest_ArgumentParser *argparser;

    SDL_Log("USAGE: %s", argv0);

    for (argparser = state->argparser; argparser; argparser = argparser->next) {
        if (argparser->finalize) {
            argparser->finalize(argparser->data);
        }
        if (argparser->usage) {
            int i;
            for (i = 0; argparser->usage[i] != NULL; i++) {
                SDL_Log("    %s", argparser->usage[i]);
            }
        }
    }
    if (options) {
        int i;
        for (i = 0; options[i] != NULL; i++) {
            SDL_Log("    %s", options[i]);
        }
    }
}

bool SDLTest_CommonDefaultArgs(SDLTest_CommonState *state, int argc, char **argv)
{
    int i = 1;
    while (i < argc) {
        const int consumed = SDLTest_CommonArg(state, i);
        if (consumed <= 0) {
            SDLTest_CommonLogUsage(state, argv[0], NULL);
            return false;
        }
        i += consumed;
    }
    return true;
}

static void SDLTest_PrintDisplayOrientation(char *text, size_t maxlen, SDL_DisplayOrientation orientation)
{
    switch (orientation) {
    case SDL_ORIENTATION_UNKNOWN:
        SDL_snprintfcat(text, maxlen, "UNKNOWN");
        break;
    case SDL_ORIENTATION_LANDSCAPE:
        SDL_snprintfcat(text, maxlen, "LANDSCAPE");
        break;
    case SDL_ORIENTATION_LANDSCAPE_FLIPPED:
        SDL_snprintfcat(text, maxlen, "LANDSCAPE_FLIPPED");
        break;
    case SDL_ORIENTATION_PORTRAIT:
        SDL_snprintfcat(text, maxlen, "PORTRAIT");
        break;
    case SDL_ORIENTATION_PORTRAIT_FLIPPED:
        SDL_snprintfcat(text, maxlen, "PORTRAIT_FLIPPED");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%8.8x", orientation);
        break;
    }
}

static void SDLTest_PrintWindowFlag(char *text, size_t maxlen, SDL_WindowFlags flag)
{
    switch (flag) {
    case SDL_WINDOW_FULLSCREEN:
        SDL_snprintfcat(text, maxlen, "FULLSCREEN");
        break;
    case SDL_WINDOW_OPENGL:
        SDL_snprintfcat(text, maxlen, "OPENGL");
        break;
    case SDL_WINDOW_OCCLUDED:
        SDL_snprintfcat(text, maxlen, "OCCLUDED");
        break;
    case SDL_WINDOW_HIDDEN:
        SDL_snprintfcat(text, maxlen, "HIDDEN");
        break;
    case SDL_WINDOW_BORDERLESS:
        SDL_snprintfcat(text, maxlen, "BORDERLESS");
        break;
    case SDL_WINDOW_RESIZABLE:
        SDL_snprintfcat(text, maxlen, "RESIZABLE");
        break;
    case SDL_WINDOW_MINIMIZED:
        SDL_snprintfcat(text, maxlen, "MINIMIZED");
        break;
    case SDL_WINDOW_MAXIMIZED:
        SDL_snprintfcat(text, maxlen, "MAXIMIZED");
        break;
    case SDL_WINDOW_MOUSE_GRABBED:
        SDL_snprintfcat(text, maxlen, "MOUSE_GRABBED");
        break;
    case SDL_WINDOW_INPUT_FOCUS:
        SDL_snprintfcat(text, maxlen, "INPUT_FOCUS");
        break;
    case SDL_WINDOW_MOUSE_FOCUS:
        SDL_snprintfcat(text, maxlen, "MOUSE_FOCUS");
        break;
    case SDL_WINDOW_EXTERNAL:
        SDL_snprintfcat(text, maxlen, "EXTERNAL");
        break;
    case SDL_WINDOW_MODAL:
        SDL_snprintfcat(text, maxlen, "MODAL");
        break;
    case SDL_WINDOW_HIGH_PIXEL_DENSITY:
        SDL_snprintfcat(text, maxlen, "HIGH_PIXEL_DENSITY");
        break;
    case SDL_WINDOW_MOUSE_CAPTURE:
        SDL_snprintfcat(text, maxlen, "MOUSE_CAPTURE");
        break;
    case SDL_WINDOW_MOUSE_RELATIVE_MODE:
        SDL_snprintfcat(text, maxlen, "MOUSE_RELATIVE_MODE");
        break;
    case SDL_WINDOW_ALWAYS_ON_TOP:
        SDL_snprintfcat(text, maxlen, "ALWAYS_ON_TOP");
        break;
    case SDL_WINDOW_UTILITY:
        SDL_snprintfcat(text, maxlen, "UTILITY");
        break;
    case SDL_WINDOW_TOOLTIP:
        SDL_snprintfcat(text, maxlen, "TOOLTIP");
        break;
    case SDL_WINDOW_POPUP_MENU:
        SDL_snprintfcat(text, maxlen, "POPUP_MENU");
        break;
    case SDL_WINDOW_KEYBOARD_GRABBED:
        SDL_snprintfcat(text, maxlen, "KEYBOARD_GRABBED");
        break;
    case SDL_WINDOW_VULKAN:
        SDL_snprintfcat(text, maxlen, "VULKAN");
        break;
    case SDL_WINDOW_METAL:
        SDL_snprintfcat(text, maxlen, "METAL");
        break;
    case SDL_WINDOW_TRANSPARENT:
        SDL_snprintfcat(text, maxlen, "TRANSPARENT");
        break;
    case SDL_WINDOW_NOT_FOCUSABLE:
        SDL_snprintfcat(text, maxlen, "NOT_FOCUSABLE");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%16.16" SDL_PRIx64, flag);
        break;
    }
}

static void SDLTest_PrintWindowFlags(char *text, size_t maxlen, SDL_WindowFlags flags)
{
    const SDL_WindowFlags window_flags[] = {
        SDL_WINDOW_FULLSCREEN,
        SDL_WINDOW_OPENGL,
        SDL_WINDOW_OCCLUDED,
        SDL_WINDOW_HIDDEN,
        SDL_WINDOW_BORDERLESS,
        SDL_WINDOW_RESIZABLE,
        SDL_WINDOW_MINIMIZED,
        SDL_WINDOW_MAXIMIZED,
        SDL_WINDOW_MOUSE_GRABBED,
        SDL_WINDOW_INPUT_FOCUS,
        SDL_WINDOW_MOUSE_FOCUS,
        SDL_WINDOW_EXTERNAL,
        SDL_WINDOW_MODAL,
        SDL_WINDOW_HIGH_PIXEL_DENSITY,
        SDL_WINDOW_MOUSE_CAPTURE,
        SDL_WINDOW_MOUSE_RELATIVE_MODE,
        SDL_WINDOW_ALWAYS_ON_TOP,
        SDL_WINDOW_UTILITY,
        SDL_WINDOW_TOOLTIP,
        SDL_WINDOW_POPUP_MENU,
        SDL_WINDOW_KEYBOARD_GRABBED,
        SDL_WINDOW_VULKAN,
        SDL_WINDOW_METAL,
        SDL_WINDOW_TRANSPARENT,
        SDL_WINDOW_NOT_FOCUSABLE
    };

    int i;
    int count = 0;
    for (i = 0; i < (sizeof(window_flags) / sizeof(window_flags[0])); ++i) {
        const SDL_WindowFlags flag = window_flags[i];
        if ((flags & flag) == flag) {
            if (count > 0) {
                SDL_snprintfcat(text, maxlen, " | ");
            }
            SDLTest_PrintWindowFlag(text, maxlen, flag);
            ++count;
        }
    }
}

static void SDLTest_PrintModStateFlag(char *text, size_t maxlen, SDL_Keymod flag)
{
    switch (flag) {
    case SDL_KMOD_LSHIFT:
        SDL_snprintfcat(text, maxlen, "LSHIFT");
        break;
    case SDL_KMOD_RSHIFT:
        SDL_snprintfcat(text, maxlen, "RSHIFT");
        break;
    case SDL_KMOD_LEVEL5:
        SDL_snprintfcat(text, maxlen, "LEVEL5");
        break;
    case SDL_KMOD_LCTRL:
        SDL_snprintfcat(text, maxlen, "LCTRL");
        break;
    case SDL_KMOD_RCTRL:
        SDL_snprintfcat(text, maxlen, "RCTRL");
        break;
    case SDL_KMOD_LALT:
        SDL_snprintfcat(text, maxlen, "LALT");
        break;
    case SDL_KMOD_RALT:
        SDL_snprintfcat(text, maxlen, "RALT");
        break;
    case SDL_KMOD_LGUI:
        SDL_snprintfcat(text, maxlen, "LGUI");
        break;
    case SDL_KMOD_RGUI:
        SDL_snprintfcat(text, maxlen, "RGUI");
        break;
    case SDL_KMOD_NUM:
        SDL_snprintfcat(text, maxlen, "NUM");
        break;
    case SDL_KMOD_CAPS:
        SDL_snprintfcat(text, maxlen, "CAPS");
        break;
    case SDL_KMOD_MODE:
        SDL_snprintfcat(text, maxlen, "MODE");
        break;
    case SDL_KMOD_SCROLL:
        SDL_snprintfcat(text, maxlen, "SCROLL");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%8.8x", (unsigned int) flag);
        break;
    }
}

static void SDLTest_PrintModState(char *text, size_t maxlen, SDL_Keymod keymod)
{
    const SDL_Keymod kmod_flags[] = {
        SDL_KMOD_LSHIFT,
        SDL_KMOD_RSHIFT,
        SDL_KMOD_LEVEL5,
        SDL_KMOD_LCTRL,
        SDL_KMOD_RCTRL,
        SDL_KMOD_LALT,
        SDL_KMOD_RALT,
        SDL_KMOD_LGUI,
        SDL_KMOD_RGUI,
        SDL_KMOD_NUM,
        SDL_KMOD_CAPS,
        SDL_KMOD_MODE,
        SDL_KMOD_SCROLL
    };

    int i;
    int count = 0;
    for (i = 0; i < SDL_arraysize(kmod_flags); ++i) {
        const SDL_Keymod flag = kmod_flags[i];
        if ((keymod & flag) == flag) {
            if (count > 0) {
                SDL_snprintfcat(text, maxlen, " | ");
            }
            SDLTest_PrintModStateFlag(text, maxlen, flag);
            ++count;
        }
    }
}

static void SDLTest_PrintButtonMask(char *text, size_t maxlen, SDL_MouseButtonFlags flags)
{
    int i;
    int count = 0;
    for (i = 1; i <= 32; ++i) {
        const Uint32 flag = SDL_BUTTON_MASK(i);
        if ((flags & flag) == flag) {
            if (count > 0) {
                SDL_snprintfcat(text, maxlen, " | ");
            }
            SDL_snprintfcat(text, maxlen, "SDL_BUTTON_MASK(%d)", i);
            ++count;
        }
    }
}

static void SDLTest_PrintPixelFormat(char *text, size_t maxlen, Uint32 format)
{
    const char *name = SDL_GetPixelFormatName(format);
    if (name) {
        if (SDL_strncmp(name, "SDL_PIXELFORMAT_", 16) == 0) {
            name += 16;
        }
        SDL_snprintfcat(text, maxlen, name);
    } else {
        SDL_snprintfcat(text, maxlen, "0x%8.8x", format);
    }
}

static void SDLTest_PrintLogicalPresentation(char *text, size_t maxlen, SDL_RendererLogicalPresentation logical_presentation)
{
    switch (logical_presentation) {
    case SDL_LOGICAL_PRESENTATION_DISABLED:
        SDL_snprintfcat(text, maxlen, "DISABLED");
        break;
    case SDL_LOGICAL_PRESENTATION_STRETCH:
        SDL_snprintfcat(text, maxlen, "STRETCH");
        break;
    case SDL_LOGICAL_PRESENTATION_LETTERBOX:
        SDL_snprintfcat(text, maxlen, "LETTERBOX");
        break;
    case SDL_LOGICAL_PRESENTATION_OVERSCAN:
        SDL_snprintfcat(text, maxlen, "OVERSCAN");
        break;
    case SDL_LOGICAL_PRESENTATION_INTEGER_SCALE:
        SDL_snprintfcat(text, maxlen, "INTEGER_SCALE");
        break;
    default:
        SDL_snprintfcat(text, maxlen, "0x%8.8x", logical_presentation);
        break;
    }
}

static void SDLTest_PrintRenderer(SDL_Renderer *renderer)
{
    const char *name;
    int i;
    char text[1024];
    int max_texture_size;
    const SDL_PixelFormat *texture_formats;

    name = SDL_GetRendererName(renderer);

    SDL_Log("  Renderer %s:", name);
    if (SDL_strcmp(name, "gpu") == 0) {
        SDL_GPUDevice *device = SDL_GetPointerProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_GPU_DEVICE_POINTER, NULL);
        SDL_Log("    Driver: %s", SDL_GetGPUDeviceDriver(device));
    }
    SDL_Log("    VSync: %d", (int)SDL_GetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_VSYNC_NUMBER, 0));

    texture_formats = (const SDL_PixelFormat *)SDL_GetPointerProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_TEXTURE_FORMATS_POINTER, NULL);
    if (texture_formats) {
        (void)SDL_snprintf(text, sizeof(text), "    Texture formats: ");
        for (i = 0; texture_formats[i]; ++i) {
            if (i > 0) {
                SDL_snprintfcat(text, sizeof(text), ", ");
            }
            SDLTest_PrintPixelFormat(text, sizeof(text), texture_formats[i]);
        }
        SDL_Log("%s", text);
    }

    max_texture_size = (int)SDL_GetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 0);
    if (max_texture_size) {
        SDL_Log("    Max Texture Size: %dx%d", max_texture_size, max_texture_size);
    }
}

static SDL_Surface *SDLTest_LoadIcon(const char *file)
{
    SDL_Surface *icon;

    /* Load the icon surface */
    icon = SDL_LoadBMP(file);
    if (!icon) {
        SDL_Log("Couldn't load %s: %s", file, SDL_GetError());
        return NULL;
    }

    if (icon->format == SDL_PIXELFORMAT_INDEX8) {
        /* Set the colorkey */
        SDL_SetSurfaceColorKey(icon, 1, *((Uint8 *)icon->pixels));
    }

    return icon;
}

static SDL_HitTestResult SDLCALL SDLTest_ExampleHitTestCallback(SDL_Window *win, const SDL_Point *area, void *data)
{
    int w, h;
    const int RESIZE_BORDER = 8;
    const int DRAGGABLE_TITLE = 32;

    /*SDL_Log("Hit test point %d,%d", area->x, area->y);*/

    SDL_GetWindowSize(win, &w, &h);

    if (area->x < RESIZE_BORDER) {
        if (area->y < RESIZE_BORDER) {
            SDL_Log("SDL_HITTEST_RESIZE_TOPLEFT");
            return SDL_HITTEST_RESIZE_TOPLEFT;
        } else if (area->y >= (h - RESIZE_BORDER)) {
            SDL_Log("SDL_HITTEST_RESIZE_BOTTOMLEFT");
            return SDL_HITTEST_RESIZE_BOTTOMLEFT;
        } else {
            SDL_Log("SDL_HITTEST_RESIZE_LEFT");
            return SDL_HITTEST_RESIZE_LEFT;
        }
    } else if (area->x >= (w - RESIZE_BORDER)) {
        if (area->y < RESIZE_BORDER) {
            SDL_Log("SDL_HITTEST_RESIZE_TOPRIGHT");
            return SDL_HITTEST_RESIZE_TOPRIGHT;
        } else if (area->y >= (h - RESIZE_BORDER)) {
            SDL_Log("SDL_HITTEST_RESIZE_BOTTOMRIGHT");
            return SDL_HITTEST_RESIZE_BOTTOMRIGHT;
        } else {
            SDL_Log("SDL_HITTEST_RESIZE_RIGHT");
            return SDL_HITTEST_RESIZE_RIGHT;
        }
    } else if (area->y >= (h - RESIZE_BORDER)) {
        SDL_Log("SDL_HITTEST_RESIZE_BOTTOM");
        return SDL_HITTEST_RESIZE_BOTTOM;
    } else if (area->y < RESIZE_BORDER) {
        SDL_Log("SDL_HITTEST_RESIZE_TOP");
        return SDL_HITTEST_RESIZE_TOP;
    } else if (area->y < DRAGGABLE_TITLE) {
        SDL_Log("SDL_HITTEST_DRAGGABLE");
        return SDL_HITTEST_DRAGGABLE;
    }
    return SDL_HITTEST_NORMAL;
}

bool SDLTest_CommonInit(SDLTest_CommonState *state)
{
    int i, j, m, n, w, h;
    char text[1024];

    if (state->flags & SDL_INIT_VIDEO) {
        if (state->verbose & VERBOSE_VIDEO) {
            n = SDL_GetNumVideoDrivers();
            if (n == 0) {
                SDL_Log("No built-in video drivers");
            } else {
                (void)SDL_snprintf(text, sizeof(text), "Built-in video drivers:");
                for (i = 0; i < n; ++i) {
                    if (i > 0) {
                        SDL_snprintfcat(text, sizeof(text), ",");
                    }
                    SDL_snprintfcat(text, sizeof(text), " %s", SDL_GetVideoDriver(i));
                }
                SDL_Log("%s", text);
            }
        }
        if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
            SDL_Log("Couldn't initialize video driver: %s",
                    SDL_GetError());
            return false;
        }
        if (state->verbose & VERBOSE_VIDEO) {
            SDL_Log("Video driver: %s",
                    SDL_GetCurrentVideoDriver());
        }

        /* Upload GL settings */
        SDL_GL_SetAttribute(SDL_GL_RED_SIZE, state->gl_red_size);
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, state->gl_green_size);
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, state->gl_blue_size);
        SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, state->gl_alpha_size);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, state->gl_double_buffer);
        SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, state->gl_buffer_size);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, state->gl_depth_size);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, state->gl_stencil_size);
        SDL_GL_SetAttribute(SDL_GL_ACCUM_RED_SIZE, state->gl_accum_red_size);
        SDL_GL_SetAttribute(SDL_GL_ACCUM_GREEN_SIZE, state->gl_accum_green_size);
        SDL_GL_SetAttribute(SDL_GL_ACCUM_BLUE_SIZE, state->gl_accum_blue_size);
        SDL_GL_SetAttribute(SDL_GL_ACCUM_ALPHA_SIZE, state->gl_accum_alpha_size);
        SDL_GL_SetAttribute(SDL_GL_STEREO, state->gl_stereo);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_RELEASE_BEHAVIOR, state->gl_release_behavior);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, state->gl_multisamplebuffers);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, state->gl_multisamplesamples);
        if (state->gl_accelerated >= 0) {
            SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL,
                                state->gl_accelerated);
        }
        SDL_GL_SetAttribute(SDL_GL_RETAINED_BACKING, state->gl_retained_backing);
        if (state->gl_major_version) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, state->gl_major_version);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, state->gl_minor_version);
        }
        if (state->gl_debug) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
        }
        if (state->gl_profile_mask) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, state->gl_profile_mask);
        }

        if (state->verbose & VERBOSE_MODES) {
            SDL_DisplayID *displays;
            SDL_Rect bounds, usablebounds;
            SDL_DisplayMode **modes;
            const SDL_DisplayMode *mode;
            int bpp;
            Uint32 Rmask, Gmask, Bmask, Amask;
#ifdef SDL_VIDEO_DRIVER_WINDOWS
            int adapterIndex = 0;
            int outputIndex = 0;
#endif
            displays = SDL_GetDisplays(&n);
            SDL_Log("Number of displays: %d", n);
            for (i = 0; i < n; ++i) {
                SDL_DisplayID displayID = displays[i];
                SDL_Log("Display %" SDL_PRIu32 ": %s", displayID, SDL_GetDisplayName(displayID));

                SDL_zero(bounds);
                SDL_GetDisplayBounds(displayID, &bounds);

                SDL_zero(usablebounds);
                SDL_GetDisplayUsableBounds(displayID, &usablebounds);

                SDL_Log("Bounds: %dx%d at %d,%d", bounds.w, bounds.h, bounds.x, bounds.y);
                SDL_Log("Usable bounds: %dx%d at %d,%d", usablebounds.w, usablebounds.h, usablebounds.x, usablebounds.y);

                mode = SDL_GetDesktopDisplayMode(displayID);
                SDL_GetMasksForPixelFormat(mode->format, &bpp, &Rmask, &Gmask,
                                           &Bmask, &Amask);
                SDL_Log("  Desktop mode: %dx%d@%gx %gHz, %d bits-per-pixel (%s)",
                        mode->w, mode->h, mode->pixel_density, mode->refresh_rate, bpp,
                        SDL_GetPixelFormatName(mode->format));
                if (Rmask || Gmask || Bmask) {
                    SDL_Log("      Red Mask   = 0x%.8" SDL_PRIx32, Rmask);
                    SDL_Log("      Green Mask = 0x%.8" SDL_PRIx32, Gmask);
                    SDL_Log("      Blue Mask  = 0x%.8" SDL_PRIx32, Bmask);
                    if (Amask) {
                        SDL_Log("      Alpha Mask = 0x%.8" SDL_PRIx32, Amask);
                    }
                }

                /* Print available fullscreen video modes */
                modes = SDL_GetFullscreenDisplayModes(displayID, &m);
                if (m == 0) {
                    SDL_Log("No available fullscreen video modes");
                } else {
                    SDL_Log("  Fullscreen video modes:");
                    for (j = 0; j < m; ++j) {
                        mode = modes[j];
                        SDL_GetMasksForPixelFormat(mode->format, &bpp, &Rmask,
                                                   &Gmask, &Bmask, &Amask);
                        SDL_Log("    Mode %d: %dx%d@%gx %gHz, %d bits-per-pixel (%s)",
                                j, mode->w, mode->h, mode->pixel_density, mode->refresh_rate, bpp,
                                SDL_GetPixelFormatName(mode->format));
                        if (Rmask || Gmask || Bmask) {
                            SDL_Log("        Red Mask   = 0x%.8" SDL_PRIx32,
                                    Rmask);
                            SDL_Log("        Green Mask = 0x%.8" SDL_PRIx32,
                                    Gmask);
                            SDL_Log("        Blue Mask  = 0x%.8" SDL_PRIx32,
                                    Bmask);
                            if (Amask) {
                                SDL_Log("        Alpha Mask = 0x%.8" SDL_PRIx32, Amask);
                            }
                        }
                    }
                }
                SDL_free(modes);

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
                /* Print the D3D9 adapter index */
                adapterIndex = SDL_GetDirect3D9AdapterIndex(displayID);
                SDL_Log("D3D9 Adapter Index: %d", adapterIndex);

                /* Print the DXGI adapter and output indices */
                SDL_GetDXGIOutputInfo(displayID, &adapterIndex, &outputIndex);
                SDL_Log("DXGI Adapter Index: %d  Output Index: %d", adapterIndex, outputIndex);
#endif
            }
            SDL_free(displays);
        }

        if (state->verbose & VERBOSE_RENDER) {
            n = SDL_GetNumRenderDrivers();
            if (n == 0) {
                SDL_Log("No built-in render drivers");
            } else {
                SDL_Log("Built-in render drivers:");
                for (i = 0; i < n; ++i) {
                    SDL_Log("  %s", SDL_GetRenderDriver(i));
                }
            }
        }

        state->displayID = SDL_GetPrimaryDisplay();
        if (state->display_index > 0) {
            SDL_DisplayID *displays = SDL_GetDisplays(&n);
            if (state->display_index < n) {
                state->displayID = displays[state->display_index];
            }
            SDL_free(displays);

            if (SDL_WINDOWPOS_ISUNDEFINED(state->window_x)) {
                state->window_x = SDL_WINDOWPOS_UNDEFINED_DISPLAY(state->displayID);
                state->window_y = SDL_WINDOWPOS_UNDEFINED_DISPLAY(state->displayID);
            } else if (SDL_WINDOWPOS_ISCENTERED(state->window_x)) {
                state->window_x = SDL_WINDOWPOS_CENTERED_DISPLAY(state->displayID);
                state->window_y = SDL_WINDOWPOS_CENTERED_DISPLAY(state->displayID);
            }
        }

        {
            bool include_high_density_modes = false;
            if (state->window_flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
                include_high_density_modes = true;
            }
            SDL_GetClosestFullscreenDisplayMode(state->displayID, state->window_w, state->window_h, state->refresh_rate, include_high_density_modes, &state->fullscreen_mode);
        }

        state->windows =
            (SDL_Window **)SDL_calloc(state->num_windows,
                                      sizeof(*state->windows));
        state->renderers =
            (SDL_Renderer **)SDL_calloc(state->num_windows,
                                        sizeof(*state->renderers));
        state->targets =
            (SDL_Texture **)SDL_calloc(state->num_windows,
                                       sizeof(*state->targets));
        if (!state->windows || !state->renderers) {
            SDL_Log("Out of memory!");
            return false;
        }
        for (i = 0; i < state->num_windows; ++i) {
            char title[1024];
            SDL_Rect r;
            SDL_PropertiesID props;

            if (state->fill_usable_bounds) {
                SDL_GetDisplayUsableBounds(state->displayID, &r);
            } else {
                r.x = state->window_x;
                r.y = state->window_y;
                r.w = state->window_w;
                r.h = state->window_h;
                if (state->auto_scale_content) {
                    float scale = SDL_GetDisplayContentScale(state->displayID);
                    r.w = (int)SDL_ceilf(r.w * scale);
                    r.h = (int)SDL_ceilf(r.h * scale);
                }
            }

            if (state->num_windows > 1) {
                (void)SDL_snprintf(title, SDL_arraysize(title), "%s %d",
                                   state->window_title, i + 1);
            } else {
                SDL_strlcpy(title, state->window_title, SDL_arraysize(title));
            }
            props = SDL_CreateProperties();
            SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, title);
            SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, r.x);
            SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, r.y);
            SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, r.w);
            SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, r.h);
            SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_FLAGS_NUMBER, state->window_flags);
            state->windows[i] = SDL_CreateWindowWithProperties(props);
            SDL_DestroyProperties(props);
            if (!state->windows[i]) {
                SDL_Log("Couldn't create window: %s",
                        SDL_GetError());
                return false;
            }
            if (state->window_minW || state->window_minH) {
                SDL_SetWindowMinimumSize(state->windows[i], state->window_minW, state->window_minH);
            }
            if (state->window_maxW || state->window_maxH) {
                SDL_SetWindowMaximumSize(state->windows[i], state->window_maxW, state->window_maxH);
            }
            if (state->window_min_aspect != 0.f || state->window_max_aspect != 0.f) {
                SDL_SetWindowAspectRatio(state->windows[i], state->window_min_aspect, state->window_max_aspect);
            }
            SDL_GetWindowSize(state->windows[i], &w, &h);
            if (!(state->window_flags & SDL_WINDOW_RESIZABLE) && (w != r.w || h != r.h)) {
                SDL_Log("Window requested size %dx%d, got %dx%d", r.w, r.h, w, h);
                state->window_w = w;
                state->window_h = h;
            }
            if (state->window_flags & SDL_WINDOW_FULLSCREEN) {
                if (state->fullscreen_exclusive) {
                    SDL_SetWindowFullscreenMode(state->windows[i], &state->fullscreen_mode);
                }
                SDL_SetWindowFullscreen(state->windows[i], true);
            }

            /* Add resize/drag areas for windows that are borderless and resizable */
            if ((state->window_flags & (SDL_WINDOW_RESIZABLE | SDL_WINDOW_BORDERLESS)) ==
                (SDL_WINDOW_RESIZABLE | SDL_WINDOW_BORDERLESS)) {
                SDL_SetWindowHitTest(state->windows[i], SDLTest_ExampleHitTestCallback, NULL);
            }

            if (state->window_icon) {
                SDL_Surface *icon = SDLTest_LoadIcon(state->window_icon);
                if (icon) {
                    SDL_SetWindowIcon(state->windows[i], icon);
                    SDL_DestroySurface(icon);
                }
            }

            if (!SDL_RectEmpty(&state->confine)) {
                SDL_SetWindowMouseRect(state->windows[i], &state->confine);
            }

            if (!state->skip_renderer && (state->renderdriver || !(state->window_flags & (SDL_WINDOW_OPENGL | SDL_WINDOW_VULKAN | SDL_WINDOW_METAL)))) {
                state->renderers[i] = SDL_CreateRenderer(state->windows[i], state->renderdriver);
                if (!state->renderers[i]) {
                    SDL_Log("Couldn't create renderer: %s",
                            SDL_GetError());
                    return false;
                }
                if (state->logical_w == 0 || state->logical_h == 0) {
                    state->logical_w = state->window_w;
                    state->logical_h = state->window_h;
                }
                if (state->render_vsync) {
                    SDL_SetRenderVSync(state->renderers[i], state->render_vsync);
                }
                if (!SDL_SetRenderLogicalPresentation(state->renderers[i], state->logical_w, state->logical_h, state->logical_presentation)) {
                    SDL_Log("Couldn't set logical presentation: %s", SDL_GetError());
                    return false;
                }
                if (state->scale != 0.0f) {
                    SDL_SetRenderScale(state->renderers[i], state->scale, state->scale);
                }
                if (state->verbose & VERBOSE_RENDER) {
                    SDL_Log("Current renderer:");
                    SDLTest_PrintRenderer(state->renderers[i]);
                }
            }

            SDL_ShowWindow(state->windows[i]);
        }
        if (state->hide_cursor) {
            SDL_HideCursor();
        }
    }

    if (state->flags & SDL_INIT_AUDIO) {
        if (state->verbose & VERBOSE_AUDIO) {
            n = SDL_GetNumAudioDrivers();
            if (n == 0) {
                SDL_Log("No built-in audio drivers");
            } else {
                (void)SDL_snprintf(text, sizeof(text), "Built-in audio drivers:");
                for (i = 0; i < n; ++i) {
                    if (i > 0) {
                        SDL_snprintfcat(text, sizeof(text), ",");
                    }
                    SDL_snprintfcat(text, sizeof(text), " %s", SDL_GetAudioDriver(i));
                }
                SDL_Log("%s", text);
            }
        }
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            SDL_Log("Couldn't initialize audio driver: %s",
                    SDL_GetError());
            return false;
        }
        if (state->verbose & VERBOSE_AUDIO) {
            SDL_Log("Audio driver: %s",
                    SDL_GetCurrentAudioDriver());
        }

        const SDL_AudioSpec spec = { state->audio_format, state->audio_channels, state->audio_freq };
        state->audio_id = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec);
        if (!state->audio_id) {
            SDL_Log("Couldn't open audio: %s", SDL_GetError());
            return false;
        }
    }

    if (state->flags & SDL_INIT_CAMERA) {
        SDL_InitSubSystem(SDL_INIT_CAMERA);
    }

    return true;
}

static const char *SystemThemeName(void)
{
    switch (SDL_GetSystemTheme()) {
#define CASE(X)               \
    case SDL_SYSTEM_THEME_##X: \
        return #X
        CASE(UNKNOWN);
        CASE(LIGHT);
        CASE(DARK);
#undef CASE
    default:
        return "???";
    }
}

static const char *DisplayOrientationName(int orientation)
{
    switch (orientation) {
#define CASE(X)               \
    case SDL_ORIENTATION_##X: \
        return #X
        CASE(UNKNOWN);
        CASE(LANDSCAPE);
        CASE(LANDSCAPE_FLIPPED);
        CASE(PORTRAIT);
        CASE(PORTRAIT_FLIPPED);
#undef CASE
    default:
        return "???";
    }
}

static const char *GamepadAxisName(const SDL_GamepadAxis axis)
{
    switch (axis) {
#define AXIS_CASE(ax)              \
    case SDL_GAMEPAD_AXIS_##ax: \
        return #ax
        AXIS_CASE(INVALID);
        AXIS_CASE(LEFTX);
        AXIS_CASE(LEFTY);
        AXIS_CASE(RIGHTX);
        AXIS_CASE(RIGHTY);
        AXIS_CASE(LEFT_TRIGGER);
        AXIS_CASE(RIGHT_TRIGGER);
#undef AXIS_CASE
    default:
        return "???";
    }
}

static const char *GamepadButtonName(const SDL_GamepadButton button)
{
    switch (button) {
#define BUTTON_CASE(btn)              \
    case SDL_GAMEPAD_BUTTON_##btn: \
        return #btn
        BUTTON_CASE(INVALID);
        BUTTON_CASE(SOUTH);
        BUTTON_CASE(EAST);
        BUTTON_CASE(WEST);
        BUTTON_CASE(NORTH);
        BUTTON_CASE(BACK);
        BUTTON_CASE(GUIDE);
        BUTTON_CASE(START);
        BUTTON_CASE(LEFT_STICK);
        BUTTON_CASE(RIGHT_STICK);
        BUTTON_CASE(LEFT_SHOULDER);
        BUTTON_CASE(RIGHT_SHOULDER);
        BUTTON_CASE(DPAD_UP);
        BUTTON_CASE(DPAD_DOWN);
        BUTTON_CASE(DPAD_LEFT);
        BUTTON_CASE(DPAD_RIGHT);
#undef BUTTON_CASE
    default:
        return "???";
    }
}

void SDLTest_PrintEvent(const SDL_Event *event)
{
    switch (event->type) {
    case SDL_EVENT_SYSTEM_THEME_CHANGED:
        SDL_Log("SDL EVENT: System theme changed to %s", SystemThemeName());
        break;
    case SDL_EVENT_DISPLAY_ADDED:
        SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " attached",
                event->display.displayID);
        break;
    case SDL_EVENT_DISPLAY_CONTENT_SCALE_CHANGED:
        {
            float scale = SDL_GetDisplayContentScale(event->display.displayID);
            SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " changed content scale to %d%%",
                    event->display.displayID, (int)(scale * 100.0f));
        }
        break;
    case SDL_EVENT_DISPLAY_DESKTOP_MODE_CHANGED:
        SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " desktop mode changed to %" SDL_PRIs32 "x%" SDL_PRIs32,
                event->display.displayID, event->display.data1, event->display.data2);
        break;
    case SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED:
        SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " current mode changed to %" SDL_PRIs32 "x%" SDL_PRIs32,
                event->display.displayID, event->display.data1, event->display.data2);
        break;
    case SDL_EVENT_DISPLAY_MOVED:
        SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " changed position",
                event->display.displayID);
        break;
    case SDL_EVENT_DISPLAY_ORIENTATION:
        SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " changed orientation to %s",
                event->display.displayID, DisplayOrientationName(event->display.data1));
        break;
    case SDL_EVENT_DISPLAY_REMOVED:
        SDL_Log("SDL EVENT: Display %" SDL_PRIu32 " removed",
                event->display.displayID);
        break;
    case SDL_EVENT_WINDOW_SHOWN:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " shown", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_HIDDEN:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " hidden", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_EXPOSED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " exposed", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_MOVED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " moved to %" SDL_PRIs32 ",%" SDL_PRIs32,
                event->window.windowID, event->window.data1, event->window.data2);
        break;
    case SDL_EVENT_WINDOW_RESIZED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " resized to %" SDL_PRIs32 "x%" SDL_PRIs32,
                event->window.windowID, event->window.data1, event->window.data2);
        break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " changed pixel size to %" SDL_PRIs32 "x%" SDL_PRIs32,
                event->window.windowID, event->window.data1, event->window.data2);
        break;
    case SDL_EVENT_WINDOW_METAL_VIEW_RESIZED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " changed metal view size",
                event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_SAFE_AREA_CHANGED: {
        SDL_Rect rect;

        SDL_GetWindowSafeArea(SDL_GetWindowFromEvent(event), &rect);
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " changed safe area to: %d,%d %dx%d",
                event->window.windowID, rect.x, rect.y, rect.w, rect.h);
        break;
    }
    case SDL_EVENT_WINDOW_MINIMIZED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " minimized", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_MAXIMIZED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " maximized", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_RESTORED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " restored", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_MOUSE_ENTER:
        SDL_Log("SDL EVENT: Mouse entered window %" SDL_PRIu32, event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_MOUSE_LEAVE:
        SDL_Log("SDL EVENT: Mouse left window %" SDL_PRIu32, event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " gained keyboard focus",
                event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " lost keyboard focus",
                event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " closed", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_HIT_TEST:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " hit test", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_ICCPROF_CHANGED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " ICC profile changed", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " display changed to %" SDL_PRIs32, event->window.windowID, event->window.data1);
        break;
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " display scale changed to %d%%", event->window.windowID, (int)(SDL_GetWindowDisplayScale(SDL_GetWindowFromEvent(event)) * 100.0f));
        break;
    case SDL_EVENT_WINDOW_OCCLUDED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " occluded", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " entered fullscreen", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " left fullscreen", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_DESTROYED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " destroyed", event->window.windowID);
        break;
    case SDL_EVENT_WINDOW_HDR_STATE_CHANGED:
        SDL_Log("SDL EVENT: Window %" SDL_PRIu32 " HDR %s", event->window.windowID, event->window.data1 ? "enabled" : "disabled");
        break;
    case SDL_EVENT_KEYBOARD_ADDED:
        SDL_Log("SDL EVENT: Keyboard %" SDL_PRIu32 " attached",
                event->kdevice.which);
        break;
    case SDL_EVENT_KEYBOARD_REMOVED:
        SDL_Log("SDL EVENT: Keyboard %" SDL_PRIu32 " removed",
                event->kdevice.which);
        break;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        char modstr[64];
        if (event->key.mod) {
            modstr[0] = '\0';
            SDLTest_PrintModState(modstr, sizeof (modstr), event->key.mod);
        } else {
            SDL_strlcpy(modstr, "NONE", sizeof (modstr));
        }

        SDL_Log("SDL EVENT: Keyboard: key %s in window %" SDL_PRIu32 ": scancode 0x%08X = %s, keycode 0x%08" SDL_PRIX32 " = %s, mods = %s",
                (event->type == SDL_EVENT_KEY_DOWN) ? "pressed" : "released",
                event->key.windowID,
                event->key.scancode,
                SDL_GetScancodeName(event->key.scancode),
                event->key.key, SDL_GetKeyName(event->key.key),
                modstr);
        break;
    }
    case SDL_EVENT_TEXT_EDITING:
        SDL_Log("SDL EVENT: Keyboard: text editing \"%s\" in window %" SDL_PRIu32,
                event->edit.text, event->edit.windowID);
        break;
    case SDL_EVENT_TEXT_EDITING_CANDIDATES:
        SDL_Log("SDL EVENT: Keyboard: text editing candidates in window %" SDL_PRIu32,
                event->edit.windowID);
        break;
    case SDL_EVENT_TEXT_INPUT:
        SDL_Log("SDL EVENT: Keyboard: text input \"%s\" in window %" SDL_PRIu32,
                event->text.text, event->text.windowID);
        break;
    case SDL_EVENT_KEYMAP_CHANGED:
        SDL_Log("SDL EVENT: Keymap changed");
        break;
    case SDL_EVENT_MOUSE_ADDED:
        SDL_Log("SDL EVENT: Mouse %" SDL_PRIu32 " attached",
                event->mdevice.which);
        break;
    case SDL_EVENT_MOUSE_REMOVED:
        SDL_Log("SDL EVENT: Mouse %" SDL_PRIu32 " removed",
                event->mdevice.which);
        break;
    case SDL_EVENT_MOUSE_MOTION:
        SDL_Log("SDL EVENT: Mouse: moved to %g,%g (%g,%g) in window %" SDL_PRIu32,
                event->motion.x, event->motion.y,
                event->motion.xrel, event->motion.yrel,
                event->motion.windowID);
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        SDL_Log("SDL EVENT: Mouse: button %d pressed at %g,%g with click count %d in window %" SDL_PRIu32,
                event->button.button, event->button.x, event->button.y, event->button.clicks,
                event->button.windowID);
        break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        SDL_Log("SDL EVENT: Mouse: button %d released at %g,%g with click count %d in window %" SDL_PRIu32,
                event->button.button, event->button.x, event->button.y, event->button.clicks,
                event->button.windowID);
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        SDL_Log("SDL EVENT: Mouse: wheel scrolled %g in x and %g in y (reversed: %d) in window %" SDL_PRIu32,
                event->wheel.x, event->wheel.y, event->wheel.direction, event->wheel.windowID);
        break;
    case SDL_EVENT_JOYSTICK_ADDED:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIu32 " attached",
                event->jdevice.which);
        break;
    case SDL_EVENT_JOYSTICK_REMOVED:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIu32 " removed",
                event->jdevice.which);
        break;
    case SDL_EVENT_JOYSTICK_AXIS_MOTION:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIu32 " axis %d value: %d",
                event->jaxis.which,
                event->jaxis.axis,
                event->jaxis.value);
        break;
    case SDL_EVENT_JOYSTICK_BALL_MOTION:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIs32 ": ball %d moved by %d,%d",
                event->jball.which, event->jball.ball, event->jball.xrel,
                event->jball.yrel);
        break;
    case SDL_EVENT_JOYSTICK_HAT_MOTION:
    {
        const char *position = "UNKNOWN";
        switch (event->jhat.value) {
        case SDL_HAT_CENTERED:
            position = "CENTER";
            break;
        case SDL_HAT_UP:
            position = "UP";
            break;
        case SDL_HAT_RIGHTUP:
            position = "RIGHTUP";
            break;
        case SDL_HAT_RIGHT:
            position = "RIGHT";
            break;
        case SDL_HAT_RIGHTDOWN:
            position = "RIGHTDOWN";
            break;
        case SDL_HAT_DOWN:
            position = "DOWN";
            break;
        case SDL_HAT_LEFTDOWN:
            position = "LEFTDOWN";
            break;
        case SDL_HAT_LEFT:
            position = "LEFT";
            break;
        case SDL_HAT_LEFTUP:
            position = "LEFTUP";
            break;
        }
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIu32 ": hat %d moved to %s",
                event->jhat.which, event->jhat.hat, position);
    } break;
    case SDL_EVENT_JOYSTICK_BUTTON_DOWN:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIu32 ": button %d pressed",
                event->jbutton.which, event->jbutton.button);
        break;
    case SDL_EVENT_JOYSTICK_BUTTON_UP:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIu32 ": button %d released",
                event->jbutton.which, event->jbutton.button);
        break;
    case SDL_EVENT_JOYSTICK_BATTERY_UPDATED:
        SDL_Log("SDL EVENT: Joystick %" SDL_PRIu32 ": battery at %d percent",
                event->jbattery.which, event->jbattery.percent);
        break;
    case SDL_EVENT_GAMEPAD_ADDED:
        SDL_Log("SDL EVENT: Gamepad %" SDL_PRIu32 " attached",
                event->gdevice.which);
        break;
    case SDL_EVENT_GAMEPAD_REMOVED:
        SDL_Log("SDL EVENT: Gamepad %" SDL_PRIu32 " removed",
                event->gdevice.which);
        break;
    case SDL_EVENT_GAMEPAD_REMAPPED:
        SDL_Log("SDL EVENT: Gamepad %" SDL_PRIu32 " mapping changed",
                event->gdevice.which);
        break;
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
        SDL_Log("SDL EVENT: Gamepad %" SDL_PRIu32 " axis %d ('%s') value: %d",
                event->gaxis.which,
                event->gaxis.axis,
                GamepadAxisName((SDL_GamepadAxis)event->gaxis.axis),
                event->gaxis.value);
        break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        SDL_Log("SDL EVENT: Gamepad %" SDL_PRIu32 "button %d ('%s') down",
                event->gbutton.which, event->gbutton.button,
                GamepadButtonName((SDL_GamepadButton)event->gbutton.button));
        break;
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
        SDL_Log("SDL EVENT: Gamepad %" SDL_PRIu32 " button %d ('%s') up",
                event->gbutton.which, event->gbutton.button,
                GamepadButtonName((SDL_GamepadButton)event->gbutton.button));
        break;
    case SDL_EVENT_CLIPBOARD_UPDATE:
        SDL_Log("SDL EVENT: Clipboard updated");
        break;

    case SDL_EVENT_FINGER_MOTION:
        SDL_Log("SDL EVENT: Finger: motion touch=%" SDL_PRIu64 ", finger=%" SDL_PRIu64 ", x=%f, y=%f, dx=%f, dy=%f, pressure=%f",
                event->tfinger.touchID,
                event->tfinger.fingerID,
                event->tfinger.x, event->tfinger.y,
                event->tfinger.dx, event->tfinger.dy, event->tfinger.pressure);
        break;
    case SDL_EVENT_FINGER_DOWN:
    case SDL_EVENT_FINGER_UP:
    case SDL_EVENT_FINGER_CANCELED:
        SDL_Log("SDL EVENT: Finger: %s touch=%" SDL_PRIu64 ", finger=%" SDL_PRIu64 ", x=%f, y=%f, dx=%f, dy=%f, pressure=%f",
                (event->type == SDL_EVENT_FINGER_DOWN) ? "down" :
                (event->type == SDL_EVENT_FINGER_UP) ? "up" : "cancel",
                event->tfinger.touchID,
                event->tfinger.fingerID,
                event->tfinger.x, event->tfinger.y,
                event->tfinger.dx, event->tfinger.dy, event->tfinger.pressure);
        break;

    case SDL_EVENT_RENDER_TARGETS_RESET:
        SDL_Log("SDL EVENT: render targets reset in window %" SDL_PRIu32, event->render.windowID);
        break;
    case SDL_EVENT_RENDER_DEVICE_RESET:
        SDL_Log("SDL EVENT: render device reset in window %" SDL_PRIu32, event->render.windowID);
        break;
    case SDL_EVENT_RENDER_DEVICE_LOST:
        SDL_Log("SDL EVENT: render device lost in window %" SDL_PRIu32, event->render.windowID);
        break;

    case SDL_EVENT_TERMINATING:
        SDL_Log("SDL EVENT: App terminating");
        break;
    case SDL_EVENT_LOW_MEMORY:
        SDL_Log("SDL EVENT: App running low on memory");
        break;
    case SDL_EVENT_WILL_ENTER_BACKGROUND:
        SDL_Log("SDL EVENT: App will enter the background");
        break;
    case SDL_EVENT_DID_ENTER_BACKGROUND:
        SDL_Log("SDL EVENT: App entered the background");
        break;
    case SDL_EVENT_WILL_ENTER_FOREGROUND:
        SDL_Log("SDL EVENT: App will enter the foreground");
        break;
    case SDL_EVENT_DID_ENTER_FOREGROUND:
        SDL_Log("SDL EVENT: App entered the foreground");
        break;
    case SDL_EVENT_DROP_BEGIN:
        SDL_Log("SDL EVENT: Drag and drop beginning in window %" SDL_PRIu32, event->drop.windowID);
        break;
    case SDL_EVENT_DROP_POSITION:
        SDL_Log("SDL EVENT: Drag and drop moving in window %" SDL_PRIu32 ": %g,%g", event->drop.windowID, event->drop.x, event->drop.y);
        break;
    case SDL_EVENT_DROP_FILE:
        SDL_Log("SDL EVENT: Drag and drop file in window %" SDL_PRIu32 ": '%s'", event->drop.windowID, event->drop.data);
        break;
    case SDL_EVENT_DROP_TEXT:
        SDL_Log("SDL EVENT: Drag and drop text in window %" SDL_PRIu32 ": '%s'", event->drop.windowID, event->drop.data);
        break;
    case SDL_EVENT_DROP_COMPLETE:
        SDL_Log("SDL EVENT: Drag and drop ending");
        break;
    case SDL_EVENT_AUDIO_DEVICE_ADDED:
        SDL_Log("SDL EVENT: Audio %s device %" SDL_PRIu32 " available",
                event->adevice.recording ? "recording" : "playback",
                event->adevice.which);
        break;
    case SDL_EVENT_AUDIO_DEVICE_REMOVED:
        SDL_Log("SDL EVENT: Audio %s device %" SDL_PRIu32 " removed",
                event->adevice.recording ? "recording" : "playback",
                event->adevice.which);
        break;
    case SDL_EVENT_AUDIO_DEVICE_FORMAT_CHANGED:
        SDL_Log("SDL EVENT: Audio %s device %" SDL_PRIu32 " format changed",
                event->adevice.recording ? "recording" : "playback",
                event->adevice.which);
        break;
    case SDL_EVENT_CAMERA_DEVICE_ADDED:
        SDL_Log("SDL EVENT: Camera device %" SDL_PRIu32 " available",
                event->cdevice.which);
        break;
    case SDL_EVENT_CAMERA_DEVICE_REMOVED:
        SDL_Log("SDL EVENT: Camera device %" SDL_PRIu32 " removed",
                event->cdevice.which);
        break;
    case SDL_EVENT_CAMERA_DEVICE_APPROVED:
        SDL_Log("SDL EVENT: Camera device %" SDL_PRIu32 " permission granted",
                event->cdevice.which);
        break;
    case SDL_EVENT_CAMERA_DEVICE_DENIED:
        SDL_Log("SDL EVENT: Camera device %" SDL_PRIu32 " permission denied",
                event->cdevice.which);
        break;
    case SDL_EVENT_SENSOR_UPDATE:
        SDL_Log("SDL EVENT: Sensor update for %" SDL_PRIu32,
                event->sensor.which);
        break;
    case SDL_EVENT_PEN_PROXIMITY_IN:
        SDL_Log("SDL EVENT: Pen %" SDL_PRIu32 " entered proximity",
                event->pproximity.which);
        break;
    case SDL_EVENT_PEN_PROXIMITY_OUT:
        SDL_Log("SDL EVENT: Pen %" SDL_PRIu32 " left proximity",
                event->ptouch.which);
        break;
    case SDL_EVENT_PEN_DOWN:
        SDL_Log("SDL EVENT: Pen %" SDL_PRIu32 " touched down at %g,%g",
                event->ptouch.which, event->ptouch.x, event->ptouch.y);
        break;
    case SDL_EVENT_PEN_UP:
        SDL_Log("SDL EVENT: Pen %" SDL_PRIu32 " lifted off at %g,%g",
                event->ptouch.which, event->ptouch.x, event->ptouch.y);
        break;
    case SDL_EVENT_PEN_BUTTON_DOWN:
        SDL_Log("SDL EVENT: Pen %" SDL_PRIu32 " button %d pressed at %g,%g",
                event->pbutton.which, event->pbutton.button, event->pbutton.x, event->pbutton.y);
        break;
    case SDL_EVENT_PEN_BUTTON_UP:
        SDL_Log("SDL EVENT: Pen %" SDL_PRIu32 " button %d released at %g,%g",
                event->pbutton.which, event->pbutton.button, event->pbutton.x, event->pbutton.y);
        break;
    case SDL_EVENT_PEN_MOTION:
        SDL_Log("SDL EVENT: Pen %" SDL_PRIu32 " moved to %g,%g",
                event->pmotion.which, event->pmotion.x, event->pmotion.y);
        break;
    case SDL_EVENT_PEN_AXIS:
        SDL_Log("SDL EVENT: Pen %" SDL_PRIu32 " axis %d changed to %.2f",
                event->paxis.which, event->paxis.axis, event->paxis.value);
        break;
    case SDL_EVENT_LOCALE_CHANGED:
        SDL_Log("SDL EVENT: Locale changed");
        break;
    case SDL_EVENT_QUIT:
        SDL_Log("SDL EVENT: Quit requested");
        break;
    case SDL_EVENT_USER:
        SDL_Log("SDL EVENT: User event %" SDL_PRIs32, event->user.code);
        break;
    default:
        SDL_Log("Unknown event 0x%4.4" SDL_PRIx32, event->type);
        break;
    }
}

#define SCREENSHOT_FILE "screenshot.bmp"

typedef struct
{
    void *image;
    size_t size;
} SDLTest_ClipboardData;

static void SDLCALL SDLTest_ScreenShotClipboardCleanup(void *context)
{
    SDLTest_ClipboardData *data = (SDLTest_ClipboardData *)context;

    SDL_Log("Cleaning up screenshot image data");

    if (data->image) {
        SDL_free(data->image);
    }
    SDL_free(data);
}

static const void * SDLCALL SDLTest_ScreenShotClipboardProvider(void *context, const char *mime_type, size_t *size)
{
    SDLTest_ClipboardData *data = (SDLTest_ClipboardData *)context;

    if (SDL_strncmp(mime_type, "text", 4) == 0) {
        SDL_Log("Providing screenshot title to clipboard!");

        /* Return "Test screenshot" */
        *size = 15;
        return "Test screenshot (but this isn't part of it)";
    }

    SDL_Log("Providing screenshot image to clipboard!");

    if (!data->image) {
        SDL_IOStream *file;

        file = SDL_IOFromFile(SCREENSHOT_FILE, "r");
        if (file) {
            size_t length = (size_t)SDL_GetIOSize(file);
            void *image = SDL_malloc(length);
            if (image) {
                if (SDL_ReadIO(file, image, length) != length) {
                    SDL_Log("Couldn't read %s: %s", SCREENSHOT_FILE, SDL_GetError());
                    SDL_free(image);
                    image = NULL;
                }
            }
            SDL_CloseIO(file);

            if (image) {
                data->image = image;
                data->size = length;
            }
        } else {
            SDL_Log("Couldn't load %s: %s", SCREENSHOT_FILE, SDL_GetError());
        }
    }

    *size = data->size;
    return data->image;
}

static void SDLTest_CopyScreenShot(SDL_Renderer *renderer)
{
    SDL_Surface *surface;
    const char *image_formats[] = {
        "text/plain;charset=utf-8",
        "image/bmp"
    };
    SDLTest_ClipboardData *clipboard_data;

    if (!renderer) {
        return;
    }

    surface = SDL_RenderReadPixels(renderer, NULL);
    if (!surface) {
        SDL_Log("Couldn't read screen: %s", SDL_GetError());
        return;
    }

    if (!SDL_SaveBMP(surface, SCREENSHOT_FILE)) {
        SDL_Log("Couldn't save %s: %s", SCREENSHOT_FILE, SDL_GetError());
        SDL_DestroySurface(surface);
        return;
    }
    SDL_DestroySurface(surface);

    clipboard_data = (SDLTest_ClipboardData *)SDL_calloc(1, sizeof(*clipboard_data));
    if (!clipboard_data) {
        SDL_Log("Couldn't allocate clipboard data");
        return;
    }
    SDL_SetClipboardData(SDLTest_ScreenShotClipboardProvider, SDLTest_ScreenShotClipboardCleanup, clipboard_data, image_formats, SDL_arraysize(image_formats));
    SDL_Log("Saved screenshot to %s and clipboard", SCREENSHOT_FILE);
}

static void SDLTest_PasteScreenShot(void)
{
    const char *image_formats[] = {
        "image/bmp",
        "image/png",
        "image/tiff",
    };
    size_t i;

    for (i = 0; i < SDL_arraysize(image_formats); ++i) {
        size_t size;
        void *data = SDL_GetClipboardData(image_formats[i], &size);
        if (data) {
            char filename[16];
            SDL_IOStream *file;

            SDL_snprintf(filename, sizeof(filename), "clipboard.%s", image_formats[i] + 6);
            file = SDL_IOFromFile(filename, "w");
            if (file) {
                SDL_Log("Writing clipboard image to %s", filename);
                SDL_WriteIO(file, data, size);
                SDL_CloseIO(file);
            }
            SDL_free(data);
            return;
        }
    }
    SDL_Log("No supported screenshot data in the clipboard");
}

static void FullscreenTo(SDLTest_CommonState *state, int index, int windowId)
{
    int num_displays;
    SDL_DisplayID *displays;
    SDL_Window *window;
    SDL_WindowFlags flags;
    const SDL_DisplayMode *mode;
    struct SDL_Rect rect = { 0, 0, 0, 0 };

    displays = SDL_GetDisplays(&num_displays);
    if (displays && index < num_displays) {
        window = SDL_GetWindowFromID(windowId);
        if (window) {
            SDL_GetDisplayBounds(displays[index], &rect);

            flags = SDL_GetWindowFlags(window);
            if (flags & SDL_WINDOW_FULLSCREEN) {
                SDL_SetWindowFullscreen(window, false);
                SDL_Delay(15);
            }

            mode = SDL_GetWindowFullscreenMode(window);
            if (mode) {
                /* Try to set the existing mode on the new display */
                SDL_DisplayMode new_mode;

                SDL_memcpy(&new_mode, mode, sizeof(new_mode));
                new_mode.displayID = displays[index];
                if (!SDL_SetWindowFullscreenMode(window, &new_mode)) {
                    /* Try again with a default mode */
                    bool include_high_density_modes = false;
                    if (state->window_flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
                        include_high_density_modes = true;
                    }
                    if (SDL_GetClosestFullscreenDisplayMode(displays[index], state->window_w, state->window_h, state->refresh_rate, include_high_density_modes, &new_mode)) {
                        SDL_SetWindowFullscreenMode(window, &new_mode);
                    }
                }
            }
            if (!mode) {
                SDL_SetWindowPosition(window, rect.x, rect.y);
            }
            SDL_SetWindowFullscreen(window, true);
        }
    }
    SDL_free(displays);
}

SDL_AppResult SDLTest_CommonEventMainCallbacks(SDLTest_CommonState *state, const SDL_Event *event)
{
    int i;

    if (state->verbose & VERBOSE_EVENT) {
        if ((event->type != SDL_EVENT_MOUSE_MOTION &&
             event->type != SDL_EVENT_FINGER_MOTION &&
             event->type != SDL_EVENT_PEN_MOTION &&
             event->type != SDL_EVENT_PEN_AXIS &&
             event->type != SDL_EVENT_JOYSTICK_AXIS_MOTION) ||
            (state->verbose & VERBOSE_MOTION)) {
            SDLTest_PrintEvent(event);
        }
    }

    switch (event->type) {
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
        if (state->auto_scale_content) {
            SDL_Window *window = SDL_GetWindowFromEvent(event);
            if (window) {
                float scale = SDL_GetDisplayContentScale(SDL_GetDisplayForWindow(window));
                int w = state->window_w;
                int h = state->window_h;

                w = (int)SDL_ceilf(w * scale);
                h = (int)SDL_ceilf(h * scale);
                SDL_SetWindowSize(window, w, h);
            }
        }
        break;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        if (state->flash_on_focus_loss) {
            SDL_Window *window = SDL_GetWindowFromEvent(event);
            if (window) {
                SDL_FlashWindow(window, SDL_FLASH_UNTIL_FOCUSED);
            }
        }
        break;
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
    {
        SDL_Window *window = SDL_GetWindowFromEvent(event);
        if (window) {
            SDL_HideWindow(window);
        }
        break;
    }
    case SDL_EVENT_KEY_DOWN:
    {
        bool withControl = !!(event->key.mod & SDL_KMOD_CTRL);
        bool withShift = !!(event->key.mod & SDL_KMOD_SHIFT);
        bool withAlt = !!(event->key.mod & SDL_KMOD_ALT);

        switch (event->key.key) {
            /* Add hotkeys here */
        case SDLK_PRINTSCREEN:
        {
            SDL_Window *window = SDL_GetWindowFromEvent(event);
            if (window) {
                for (i = 0; i < state->num_windows; ++i) {
                    if (window == state->windows[i]) {
                        SDLTest_CopyScreenShot(state->renderers[i]);
                    }
                }
            }
        } break;
        case SDLK_EQUALS:
            if (withControl) {
                /* Ctrl-+ double the size of the window */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    int w, h;
                    SDL_GetWindowSize(window, &w, &h);
                    SDL_SetWindowSize(window, w * 2, h * 2);
                }
            }
            break;
        case SDLK_MINUS:
            if (withControl) {
                /* Ctrl-- half the size of the window */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    int w, h;
                    SDL_GetWindowSize(window, &w, &h);
                    SDL_SetWindowSize(window, w / 2, h / 2);
                }
            }
            break;
        case SDLK_UP:
        case SDLK_DOWN:
        case SDLK_LEFT:
        case SDLK_RIGHT:
            if (withAlt) {
                /* Alt-Up/Down/Left/Right switches between displays */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    int num_displays;
                    const SDL_DisplayID *displays = SDL_GetDisplays(&num_displays);
                    if (displays) {
                        SDL_DisplayID displayID = SDL_GetDisplayForWindow(window);
                        int current_index = -1;

                        for (i = 0; i < num_displays; ++i) {
                            if (displayID == displays[i]) {
                                current_index = i;
                                break;
                            }
                        }
                        if (current_index >= 0) {
                            SDL_DisplayID dest;
                            if (event->key.key == SDLK_UP || event->key.key == SDLK_LEFT) {
                                dest = displays[(current_index + num_displays - 1) % num_displays];
                            } else {
                                dest = displays[(current_index + num_displays + 1) % num_displays];
                            }
                            SDL_Log("Centering on display (%" SDL_PRIu32 ")", dest);
                            SDL_SetWindowPosition(window,
                                                  SDL_WINDOWPOS_CENTERED_DISPLAY(dest),
                                                  SDL_WINDOWPOS_CENTERED_DISPLAY(dest));
                        }
                    }
                }
            }
            if (withShift) {
                /* Shift-Up/Down/Left/Right shift the window by 100px */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    const int delta = 100;
                    int x, y;
                    SDL_GetWindowPosition(window, &x, &y);

                    if (event->key.key == SDLK_UP) {
                        y -= delta;
                    }
                    if (event->key.key == SDLK_DOWN) {
                        y += delta;
                    }
                    if (event->key.key == SDLK_LEFT) {
                        x -= delta;
                    }
                    if (event->key.key == SDLK_RIGHT) {
                        x += delta;
                    }

                    SDL_Log("Setting position to (%d, %d)", x, y);
                    SDL_SetWindowPosition(window, x, y);
                }
            }
            break;
        case SDLK_O:
            if (withControl) {
                /* Ctrl-O (or Ctrl-Shift-O) changes window opacity. */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    float opacity = SDL_GetWindowOpacity(window);
                    if (withShift) {
                        opacity += 0.20f;
                    } else {
                        opacity -= 0.20f;
                    }
                    SDL_SetWindowOpacity(window, opacity);
                }
            }
            break;
        case SDLK_H:
            if (withControl) {
                /* Ctrl-H changes cursor visibility. */
                if (SDL_CursorVisible()) {
                    SDL_HideCursor();
                } else {
                    SDL_ShowCursor();
                }
            }
            break;
        case SDLK_C:
            if (withAlt) {
                /* Alt-C copy awesome text to the primary selection! */
                SDL_SetPrimarySelectionText("SDL rocks!\nYou know it!");
                SDL_Log("Copied text to primary selection");

            } else if (withControl) {
                if (withShift) {
                    /* Ctrl-Shift-C copy screenshot! */
                    SDL_Window *window = SDL_GetWindowFromEvent(event);
                    if (window) {
                        for (i = 0; i < state->num_windows; ++i) {
                            if (window == state->windows[i]) {
                                SDLTest_CopyScreenShot(state->renderers[i]);
                            }
                        }
                    }
                } else {
                    /* Ctrl-C copy awesome text! */
                    SDL_SetClipboardText("SDL rocks!\nYou know it!");
                    SDL_Log("Copied text to clipboard");
                }
                break;
            }
            break;
        case SDLK_V:
            if (withAlt) {
                /* Alt-V paste awesome text from the primary selection! */
                char *text = SDL_GetPrimarySelectionText();
                if (*text) {
                    SDL_Log("Primary selection: %s", text);
                } else {
                    SDL_Log("Primary selection is empty");
                }
                SDL_free(text);

            } else if (withControl) {
                if (withShift) {
                    /* Ctrl-Shift-V paste screenshot! */
                    SDLTest_PasteScreenShot();
                } else {
                    /* Ctrl-V paste awesome text! */
                    char *text = SDL_GetClipboardText();
                    if (*text) {
                        SDL_Log("Clipboard: %s", text);
                    } else {
                        SDL_Log("Clipboard is empty");
                    }
                    SDL_free(text);
                }
            }
            break;
        case SDLK_F:
            if (withControl) {
                /* Ctrl-F flash the window */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    SDL_FlashWindow(window, SDL_FLASH_BRIEFLY);
                }
            }
            break;
        case SDLK_P:
            if (withAlt) {
                /* Alt-P cycle through progress states */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    const char *name;
                    SDL_ProgressState progress_state = SDL_GetWindowProgressState(window);
                    progress_state += 1;
                    if (progress_state > SDL_PROGRESS_STATE_ERROR) {
                        progress_state = SDL_PROGRESS_STATE_NONE;
                    }
                    switch (progress_state) {
                    case SDL_PROGRESS_STATE_NONE:
                        name = "NONE";
                        break;
                    case SDL_PROGRESS_STATE_INDETERMINATE:
                        name = "INDETERMINATE";
                        break;
                    case SDL_PROGRESS_STATE_NORMAL:
                        name = "NORMAL";
                        break;
                    case SDL_PROGRESS_STATE_PAUSED:
                        name = "PAUSED";
                        break;
                    case SDL_PROGRESS_STATE_ERROR:
                        name = "ERROR";
                        break;
                    default:
                        name = "UNKNOWN";
                        break;
                    }
                    SDL_Log("Setting progress state to %s", name);
                    SDL_SetWindowProgressState(window, progress_state);
                }
            }
            else if (withControl)
            {
                /* Ctrl-P increase progress value */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    float progress_value = SDL_GetWindowProgressValue(window);
                    if (withShift) {
                        progress_value -= 0.1f;
                    } else {
                        progress_value += 0.1f;
                    }
                    SDL_Log("Setting progress value to %.1f", progress_value);
                    SDL_SetWindowProgressValue(window, progress_value);
                }
            }
            break;
        case SDLK_G:
            if (withControl) {
                /* Ctrl-G toggle mouse grab */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    if (SDL_RectEmpty(SDL_GetWindowMouseRect(window))) {
                        SDL_Rect r = { 10, 10, 200, 200};
                        SDL_SetWindowMouseRect(window, &r);
                    } else {
                        SDL_SetWindowMouseRect(window, NULL);
                    }
                    //SDL_SetWindowMouseGrab(window, !SDL_GetWindowMouseGrab(window));
                }
            }
            break;
        case SDLK_K:
            if (withControl) {
                /* Ctrl-K toggle keyboard grab */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    SDL_SetWindowKeyboardGrab(window, !SDL_GetWindowKeyboardGrab(window));
                }
            }
            break;
        case SDLK_M:
            if (withControl) {
                /* Ctrl-M maximize */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    SDL_WindowFlags flags = SDL_GetWindowFlags(window);
                    if (!(flags & SDL_WINDOW_RESIZABLE)) {
                        SDL_SetWindowResizable(window, true);
                    }
                    if (flags & SDL_WINDOW_MAXIMIZED) {
                        SDL_RestoreWindow(window);
                    } else {
                        SDL_MaximizeWindow(window);
                    }
                    if (!(flags & SDL_WINDOW_RESIZABLE)) {
                        SDL_SetWindowResizable(window, false);
                    }
                }
            }
            if (withShift) {
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    const bool shouldCapture = !(SDL_GetWindowFlags(window) & SDL_WINDOW_MOUSE_CAPTURE);
                    const bool rc = SDL_CaptureMouse(shouldCapture);
                    SDL_Log("%sapturing mouse %s!", shouldCapture ? "C" : "Unc", rc ? "succeeded" : "failed");
                }
            }
            break;
        case SDLK_R:
            if (withControl) {
                /* Ctrl-R toggle mouse relative mode */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    SDL_SetWindowRelativeMouseMode(window, !SDL_GetWindowRelativeMouseMode(window));
                }
            }
            break;
        case SDLK_T:
            if (withControl) {
                /* Ctrl-T toggle topmost mode */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    SDL_WindowFlags flags = SDL_GetWindowFlags(window);
                    if (flags & SDL_WINDOW_ALWAYS_ON_TOP) {
                        SDL_SetWindowAlwaysOnTop(window, false);
                    } else {
                        SDL_SetWindowAlwaysOnTop(window, true);
                    }
                }
            }
            break;
        case SDLK_Z:
            if (withControl) {
                /* Ctrl-Z minimize */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    SDL_MinimizeWindow(window);
                }
            }
            break;
        case SDLK_RETURN:
            if (withControl) {
                /* Ctrl-Enter toggle fullscreen */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    SDL_WindowFlags flags = SDL_GetWindowFlags(window);
                    if (!(flags & SDL_WINDOW_FULLSCREEN) ||
                        !SDL_GetWindowFullscreenMode(window)) {
                        SDL_SetWindowFullscreenMode(window, &state->fullscreen_mode);
                        SDL_SetWindowFullscreen(window, true);
                    } else {
                        SDL_SetWindowFullscreen(window, false);
                    }
                }
            } else if (withAlt) {
                /* Alt-Enter toggle fullscreen desktop */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    SDL_WindowFlags flags = SDL_GetWindowFlags(window);
                    if (!(flags & SDL_WINDOW_FULLSCREEN) ||
                        SDL_GetWindowFullscreenMode(window)) {
                        SDL_SetWindowFullscreenMode(window, NULL);
                        SDL_SetWindowFullscreen(window, true);
                    } else {
                        SDL_SetWindowFullscreen(window, false);
                    }
                }
            }

            break;
        case SDLK_B:
            if (withControl) {
                /* Ctrl-B toggle window border */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    const SDL_WindowFlags flags = SDL_GetWindowFlags(window);
                    const bool b = (flags & SDL_WINDOW_BORDERLESS) ? true : false;
                    SDL_SetWindowBordered(window, b);
                }
            }
            break;
        case SDLK_A:
            if (withControl) {
                /* Ctrl-A toggle aspect ratio */
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                if (window) {
                    float min_aspect = 0.0f, max_aspect = 0.0f;

                    SDL_GetWindowAspectRatio(window, &min_aspect, &max_aspect);
                    if (min_aspect > 0.0f || max_aspect > 0.0f) {
                        min_aspect = 0.0f;
                        max_aspect = 0.0f;
                    } else {
                        min_aspect = 1.0f;
                        max_aspect = 1.0f;
                    }
                    SDL_SetWindowAspectRatio(window, min_aspect, max_aspect);
                }
            }
            break;
        case SDLK_0:
            if (withControl) {
                SDL_Window *window = SDL_GetWindowFromEvent(event);
                SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Test Message", "You're awesome!", window);
            }
            break;
        case SDLK_1:
            if (withControl) {
                FullscreenTo(state, 0, event->key.windowID);
            }
            break;
        case SDLK_2:
            if (withControl) {
                FullscreenTo(state, 1, event->key.windowID);
            }
            break;
        case SDLK_ESCAPE:
            return SDL_APP_SUCCESS;
        default:
            break;
        }
        break;
    }
    case SDL_EVENT_QUIT:
        return SDL_APP_SUCCESS;
    default:
        break;
    }

    return SDL_APP_CONTINUE;
}

void SDLTest_CommonEvent(SDLTest_CommonState *state, SDL_Event *event, int *done)
{
    if (SDLTest_CommonEventMainCallbacks(state, event) != SDL_APP_CONTINUE) {
        *done = 1;
    }
}

void SDLTest_CommonQuit(SDLTest_CommonState *state)
{
    if (state) {
        int i;

        if (state->targets) {
            for (i = 0; i < state->num_windows; ++i) {
                if (state->targets[i]) {
                    SDL_DestroyTexture(state->targets[i]);
                }
            }
            SDL_free(state->targets);
        }
        if (state->renderers) {
            for (i = 0; i < state->num_windows; ++i) {
                if (state->renderers[i]) {
                    SDL_DestroyRenderer(state->renderers[i]);
                }
            }
            SDL_free(state->renderers);
        }
        if (state->windows) {
            for (i = 0; i < state->num_windows; i++) {
                SDL_DestroyWindow(state->windows[i]);
            }
            SDL_free(state->windows);
        }
    }
    SDL_Quit();
    SDLTest_CommonDestroyState(state);
}

void SDLTest_CommonDrawWindowInfo(SDL_Renderer *renderer, SDL_Window *window, float *usedHeight)
{
    char text[1024];
    float textY = 0.0f;
    const int lineHeight = 10;
    int x, y, w, h;
    float fx, fy;
    SDL_Rect rect;
    const SDL_DisplayMode *mode;
    float scaleX, scaleY;
    SDL_MouseButtonFlags flags;
    SDL_DisplayID windowDisplayID = SDL_GetDisplayForWindow(window);
    const char *name;
    SDL_RendererLogicalPresentation logical_presentation;

    /* Video */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0.0f, textY, "-- Video --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetCurrentVideoDriver: %s", SDL_GetCurrentVideoDriver());
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    /* Renderer */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0.0f, textY, "-- Renderer --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    name = SDL_GetRendererName(renderer);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetRendererName: %s", name);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    if (SDL_GetRenderOutputSize(renderer, &w, &h)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetRenderOutputSize: %dx%d", w, h);
        SDLTest_DrawString(renderer, 0.0f, textY, text);
        textY += lineHeight;
    }

    if (SDL_GetCurrentRenderOutputSize(renderer, &w, &h)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetCurrentRenderOutputSize: %dx%d", w, h);
        SDLTest_DrawString(renderer, 0.0f, textY, text);
        textY += lineHeight;
    }

    SDL_GetRenderViewport(renderer, &rect);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetRenderViewport: %d,%d, %dx%d",
                       rect.x, rect.y, rect.w, rect.h);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    SDL_GetRenderScale(renderer, &scaleX, &scaleY);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetRenderScale: %g,%g",
                       scaleX, scaleY);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    SDL_GetRenderLogicalPresentation(renderer, &w, &h, &logical_presentation);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetRenderLogicalPresentation: %dx%d ", w, h);
    SDLTest_PrintLogicalPresentation(text, sizeof(text), logical_presentation);
    textY += lineHeight;

    /* Window */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0.0f, textY, "-- Window --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    SDL_GetWindowPosition(window, &x, &y);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowPosition: %d,%d", x, y);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    SDL_GetWindowSize(window, &w, &h);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowSize: %dx%d", w, h);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    SDL_GetWindowSafeArea(window, &rect);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowSafeArea: %d,%d %dx%d", rect.x, rect.y, rect.w, rect.h);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowFlags: ");
    SDLTest_PrintWindowFlags(text, sizeof(text), SDL_GetWindowFlags(window));
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    mode = SDL_GetWindowFullscreenMode(window);
    if (mode) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetWindowFullscreenMode: %dx%d@%gx %gHz, (%s)",
                           mode->w, mode->h, mode->pixel_density, mode->refresh_rate, SDL_GetPixelFormatName(mode->format));
        SDLTest_DrawString(renderer, 0.0f, textY, text);
        textY += lineHeight;
    }

    /* Display */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0.0f, textY, "-- Display --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetDisplayForWindow: %" SDL_PRIu32, windowDisplayID);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetDisplayName: %s", SDL_GetDisplayName(windowDisplayID));
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    if (SDL_GetDisplayBounds(windowDisplayID, &rect)) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetDisplayBounds: %d,%d, %dx%d",
                           rect.x, rect.y, rect.w, rect.h);
        SDLTest_DrawString(renderer, 0.0f, textY, text);
        textY += lineHeight;
    }

    mode = SDL_GetCurrentDisplayMode(windowDisplayID);
    if (mode) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetCurrentDisplayMode: %dx%d@%gx %gHz, (%s)",
                           mode->w, mode->h, mode->pixel_density, mode->refresh_rate, SDL_GetPixelFormatName(mode->format));
        SDLTest_DrawString(renderer, 0.0f, textY, text);
        textY += lineHeight;
    }

    mode = SDL_GetDesktopDisplayMode(windowDisplayID);
    if (mode) {
        (void)SDL_snprintf(text, sizeof(text), "SDL_GetDesktopDisplayMode: %dx%d@%gx %gHz, (%s)",
                           mode->w, mode->h, mode->pixel_density, mode->refresh_rate, SDL_GetPixelFormatName(mode->format));
        SDLTest_DrawString(renderer, 0.0f, textY, text);
        textY += lineHeight;
    }

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetNaturalDisplayOrientation: ");
    SDLTest_PrintDisplayOrientation(text, sizeof(text), SDL_GetNaturalDisplayOrientation(windowDisplayID));
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetCurrentDisplayOrientation: ");
    SDLTest_PrintDisplayOrientation(text, sizeof(text), SDL_GetCurrentDisplayOrientation(windowDisplayID));
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetDisplayContentScale: %g", SDL_GetDisplayContentScale(windowDisplayID));
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    /* Mouse */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0.0f, textY, "-- Mouse --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    flags = SDL_GetMouseState(&fx, &fy);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetMouseState: %g,%g ", fx, fy);
    SDLTest_PrintButtonMask(text, sizeof(text), flags);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    flags = SDL_GetGlobalMouseState(&fx, &fy);
    (void)SDL_snprintf(text, sizeof(text), "SDL_GetGlobalMouseState: %g,%g ", fx, fy);
    SDLTest_PrintButtonMask(text, sizeof(text), flags);
    SDLTest_DrawString(renderer, 0.0f, textY, text);
    textY += lineHeight;

    /* Keyboard */

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDLTest_DrawString(renderer, 0, textY, "-- Keyboard --");
    textY += lineHeight;

    SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);

    (void)SDL_snprintf(text, sizeof(text), "SDL_GetModState: ");
    SDLTest_PrintModState(text, sizeof(text), SDL_GetModState());
    SDLTest_DrawString(renderer, 0, textY, text);
    textY += lineHeight;

    if (usedHeight) {
        *usedHeight = textY;
    }
}

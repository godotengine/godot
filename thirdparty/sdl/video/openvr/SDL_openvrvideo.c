/*
  Simple DirectMedia Layer
  Copyright (C) 2022 Charles Lohr <charlesl@valvesoftware.com>

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
#include "SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_OPENVR

#define DEBUG_OPENVR

#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_events_c.h"
#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../SDL_egl_c.h"
#include "SDL_openvrvideo.h"

#include <SDL3/SDL_opengl.h>

#ifdef SDL_VIDEO_DRIVER_WINDOWS
#include "../windows/SDL_windowsopengles.h"
#include "../windows/SDL_windowsopengl.h"
#include "../windows/SDL_windowsvulkan.h"
#define DEFAULT_OPENGL "OPENGL32.DLL"
static bool OPENVR_GL_LoadLibrary(SDL_VideoDevice *_this, const char *path);
static SDL_GLContext OPENVR_GL_CreateContext(SDL_VideoDevice *_this, SDL_Window *window);

struct SDL_GLContextState
{
    HGLRC hglrc;
};

#else
#include <SDL3/SDL_opengles2_gl2.h>
#endif

#define MARKER_ID 0
#define MARKER_STR "vr-marker,frame_end,type,application"

#undef EXTERN_C

// For access to functions that don't get the video data context.
SDL_VideoData * global_openvr_driver;

static void InitializeMouseFunctions();

struct SDL_CursorData
{
    unsigned texture_id_handle;
    int hot_x, hot_y;
    int w, h;
};

// GL Extensions for functions we will be using.
static void (APIENTRY *ov_glGenFramebuffers)(GLsizei n, GLuint *framebuffers);
static void (APIENTRY *ov_glGenRenderbuffers)(GLsizei n, GLuint *renderbuffers);
static void (APIENTRY *ov_glBindFramebuffer)(GLenum target, GLuint framebuffer);
static void (APIENTRY *ov_glBindRenderbuffer)(GLenum target, GLuint renderbuffer);
static void (APIENTRY *ov_glRenderbufferStorage)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
static void (APIENTRY *ov_glFramebufferRenderbuffer)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
static void (APIENTRY *ov_glFramebufferTexture2D)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
static GLenum (APIENTRY *ov_glCheckNamedFramebufferStatus)(GLuint framebuffer, GLenum target);
static GLenum (APIENTRY *ov_glGetError)();
static void (APIENTRY *ov_glFlush)();
static void (APIENTRY *ov_glFinish)();
static void (APIENTRY *ov_glGenTextures)(GLsizei n, GLuint *textures);
static void (APIENTRY *ov_glDeleteTextures)(GLsizei n, GLuint *textures);
static void (APIENTRY *ov_glTexParameterf)(GLenum target, GLenum pname, GLfloat param);
static void (APIENTRY *ov_glTexParameteri)(GLenum target, GLenum pname, GLenum param);
static void (APIENTRY *ov_glTexImage2D)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *data);
static void (APIENTRY *ov_glBindTexture)(GLenum target, GLuint texture);
static void (APIENTRY *ov_glDrawBuffers)(GLsizei n, const GLenum *bufs);
static void (APIENTRY *ov_glGetIntegerv)(GLenum pname, GLint * data);
static const GLubyte *(APIENTRY *ov_glGetStringi)(GLenum name, GLuint index);
static void (APIENTRY *ov_glClear)(GLbitfield mask);
static void (APIENTRY *ov_glClearColor)(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
static void (APIENTRY *ov_glColorMask)(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
static void (APIENTRY *ov_glDebugMessageInsert)(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char *message);

#ifdef SDL_VIDEO_DRIVER_WINDOWS
static PROC (*ov_wglGetProcAddress)(LPCSTR);
static HGLRC (*ov_wglCreateContext)(HDC);
static BOOL (*ov_wglDeleteContext)(HGLRC);
static BOOL (*ov_wglMakeCurrent)(HDC, HGLRC);
//static HGLRC (*ov_wglGetCurrentContext)(void);
#endif


#define OPENVR_DEFAULT_WIDTH 1920
#define OPENVR_DEFAULT_HEIGHT 1080

#define OPENVR_SetupProc(proc) { proc = (void*)SDL_GL_GetProcAddress((#proc)+3); if (!proc) { failed_extension = (#proc)+3; } }

static bool OPENVR_InitExtensions(SDL_VideoDevice *_this)
{
    if (!ov_glGetError) {
        const char * failed_extension = 0;
        OPENVR_SetupProc(ov_glGenFramebuffers);
        OPENVR_SetupProc(ov_glGenRenderbuffers);
        OPENVR_SetupProc(ov_glBindFramebuffer);
        OPENVR_SetupProc(ov_glBindRenderbuffer);
        OPENVR_SetupProc(ov_glRenderbufferStorage);
        OPENVR_SetupProc(ov_glFramebufferRenderbuffer);
        OPENVR_SetupProc(ov_glFramebufferTexture2D);
        OPENVR_SetupProc(ov_glCheckNamedFramebufferStatus);
        OPENVR_SetupProc(ov_glGetError);
        OPENVR_SetupProc(ov_glFlush);
        OPENVR_SetupProc(ov_glFinish);
        OPENVR_SetupProc(ov_glGenTextures);
        OPENVR_SetupProc(ov_glDeleteTextures);
        OPENVR_SetupProc(ov_glTexParameterf);
        OPENVR_SetupProc(ov_glTexParameteri);
        OPENVR_SetupProc(ov_glTexImage2D);
        OPENVR_SetupProc(ov_glBindTexture);
        OPENVR_SetupProc(ov_glDrawBuffers);
        OPENVR_SetupProc(ov_glClear);
        OPENVR_SetupProc(ov_glClearColor);
        OPENVR_SetupProc(ov_glColorMask);
        OPENVR_SetupProc(ov_glGetStringi);
        OPENVR_SetupProc(ov_glGetIntegerv);
        OPENVR_SetupProc(ov_glDebugMessageInsert);
        if (failed_extension) {
            return SDL_SetError("Error loading GL extension for %s", failed_extension);
        }
    }
    return true;
}

static bool OPENVR_SetOverlayError(EVROverlayError e)
{
    switch (e) {
#define CASE(X) case EVROverlayError_VROverlayError_##X: return SDL_SetError("VROverlayError %s", #X)
    CASE(UnknownOverlay);
    CASE(InvalidHandle);
    CASE(PermissionDenied);
    CASE(OverlayLimitExceeded);
    CASE(WrongVisibilityType);
    CASE(KeyTooLong);
    CASE(NameTooLong);
    CASE(KeyInUse);
    CASE(WrongTransformType);
    CASE(InvalidTrackedDevice);
    CASE(InvalidParameter);
    CASE(ThumbnailCantBeDestroyed);
    CASE(ArrayTooSmall);
    CASE(RequestFailed);
    CASE(InvalidTexture);
    CASE(UnableToLoadFile);
    CASE(KeyboardAlreadyInUse);
    CASE(NoNeighbor);
    CASE(TooManyMaskPrimitives);
    CASE(BadMaskPrimitive);
    CASE(TextureAlreadyLocked);
    CASE(TextureLockCapacityReached);
    CASE(TextureNotLocked);
    CASE(TimedOut);
#undef CASE
    default:
        return SDL_SetError("Unknown VROverlayError %d", e);
    }
}

static bool OPENVR_InitializeOverlay(SDL_VideoDevice *_this, SDL_Window *window);

static bool OPENVR_VideoInit(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->internal;

    const char * hintWidth = SDL_GetHint("SDL_DEFAULT_WIDTH");
    const char * hintHeight = SDL_GetHint("SDL_DEFAULT_HEIGHT");
    const char * hintFPS = SDL_GetHint("SDL_DEFAULT_FPS");
    int width = hintWidth ? SDL_atoi(hintWidth) : 0;
    int height = hintHeight ? SDL_atoi(hintHeight) : 0;
    int fps = hintFPS ? SDL_atoi(hintFPS) : 0;

    SDL_VideoDisplay display;
    SDL_zero(display);
    display.desktop_mode.format = SDL_PIXELFORMAT_RGBA32;
    display.desktop_mode.w = OPENVR_DEFAULT_WIDTH;
    display.desktop_mode.h = OPENVR_DEFAULT_HEIGHT;
    display.natural_orientation = SDL_ORIENTATION_LANDSCAPE;
    display.current_orientation = SDL_ORIENTATION_LANDSCAPE;
    display.content_scale = 1.0f;
    if (height > 0 && width > 0) {
        display.desktop_mode.w = width;
        display.desktop_mode.h = height;
    }
    if (fps) {
        display.desktop_mode.refresh_rate = fps;
    } else {
        display.desktop_mode.refresh_rate = data->oSystem->GetFloatTrackedDeviceProperty(k_unTrackedDeviceIndex_Hmd, ETrackedDeviceProperty_Prop_DisplayFrequency_Float, 0);
    }

    display.internal = (SDL_DisplayData *)data;
    display.name = (char*)"OpenVRDisplay";
    SDL_AddVideoDisplay(&display, false);

    return true;
}

static void OPENVR_VideoQuit(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    if (videodata->bDidCreateOverlay && videodata->overlayID != 0) {
        videodata->oOverlay->DestroyOverlay(videodata->overlayID);
    }
}

static void OPENVR_Destroy(SDL_VideoDevice *device)
{
    SDL_VideoData *data = device->internal;

#ifdef SDL_PLATFORM_WINDOWS
    SDL_UnregisterApp();
#endif

    if (data) {
        if (data->openVRLIB) {
            SDL_UnloadObject(data->openVRLIB);
        }
    }
    SDL_free(device->internal);
    SDL_free(device);
}

static uint32_t *ImageSDLToOpenVRGL(SDL_Surface * surf, bool bFlipY)
{
    int w = surf->w;
    int h = surf->h;
    int pitch = surf->pitch;
    int x, y;
    uint32_t * pxd = SDL_malloc(4 * surf->w * surf->h);
    for(y = 0; y < h; y++) {
        uint32_t * iline = (uint32_t*)&(((uint8_t*)surf->pixels)[y*pitch]);
        uint32_t * oline = &pxd[(bFlipY?(h-y-1):y)*w];
        for(x = 0; x < w; x++)
        {
            uint32_t pr = iline[x];
            oline[x] = (pr & 0xff00ff00) | ((pr & 0xff) << 16) | ((pr & 0xff0000)>>16);
        }
    }
    return pxd;
}

static bool OPENVR_CheckRenderbuffer(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;

    if (videodata->targw == 0 || videodata->targh == 0) {
        videodata->targw = OPENVR_DEFAULT_WIDTH;
        videodata->targh = OPENVR_DEFAULT_HEIGHT;
    }

    if (videodata->targh != videodata->last_targh
     || videodata->targw != videodata->last_targw) {

        struct HmdVector2_t ms;
        int status;

        if (videodata->fbo <= 0) {
            ov_glGenFramebuffers(1, &videodata->fbo);
            ov_glGenRenderbuffers(1, &videodata->rbo);
            ov_glGenTextures(1, &videodata->overlaytexture);
        }

        // Generate the OpenGL Backing buffers/etc.
        ov_glBindFramebuffer(GL_FRAMEBUFFER, videodata->fbo);
        ov_glBindRenderbuffer(GL_RENDERBUFFER, videodata->rbo);
        ov_glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, videodata->targw, videodata->targh);
        ov_glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, videodata->rbo);
        ov_glBindTexture(GL_TEXTURE_2D, videodata->overlaytexture);
        ov_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        ov_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        ov_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, videodata->targw, videodata->targh, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        ov_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, videodata->overlaytexture, 0);
        status = ov_glCheckNamedFramebufferStatus(videodata->fbo, GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            return SDL_SetError("OPENVR: Can't generate overlay buffer");
        }
        ov_glBindFramebuffer(GL_FRAMEBUFFER, 0);

        ms.v[0] = (float)videodata->targw;
        ms.v[1] = (float)videodata->targh;
        videodata->oOverlay->SetOverlayMouseScale(videodata->overlayID, &ms);

        videodata->last_targh = videodata->targh;
        videodata->last_targw = videodata->targw;
    }
    return true;
}

static bool OPENVR_VirtualControllerRumble(void *userdata, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    // On XBOX Controllers Low/High maps to Left/Right
    SDL_VideoData *videodata = (SDL_VideoData *)userdata;

    const float k_flIntensity = 320.f; // Maximum frequency
    float flLeftFrequency = (float)low_frequency_rumble * k_flIntensity / 65535.f;
    float flRightFrequency = (float)high_frequency_rumble * k_flIntensity / 65535.f;
    float flDurationSeconds = 2.f;
    float flAmplitude = 1.f;

    videodata->oInput->TriggerHapticVibrationAction(videodata->input_action_handles_haptics[0], 0, flDurationSeconds, flLeftFrequency, flAmplitude, 0);
    videodata->oInput->TriggerHapticVibrationAction(videodata->input_action_handles_haptics[1], 0, flDurationSeconds, flRightFrequency, flAmplitude, 0);

    return true;
}

static bool OPENVR_VirtualControllerRumbleTriggers(void *userdata, Uint16 left_rumble, Uint16 right_rumble)
{
    SDL_VideoData *videodata = (SDL_VideoData *)userdata;
    videodata->oInput->TriggerHapticVibrationAction(videodata->input_action_handles_haptics[0], 0, 0.1f, left_rumble, 1.0, 0);
    videodata->oInput->TriggerHapticVibrationAction(videodata->input_action_handles_haptics[1], 0, 0.1f, right_rumble, 1.0, 0);
    return true;
}

static void OPENVR_VirtualControllerUpdate(void *userdata)
{
    SDL_VideoData *videodata = (SDL_VideoData *)userdata;
    SDL_Joystick * joystick = videodata->virtual_joystick;
    InputDigitalActionData_t digital_input_action;
    InputAnalogActionData_t analog_input_action;
    EVRInputError e;
#ifdef DEBUG_OPENVR
    //char cts[10240];
    //char * ctsx = cts;
#endif
    VRActiveActionSet_t actionSet = { 0 };
    actionSet.ulActionSet = videodata->input_action_set;
    e = videodata->oInput->UpdateActionState(&actionSet, sizeof(actionSet), 1);
    if (e)
    {
#ifdef DEBUG_OPENVR
        SDL_Log("ERROR: Failed to update action state");
#endif
        return;
    }

    for (int d = 0; d < videodata->input_action_handles_buttons_count; d++)
    {
        if (videodata->input_action_handles_buttons[d] == k_ulInvalidActionHandle)
            continue;
        e = videodata->oInput->GetDigitalActionData(videodata->input_action_handles_buttons[d], &digital_input_action, sizeof(digital_input_action), k_ulInvalidInputValueHandle);
        if (e)
        {
#ifdef DEBUG_OPENVR
            SDL_Log("ERROR: Failed to get digital action data: %d", d);
#endif
            return;
        }
        SDL_SetJoystickVirtualButton(joystick, d, digital_input_action.bState);
#ifdef DEBUG_OPENVR
        //ctsx+=sprintf(ctsx,"%d", digital_input_action.bState);
#endif
    }

    // Left Stick
    e = videodata->oInput->GetAnalogActionData(videodata->input_action_handles_axes[0], &analog_input_action, sizeof(analog_input_action), k_ulInvalidInputValueHandle);
    if (e)
    {
#ifdef DEBUG_OPENVR
        SDL_Log("ERROR: Failed to get analog action data: left stick");
#endif
        return;
    }
    SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_LEFTX, (Sint16)(analog_input_action.x * SDL_JOYSTICK_AXIS_MAX));
    SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_LEFTY, (Sint16)(-analog_input_action.y * SDL_JOYSTICK_AXIS_MAX));

    // Right Stick
    e = videodata->oInput->GetAnalogActionData(videodata->input_action_handles_axes[1], &analog_input_action, sizeof(analog_input_action), k_ulInvalidInputValueHandle);
    if (e)
    {
#ifdef DEBUG_OPENVR
        SDL_Log("ERROR: Failed to get analog action data: right stick");
#endif
        return;
    }
    SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_RIGHTX, (Sint16)(analog_input_action.x * SDL_JOYSTICK_AXIS_MAX));
    SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_RIGHTY, (Sint16)(-analog_input_action.y * SDL_JOYSTICK_AXIS_MAX));

    // Left Trigger
    e = videodata->oInput->GetAnalogActionData(videodata->input_action_handles_axes[2], &analog_input_action, sizeof(analog_input_action), k_ulInvalidInputValueHandle);
    if (e)
    {
#ifdef DEBUG_OPENVR
        SDL_Log("ERROR: Failed to get analog action data: left trigger");
#endif
        return;
    }
    SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, (Sint16)((analog_input_action.x * 2.0f - 1.0f) * SDL_JOYSTICK_AXIS_MAX));

    // Right Trigger
    e = videodata->oInput->GetAnalogActionData(videodata->input_action_handles_axes[3], &analog_input_action, sizeof(analog_input_action), k_ulInvalidInputValueHandle);
    if (e)
    {
#ifdef DEBUG_OPENVR
        SDL_Log("ERROR: Failed to get analog action data: right trigger");
#endif
        return;
    }
    SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, (Sint16)((analog_input_action.x * 2.0f - 1.0f) * SDL_JOYSTICK_AXIS_MAX));

#if 0
    for (a = 0; a < videodata->input_action_handles_axes_count; a++)
    {
        float xval = 0.0f;
        e = videodata->oInput->GetAnalogActionData(videodata->input_action_handles_axes[a], &analog_input_action, sizeof(analog_input_action), k_ulInvalidInputValueHandle);
        if (e) goto updatefail;
        xval = analog_input_action.x;
        if (a == SDL_CONTROLLER_AXIS_LEFTY || a == SDL_CONTROLLER_AXIS_RIGHTY)
          xval *= -1.0f;
        if (a == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || a == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)
          xval = xval * 2.0f - 1.0f;
        //SDL_SetJoystickVirtualAxis(joystick, a, analog_input_action.x*32767);
        xval *= SDL_JOYSTICK_AXIS_MAX;
        SDL_SetJoystickVirtualAxis(joystick, a, xval);
#ifdef DEBUG_OPENVR
        //ctsx+=sprintf(ctsx,"[%f]", analog_input_action.x);
#endif
    }
#endif
#ifdef DEBUG_OPENVR
    //SDL_Log("Debug Input States: %s", cts);
#endif
    return;
}

static bool OPENVR_SetupJoystickBasedOnLoadedActionManifest(SDL_VideoData * videodata)
{
    SDL_VirtualJoystickDesc desc;
    int virtual_index;

    EVRInputError e = 0;

    char * k_pchBooleanActionPaths[SDL_GAMEPAD_BUTTON_COUNT] = {
        "/actions/virtualgamepad/in/a",
        "/actions/virtualgamepad/in/b",
        "/actions/virtualgamepad/in/x",
        "/actions/virtualgamepad/in/y",
        "/actions/virtualgamepad/in/back",
        "/actions/virtualgamepad/in/guide",
        "/actions/virtualgamepad/in/start",
        "/actions/virtualgamepad/in/stick_click_left",
        "/actions/virtualgamepad/in/stick_click_right",
        "/actions/virtualgamepad/in/shoulder_left",
        "/actions/virtualgamepad/in/shoulder_right",
        "/actions/virtualgamepad/in/dpad_up",
        "/actions/virtualgamepad/in/dpad_down",
        "/actions/virtualgamepad/in/dpad_left",
        "/actions/virtualgamepad/in/dpad_right",
        "/actions/virtualgamepad/in/misc_1",
        "/actions/virtualgamepad/in/paddle_1",
        "/actions/virtualgamepad/in/paddle_2",
        "/actions/virtualgamepad/in/paddle_3",
        "/actions/virtualgamepad/in/paddle_4",
        "/actions/virtualgamepad/in/touchpad_click",
        "/actions/virtualgamepad/in/misc_2",
        "/actions/virtualgamepad/in/misc_3",
        "/actions/virtualgamepad/in/misc_4",
        "/actions/virtualgamepad/in/misc_5",
        "/actions/virtualgamepad/in/misc_6",
    };
    char * k_pchAnalogActionPaths[4] = {
        "/actions/virtualgamepad/in/stick_left",
        "/actions/virtualgamepad/in/stick_right",
        "/actions/virtualgamepad/in/trigger_left",
        "/actions/virtualgamepad/in/trigger_right",
    };

    if ((e = videodata->oInput->GetActionSetHandle("/actions/virtualgamepad", &videodata->input_action_set)) != EVRInputError_VRInputError_None)
    {
#ifdef DEBUG_OPENVR
        SDL_Log("ERROR: Failed to get action set handle: %d", e);
#endif
        return SDL_SetError("Failed to get action set handle");
    }

    videodata->input_action_handles_buttons_count = sizeof(k_pchBooleanActionPaths) / sizeof(k_pchBooleanActionPaths[0]);
    videodata->input_action_handles_buttons = SDL_malloc(videodata->input_action_handles_buttons_count * sizeof(VRActionHandle_t));

    for (int i = 0; i < videodata->input_action_handles_buttons_count; i++)
    {
        e = videodata->oInput->GetActionHandle(k_pchBooleanActionPaths[i], &videodata->input_action_handles_buttons[i]);
        if (e)
        {
            SDL_Log("ERROR: Failed to get button action %d ('%s')", i, k_pchBooleanActionPaths[i]);
            return SDL_SetError("ERROR: Failed to get button action");
        }
    }

    videodata->input_action_handles_axes_count = sizeof(k_pchAnalogActionPaths) / sizeof(k_pchAnalogActionPaths[0]);
    videodata->input_action_handles_axes = SDL_malloc(videodata->input_action_handles_axes_count * sizeof(VRActionHandle_t));

    for (int i = 0; i < videodata->input_action_handles_axes_count; i++)
    {
        e = videodata->oInput->GetActionHandle(k_pchAnalogActionPaths[i], &videodata->input_action_handles_axes[i]);
        if (e)
        {
            SDL_Log("ERROR: Failed to get analog action %d ('%s')", i, k_pchAnalogActionPaths[i]);
            return SDL_SetError("ERROR: Failed to get analog action");
        }
    }

    e  = videodata->oInput->GetActionHandle("/actions/virtualgamepad/out/haptic_left", &videodata->input_action_handles_haptics[0]);
    e |= videodata->oInput->GetActionHandle("/actions/virtualgamepad/out/haptic_right", &videodata->input_action_handles_haptics[1]);
    if (e)
    {
#ifdef DEBUG_OPENVR
        SDL_Log("ERROR: Failed to get haptics action");
#endif
        return SDL_SetError("ERROR: Failed to get haptics action");
    }

    // Create a virtual joystick.
    SDL_INIT_INTERFACE(&desc);
    desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
    desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    desc.Rumble = OPENVR_VirtualControllerRumble;
    desc.RumbleTriggers = OPENVR_VirtualControllerRumbleTriggers;
    desc.Update = OPENVR_VirtualControllerUpdate;
    desc.userdata = videodata;
    virtual_index = SDL_AttachVirtualJoystick(&desc);

    if (virtual_index < 0) {
        return SDL_SetError("OPENVR: Couldn't open virtual joystick device: %s", SDL_GetError());
    } else {
        videodata->virtual_joystick = SDL_OpenJoystick(virtual_index);
        if (!videodata->virtual_joystick) {
            return SDL_SetError("OPENVR: Couldn't open virtual joystick device: %s", SDL_GetError());
        }
    }

#ifdef DEBUG_OPENVR
    SDL_Log("Loaded virtual joystick with %d buttons and %d axes", videodata->input_action_handles_buttons_count, videodata->input_action_handles_axes_count);
#endif

    return false;
}

static bool OPENVR_InitializeOverlay(SDL_VideoDevice *_this,SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;

    // Wait til here, to make sure we have our context setup correctly.
    if (!OPENVR_InitExtensions(_this)) {
        return false;
    }

    // Generate the overlay.
    {
        const char * hint = SDL_GetHint("SDL_OPENVR_OVERLAY_NAME");
        char * cursorname = 0;
        if (!hint) {
            hint = "sdl";
        }

        SDL_asprintf(&videodata->sOverlayName, "%s-overlay",hint);
        if (!videodata->sOverlayName) {
            return false;
        }
        SDL_asprintf(&cursorname, "%s-cursor",hint);
        if (!cursorname) {
            return false;
        }

        EVROverlayError result = videodata->oOverlay->CreateDashboardOverlay(videodata->sOverlayName,
            window->title, &videodata->overlayID, &videodata->thumbID);
        if (result != EVROverlayError_VROverlayError_None) {
            SDL_free(cursorname);
            return SDL_SetError("Could not create dashboard overlay (%d)", result );
        }
        result = videodata->oOverlay->CreateOverlay(cursorname, window->title, &videodata->cursorID);
        if (result != EVROverlayError_VROverlayError_None) {
            SDL_free(cursorname);
            return SDL_SetError("Could not create cursor overlay (%d)", result );
        }
        SDL_PropertiesID props = SDL_GetWindowProperties(window);
        SDL_SetNumberProperty(props, SDL_PROP_WINDOW_OPENVR_OVERLAY_ID_NUMBER, videodata->overlayID);
        SDL_free(cursorname);
        videodata->bHasShownOverlay = false;
    }
    {
        const char * hint = SDL_GetHint("SDL_OPENVR_OVERLAY_PANEL_WIDTH");
        float fWidth = hint ? (float)SDL_atof(hint) : 1.0f;
        videodata->oOverlay->SetOverlayWidthInMeters(videodata->overlayID, fWidth);
    }
    {
        const char * hint = SDL_GetHint("SDL_OPENVR_CURSOR_WIDTH");
        // Default is what SteamVR Does
        float fCursorWidth = hint ? (float)SDL_atof(hint) : 0.06f;
        videodata->oOverlay->SetOverlayWidthInMeters(videodata->cursorID, fCursorWidth * 0.5f);
    }
    {
        const char * hint = SDL_GetHint("SDL_OPENVR_WINDOW_ICON_FILE");
        videodata->bIconOverridden = false;
        if (hint) {
            char * tmpcopy = SDL_strdup(hint);
            EVROverlayError err = videodata->oOverlay->SetOverlayFromFile(videodata->thumbID, tmpcopy);
            SDL_free(tmpcopy);
            if (err == EVROverlayError_VROverlayError_None) {
                videodata->bIconOverridden = SDL_GetHintBoolean("SDL_OPENVR_WINDOW_ICON_OVERRIDE",false);
            }
        }
    }
    {
        VRTextureBounds_t bounds;
        bounds.uMin = 0;
        bounds.uMax = 1;
        bounds.vMin = 0;
        bounds.vMax = 1;
        videodata->oOverlay->SetOverlayTextureBounds(videodata->overlayID, &bounds);
    }

    if (!OPENVR_CheckRenderbuffer(_this)) {
        return false;
    }


    global_openvr_driver = videodata;
    InitializeMouseFunctions();

    // Actually show the overlay.
    videodata->oOverlay->SetOverlayFlag(videodata->overlayID, 1<<23, true); //vr::VROverlayFlags_EnableControlBar
    videodata->oOverlay->SetOverlayFlag(videodata->overlayID, 1<<24, true); //vr::VROverlayFlags_EnableControlBarKeyboard
    videodata->oOverlay->SetOverlayFlag(videodata->overlayID, 1<<25, true); //vr::VROverlayFlags_EnableControlBarClose
    videodata->oOverlay->SetOverlayName(videodata->overlayID, window->title);

    videodata->bDidCreateOverlay = true;
    videodata->window = window;

    return true;
}


static bool OPENVR_SetupFrame(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    static const GLenum buffers[8] = { GL_COLOR_ATTACHMENT0_EXT };

    videodata->is_buffer_rendering = true;

#ifdef DEBUG_OPENVR
    {
        int error = ov_glGetError();
        if (error)
            SDL_Log("Found GL Error before beginning frame: %d / (Framebuffer:%d)", error, ov_glCheckNamedFramebufferStatus(videodata->fbo, GL_FRAMEBUFFER));
    }
#endif

    ov_glBindFramebuffer(GL_FRAMEBUFFER, videodata->fbo);
    ov_glDrawBuffers(1, buffers);

    // Set the alpha channel for non-transparent windows
    if (!(window->flags & SDL_WINDOW_TRANSPARENT)) {
        ov_glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        ov_glColorMask(false, false, false, true);
        ov_glClear(GL_COLOR_BUFFER_BIT);
        ov_glColorMask(true, true, true, true);
    }

    ov_glBindTexture( GL_TEXTURE_2D, videodata->saved_texture_state );

    return true;
}

static bool OPENVR_ReleaseFrame(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    ov_glGetIntegerv(GL_TEXTURE_BINDING_2D, &videodata->saved_texture_state);

    if (!ov_glGetError) {
        return true;
    }

    if (!videodata->is_buffer_rendering) {
        return true;
    }

#ifdef DEBUG_OPENVR
    {
        int error = ov_glGetError();
        if (error) {
            SDL_Log("Found GL Error before release frame: %d / (Framebuffer:%d)", error, ov_glCheckNamedFramebufferStatus(videodata->fbo, GL_FRAMEBUFFER));
        }
    }
#endif

    videodata->is_buffer_rendering = false;

    ov_glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (videodata->overlaytexture != 0 &&
        videodata->targh == videodata->last_targh &&
        videodata->targw == videodata->last_targw) {
        // Only submit frames to OpenVR if the textu re exists.
        struct Texture_t tex;

        // Setup a Texture_t object to send in the texture.
        tex.eColorSpace = EColorSpace_ColorSpace_Auto;
        tex.eType = ETextureType_TextureType_OpenGL;
        tex.handle = (void *)(intptr_t)videodata->overlaytexture;

        // Send texture into OpenVR as the overlay.
        videodata->oOverlay->SetOverlayTexture(videodata->overlayID, &tex);
    }

    if (!videodata->bHasShownOverlay && videodata->bDidCreateOverlay) {
        videodata->oOverlay->ShowDashboard(videodata->sOverlayName);
        videodata->bHasShownOverlay = true;
    }

    if (videodata->renderdoc_debugmarker_frame_end) {
        ov_glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION,
            GL_DEBUG_TYPE_MARKER, MARKER_ID, GL_DEBUG_SEVERITY_NOTIFICATION, -1,
            MARKER_STR);
    }

    return OPENVR_CheckRenderbuffer(_this);
}

static void OPENVR_HandleResize(SDL_VideoDevice *_this, int w, int h)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->internal;
    data->targw = w;
    data->targh = h;
}

static bool OPENVR_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    return true;
}


#ifdef SDL_VIDEO_DRIVER_WINDOWS
static LRESULT CALLBACK OpenVRVideoWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_DESTROY:
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

static bool OPENVR_GL_LoadLibrary(SDL_VideoDevice *_this, const char *path)
{
    if (path == NULL) {
        path = SDL_GetHint(SDL_HINT_OPENGL_LIBRARY);
    }

    if (path == NULL) {
        path = DEFAULT_OPENGL;
    }
    _this->gl_config.dll_handle = SDL_LoadObject(path);
    if (!_this->gl_config.dll_handle) {
        return false;
    }
    SDL_strlcpy(_this->gl_config.driver_path, path,
                 SDL_arraysize(_this->gl_config.driver_path));

    // Allocate OpenGL memory
    _this->gl_data = (struct SDL_GLDriverData *)SDL_calloc(1, sizeof(struct SDL_GLDriverData));
    if (!_this->gl_data) {
        return false;
    }
    _this->gl_config.driver_loaded = true;

    return true;
}

static SDL_FunctionPointer OPENVR_GL_GetProcAddress(SDL_VideoDevice *_this, const char *proc)
{
    SDL_FunctionPointer result = NULL;
    if (ov_wglGetProcAddress) {
        result = (SDL_FunctionPointer)ov_wglGetProcAddress(proc);
        if (result) {
            return result;
        }
    }

    return SDL_LoadFunction(_this->gl_config.dll_handle, proc);
}

static void OPENVR_GL_UnloadLibrary(SDL_VideoDevice *_this)
{
    SDL_GL_UnloadLibrary();
}

static SDL_GLContext OPENVR_GL_CreateContext(SDL_VideoDevice *_this, SDL_Window *window)
{
    GLint numExtensions;
    int i;
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    if (!videodata->hglrc) {
        // Crate a surfaceless EGL Context
        HWND hwnd;

        WNDCLASSA wnd;
        wnd.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        wnd.lpfnWndProc = OpenVRVideoWndProc;
        wnd.cbClsExtra = 0;
        wnd.cbWndExtra = 0;
        wnd.hInstance = GetModuleHandle(NULL);
        wnd.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        wnd.hCursor = LoadCursor(NULL, IDC_ARROW);
        wnd.hbrBackground = (HBRUSH)(COLOR_BACKGROUND);
        wnd.lpszMenuName = NULL;
        wnd.lpszClassName = "SDL_openvrvideo_classname";
        RegisterClassA(&wnd);
        hwnd = CreateWindowA("SDL_openvrvideo_classname", "SDL_openvrvideo_windowname", (WS_OVERLAPPEDWINDOW), 0, 0,
                              100, 100, NULL, NULL, GetModuleHandle(NULL), NULL);

        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        videodata->hdc = GetDC(hwnd);

        static  PIXELFORMATDESCRIPTOR pfd =
        {
            sizeof(PIXELFORMATDESCRIPTOR),
            1,
            PFD_DRAW_TO_WINDOW |
            PFD_SUPPORT_OPENGL |
            PFD_DOUBLEBUFFER,
            PFD_TYPE_RGBA,
            24,
            8, 0, 8, 8, 8, 16,
            8,
            24,
            32,
            8, 8, 8, 8,
            16,
            0,
            0,
            PFD_MAIN_PLANE,
            0,
            0, 0, 0
        };
        GLuint PixelFormat = ChoosePixelFormat(videodata->hdc, &pfd);
        if (!SetPixelFormat(videodata->hdc, PixelFormat, &pfd)) {
            SDL_SetError( "Could not set pixel format" );
            return NULL;
        }
        HMODULE opengl = GetModuleHandleA(DEFAULT_OPENGL);
        if (!opengl) {
            SDL_SetError("Could not open OpenGL Library %s", DEFAULT_OPENGL);
            return NULL;
        }

        ov_wglMakeCurrent = (BOOL(*)(HDC, HGLRC))GetProcAddress(opengl, "wglMakeCurrent");
        ov_wglCreateContext = (HGLRC(*)(HDC))GetProcAddress(opengl, "wglCreateContext");
        ov_wglGetProcAddress = (PROC(*)(LPCSTR))GetProcAddress(opengl, "wglGetProcAddress");
        ov_wglDeleteContext = (BOOL(*)(HGLRC))GetProcAddress(opengl, "wglDeleteContext");
        if (!ov_wglMakeCurrent || !ov_wglCreateContext) {
            SDL_SetError("Cannot get wgl context procs(%p, %p)", ov_wglMakeCurrent, ov_wglCreateContext);
            return NULL;
        }

        videodata->hglrc = ov_wglCreateContext(videodata->hdc);
        if (!videodata->hglrc || !ov_wglMakeCurrent(videodata->hdc, videodata->hglrc)) {
            SDL_SetError("Could not make current OpenGL context.");
            return NULL;
        }
    }

    i = OPENVR_InitExtensions(_this);
    if (i == 0) {
        return NULL;
    }

    videodata->renderdoc_debugmarker_frame_end = false;

    ov_glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
    for (i = 0; i < numExtensions; i++) {
        const char *ccc = (const char *)ov_glGetStringi(GL_EXTENSIONS, i);
        if (SDL_strcmp(ccc, "GL_KHR_debug") == 0) {
#ifdef DEBUG_OPENVR
            SDL_Log("Found renderdoc debug extension.");
#endif
            videodata->renderdoc_debugmarker_frame_end = true;
        }
    }

    if (!videodata->bDidCreateOverlay) {
        if (!OPENVR_InitializeOverlay(_this, window)) {
            return NULL;
        }
    }

    OPENVR_CheckRenderbuffer(_this);

    OPENVR_SetupFrame(_this, window);

    SDL_GLContext result = SDL_malloc(sizeof(struct SDL_GLContextState));
    if (!result) {
        return NULL;
    }
    result->hglrc = videodata->hglrc;
    return result;
}

static bool OPENVR_GL_MakeCurrent(SDL_VideoDevice *_this, SDL_Window *wnd, SDL_GLContext context)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    ov_wglMakeCurrent(videodata->hdc, videodata->hglrc);
    return true;
}

static bool OPENVR_GL_SetSwapInterval(SDL_VideoDevice *_this, int interval)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    videodata->swap_interval = interval;
    return true;
}

static bool OPENVR_GL_GetSwapInterval(SDL_VideoDevice *_this, int *swapInterval)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    if (swapInterval)
        *swapInterval = videodata->swap_interval;
    else
        return SDL_SetError("OPENVR: null passed in for GetSwapInterval");
    return true;
}

static bool OPENVR_GL_DestroyContext(SDL_VideoDevice *_this, SDL_GLContext context)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    ov_wglMakeCurrent(videodata->hdc, NULL);
    ov_wglDeleteContext(videodata->hglrc);
    return true;
}


#else

static EGLint context_attribs[] = {
    EGL_CONTEXT_CLIENT_VERSION, 2,
    EGL_NONE
};

static bool SDL_EGL_InitInternal(SDL_VideoData * vd)
{
    // Crate a surfaceless EGL Context
    EGLint major, minor;
    EGLConfig eglCfg=NULL;
    EGLBoolean b;

    vd->eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
#ifdef DEBUG_OPENVR
    SDL_Log("EGL Display: %p", vd->eglDpy);
#endif

    if (vd->eglDpy == 0) {
        return SDL_SetError("No EGL Display");
    }

    b = eglInitialize(vd->eglDpy, &major, &minor);
    if (!b) {
        return SDL_SetError("eglInitialize failed");
    }

    eglBindAPI(EGL_OPENGL_API);
#ifdef DEBUG_OPENVR
    SDL_Log("EGL Major Minor: %d %d = %d", major, minor, b);
#endif

    vd->eglCtx = eglCreateContext(vd->eglDpy, eglCfg, EGL_NO_CONTEXT, context_attribs);

#ifdef DEBUG_OPENVR
    {
        int err = eglGetError();
        if (err != EGL_SUCCESS) {
            return SDL_SetError("EGL Error after eglCreateContext %d", err);
        }
    }
#endif

    if (!vd->eglCtx) {
        return SDL_SetError("No EGL context available");
    }

    eglMakeCurrent(vd->eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, vd->eglCtx);

    return true;
}

// Linux, EGL, etc.
static bool OVR_EGL_LoadLibrary(SDL_VideoDevice *_this, const char *path)
{
    return SDL_EGL_LoadLibrary(_this, path, /*displaydata->native_display*/0, 0);
}

static SDL_FunctionPointer OVR_EGL_GetProcAddress(SDL_VideoDevice *_this, const char *proc)
{
    return SDL_EGL_GetProcAddress(proc);
}
static void OVR_EGL_UnloadLibrary(SDL_VideoDevice *_this)
{
    return SDL_EGL_UnloadLibrary(_this);
}
static SDL_GLContext OVR_EGL_CreateContext(SDL_VideoDevice *_this, SDL_Window * window)
{
    GLint numExtensions;
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    if (!videodata->eglCtx) {
        if (!SDL_EGL_InitInternal(videodata)) {
            return NULL;
        }
    }

    if (!OPENVR_InitExtensions(_this)) {
        return NULL;
    }

    videodata->renderdoc_debugmarker_frame_end = false;

    ov_glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
    for(int i = 0; i < numExtensions; i++) {
        const char * ccc = (const char*)ov_glGetStringi(GL_EXTENSIONS, i);
        if (SDL_strcmp(ccc, "GL_KHR_debug") == 0) {
#ifdef DEBUG_OPENVR
           SDL_Log("Found renderdoc debug extension.");
#endif
           videodata->renderdoc_debugmarker_frame_end = true;
        }
    }

    if (!videodata->bDidCreateOverlay) {
        if (!OPENVR_InitializeOverlay(_this, window)) {
            return NULL;
        }
    }

    OPENVR_CheckRenderbuffer(_this);

    OPENVR_SetupFrame(_this, window);

    return videodata->eglCtx;
}

static bool OVR_EGL_MakeCurrent(SDL_VideoDevice *_this, SDL_Window * wnd, SDL_GLContext context)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    eglMakeCurrent(videodata->eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, videodata->eglCtx);
    return true;
}

static bool OVR_EGL_SetSwapInterval(SDL_VideoDevice *_this, int interval)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    videodata->swap_interval = interval;
    return true;
}

static bool OVR_EGL_GetSwapInterval(SDL_VideoDevice *_this, int * swapInterval)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    if (swapInterval)
        *swapInterval = videodata->swap_interval;
    else
        return SDL_SetError("OPENVR: null passed in for GetSwapInterval");
    return true;
}

static bool OVR_EGL_DestroyContext(SDL_VideoDevice *_this, SDL_GLContext context)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    if (videodata->eglDpy) {
        eglTerminate(videodata->eglDpy);
    }
    return true;
}

#endif

static bool OPENVR_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props)
{
    SDL_WindowData *data;

    // Allocate window internal data
    data = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (data == NULL) {
        return SDL_OutOfMemory();
    }

    window->max_w = 4096;
    window->max_h = 4096;
    window->min_w = 1;
    window->min_h = 1;

    // Setup driver data for this window
    window->internal = data;
    return true;
}


static void OPENVR_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data;

    data = window->internal;
    if (data) {
        SDL_free(data);
    }
    window->internal = NULL;
}

static void OPENVR_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData * data = (SDL_VideoData *)_this->internal;
    if (data->bDidCreateOverlay) {
        data->oOverlay->SetOverlayName(data->overlayID, window->title);
    }
}

static void OPENVR_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->internal;

    if (window->pending.w != window->w) {
        window->w = window->pending.w;
    }

    if (window->pending.h != window->h) {
        window->h = window->pending.h;
    }

    if (data->targh != window->h || data->targw != window->w) {
        OPENVR_HandleResize(_this, window->w, window->h);
    }

    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESIZED, window->w, window->h);
}

static void OPENVR_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->internal;
    *w = data->targw;
    *h = data->targh;
}

static void OPENVR_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->internal;
    if (data->targh != window->h || data->targw != window->w) {
        OPENVR_HandleResize(_this, window->w, window->h);
    }

    data->oOverlay->ShowDashboard(data->sOverlayName);

    window->flags |= (SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_MOUSE_FOCUS);
    SDL_SetKeyboardFocus(window);
}

static void OPENVR_HideWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    return;
}

static bool OPENVR_GL_SwapWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;

    // This is a little weird.   On Windows, we don't necessarily call the normal
    // context creation function, and we might get here without having our buffers
    // initialized.
    if (!videodata->bDidCreateOverlay) {
        if (!OPENVR_InitializeOverlay(_this, window)) {
            return false;
        }
    }

    if (!OPENVR_ReleaseFrame(_this)) {
        return false;
    }

    // If swap_interval is nonzero (i.e. -1 or 1) we want to wait for vsync on the compositor.
    if (videodata->swap_interval != 0) {
        videodata->oOverlay->WaitFrameSync(100);
    }

    if (!OPENVR_SetupFrame(_this, window)) {
        return false;
    }

    return true;
}

static void OPENVR_HandleMouse(float x, float y, int btn, int evt)
{
    if (evt == 2) {
        SDL_SendMouseMotion(0, NULL, SDL_GLOBAL_MOUSE_ID, false, x, y);
    } else {
        const Uint8 button = SDL_BUTTON_LEFT + btn;
        const bool down = (evt != 0);
        SDL_SendMouseButton(0, NULL, SDL_GLOBAL_MOUSE_ID, button, down);
    }
}


static bool OPENVR_HasScreenKeyboardSupport(SDL_VideoDevice *_this)
{
    return true;
}

static void OPENVR_ShowScreenKeyboard(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    if (!videodata ||
        videodata->oOverlay == 0 ||
        videodata->overlayID == 0) {
        return;
    }
    EGamepadTextInputMode input_mode;
    switch (SDL_GetTextInputType(props)) {
    case SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_HIDDEN:
    case SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_HIDDEN:
        input_mode = EGamepadTextInputMode_k_EGamepadTextInputModePassword;
        break;
    default:
        input_mode = EGamepadTextInputMode_k_EGamepadTextInputModeNormal;
        break;
    }
    EGamepadTextInputLineMode line_mode;
    if (SDL_GetTextInputMultiline(props)) {
        line_mode = EGamepadTextInputLineMode_k_EGamepadTextInputLineModeMultipleLines;
    } else {
        line_mode = EGamepadTextInputLineMode_k_EGamepadTextInputLineModeSingleLine;
    }
    videodata->oOverlay->ShowKeyboardForOverlay(videodata->overlayID,
           input_mode, line_mode,
           EKeyboardFlags_KeyboardFlag_Minimal, "Virtual Keyboard", 128, "", 0);
    videodata->bKeyboardShown = true;
}

static void OPENVR_HideScreenKeyboard(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    videodata->oOverlay->HideKeyboard();
    videodata->bKeyboardShown = false;
}

static bool OPENVR_IsScreenKeyboardShown(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    return videodata->bKeyboardShown;
}

static SDL_Cursor *OPENVR_CreateCursor(SDL_Surface * surface, int hot_x, int hot_y)
{
    SDL_Cursor *result = SDL_calloc(1, sizeof(SDL_Cursor));
    if (!result) {
        return NULL;
    }

    uint32_t * pixels = ImageSDLToOpenVRGL(surface, false);
    SDL_CursorData *ovrc = (SDL_CursorData *)SDL_calloc(1, sizeof(*ovrc));
    if (!ovrc) {
        SDL_free(result);
        return NULL;
    }
    result->internal = ovrc;

    ov_glGenTextures(1, &ovrc->texture_id_handle);
    ov_glBindTexture(GL_TEXTURE_2D, ovrc->texture_id_handle);
    ov_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surface->w, surface->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    SDL_free(pixels);
    ov_glBindTexture(GL_TEXTURE_2D, 0);

    ovrc->hot_x = hot_x;
    ovrc->hot_y = hot_y;
    ovrc->w = surface->w;
    ovrc->h = surface->h;

    return result;
}

static bool OPENVR_ShowCursor(SDL_Cursor * cursor)
{
    SDL_CursorData * ovrc;
    EVROverlayError e;
    Texture_t texture;
    HmdVector2_t hotspot;
    VRTextureBounds_t tb;

    if (!cursor) {
        global_openvr_driver->oOverlay->SetOverlayFlag(global_openvr_driver->overlayID, VROverlayFlags_HideLaserIntersection, true);
        e = global_openvr_driver->oOverlay->SetOverlayCursor(global_openvr_driver->overlayID, k_ulOverlayHandleInvalid);
        if (e != EVROverlayError_VROverlayError_None) {
            return OPENVR_SetOverlayError(e);
        }
        return true;
    }

    global_openvr_driver->oOverlay->SetOverlayFlag(global_openvr_driver->overlayID, VROverlayFlags_HideLaserIntersection, false);

    ovrc = cursor->internal;

    if (!ovrc) {
        // Sometimes at boot there is a race condition where this is not ready.
        return true;
    }

    hotspot.v[0] = (float)ovrc->hot_x / (float)ovrc->w;
    hotspot.v[1] = (float)ovrc->hot_y / (float)ovrc->h;

    texture.handle = (void*)(intptr_t)(ovrc->texture_id_handle);
    texture.eType = ETextureType_TextureType_OpenGL;
    texture.eColorSpace = EColorSpace_ColorSpace_Auto;

    tb.uMin = 0;
    tb.uMax = 1;
    tb.vMin = 1;
    tb.vMax = 0;

    e = global_openvr_driver->oOverlay->SetOverlayTextureBounds(global_openvr_driver->cursorID, &tb);
    if (e != EVROverlayError_VROverlayError_None) {
        return OPENVR_SetOverlayError(e);
    }

    e = global_openvr_driver->oOverlay->SetOverlayTransformCursor(global_openvr_driver->cursorID, &hotspot);
    if (e != EVROverlayError_VROverlayError_None) {
        return OPENVR_SetOverlayError(e);
    }

    e = global_openvr_driver->oOverlay->SetOverlayTexture(global_openvr_driver->cursorID, &texture);
    if (e != EVROverlayError_VROverlayError_None) {
        return OPENVR_SetOverlayError(e);
    }

    e = global_openvr_driver->oOverlay->SetOverlayCursor(global_openvr_driver->overlayID, global_openvr_driver->cursorID);
    if (e != EVROverlayError_VROverlayError_None) {
        return OPENVR_SetOverlayError(e);
    }

    return true;
}

static void OPENVR_FreeCursor(SDL_Cursor * cursor)
{
    if (cursor) {
        SDL_CursorData *ovrc = cursor->internal;
        if (ovrc) {
            ov_glDeleteTextures(1, &ovrc->texture_id_handle);
            SDL_free(ovrc);
        }
        SDL_free(cursor);
    }
}


static bool OPENVR_SetWindowIcon(SDL_VideoDevice *_this, SDL_Window * window, SDL_Surface * icon)
{
    if (!global_openvr_driver) {
        return SDL_SetError("OpenVR Overlay not initialized");
    }

    unsigned texture_id_handle;
    EVROverlayError e;
    Texture_t texture;
    uint32_t * pixels;
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    if (videodata->bIconOverridden) {
        return SDL_SetError("OpenVR Icon is overridden.");
    }

    pixels = ImageSDLToOpenVRGL(icon, true);

    ov_glGenTextures(1, &texture_id_handle);
    ov_glBindTexture(GL_TEXTURE_2D, texture_id_handle);
    ov_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, icon->w, icon->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    SDL_free(pixels);
    ov_glBindTexture(GL_TEXTURE_2D, 0);

    texture.handle = (void*)(intptr_t)(texture_id_handle);
    texture.eType = ETextureType_TextureType_OpenGL;
    texture.eColorSpace = EColorSpace_ColorSpace_Auto;

    e = global_openvr_driver->oOverlay->SetOverlayTexture(videodata->thumbID, &texture);
    if (e != EVROverlayError_VROverlayError_None) {
        return OPENVR_SetOverlayError(e);
    }
    return true;
}

static bool OPENVR_ShowMessageBox(SDL_VideoDevice *_this,const SDL_MessageBoxData *messageboxdata, int *buttonid)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    char empty = 0;
    char * message = SDL_strdup(messageboxdata->message?messageboxdata->message:"");
    char * title = SDL_strdup(messageboxdata->message?messageboxdata->message:"");
    char * ok = SDL_strdup("Ok");
    videodata->oOverlay->ShowMessageOverlay(message, title, ok, &empty, &empty, &empty);
    SDL_free(ok);
    SDL_free(title);
    SDL_free(message);
    return true;
}

static void InitializeMouseFunctions()
{
    SDL_Mouse *mouse = SDL_GetMouse();
    mouse->CreateCursor = OPENVR_CreateCursor;
    mouse->ShowCursor = OPENVR_ShowCursor;
    mouse->FreeCursor = OPENVR_FreeCursor;
}

static void OPENVR_PumpEvents(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->internal;
    struct VREvent_t nEvent;
    if (videodata->overlayID) {
        while (videodata->oOverlay->PollNextOverlayEvent(videodata->overlayID, &nEvent, sizeof(nEvent))) {
            switch (nEvent.eventType) {
            case EVREventType_VREvent_ButtonPress:
            case EVREventType_VREvent_ButtonUnpress:
                break;
            case EVREventType_VREvent_MouseMove:
                OPENVR_HandleMouse(nEvent.data.mouse.x, videodata->targh - nEvent.data.mouse.y, nEvent.data.mouse.button, 2);
                break;
            case EVREventType_VREvent_MouseButtonDown:
                OPENVR_HandleMouse(nEvent.data.mouse.x, videodata->targh - nEvent.data.mouse.y, 0, 1);
                break;
            case EVREventType_VREvent_MouseButtonUp:
                OPENVR_HandleMouse(nEvent.data.mouse.x, videodata->targh - nEvent.data.mouse.y, 0, 0);
                break;
            case EVREventType_VREvent_KeyboardCharInput:
                SDL_SendKeyboardUnicodeKey(SDL_GetTicksNS(), nEvent.data.keyboard.cNewInput[0]);
                break;
            case EVREventType_VREvent_OverlayShown:
                SDL_SetKeyboardFocus(videodata->window);
                SDL_SendWindowEvent(videodata->window, SDL_EVENT_WINDOW_RESTORED, 0, 0);
                SDL_SendWindowEvent(videodata->window, SDL_EVENT_WINDOW_SHOWN, 0, 0);
                break;
            case EVREventType_VREvent_OverlayHidden:
                SDL_SendWindowEvent(videodata->window, SDL_EVENT_WINDOW_HIDDEN, 0, 0);
                SDL_SendWindowEvent(videodata->window, SDL_EVENT_WINDOW_MINIMIZED, 0, 0);
                SDL_SetKeyboardFocus(NULL);
                break;
            case EVREventType_VREvent_OverlayClosed:
            case EVREventType_VREvent_Quit:
                SDL_Quit();
                break;
            }
        }
    }
}


static SDL_VideoDevice *OPENVR_CreateDevice(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *data;

#ifdef SDL_PLATFORM_WINDOWS
    SDL_RegisterApp(NULL, 0, NULL);
#endif

    // Initialize all variables that we clean on shutdown
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (device) {
        data = (struct SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    } else {
        data = NULL;
    }
    if (!data) {
#ifdef SDL_PLATFORM_WINDOWS
        SDL_UnregisterApp();
#endif
        SDL_free(device);
        return NULL;
    }
    device->internal = data;

    {
        const char * hint = SDL_GetHint(SDL_HINT_OPENVR_LIBRARY);
        if (hint)
            data->openVRLIB = SDL_LoadObject(hint);
#ifdef SDL_PLATFORM_WINDOWS
        if (!data->openVRLIB)
            data->openVRLIB = SDL_LoadObject("openvr_api.dll");
#else
        if (!data->openVRLIB)
            data->openVRLIB = SDL_LoadObject("openvr_api.so");
#endif
    }

    if (!data->openVRLIB) {
        SDL_SetError("Could not open OpenVR API Library");
        goto error;
    }

    data->FN_VR_InitInternal = (intptr_t(*)(EVRInitError * peError, EVRApplicationType eType))SDL_LoadFunction(data->openVRLIB, "VR_InitInternal");
    data->FN_VR_GetVRInitErrorAsEnglishDescription = (const char *(*)(EVRInitError error))SDL_LoadFunction(data->openVRLIB, "VR_GetVRInitErrorAsEnglishDescription");
    data->FN_VR_GetGenericInterface = (intptr_t (*)(const char *pchInterfaceVersion, EVRInitError * peError))SDL_LoadFunction(data->openVRLIB, "VR_GetGenericInterface");
    if (!data->FN_VR_InitInternal || !data->FN_VR_GetVRInitErrorAsEnglishDescription || !data->FN_VR_GetGenericInterface) {
        goto error;
    }

    char fnname[128];
    EVRInitError e;
    data->vrtoken = data->FN_VR_InitInternal(&e, EVRApplicationType_VRApplication_Overlay);
    if (!data->vrtoken) {
        const char *err = "Can't get english description";
        if (data->FN_VR_GetVRInitErrorAsEnglishDescription != NULL)
            err = data->FN_VR_GetVRInitErrorAsEnglishDescription(e);
        SDL_SetError("Could not generate OpenVR Context (%s)", err);
        goto error;
    }

    SDL_snprintf(fnname, 127, "FnTable:%s", IVRSystem_Version);
    data->oSystem = (struct VR_IVRSystem_FnTable *)data->FN_VR_GetGenericInterface(fnname, &e);
    SDL_snprintf(fnname, 127, "FnTable:%s", IVROverlay_Version);
    data->oOverlay = (struct VR_IVROverlay_FnTable *)data->FN_VR_GetGenericInterface(fnname, &e);
    SDL_snprintf(fnname, 127, "FnTable:%s", IVRInput_Version);
    data->oInput = (struct VR_IVRInput_FnTable *)data->FN_VR_GetGenericInterface(fnname, &e);

    if (!data->oOverlay || !data->oSystem || !data->oInput) {
        SDL_SetError("Could not get interfaces for the OpenVR System (%s), Overlay (%s) and Input (%s) versions", IVRSystem_Version, IVROverlay_Version, IVRInput_Version);
    }

    const char *hint = SDL_GetHint("SDL_OPENVR_INPUT_PROFILE");
    char *loadpath = 0;
    EVRInputError err;

    if (hint) {
        SDL_asprintf(&loadpath, "%s", hint);
    } else {
        const char *basepath = SDL_GetBasePath();
        SDL_asprintf(&loadpath, "%ssdloverlay_actions.json", basepath);
    }
    if (!loadpath) {
        goto error;
    }

    err = data->oInput->SetActionManifestPath(loadpath);
#ifdef DEBUG_OPENVR
    SDL_Log("Loaded action manifest at %s (%d)", loadpath, err);
#endif
    SDL_free(loadpath);
    if (err != EVRInputError_VRInputError_None) {
        // I know we don't normally log, but this _really_ should be percolated
        // up as far as we can.
        SDL_Log("Could not load action manifest path");
        // If we didn't have a hint, this is a soft fail.
        // If we did have the hint, then it's a hard fail.
        if (hint) {
            goto error;
        }
    } else {
        if(!OPENVR_SetupJoystickBasedOnLoadedActionManifest(data)) {
            goto error;
        }
    }

    // Setup amount of available displays
    device->num_displays = 0;

    // Set device free function
    device->free = OPENVR_Destroy;

    // Setup all functions which we can handle
    device->VideoInit = OPENVR_VideoInit;
    device->VideoQuit = OPENVR_VideoQuit;
    device->SetDisplayMode = OPENVR_SetDisplayMode;
    device->CreateSDLWindow = OPENVR_CreateWindow;
    device->SetWindowTitle = OPENVR_SetWindowTitle;
    device->SetWindowSize = OPENVR_SetWindowSize;
    device->GetWindowSizeInPixels = OPENVR_GetWindowSizeInPixels;
    device->ShowWindow = OPENVR_ShowWindow;
    device->HideWindow = OPENVR_HideWindow;
    device->DestroyWindow = OPENVR_DestroyWindow;
    device->ShowMessageBox = OPENVR_ShowMessageBox;

#ifdef SDL_VIDEO_DRIVER_WINDOWS
#ifdef SDL_VIDEO_OPENGL_WGL
    device->GL_LoadLibrary = OPENVR_GL_LoadLibrary;
    device->GL_GetProcAddress = OPENVR_GL_GetProcAddress;
    device->GL_UnloadLibrary = OPENVR_GL_UnloadLibrary;
    device->GL_CreateContext = OPENVR_GL_CreateContext;
    device->GL_MakeCurrent = OPENVR_GL_MakeCurrent;
    device->GL_SetSwapInterval = OPENVR_GL_SetSwapInterval;
    device->GL_GetSwapInterval = OPENVR_GL_GetSwapInterval;
    device->GL_SwapWindow = OPENVR_GL_SwapWindow;
    device->GL_DestroyContext = OPENVR_GL_DestroyContext;
#elif SDL_VIDEO_OPENGL_EGL
    device->GL_LoadLibrary = WIN_GLES_LoadLibrary;
    device->GL_GetProcAddress = WIN_GLES_GetProcAddress;
    device->GL_UnloadLibrary = WIN_GLES_UnloadLibrary;
    device->GL_CreateContext = WIN_GLES_CreateContext;
    device->GL_MakeCurrent = WIN_GLES_MakeCurrent;
    device->GL_SetSwapInterval = WIN_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = WIN_GLES_GetSwapInterval;
    device->GL_SwapWindow = WIN_GLES_SwapWindow;
    device->GL_DestroyContext = WIN_GLES_DestroyContext;
#endif
#else
    device->GL_LoadLibrary = OVR_EGL_LoadLibrary;
    device->GL_GetProcAddress = OVR_EGL_GetProcAddress;
    device->GL_UnloadLibrary = OVR_EGL_UnloadLibrary;
    device->GL_CreateContext = OVR_EGL_CreateContext;
    device->GL_MakeCurrent = OVR_EGL_MakeCurrent;
    device->GL_SetSwapInterval = OVR_EGL_SetSwapInterval;
    device->GL_GetSwapInterval = OVR_EGL_GetSwapInterval;
    device->GL_DestroyContext = OVR_EGL_DestroyContext;
    device->GL_SwapWindow = OPENVR_GL_SwapWindow;
#endif

#if defined(SDL_VIDEO_VULKAN) && defined(SDL_VIDEO_DRIVER_WINDOWS)
    device->Vulkan_LoadLibrary = WIN_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = WIN_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = WIN_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = WIN_Vulkan_CreateSurface;
#else
    device->Vulkan_LoadLibrary = 0;
    device->Vulkan_UnloadLibrary = 0;
    device->Vulkan_GetInstanceExtensions = 0;
    device->Vulkan_CreateSurface = 0;
#endif

    device->PumpEvents = OPENVR_PumpEvents;
    device->VideoInit = OPENVR_VideoInit;
    device->VideoQuit = OPENVR_VideoQuit;

    device->HasScreenKeyboardSupport = OPENVR_HasScreenKeyboardSupport;
    device->ShowScreenKeyboard = OPENVR_ShowScreenKeyboard;
    device->HideScreenKeyboard = OPENVR_HideScreenKeyboard;
    device->IsScreenKeyboardShown = OPENVR_IsScreenKeyboardShown;
    device->SetWindowIcon = OPENVR_SetWindowIcon;

    return device;

error:
    OPENVR_Destroy(device);
    return NULL;
}

VideoBootStrap OPENVR_bootstrap = {
    "openvr", "SDL OpenVR video driver", OPENVR_CreateDevice, NULL, false
};

#endif // SDL_VIDEO_DRIVER_WINDOWS


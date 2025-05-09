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
#include "SDL_internal.h"

#ifndef SDL_android_h
#define SDL_android_h

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
/* *INDENT-OFF* */
extern "C" {
/* *INDENT-ON* */
#endif

#include <EGL/eglplatform.h>
#include <android/native_window_jni.h>

#include "../../audio/SDL_sysaudio.h"

// this appears to be broken right now (on Android, not SDL, I think...?).
#define ALLOW_MULTIPLE_ANDROID_AUDIO_DEVICES 0

// Life cycle
typedef enum
{
    SDL_ANDROID_LIFECYCLE_WAKE,
    SDL_ANDROID_LIFECYCLE_PAUSE,
    SDL_ANDROID_LIFECYCLE_RESUME,
    SDL_ANDROID_LIFECYCLE_LOWMEMORY,
    SDL_ANDROID_LIFECYCLE_DESTROY,
    SDL_NUM_ANDROID_LIFECYCLE_EVENTS
} SDL_AndroidLifecycleEvent;

void Android_SendLifecycleEvent(SDL_AndroidLifecycleEvent event);
bool Android_WaitLifecycleEvent(SDL_AndroidLifecycleEvent *event, Sint64 timeoutNS);

void Android_LockActivityMutex(void);
void Android_UnlockActivityMutex(void);

void Android_SetAllowRecreateActivity(bool enabled);

// Interface from the SDL library into the Android Java activity
extern void Android_JNI_SetActivityTitle(const char *title);
extern void Android_JNI_SetWindowStyle(bool fullscreen);
extern void Android_JNI_SetOrientation(int w, int h, int resizable, const char *hint);
extern void Android_JNI_MinizeWindow(void);
extern bool Android_JNI_ShouldMinimizeOnFocusLoss(void);

extern bool Android_JNI_GetAccelerometerValues(float values[3]);
extern void Android_JNI_ShowScreenKeyboard(int input_type, SDL_Rect *inputRect);
extern void Android_JNI_HideScreenKeyboard(void);
extern bool Android_JNI_IsScreenKeyboardShown(void);
extern ANativeWindow *Android_JNI_GetNativeWindow(void);

extern SDL_DisplayOrientation Android_JNI_GetDisplayNaturalOrientation(void);
extern SDL_DisplayOrientation Android_JNI_GetDisplayCurrentOrientation(void);

// Audio support
void Android_StartAudioHotplug(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording);
void Android_StopAudioHotplug(void);
extern void Android_AudioThreadInit(SDL_AudioDevice *device);

// Detecting device type
extern bool Android_IsDeXMode(void);
extern bool Android_IsChromebook(void);

bool Android_JNI_FileOpen(void **puserdata, const char *fileName, const char *mode);
Sint64 Android_JNI_FileSize(void *userdata);
Sint64 Android_JNI_FileSeek(void *userdata, Sint64 offset, SDL_IOWhence whence);
size_t Android_JNI_FileRead(void *userdata, void *buffer, size_t size, SDL_IOStatus *status);
size_t Android_JNI_FileWrite(void *userdata, const void *buffer, size_t size, SDL_IOStatus *status);
bool Android_JNI_FileClose(void *userdata);

// Environment support
void Android_JNI_GetManifestEnvironmentVariables(void);
int Android_JNI_OpenFileDescriptor(const char *uri, const char *mode);

// Clipboard support
bool Android_JNI_SetClipboardText(const char *text);
char *Android_JNI_GetClipboardText(void);
bool Android_JNI_HasClipboardText(void);

// Power support
int Android_JNI_GetPowerInfo(int *plugged, int *charged, int *battery, int *seconds, int *percent);

// Joystick support
void Android_JNI_PollInputDevices(void);

// Haptic support
void Android_JNI_PollHapticDevices(void);
void Android_JNI_HapticRun(int device_id, float intensity, int length);
void Android_JNI_HapticRumble(int device_id, float low_frequency_intensity, float high_frequency_intensity, int length);
void Android_JNI_HapticStop(int device_id);

// Video
bool Android_JNI_SuspendScreenSaver(bool suspend);

// Touch support
void Android_JNI_InitTouch(void);

// Threads
#include <jni.h>
JNIEnv *Android_JNI_GetEnv(void);
bool Android_JNI_SetupThread(void);

// Locale
bool Android_JNI_GetLocale(char *buf, size_t buflen);

// Generic messages
bool Android_JNI_SendMessage(int command, int param);

// MessageBox
bool Android_JNI_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID);

// Cursor support
int Android_JNI_CreateCustomCursor(SDL_Surface *surface, int hot_x, int hot_y);
void Android_JNI_DestroyCustomCursor(int cursorID);
bool Android_JNI_SetCustomCursor(int cursorID);
bool Android_JNI_SetSystemCursor(int cursorID);

// Relative mouse support
bool Android_JNI_SupportsRelativeMouse(void);
bool Android_JNI_SetRelativeMouseEnabled(bool enabled);

// Show toast notification
bool Android_JNI_ShowToast(const char *message, int duration, int gravity, int xOffset, int yOffset);

bool Android_JNI_OpenURL(const char *url);

int SDL_GetAndroidSDKVersion(void);

bool SDL_IsAndroidTablet(void);
bool SDL_IsAndroidTV(void);

// File Dialogs
bool Android_JNI_OpenFileDialog(SDL_DialogFileCallback callback, void* userdata,
    const SDL_DialogFileFilter *filters, int nfilters, bool forwrite,
    bool multiple);

// Ends C function definitions when using C++
#ifdef __cplusplus
/* *INDENT-OFF* */
}
/* *INDENT-ON* */
#endif

#endif // SDL_android_h

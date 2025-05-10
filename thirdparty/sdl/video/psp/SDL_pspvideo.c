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

#ifdef SDL_VIDEO_DRIVER_PSP

// SDL internals
#include "../SDL_sysvideo.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"

// PSP declarations
#include "SDL_pspvideo.h"
#include "SDL_pspevents_c.h"
#include "SDL_pspgl_c.h"
#include "../../render/psp/SDL_render_psp_c.h"

#include <psputility.h>
#include <pspgu.h>
#include <pspdisplay.h>
#include <vram.h>

/* unused
static bool PSP_initialized = false;
*/

static void PSP_Destroy(SDL_VideoDevice *device)
{
    SDL_free(device->internal);
    SDL_free(device);
}

static SDL_VideoDevice *PSP_Create(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *phdata;
    SDL_GLDriverData *gldata;

    // Initialize SDL_VideoDevice structure
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        return NULL;
    }

    // Initialize internal PSP specific data
    phdata = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!phdata) {
        SDL_free(device);
        return NULL;
    }

    gldata = (SDL_GLDriverData *)SDL_calloc(1, sizeof(SDL_GLDriverData));
    if (!gldata) {
        SDL_free(device);
        SDL_free(phdata);
        return NULL;
    }
    device->gl_data = gldata;

    device->internal = phdata;

    phdata->egl_initialized = true;

    // Setup amount of available displays
    device->num_displays = 0;

    // Set device free function
    device->free = PSP_Destroy;

    // Setup all functions which we can handle
    device->VideoInit = PSP_VideoInit;
    device->VideoQuit = PSP_VideoQuit;
    device->GetDisplayModes = PSP_GetDisplayModes;
    device->SetDisplayMode = PSP_SetDisplayMode;
    device->CreateSDLWindow = PSP_CreateWindow;
    device->SetWindowTitle = PSP_SetWindowTitle;
    device->SetWindowPosition = PSP_SetWindowPosition;
    device->SetWindowSize = PSP_SetWindowSize;
    device->ShowWindow = PSP_ShowWindow;
    device->HideWindow = PSP_HideWindow;
    device->RaiseWindow = PSP_RaiseWindow;
    device->MaximizeWindow = PSP_MaximizeWindow;
    device->MinimizeWindow = PSP_MinimizeWindow;
    device->RestoreWindow = PSP_RestoreWindow;
    device->DestroyWindow = PSP_DestroyWindow;
    device->GL_LoadLibrary = PSP_GL_LoadLibrary;
    device->GL_GetProcAddress = PSP_GL_GetProcAddress;
    device->GL_UnloadLibrary = PSP_GL_UnloadLibrary;
    device->GL_CreateContext = PSP_GL_CreateContext;
    device->GL_MakeCurrent = PSP_GL_MakeCurrent;
    device->GL_SetSwapInterval = PSP_GL_SetSwapInterval;
    device->GL_GetSwapInterval = PSP_GL_GetSwapInterval;
    device->GL_SwapWindow = PSP_GL_SwapWindow;
    device->GL_DestroyContext = PSP_GL_DestroyContext;
    device->HasScreenKeyboardSupport = PSP_HasScreenKeyboardSupport;
    device->ShowScreenKeyboard = PSP_ShowScreenKeyboard;
    device->HideScreenKeyboard = PSP_HideScreenKeyboard;
    device->IsScreenKeyboardShown = PSP_IsScreenKeyboardShown;

    device->PumpEvents = PSP_PumpEvents;

    return device;
}

static void configure_dialog(pspUtilityMsgDialogParams *dialog, size_t dialog_size)
{
	// clear structure and setup size
	SDL_memset(dialog, 0, dialog_size);
	dialog->base.size = dialog_size;

	// set language
	sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_LANGUAGE, &dialog->base.language);

	// set X/O swap
	sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_UNKNOWN, &dialog->base.buttonSwap);

	// set thread priorities
	// TODO: understand how these work
	dialog->base.soundThread = 0x10;
	dialog->base.graphicsThread = 0x11;
	dialog->base.fontThread = 0x12;
	dialog->base.accessThread = 0x13;
}

static void *setup_temporal_gu(void *list)
{
    // Using GU_PSM_8888 for the framebuffer
	int bpp = 4;

	void *doublebuffer = vramalloc(PSP_FRAME_BUFFER_SIZE * bpp * 2);
    void *backbuffer = doublebuffer;
    void *frontbuffer = ((uint8_t *)doublebuffer) + PSP_FRAME_BUFFER_SIZE * bpp;

    sceGuInit();

    sceGuStart(GU_DIRECT,list);
	sceGuDrawBuffer(GU_PSM_8888, vrelptr(frontbuffer), PSP_FRAME_BUFFER_WIDTH);
    sceGuDispBuffer(PSP_SCREEN_WIDTH, PSP_SCREEN_HEIGHT, vrelptr(backbuffer), PSP_FRAME_BUFFER_WIDTH);

    sceGuOffset(2048 - (PSP_SCREEN_WIDTH >> 1), 2048 - (PSP_SCREEN_HEIGHT >> 1));
    sceGuViewport(2048, 2048, PSP_SCREEN_WIDTH, PSP_SCREEN_HEIGHT);

    sceGuDisable(GU_DEPTH_TEST);

    // Scissoring
    sceGuScissor(0, 0, PSP_SCREEN_WIDTH, PSP_SCREEN_HEIGHT);
    sceGuEnable(GU_SCISSOR_TEST);

    sceGuFinish();
    sceGuSync(0,0);

    sceDisplayWaitVblankStart();
    sceGuDisplay(GU_TRUE);

	return doublebuffer;
}

static void term_temporal_gu(void *guBuffer)
{
	sceGuTerm();
	vfree(guBuffer);
	sceDisplayWaitVblankStart();
}

bool PSP_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
	unsigned char list[64] __attribute__((aligned(64)));
	pspUtilityMsgDialogParams dialog;
	int status;
    void *guBuffer = NULL;

	// check if it's possible to use existing video context
	if (SDL_GetKeyboardFocus() == NULL) {
		guBuffer = setup_temporal_gu(list);
	}

	// configure dialog
	configure_dialog(&dialog, sizeof(dialog));

	// setup dialog options for text
	dialog.mode = PSP_UTILITY_MSGDIALOG_MODE_TEXT;
	dialog.options = PSP_UTILITY_MSGDIALOG_OPTION_TEXT;

	// copy the message in, 512 bytes max
	SDL_snprintf(dialog.message, sizeof(dialog.message), "%s\r\n\r\n%s", messageboxdata->title, messageboxdata->message);

	// too many buttons
	if (messageboxdata->numbuttons > 2)
		return SDL_SetError("messageboxdata->numbuttons valid values are 0, 1, 2");

	// we only have two options, "yes/no" or "ok"
	if (messageboxdata->numbuttons == 2)
		dialog.options |= PSP_UTILITY_MSGDIALOG_OPTION_YESNO_BUTTONS | PSP_UTILITY_MSGDIALOG_OPTION_DEFAULT_NO;

	// start dialog
	if (sceUtilityMsgDialogInitStart(&dialog) != 0)
		return SDL_SetError("sceUtilityMsgDialogInitStart() failed for some reason");

	// loop while the dialog is active
	status = PSP_UTILITY_DIALOG_NONE;
	do
	{
		sceGuStart(GU_DIRECT, list);
		sceGuClearColor(0);
		sceGuClearDepth(0);
		sceGuClear(GU_COLOR_BUFFER_BIT|GU_DEPTH_BUFFER_BIT);
		sceGuFinish();
		sceGuSync(0,0);

		status = sceUtilityMsgDialogGetStatus();

		switch (status)
		{
			case PSP_UTILITY_DIALOG_VISIBLE:
				sceUtilityMsgDialogUpdate(1);
				break;

			case PSP_UTILITY_DIALOG_QUIT:
				sceUtilityMsgDialogShutdownStart();
				break;
		}

		sceDisplayWaitVblankStart();
		sceGuSwapBuffers();

	} while (status != PSP_UTILITY_DIALOG_NONE);

    // cleanup
	if (guBuffer)
	{
		term_temporal_gu(guBuffer);
	}

	// success
	if (dialog.buttonPressed == PSP_UTILITY_MSGDIALOG_RESULT_YES)
		*buttonID = messageboxdata->buttons[0].buttonID;
	else if (dialog.buttonPressed == PSP_UTILITY_MSGDIALOG_RESULT_NO)
		*buttonID = messageboxdata->buttons[1].buttonID;
	else
		*buttonID = messageboxdata->buttons[0].buttonID;

	return true;
}

VideoBootStrap PSP_bootstrap = {
    "psp",
    "PSP Video Driver",
    PSP_Create,
    PSP_ShowMessageBox,
    false
};

/*****************************************************************************/
// SDL Video and Display initialization/handling functions
/*****************************************************************************/
bool PSP_VideoInit(SDL_VideoDevice *_this)
{
    SDL_DisplayMode mode;

    if (!PSP_EventInit(_this)) {
        return false;  // error string would already be set
    }

    SDL_zero(mode);
    mode.w = PSP_SCREEN_WIDTH;
    mode.h = PSP_SCREEN_HEIGHT;
    mode.refresh_rate = 60.0f;

    // 32 bpp for default
    mode.format = SDL_PIXELFORMAT_ABGR8888;

    if (SDL_AddBasicVideoDisplay(&mode) == 0) {
        return false;
    }
    return true;
}

void PSP_VideoQuit(SDL_VideoDevice *_this)
{
    PSP_EventQuit(_this);
}

bool PSP_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *display)
{
    SDL_DisplayMode mode;

    SDL_zero(mode);
    mode.w = PSP_SCREEN_WIDTH;
    mode.h = PSP_SCREEN_HEIGHT;
    mode.refresh_rate = 60.0f;

    // 32 bpp for default
    mode.format = SDL_PIXELFORMAT_ABGR8888;
    SDL_AddFullscreenDisplayMode(display, &mode);

    // 16 bpp secondary mode
    mode.format = SDL_PIXELFORMAT_BGR565;
    SDL_AddFullscreenDisplayMode(display, &mode);
    return true;
}

bool PSP_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    return true;
}

#define EGLCHK(stmt)                           \
    do {                                       \
        EGLint err;                            \
                                               \
        stmt;                                  \
        err = eglGetError();                   \
        if (err != EGL_SUCCESS) {              \
            SDL_SetError("EGL error %d", err); \
            return true;                          \
        }                                      \
    } while (0)

bool PSP_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props)
{
    SDL_WindowData *wdata;

    // Allocate window internal data
    wdata = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (!wdata) {
        return false;
    }

    // Setup driver data for this window
    window->internal = wdata;

    SDL_SetKeyboardFocus(window);

    // Window has been successfully created
    return true;
}

void PSP_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window)
{
}
bool PSP_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window)
{
    return SDL_Unsupported();
}
void PSP_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void PSP_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void PSP_HideWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void PSP_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void PSP_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void PSP_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void PSP_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void PSP_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}

bool PSP_HasScreenKeyboardSupport(SDL_VideoDevice *_this)
{
    return true;
}

void PSP_ShowScreenKeyboard(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props)
{
    char list[0x20000] __attribute__((aligned(64)));  // Needed for sceGuStart to work
    int i;
    int done = 0;
    int input_text_length = 32; // SDL_SendKeyboardText supports up to 32 characters per event
    unsigned short outtext[input_text_length];
    char text_string[input_text_length];

    SceUtilityOskData data;
    SceUtilityOskParams params;

    SDL_memset(outtext, 0, input_text_length * sizeof(unsigned short));

    data.language = PSP_UTILITY_OSK_LANGUAGE_DEFAULT;
    data.lines = 1;
    data.unk_24 = 1;
    switch (SDL_GetTextInputType(props)) {
    default:
    case SDL_TEXTINPUT_TYPE_TEXT:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_ALL;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_NAME:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_ALL;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_EMAIL:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_ALL;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_USERNAME:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_ALL;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_HIDDEN:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_ALL;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_VISIBLE:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_ALL;
        break;
    case SDL_TEXTINPUT_TYPE_NUMBER:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_LATIN_DIGIT;
        break;
    case SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_HIDDEN:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_LATIN_DIGIT;
        break;
    case SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_VISIBLE:
        data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_LATIN_DIGIT;
        break;
    }
    data.desc = NULL;
    data.intext = NULL;
    data.outtextlength = input_text_length;
    data.outtextlimit = input_text_length;
    data.outtext = outtext;

    params.base.size = sizeof(params);
    sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_LANGUAGE, &params.base.language);
    sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_UNKNOWN, &params.base.buttonSwap);
    params.base.graphicsThread = 17;
    params.base.accessThread = 19;
    params.base.fontThread = 18;
    params.base.soundThread = 16;
    params.datacount = 1;
    params.data = &data;

    sceUtilityOskInitStart(&params);

    while(!done) {
        sceGuStart(GU_DIRECT, list);
        sceGuClearColor(0);
        sceGuClearDepth(0);
        sceGuClear(GU_COLOR_BUFFER_BIT|GU_DEPTH_BUFFER_BIT);
        sceGuFinish();
        sceGuSync(0,0);

        switch(sceUtilityOskGetStatus())
        {
            case PSP_UTILITY_DIALOG_VISIBLE:
                sceUtilityOskUpdate(1);
                break;
            case PSP_UTILITY_DIALOG_QUIT:
                sceUtilityOskShutdownStart();
                break;
            case PSP_UTILITY_DIALOG_NONE:
                done = 1;
                break;
            default :
                break;
        }
        sceDisplayWaitVblankStart();
        sceGuSwapBuffers();
    }

    // Convert input list to string
    for (i = 0; i < input_text_length; i++) {
        text_string[i] = outtext[i];
    }
    SDL_SendKeyboardText((const char *) text_string);
}
void PSP_HideScreenKeyboard(SDL_VideoDevice *_this, SDL_Window *window)
{
}
bool PSP_IsScreenKeyboardShown(SDL_VideoDevice *_this, SDL_Window *window)
{
    return false;
}

#endif // SDL_VIDEO_DRIVER_PSP

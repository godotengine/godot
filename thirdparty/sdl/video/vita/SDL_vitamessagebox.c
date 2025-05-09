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

#ifdef SDL_VIDEO_DRIVER_VITA

#include "SDL_vitavideo.h"
#include "SDL_vitamessagebox.h"
#include <psp2/message_dialog.h>

#ifdef SDL_VIDEO_RENDER_VITA_GXM
#include "../../render/vitagxm/SDL_render_vita_gxm_tools.h"
#endif // SDL_VIDEO_RENDER_VITA_GXM

bool VITA_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
#ifdef SDL_VIDEO_RENDER_VITA_GXM
    SceMsgDialogParam param;
    SceMsgDialogUserMessageParam msgParam;
    SceMsgDialogButtonsParam buttonParam;
    SceDisplayFrameBuf dispparam;
    char message[512];

    SceMsgDialogResult dialog_result;
    SceCommonDialogErrorCode init_result;
    bool setup_minimal_gxm = false;

    if (messageboxdata->numbuttons > 3) {
        return false;
    }

    SDL_zero(param);
    sceMsgDialogParamInit(&param);
    param.mode = SCE_MSG_DIALOG_MODE_USER_MSG;

    SDL_zero(msgParam);
    SDL_snprintf(message, sizeof(message), "%s\r\n\r\n%s", messageboxdata->title, messageboxdata->message);

    msgParam.msg = (const SceChar8 *)message;
    SDL_zero(buttonParam);

    if (messageboxdata->numbuttons == 3) {
        msgParam.buttonType = SCE_MSG_DIALOG_BUTTON_TYPE_3BUTTONS;
        msgParam.buttonParam = &buttonParam;
        buttonParam.msg1 = messageboxdata->buttons[0].text;
        buttonParam.msg2 = messageboxdata->buttons[1].text;
        buttonParam.msg3 = messageboxdata->buttons[2].text;
    } else if (messageboxdata->numbuttons == 2) {
        msgParam.buttonType = SCE_MSG_DIALOG_BUTTON_TYPE_YESNO;
    } else if (messageboxdata->numbuttons == 1) {
        msgParam.buttonType = SCE_MSG_DIALOG_BUTTON_TYPE_OK;
    }
    param.userMsgParam = &msgParam;

    dispparam.size = sizeof(dispparam);

    init_result = sceMsgDialogInit(&param);

    // Setup display if it hasn't been initialized before
    if (init_result == SCE_COMMON_DIALOG_ERROR_GXM_IS_UNINITIALIZED) {
        gxm_minimal_init_for_common_dialog();
        init_result = sceMsgDialogInit(&param);
        setup_minimal_gxm = true;
    }

    gxm_init_for_common_dialog();

    if (init_result >= 0) {
        while (sceMsgDialogGetStatus() == SCE_COMMON_DIALOG_STATUS_RUNNING) {
            gxm_swap_for_common_dialog();
        }
        SDL_zero(dialog_result);
        sceMsgDialogGetResult(&dialog_result);

        if (dialog_result.buttonId == SCE_MSG_DIALOG_BUTTON_ID_BUTTON1) {
            *buttonID = messageboxdata->buttons[0].buttonID;
        } else if (dialog_result.buttonId == SCE_MSG_DIALOG_BUTTON_ID_BUTTON2) {
            *buttonID = messageboxdata->buttons[1].buttonID;
        } else if (dialog_result.buttonId == SCE_MSG_DIALOG_BUTTON_ID_BUTTON3) {
            *buttonID = messageboxdata->buttons[2].buttonID;
        } else if (dialog_result.buttonId == SCE_MSG_DIALOG_BUTTON_ID_YES) {
            *buttonID = messageboxdata->buttons[0].buttonID;
        } else if (dialog_result.buttonId == SCE_MSG_DIALOG_BUTTON_ID_NO) {
            *buttonID = messageboxdata->buttons[1].buttonID;
        } else if (dialog_result.buttonId == SCE_MSG_DIALOG_BUTTON_ID_OK) {
            *buttonID = messageboxdata->buttons[0].buttonID;
        }
        sceMsgDialogTerm();
    } else {
        return false;
    }

    gxm_term_for_common_dialog();

    if (setup_minimal_gxm) {
        gxm_minimal_term_for_common_dialog();
    }

    return true;
#else
    (void)messageboxdata;
    (void)buttonID;
    return SDL_Unsupported();
#endif // SDL_VIDEO_RENDER_VITA_GXM
}

#endif // SDL_VIDEO_DRIVER_VITA

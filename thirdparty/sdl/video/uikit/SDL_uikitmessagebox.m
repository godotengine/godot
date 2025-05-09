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

#ifdef SDL_VIDEO_DRIVER_UIKIT

#include "SDL_uikitvideo.h"
#include "SDL_uikitwindow.h"

// Display a UIKit message box

static bool s_showingMessageBox = false;

bool UIKit_ShowingMessageBox(void)
{
    return s_showingMessageBox;
}

static void UIKit_WaitUntilMessageBoxClosed(const SDL_MessageBoxData *messageboxdata, int *clickedindex)
{
    *clickedindex = messageboxdata->numbuttons;

    @autoreleasepool {
        // Run the main event loop until the alert has finished
        // Note that this needs to be done on the main thread
        s_showingMessageBox = true;
        while ((*clickedindex) == messageboxdata->numbuttons) {
            [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate distantFuture]];
        }
        s_showingMessageBox = false;
    }
}

static BOOL UIKit_ShowMessageBoxAlertController(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    int i;
    int __block clickedindex = messageboxdata->numbuttons;
    UIWindow *window = nil;
    UIWindow *alertwindow = nil;

    if (![UIAlertController class]) {
        return NO;
    }

    UIAlertController *alert;
    alert = [UIAlertController alertControllerWithTitle:@(messageboxdata->title)
                                                message:@(messageboxdata->message)
                                         preferredStyle:UIAlertControllerStyleAlert];

    for (i = 0; i < messageboxdata->numbuttons; i++) {
        UIAlertAction *action;
        UIAlertActionStyle style = UIAlertActionStyleDefault;
        const SDL_MessageBoxButtonData *sdlButton;

        if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_RIGHT_TO_LEFT) {
            sdlButton = &messageboxdata->buttons[messageboxdata->numbuttons - 1 - i];
        } else {
            sdlButton = &messageboxdata->buttons[i];
        }

        if (sdlButton->flags & SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT) {
            style = UIAlertActionStyleCancel;
        }

        action = [UIAlertAction actionWithTitle:@(sdlButton->text)
                                          style:style
                                        handler:^(UIAlertAction *alertAction) {
                                          clickedindex = (int)(sdlButton - messageboxdata->buttons);
                                        }];
        [alert addAction:action];

        if (sdlButton->flags & SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT) {
            alert.preferredAction = action;
        }
    }

    if (messageboxdata->window) {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)messageboxdata->window->internal;
        window = data.uiwindow;
    }

    if (window == nil || window.rootViewController == nil) {
#ifdef SDL_PLATFORM_VISIONOS
        alertwindow = [[UIWindow alloc] init];
#else
        alertwindow = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
#endif
        alertwindow.rootViewController = [UIViewController new];
        alertwindow.windowLevel = UIWindowLevelAlert;

        window = alertwindow;

        [alertwindow makeKeyAndVisible];
    }

    [window.rootViewController presentViewController:alert animated:YES completion:nil];
    UIKit_WaitUntilMessageBoxClosed(messageboxdata, &clickedindex);

    if (alertwindow) {
        alertwindow.hidden = YES;
    }

    UIKit_ForceUpdateHomeIndicator();

    *buttonID = messageboxdata->buttons[clickedindex].buttonID;
    return YES;
}

static void UIKit_ShowMessageBoxImpl(const SDL_MessageBoxData *messageboxdata, int *buttonID, int *result)
{
    @autoreleasepool {
        if (UIKit_ShowMessageBoxAlertController(messageboxdata, buttonID)) {
            *result = true;
        } else {
            *result = SDL_SetError("Could not show message box.");
        }
    }
}

bool UIKit_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    @autoreleasepool {
        __block int result = true;

        if ([NSThread isMainThread]) {
            UIKit_ShowMessageBoxImpl(messageboxdata, buttonID, &result);
        } else {
            dispatch_sync(dispatch_get_main_queue(), ^{
              UIKit_ShowMessageBoxImpl(messageboxdata, buttonID, &result);
            });
        }
        return result;
    }
}

#endif // SDL_VIDEO_DRIVER_UIKIT

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

#ifdef SDL_VIDEO_DRIVER_COCOA

#include "SDL_cocoavideo.h"

@interface SDL3MessageBoxPresenter : NSObject
{
  @public
    NSInteger clicked;
    NSWindow *nswindow;
}
- (id)initWithParentWindow:(SDL_Window *)window;
@end

@implementation SDL3MessageBoxPresenter
- (id)initWithParentWindow:(SDL_Window *)window
{
    self = [super init];
    if (self) {
        clicked = -1;

        // Retain the NSWindow because we'll show the alert later on the main thread
        if (window) {
            nswindow = ((__bridge SDL_CocoaWindowData *)window->internal).nswindow;
        } else {
            nswindow = nil;
        }
    }

    return self;
}

- (void)showAlert:(NSAlert *)alert
{
    if (nswindow) {
        [alert beginSheetModalForWindow:nswindow
                      completionHandler:^(NSModalResponse returnCode) {
                        [NSApp stopModalWithCode:returnCode];
                      }];
        clicked = [NSApp runModalForWindow:nswindow];
        nswindow = nil;
    } else {
        clicked = [alert runModal];
    }
}
@end

static void Cocoa_ShowMessageBoxImpl(const SDL_MessageBoxData *messageboxdata, int *buttonID, bool *result)
{
    NSAlert *alert;
    const SDL_MessageBoxButtonData *buttons = messageboxdata->buttons;
    SDL3MessageBoxPresenter *presenter;
    NSInteger clicked;
    int i;
    Cocoa_RegisterApp();

    alert = [[NSAlert alloc] init];

    if (messageboxdata->flags & SDL_MESSAGEBOX_ERROR) {
        [alert setAlertStyle:NSAlertStyleCritical];
    } else if (messageboxdata->flags & SDL_MESSAGEBOX_WARNING) {
        [alert setAlertStyle:NSAlertStyleWarning];
    } else {
        [alert setAlertStyle:NSAlertStyleInformational];
    }

    [alert setMessageText:[NSString stringWithUTF8String:messageboxdata->title]];
    [alert setInformativeText:[NSString stringWithUTF8String:messageboxdata->message]];

    for (i = 0; i < messageboxdata->numbuttons; ++i) {
        const SDL_MessageBoxButtonData *sdlButton;
        NSButton *button;

        if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_RIGHT_TO_LEFT) {
            sdlButton = &messageboxdata->buttons[messageboxdata->numbuttons - 1 - i];
        } else {
            sdlButton = &messageboxdata->buttons[i];
        }

        button = [alert addButtonWithTitle:[NSString stringWithUTF8String:sdlButton->text]];
        if (sdlButton->flags & SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT) {
            [button setKeyEquivalent:@"\r"];
        } else if (sdlButton->flags & SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT) {
            [button setKeyEquivalent:@"\033"];
        } else {
            [button setKeyEquivalent:@""];
        }
    }

    presenter = [[SDL3MessageBoxPresenter alloc] initWithParentWindow:messageboxdata->window];

    [presenter showAlert:alert];

    clicked = presenter->clicked;
    if (clicked >= NSAlertFirstButtonReturn) {
        clicked -= NSAlertFirstButtonReturn;
        if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_RIGHT_TO_LEFT) {
            clicked = messageboxdata->numbuttons - 1 - clicked;
        }
        *buttonID = buttons[clicked].buttonID;
        *result = true;
    } else {
        *result = SDL_SetError("Did not get a valid `clicked button' id: %ld", (long)clicked);
    }
}

// Display a Cocoa message box
bool Cocoa_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    @autoreleasepool {
        __block bool result = 0;

        if ([NSThread isMainThread]) {
            Cocoa_ShowMessageBoxImpl(messageboxdata, buttonID, &result);
        } else {
            dispatch_sync(dispatch_get_main_queue(), ^{
              Cocoa_ShowMessageBoxImpl(messageboxdata, buttonID, &result);
            });
        }
        return result;
    }
}

#endif // SDL_VIDEO_DRIVER_COCOA

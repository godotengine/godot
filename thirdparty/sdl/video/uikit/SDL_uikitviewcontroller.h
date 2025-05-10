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

#import <UIKit/UIKit.h>

#include "../SDL_sysvideo.h"

#ifdef SDL_PLATFORM_TVOS
#import <GameController/GameController.h>
#define SDLRootViewController GCEventViewController
#else
#define SDLRootViewController UIViewController
#endif

@interface SDLUITextField : UITextField
- (BOOL)canPerformAction:(SEL)action withSender:(id)sender;
@end

#ifdef SDL_IPHONE_KEYBOARD
@interface SDL_uikitviewcontroller : SDLRootViewController <UITextFieldDelegate>
#else
@interface SDL_uikitviewcontroller : SDLRootViewController
#endif

@property(nonatomic, assign) SDL_Window *window;

- (instancetype)initWithSDLWindow:(SDL_Window *)_window;

- (void)traitCollectionDidChange:(UITraitCollection *)previousTraitCollection;

- (void)setAnimationCallback:(int)interval
                    callback:(void (*)(void *))callback
               callbackParam:(void *)callbackParam;

- (void)startAnimation;
- (void)stopAnimation;

- (void)doLoop:(CADisplayLink *)sender;

- (void)loadView;
- (void)viewDidLayoutSubviews;

#ifndef SDL_PLATFORM_TVOS
- (NSUInteger)supportedInterfaceOrientations;
- (BOOL)prefersStatusBarHidden;
- (BOOL)prefersHomeIndicatorAutoHidden;
- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures;

@property(nonatomic, assign) int homeIndicatorHidden;
#endif

#ifdef SDL_IPHONE_KEYBOARD
- (bool)startTextInput;
- (bool)stopTextInput;
- (void)initKeyboard;
- (void)deinitKeyboard;

- (void)keyboardWillShow:(NSNotification *)notification;
- (void)keyboardWillHide:(NSNotification *)notification;

- (void)updateKeyboard;

@property(nonatomic, assign, getter=isTextFieldFocused) BOOL textFieldFocused;
@property(nonatomic, assign) SDL_Rect textInputRect;
@property(nonatomic, assign) int keyboardHeight;
#endif

@end

#ifdef SDL_IPHONE_KEYBOARD
bool UIKit_HasScreenKeyboardSupport(SDL_VideoDevice *_this);
bool UIKit_StartTextInput(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props);
bool UIKit_StopTextInput(SDL_VideoDevice *_this, SDL_Window *window);
void UIKit_SetTextInputProperties(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props);
bool UIKit_IsScreenKeyboardShown(SDL_VideoDevice *_this, SDL_Window *window);
bool UIKit_UpdateTextInputArea(SDL_VideoDevice *_this, SDL_Window *window);
#endif

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

#include "../SDL_sysvideo.h"
#include "../../events/SDL_events_c.h"

#include "SDL_uikitviewcontroller.h"
#include "SDL_uikitmessagebox.h"
#include "SDL_uikitevents.h"
#include "SDL_uikitvideo.h"
#include "SDL_uikitmodes.h"
#include "SDL_uikitwindow.h"
#include "SDL_uikitopengles.h"

#ifdef SDL_PLATFORM_TVOS
static void SDLCALL SDL_AppleTVControllerUIHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    @autoreleasepool {
        SDL_uikitviewcontroller *viewcontroller = (__bridge SDL_uikitviewcontroller *)userdata;
        viewcontroller.controllerUserInteractionEnabled = hint && (*hint != '0');
    }
}
#endif

#ifndef SDL_PLATFORM_TVOS
static void SDLCALL SDL_HideHomeIndicatorHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    @autoreleasepool {
        SDL_uikitviewcontroller *viewcontroller = (__bridge SDL_uikitviewcontroller *)userdata;
        viewcontroller.homeIndicatorHidden = (hint && *hint) ? SDL_atoi(hint) : -1;
        [viewcontroller setNeedsUpdateOfHomeIndicatorAutoHidden];
        [viewcontroller setNeedsUpdateOfScreenEdgesDeferringSystemGestures];
    }
}
#endif

@implementation SDLUITextField : UITextField
- (BOOL)canPerformAction:(SEL)action withSender:(id)sender
{
    if (action == @selector(paste:)) {
        return NO;
    }

    return [super canPerformAction:action withSender:sender];
}
@end

@implementation SDL_uikitviewcontroller
{
    CADisplayLink *displayLink;
    int animationInterval;
    void (*animationCallback)(void *);
    void *animationCallbackParam;

#ifdef SDL_IPHONE_KEYBOARD
    SDLUITextField *textField;
    BOOL hidingKeyboard;
    BOOL rotatingOrientation;
    NSString *committedText;
    NSString *obligateForBackspace;
#endif
}

@synthesize window;

- (instancetype)initWithSDLWindow:(SDL_Window *)_window
{
    if (self = [super initWithNibName:nil bundle:nil]) {
        self.window = _window;

#ifdef SDL_IPHONE_KEYBOARD
        [self initKeyboard];
        hidingKeyboard = NO;
        rotatingOrientation = NO;
#endif

#ifdef SDL_PLATFORM_TVOS
        SDL_AddHintCallback(SDL_HINT_APPLE_TV_CONTROLLER_UI_EVENTS,
                            SDL_AppleTVControllerUIHintChanged,
                            (__bridge void *)self);
#endif

#ifndef SDL_PLATFORM_TVOS
        SDL_AddHintCallback(SDL_HINT_IOS_HIDE_HOME_INDICATOR,
                            SDL_HideHomeIndicatorHintChanged,
                            (__bridge void *)self);
#endif

        // Enable high refresh rates on iOS
        // To enable this on phones, you should add the following line to Info.plist:
        // <key>CADisableMinimumFrameDurationOnPhone</key> <true/>
        if (@available(iOS 15.0, tvOS 15.0, *)) {
            const SDL_DisplayMode *mode = SDL_GetDesktopDisplayMode(SDL_GetPrimaryDisplay());
            if (mode && mode->refresh_rate > 60.0f) {
                int frame_rate = (int)mode->refresh_rate;
                displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(doLoop:)];
                displayLink.preferredFrameRateRange = CAFrameRateRangeMake((frame_rate * 2) / 3, frame_rate, frame_rate);
                [displayLink addToRunLoop:NSRunLoop.currentRunLoop forMode:NSDefaultRunLoopMode];
            }
        }
    }
    return self;
}

- (void)dealloc
{
#ifdef SDL_IPHONE_KEYBOARD
    [self deinitKeyboard];
#endif

#ifdef SDL_PLATFORM_TVOS
    SDL_RemoveHintCallback(SDL_HINT_APPLE_TV_CONTROLLER_UI_EVENTS,
                        SDL_AppleTVControllerUIHintChanged,
                        (__bridge void *)self);
#endif

#ifndef SDL_PLATFORM_TVOS
    SDL_RemoveHintCallback(SDL_HINT_IOS_HIDE_HOME_INDICATOR,
                        SDL_HideHomeIndicatorHintChanged,
                        (__bridge void *)self);
#endif
}

- (void)traitCollectionDidChange:(UITraitCollection *)previousTraitCollection
{
    SDL_SetSystemTheme(UIKit_GetSystemTheme());
}

- (void)setAnimationCallback:(int)interval
                    callback:(void (*)(void *))callback
               callbackParam:(void *)callbackParam
{
    [self stopAnimation];

    if (interval <= 0) {
        interval = 1;
    }
    animationInterval = interval;
    animationCallback = callback;
    animationCallbackParam = callbackParam;

    if (animationCallback) {
        [self startAnimation];
    }
}

- (void)startAnimation
{
    displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(doLoop:)];

#ifdef SDL_PLATFORM_VISIONOS
    displayLink.preferredFramesPerSecond = 90 / animationInterval;      //TODO: Get frame max frame rate on visionOS
#else
    SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;

    displayLink.preferredFramesPerSecond = data.uiwindow.screen.maximumFramesPerSecond / animationInterval;
#endif

    [displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
}

- (void)stopAnimation
{
    [displayLink invalidate];
    displayLink = nil;
}

- (void)doLoop:(CADisplayLink *)sender
{
    // Don't run the game loop while a messagebox is up
    if (animationCallback && !UIKit_ShowingMessageBox()) {
        // See the comment in the function definition.
#if defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
        UIKit_GL_RestoreCurrentContext();
#endif

        animationCallback(animationCallbackParam);
    }
}

- (void)loadView
{
    // Do nothing.
}

- (void)viewDidLayoutSubviews
{
    const CGSize size = self.view.bounds.size;
    int w = (int)size.width;
    int h = (int)size.height;

    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESIZED, w, h);
}

#ifndef SDL_PLATFORM_TVOS
- (NSUInteger)supportedInterfaceOrientations
{
    return UIKit_GetSupportedOrientations(window);
}

- (BOOL)prefersStatusBarHidden
{
    BOOL hidden = (window->flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS)) != 0;
    return hidden;
}

- (BOOL)prefersHomeIndicatorAutoHidden
{
    BOOL hidden = NO;
    if (self.homeIndicatorHidden == 1) {
        hidden = YES;
    }
    return hidden;
}

- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures
{
    if (self.homeIndicatorHidden >= 0) {
        if (self.homeIndicatorHidden == 2) {
            return UIRectEdgeAll;
        } else {
            return UIRectEdgeNone;
        }
    }

    // By default, fullscreen and borderless windows get all screen gestures
    if ((window->flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS)) != 0) {
        return UIRectEdgeAll;
    } else {
        return UIRectEdgeNone;
    }
}

- (BOOL)prefersPointerLocked
{
    return SDL_GCMouseRelativeMode() ? YES : NO;
}

#endif // !SDL_PLATFORM_TVOS

/*
 ---- Keyboard related functionality below this line ----
 */
#ifdef SDL_IPHONE_KEYBOARD

@synthesize textInputRect;
@synthesize keyboardHeight;
@synthesize textFieldFocused;

// Set ourselves up as a UITextFieldDelegate
- (void)initKeyboard
{
    obligateForBackspace = @"                                                                "; // 64 space
    textField = [[SDLUITextField alloc] initWithFrame:CGRectZero];
    textField.delegate = self;
    // placeholder so there is something to delete!
    textField.text = obligateForBackspace;
    committedText = textField.text;

    textField.hidden = YES;
    textFieldFocused = NO;

    NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
#ifndef SDL_PLATFORM_TVOS
    [center addObserver:self
               selector:@selector(keyboardWillShow:)
                   name:UIKeyboardWillShowNotification
                 object:nil];
    [center addObserver:self
               selector:@selector(keyboardWillHide:)
                   name:UIKeyboardWillHideNotification
                 object:nil];
    [center addObserver:self
               selector:@selector(keyboardDidHide:)
                   name:UIKeyboardDidHideNotification
                 object:nil];
#endif
    [center addObserver:self
               selector:@selector(textFieldTextDidChange:)
                   name:UITextFieldTextDidChangeNotification
                 object:nil];
}

- (NSArray *)keyCommands
{
    NSMutableArray *commands = [[NSMutableArray alloc] init];
    [commands addObject:[UIKeyCommand keyCommandWithInput:UIKeyInputUpArrow modifierFlags:kNilOptions action:@selector(handleCommand:)]];
    [commands addObject:[UIKeyCommand keyCommandWithInput:UIKeyInputDownArrow modifierFlags:kNilOptions action:@selector(handleCommand:)]];
    [commands addObject:[UIKeyCommand keyCommandWithInput:UIKeyInputLeftArrow modifierFlags:kNilOptions action:@selector(handleCommand:)]];
    [commands addObject:[UIKeyCommand keyCommandWithInput:UIKeyInputRightArrow modifierFlags:kNilOptions action:@selector(handleCommand:)]];
    [commands addObject:[UIKeyCommand keyCommandWithInput:UIKeyInputEscape modifierFlags:kNilOptions action:@selector(handleCommand:)]];
    return [NSArray arrayWithArray:commands];
}

- (void)handleCommand:(UIKeyCommand *)keyCommand
{
    SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;
    NSString *input = keyCommand.input;

    if (input == UIKeyInputUpArrow) {
        scancode = SDL_SCANCODE_UP;
    } else if (input == UIKeyInputDownArrow) {
        scancode = SDL_SCANCODE_DOWN;
    } else if (input == UIKeyInputLeftArrow) {
        scancode = SDL_SCANCODE_LEFT;
    } else if (input == UIKeyInputRightArrow) {
        scancode = SDL_SCANCODE_RIGHT;
    } else if (input == UIKeyInputEscape) {
        scancode = SDL_SCANCODE_ESCAPE;
    }

    if (scancode != SDL_SCANCODE_UNKNOWN) {
        SDL_SendKeyboardKeyAutoRelease(0, scancode);
    }
}

- (void)setView:(UIView *)view
{
    [super setView:view];

    [view addSubview:textField];

    if (textFieldFocused) {
        /* startTextInput has been called before the text field was added to the view,
         * call it again for the text field to actually become first responder. */
        [self startTextInput];
    }
}

- (void)viewWillTransitionToSize:(CGSize)size withTransitionCoordinator:(id<UIViewControllerTransitionCoordinator>)coordinator
{
    [super viewWillTransitionToSize:size withTransitionCoordinator:coordinator];
    rotatingOrientation = YES;
    [coordinator
        animateAlongsideTransition:^(id<UIViewControllerTransitionCoordinatorContext> context) {
        }
        completion:^(id<UIViewControllerTransitionCoordinatorContext> context) {
          self->rotatingOrientation = NO;
        }];
}

- (void)deinitKeyboard
{
    NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
#ifndef SDL_PLATFORM_TVOS
    [center removeObserver:self
                      name:UIKeyboardWillShowNotification
                    object:nil];
    [center removeObserver:self
                      name:UIKeyboardWillHideNotification
                    object:nil];
    [center removeObserver:self
                      name:UIKeyboardDidHideNotification
                    object:nil];
#endif
    [center removeObserver:self
                      name:UITextFieldTextDidChangeNotification
                    object:nil];
}

- (void)setTextFieldProperties:(SDL_PropertiesID) props
{
    textField.secureTextEntry = NO;

    switch (SDL_GetTextInputType(props)) {
    default:
    case SDL_TEXTINPUT_TYPE_TEXT:
        textField.keyboardType = UIKeyboardTypeDefault;
        textField.textContentType = nil;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_NAME:
        textField.keyboardType = UIKeyboardTypeDefault;
        textField.textContentType = UITextContentTypeName;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_EMAIL:
        textField.keyboardType = UIKeyboardTypeEmailAddress;
        textField.textContentType = UITextContentTypeEmailAddress;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_USERNAME:
        textField.keyboardType = UIKeyboardTypeDefault;
        textField.textContentType = UITextContentTypeUsername;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_HIDDEN:
        textField.keyboardType = UIKeyboardTypeDefault;
        textField.textContentType = UITextContentTypePassword;
        textField.secureTextEntry = YES;
        break;
    case SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_VISIBLE:
        textField.keyboardType = UIKeyboardTypeDefault;
        textField.textContentType = UITextContentTypePassword;
        break;
    case SDL_TEXTINPUT_TYPE_NUMBER:
        textField.keyboardType = UIKeyboardTypeDecimalPad;
        textField.textContentType = nil;
        break;
    case SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_HIDDEN:
        textField.keyboardType = UIKeyboardTypeNumberPad;
        if (@available(iOS 12.0, tvOS 12.0, *)) {
            textField.textContentType = UITextContentTypeOneTimeCode;
        } else {
            textField.textContentType = nil;
        }
        textField.secureTextEntry = YES;
        break;
    case SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_VISIBLE:
        textField.keyboardType = UIKeyboardTypeNumberPad;
        if (@available(iOS 12.0, tvOS 12.0, *)) {
            textField.textContentType = UITextContentTypeOneTimeCode;
        } else {
            textField.textContentType = nil;
        }
        break;
    }

    switch (SDL_GetTextInputCapitalization(props)) {
    default:
    case SDL_CAPITALIZE_NONE:
        textField.autocapitalizationType = UITextAutocapitalizationTypeNone;
        break;
    case SDL_CAPITALIZE_LETTERS:
        textField.autocapitalizationType = UITextAutocapitalizationTypeAllCharacters;
        break;
    case SDL_CAPITALIZE_WORDS:
        textField.autocapitalizationType = UITextAutocapitalizationTypeWords;
        break;
    case SDL_CAPITALIZE_SENTENCES:
        textField.autocapitalizationType = UITextAutocapitalizationTypeSentences;
        break;
    }

    if (SDL_GetTextInputAutocorrect(props)) {
        textField.autocorrectionType = UITextAutocorrectionTypeYes;
        textField.spellCheckingType = UITextSpellCheckingTypeYes;
    } else {
        textField.autocorrectionType = UITextAutocorrectionTypeNo;
        textField.spellCheckingType = UITextSpellCheckingTypeNo;
    }

    if (SDL_GetTextInputMultiline(props)) {
        textField.enablesReturnKeyAutomatically = YES;
    } else {
        textField.enablesReturnKeyAutomatically = NO;
    }

    if (!textField.window) {
        /* textField has not been added to the view yet,
         we don't have to do anything. */
        return;
    }

    // the text field needs to be re-added to the view in order to update correctly.
    UIView *superview = textField.superview;
    [textField removeFromSuperview];
    [superview addSubview:textField];

    if (SDL_TextInputActive(window)) {
        [textField becomeFirstResponder];
    }
}

/* requests the SDL text field to become focused and accept text input.
 * also shows the onscreen virtual keyboard if no hardware keyboard is attached. */
- (bool)startTextInput
{
    textFieldFocused = YES;
    if (!textField.window) {
        /* textField has not been added to the view yet,
         * we will try again when that happens. */
        return true;
    }

    return [textField becomeFirstResponder];
}

/* requests the SDL text field to lose focus and stop accepting text input.
 * also hides the onscreen virtual keyboard if no hardware keyboard is attached. */
- (bool)stopTextInput
{
    textFieldFocused = NO;
    if (!textField.window) {
        /* textField has not been added to the view yet,
         * we will try again when that happens. */
        return true;
    }

    [self resetTextState];
    return [textField resignFirstResponder];
}

- (void)keyboardWillShow:(NSNotification *)notification
{
#ifndef SDL_PLATFORM_TVOS
    CGRect kbrect = [[notification userInfo][UIKeyboardFrameEndUserInfoKey] CGRectValue];

    /* The keyboard rect is in the coordinate space of the screen/window, but we
     * want its height in the coordinate space of the view. */
    kbrect = [self.view convertRect:kbrect fromView:nil];

    [self setKeyboardHeight:(int)kbrect.size.height];
#endif

    /* A keyboard hide transition has been interrupted with a show (keyboardWillHide has been called but keyboardDidHide didn't).
     * since text input was stopped by the hide, we have to start it again. */
    if (hidingKeyboard) {
        SDL_StartTextInput(window);
        hidingKeyboard = NO;
    }
}

- (void)keyboardWillHide:(NSNotification *)notification
{
    hidingKeyboard = YES;
    [self setKeyboardHeight:0];

    /* When the user dismisses the software keyboard by the "hide" button in the bottom right corner,
     * we want to reflect that on SDL_TextInputActive by calling SDL_StopTextInput...on certain conditions */
    if (SDL_TextInputActive(window)
        /* keyboardWillHide gets called when a hardware keyboard is attached,
         * keep text input state active if hiding while there is a hardware keyboard.
         * if the hardware keyboard gets detached, the software keyboard will appear anyway. */
        && !SDL_HasKeyboard()
        /* When the device changes orientation, a sequence of hide and show transitions are triggered.
         * keep text input state active in this case. */
        && !rotatingOrientation) {
        SDL_StopTextInput(window);
    }
}

- (void)keyboardDidHide:(NSNotification *)notification
{
    hidingKeyboard = NO;
}

- (void)textFieldTextDidChange:(NSNotification *)notification
{
    if (textField.markedTextRange == nil) {
        NSUInteger compareLength = SDL_min(textField.text.length, committedText.length);
        NSUInteger matchLength;

        // Backspace over characters that are no longer in the string
        for (matchLength = 0; matchLength < compareLength; ++matchLength) {
            if ([committedText characterAtIndex:matchLength] != [textField.text characterAtIndex:matchLength]) {
                break;
            }
        }
        if (matchLength < committedText.length) {
            size_t deleteLength = SDL_utf8strlen([[committedText substringFromIndex:matchLength] UTF8String]);
            while (deleteLength > 0) {
                // Send distinct down and up events for each backspace action
                SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_BACKSPACE, true);
                SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_BACKSPACE, false);
                --deleteLength;
            }
        }

        if (matchLength < textField.text.length) {
            NSString *pendingText = [textField.text substringFromIndex:matchLength];
            if (!SDL_HardwareKeyboardKeyPressed()) {
                /* Go through all the characters in the string we've been sent and
                 * convert them to key presses */
                NSUInteger i;
                for (i = 0; i < pendingText.length; i++) {
                    SDL_SendKeyboardUnicodeKey(0, [pendingText characterAtIndex:i]);
                }
            }
            SDL_SendKeyboardText([pendingText UTF8String]);
        }
        committedText = textField.text;
    }
}

- (void)updateKeyboard
{
    SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *) window->internal;

    CGAffineTransform t = self.view.transform;
    CGPoint offset = CGPointMake(0.0, 0.0);
#ifdef SDL_PLATFORM_VISIONOS
    CGRect frame = UIKit_ComputeViewFrame(window);
#else
    CGRect frame = UIKit_ComputeViewFrame(window, data.uiwindow.screen);
#endif

    if (self.keyboardHeight && self.textInputRect.h) {
        int rectbottom = (int)(self.textInputRect.y + self.textInputRect.h);
        int keybottom = (int)(self.view.bounds.size.height - self.keyboardHeight);
        if (keybottom < rectbottom) {
            offset.y = keybottom - rectbottom;
        }
    }

    /* Apply this view's transform (except any translation) to the offset, in
     * order to orient it correctly relative to the frame's coordinate space. */
    t.tx = 0.0;
    t.ty = 0.0;
    offset = CGPointApplyAffineTransform(offset, t);

    // Apply the updated offset to the view's frame.
    frame.origin.x += offset.x;
    frame.origin.y += offset.y;

    self.view.frame = frame;
}

- (void)setKeyboardHeight:(int)height
{
    keyboardHeight = height;
    [self updateKeyboard];
}

// UITextFieldDelegate method.  Invoked when user types something.
- (BOOL)textField:(UITextField *)_textField shouldChangeCharactersInRange:(NSRange)range replacementString:(NSString *)string
{
    if (textField.markedTextRange == nil) {
        if (textField.text.length < 16) {
            [self resetTextState];
        }
    }
    return YES;
}

// Terminates the editing session
- (BOOL)textFieldShouldReturn:(UITextField *)_textField
{
    SDL_SendKeyboardKeyAutoRelease(0, SDL_SCANCODE_RETURN);
    if (textFieldFocused &&
        SDL_GetHintBoolean(SDL_HINT_RETURN_KEY_HIDES_IME, false)) {
        SDL_StopTextInput(window);
    }
    return YES;
}

- (void)resetTextState
{
    textField.text = obligateForBackspace;
    committedText = textField.text;
}

#endif

@end

// iPhone keyboard addition functions
#ifdef SDL_IPHONE_KEYBOARD

static SDL_uikitviewcontroller *GetWindowViewController(SDL_Window *window)
{
    if (!window || !window->internal) {
        SDL_SetError("Invalid window");
        return nil;
    }

    SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;

    return data.viewcontroller;
}

bool UIKit_HasScreenKeyboardSupport(SDL_VideoDevice *_this)
{
    return true;
}

bool UIKit_StartTextInput(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props)
{
    @autoreleasepool {
        SDL_uikitviewcontroller *vc = GetWindowViewController(window);
        return [vc startTextInput];
    }
}

bool UIKit_StopTextInput(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_uikitviewcontroller *vc = GetWindowViewController(window);
        return [vc stopTextInput];
    }
}

void UIKit_SetTextInputProperties(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props)
{
    @autoreleasepool {
        SDL_uikitviewcontroller *vc = GetWindowViewController(window);
        [vc setTextFieldProperties:props];
    }
}

bool UIKit_IsScreenKeyboardShown(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_uikitviewcontroller *vc = GetWindowViewController(window);
        if (vc != nil) {
            return vc.textFieldFocused;
        }
        return false;
    }
}

bool UIKit_UpdateTextInputArea(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_uikitviewcontroller *vc = GetWindowViewController(window);
        if (vc != nil) {
            vc.textInputRect = window->text_input_rect;

            if (vc.textFieldFocused) {
                [vc updateKeyboard];
            }
        }
    }
    return true;
}

#endif // SDL_IPHONE_KEYBOARD

#endif // SDL_VIDEO_DRIVER_UIKIT

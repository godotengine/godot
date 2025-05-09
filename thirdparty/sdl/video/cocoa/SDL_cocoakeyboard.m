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

#include "../../events/SDL_events_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/scancodes_darwin.h"

#include <Carbon/Carbon.h>

#if 0
#define DEBUG_IME NSLog
#else
#define DEBUG_IME(...)
#endif

@interface SDL3TranslatorResponder : NSView <NSTextInputClient>
{
    NSString *_markedText;
    NSRange _markedRange;
    NSRange _selectedRange;
    SDL_Rect _inputRect;
    int _pendingRawCode;
    SDL_Scancode _pendingScancode;
    Uint64 _pendingTimestamp;
}
- (void)doCommandBySelector:(SEL)myselector;
- (void)setInputRect:(const SDL_Rect *)rect;
- (void)setPendingKey:(int)rawcode scancode:(SDL_Scancode)scancode timestamp:(Uint64)timestamp;
- (void)sendPendingKey;
- (void)clearPendingKey;
@end

@implementation SDL3TranslatorResponder

- (void)setInputRect:(const SDL_Rect *)rect
{
    SDL_copyp(&_inputRect, rect);
}

- (void)insertText:(id)aString replacementRange:(NSRange)replacementRange
{
    const char *str;

    DEBUG_IME(@"insertText: %@ replacementRange: (%d, %d)", aString,
              (int)replacementRange.location, (int)replacementRange.length);

    /* Could be NSString or NSAttributedString, so we have
     * to test and convert it before return as SDL event */
    if ([aString isKindOfClass:[NSAttributedString class]]) {
        str = [[aString string] UTF8String];
    } else {
        str = [aString UTF8String];
    }

    // We're likely sending the composed text, so we reset the IME status.
    if ([self hasMarkedText]) {
        [self unmarkText];
    }

    // Deliver the raw key event that generated this text
    [self sendPendingKey];

    if ((int)replacementRange.location != -1) {
        // We're replacing the last character
        SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_BACKSPACE, true);
        SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_BACKSPACE, false);
    }

    SDL_SendKeyboardText(str);
}

- (void)doCommandBySelector:(SEL)myselector
{
    /* No need to do anything since we are not using Cocoa
       selectors to handle special keys, instead we use SDL
       key events to do the same job.
    */
}

- (BOOL)hasMarkedText
{
    return _markedText != nil;
}

- (NSRange)markedRange
{
    return _markedRange;
}

- (NSRange)selectedRange
{
    return _selectedRange;
}

- (void)setMarkedText:(id)aString selectedRange:(NSRange)selectedRange replacementRange:(NSRange)replacementRange
{
    if ([aString isKindOfClass:[NSAttributedString class]]) {
        aString = [aString string];
    }

    if ([aString length] == 0) {
        [self unmarkText];
        return;
    }

    if (_markedText != aString) {
        _markedText = aString;
    }

    _selectedRange = selectedRange;
    _markedRange = NSMakeRange(0, [aString length]);

    // This key event was consumed by the IME
    [self clearPendingKey];

    NSUInteger utf32SelectedRangeLocation = [[aString substringToIndex:selectedRange.location] lengthOfBytesUsingEncoding:NSUTF32StringEncoding] / 4;
    NSUInteger utf32SelectionRangeEnd = [[aString substringToIndex:(selectedRange.location + selectedRange.length)] lengthOfBytesUsingEncoding:NSUTF32StringEncoding] / 4;
    NSUInteger utf32SelectionRangeLength = utf32SelectionRangeEnd - utf32SelectedRangeLocation;

    SDL_SendEditingText([aString UTF8String],
                        (int)utf32SelectedRangeLocation, (int)utf32SelectionRangeLength);

    DEBUG_IME(@"setMarkedText: %@, (%d, %d) replacement range (%d, %d)", _markedText,
              (int)selectedRange.location, (int)selectedRange.length,
              (int)replacementRange.location, (int)replacementRange.length);
}

- (void)unmarkText
{
    _markedText = nil;

    // This key event was consumed by the IME
    [self clearPendingKey];

    SDL_SendEditingText("", 0, 0);
}

- (NSRect)firstRectForCharacterRange:(NSRange)aRange actualRange:(NSRangePointer)actualRange
{
    NSWindow *window = [self window];
    NSRect contentRect = [window contentRectForFrameRect:[window frame]];
    float windowHeight = contentRect.size.height;
    NSRect rect = NSMakeRect(_inputRect.x, windowHeight - _inputRect.y - _inputRect.h,
                             _inputRect.w, _inputRect.h);

    if (actualRange) {
        *actualRange = aRange;
    }

    DEBUG_IME(@"firstRectForCharacterRange: (%d, %d): windowHeight = %g, rect = %@",
              (int)aRange.location, (int)aRange.length, windowHeight,
              NSStringFromRect(rect));

    rect = [window convertRectToScreen:rect];

    return rect;
}

- (NSAttributedString *)attributedSubstringForProposedRange:(NSRange)aRange actualRange:(NSRangePointer)actualRange
{
    DEBUG_IME(@"attributedSubstringFromRange: (%d, %d)", (int)aRange.location, (int)aRange.length);
    return nil;
}

- (NSInteger)conversationIdentifier
{
    return (NSInteger)self;
}

/* This method returns the index for character that is
 * nearest to thePoint.  thPoint is in screen coordinate system.
 */
- (NSUInteger)characterIndexForPoint:(NSPoint)thePoint
{
    DEBUG_IME(@"characterIndexForPoint: (%g, %g)", thePoint.x, thePoint.y);
    return 0;
}

/* This method is the key to attribute extension.
 * We could add new attributes through this method.
 * NSInputServer examines the return value of this
 * method & constructs appropriate attributed string.
 */
- (NSArray *)validAttributesForMarkedText
{
    return [NSArray array];
}

- (void)setPendingKey:(int)rawcode scancode:(SDL_Scancode)scancode timestamp:(Uint64)timestamp
{
    _pendingRawCode = rawcode;
    _pendingScancode = scancode;
    _pendingTimestamp = timestamp;
}

- (void)sendPendingKey
{
    if (_pendingRawCode < 0) {
        return;
    }

    SDL_SendKeyboardKey(_pendingTimestamp, SDL_DEFAULT_KEYBOARD_ID, _pendingRawCode, _pendingScancode, true);
    [self clearPendingKey];
}

- (void)clearPendingKey
{
    _pendingRawCode = -1;
}

@end

static bool IsModifierKeyPressed(unsigned int flags,
                                 unsigned int target_mask,
                                 unsigned int other_mask,
                                 unsigned int either_mask)
{
    bool target_pressed = (flags & target_mask) != 0;
    bool other_pressed = (flags & other_mask) != 0;
    bool either_pressed = (flags & either_mask) != 0;

    if (either_pressed != (target_pressed || other_pressed))
        return either_pressed;

    return target_pressed;
}

static void HandleModifiers(SDL_VideoDevice *_this, SDL_Scancode code, unsigned int modifierFlags)
{
    bool pressed = false;

    if (code == SDL_SCANCODE_LSHIFT) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICELSHIFTKEYMASK,
                                       NX_DEVICERSHIFTKEYMASK, NX_SHIFTMASK);
    } else if (code == SDL_SCANCODE_LCTRL) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICELCTLKEYMASK,
                                       NX_DEVICERCTLKEYMASK, NX_CONTROLMASK);
    } else if (code == SDL_SCANCODE_LALT) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICELALTKEYMASK,
                                       NX_DEVICERALTKEYMASK, NX_ALTERNATEMASK);
    } else if (code == SDL_SCANCODE_LGUI) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICELCMDKEYMASK,
                                       NX_DEVICERCMDKEYMASK, NX_COMMANDMASK);
    } else if (code == SDL_SCANCODE_RSHIFT) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICERSHIFTKEYMASK,
                                       NX_DEVICELSHIFTKEYMASK, NX_SHIFTMASK);
    } else if (code == SDL_SCANCODE_RCTRL) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICERCTLKEYMASK,
                                       NX_DEVICELCTLKEYMASK, NX_CONTROLMASK);
    } else if (code == SDL_SCANCODE_RALT) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICERALTKEYMASK,
                                       NX_DEVICELALTKEYMASK, NX_ALTERNATEMASK);
    } else if (code == SDL_SCANCODE_RGUI) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICERCMDKEYMASK,
                                       NX_DEVICELCMDKEYMASK, NX_COMMANDMASK);
    } else {
        return;
    }

    if (pressed) {
        SDL_SendKeyboardKey(0, SDL_DEFAULT_KEYBOARD_ID, 0, code, true);
    } else {
        SDL_SendKeyboardKey(0, SDL_DEFAULT_KEYBOARD_ID, 0, code, false);
    }
}

static void UpdateKeymap(SDL_CocoaVideoData *data, bool send_event)
{
    TISInputSourceRef key_layout;
    UCKeyboardLayout *keyLayoutPtr = NULL;
    CFDataRef uchrDataRef;

    // See if the keymap needs to be updated
    key_layout = TISCopyCurrentKeyboardLayoutInputSource();
    if (key_layout == data.key_layout) {
        return;
    }
    data.key_layout = key_layout;

    // Try Unicode data first
    uchrDataRef = TISGetInputSourceProperty(key_layout, kTISPropertyUnicodeKeyLayoutData);
    if (uchrDataRef) {
        keyLayoutPtr = (UCKeyboardLayout *)CFDataGetBytePtr(uchrDataRef);
    }

    if (!keyLayoutPtr) {
        CFRelease(key_layout);
        return;
    }

    static struct {
        int flags;
        SDL_Keymod modstate;
    } mods[] = {
        { 0, SDL_KMOD_NONE },
        { shiftKey, SDL_KMOD_SHIFT },
        { alphaLock, SDL_KMOD_CAPS },
        { (shiftKey | alphaLock), (SDL_KMOD_SHIFT | SDL_KMOD_CAPS) },
        { optionKey, SDL_KMOD_ALT },
        { (optionKey | shiftKey), (SDL_KMOD_ALT | SDL_KMOD_SHIFT) },
        { (optionKey | alphaLock), (SDL_KMOD_ALT | SDL_KMOD_CAPS) },
        { (optionKey | shiftKey | alphaLock), (SDL_KMOD_ALT | SDL_KMOD_SHIFT | SDL_KMOD_CAPS) }
    };

    UInt32 keyboard_type = LMGetKbdType();

    SDL_Keymap *keymap = SDL_CreateKeymap(true);
    for (int m = 0; m < SDL_arraysize(mods); ++m) {
        for (int i = 0; i < SDL_arraysize(darwin_scancode_table); i++) {
            OSStatus err;
            UniChar s[8];
            UniCharCount len;
            UInt32 dead_key_state;

            // Make sure this scancode is a valid character scancode
            SDL_Scancode scancode = darwin_scancode_table[i];
            if (scancode == SDL_SCANCODE_UNKNOWN ||
                scancode == SDL_SCANCODE_DELETE ||
                (SDL_GetKeymapKeycode(NULL, scancode, SDL_KMOD_NONE) & SDLK_SCANCODE_MASK)) {
                continue;
            }

            /*
             * Swap the scancode for these two wrongly translated keys
             * UCKeyTranslate() function does not do its job properly for ISO layout keyboards, where the key '@',
             * which is located in the top left corner of the keyboard right under the Escape key, and the additional
             * key '<', which is on the right of the Shift key, are inverted
            */
            if ((scancode == SDL_SCANCODE_NONUSBACKSLASH || scancode == SDL_SCANCODE_GRAVE) && KBGetLayoutType(LMGetKbdType()) == kKeyboardISO) {
                // see comments in scancodes_darwin.h
                scancode = (SDL_Scancode)((SDL_SCANCODE_NONUSBACKSLASH + SDL_SCANCODE_GRAVE) - scancode);
            }

            dead_key_state = 0;
            err = UCKeyTranslate(keyLayoutPtr, i, kUCKeyActionDown,
                                 ((mods[m].flags >> 8) & 0xFF), keyboard_type,
                                 kUCKeyTranslateNoDeadKeysMask,
                                 &dead_key_state, 8, &len, s);
            if (err != noErr) {
                continue;
            }

            if (len > 0 && s[0] != 0x10) {
                SDL_SetKeymapEntry(keymap, scancode, mods[m].modstate, s[0]);
            } else {
                // The default keymap doesn't have any SDL_KMOD_ALT entries, so we don't need to override them
                if (!(mods[m].modstate & SDL_KMOD_ALT)) {
                    SDL_SetKeymapEntry(keymap, scancode, mods[m].modstate, SDLK_UNKNOWN);
                }
            }
        }
    }
    SDL_SetKeymap(keymap, send_event);
}

static void SDLCALL SDL_MacOptionAsAltChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_VideoDevice *_this = (SDL_VideoDevice *)userdata;
    SDL_CocoaVideoData *data = (__bridge SDL_CocoaVideoData *)_this->internal;

    if (hint && *hint) {
        if (SDL_strcmp(hint, "none") == 0) {
            data.option_as_alt = OptionAsAltNone;
        } else if (SDL_strcmp(hint, "only_left") == 0) {
            data.option_as_alt = OptionAsAltOnlyLeft;
        } else if (SDL_strcmp(hint, "only_right") == 0) {
            data.option_as_alt = OptionAsAltOnlyRight;
        } else if (SDL_strcmp(hint, "both") == 0) {
            data.option_as_alt = OptionAsAltBoth;
        }
    } else {
        data.option_as_alt = OptionAsAltNone;
    }
}

void Cocoa_InitKeyboard(SDL_VideoDevice *_this)
{
    SDL_CocoaVideoData *data = (__bridge SDL_CocoaVideoData *)_this->internal;

    UpdateKeymap(data, false);

    // Set our own names for the platform-dependent but layout-independent keys
    // This key is NumLock on the MacBook keyboard. :)
    // SDL_SetScancodeName(SDL_SCANCODE_NUMLOCKCLEAR, "Clear");
    SDL_SetScancodeName(SDL_SCANCODE_LALT, "Left Option");
    SDL_SetScancodeName(SDL_SCANCODE_LGUI, "Left Command");
    SDL_SetScancodeName(SDL_SCANCODE_RALT, "Right Option");
    SDL_SetScancodeName(SDL_SCANCODE_RGUI, "Right Command");

    data.modifierFlags = (unsigned int)[NSEvent modifierFlags];
    SDL_ToggleModState(SDL_KMOD_CAPS, (data.modifierFlags & NSEventModifierFlagCapsLock) ? true : false);

    SDL_AddHintCallback(SDL_HINT_MAC_OPTION_AS_ALT, SDL_MacOptionAsAltChanged, _this);
}

bool Cocoa_StartTextInput(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props)
{
    @autoreleasepool {
        NSView *parentView;
        SDL_CocoaVideoData *data = (__bridge SDL_CocoaVideoData *)_this->internal;
        NSWindow *nswindow = ((__bridge SDL_CocoaWindowData *)window->internal).nswindow;

        parentView = [nswindow contentView];

        /* We only keep one field editor per process, since only the front most
         * window can receive text input events, so it make no sense to keep more
         * than one copy. When we switched to another window and requesting for
         * text input, simply remove the field editor from its superview then add
         * it to the front most window's content view */
        if (!data.fieldEdit) {
            data.fieldEdit = [[SDL3TranslatorResponder alloc] initWithFrame:NSMakeRect(0.0, 0.0, 0.0, 0.0)];
        }

        if (![[data.fieldEdit superview] isEqual:parentView]) {
            // DEBUG_IME(@"add fieldEdit to window contentView");
            [data.fieldEdit removeFromSuperview];
            [parentView addSubview:data.fieldEdit];
            [nswindow makeFirstResponder:data.fieldEdit];
        }
    }
    return Cocoa_UpdateTextInputArea(_this, window);
}

bool Cocoa_StopTextInput(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaVideoData *data = (__bridge SDL_CocoaVideoData *)_this->internal;

        if (data && data.fieldEdit) {
            [data.fieldEdit removeFromSuperview];
            data.fieldEdit = nil;
        }
    }
    return true;
}

bool Cocoa_UpdateTextInputArea(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_CocoaVideoData *data = (__bridge SDL_CocoaVideoData *)_this->internal;
    if (data.fieldEdit) {
        [data.fieldEdit setInputRect:&window->text_input_rect];
    }
    return true;
}

static NSEvent *ReplaceEvent(NSEvent *event, OptionAsAlt option_as_alt)
{
    if (option_as_alt == OptionAsAltNone) {
        return event;
    }

    const unsigned int modflags = (unsigned int)[event modifierFlags];

    bool ignore_alt_characters = false;

    bool lalt_pressed = IsModifierKeyPressed(modflags, NX_DEVICELALTKEYMASK,
                                             NX_DEVICERALTKEYMASK, NX_ALTERNATEMASK);
    bool ralt_pressed = IsModifierKeyPressed(modflags, NX_DEVICERALTKEYMASK,
                                             NX_DEVICELALTKEYMASK, NX_ALTERNATEMASK);

    if (option_as_alt == OptionAsAltOnlyLeft && lalt_pressed) {
        ignore_alt_characters = true;
    } else if (option_as_alt == OptionAsAltOnlyRight && ralt_pressed) {
        ignore_alt_characters = true;
    } else if (option_as_alt == OptionAsAltBoth && (lalt_pressed || ralt_pressed)) {
        ignore_alt_characters = true;
    }

    bool cmd_pressed = modflags & NX_COMMANDMASK;
    bool ctrl_pressed = modflags & NX_CONTROLMASK;

    ignore_alt_characters = ignore_alt_characters && !cmd_pressed && !ctrl_pressed;

    if (ignore_alt_characters) {
        NSString *charactersIgnoringModifiers = [event charactersIgnoringModifiers];
        return [NSEvent keyEventWithType:[event type]
                                location:[event locationInWindow]
                           modifierFlags:modflags
                               timestamp:[event timestamp]
                            windowNumber:[event windowNumber]
                                 context:nil
                              characters:charactersIgnoringModifiers
             charactersIgnoringModifiers:charactersIgnoringModifiers
                               isARepeat:[event isARepeat]
                                 keyCode:[event keyCode]];
    }

    return event;
}

void Cocoa_HandleKeyEvent(SDL_VideoDevice *_this, NSEvent *event)
{
    unsigned short scancode;
    SDL_Scancode code;
    SDL_CocoaVideoData *data = _this ? ((__bridge SDL_CocoaVideoData *)_this->internal) : nil;
    if (!data) {
        return; // can happen when returning from fullscreen Space on shutdown
    }

    if ([event type] == NSEventTypeKeyDown || [event type] == NSEventTypeKeyUp) {
        event = ReplaceEvent(event, data.option_as_alt);
    }

    scancode = [event keyCode];

    if ((scancode == 10 || scancode == 50) && KBGetLayoutType(LMGetKbdType()) == kKeyboardISO) {
        // see comments in scancodes_darwin.h
        scancode = 60 - scancode;
    }

    if (scancode < SDL_arraysize(darwin_scancode_table)) {
        code = darwin_scancode_table[scancode];
    } else {
        // Hmm, does this ever happen?  If so, need to extend the keymap...
        code = SDL_SCANCODE_UNKNOWN;
    }

    switch ([event type]) {
    case NSEventTypeKeyDown:
        if (![event isARepeat]) {
            // See if we need to rebuild the keyboard layout
            UpdateKeymap(data, true);
        }

#ifdef DEBUG_SCANCODES
        if (code == SDL_SCANCODE_UNKNOWN) {
            SDL_Log("The key you just pressed is not recognized by SDL. To help get this fixed, report this to the SDL forums/mailing list <https://discourse.libsdl.org/> or to Christian Walther <cwalther@gmx.ch>. Mac virtual key code is %d.", scancode);
        }
#endif
        if (SDL_TextInputActive(SDL_GetKeyboardFocus())) {
            [data.fieldEdit setPendingKey:scancode scancode:code timestamp:Cocoa_GetEventTimestamp([event timestamp])];
            [data.fieldEdit interpretKeyEvents:[NSArray arrayWithObject:event]];
            [data.fieldEdit sendPendingKey];
        } else if (SDL_GetKeyboardFocus()) {
            SDL_SendKeyboardKey(Cocoa_GetEventTimestamp([event timestamp]), SDL_DEFAULT_KEYBOARD_ID, scancode, code, true);
        }
        break;
    case NSEventTypeKeyUp:
        SDL_SendKeyboardKey(Cocoa_GetEventTimestamp([event timestamp]), SDL_DEFAULT_KEYBOARD_ID, scancode, code, false);
        break;
    case NSEventTypeFlagsChanged: {
        // see if the new modifierFlags mean any existing keys should be pressed/released...
        const unsigned int modflags = (unsigned int)[event modifierFlags];
        HandleModifiers(_this, SDL_SCANCODE_LSHIFT, modflags);
        HandleModifiers(_this, SDL_SCANCODE_LCTRL, modflags);
        HandleModifiers(_this, SDL_SCANCODE_LALT, modflags);
        HandleModifiers(_this, SDL_SCANCODE_LGUI, modflags);
        HandleModifiers(_this, SDL_SCANCODE_RSHIFT, modflags);
        HandleModifiers(_this, SDL_SCANCODE_RCTRL, modflags);
        HandleModifiers(_this, SDL_SCANCODE_RALT, modflags);
        HandleModifiers(_this, SDL_SCANCODE_RGUI, modflags);
        break;
    }
    default: // just to avoid compiler warnings
        break;
    }
}

void Cocoa_QuitKeyboard(SDL_VideoDevice *_this)
{
}

typedef int CGSConnection;
typedef enum
{
    CGSGlobalHotKeyEnable = 0,
    CGSGlobalHotKeyDisable = 1,
} CGSGlobalHotKeyOperatingMode;

extern CGSConnection _CGSDefaultConnection(void);
extern CGError CGSSetGlobalHotKeyOperatingMode(CGSConnection connection, CGSGlobalHotKeyOperatingMode mode);

bool Cocoa_SetWindowKeyboardGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed)
{
#ifdef SDL_MAC_NO_SANDBOX
    CGSSetGlobalHotKeyOperatingMode(_CGSDefaultConnection(), grabbed ? CGSGlobalHotKeyDisable : CGSGlobalHotKeyEnable);
#endif
    return true;
}

#endif // SDL_VIDEO_DRIVER_COCOA

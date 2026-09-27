/**************************************************************************/
/*  godot_keyboard_input_field.mm                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#import "godot_keyboard_input_field.h"

#include "core/os/keyboard.h"
#import "drivers/apple_embedded/display_server_apple_embedded.h"
#import "drivers/apple_embedded/os_apple_embedded.h"

@interface GDTKeyboardInputField () <UITextFieldDelegate>

@property(nonatomic, copy) NSString *previousText;
@property(nonatomic, assign) NSUInteger godotOpenedLength;
@property(nonatomic, assign) BOOL godotExpectResign;

@end

@implementation GDTKeyboardInputField

- (instancetype)initWithFrame:(CGRect)frame {
	self = [super initWithFrame:frame];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (instancetype)initWithCoder:(NSCoder *)coder {
	self = [super initWithCoder:coder];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (void)godot_commonInit {
	self.hidden = YES;
	self.delegate = self;

	[[NSNotificationCenter defaultCenter] addObserver:self
											 selector:@selector(observeTextChange:)
												 name:UITextFieldTextDidChangeNotification
											   object:self];
}

- (void)dealloc {
	self.delegate = nil;
	[[NSNotificationCenter defaultCenter] removeObserver:self];
}

// MARK: Keyboard

- (BOOL)canBecomeFirstResponder {
	return YES;
}

- (BOOL)becomeFirstResponderWithString:(NSString *)existingString cursorStart:(NSInteger)start cursorEnd:(NSInteger)end {
	// The tvOS keyboard owns its cursor (append-only, no caret navigation),
	// so Godot's caret position is intentionally not mirrored: every change
	// below replays the whole text, which converges whatever either side
	// holds. The range arguments only exist to match the input view's shape.
	(void)start;
	(void)end;
	self.text = existingString;
	self.previousText = existingString;
	self.godotOpenedLength = existingString.length;
	return [self becomeFirstResponder];
}

- (BOOL)resignFirstResponder {
	self.text = nil;
	self.previousText = nil;
	self.godotOpenedLength = 0;
	return [super resignFirstResponder];
}

// Hiding from Godot's side: no key back, the LineEdit already unedited (or is
// about to) and an Escape would unedit whatever it is editing instead. Only
// arm when a resign will actually follow: hiding an already-resigned field
// fires no delegate call, and a stuck flag would swallow the next real one.
- (void)godot_hideKeyboard {
	if ([self isFirstResponder]) {
		self.godotExpectResign = YES;
		[self resignFirstResponder];
	}
}

// MARK: UITextFieldDelegate

- (BOOL)textField:(UITextField *)textField shouldChangeCharactersInRange:(NSRange)range replacementString:(NSString *)string {
	if (self.godotMaxLength <= 0) {
		return YES;
	}
	NSInteger newLength = (NSInteger)textField.text.length - (NSInteger)range.length + (NSInteger)string.length;
	return newLength <= self.godotMaxLength;
}

- (BOOL)textFieldShouldReturn:(UITextField *)textField {
	// A field ends editing on return where the view inserts a newline; Godot
	// reads the newline as Enter, so send that and stay up instead.
	[self enterText:@"\n"];
	return NO;
}

- (void)textFieldDidEndEditing:(UITextField *)textField {
	BOOL expected = self.godotExpectResign;
	self.godotExpectResign = NO;
	if (expected) {
		return;
	}
	// Menu-dismissed from the remote: tvOS swallows the press, so Godot never
	// sees it and its field would sit editing with no keyboard up. Escape is
	// ui_cancel, which unedits a LineEdit without submitting.
	DisplayServerAppleEmbedded::get_singleton()->key(Key::ESCAPE, 0, Key::ESCAPE, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
	DisplayServerAppleEmbedded::get_singleton()->key(Key::ESCAPE, 0, Key::ESCAPE, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
}

// MARK: OS Messages

- (void)deleteText:(NSInteger)charactersToDelete {
	for (int i = 0; i < charactersToDelete; i++) {
		DisplayServerAppleEmbedded::get_singleton()->key(Key::BACKSPACE, 0, Key::BACKSPACE, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
		DisplayServerAppleEmbedded::get_singleton()->key(Key::BACKSPACE, 0, Key::BACKSPACE, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
	}
}

- (void)caretToEnd:(NSInteger)overshoot {
	for (int i = 0; i < overshoot; i++) {
		DisplayServerAppleEmbedded::get_singleton()->key(Key::RIGHT, 0, Key::RIGHT, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
		DisplayServerAppleEmbedded::get_singleton()->key(Key::RIGHT, 0, Key::RIGHT, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
	}
}

- (void)enterText:(NSString *)substring {
	String characters = String::utf8([substring UTF8String]);

	for (int i = 0; i < characters.size(); i++) {
		int character = characters[i];
		Key key = Key::NONE;

		if (character == '\t') { // 0x09
			key = Key::TAB;
		} else if (character == '\n') { // 0x0A
			key = Key::ENTER;
		} else if (character == 0x2006) {
			key = Key::SPACE;
		}

		DisplayServerAppleEmbedded::get_singleton()->key(key, character, key, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
		DisplayServerAppleEmbedded::get_singleton()->key(key, character, key, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
	}
}

// MARK: Observer

- (void)observeTextChange:(NSNotification *)notification {
	if (notification.object != self) {
		return;
	}

	// Mirror the whole new text into Godot: caret to the end, clear, retype.
	// Deliberately stateless - the tvOS keyboard owns its cursor and never
	// reports caret moves, so diffing keystroke by keystroke against a
	// recorded cursor diverges (old text merged with new). Overshoots are
	// no-ops past the ends, and the counts cover the opened text, anything
	// typed since, and the field's max length.
	NSString *fullText = self.text ?: @"";
	NSUInteger longest = MAX(self.previousText.length, self.godotOpenedLength);
	if (self.godotMaxLength > 0) {
		longest = MAX(longest, (NSUInteger)self.godotMaxLength);
	}
	NSInteger overshoot = longest + 64;
	[self caretToEnd:overshoot];
	[self deleteText:overshoot];
	[self enterText:fullText];

	self.previousText = fullText;
}

@end

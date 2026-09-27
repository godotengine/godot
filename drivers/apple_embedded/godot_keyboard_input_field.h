/**************************************************************************/
/*  godot_keyboard_input_field.h                                          */
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

#pragma once

#import <UIKit/UIKit.h>

// tvOS counterpart of GDTKeyboardInputView. tvOS does not summon its keyboard
// for a UITextView - a plain one refuses first responder outright - while a
// UITextField brings it up, hidden or not. Same interface, same key synthesis.
@interface GDTKeyboardInputField : UITextField

- (BOOL)becomeFirstResponderWithString:(NSString *)existingString cursorStart:(NSInteger)start cursorEnd:(NSInteger)end;
- (void)godot_hideKeyboard;

// Mirrors the LineEdit's max length (<= 0 is unlimited). Without it the
// field can outgrow the LineEdit, which rejects the overflow, and the two
// sides diverge - every later keystroke diffs against the wrong text.
@property(nonatomic, assign) NSInteger godotMaxLength;

@end

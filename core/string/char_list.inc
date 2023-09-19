/**************************************************************************/
/*  char_list.inc                                                         */
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

#ifndef CHAR_LIST_INC
#define CHAR_LIST_INC

#define CHAR_NULL '\0'
#define CHAR_START_OF_HEADING 0x0001
#define CHAR_START_OF_TEXT 0x0002
#define CHAR_END_OF_TEXT 0x0003
#define CHAR_END_OF_TRANSMISSION 0x0004
#define CHAR_ENQUIRY 0x0005
#define CHAR_ACKNOWLEDGE 0x0006
#define CHAR_BELL '\a'
#define CHAR_BACKSPACE '\b'
#define CHAR_HORIZONTAL_TAB '\t'
#define CHAR_NEWLINE '\n'
#define CHAR_VERTICAL_TAB '\v'
#define CHAR_FORM_FEED '\f'
#define CHAR_CARRIAGE_RETURN '\r'
#define CHAR_SHIFT_OUT 0x000E
#define CHAR_SHIFT_IN 0x000F
#define CHAR_DATA_LINK_ESCAPE 0x0010
#define CHAR_DEVICE_CONTROL_ONE 0x0011
#define CHAR_DEVICE_CONTROL_TWO 0x0012
#define CHAR_DEVICE_CONTROL_THREE 0x0013
#define CHAR_DEVICE_CONTROL_FOUR 0x0014
#define CHAR_NEGATIVE_ACKNOWLEDGE 0x0015
#define CHAR_SYNCHRONOUS_IDLE 0x0016
#define CHAR_END_OF_TRANSMISSION_BLOCK 0x0017
#define CHAR_CANCEL 0x0018
#define CHAR_END_OF_MEDIUM 0x0019
#define CHAR_SUBSTITUTE 0x001A
#define CHAR_ESCAPE 0x001B
#define CHAR_FILE_SEPARATOR 0x001C
#define CHAR_GROUP_SEPARATOR 0x001D
#define CHAR_RECORD_SEPARATOR 0x001E
#define CHAR_UNIT_SEPARATOR 0x001F
#define CHAR_DELETE 0x007F

// Latin-1 Supplement
#define CHAR_NEXT_LINE 0x0085
#define CHAR_APPLICATION_PROGRAM_COMMAND 0x009F
#define CHAR_NO_BREAK_SPACE 0x00a0

// Ogham
#define CHAR_OGHAM_SPACE_MARK 0x1680

// General Punctuation
#define CHAR_EN_QUAD 0x2000
#define CHAR_HAIR_SPACE 0x200A
#define CHAR_LEFT_TO_RIGHT_MARK 0x200E
#define CHAR_RIGHT_TO_LEFT_MARK 0x200F
#define CHAR_LINE_SEPARATOR 0x2028
#define CHAR_PARAGRAPH_SEPARATOR 0x2029
#define CHAR_LEFT_TO_RIGHT_EMBEDDING 0x202A
#define CHAR_RIGHT_TO_LEFT_EMBEDDING 0x202B
#define CHAR_POP_DIRECTIONAL_FORMATING 0x202C
#define CHAR_LEFT_TO_RIGHT_OVERRIDE 0x202D
#define CHAR_RIGHT_TO_LEFT_OVERRIDE 0x202E
#define CHAR_NARROW_NO_BREAK_SPACE 0x202F
#define CHAR_MEDIUM_MATHEMATICAL_SPACE 0x205F
#define CHAR_LEFT_TO_RIGHT_ISOLATE 0x2066
#define CHAR_RIGHT_TO_LEFT_ISOLATE 0x2069
#define CHAR_NOMINAL_DIGIT_SHAPES 0x206F

// Symbols and Punctuation, CJK
#define CHAR_IDEOGRAPHIC_SPACE 0x3000
#define CHAR_IDEOGRAPHIC_HALF_FILL_SPACE 0x303f

#endif // CHAR_LIST_INC

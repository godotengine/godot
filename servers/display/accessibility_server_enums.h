/**************************************************************************/
/*  accessibility_server_enums.h                                          */
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

namespace AccessibilityServerEnums {

enum AccessibilityMode {
	ACCESSIBILITY_AUTO,
	ACCESSIBILITY_ALWAYS,
	ACCESSIBILITY_DISABLED,
};

enum AccessibilityRole {
	ROLE_UNKNOWN,
	ROLE_DEFAULT_BUTTON,
	ROLE_AUDIO,
	ROLE_VIDEO,
	ROLE_STATIC_TEXT,
	ROLE_CONTAINER,
	ROLE_PANEL,
	ROLE_BUTTON,
	ROLE_LINK,
	ROLE_CHECK_BOX,
	ROLE_RADIO_BUTTON,
	ROLE_CHECK_BUTTON,
	ROLE_SCROLL_BAR,
	ROLE_SCROLL_VIEW,
	ROLE_SPLITTER,
	ROLE_SLIDER,
	ROLE_SPIN_BUTTON,
	ROLE_PROGRESS_INDICATOR,
	ROLE_TEXT_FIELD,
	ROLE_MULTILINE_TEXT_FIELD,
	ROLE_COLOR_PICKER,
	ROLE_TABLE,
	ROLE_CELL,
	ROLE_ROW,
	ROLE_ROW_GROUP,
	ROLE_ROW_HEADER,
	ROLE_COLUMN_HEADER,
	ROLE_TREE,
	ROLE_TREE_ITEM,
	ROLE_LIST,
	ROLE_LIST_ITEM,
	ROLE_LIST_BOX,
	ROLE_LIST_BOX_OPTION,
	ROLE_TAB_BAR,
	ROLE_TAB,
	ROLE_TAB_PANEL,
	ROLE_MENU_BAR,
	ROLE_MENU,
	ROLE_MENU_ITEM,
	ROLE_MENU_ITEM_CHECK_BOX,
	ROLE_MENU_ITEM_RADIO,
	ROLE_IMAGE,
	ROLE_WINDOW,
	ROLE_TITLE_BAR,
	ROLE_DIALOG,
	ROLE_TOOLTIP,
	ROLE_REGION,
	ROLE_TEXT_RUN,
};

enum AccessibilityPopupType {
	POPUP_MENU,
	POPUP_LIST,
	POPUP_TREE,
	POPUP_DIALOG,
};

enum AccessibilityFlags {
	FLAG_HIDDEN,
	FLAG_MULTISELECTABLE,
	FLAG_REQUIRED,
	FLAG_VISITED,
	FLAG_BUSY,
	FLAG_MODAL,
	FLAG_TOUCH_PASSTHROUGH,
	FLAG_READONLY,
	FLAG_DISABLED,
	FLAG_CLIPS_CHILDREN,
};

enum AccessibilityAction {
	ACTION_CLICK,
	ACTION_FOCUS,
	ACTION_BLUR,
	ACTION_COLLAPSE,
	ACTION_EXPAND,
	ACTION_DECREMENT,
	ACTION_INCREMENT,
	ACTION_HIDE_TOOLTIP,
	ACTION_SHOW_TOOLTIP,
	ACTION_SET_TEXT_SELECTION,
	ACTION_REPLACE_SELECTED_TEXT,
	ACTION_SCROLL_BACKWARD,
	ACTION_SCROLL_DOWN,
	ACTION_SCROLL_FORWARD,
	ACTION_SCROLL_LEFT,
	ACTION_SCROLL_RIGHT,
	ACTION_SCROLL_UP,
	ACTION_SCROLL_INTO_VIEW,
	ACTION_SCROLL_TO_POINT,
	ACTION_SET_SCROLL_OFFSET,
	ACTION_SET_VALUE,
	ACTION_SHOW_CONTEXT_MENU,
	ACTION_CUSTOM,
};

enum AccessibilityLiveMode {
	LIVE_OFF,
	LIVE_POLITE,
	LIVE_ASSERTIVE,
};

enum AccessibilityScrollUnit {
	SCROLL_UNIT_ITEM,
	SCROLL_UNIT_PAGE,
};

enum AccessibilityScrollHint {
	SCROLL_HINT_TOP_LEFT,
	SCROLL_HINT_BOTTOM_RIGHT,
	SCROLL_HINT_TOP_EDGE,
	SCROLL_HINT_BOTTOM_EDGE,
	SCROLL_HINT_LEFT_EDGE,
	SCROLL_HINT_RIGHT_EDGE,
};

}; // namespace AccessibilityServerEnums

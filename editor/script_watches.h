/*************************************************************************/
/*  script_watches.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef SCRIPT_WATCHES_H
#define SCRIPT_WATCHES_H

#include "scene/gui/box_container.h"
#include "scene/gui/scroll_container.h"

class VBoxContainer;
class HBoxContainer;
class LineEdit;
class RichTextLabel;
class CheckBox;
class ToolButton;

class ScriptWatch;
class ScriptEditorDebugger;

class ScriptWatches : public ScrollContainer {

	GDCLASS(ScriptWatches, ScrollContainer)
private:
	VBoxContainer *main_vbox;

	Vector<ScriptWatch *> watches;

	void _watch_changed(Object *p_watch);
	void _lock_changed(Object *p_watch, bool p_locked);
	void _tracking_changed(Object *p_watch, bool p_tracking);
	void _watch_removed(Object *p_watch);
	void _lost_focus(Object *p_watch);
	void _breaked(bool p_breaked, bool p_can_debug);

	ScriptWatch *_get_watch_ghost() const;
	void _create_watch_ghost();
	void _remove_watch(ScriptWatch *p_watch);
	void _update_watch_ids(int p_from_index = 0);

protected:
	static void _bind_methods();

public:
	void add_watch(const String &p_expression);
	void update_watch_result(int p_index, const String &result, bool success);
	void remove_watch(int p_index);

	int get_watch_count() const;
	String get_watch_expression(int p_index);

	void disable(bool p_disable);

	ScriptWatches(ScriptEditorDebugger *p_debugger);
};

#endif // SCRIPT_WATCHES_H

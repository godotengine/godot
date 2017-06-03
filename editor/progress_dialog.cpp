/*************************************************************************/
/*  progress_dialog.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "progress_dialog.h"

#include "editor_scale.h"
#include "main/main.h"
#include "message_queue.h"
#include "os/os.h"

void BackgroundProgress::_add_task(const String &p_task, const String &p_label, int p_steps) {

	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(tasks.has(p_task));
	Task t;
	t.hb = memnew(HBoxContainer);
	Label *l = memnew(Label);
	l->set_text(p_label + " ");
	t.hb->add_child(l);
	t.progress = memnew(ProgressBar);
	t.progress->set_max(p_steps);
	t.progress->set_value(p_steps);
	Control *ec = memnew(Control);
	ec->set_h_size_flags(SIZE_EXPAND_FILL);
	ec->set_v_size_flags(SIZE_EXPAND_FILL);
	t.progress->set_area_as_parent_rect();
	ec->add_child(t.progress);
	ec->set_custom_minimum_size(Size2(80, 5) * EDSCALE);
	t.hb->add_child(ec);

	add_child(t.hb);

	tasks[p_task] = t;
}

void BackgroundProgress::_update() {

	_THREAD_SAFE_METHOD_

	for (Map<String, int>::Element *E = updates.front(); E; E = E->next()) {

		if (tasks.has(E->key())) {
			_task_step(E->key(), E->get());
		}
	}

	updates.clear();
}

void BackgroundProgress::_task_step(const String &p_task, int p_step) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!tasks.has(p_task));

	Task &t = tasks[p_task];
	if (p_step < 0)
		t.progress->set_value(t.progress->get_value() + 1);
	else
		t.progress->set_value(p_step);
}
void BackgroundProgress::_end_task(const String &p_task) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!tasks.has(p_task));
	Task &t = tasks[p_task];

	memdelete(t.hb);
	tasks.erase(p_task);
}

void BackgroundProgress::_bind_methods() {

	ClassDB::bind_method("_add_task", &BackgroundProgress::_add_task);
	ClassDB::bind_method("_task_step", &BackgroundProgress::_task_step);
	ClassDB::bind_method("_end_task", &BackgroundProgress::_end_task);
	ClassDB::bind_method("_update", &BackgroundProgress::_update);
}

void BackgroundProgress::add_task(const String &p_task, const String &p_label, int p_steps) {

	MessageQueue::get_singleton()->push_call(this, "_add_task", p_task, p_label, p_steps);
}
void BackgroundProgress::task_step(const String &p_task, int p_step) {

	//this code is weird, but it prevents deadlock.
	bool no_updates;
	{
		_THREAD_SAFE_METHOD_
		no_updates = updates.empty();
	}

	if (no_updates)
		MessageQueue::get_singleton()->push_call(this, "_update");

	{
		_THREAD_SAFE_METHOD_
		updates[p_task] = p_step;
	}
}

void BackgroundProgress::end_task(const String &p_task) {

	MessageQueue::get_singleton()->push_call(this, "_end_task", p_task);
}

////////////////////////////////////////////////

ProgressDialog *ProgressDialog::singleton = NULL;

void ProgressDialog::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_DRAW: {

			Ref<StyleBox> style = get_stylebox("panel", "PopupMenu");
			draw_style_box(style, Rect2(Point2(), get_size()));

		} break;
	}
}

void ProgressDialog::_popup() {

	Size2 ms = main->get_combined_minimum_size();
	ms.width = MAX(500 * EDSCALE, ms.width);

	Ref<StyleBox> style = get_stylebox("panel", "PopupMenu");
	ms += style->get_minimum_size();
	for (int i = 0; i < 4; i++) {
		main->set_margin(Margin(i), style->get_margin(Margin(i)));
	}

	popup_centered(ms);
}

void ProgressDialog::add_task(const String &p_task, const String &p_label, int p_steps) {

	ERR_FAIL_COND(tasks.has(p_task));
	Task t;
	t.vb = memnew(VBoxContainer);
	VBoxContainer *vb2 = memnew(VBoxContainer);
	t.vb->add_margin_child(p_label, vb2);
	t.progress = memnew(ProgressBar);
	t.progress->set_max(p_steps);
	t.progress->set_value(p_steps);
	vb2->add_child(t.progress);
	t.state = memnew(Label);
	t.state->set_clip_text(true);
	vb2->add_child(t.state);
	main->add_child(t.vb);

	tasks[p_task] = t;
	_popup();
}

void ProgressDialog::task_step(const String &p_task, const String &p_state, int p_step, bool p_force_redraw) {

	ERR_FAIL_COND(!tasks.has(p_task));

	if (!p_force_redraw) {
		uint64_t tus = OS::get_singleton()->get_ticks_usec();
		if (tus - last_progress_tick < 50000) //50ms
			return;
	}

	Task &t = tasks[p_task];
	if (p_step < 0)
		t.progress->set_value(t.progress->get_value() + 1);
	else
		t.progress->set_value(p_step);

	t.state->set_text(p_state);
	last_progress_tick = OS::get_singleton()->get_ticks_usec();
	Main::iteration(); // this will not work on a lot of platforms, so it's only meant for the editor
}

void ProgressDialog::end_task(const String &p_task) {

	ERR_FAIL_COND(!tasks.has(p_task));
	Task &t = tasks[p_task];

	memdelete(t.vb);
	tasks.erase(p_task);

	if (tasks.empty())
		hide();
	else
		_popup();
}

ProgressDialog::ProgressDialog() {

	main = memnew(VBoxContainer);
	add_child(main);
	main->set_area_as_parent_rect();
	set_exclusive(true);
	last_progress_tick = 0;
	singleton = this;
}

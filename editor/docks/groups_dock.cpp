/**************************************************************************/
/*  groups_dock.cpp                                                       */
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

#include "groups_dock.h"

#include "editor/settings/editor_command_palette.h"

void GroupsDock::set_selection(const Vector<Node *> &p_nodes) {
	groups->set_selection(p_nodes);
}

GroupsDock::GroupsDock() {
	singleton = this;
	set_name(TTRC("Groups"));
	set_icon_name("Groups");
	set_dock_shortcut(ED_SHORTCUT_AND_COMMAND("docks/open_groups", TTRC("Open Groups Dock")));
	set_default_slot(DockConstants::DOCK_SLOT_RIGHT_UL);

	groups = memnew(GroupsEditor);
	groups->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(groups);
}

GroupsDock::~GroupsDock() {
	singleton = nullptr;
}

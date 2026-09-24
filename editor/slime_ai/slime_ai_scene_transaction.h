/**************************************************************************/
/*  slime_ai_scene_transaction.h                                        */
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

#include "editor/slime_ai/slime_ai_journal.h"
#include "core/object/undo_redo.h"

class Node;

namespace SlimeAI {

class SceneTransaction {
	public:
	enum ApprovalMode { MANUAL, PROTECTED, FREEDOM };
	private:
	Journal journal;
	UndoRedo test_undo_redo;
	Dictionary preview_data;
	String preview_id;
	bool granted = false;
	bool inject_failure_after_effect = false;
	uint64_t sequence = 0;
	ApprovalMode approval_mode = MANUAL;
	String approved_scene_scope;

public:
	explicit SceneTransaction(const String &p_journal_path);
	Dictionary set_mode(ApprovalMode p_mode, Node *p_selected_root);
	Dictionary preview(Node *p_root, const String &p_operation_id, const Dictionary &p_proposal, bool p_editor_unsaved);
	Dictionary grant(const String &p_preview_id);
	Dictionary cancel(const String &p_preview_id);
	Dictionary apply(Node *p_root, const String &p_preview_id, bool p_editor_unsaved);
	Dictionary status(Node *p_root, const String &p_operation_id) const;
	Dictionary resolve(Node *p_root, const String &p_operation_id, const String &p_expected_revision, const String &p_observed_effect, bool p_editor_unsaved);
	Array recorded_operation_ids() const { return journal.operation_ids(); }
	Dictionary undo(Node *p_root);
	Dictionary redo(Node *p_root);
	void set_inject_failure_after_effect(bool p_enabled) { inject_failure_after_effect = p_enabled; }
	String unresolved_operation(Node *p_root, const String &p_except_id = String()) const;
};

} // namespace SlimeAI

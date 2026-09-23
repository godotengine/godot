/**************************************************************************/
/*  slime_ai_scene_command_executor.h                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietzky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/object/undo_redo.h"
#include "core/variant/dictionary.h"
#include "scene/main/node.h"

namespace SlimeAI {

bool execute_scene_operation(Node *p_root, const Dictionary &p_operation, const String &p_operation_id, UndoRedo *p_test_undo, Node *&r_created);

} // namespace SlimeAI

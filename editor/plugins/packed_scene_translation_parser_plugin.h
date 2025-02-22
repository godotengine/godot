/**************************************************************************/
/*  packed_scene_translation_parser_plugin.h                              */
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

#ifndef PACKED_SCENE_TRANSLATION_PARSER_PLUGIN_H
#define PACKED_SCENE_TRANSLATION_PARSER_PLUGIN_H

#include "editor/editor_translation_parser.h"

class PackedSceneEditorTranslationParserPlugin : public EditorTranslationParserPlugin {
	GDCLASS(PackedSceneEditorTranslationParserPlugin, EditorTranslationParserPlugin);

	// Scene Node's properties that contain translation strings.
	HashSet<String> lookup_properties;
	// Properties from specific Nodes that should be ignored.
	HashMap<String, Vector<String>> exception_list;

public:
	virtual Error parse_file(const String &p_path, Vector<String> *r_ids, Vector<Vector<String>> *r_ids_ctx_plural) override;
	bool match_property(const String &p_property_name, const String &p_node_type);
	virtual void get_recognized_extensions(List<String> *r_extensions) const override;

	PackedSceneEditorTranslationParserPlugin();
};

#endif // PACKED_SCENE_TRANSLATION_PARSER_PLUGIN_H

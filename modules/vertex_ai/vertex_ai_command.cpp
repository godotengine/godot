/**************************************************************************/
/*  vertex_ai_command.cpp                                                */
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

#include "vertex_ai_command.h"

#include "core/object/class_db.h"

Dictionary VertexAICommand::to_dictionary() const {
	Dictionary d;
	d["name"] = name;
	d["description"] = description;
	d["is_destructive"] = is_destructive;
	return d;
}

void VertexAICommand::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_name", "name"), &VertexAICommand::set_name);
	ClassDB::bind_method(D_METHOD("get_name"), &VertexAICommand::get_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "name"), "set_name", "get_name");

	ClassDB::bind_method(D_METHOD("set_description", "description"), &VertexAICommand::set_description);
	ClassDB::bind_method(D_METHOD("get_description"), &VertexAICommand::get_description);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description"), "set_description", "get_description");

	ClassDB::bind_method(D_METHOD("set_is_destructive", "destructive"), &VertexAICommand::set_is_destructive);
	ClassDB::bind_method(D_METHOD("get_is_destructive"), &VertexAICommand::get_is_destructive);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_destructive"), "set_is_destructive", "get_is_destructive");

	ClassDB::bind_method(D_METHOD("set_handler", "handler"), &VertexAICommand::set_handler);
	ClassDB::bind_method(D_METHOD("get_handler"), &VertexAICommand::get_handler);

	ClassDB::bind_method(D_METHOD("to_dictionary"), &VertexAICommand::to_dictionary);
}

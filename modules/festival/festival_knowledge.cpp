/**************************************************************************/
/*  festival_knowledge.cpp                                                */
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

#include "festival_knowledge.h"

#include "core/object/class_db.h"

void FestivalKnowledge::set_id(const StringName &p_id) { id = p_id; }
StringName FestivalKnowledge::get_id() const { return id; }

void FestivalKnowledge::set_category(Category p_category) { category = p_category; }
FestivalKnowledge::Category FestivalKnowledge::get_category() const { return category; }

void FestivalKnowledge::set_subject(const StringName &p_subject) { subject = p_subject; }
StringName FestivalKnowledge::get_subject() const { return subject; }

void FestivalKnowledge::set_title(const String &p_title) { title = p_title; }
String FestivalKnowledge::get_title() const { return title; }

void FestivalKnowledge::set_body(const String &p_body) { body = p_body; }
String FestivalKnowledge::get_body() const { return body; }

void FestivalKnowledge::set_veracity(bool p_veracity) { veracity = p_veracity; }
bool FestivalKnowledge::get_veracity() const { return veracity; }

void FestivalKnowledge::set_census_data(const Dictionary &p_census_data) { census_data = p_census_data; }
Dictionary FestivalKnowledge::get_census_data() const { return census_data; }

void FestivalKnowledge::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_id", "id"), &FestivalKnowledge::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &FestivalKnowledge::get_id);
	ClassDB::bind_method(D_METHOD("set_category", "category"), &FestivalKnowledge::set_category);
	ClassDB::bind_method(D_METHOD("get_category"), &FestivalKnowledge::get_category);
	ClassDB::bind_method(D_METHOD("set_subject", "subject"), &FestivalKnowledge::set_subject);
	ClassDB::bind_method(D_METHOD("get_subject"), &FestivalKnowledge::get_subject);
	ClassDB::bind_method(D_METHOD("set_title", "title"), &FestivalKnowledge::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &FestivalKnowledge::get_title);
	ClassDB::bind_method(D_METHOD("set_body", "body"), &FestivalKnowledge::set_body);
	ClassDB::bind_method(D_METHOD("get_body"), &FestivalKnowledge::get_body);
	ClassDB::bind_method(D_METHOD("set_veracity", "veracity"), &FestivalKnowledge::set_veracity);
	ClassDB::bind_method(D_METHOD("get_veracity"), &FestivalKnowledge::get_veracity);
	ClassDB::bind_method(D_METHOD("set_census_data", "census_data"), &FestivalKnowledge::set_census_data);
	ClassDB::bind_method(D_METHOD("get_census_data"), &FestivalKnowledge::get_census_data);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "category", PROPERTY_HINT_ENUM, "Secret,Dark Secret,Rumor,False Rumor,Password,Schedule,Route,Fact"), "set_category", "get_category");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "subject"), "set_subject", "get_subject");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "body", PROPERTY_HINT_MULTILINE_TEXT), "set_body", "get_body");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "veracity"), "set_veracity", "get_veracity");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "census_data"), "set_census_data", "get_census_data");

	BIND_ENUM_CONSTANT(CATEGORY_SECRET);
	BIND_ENUM_CONSTANT(CATEGORY_DARK_SECRET);
	BIND_ENUM_CONSTANT(CATEGORY_RUMOR);
	BIND_ENUM_CONSTANT(CATEGORY_FALSE_RUMOR);
	BIND_ENUM_CONSTANT(CATEGORY_PASSWORD);
	BIND_ENUM_CONSTANT(CATEGORY_SCHEDULE);
	BIND_ENUM_CONSTANT(CATEGORY_ROUTE);
	BIND_ENUM_CONSTANT(CATEGORY_FACT);
}

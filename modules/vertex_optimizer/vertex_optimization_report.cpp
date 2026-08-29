/**************************************************************************/
/*  vertex_optimization_report.cpp                                       */
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

#include "vertex_optimization_report.h"

#include "core/object/class_db.h"

void VertexOptimizationReport::add_recommendation(Severity p_severity, const String &p_message, const String &p_action, const String &p_target) {
	Dictionary r;
	r["severity"] = p_severity;
	r["message"] = p_message;
	r["action"] = p_action;
	r["target"] = p_target;
	recommendations.push_back(r);
}

Dictionary VertexOptimizationReport::get_recommendation(int p_index) const {
	if (p_index < 0 || p_index >= recommendations.size()) {
		return Dictionary();
	}
	return recommendations[p_index];
}

int VertexOptimizationReport::get_critical_count() const {
	int c = 0;
	for (const Dictionary &r : recommendations) {
		if ((int)r["severity"] == SEVERITY_CRITICAL) {
			c++;
		}
	}
	return c;
}

int VertexOptimizationReport::get_warning_count() const {
	int c = 0;
	for (const Dictionary &r : recommendations) {
		if ((int)r["severity"] == SEVERITY_WARNING) {
			c++;
		}
	}
	return c;
}

Dictionary VertexOptimizationReport::to_dictionary() const {
	Dictionary d;
	d["metrics"] = metrics;
	Array recs;
	for (const Dictionary &r : recommendations) {
		recs.push_back(r);
	}
	d["recommendations"] = recs;
	d["critical_count"] = get_critical_count();
	d["warning_count"] = get_warning_count();
	return d;
}

void VertexOptimizationReport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_metrics", "metrics"), &VertexOptimizationReport::set_metrics);
	ClassDB::bind_method(D_METHOD("get_metrics"), &VertexOptimizationReport::get_metrics);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "metrics"), "set_metrics", "get_metrics");

	ClassDB::bind_method(D_METHOD("add_recommendation", "severity", "message", "action", "target"), &VertexOptimizationReport::add_recommendation, DEFVAL(String()), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("get_recommendation_count"), &VertexOptimizationReport::get_recommendation_count);
	ClassDB::bind_method(D_METHOD("get_recommendation", "index"), &VertexOptimizationReport::get_recommendation);
	ClassDB::bind_method(D_METHOD("get_recommendations"), &VertexOptimizationReport::get_recommendations);
	ClassDB::bind_method(D_METHOD("get_critical_count"), &VertexOptimizationReport::get_critical_count);
	ClassDB::bind_method(D_METHOD("get_warning_count"), &VertexOptimizationReport::get_warning_count);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &VertexOptimizationReport::to_dictionary);

	BIND_ENUM_CONSTANT(SEVERITY_INFO);
	BIND_ENUM_CONSTANT(SEVERITY_WARNING);
	BIND_ENUM_CONSTANT(SEVERITY_CRITICAL);
}

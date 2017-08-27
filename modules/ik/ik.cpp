/*************************************************************************/
/*  ik.cpp                                                               */
/* Copyright (c) 2016 Sergey Lapin <slapinid@gmail.com>                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "ik.h"

bool InverseKinematics::_get(const StringName &p_name, Variant &r_ret) const {

	if (String(p_name) == "ik_bone") {

		r_ret = get_bone_name();
		return true;
	}

	return false;
}

bool InverseKinematics::_set(const StringName &p_name, const Variant &p_value) {

	if (String(p_name) == "ik_bone") {

		set_bone_name(p_value);
		changed = true;
		return true;
	}

	return false;
}

void InverseKinematics::_get_property_list(List<PropertyInfo> *p_list) const {

	Skeleton *parent = NULL;
	if (get_parent())
		parent = get_parent()->cast_to<Skeleton>();

	if (parent) {

		String names;
		for (int i = 0; i < parent->get_bone_count(); i++) {
			if (i > 0)
				names += ",";
			names += parent->get_bone_name(i);
		}

		p_list->push_back(PropertyInfo(Variant::STRING, "ik_bone", PROPERTY_HINT_ENUM, names));
	} else {

		p_list->push_back(PropertyInfo(Variant::STRING, "ik_bone"));
	}
}

void InverseKinematics::_check_bind() {

	if (get_parent() && get_parent()->cast_to<Skeleton>()) {
		Skeleton *sk = get_parent()->cast_to<Skeleton>();
		int idx = sk->find_bone(ik_bone);
		if (idx != -1) {
			ik_bone_no = idx;
			bound = true;
		}
		skel = sk;
	}
}

void InverseKinematics::_check_unbind() {

	if (bound) {

		if (get_parent() && get_parent()->cast_to<Skeleton>()) {
			Skeleton *sk = get_parent()->cast_to<Skeleton>();
			int idx = sk->find_bone(ik_bone);
			if (idx != -1)
				ik_bone_no = idx;
			else
				ik_bone_no = 0;
			skel = sk;
		}
		bound = false;
	}
}

void InverseKinematics::set_bone_name(const String &p_name) {

	if (is_inside_tree())
		_check_unbind();

	ik_bone = p_name;

	if (is_inside_tree())
		_check_bind();
	changed = true;
}

String InverseKinematics::get_bone_name() const {

	return ik_bone;
}

void InverseKinematics::set_iterations(int itn) {

	if (is_inside_tree())
		_check_unbind();

	iterations = itn;

	if (is_inside_tree())
		_check_bind();
	changed = true;
}

int InverseKinematics::get_iterations() const {

	return iterations;
}

void InverseKinematics::set_chain_size(int cs) {
	if (is_inside_tree())
		_check_unbind();

	chain_size = cs;
	chain.clear();
	if (bound)
		update_parameters();

	if (is_inside_tree())
		_check_bind();
	changed = true;
}

int InverseKinematics::get_chain_size() const {

	return chain_size;
}

void InverseKinematics::set_precision(float p) {

	if (is_inside_tree())
		_check_unbind();

	precision = p;

	if (is_inside_tree())
		_check_bind();
	changed = true;
}

float InverseKinematics::get_precision() const {

	return precision;
}

void InverseKinematics::set_speed(float p) {

	if (is_inside_tree())
		_check_unbind();

	speed = p;

	if (is_inside_tree())
		_check_bind();
	changed = true;
}

float InverseKinematics::get_speed() const {

	return speed;
}

void InverseKinematics::update_parameters() {
	tail_bone = -1;
	for (int i = 0; i < skel->get_bone_count(); i++)
		if (skel->get_bone_parent(i) == ik_bone_no)
			tail_bone = i;
	int cur_bone = ik_bone_no;
	int its = chain_size;
	while (its > 0 && cur_bone >= 0) {
		chain.push_back(cur_bone);
		cur_bone = skel->get_bone_parent(cur_bone);
		its--;
	}
}

void InverseKinematics::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			_check_bind();
			if (bound) {
				update_parameters();
				changed = false;
				set_process(true);
			}
		} break;
		case NOTIFICATION_PROCESS: {

			Spatial *sksp = skel->cast_to<Spatial>();
			if (!bound)
				break;
			if (!sksp)
				break;
			if (changed) {
				update_parameters();
				changed = false;
			}
			Vector3 to = get_translation();
			for (int hump = 0; hump < iterations; hump++) {
				int depth = 0;
				float olderr = 1000.0;
				float psign = 1.0;
				bool reached = false;

				for (List<int>::Element *b = chain.front(); b; b = b->next()) {
					int cur_bone = b->get();
					Vector3 d = skel->get_bone_global_pose(tail_bone).origin;
					Vector3 rg = to;
					float err = d.distance_squared_to(rg);
					if (err < precision) {
						if (!reached && err < precision)
							reached = true;
						break;
					} else if (reached)
						reached = false;
					if (err > olderr)
						psign = -psign;
					Transform mod = skel->get_bone_global_pose(cur_bone);
					Quat q1 = Quat(mod.basis).normalized();
					Transform mod2 = mod.looking_at(to, Vector3(0.0, 1.0, 0.0));
					Quat q2 = Quat(mod2.basis).normalized();
					if (psign < 0.0)
						q2 = q2.inverse();
					Quat q = q1.slerp(q2, speed / (1.0 + 500.0 * depth)).normalized();
					Transform fin = Transform(q);
					fin.origin = mod.origin;
					skel->set_bone_global_pose(cur_bone, fin);
					depth++;
				}
				if (reached)
					break;
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			set_process(false);

			_check_unbind();
		} break;
	}
}
void InverseKinematics::_bind_methods() {
	ObjectTypeDB::bind_method(_MD("set_bone_name", "ik_bone"), &InverseKinematics::set_bone_name);
	ObjectTypeDB::bind_method(_MD("get_bone_name"), &InverseKinematics::get_bone_name);
	ObjectTypeDB::bind_method(_MD("set_iterations", "iterations"), &InverseKinematics::set_iterations);
	ObjectTypeDB::bind_method(_MD("get_iterations"), &InverseKinematics::get_iterations);
	ObjectTypeDB::bind_method(_MD("set_chain_size", "chain_size"), &InverseKinematics::set_chain_size);
	ObjectTypeDB::bind_method(_MD("get_chain_size"), &InverseKinematics::get_chain_size);
	ObjectTypeDB::bind_method(_MD("set_precision", "precision"), &InverseKinematics::set_precision);
	ObjectTypeDB::bind_method(_MD("get_precision"), &InverseKinematics::get_precision);
	ObjectTypeDB::bind_method(_MD("set_speed", "speed"), &InverseKinematics::set_speed);
	ObjectTypeDB::bind_method(_MD("get_speed"), &InverseKinematics::get_speed);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "iterations"), _SCS("set_iterations"), _SCS("get_iterations"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "chain_size"), _SCS("set_chain_size"), _SCS("get_chain_size"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "precision"), _SCS("set_precision"), _SCS("get_precision"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "speed"), _SCS("set_speed"), _SCS("get_speed"));
}

InverseKinematics::InverseKinematics() {
	bound = false;
	chain_size = 2;
	iterations = 100;
	precision = 0.001;
	speed = 0.2;
}

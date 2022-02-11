/*************************************************************************/
/*  occluder.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "occluder.h"

#include "core/engine.h"
#include "servers/visual/portals/portal_occlusion_culler.h"

void Occluder::resource_changed(RES res) {
	update_gizmo();
}

void Occluder::set_shape(const Ref<OccluderShape> &p_shape) {
	if (p_shape == _shape) {
		return;
	}
	if (!_shape.is_null()) {
		_shape->unregister_owner(this);
	}
	_shape = p_shape;
	if (_shape.is_valid()) {
		_shape->register_owner(this);

		if (is_inside_world() && get_world().is_valid()) {
			if (_occluder_instance.is_valid()) {
				VisualServer::get_singleton()->occluder_instance_link_resource(_occluder_instance, p_shape->get_rid());
			}
		}
	}

	update_gizmo();
	update_configuration_warning();
}
Ref<OccluderShape> Occluder::get_shape() const {
	return _shape;
}

#ifdef TOOLS_ENABLED
AABB Occluder::get_fallback_gizmo_aabb() const {
	if (_shape.is_valid()) {
		return _shape->get_fallback_gizmo_aabb();
	}
	return Spatial::get_fallback_gizmo_aabb();
}
#endif

String Occluder::get_configuration_warning() const {
	String warning = Spatial::get_configuration_warning();

	if (!_shape.is_valid()) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("No shape is set.");
		return warning;
	}

#ifdef TOOLS_ENABLED
	if (_shape.ptr()->requires_uniform_scale()) {
		Transform tr = get_global_transform();
		Vector3 scale = tr.basis.get_scale();

		if ((!Math::is_equal_approx(scale.x, scale.y, 0.01f)) ||
				(!Math::is_equal_approx(scale.x, scale.z, 0.01f))) {
			if (!warning.empty()) {
				warning += "\n\n";
			}
			warning += TTR("Only uniform scales are supported.");
		}
	}
#endif

	return warning;
}

void Occluder::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {
			ERR_FAIL_COND(get_world().is_null());

			if (_occluder_instance.is_valid()) {
				VisualServer::get_singleton()->occluder_instance_set_scenario(_occluder_instance, get_world()->get_scenario());
				if (get_shape().is_valid()) {
					VisualServer::get_singleton()->occluder_instance_link_resource(_occluder_instance, get_shape()->get_rid());
				}
				VisualServer::get_singleton()->occluder_instance_set_active(_occluder_instance, is_visible_in_tree());
				VisualServer::get_singleton()->occluder_instance_set_transform(_occluder_instance, get_global_transform());
			}

#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				set_process_internal(true);
			}
#endif
		} break;
		case NOTIFICATION_EXIT_WORLD: {
			if (_occluder_instance.is_valid()) {
				VisualServer::get_singleton()->occluder_instance_set_scenario(_occluder_instance, RID());
			}
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				set_process_internal(false);
			}
#endif
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (_occluder_instance.is_valid() && is_inside_tree()) {
				VisualServer::get_singleton()->occluder_instance_set_active(_occluder_instance, is_visible_in_tree());
			}
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (_occluder_instance.is_valid()) {
				VisualServer::get_singleton()->occluder_instance_set_transform(_occluder_instance, get_global_transform());
#ifdef TOOLS_ENABLED
				if (Engine::get_singleton()->is_editor_hint()) {
					update_configuration_warning();
				}
#endif
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (PortalOcclusionCuller::_redraw_gizmo) {
				PortalOcclusionCuller::_redraw_gizmo = false;
				update_gizmo();
			}
		} break;
	}
}

void Occluder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("resource_changed", "resource"), &Occluder::resource_changed);
	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &Occluder::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &Occluder::get_shape);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "OccluderShape"), "set_shape", "get_shape");
}

Occluder::Occluder() {
	_occluder_instance = RID_PRIME(VisualServer::get_singleton()->occluder_instance_create());
	set_notify_transform(true);
}

Occluder::~Occluder() {
	if (_occluder_instance != RID()) {
		VisualServer::get_singleton()->free(_occluder_instance);
	}

	if (!_shape.is_null()) {
		_shape->unregister_owner(this);
	}
}

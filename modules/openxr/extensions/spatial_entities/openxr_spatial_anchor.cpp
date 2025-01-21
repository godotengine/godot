/**************************************************************************/
/*  openxr_spatial_anchor.cpp                                             */
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

#include "openxr_spatial_anchor.h"
#include "../openxr_future_extension.h"

#include "../../openxr_api.h"
#include "../../openxr_util.h"
#include "core/config/project_settings.h"
#include "servers/xr_server.h"

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialCapabilityConfigurationAnchor

void OpenXRSpatialCapabilityConfigurationAnchor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_enabled_components"), &OpenXRSpatialCapabilityConfigurationAnchor::_get_enabled_components);
}

bool OpenXRSpatialCapabilityConfigurationAnchor::has_valid_configuration() const {
	OpenXRSpatialAnchorCapability *capability = OpenXRSpatialAnchorCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, false);

	return capability->is_spatial_anchor_supported();
}

XrSpatialCapabilityConfigurationBaseHeaderEXT *OpenXRSpatialCapabilityConfigurationAnchor::get_configuration() {
	OpenXRSpatialAnchorCapability *capability = OpenXRSpatialAnchorCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, nullptr);

	if (capability->is_spatial_anchor_supported()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL_V(se_extension, nullptr);

		anchor_enabled_components.clear();

		// Guaranteed components:
		anchor_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_ANCHOR_EXT);

		// enable optional components
		if (capability->is_spatial_persistence_supported()) {
			anchor_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_PERSISTENCE_EXT);
		}

		anchor_config.enabledComponentCount = anchor_enabled_components.size();
		anchor_config.enabledComponents = anchor_enabled_components.ptr();

		// and return this.
		return (XrSpatialCapabilityConfigurationBaseHeaderEXT *)&anchor_config;
	}

	return nullptr;
}

PackedInt64Array OpenXRSpatialCapabilityConfigurationAnchor::_get_enabled_components() const {
	PackedInt64Array components;

	for (const XrSpatialComponentTypeEXT &component_type : anchor_enabled_components) {
		components.push_back((int64_t)component_type);
	}

	return components;
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialComponentAnchorList

void OpenXRSpatialComponentAnchorList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_entity_pose", "index"), &OpenXRSpatialComponentAnchorList::get_entity_pose);
}

void OpenXRSpatialComponentAnchorList::set_capacity(uint32_t p_entity_id_capacity, uint32_t p_entity_state_capacity) {
	entity_poses.resize(p_entity_id_capacity);

	anchor_list.locationCount = uint32_t(entity_poses.size());
	anchor_list.locations = entity_poses.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentAnchorList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_ANCHOR_EXT;
}

void *OpenXRSpatialComponentAnchorList::get_structure_data(void *p_next) {
	anchor_list.next = p_next;
	return &anchor_list;
}

Transform3D OpenXRSpatialComponentAnchorList::get_entity_pose(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, entity_poses.size(), Transform3D());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Transform3D());

	return openxr_api->transform_from_pose(entity_poses[p_index]);
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialContextPersistenceConfig1

void OpenXRSpatialContextPersistenceConfig::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_persistence_context", "persistence_context"), &OpenXRSpatialContextPersistenceConfig::add_persistence_context);
	ClassDB::bind_method(D_METHOD("remove_persistence_context", "persistence_context"), &OpenXRSpatialContextPersistenceConfig::remove_persistence_context);
}

bool OpenXRSpatialContextPersistenceConfig::has_valid_configuration() const {
	OpenXRSpatialAnchorCapability *capability = OpenXRSpatialAnchorCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, false);

	if (!capability->is_spatial_persistence_supported()) {
		return false;
	}

	// Check if we have a valid config.
	if (persistence_contexts.is_empty()) {
		return false;
	}

	return true;
}

void *OpenXRSpatialContextPersistenceConfig::get_header(void *p_next) {
	void *n = p_next;
	if (get_next().is_valid()) {
		n = get_next()->get_header(n);
	}

	if (has_valid_configuration()) {
		OpenXRSpatialAnchorCapability *anchor_capability = OpenXRSpatialAnchorCapability::get_singleton();
		ERR_FAIL_NULL_V(anchor_capability, nullptr);

		// Prepare our buffer.
		context_handles.resize(persistence_contexts.size());

		// Copy our handles.
		XrSpatialPersistenceContextEXT *ptr = context_handles.ptrw();
		int i = 0;
		for (const RID &rid : persistence_contexts) {
			ptr[i++] = anchor_capability->get_persistence_context_handle(rid);
		}

		persistence_config.next = n;
		persistence_config.persistenceContextCount = (uint32_t)context_handles.size();
		persistence_config.persistenceContexts = context_handles.ptr();

		// and return this.
		return (XrSpatialCapabilityConfigurationBaseHeaderEXT *)&persistence_config;
	}

	return n;
}

void OpenXRSpatialContextPersistenceConfig::add_persistence_context(RID p_persistence_context) {
	ERR_FAIL_COND(persistence_contexts.has(p_persistence_context));

	persistence_contexts.push_back(p_persistence_context);
}

void OpenXRSpatialContextPersistenceConfig::remove_persistence_context(RID p_persistence_context) {
	ERR_FAIL_COND(!persistence_contexts.has(p_persistence_context));

	persistence_contexts.erase(p_persistence_context);
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialComponentPersistenceList

void OpenXRSpatialComponentPersistenceList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_persistent_uuid", "index"), &OpenXRSpatialComponentPersistenceList::_get_persistent_uuid);
	ClassDB::bind_method(D_METHOD("get_persistent_state", "index"), &OpenXRSpatialComponentPersistenceList::_get_persistent_state);
}

void OpenXRSpatialComponentPersistenceList::set_capacity(uint32_t p_entity_id_capacity, uint32_t p_entity_state_capacity) {
	persist_data.resize(p_entity_id_capacity);

	persistence_list.persistDataCount = uint32_t(persist_data.size());
	persistence_list.persistData = persist_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentPersistenceList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_PERSISTENCE_EXT;
}

void *OpenXRSpatialComponentPersistenceList::get_structure_data(void *p_next) {
	persistence_list.next = p_next;
	return &persistence_list;
}

XrUuid OpenXRSpatialComponentPersistenceList::get_persistent_uuid(int64_t p_index) const {
	XrUuid null_uuid = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	ERR_FAIL_INDEX_V(p_index, persist_data.size(), null_uuid);

	return persist_data[p_index].persistUuid;
}

String OpenXRSpatialComponentPersistenceList::_get_persistent_uuid(int64_t p_index) const {
	String ret;
	XrUuid uuid = get_persistent_uuid(p_index);
	bool non_zero = false;

	// TODO move this code more central, maybe add functions to openxr_util.h ?

	for (int i = 0; i < XR_UUID_SIZE; i++) {
		non_zero |= uuid.data[i] != 0;

		char a = uuid.data[i] & 0xF0 >> 4;
		char b = uuid.data[i] & 0x0F;

		if (a < 10) {
			ret += '0' + a;
		} else {
			ret += 'a' + a;
		}

		if (b < 10) {
			ret += '0' + b;
		} else {
			ret += 'a' + b;
		}
	}

	if (non_zero) {
		return ret;
	} else {
		return "";
	}
}

XrSpatialPersistenceStateEXT OpenXRSpatialComponentPersistenceList::get_persistent_state(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, persist_data.size(), XR_SPATIAL_PERSISTENCE_STATE_MAX_ENUM_EXT);

	return persist_data[p_index].persistState;
}

uint64_t OpenXRSpatialComponentPersistenceList::_get_persistent_state(int64_t p_index) const {
	// TODO make a Godot constant that mirrors XrSpatialPersistenceStateEXT and return that
	return (uint64_t)get_persistent_state(p_index);
}

String OpenXRSpatialComponentPersistenceList::get_persistence_state_name(XrSpatialPersistenceStateEXT p_state) {
	XR_ENUM_SWITCH(XrSpatialPersistenceStateEXT, p_state)
}

////////////////////////////////////////////////////////////////////////////
// OpenXRAnchorTracker

void OpenXRAnchorTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_uuid"), &OpenXRAnchorTracker::has_uuid);
	ClassDB::bind_method(D_METHOD("set_uuid", "uuid"), &OpenXRAnchorTracker::_set_uuid);
	ClassDB::bind_method(D_METHOD("get_uuid"), &OpenXRAnchorTracker::_get_uuid);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "uuid"), "set_uuid", "get_uuid");

	ADD_SIGNAL(MethodInfo("uuid_changed"));
}

bool OpenXRAnchorTracker::has_uuid() const {
	for (int i = 0; i < XR_UUID_SIZE; i++) {
		if (uuid.data[i] != 0) {
			return true;
		}
	}

	return false;
}

XrUuid OpenXRAnchorTracker::get_uuid() const {
	return uuid;
}

void OpenXRAnchorTracker::set_uuid(const XrUuid &p_uuid) {
	if (uuid_is_equal(uuid, p_uuid)) {
		return;
	}

	uuid = p_uuid;

	emit_signal(SNAME("uuid_changed"));
}

String OpenXRAnchorTracker::_get_uuid() const {
	String ret;
	bool non_zero = false;

	// TODO move this code more central, maybe add functions to openxr_util.h ?

	for (int i = 0; i < XR_UUID_SIZE; i++) {
		non_zero |= uuid.data[i] != 0;

		char a = (uuid.data[i] & 0xF0) >> 4;
		char b = uuid.data[i] & 0x0F;

		if (a < 10) {
			ret += '0' + a;
		} else {
			ret += 'a' + a - 10;
		}

		if (b < 10) {
			ret += '0' + b;
		} else {
			ret += 'a' + b - 10;
		}
	}

	if (non_zero) {
		return ret;
	} else {
		return "";
	}
}

void OpenXRAnchorTracker::_set_uuid(const String &p_uuid) {
	ERR_FAIL_COND(p_uuid.size() != XR_UUID_SIZE);

	// TODO move this code more central, maybe add functions to openxr_util.h ?

	XrUuid new_uuid = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int len = p_uuid.length();
	if (len == 0) {
		set_uuid(new_uuid);
		return;
	} else if (len != (2 * XR_UUID_SIZE)) {
		WARN_PRINT("OpenXR : unexpected UUID length: " + String::num_int64(len) + " != " + String::num_int64(2 * XR_UUID_SIZE));
	}

	int j = 0;
	for (int i = 0; i < XR_UUID_SIZE; i++) {
		uint8_t val = 0;

		// 2 chars per byte.
		for (int k = 0; k < 2; k++) {
			if (j < len) {
				val <<= 4;

				char32_t c = p_uuid[j++];
				if (c >= '0' && c <= '9') {
					val += uint8_t(c - '0');
				} else if (c >= 'a' && c <= 'f') {
					val += uint8_t(10 + c - 'a');
				} else if (c >= 'A' && c <= 'F') {
					val += uint8_t(10 + c - 'A');
				} else {
					WARN_PRINT("OpenXR : unexpected character in UUID: " + String::num_int64(c));
				}
			}
		}

		new_uuid.data[i] = val;
	}

	set_uuid(new_uuid);
}

bool OpenXRAnchorTracker::uuid_is_equal(const XrUuid &p_a, const XrUuid &p_b) {
	for (int i = 0; i < XR_UUID_SIZE; i++) {
		if (p_a.data[i] != p_b.data[i]) {
			return false;
		}
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialAnchorCapability

OpenXRSpatialAnchorCapability *OpenXRSpatialAnchorCapability::singleton = nullptr;

OpenXRSpatialAnchorCapability *OpenXRSpatialAnchorCapability::get_singleton() {
	return singleton;
}

OpenXRSpatialAnchorCapability::OpenXRSpatialAnchorCapability() {
	singleton = this;
}

OpenXRSpatialAnchorCapability::~OpenXRSpatialAnchorCapability() {
	singleton = nullptr;
}

void OpenXRSpatialAnchorCapability::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_spatial_anchor_supported"), &OpenXRSpatialAnchorCapability::is_spatial_anchor_supported);
	ClassDB::bind_method(D_METHOD("is_spatial_persistence_supported"), &OpenXRSpatialAnchorCapability::is_spatial_persistence_supported);

	ClassDB::bind_method(D_METHOD("is_persistence_scope_supported", "scope"), &OpenXRSpatialAnchorCapability::_is_persistence_scope_supported);
	ClassDB::bind_method(D_METHOD("create_persistence_context", "scope", "user_callback"), &OpenXRSpatialAnchorCapability::_create_persistence_context, DEFVAL(Callable()));
	ClassDB::bind_method(D_METHOD("get_persistence_context_handle", "persistence_context"), &OpenXRSpatialAnchorCapability::_get_persistence_context_handle);
	ClassDB::bind_method(D_METHOD("free_persistence_context", "persistence_context"), &OpenXRSpatialAnchorCapability::free_persistence_context);

	ClassDB::bind_method(D_METHOD("create_new_anchor", "transform", "spatial_context"), &OpenXRSpatialAnchorCapability::create_new_anchor, DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("remove_anchor", "anchor_tracker"), &OpenXRSpatialAnchorCapability::remove_anchor);
	ClassDB::bind_method(D_METHOD("make_anchor_persistent", "anchor_tracker", "persistence_context", "callback"), &OpenXRSpatialAnchorCapability::make_anchor_persistent, DEFVAL(RID()), DEFVAL(Callable()));
	ClassDB::bind_method(D_METHOD("make_anchor_unpersistent", "anchor_tracker", "persistence_context", "callback"), &OpenXRSpatialAnchorCapability::make_anchor_unpersistent, DEFVAL(RID()), DEFVAL(Callable()));

	ADD_SIGNAL(MethodInfo("persistence_creation_failed", PropertyInfo(Variant::RID, "persistence_context")));
	ADD_SIGNAL(MethodInfo("anchor_persistent_failed", PropertyInfo(Variant::OBJECT, "tracker", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRAnchorTracker")));
	ADD_SIGNAL(MethodInfo("anchor_unpersistent_failed", PropertyInfo(Variant::OBJECT, "tracker", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRAnchorTracker")));

	BIND_ENUM_CONSTANT(OPENXR_SPATIAL_PERSISTENCE_SCOPE_SYSTEM_MANAGED);
	BIND_ENUM_CONSTANT(OPENXR_SPATIAL_PERSISTENCE_SCOPE_LOCAL_ANCHORS);
}

HashMap<String, bool *> OpenXRSpatialAnchorCapability::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	if (GLOBAL_GET("xr/openxr/extensions/spatial_entity/enabled") && GLOBAL_GET("xr/openxr/extensions/spatial_entity/enable_spatial_anchors")) {
		request_extensions[XR_EXT_SPATIAL_ANCHOR_EXTENSION_NAME] = &spatial_anchor_ext;
		if (GLOBAL_GET("xr/openxr/extensions/spatial_entity/enable_persistent_anchors")) {
			request_extensions[XR_EXT_SPATIAL_PERSISTENCE_EXTENSION_NAME] = &spatial_persistence_ext;
			request_extensions[XR_EXT_SPATIAL_PERSISTENCE_OPERATIONS_EXTENSION_NAME] = &spatial_persistence_operations_ext;
		}
	}

	return request_extensions;
}

void OpenXRSpatialAnchorCapability::on_instance_created(const XrInstance p_instance) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	if (spatial_anchor_ext && !se_extension->supports_capability(XR_SPATIAL_CAPABILITY_ANCHOR_EXT)) {
		// Supported by XR runtime but not by device? Disable this!
		spatial_anchor_ext = false;
	}

	if (spatial_anchor_ext) {
		EXT_INIT_XR_FUNC(xrCreateSpatialAnchorEXT);
	}

	if (spatial_persistence_ext) {
		EXT_INIT_XR_FUNC(xrEnumerateSpatialPersistenceScopesEXT);
		EXT_INIT_XR_FUNC(xrCreateSpatialPersistenceContextAsyncEXT);
		EXT_INIT_XR_FUNC(xrCreateSpatialPersistenceContextCompleteEXT);
		EXT_INIT_XR_FUNC(xrDestroySpatialPersistenceContextEXT);
	}

	if (spatial_persistence_operations_ext) {
		EXT_INIT_XR_FUNC(xrPersistSpatialEntityAsyncEXT);
		EXT_INIT_XR_FUNC(xrPersistSpatialEntityCompleteEXT);
		EXT_INIT_XR_FUNC(xrUnpersistSpatialEntityAsyncEXT);
		EXT_INIT_XR_FUNC(xrUnpersistSpatialEntityCompleteEXT);
	}
}

void OpenXRSpatialAnchorCapability::on_instance_destroyed() {
	xrCreateSpatialAnchorEXT_ptr = nullptr;

	xrEnumerateSpatialPersistenceScopesEXT_ptr = nullptr;
	xrCreateSpatialPersistenceContextAsyncEXT_ptr = nullptr;
	xrCreateSpatialPersistenceContextCompleteEXT_ptr = nullptr;
	xrDestroySpatialPersistenceContextEXT_ptr = nullptr;
}

void OpenXRSpatialAnchorCapability::on_session_created(const XrSession p_session) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	if (!spatial_anchor_ext) {
		return;
	}

	_load_supported_persistence_scopes();

	se_extension->connect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialAnchorCapability::_on_spatial_discovery_recommended));

	if (GLOBAL_GET("xr/openxr/extensions/spatial_entity/enable_builtin_anchor_detection")) {
		if (spatial_persistence_ext && !supported_persistence_scopes.is_empty()) {
			// TODO make something nicer to select the persistence scope we want.
			// We may even want to create multiple here so we get access to all
			// but then mark one that we use to create our persistent anchors on.

			XrSpatialPersistenceScopeEXT scope = XR_SPATIAL_PERSISTENCE_SCOPE_MAX_ENUM_EXT;

			// Lets check these in order of importance to us and find the best applicable scope.
			if (supported_persistence_scopes.has(XR_SPATIAL_PERSISTENCE_SCOPE_LOCAL_ANCHORS_EXT)) {
				// This scope allows for local storage and is required if we want to create our own anchors.
				scope = XR_SPATIAL_PERSISTENCE_SCOPE_LOCAL_ANCHORS_EXT;
			} else if (supported_persistence_scopes.has(XR_SPATIAL_PERSISTENCE_SCOPE_SYSTEM_MANAGED_EXT)) {
				// The system managed scope is a read only scope with system managed anchors.
				scope = XR_SPATIAL_PERSISTENCE_SCOPE_SYSTEM_MANAGED_EXT;
			} else {
				// Just use the first supported scope, but this will be an unknown type.
				scope = supported_persistence_scopes[0];
			}

			// Output what we're using:
			print_verbose("OpenXR: Using persistence scope " + get_spatial_persistence_scope_name(scope));

			// Start by creating our persistence context.
			create_persistence_context(scope, callable_mp(this, &OpenXRSpatialAnchorCapability::_on_persistence_context_completed));
		} else {
			// Start by creating our spatial context
			_create_spatial_context();
		}
	}
}

void OpenXRSpatialAnchorCapability::on_session_destroyed() {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	// Free and unregister our anchors
	for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRAnchorTracker>> &anchor : anchors) {
		xr_server->remove_tracker(anchor.value);
	}
	anchors.clear();

	// Free our configurations
	anchor_configuration.unref();

	// Free our spatial context
	if (spatial_context.is_valid()) {
		se_extension->free_spatial_context(spatial_context);
		spatial_context = RID();
	}

	// Free our persistence context
	if (persistence_context.is_valid()) {
		free_persistence_context(persistence_context);
		persistence_context = RID();
	}

	se_extension->disconnect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialAnchorCapability::_on_spatial_discovery_recommended));

	supported_persistence_scopes.clear();

	// Clean up all remaining persistence context RIDs.
	LocalVector<RID> persistence_context_rids = persistence_context_owner.get_owned_list();
	for (const RID &rid : persistence_context_rids) {
		// Q should we log this? Means someone forgot to clean up.
		free_persistence_context(rid);
	}
}

void OpenXRSpatialAnchorCapability::on_process() {
	// If we have a valid spatial context, and we have anchors, we want updates!
	if (spatial_context.is_valid() && !anchors.is_empty()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL(se_extension);

		// We want updates for all anchors
		Vector<RID> entities;
		entities.resize(anchors.size());
		RID *entity = entities.ptrw();
		for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRAnchorTracker>> &e : anchors) {
			*entity = e.value->get_entity();
			entity++;
		}

		// We just want our anchor component
		Vector<XrSpatialComponentTypeEXT> component_types;
		component_types.push_back(XR_SPATIAL_COMPONENT_TYPE_ANCHOR_EXT);

		// And we get our update snapshot, this is NOT async!
		RID snapshot = se_extension->update_spatial_entities(spatial_context, entities, component_types, nullptr);
		if (snapshot.is_valid()) {
			_on_spatial_discovery_completed(snapshot, false);
		}
	}
}

bool OpenXRSpatialAnchorCapability::is_spatial_anchor_supported() {
	if (!spatial_anchor_ext) {
		return false;
	}

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, false);

	return se_extension->supports_component_type(XR_SPATIAL_CAPABILITY_ANCHOR_EXT, XR_SPATIAL_COMPONENT_TYPE_ANCHOR_EXT);
}

bool OpenXRSpatialAnchorCapability::is_spatial_persistence_supported() {
	// Need anchor support for persistence to be usable
	if (!is_spatial_anchor_supported()) {
		return false;
	}

	return spatial_persistence_ext;
}

////////////////////////////////////////////////////////////////////////////
// Persistence scopes

bool OpenXRSpatialAnchorCapability::_load_supported_persistence_scopes() {
	ERR_FAIL_COND_V(!spatial_persistence_ext, false);

	if (supported_persistence_scopes.size() == 0) {
		OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
		ERR_FAIL_NULL_V(openxr_api, false);

		uint32_t size;
		XrInstance instance = openxr_api->get_instance();
		XrSystemId system_id = openxr_api->get_system_id();

		ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, false);

		XrResult result = xrEnumerateSpatialPersistenceScopesEXT(instance, system_id, 0, &size, nullptr);
		if (XR_FAILED(result)) {
			ERR_FAIL_V_MSG(false, "OpenXR: Failed to query persistence scopes count [" + openxr_api->get_error_string(result) + "]");
		}

		if (size > 0) {
			supported_persistence_scopes.resize(size);
			result = xrEnumerateSpatialPersistenceScopesEXT(instance, system_id, supported_persistence_scopes.size(), &size, supported_persistence_scopes.ptrw());
			if (XR_FAILED(result)) {
				ERR_FAIL_V_MSG(false, "OpenXR: Failed to query persistence scopes [" + openxr_api->get_error_string(result) + "]");
			}
		}

		if (is_print_verbose_enabled()) {
			if (supported_persistence_scopes.size() > 0) {
				print_verbose("OpenXR: Supported spatial persistence scopes:");
				for (const XrSpatialPersistenceScopeEXT &scope : supported_persistence_scopes) {
					print_verbose(" - " + get_spatial_persistence_scope_name(scope));
				}
			} else {
				WARN_PRINT("OpenXR: No persistence scopes found!");
			}
		}
	}

	return true;
}

bool OpenXRSpatialAnchorCapability::is_persistence_scope_supported(XrSpatialPersistenceScopeEXT p_scope) {
	if (!is_spatial_persistence_supported()) {
		return false;
	}

	if (!_load_supported_persistence_scopes()) {
		return false;
	}

	return supported_persistence_scopes.has(p_scope);
}

bool OpenXRSpatialAnchorCapability::_is_persistence_scope_supported(OpenXrSpatialPersistenceScope p_scope) {
	return is_persistence_scope_supported((XrSpatialPersistenceScopeEXT)p_scope);
}

Ref<OpenXRFutureResult> OpenXRSpatialAnchorCapability::create_persistence_context(XrSpatialPersistenceScopeEXT p_scope, Callable p_user_callback) {
	if (!is_spatial_persistence_supported()) {
		return nullptr;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, nullptr);

	OpenXRFutureExtension *future_api = OpenXRFutureExtension::get_singleton();
	ERR_FAIL_NULL_V(future_api, nullptr);

	XrSpatialPersistenceContextCreateInfoEXT create_info = {
		XR_TYPE_SPATIAL_PERSISTENCE_CONTEXT_CREATE_INFO_EXT, // type
		nullptr, // next
		p_scope // scope
	};
	XrFutureEXT future = XR_NULL_HANDLE;
	XrResult result = xrCreateSpatialPersistenceContextAsyncEXT(openxr_api->get_session(), &create_info, &future);
	if (XR_FAILED(result)) {
		// Not successful? then exit.
		ERR_PRINT("OpenXR: Failed to create persistence scope [" + openxr_api->get_error_string(result) + "]");
		return nullptr;
	}

	// Create our future result
	Ref<OpenXRFutureResult> future_result = future_api->register_future(future, callable_mp(this, &OpenXRSpatialAnchorCapability::_on_persistence_context_ready).bind(p_user_callback).bind((uint64_t)p_scope));

	return future_result;
}

Ref<OpenXRFutureResult> OpenXRSpatialAnchorCapability::_create_persistence_context(OpenXrSpatialPersistenceScope p_scope, Callable p_user_callback) {
	return create_persistence_context((XrSpatialPersistenceScopeEXT)p_scope, p_user_callback);
}

void OpenXRSpatialAnchorCapability::_on_persistence_context_ready(Ref<OpenXRFutureResult> p_future_result, uint64_t p_scope, Callable p_user_callback) {
	// Complete context creation...
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	XrCreateSpatialPersistenceContextCompletionEXT completion = {
		XR_TYPE_CREATE_SPATIAL_PERSISTENCE_CONTEXT_COMPLETION_EXT, // type
		nullptr, // next
		XR_RESULT_MAX_ENUM, // futureResult
		XR_SPATIAL_PERSISTENCE_CONTEXT_RESULT_MAX_ENUM_EXT, // createResult
		XR_NULL_HANDLE // persistenceContext
	};
	XrResult result = xrCreateSpatialPersistenceContextCompleteEXT(openxr_api->get_session(), p_future_result->get_future(), &completion);
	if (XR_FAILED(result)) { // Did our xrCreateSpatialContextCompleteEXT call fail?
		// Log issue and fail.
		ERR_FAIL_MSG("OpenXR: Failed to complete persistence context create future [" + openxr_api->get_error_string(result) + "]");
	}
	if (XR_FAILED(completion.futureResult)) { // Did our completion fail?
		// Log issue and fail.
		ERR_FAIL_MSG("OpenXR: Failed to complete persistence context creation [" + openxr_api->get_error_string(completion.futureResult) + "]");
	}
	if (completion.createResult != XR_SPATIAL_PERSISTENCE_CONTEXT_RESULT_SUCCESS_EXT) { // Did our persist fail?
		// Log issue and fail.
		ERR_FAIL_MSG("OpenXR: Failed to complete persistence context creation [" + get_spatial_persistence_context_result_name(completion.createResult) + "]");
	}

	// Wrap our persistence context
	PersistenceContextData persistence_context_data;

	// Update our spatial context data
	persistence_context_data.scope = (XrSpatialPersistenceScopeEXT)p_scope;
	persistence_context_data.persistence_context = completion.persistenceContext;

	// Store this as a RID so we keep track of it.
	RID context_rid = persistence_context_owner.make_rid(persistence_context_data);

	// Set our rid as our result value on our future.
	p_future_result->set_result_value(context_rid);

	// And perform our callback if we have one.
	if (p_user_callback.is_valid()) {
		p_user_callback.call(context_rid);
	}
}

XrSpatialPersistenceContextEXT OpenXRSpatialAnchorCapability::get_persistence_context_handle(RID p_persistence_context) const {
	PersistenceContextData *persistence_context_data = persistence_context_owner.get_or_null(p_persistence_context);
	ERR_FAIL_NULL_V(persistence_context_data, XR_NULL_HANDLE);

	return persistence_context_data->persistence_context;
}

uint64_t OpenXRSpatialAnchorCapability::_get_persistence_context_handle(RID p_persistence_context) const {
	return (uint64_t)get_persistence_context_handle(p_persistence_context);
}

void OpenXRSpatialAnchorCapability::free_persistence_context(RID p_persistence_context) {
	PersistenceContextData *persistence_context_data = persistence_context_owner.get_or_null(p_persistence_context);
	ERR_FAIL_NULL(persistence_context_data);

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	if (persistence_context_data->persistence_context != XR_NULL_HANDLE) {
		// Destroy our spatial context
		XrResult result = xrDestroySpatialPersistenceContextEXT(persistence_context_data->persistence_context);
		if (XR_FAILED(result)) {
			WARN_PRINT("OpenXR: Failed to destroy the persistence context [" + openxr_api->get_error_string(result) + "]");
		}
		persistence_context_data->persistence_context = XR_NULL_HANDLE;
	}

	// And remove our RID.
	persistence_context_owner.free(p_persistence_context);
}

////////////////////////////////////////////////////////////////////////////
// Discovery logic

void OpenXRSpatialAnchorCapability::_on_persistence_context_completed(RID p_persistence_context) {
	persistence_context = p_persistence_context;

	_create_spatial_context();
}

Ref<OpenXRFutureResult> OpenXRSpatialAnchorCapability::_create_spatial_context() {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, nullptr);

	TypedArray<OpenXRSpatialCapabilityConfigurationBaseHeader> capability_configurations;

	// Create our configuration objects.
	anchor_configuration.instantiate();
	capability_configurations.push_back(anchor_configuration);

	if (persistence_context.is_valid()) {
		persistence_configuration.instantiate();
		persistence_configuration->add_persistence_context(persistence_context);
	} else {
		// Shouldn't be instantiated in the first place but JIC
		persistence_configuration.unref();
	}

	return se_extension->create_spatial_context(capability_configurations, persistence_configuration, callable_mp(this, &OpenXRSpatialAnchorCapability::_on_spatial_context_created));
}

void OpenXRSpatialAnchorCapability::_on_spatial_context_created(RID p_spatial_context) {
	spatial_context = p_spatial_context;

	_start_entity_discovery();
}

void OpenXRSpatialAnchorCapability::_on_spatial_discovery_recommended(RID p_spatial_context) {
	if (p_spatial_context == spatial_context) {
		// Trigger new discovery
		_start_entity_discovery();
	}
}

Ref<OpenXRFutureResult> OpenXRSpatialAnchorCapability::_start_entity_discovery() {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, nullptr);

	// It makes no sense to discover non persistent anchors as we'd have created them during this session.
	if (!persistence_context.is_valid()) {
		return nullptr;
	}

	// Already running, or ran discovery, cancel/cleanup
	if (discovery_query_result.is_valid()) {
		discovery_query_result->cancel_future();
		discovery_query_result.unref();
	}

	// We want both our anchor and persistence component
	Vector<XrSpatialComponentTypeEXT> component_types;
	component_types.push_back(XR_SPATIAL_COMPONENT_TYPE_ANCHOR_EXT);
	component_types.push_back(XR_SPATIAL_COMPONENT_TYPE_PERSISTENCE_EXT);

	// Start our new snapshot.
	discovery_query_result = se_extension->discover_spatial_entities(spatial_context, component_types, nullptr, callable_mp(this, &OpenXRSpatialAnchorCapability::_on_spatial_discovery_completed).bind(true));

	return discovery_query_result;
}

void OpenXRSpatialAnchorCapability::_on_spatial_discovery_completed(RID p_snapshot, bool p_get_persistence_data) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Build our component data
	TypedArray<OpenXRSpatialComponentData> component_data;

	// We always need a query result data object
	Ref<OpenXRSpatialQueryResultData> query_result_data;
	query_result_data.instantiate();
	component_data.push_back(query_result_data);

	// And a anchor list object
	Ref<OpenXRSpatialComponentAnchorList> anchor_list_data;
	anchor_list_data.instantiate();
	component_data.push_back(anchor_list_data);

	Ref<OpenXRSpatialComponentPersistenceList> persistence_list_data;
	if (p_get_persistence_data) {
		// Note that adding this data object means our snapshot will only return persistent anchors!
		persistence_list_data.instantiate();
		component_data.push_back(persistence_list_data);
	}

	if (se_extension->query_snapshot(p_snapshot, component_data, nullptr)) {
		// Now loop through our data and update our anchors.
		// Q we're assuming entity id size and state size are equal, is there ever a situation where they would not be?
		int64_t size = query_result_data->get_entity_id_size();
		for (int64_t i = 0; i < size; i++) {
			XrSpatialEntityIdEXT entity_id = query_result_data->get_entity_id(i);
			XrSpatialEntityTrackingStateEXT entity_state = query_result_data->get_entity_state(i);

			if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT) {
				if (anchors.has(entity_id)) {
					Ref<OpenXRAnchorTracker> anchor = anchors[entity_id];
					anchor->invalidate_pose(SNAME("default"));
					anchor->set_spatial_tracking_state(OpenXRSpatialEntityTracker::OPENXR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED);
				}
			} else {
				// Process our entity
				bool add_to_xr_server = false;
				Ref<OpenXRAnchorTracker> anchor;

				if (anchors.has(entity_id)) {
					// We know about this one already
					anchor = anchors[entity_id];
				} else {
					// Create a new anchor
					anchor.instantiate();
					anchor->set_entity(se_extension->make_spatial_entity(se_extension->get_spatial_snapshot_context(p_snapshot), entity_id));
					anchors[entity_id] = anchor;

					add_to_xr_server = true;
				}

				// Handle component data
				if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT) {
					anchor->invalidate_pose(SNAME("default"));
					anchor->set_spatial_tracking_state(OpenXRSpatialEntityTracker::OPENXR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED);

					// No further component data will be valid in this state, we need to ignore it!
				} else if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING_EXT) {
					Transform3D transform = anchor_list_data->get_entity_pose(i);
					anchor->set_pose(SNAME("default"), transform, Vector3(), Vector3());
					anchor->set_spatial_tracking_state(OpenXRSpatialEntityTracker::OPENXR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING);
				}

				// Note: Persistence is the only component that will contain valid data if entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT.
				if (persistence_list_data.is_valid()) {
					const XrSpatialPersistenceStateEXT persistent_state = persistence_list_data->get_persistent_state(i);

					if (persistent_state == XR_SPATIAL_PERSISTENCE_STATE_LOADED_EXT) {
						anchor->set_uuid(persistence_list_data->get_persistent_uuid(i));
					}
				}

				if (add_to_xr_server) {
					// Register with XR server
					xr_server->add_tracker(anchor);
				}
			}
		}

		// We don't remove trackers here, users will be removing anchors.
		// Maybe at some point when shared anchors between headsets result
		// in another device removing the shared anchor we need to deal with this.
	}

	// Now that we're done, clean up our snapshot!
	se_extension->free_spatial_snapshot(p_snapshot);

	// And if this was our discovery snapshot, lets reset it
	if (discovery_query_result.is_valid() && discovery_query_result->get_result_value() == p_snapshot) {
		discovery_query_result.unref();
	}
}

////////////////////////////////////////////////////////////////////////////
// Anchor creation

Ref<OpenXRAnchorTracker> OpenXRSpatialAnchorCapability::create_new_anchor(const Transform3D &p_transform, RID p_spatial_context) {
	Ref<OpenXRAnchorTracker> tracker;

	ERR_FAIL_COND_V_MSG(!spatial_anchor_ext, tracker, "OpenXR: Spatial entity anchor capability is not supported on this hardware!");

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, tracker);
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, tracker);
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, tracker);

	// TODO reverse apply world scale and reference frame to transform

	XrPosef pose = openxr_api->pose_from_transform(p_transform);

	RID sc = p_spatial_context.is_valid() ? p_spatial_context : spatial_context;
	ERR_FAIL_COND_V(sc.is_null(), tracker);

	XrSpatialAnchorCreateInfoEXT create_info = {
		XR_TYPE_SPATIAL_ANCHOR_CREATE_INFO_EXT, // type
		nullptr, // next
		openxr_api->get_play_space(), // baseSpace
		openxr_api->get_predicted_display_time(), // time
		pose // pose
	};
	XrSpatialEntityIdEXT entity_id;
	XrSpatialEntityEXT entity;
	XrResult result = xrCreateSpatialAnchorEXT(se_extension->get_spatial_context_handle(sc), &create_info, &entity_id, &entity);
	if (XR_FAILED(result)) { // Did our xrCreateSpatialContextCompleteEXT call fail?
		ERR_FAIL_V_MSG(tracker, "OpenXR: Failed to create anchor [" + openxr_api->get_error_string(result) + "]");
	}

	tracker.instantiate();
	tracker->set_entity(se_extension->add_spatial_entity(sc, entity_id, entity));
	tracker->set_tracker_desc("Anchor");
	tracker->set_pose(SNAME("default"), p_transform, Vector3(), Vector3());

	// Remember our tracker
	anchors[entity_id] = tracker;
	xr_server->add_tracker(tracker);

	return tracker;
}

void OpenXRSpatialAnchorCapability::remove_anchor(Ref<OpenXRAnchorTracker> p_anchor_tracker) {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	// We check for this here. We could do this asynchronous but the caller may than wrongly expect this method to be instant.
	ERR_FAIL_COND_MSG(p_anchor_tracker->has_uuid(), "This anchor is persistent. It must first be made unpersistent.");

	// Attempt to unregister it from our xr_server.
	xr_server->remove_tracker(p_anchor_tracker);

	// Get our entity
	RID entity = p_anchor_tracker->get_entity();
	ERR_FAIL_COND(entity.is_null());

	// Get our entity id.
	XrSpatialEntityIdEXT entity_id = se_extension->get_spatial_entity_id(entity);
	ERR_FAIL_COND(entity_id == XR_NULL_ENTITY);

	// Remove it from our entity list
	if (anchors.has(entity_id)) {
		anchors.erase(entity_id);
	}

	// The rest will be freed up once all references to the anchor go out of scope.
}

bool OpenXRSpatialAnchorCapability::make_anchor_persistent(Ref<OpenXRAnchorTracker> p_anchor_tracker, RID p_persistence_context, Callable p_callback) {
	ERR_FAIL_COND_V(!is_spatial_persistence_supported(), false);

	RID pc = p_persistence_context.is_valid() ? p_persistence_context : persistence_context;
	ERR_FAIL_COND_V(pc.is_null(), false);
	ERR_FAIL_COND_V(p_anchor_tracker.is_null(), false);
	PersistenceContextData *persistence_context_data = persistence_context_owner.get_or_null(pc);
	ERR_FAIL_NULL_V(persistence_context_data, false);
	const XrSpatialPersistenceContextEXT persistence_context_handle = persistence_context_data->persistence_context;

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, false);
	OpenXRFutureExtension *future_api = OpenXRFutureExtension::get_singleton();
	ERR_FAIL_NULL_V(future_api, false);
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, false);

	RID entity = p_anchor_tracker->get_entity();
	ERR_FAIL_COND_V(entity.is_null(), false);

	XrSpatialEntityIdEXT entity_id = se_extension->get_spatial_entity_id(entity);
	ERR_FAIL_COND_V(entity_id == XR_NULL_ENTITY, false);

	RID spatial_context_rid = se_extension->get_spatial_entity_context(entity);
	const XrSpatialContextEXT spatial_context_handle = se_extension->get_spatial_context_handle(spatial_context_rid);

	XrFutureEXT future = XR_NULL_HANDLE;

	XrSpatialEntityPersistInfoEXT persist_info = {
		XR_TYPE_SPATIAL_ENTITY_PERSIST_INFO_EXT, // type
		nullptr, // next
		spatial_context_handle, // spatialContext
		entity_id // entityId
	};
	XrResult result = xrPersistSpatialEntityAsyncEXT(persistence_context_handle, &persist_info, &future);
	if (XR_FAILED(result)) {
		ERR_FAIL_V_MSG(false, "OpenXR: Failed to start making anchor persistent [" + openxr_api->get_error_string(result) + "]");
	}

	// Register our future to trigger our callback.
	future_api->register_future(future, callable_mp(this, &OpenXRSpatialAnchorCapability::_on_made_anchor_persistent).bind(p_callback).bind(p_anchor_tracker).bind(pc));

	return true;
}

void OpenXRSpatialAnchorCapability::_on_made_anchor_persistent(Ref<OpenXRFutureResult> p_future, RID p_persistence_context, Ref<OpenXRAnchorTracker> p_anchor_tracker, Callable p_callback) {
	ERR_FAIL_COND(p_anchor_tracker.is_null());
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	PersistenceContextData *persistence_context_data = persistence_context_owner.get_or_null(p_persistence_context);
	ERR_FAIL_NULL(persistence_context_data);

	XrFutureEXT future = p_future->get_future();

	XrPersistSpatialEntityCompletionEXT completion = {
		XR_TYPE_PERSIST_SPATIAL_ENTITY_COMPLETION_EXT, // type
		nullptr, // next
		XR_RESULT_MAX_ENUM, // futureResult
		XR_SPATIAL_PERSISTENCE_CONTEXT_RESULT_MAX_ENUM_EXT, // persistResult
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } // persistUuid
	};
	XrResult result = xrPersistSpatialEntityCompleteEXT(persistence_context_data->persistence_context, future, &completion);
	if (XR_FAILED(result)) { // Did our xrCreateSpatialContextCompleteEXT call fail?
		// Signal our failure.
		emit_signal(SNAME("anchor_persistent_failed"), p_anchor_tracker);

		// And log issue.
		ERR_FAIL_MSG("OpenXR: Failed to complete anchor persistent future [" + openxr_api->get_error_string(result) + "]");
	}
	if (XR_FAILED(completion.futureResult)) { // Did our completion fail?
		// Signal our failure.
		emit_signal(SNAME("anchor_persistent_failed"), p_anchor_tracker);

		// And log issue.
		ERR_FAIL_MSG("OpenXR: Failed to complete making anchor persistent [" + openxr_api->get_error_string(completion.futureResult) + "]");
	}
	if (completion.persistResult != XR_SPATIAL_PERSISTENCE_CONTEXT_RESULT_SUCCESS_EXT) { // Did our process fail?
		// Signal our failure.
		emit_signal(SNAME("anchor_persistent_failed"), p_anchor_tracker);

		// And log issue.
		ERR_FAIL_MSG("OpenXR: Failed to make anchor persistent [" + get_spatial_persistence_context_result_name(completion.persistResult) + "]");
	}

	// Set our new UUID
	p_anchor_tracker->set_uuid(completion.persistUuid);

	// Do our callback
	p_callback.call(p_anchor_tracker);
}

bool OpenXRSpatialAnchorCapability::make_anchor_unpersistent(Ref<OpenXRAnchorTracker> p_anchor_tracker, RID p_persistence_context, Callable p_callback) {
	ERR_FAIL_COND_V(!is_spatial_persistence_supported(), false);

	RID pc = p_persistence_context.is_valid() ? p_persistence_context : persistence_context;
	ERR_FAIL_COND_V(pc.is_null(), false);
	ERR_FAIL_COND_V(p_anchor_tracker.is_null(), false);
	PersistenceContextData *persistence_context_data = persistence_context_owner.get_or_null(pc);
	ERR_FAIL_NULL_V(persistence_context_data, false);

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, false);
	OpenXRFutureExtension *future_api = OpenXRFutureExtension::get_singleton();
	ERR_FAIL_NULL_V(future_api, false);
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, false);

	XrFutureEXT future;

	XrSpatialEntityUnpersistInfoEXT unpersist_info = {
		XR_TYPE_SPATIAL_ENTITY_PERSIST_INFO_EXT, // type
		nullptr, // next
		p_anchor_tracker->get_uuid() // persistUuid
	};
	XrResult result = xrUnpersistSpatialEntityAsyncEXT(persistence_context_data->persistence_context, &unpersist_info, &future);
	if (XR_FAILED(result)) {
		ERR_FAIL_V_MSG(false, "OpenXR: Failed to make anchor unpersistent [" + openxr_api->get_error_string(result) + "]");
	}

	// Register our future to trigger our callback.
	future_api->register_future(future, callable_mp(this, &OpenXRSpatialAnchorCapability::_on_made_anchor_unpersistent).bind(p_callback).bind(p_anchor_tracker).bind(pc));

	return true;
}

void OpenXRSpatialAnchorCapability::_on_made_anchor_unpersistent(Ref<OpenXRFutureResult> p_future, RID p_persistence_context, Ref<OpenXRAnchorTracker> p_anchor_tracker, Callable p_callback) {
	ERR_FAIL_COND(p_anchor_tracker.is_null());
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);
	PersistenceContextData *persistence_context_data = persistence_context_owner.get_or_null(p_persistence_context);
	ERR_FAIL_NULL(persistence_context_data);

	XrFutureEXT future = p_future->get_future();

	XrUnpersistSpatialEntityCompletionEXT completion = {
		XR_TYPE_UNPERSIST_SPATIAL_ENTITY_COMPLETION_EXT, // type
		nullptr, // next
		XR_RESULT_MAX_ENUM, // futureResult
		XR_SPATIAL_PERSISTENCE_CONTEXT_RESULT_MAX_ENUM_EXT, // unpersistResult
	};
	XrResult result = xrUnpersistSpatialEntityCompleteEXT(persistence_context_data->persistence_context, future, &completion);
	if (XR_FAILED(result)) { // Did our xrCreateSpatialContextCompleteEXT call fail?
		// Signal our failure.
		emit_signal(SNAME("anchor_unpersistent_failed"), p_anchor_tracker);

		// And log issue.
		ERR_FAIL_MSG("OpenXR: Failed to complete anchor unpersistent future [" + openxr_api->get_error_string(result) + "]");
	}
	if (XR_FAILED(completion.futureResult)) { // Did our completion fail?
		// Signal our failure.
		emit_signal(SNAME("anchor_unpersistent_failed"), p_anchor_tracker);

		// And log issue.
		ERR_FAIL_MSG("OpenXR: Failed to complete making anchor unpersistent [" + openxr_api->get_error_string(completion.futureResult) + "]");
	}
	if (completion.unpersistResult != XR_SPATIAL_PERSISTENCE_CONTEXT_RESULT_SUCCESS_EXT) { // Did our process fail?
		// Signal our failure.
		emit_signal(SNAME("anchor_unpersistent_failed"), p_anchor_tracker);

		// And log issue.
		ERR_FAIL_MSG("OpenXR: Failed to make anchor unpersistent [" + get_spatial_persistence_context_result_name(completion.unpersistResult) + "]");
	}

	// Unset our UUID
	XrUuid empty_uid = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	p_anchor_tracker->set_uuid(empty_uid);

	// Do our callback
	p_callback.call(p_anchor_tracker);
}

String OpenXRSpatialAnchorCapability::get_spatial_persistence_scope_name(XrSpatialPersistenceScopeEXT p_scope){
	XR_ENUM_SWITCH(XrSpatialPersistenceScopeEXT, p_scope)
}

String OpenXRSpatialAnchorCapability::get_spatial_persistence_context_result_name(XrSpatialPersistenceContextResultEXT p_result) {
	XR_ENUM_SWITCH(XrSpatialPersistenceContextResultEXT, p_result)
}

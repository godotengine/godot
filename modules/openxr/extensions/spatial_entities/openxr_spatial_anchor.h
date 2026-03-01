/**************************************************************************/
/*  openxr_spatial_anchor.h                                               */
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

#include "../../openxr_util.h"
#include "openxr_spatial_entities.h"
#include "openxr_spatial_entity_extension.h"

// Anchor capability configuration
class OpenXRSpatialCapabilityConfigurationAnchor : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDCLASS(OpenXRSpatialCapabilityConfigurationAnchor, OpenXRSpatialCapabilityConfigurationBaseHeader);

public:
	virtual bool has_valid_configuration() const override;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration() override;

	Vector<XrSpatialComponentTypeEXT> get_enabled_components() const { return anchor_enabled_components; }

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialComponentTypeEXT> anchor_enabled_components;
	XrSpatialCapabilityConfigurationAnchorEXT anchor_config = { XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_ANCHOR_EXT, nullptr, XR_SPATIAL_CAPABILITY_ANCHOR_EXT, 0, nullptr };

	PackedInt64Array _get_enabled_components() const;
};

// Anchor component anchor list
class OpenXRSpatialComponentAnchorList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentAnchorList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	Transform3D get_entity_pose(int64_t p_index) const;

private:
	Vector<XrPosef> entity_poses;

	XrSpatialComponentAnchorListEXT anchor_list = { XR_TYPE_SPATIAL_COMPONENT_ANCHOR_LIST_EXT, nullptr, 0, nullptr };
};

// Persistence configuration
class OpenXRSpatialContextPersistenceConfig : public OpenXRStructureBase {
	GDCLASS(OpenXRSpatialContextPersistenceConfig, OpenXRStructureBase);

public:
	bool has_valid_configuration() const;
	virtual void *get_header(void *p_next) override;
	virtual XrStructureType get_structure_type() override;

	void add_persistence_context(RID p_persistence_context);
	void remove_persistence_context(RID p_persistence_context);

protected:
	static void _bind_methods();

private:
	Vector<RID> persistence_contexts;
	Vector<XrSpatialPersistenceContextEXT> context_handles;

	XrSpatialContextPersistenceConfigEXT persistence_config = { XR_TYPE_SPATIAL_CONTEXT_PERSISTENCE_CONFIG_EXT, nullptr, 0, nullptr };
};

// Component persistence list
class OpenXRSpatialComponentPersistenceList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentPersistenceList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	XrUuid get_persistent_uuid(int64_t p_index) const;
	XrSpatialPersistenceStateEXT get_persistent_state(int64_t p_index) const;

	static String get_persistence_state_name(XrSpatialPersistenceStateEXT p_state);

private:
	Vector<XrSpatialPersistenceDataEXT> persist_data;

	XrSpatialComponentPersistenceListEXT persistence_list = { XR_TYPE_SPATIAL_COMPONENT_PERSISTENCE_LIST_EXT, nullptr, 0, nullptr };

	String _get_persistent_uuid(int64_t p_index) const;
	uint64_t _get_persistent_state(int64_t p_index) const;
};

// Anchor tracker, this adds no new logic, it's purely for typing!
class OpenXRAnchorTracker : public OpenXRSpatialEntityTracker {
	GDCLASS(OpenXRAnchorTracker, OpenXRSpatialEntityTracker);

protected:
	static void _bind_methods();

public:
	bool has_uuid() const;
	XrUuid get_uuid() const;
	void set_uuid(const XrUuid &p_uuid);

private:
	XrUuid uuid = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	String _get_uuid() const;
	void _set_uuid(const String &p_uuid);

	bool uuid_is_equal(const XrUuid &p_a, const XrUuid &p_b);
};

// (Persistent) anchor logic
class OpenXRSpatialAnchorCapability : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRSpatialAnchorCapability, OpenXRExtensionWrapper);

public:
	enum PersistenceScope {
		PERSISTENCE_SCOPE_SYSTEM_MANAGED = XR_SPATIAL_PERSISTENCE_SCOPE_SYSTEM_MANAGED_EXT,
		PERSISTENCE_SCOPE_LOCAL_ANCHORS = XR_SPATIAL_PERSISTENCE_SCOPE_LOCAL_ANCHORS_EXT,
	};

	static OpenXRSpatialAnchorCapability *get_singleton();

	OpenXRSpatialAnchorCapability();
	virtual ~OpenXRSpatialAnchorCapability() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_instance_destroyed() override;
	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;

	virtual void on_process() override;

	bool is_spatial_anchor_supported();
	bool is_spatial_persistence_supported();

	// Persistence scopes
	bool is_persistence_scope_supported(XrSpatialPersistenceScopeEXT p_scope);
	Ref<OpenXRFutureResult> create_persistence_context(XrSpatialPersistenceScopeEXT p_scope, const Callable &p_user_callback = Callable());
	XrSpatialPersistenceContextEXT get_persistence_context_handle(RID p_persistence_context) const;
	void free_persistence_context(RID p_persistence_context);

	Ref<OpenXRAnchorTracker> create_new_anchor(const Transform3D &p_transform, RID p_spatial_context = RID());
	void remove_anchor(Ref<OpenXRAnchorTracker> p_anchor_tracker);
	Ref<OpenXRFutureResult> persist_anchor(Ref<OpenXRAnchorTracker> p_anchor_tracker, RID p_persistence_context = RID(), const Callable &p_user_callback = Callable());
	Ref<OpenXRFutureResult> unpersist_anchor(Ref<OpenXRAnchorTracker> p_anchor_tracker, RID p_persistence_context = RID(), const Callable &p_user_callback = Callable());

	static String get_spatial_persistence_scope_name(XrSpatialPersistenceScopeEXT p_scope);
	static String get_spatial_persistence_context_result_name(XrSpatialPersistenceContextResultEXT p_result);

protected:
	static void _bind_methods();

private:
	static OpenXRSpatialAnchorCapability *singleton;

	bool spatial_anchor_ext = false;
	bool spatial_persistence_ext = false;
	bool spatial_persistence_operations_ext = false;
	bool spatial_anchor_supported = false;

	RID spatial_context;
	RID persistence_context;
	bool need_discovery = false;
	int discovery_cooldown = 0;
	Ref<OpenXRFutureResult> discovery_query_result;

	Ref<OpenXRSpatialCapabilityConfigurationAnchor> anchor_configuration;
	Ref<OpenXRSpatialContextPersistenceConfig> persistence_configuration;

	Vector<XrSpatialPersistenceScopeEXT> supported_persistence_scopes;
	bool _load_supported_persistence_scopes();

	// Persistence scopes
	struct PersistenceContextData {
		XrSpatialPersistenceScopeEXT scope;
		XrSpatialPersistenceContextEXT persistence_context = XR_NULL_HANDLE;
	};
	mutable RID_Owner<PersistenceContextData> persistence_context_owner;

	bool _is_persistence_scope_supported(PersistenceScope p_scope);
	Ref<OpenXRFutureResult> _create_persistence_context(PersistenceScope p_scope, Callable p_user_callback = Callable());

	uint64_t _get_persistence_context_handle(RID p_persistence_context) const;
	void _on_persistence_context_ready(Ref<OpenXRFutureResult> p_future_result, uint64_t p_scope, Callable p_user_callback = Callable());

	// Discovery logic
	void _on_persistence_context_completed(RID p_persistence_context);

	Ref<OpenXRFutureResult> _create_spatial_context();
	void _on_spatial_context_created(RID p_spatial_context);

	void _on_spatial_discovery_recommended(RID p_spatial_context);

	Ref<OpenXRFutureResult> _start_entity_discovery();
	void _process_discovery_snapshot(RID p_snapshot);
	void _process_update_snapshot(RID p_snapshot);

	// Entities
	void _on_made_anchor_persistent(Ref<OpenXRFutureResult> p_future_result, RID p_persistence_context, Ref<OpenXRAnchorTracker> p_anchor_tracker, const Callable &p_callback);
	void _on_made_anchor_unpersistent(Ref<OpenXRFutureResult> p_future_result, RID p_persistence_context, Ref<OpenXRAnchorTracker> p_anchor_tracker, const Callable &p_callback);

	// Trackers
	HashMap<XrSpatialEntityIdEXT, Ref<OpenXRAnchorTracker>> anchors;

	// OpenXR API call wrappers
	EXT_PROTO_XRRESULT_FUNC4(xrCreateSpatialAnchorEXT, (XrSpatialContextEXT), spatialContext, (const XrSpatialAnchorCreateInfoEXT *), create_info, (XrSpatialEntityIdEXT *), anchor_entity_id, (XrSpatialEntityEXT *), anchor_entity);

	EXT_PROTO_XRRESULT_FUNC5(xrEnumerateSpatialPersistenceScopesEXT, (XrInstance), instance, (XrSystemId), system_id, (uint32_t), persistence_scope_capacity_input, (uint32_t *), persistence_scope_count_output, (XrSpatialPersistenceScopeEXT *), persistence_scopes);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialPersistenceContextAsyncEXT, (XrSession), session, (const XrSpatialPersistenceContextCreateInfoEXT *), create_info, (XrFutureEXT *), future);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialPersistenceContextCompleteEXT, (XrSession), session, (XrFutureEXT), future, (XrCreateSpatialPersistenceContextCompletionEXT *), completion);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroySpatialPersistenceContextEXT, (XrSpatialPersistenceContextEXT), persistence_context);

	EXT_PROTO_XRRESULT_FUNC3(xrPersistSpatialEntityAsyncEXT, (XrSpatialPersistenceContextEXT), persistence_context, (const XrSpatialEntityPersistInfoEXT *), persist_info, (XrFutureEXT *), future);
	EXT_PROTO_XRRESULT_FUNC3(xrPersistSpatialEntityCompleteEXT, (XrSpatialPersistenceContextEXT), persistence_context, (XrFutureEXT), future, (XrPersistSpatialEntityCompletionEXT *), completion);
	EXT_PROTO_XRRESULT_FUNC3(xrUnpersistSpatialEntityAsyncEXT, (XrSpatialPersistenceContextEXT), persistence_context, (const XrSpatialEntityUnpersistInfoEXT *), unpersist_info, (XrFutureEXT *), future);
	EXT_PROTO_XRRESULT_FUNC3(xrUnpersistSpatialEntityCompleteEXT, (XrSpatialPersistenceContextEXT), persistence_context, (XrFutureEXT), future, (XrUnpersistSpatialEntityCompletionEXT *), completion);
};

VARIANT_ENUM_CAST(OpenXRSpatialAnchorCapability::PersistenceScope);

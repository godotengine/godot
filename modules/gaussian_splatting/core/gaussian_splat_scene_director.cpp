#include "gaussian_splat_scene_director.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "../logger/gs_logger.h"
#include "../logger/gs_debug_trace.h"
#include "gaussian_splat_manager.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../renderer/render_debug_state_orchestrator.h"
#include "scene/3d/node_3d.h"

static bool _is_scene_director_log_enabled() {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	if (!ps) {
		return false;
	}
	if (ps->get_setting("rendering/gaussian_splatting/debug/enable_all_debug", false)) {
		return true;
	}
	if (ps->get_setting("rendering/gaussian_splatting/debug/enable_frame_logging", false)) {
		return true;
	}
	return ps->get_setting("rendering/gaussian_splatting/debug/enable_data_logging", false);
}

static void _bump_instance_generation(uint64_t &r_generation) {
	r_generation++;
	if (r_generation == 0) {
		r_generation = 1;
	}
}

static bool _dict_get_bool(const Dictionary &p_dict, const StringName &p_key, bool p_default) {
	if (!p_dict.has(p_key)) {
		return p_default;
	}
	const Variant value = p_dict[p_key];
	if (value.get_type() == Variant::BOOL) {
		return (bool)value;
	}
	if (value.get_type() == Variant::INT) {
		return int64_t(value) != 0;
	}
	return p_default;
}

static int _dict_get_int(const Dictionary &p_dict, const StringName &p_key, int p_default) {
	if (!p_dict.has(p_key)) {
		return p_default;
	}
	const Variant value = p_dict[p_key];
	if (value.get_type() == Variant::FLOAT) {
		return int((double)value);
	}
	return int(value);
}

static float _dict_get_float(const Dictionary &p_dict, const StringName &p_key, float p_default) {
	if (!p_dict.has(p_key)) {
		return p_default;
	}
	const Variant value = p_dict[p_key];
	if (value.get_type() == Variant::INT) {
		return (float)int64_t(value);
	}
	return (float)(double)value;
}

static String _dict_get_string(const Dictionary &p_dict, const StringName &p_key, const String &p_default = String()) {
	if (!p_dict.has(p_key)) {
		return p_default;
	}
	return String(p_dict[p_key]);
}

static Dictionary _dict_get_dictionary(const Dictionary &p_dict, const StringName &p_key) {
	if (!p_dict.has(p_key)) {
		return Dictionary();
	}
	const Variant value = p_dict[p_key];
	return value.get_type() == Variant::DICTIONARY ? Dictionary(value) : Dictionary();
}

static const StringName WORLD_OVERRIDE_LOD_ENABLED("lod_enabled");
static const StringName WORLD_OVERRIDE_LOD_BIAS("lod_bias");
static const StringName WORLD_OVERRIDE_LOD_MAX_DISTANCE("lod_max_distance");
static const StringName WORLD_OVERRIDE_MAX_SPLATS("max_splats");
static const StringName WORLD_OVERRIDE_FRUSTUM_CULLING("frustum_culling");
static const StringName WORLD_OVERRIDE_ASYNC_UPLOAD_ENABLED("async_upload_enabled");
static const StringName WORLD_OVERRIDE_OPACITY_MULTIPLIER("opacity_multiplier");
static const StringName WORLD_OVERRIDE_STREAMING("streaming");

static const StringName WORLD_STREAMING_OVERRIDE_PREFETCH("override_prefetch");
static const StringName WORLD_STREAMING_PREDICTIVE_PREFETCH_ENABLED("predictive_prefetch_enabled");
static const StringName WORLD_STREAMING_PREFETCH_LOOKAHEAD_DISTANCE("prefetch_lookahead_distance");
static const StringName WORLD_STREAMING_OVERRIDE_VRAM_BUDGET("override_vram_budget");
static const StringName WORLD_STREAMING_VRAM_BUDGET_MB("vram_budget_mb");
static const StringName WORLD_STREAMING_VRAM_MIN_CHUNKS("vram_min_chunks");
static const StringName WORLD_STREAMING_VRAM_MAX_CHUNKS("vram_max_chunks");
static const StringName WORLD_STREAMING_OVERRIDE_IO_SOURCE("override_io_source");
static const StringName WORLD_STREAMING_IO_SOURCE_PATH("io_source_path");

struct WorldSubmissionRendererConfigSnapshot {
	bool valid = false;
	bool lod_enabled = true;
	float lod_bias = 1.0f;
	float lod_max_distance = 0.0f;
	bool frustum_culling = true;
	bool async_upload_enabled = true;
	float opacity_multiplier = 1.0f;
	int max_splats = 1000000;
	GaussianStreamingTypes::ConfigOverrides streaming_overrides;
};

static WorldSubmissionRendererConfigSnapshot _snapshot_world_submission_renderer_config(const Ref<GaussianSplatRenderer> &p_renderer) {
	WorldSubmissionRendererConfigSnapshot snapshot;
	if (p_renderer.is_null()) {
		return snapshot;
	}

	snapshot.valid = true;
	snapshot.lod_enabled = p_renderer->get_lod_enabled();
	snapshot.lod_bias = p_renderer->get_lod_bias();
	snapshot.lod_max_distance = p_renderer->get_lod_max_distance();
	snapshot.frustum_culling = p_renderer->get_frustum_culling();
	snapshot.async_upload_enabled = p_renderer->get_async_upload_enabled();
	snapshot.opacity_multiplier = p_renderer->get_opacity_multiplier();
	snapshot.max_splats = p_renderer->get_max_splats();
	snapshot.streaming_overrides = p_renderer->get_streaming_config_overrides();
	return snapshot;
}

static void _restore_world_submission_renderer_config(const Ref<GaussianSplatRenderer> &p_renderer,
		const WorldSubmissionRendererConfigSnapshot &p_snapshot) {
	if (p_renderer.is_null() || !p_snapshot.valid) {
		return;
	}

	p_renderer->set_lod_enabled(p_snapshot.lod_enabled);
	p_renderer->set_lod_bias(p_snapshot.lod_bias);
	p_renderer->set_lod_max_distance(p_snapshot.lod_max_distance);
	p_renderer->set_frustum_culling(p_snapshot.frustum_culling);
	p_renderer->set_async_upload_enabled(p_snapshot.async_upload_enabled);
	p_renderer->set_opacity_multiplier(p_snapshot.opacity_multiplier);
	p_renderer->set_max_splats(p_snapshot.max_splats);
	p_renderer->set_streaming_config_overrides(p_snapshot.streaming_overrides);
}

GaussianSplatSceneDirector *GaussianSplatSceneDirector::singleton = nullptr;

GaussianSplatSceneDirector *GaussianSplatSceneDirector::get_singleton() {
    return singleton;
}

GaussianSplatSceneDirector::GaussianSplatSceneDirector() {
    if (!singleton) {
        singleton = this;
    }
}

GaussianSplatSceneDirector::~GaussianSplatSceneDirector() {
    // Release all SharedWorld entries so their Ref<GaussianSplatRenderer>
    // instances are unreferenced, allowing GPU resources (compute/shader
    // RIDs, buffers) to be freed.  Without this, each F6 runtime cycle
    // leaks an entire renderer's worth of GPU allocations.
    worlds.clear();
    if (singleton == this) {
        singleton = nullptr;
    }
}

void GaussianSplatSceneDirector::_bind_methods() {
}

GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_get_or_create_world_for_scenario(const RID &p_scenario) {
	if (!p_scenario.is_valid()) {
		return nullptr;
	}

	SharedWorld *entry = worlds.getptr(p_scenario);
	if (!entry) {
		SharedWorld world;
		world.scenario = p_scenario;
		worlds.insert(p_scenario, world);
		entry = worlds.getptr(p_scenario);
	}

	if (entry && !entry->renderer.is_valid()) {
		GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
		RenderingDevice *device = manager ? manager->get_primary_rendering_device() : nullptr;
		if (!device) {
			static bool warned_missing_device = false;
			if (!warned_missing_device) {
				warned_missing_device = true;
				GS_LOG_RENDERER_ERROR(
						"[GaussianSplatSceneDirector] Unable to acquire primary RenderingDevice for shared renderer (scenario=" +
						String::num_uint64((uint64_t)p_scenario.get_id()) +
						"). Gaussian splat instances in this world will be collected but skipped because no renderer can be attached.");
			}
			return entry;
		}

		entry->renderer = Ref<GaussianSplatRenderer>(memnew(GaussianSplatRenderer(device)));
		if (_is_scene_director_log_enabled()) {
			GS_LOG_RENDERER_DEBUG("[SceneDirector] Created shared renderer (deferred initialization)");
		}
		if (entry->world_submission.active) {
			_apply_world_submission_to_renderer(*entry, entry->world_submission);
		}
	}

	return entry;
}

GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_get_or_create_world(World3D *p_world) {
	ERR_FAIL_NULL_V(p_world, nullptr);
	return _get_or_create_world_for_scenario(p_world->get_scenario());
}

GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_get_world_for_instance(ObjectID p_node_id) {
	Object *obj = ObjectDB::get_instance(p_node_id);
	Node3D *node = Object::cast_to<Node3D>(obj);
	if (!node) {
		return nullptr;
	}
	if (!node->is_inside_world()) {
		return nullptr;
	}
	World3D *world = node->get_world_3d().ptr();
	if (!world) {
		return nullptr;
	}
	return _get_or_create_world(world);
}

GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_find_world_for_instance(ObjectID p_node_id) {
	for (KeyValue<RID, SharedWorld> &E : worlds) {
		if (E.value.instance_lookup.has(p_node_id)) {
			return &E.value;
		}
	}
	return nullptr;
}

GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_find_world_for_renderer(const GaussianSplatRenderer *p_renderer) {
	if (!p_renderer) {
		return nullptr;
	}
	for (KeyValue<RID, SharedWorld> &E : worlds) {
		if (E.value.renderer.ptr() == p_renderer) {
			return &E.value;
		}
	}
	if (GaussianSplatting::debug_trace_is_enabled()) {
		GaussianSplatting::debug_trace_record_event("world_lookup",
				vformat("renderer=%d not found (worlds=%d)",
						(int64_t)(uintptr_t)p_renderer, (int)worlds.size()),
				true);
	}
	return nullptr;
}

const GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_find_world_for_renderer(const GaussianSplatRenderer *p_renderer) const {
	if (!p_renderer) {
		GaussianSplatting::debug_trace_record_event("world_lookup", "renderer=NULL", true);
		return nullptr;
	}
	for (const KeyValue<RID, SharedWorld> &E : worlds) {
		if (E.value.renderer.ptr() == p_renderer) {
			return &E.value;
		}
	}
	if (GaussianSplatting::debug_trace_is_enabled()) {
		GaussianSplatting::debug_trace_record_event("world_lookup",
				vformat("renderer=%d not found (worlds=%d)",
						(int64_t)(uintptr_t)p_renderer, (int)worlds.size()),
				true);
	}
	return nullptr;
}

GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_find_world_for_world_submission(ObjectID p_owner_id) {
	if (p_owner_id == ObjectID()) {
		return nullptr;
	}
	for (KeyValue<RID, SharedWorld> &E : worlds) {
		if (E.value.world_submission.active && E.value.world_submission.owner_id == p_owner_id) {
			return &E.value;
		}
	}
	return nullptr;
}

const GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_find_world_for_world_submission(ObjectID p_owner_id) const {
	if (p_owner_id == ObjectID()) {
		return nullptr;
	}
	for (const KeyValue<RID, SharedWorld> &E : worlds) {
		if (E.value.world_submission.active && E.value.world_submission.owner_id == p_owner_id) {
			return &E.value;
		}
	}
	return nullptr;
}

GaussianSplatSceneDirector::PreviewSubmissionRecord *GaussianSplatSceneDirector::_find_preview_submission_for_renderer(const GaussianSplatRenderer *p_renderer) {
	if (!p_renderer) {
		return nullptr;
	}
	for (KeyValue<ObjectID, PreviewSubmissionRecord> &E : preview_submissions) {
		if (E.value.renderer.ptr() == p_renderer) {
			return &E.value;
		}
	}
	return nullptr;
}

const GaussianSplatSceneDirector::PreviewSubmissionRecord *GaussianSplatSceneDirector::_find_preview_submission_for_renderer(const GaussianSplatRenderer *p_renderer) const {
	if (!p_renderer) {
		return nullptr;
	}
	for (const KeyValue<ObjectID, PreviewSubmissionRecord> &E : preview_submissions) {
		if (E.value.renderer.ptr() == p_renderer) {
			return &E.value;
		}
	}
	return nullptr;
}

bool GaussianSplatSceneDirector::_populate_gaussian_data_from_asset(const Ref<GaussianSplatAsset> &p_asset, Ref<GaussianData> &r_data) {
	if (p_asset.is_null()) {
		return false;
	}

	if (p_asset->get_asset_type() == GaussianSplatAsset::ASSET_TYPE_DYNAMIC) {
		return p_asset->populate_gaussian_data(r_data);
	}

	Ref<GaussianData> shared_data = p_asset->get_gaussian_data();
	if (shared_data.is_null()) {
		return false;
	}
	r_data = shared_data;
	return true;
}

bool GaussianSplatSceneDirector::_retain_asset_record(SharedWorld &p_world, const Ref<GaussianSplatAsset> &p_asset, uint32_t p_asset_id) {
	if (p_asset.is_null()) {
		return false;
	}
	uint32_t edited_version = 0;
#ifdef TOOLS_ENABLED
	edited_version = p_asset->get_edited_version();
#endif
	SharedWorld::AssetRecord *record = p_world.asset_records.getptr(p_asset_id);
	if (!record) {
		SharedWorld::AssetRecord new_record;
		new_record.asset = p_asset;
		if (!_populate_gaussian_data_from_asset(p_asset, new_record.data)) {
			GS_LOG_WARN_DEFAULT("[GaussianSplatSceneDirector] Failed to build GaussianData from asset.");
			return false;
		}
		new_record.edited_version = edited_version;
		new_record.refcount = 1;
		p_world.asset_records.insert(p_asset_id, new_record);
		_bump_instance_generation(p_world.instance_generation);
		return true;
	}

	record->asset = p_asset;
	if (record->data.is_null() || record->edited_version != edited_version) {
		Ref<GaussianData> refreshed_data;
		if (!_populate_gaussian_data_from_asset(p_asset, refreshed_data)) {
			GS_LOG_WARN_DEFAULT("[GaussianSplatSceneDirector] Failed to rebuild GaussianData from asset.");
			return false;
		}
		record->data = refreshed_data;
		record->edited_version = edited_version;
		_bump_instance_generation(p_world.instance_generation);
	}
	record->refcount++;
	return true;
}

bool GaussianSplatSceneDirector::_refresh_asset_record(SharedWorld &p_world, const Ref<GaussianSplatAsset> &p_asset, uint32_t p_asset_id) {
	if (p_asset.is_null()) {
		return false;
	}
	SharedWorld::AssetRecord *record = p_world.asset_records.getptr(p_asset_id);
	if (!record) {
		return false;
	}
	uint32_t edited_version = 0;
#ifdef TOOLS_ENABLED
	edited_version = p_asset->get_edited_version();
#endif
	if (!record->data.is_null() && record->edited_version == edited_version) {
		return true;
	}
	Ref<GaussianData> refreshed_data;
	if (!_populate_gaussian_data_from_asset(p_asset, refreshed_data)) {
		GS_LOG_WARN_DEFAULT("[GaussianSplatSceneDirector] Failed to refresh GaussianData from asset.");
		return false;
	}
	record->asset = p_asset;
	record->data = refreshed_data;
	record->edited_version = edited_version;
	_bump_instance_generation(p_world.instance_generation);
	return true;
}

void GaussianSplatSceneDirector::_release_asset_record(SharedWorld &p_world, uint32_t p_asset_id) {
	SharedWorld::AssetRecord *record = p_world.asset_records.getptr(p_asset_id);
	if (!record) {
		return;
	}
	if (record->refcount > 0) {
		record->refcount--;
	}
	if (record->refcount == 0) {
		p_world.asset_records.erase(p_asset_id);
	}
}

bool GaussianSplatSceneDirector::_is_world_submission_owner_live(ObjectID p_owner_id) {
	if (p_owner_id == ObjectID()) {
		return false;
	}
	return ObjectDB::get_instance(p_owner_id) != nullptr;
}

void GaussianSplatSceneDirector::_store_world_submission_record(SharedWorld::WorldSubmissionRecord &r_record,
		const WorldSubmission &p_submission) {
	r_record.owner_id = p_submission.owner_id;
	r_record.gaussian_data = p_submission.gaussian_data;
	r_record.static_chunks = p_submission.static_chunks;
	r_record.bounds = p_submission.bounds;
	r_record.metadata = p_submission.metadata;
	r_record.has_desired_residency_hint = p_submission.has_desired_residency_hint;
	r_record.desired_residency_hint = p_submission.desired_residency_hint;
	r_record.desired_renderer_overrides = p_submission.desired_renderer_overrides;
	r_record.active = true;
}

void GaussianSplatSceneDirector::_copy_world_submission_record(const SharedWorld &p_world,
		const SharedWorld::WorldSubmissionRecord &p_record, WorldSubmission *r_submission) {
	ERR_FAIL_NULL(r_submission);

	r_submission->owner_id = p_record.owner_id;
	r_submission->scenario = p_world.scenario;
	r_submission->gaussian_data = p_record.gaussian_data;
	r_submission->static_chunks = p_record.static_chunks;
	r_submission->bounds = p_record.bounds;
	r_submission->metadata = p_record.metadata;
	r_submission->has_desired_residency_hint = p_record.has_desired_residency_hint;
	r_submission->desired_residency_hint = p_record.desired_residency_hint;
	r_submission->desired_renderer_overrides = p_record.desired_renderer_overrides;
}

void GaussianSplatSceneDirector::_clear_world_submission_renderer(SharedWorld &p_world) {
	if (!p_world.renderer.is_valid()) {
		return;
	}
	p_world.renderer->set_gaussian_data(Ref<GaussianData>());
	p_world.renderer->clear_static_chunks();
}

bool GaussianSplatSceneDirector::_apply_world_submission_to_renderer(SharedWorld &p_world,
		const SharedWorld::WorldSubmissionRecord &p_record) {
	if (!p_record.active || !p_world.renderer.is_valid()) {
		return true;
	}

	GaussianSplatRenderer *renderer = p_world.renderer.ptr();
	ERR_FAIL_NULL_V(renderer, false);

	const Dictionary &overrides = p_record.desired_renderer_overrides;
	renderer->set_lod_enabled(_dict_get_bool(overrides, WORLD_OVERRIDE_LOD_ENABLED, renderer->get_lod_enabled()));
	renderer->set_lod_bias(_dict_get_float(overrides, WORLD_OVERRIDE_LOD_BIAS, renderer->get_lod_bias()));
	renderer->set_lod_max_distance(_dict_get_float(overrides, WORLD_OVERRIDE_LOD_MAX_DISTANCE, renderer->get_lod_max_distance()));
	renderer->set_frustum_culling(_dict_get_bool(overrides, WORLD_OVERRIDE_FRUSTUM_CULLING, renderer->get_frustum_culling()));
	renderer->set_async_upload_enabled(_dict_get_bool(overrides, WORLD_OVERRIDE_ASYNC_UPLOAD_ENABLED, renderer->get_async_upload_enabled()));
	renderer->set_opacity_multiplier(_dict_get_float(overrides, WORLD_OVERRIDE_OPACITY_MULTIPLIER, renderer->get_opacity_multiplier()));

	if (overrides.has(WORLD_OVERRIDE_STREAMING)) {
		GaussianStreamingTypes::ConfigOverrides streaming_overrides;
		const Dictionary streaming_dict = _dict_get_dictionary(overrides, WORLD_OVERRIDE_STREAMING);
		streaming_overrides.override_prefetch = _dict_get_bool(streaming_dict, WORLD_STREAMING_OVERRIDE_PREFETCH, false);
		streaming_overrides.predictive_prefetch_enabled = _dict_get_bool(streaming_dict, WORLD_STREAMING_PREDICTIVE_PREFETCH_ENABLED,
				streaming_overrides.predictive_prefetch_enabled);
		streaming_overrides.prefetch_lookahead_distance = _dict_get_float(streaming_dict, WORLD_STREAMING_PREFETCH_LOOKAHEAD_DISTANCE,
				streaming_overrides.prefetch_lookahead_distance);
		streaming_overrides.override_vram_budget = _dict_get_bool(streaming_dict, WORLD_STREAMING_OVERRIDE_VRAM_BUDGET, false);
		streaming_overrides.vram_budget_config.budget_mb = MAX(0, _dict_get_int(streaming_dict, WORLD_STREAMING_VRAM_BUDGET_MB,
				int(streaming_overrides.vram_budget_config.budget_mb)));
		streaming_overrides.vram_budget_config.min_chunks = MAX(0, _dict_get_int(streaming_dict, WORLD_STREAMING_VRAM_MIN_CHUNKS,
				int(streaming_overrides.vram_budget_config.min_chunks)));
		streaming_overrides.vram_budget_config.max_chunks = MAX(0, _dict_get_int(streaming_dict, WORLD_STREAMING_VRAM_MAX_CHUNKS,
				int(streaming_overrides.vram_budget_config.max_chunks)));
		streaming_overrides.override_io_source = _dict_get_bool(streaming_dict, WORLD_STREAMING_OVERRIDE_IO_SOURCE, false);
		streaming_overrides.io_source_path = _dict_get_string(streaming_dict, WORLD_STREAMING_IO_SOURCE_PATH);
		if (streaming_overrides.override_vram_budget) {
			streaming_overrides.vram_budget_config.min_chunks = MIN(streaming_overrides.vram_budget_config.min_chunks,
					streaming_overrides.vram_budget_config.max_chunks);
		}
		renderer->set_streaming_config_overrides(streaming_overrides);
	}

	const uint32_t data_count = p_record.gaussian_data.is_valid() ? p_record.gaussian_data->get_count() : 0;
	const int requested_max_splats = MAX(0, _dict_get_int(overrides, WORLD_OVERRIDE_MAX_SPLATS, renderer->get_max_splats()));
	int effective_max_splats = requested_max_splats;
	if (effective_max_splats > 0 && data_count > 0) {
		effective_max_splats = MIN(effective_max_splats, int(data_count));
	} else if (data_count > 0) {
		effective_max_splats = int(data_count);
	} else {
		effective_max_splats = 1000;
	}
	renderer->set_max_splats(MAX(1000, effective_max_splats));

	const auto &resource_state = renderer->get_resource_state();
	if (!resource_state.gpu_resources_initialized && !resource_state.gpu_initialization_pending) {
		renderer->initialize();
	}

	const Error err = renderer->set_gaussian_data(p_record.gaussian_data);
	if (err != OK) {
		GS_LOG_RENDERER_ERROR(vformat("[GaussianSplatSceneDirector] Failed to apply world submission (err=%d).", err));
		renderer->clear_static_chunks();
		return false;
	}

	if (data_count == 0) {
		const String world_path = _dict_get_string(p_record.metadata, StringName("world_path"));
		if (world_path.is_empty()) {
			WARN_PRINT("[GaussianSplatSceneDirector] World submission has zero splats; renderer will stay disconnected.");
		} else {
			WARN_PRINT(vformat("[GaussianSplatSceneDirector] World submission '%s' has zero splats; renderer will stay disconnected.",
					world_path));
		}
		renderer->clear_static_chunks();
		return true;
	}

	renderer->set_static_chunks(p_record.static_chunks);
	return true;
}


void GaussianSplatSceneDirector::register_instance(ObjectID p_node_id, const Ref<GaussianSplatAsset> &p_asset,
        const Transform3D &p_transform, float p_opacity, float p_lod_bias, uint32_t p_flags, bool p_casts_shadow,
        float p_wind_intensity, uint32_t p_wind_mode, const Vector3 &p_wind_direction, float p_wind_frequency,
        bool p_visible, bool p_has_desired_residency_hint, int32_t p_desired_residency_hint) {
	MutexLock lock(world_mutex);
	SharedWorld *world = _get_world_for_instance(p_node_id);
	if (!world) {
		GaussianSplatting::debug_trace_record_event("instance_reg", "FAIL: world=NULL", true);
		return;
	}
	if (world->renderer.is_valid()) {
		const auto &resource_state = world->renderer->get_resource_state();
		if (!resource_state.gpu_resources_initialized && !resource_state.gpu_initialization_pending) {
			world->renderer->initialize();
		}
	}
	if (p_asset.is_null()) {
		GaussianSplatting::debug_trace_record_event("instance_reg", "FAIL: asset=null", true);
		return;
	}
	const uint64_t asset_id_u64 = p_asset->get_instance_id();
	const uint32_t asset_id = static_cast<uint32_t>(asset_id_u64);
	const float wind_intensity = MAX(0.0f, p_wind_intensity);
	const uint32_t wind_mode = MIN(p_wind_mode, (uint32_t)INSTANCE_WIND_FORCE_ENABLED);
	const float wind_frequency = MAX(0.0f, p_wind_frequency);
	if (asset_id == 0) {
		GaussianSplatting::debug_trace_record_event("instance_reg", "FAIL: asset_id=0", true);
		return;
	}
	GaussianSplatting::debug_trace_record_event("instance_reg",
			vformat("OK: asset_id=%d instances_before=%d", asset_id, world->instances.size()),
			false);

	uint32_t *index_ptr = world->instance_lookup.getptr(p_node_id);
	if (index_ptr && *index_ptr < world->instances.size()) {
		InstanceRecord &record = world->instances[*index_ptr];
		bool dirty = false;
		if (!record.transform.is_equal_approx(p_transform)) {
			record.transform = p_transform;
			dirty = true;
		}
		if (!Math::is_equal_approx(record.opacity, p_opacity)) {
			record.opacity = p_opacity;
			dirty = true;
		}
		if (!Math::is_equal_approx(record.lod_bias, p_lod_bias)) {
			record.lod_bias = p_lod_bias;
			dirty = true;
		}
		if (record.flags != p_flags) {
			record.flags = p_flags;
			dirty = true;
		}
		if (record.casts_shadow != p_casts_shadow) {
			record.casts_shadow = p_casts_shadow;
			dirty = true;
		}
		if (record.visible != p_visible) {
			record.visible = p_visible;
			dirty = true;
		}
		if (!Math::is_equal_approx(record.wind_intensity, wind_intensity)) {
			record.wind_intensity = wind_intensity;
			dirty = true;
		}
		if (record.wind_mode != wind_mode) {
			record.wind_mode = wind_mode;
			dirty = true;
		}
		if (!record.wind_direction.is_equal_approx(p_wind_direction)) {
			record.wind_direction = p_wind_direction;
			dirty = true;
		}
		if (!Math::is_equal_approx(record.wind_frequency, wind_frequency)) {
			record.wind_frequency = wind_frequency;
			dirty = true;
		}
		if (record.has_desired_residency_hint != p_has_desired_residency_hint) {
			record.has_desired_residency_hint = p_has_desired_residency_hint;
			dirty = true;
		}
		if (record.desired_residency_hint != p_desired_residency_hint) {
			record.desired_residency_hint = p_desired_residency_hint;
			dirty = true;
		}
		if (record.asset_id == asset_id) {
			if (world->asset_records.has(asset_id)) {
				if (!_refresh_asset_record(*world, p_asset, asset_id)) {
					return;
				}
			} else {
				if (!_retain_asset_record(*world, p_asset, asset_id)) {
					return;
				}
			}
		}
		if (record.asset_id != asset_id) {
			if (!_retain_asset_record(*world, p_asset, asset_id)) {
				return;
			}
			_release_asset_record(*world, record.asset_id);
			record.asset_id = asset_id;
			record.last_lod = 0;
			dirty = true;
		}
		record.dirty = record.dirty || dirty;
		if (dirty) {
			_bump_instance_generation(world->instance_generation);
		}
		return;
	}

	if (!_retain_asset_record(*world, p_asset, asset_id)) {
		return;
	}

	InstanceRecord record;
	record.node_id = p_node_id;
	record.transform = p_transform;
	record.opacity = p_opacity;
	record.lod_bias = p_lod_bias;
	record.wind_intensity = wind_intensity;
	record.wind_mode = wind_mode;
	record.wind_direction = p_wind_direction;
	record.wind_frequency = wind_frequency;
	record.asset_id = asset_id;
	record.flags = p_flags;
	record.last_lod = 0;
	record.casts_shadow = p_casts_shadow;
	record.visible = p_visible;
	record.has_desired_residency_hint = p_has_desired_residency_hint;
	record.desired_residency_hint = p_desired_residency_hint;
	record.dirty = true;

	world->instance_lookup[p_node_id] = world->instances.size();
	world->instances.push_back(record);
	_bump_instance_generation(world->instance_generation);
	GaussianSplatting::debug_trace_record_event("instance_reg",
			vformat("ADDED: instances_after=%d", world->instances.size()),
			false);
}

void GaussianSplatSceneDirector::update_instance_transform(ObjectID p_node_id, const Transform3D &p_transform) {
	MutexLock lock(world_mutex);
	SharedWorld *world = _get_world_for_instance(p_node_id);
	if (!world) {
		world = _find_world_for_instance(p_node_id);
	}
	if (!world) {
		return;
	}

	uint32_t *index_ptr = world->instance_lookup.getptr(p_node_id);
	if (!index_ptr || *index_ptr >= world->instances.size()) {
		return;
	}

	InstanceRecord &record = world->instances[*index_ptr];
	if (record.transform.is_equal_approx(p_transform)) {
		return;
	}
	record.transform = p_transform;
	record.dirty = true;
	_bump_instance_generation(world->instance_generation);
}

void GaussianSplatSceneDirector::update_instance_params(ObjectID p_node_id, float p_opacity, float p_lod_bias,
		uint32_t p_flags, bool p_casts_shadow, float p_wind_intensity, uint32_t p_wind_mode,
		const Vector3 &p_wind_direction, float p_wind_frequency, bool p_visible,
		bool p_has_desired_residency_hint, int32_t p_desired_residency_hint) {
	MutexLock lock(world_mutex);
	SharedWorld *world = _get_world_for_instance(p_node_id);
	if (!world) {
		world = _find_world_for_instance(p_node_id);
	}
	if (!world) {
		return;
	}

	uint32_t *index_ptr = world->instance_lookup.getptr(p_node_id);
	if (!index_ptr || *index_ptr >= world->instances.size()) {
		return;
	}

	InstanceRecord &record = world->instances[*index_ptr];
	const float wind_intensity = MAX(0.0f, p_wind_intensity);
	const uint32_t wind_mode = MIN(p_wind_mode, (uint32_t)INSTANCE_WIND_FORCE_ENABLED);
	const float wind_frequency = MAX(0.0f, p_wind_frequency);
	bool dirty = false;
	if (!Math::is_equal_approx(record.opacity, p_opacity)) {
		record.opacity = p_opacity;
		dirty = true;
	}
	if (!Math::is_equal_approx(record.lod_bias, p_lod_bias)) {
		record.lod_bias = p_lod_bias;
		dirty = true;
	}
	if (record.flags != p_flags) {
		record.flags = p_flags;
		dirty = true;
	}
	if (record.casts_shadow != p_casts_shadow) {
		record.casts_shadow = p_casts_shadow;
		dirty = true;
	}
	if (record.visible != p_visible) {
		record.visible = p_visible;
		dirty = true;
	}
	if (!Math::is_equal_approx(record.wind_intensity, wind_intensity)) {
		record.wind_intensity = wind_intensity;
		dirty = true;
	}
	if (record.wind_mode != wind_mode) {
		record.wind_mode = wind_mode;
		dirty = true;
	}
	if (!record.wind_direction.is_equal_approx(p_wind_direction)) {
		record.wind_direction = p_wind_direction;
		dirty = true;
	}
	if (!Math::is_equal_approx(record.wind_frequency, wind_frequency)) {
		record.wind_frequency = wind_frequency;
		dirty = true;
	}
	if (record.has_desired_residency_hint != p_has_desired_residency_hint) {
		record.has_desired_residency_hint = p_has_desired_residency_hint;
		dirty = true;
	}
	if (record.desired_residency_hint != p_desired_residency_hint) {
		record.desired_residency_hint = p_desired_residency_hint;
		dirty = true;
	}
	record.dirty = record.dirty || dirty;
	if (dirty) {
		_bump_instance_generation(world->instance_generation);
	}
}

void GaussianSplatSceneDirector::unregister_instance(ObjectID p_node_id) {
	MutexLock lock(world_mutex);
	SharedWorld *world = _get_world_for_instance(p_node_id);
	if (!world) {
		world = _find_world_for_instance(p_node_id);
	}
	if (!world) {
		return;
	}

	uint32_t *index_ptr = world->instance_lookup.getptr(p_node_id);
	if (!index_ptr || *index_ptr >= world->instances.size()) {
		return;
	}

	uint32_t index = *index_ptr;
	const uint32_t asset_id = world->instances[index].asset_id;
	uint32_t last_index = world->instances.size() - 1;
	if (index != last_index) {
		world->instances[index] = world->instances[last_index];
		world->instance_lookup[world->instances[index].node_id] = index;
	}
	world->instances.remove_at(last_index);
	world->instance_lookup.erase(p_node_id);
	_release_asset_record(*world, asset_id);
	_bump_instance_generation(world->instance_generation);

	// Free the SharedWorld (and its renderer's GPU resources) once the
	// last instance leaves.  This prevents GPU resource accumulation
	// across F6 runtime cycles.
	if (world->instances.is_empty()) {
		RID erase_key;
		for (const KeyValue<RID, SharedWorld> &kv : worlds) {
			if (&kv.value == world) {
				erase_key = kv.key;
				break;
			}
		}
		if (erase_key.is_valid()) {
			worlds.erase(erase_key);
		}
	}
}

void GaussianSplatSceneDirector::register_instance_submission(ObjectID p_node_id, const Ref<GaussianSplatAsset> &p_asset,
		const Transform3D &p_transform, float p_opacity, float p_lod_bias, uint32_t p_flags, bool p_casts_shadow,
		float p_wind_intensity, uint32_t p_wind_mode, const Vector3 &p_wind_direction, float p_wind_frequency,
		bool p_visible, bool p_has_desired_residency_hint, int32_t p_desired_residency_hint) {
	register_instance(p_node_id, p_asset, p_transform, p_opacity, p_lod_bias, p_flags, p_casts_shadow,
			p_wind_intensity, p_wind_mode, p_wind_direction, p_wind_frequency, p_visible,
			p_has_desired_residency_hint, p_desired_residency_hint);
}

void GaussianSplatSceneDirector::update_instance_submission_transform(ObjectID p_node_id, const Transform3D &p_transform) {
	update_instance_transform(p_node_id, p_transform);
}

void GaussianSplatSceneDirector::update_instance_submission_params(ObjectID p_node_id, float p_opacity, float p_lod_bias,
		uint32_t p_flags, bool p_casts_shadow, float p_wind_intensity, uint32_t p_wind_mode,
		const Vector3 &p_wind_direction, float p_wind_frequency, bool p_visible,
		bool p_has_desired_residency_hint, int32_t p_desired_residency_hint) {
	update_instance_params(p_node_id, p_opacity, p_lod_bias, p_flags, p_casts_shadow, p_wind_intensity,
			p_wind_mode, p_wind_direction, p_wind_frequency, p_visible,
			p_has_desired_residency_hint, p_desired_residency_hint);
}

void GaussianSplatSceneDirector::unregister_instance_submission(ObjectID p_node_id) {
	unregister_instance(p_node_id);
}

bool GaussianSplatSceneDirector::get_instance_submission(ObjectID p_node_id, InstanceSubmission *r_submission) const {
	ERR_FAIL_NULL_V(r_submission, false);

	MutexLock lock(world_mutex);
	for (const KeyValue<RID, SharedWorld> &E : worlds) {
		const SharedWorld &world = E.value;
		const uint32_t *index_ptr = world.instance_lookup.getptr(p_node_id);
		if (!index_ptr || *index_ptr >= world.instances.size()) {
			continue;
		}

		const InstanceRecord &record = world.instances[*index_ptr];
		const SharedWorld::AssetRecord *asset_record = world.asset_records.getptr(record.asset_id);

		r_submission->node_id = record.node_id;
		r_submission->scenario = world.scenario;
		r_submission->renderer = world.renderer;
		r_submission->asset = asset_record ? asset_record->asset : Ref<GaussianSplatAsset>();
		r_submission->transform = record.transform;
		r_submission->opacity = record.opacity;
		r_submission->lod_bias = record.lod_bias;
		r_submission->wind_intensity = record.wind_intensity;
		r_submission->wind_mode = record.wind_mode;
		r_submission->wind_direction = record.wind_direction;
		r_submission->wind_frequency = record.wind_frequency;
		r_submission->flags = record.flags;
		r_submission->last_lod = record.last_lod;
		r_submission->casts_shadow = record.casts_shadow;
		r_submission->visible = record.visible;
		r_submission->has_desired_residency_hint = record.has_desired_residency_hint;
		r_submission->desired_residency_hint = record.desired_residency_hint;
		return true;
	}

	return false;
}

void GaussianSplatSceneDirector::update_instance_lods(const Vector3 &p_camera_pos, const LODConfig &p_lod_config,
		float p_hysteresis_zone) {
	MutexLock lock(world_mutex);
	const int max_lod = MAX(0, p_lod_config.num_levels - 1);
	const bool use_fallback = p_hysteresis_zone <= 0.0f;
	const bool log_enabled = _is_scene_director_log_enabled();

	for (KeyValue<RID, SharedWorld> &E : worlds) {
		SharedWorld &world = E.value;
		if (world.instances.is_empty()) {
			continue;
		}
		bool any_changed = false;
		for (uint32_t i = 0; i < world.instances.size(); i++) {
			InstanceRecord &record = world.instances[i];
			const float distance = p_camera_pos.distance_to(record.transform.origin);
			const float bias = MAX(record.lod_bias, 0.0001f);
			const float effective_distance = distance * bias;
			int desired_lod = p_lod_config.calculate_lod_level(effective_distance);
			desired_lod = CLAMP(desired_lod, 0, max_lod);

			uint32_t current_lod = record.last_lod;
			if (current_lod > static_cast<uint32_t>(max_lod)) {
				current_lod = static_cast<uint32_t>(max_lod);
				record.last_lod = current_lod;
				record.dirty = true;
				any_changed = true;
			}
			if (desired_lod == static_cast<int>(current_lod)) {
				if (log_enabled) {
					GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%s asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (no change)",
							String::num_uint64((uint64_t)record.node_id), record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
				}
				continue;
			}

			if (desired_lod > static_cast<int>(current_lod)) {
				const float threshold = p_lod_config.get_distance_threshold(desired_lod);
				const float zone = use_fallback ? MAX(0.5f, 0.05f * threshold) : p_hysteresis_zone;
				if (effective_distance < threshold + zone) {
					if (log_enabled) {
						GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%s asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (hold-up)",
								String::num_uint64((uint64_t)record.node_id), record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
					}
					continue;
				}
			} else {
				const float threshold = p_lod_config.get_distance_threshold(static_cast<int>(current_lod));
				const float zone = use_fallback ? MAX(0.5f, 0.05f * threshold) : p_hysteresis_zone;
				if (effective_distance > threshold - zone) {
					if (log_enabled) {
						GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%s asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (hold-down)",
								String::num_uint64((uint64_t)record.node_id), record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
					}
					continue;
				}
			}

			record.last_lod = static_cast<uint32_t>(desired_lod);
			record.dirty = true;
			any_changed = true;
			if (log_enabled) {
				GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%s asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u -> %u",
						String::num_uint64((uint64_t)record.node_id), record.asset_id, distance, bias, effective_distance,
						current_lod, record.last_lod));
			}
		}
		if (any_changed) {
			_bump_instance_generation(world.instance_generation);
		}
	}
}

void GaussianSplatSceneDirector::update_instance_lods_for_renderer(const GaussianSplatRenderer *p_renderer,
		const Vector3 &p_camera_pos, const LODConfig &p_lod_config, float p_hysteresis_zone) {
	MutexLock lock(world_mutex);
	SharedWorld *world = _find_world_for_renderer(p_renderer);
	if (!world || world->instances.is_empty()) {
		return;
	}

	const int max_lod = MAX(0, p_lod_config.num_levels - 1);
	const bool use_fallback = p_hysteresis_zone <= 0.0f;
	const bool log_enabled = _is_scene_director_log_enabled();
	bool any_changed = false;

	for (uint32_t i = 0; i < world->instances.size(); i++) {
		InstanceRecord &record = world->instances[i];
		const float distance = p_camera_pos.distance_to(record.transform.origin);
		const float bias = MAX(record.lod_bias, 0.0001f);
		const float effective_distance = distance * bias;
		int desired_lod = p_lod_config.calculate_lod_level(effective_distance);
		desired_lod = CLAMP(desired_lod, 0, max_lod);

		uint32_t current_lod = record.last_lod;
		if (current_lod > static_cast<uint32_t>(max_lod)) {
			current_lod = static_cast<uint32_t>(max_lod);
			record.last_lod = current_lod;
			record.dirty = true;
			any_changed = true;
		}
		if (desired_lod == static_cast<int>(current_lod)) {
			if (log_enabled) {
				GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%s asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (no change)",
						String::num_uint64((uint64_t)record.node_id), record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
			}
			continue;
		}

		if (desired_lod > static_cast<int>(current_lod)) {
			const float threshold = p_lod_config.get_distance_threshold(desired_lod);
			const float zone = use_fallback ? MAX(0.5f, 0.05f * threshold) : p_hysteresis_zone;
			if (effective_distance < threshold + zone) {
				if (log_enabled) {
					GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%s asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (hold-up)",
							String::num_uint64((uint64_t)record.node_id), record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
				}
				continue;
			}
		} else {
			const float threshold = p_lod_config.get_distance_threshold(static_cast<int>(current_lod));
			const float zone = use_fallback ? MAX(0.5f, 0.05f * threshold) : p_hysteresis_zone;
			if (effective_distance > threshold - zone) {
				if (log_enabled) {
					GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%s asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (hold-down)",
							String::num_uint64((uint64_t)record.node_id), record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
				}
				continue;
			}
		}

		record.last_lod = static_cast<uint32_t>(desired_lod);
		record.dirty = true;
		any_changed = true;
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%s asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u -> %u",
					String::num_uint64((uint64_t)record.node_id), record.asset_id, distance, bias, effective_distance,
					current_lod, record.last_lod));
		}
	}
	if (any_changed) {
		_bump_instance_generation(world->instance_generation);
	}
}

void GaussianSplatSceneDirector::build_instance_buffer(LocalVector<InstanceDataGPU> &out) const {
	MutexLock lock(world_mutex);
	out.clear();

	uint32_t total_instances = 0;
	for (const KeyValue<RID, SharedWorld> &E : worlds) {
		total_instances += E.value.instances.size();
	}
	if (total_instances == 0) {
		return;
	}
	out.reserve(total_instances);

	for (const KeyValue<RID, SharedWorld> &E : worlds) {
		const SharedWorld &world = E.value;
		for (const InstanceRecord &record : world.instances) {
			if (!record.visible) {
				continue;
			}
			const SharedWorld::AssetRecord *asset_record = world.asset_records.getptr(record.asset_id);
			if (!asset_record || asset_record->data.is_null()) {
				continue;
			}
			InstanceDataGPU entry = {};

			const Basis &basis = record.transform.basis;
			const Vector3 scale = basis.get_scale();
			const float sx = Math::abs(scale.x);
			const float sy = Math::abs(scale.y);
			const float sz = Math::abs(scale.z);
			const float uniform_scale = MAX(sx, MAX(sy, sz));

			Quaternion rotation = basis.get_rotation_quaternion().normalized();
			Quaternion inv_rotation = rotation.inverse();

			entry.rotation[0] = rotation.x;
			entry.rotation[1] = rotation.y;
			entry.rotation[2] = rotation.z;
			entry.rotation[3] = rotation.w;

			entry.inv_rotation[0] = inv_rotation.x;
			entry.inv_rotation[1] = inv_rotation.y;
			entry.inv_rotation[2] = inv_rotation.z;
			entry.inv_rotation[3] = inv_rotation.w;

			entry.translation_scale[0] = record.transform.origin.x;
			entry.translation_scale[1] = record.transform.origin.y;
			entry.translation_scale[2] = record.transform.origin.z;
			entry.translation_scale[3] = uniform_scale;

			entry.params[0] = record.opacity;
			entry.params[1] = record.lod_bias;
			entry.params[2] = record.wind_intensity;
			entry.params[3] = float(record.wind_mode);

			entry.ids[0] = record.asset_id;
			uint32_t flags = record.flags;
			if (rotation.is_equal_approx(Quaternion())) {
				flags |= GS_INSTANCE_FLAG_ROTATION_IDENTITY;
			}
			if (Math::is_equal_approx(uniform_scale, 1.0f)) {
				flags |= GS_INSTANCE_FLAG_SCALE_IDENTITY;
			}
			if (record.transform.origin.is_zero_approx()) {
				flags |= GS_INSTANCE_FLAG_TRANSLATION_ZERO;
			}
			entry.ids[1] = flags;

			entry.lod[0] = record.last_lod;
			entry.lod[1] = 0;
			entry.wind_params[0] = record.wind_direction.x;
			entry.wind_params[1] = record.wind_direction.y;
			entry.wind_params[2] = record.wind_direction.z;
			entry.wind_params[3] = record.wind_frequency;

			out.push_back(entry);
		}
	}
}

void GaussianSplatSceneDirector::build_instance_buffer_for_renderer(const GaussianSplatRenderer *p_renderer,
		LocalVector<InstanceDataGPU> &out, bool p_shadow_casters_only) const {
	MutexLock lock(world_mutex);
	out.clear();

	const SharedWorld *world = _find_world_for_renderer(p_renderer);
	if (!world || world->instances.is_empty()) {
		return;
	}

	const bool log_enabled = _is_scene_director_log_enabled();
	const bool trace_enabled = GaussianSplatting::debug_trace_is_enabled();
	out.reserve(world->instances.size());
	uint32_t skipped_instances = 0;
	uint32_t traced_total = 0;
	uint32_t traced_rotation_identity = 0;
	uint32_t traced_scale_identity = 0;
	uint32_t traced_translation_zero = 0;
	uint32_t traced_fully_identity = 0;
	for (const InstanceRecord &record : world->instances) {
		if (!record.visible) {
			continue;
		}
		if (p_shadow_casters_only && !record.casts_shadow) {
			continue;
		}
		const SharedWorld::AssetRecord *asset_record = world->asset_records.getptr(record.asset_id);
		if (!asset_record || asset_record->data.is_null()) {
			if (log_enabled) {
				GS_LOG_RENDERER_DEBUG(vformat("[InstanceBuffer] SKIP asset_id=%d record=%s data=%s",
						record.asset_id,
						asset_record ? "found" : "NULL",
						(asset_record && asset_record->data.is_valid()) ? "valid" : "null"));
			}
			skipped_instances++;
			continue;
		}
		InstanceDataGPU entry = {};

		const Basis &basis = record.transform.basis;
		const Vector3 scale = basis.get_scale();
		const float sx = Math::abs(scale.x);
		const float sy = Math::abs(scale.y);
		const float sz = Math::abs(scale.z);
		const float uniform_scale = MAX(sx, MAX(sy, sz));

		Quaternion rotation = basis.get_rotation_quaternion().normalized();
		Quaternion inv_rotation = rotation.inverse();

		entry.rotation[0] = rotation.x;
		entry.rotation[1] = rotation.y;
		entry.rotation[2] = rotation.z;
		entry.rotation[3] = rotation.w;

		entry.inv_rotation[0] = inv_rotation.x;
		entry.inv_rotation[1] = inv_rotation.y;
		entry.inv_rotation[2] = inv_rotation.z;
		entry.inv_rotation[3] = inv_rotation.w;

		entry.translation_scale[0] = record.transform.origin.x;
		entry.translation_scale[1] = record.transform.origin.y;
		entry.translation_scale[2] = record.transform.origin.z;
		entry.translation_scale[3] = uniform_scale;

		entry.params[0] = record.opacity;
		entry.params[1] = record.lod_bias;
		entry.params[2] = record.wind_intensity;
		entry.params[3] = float(record.wind_mode);

		entry.ids[0] = record.asset_id;
		uint32_t flags = record.flags;
		if (rotation.is_equal_approx(Quaternion())) {
			flags |= GS_INSTANCE_FLAG_ROTATION_IDENTITY;
		}
		if (Math::is_equal_approx(uniform_scale, 1.0f)) {
			flags |= GS_INSTANCE_FLAG_SCALE_IDENTITY;
		}
		if (record.transform.origin.is_zero_approx()) {
			flags |= GS_INSTANCE_FLAG_TRANSLATION_ZERO;
		}
		entry.ids[1] = flags;

		entry.lod[0] = record.last_lod;
		entry.lod[1] = 0;
		entry.wind_params[0] = record.wind_direction.x;
		entry.wind_params[1] = record.wind_direction.y;
		entry.wind_params[2] = record.wind_direction.z;
		entry.wind_params[3] = record.wind_frequency;

		out.push_back(entry);
		if (trace_enabled) {
			const bool rotation_identity = (flags & GS_INSTANCE_FLAG_ROTATION_IDENTITY) != 0u;
			const bool scale_identity = (flags & GS_INSTANCE_FLAG_SCALE_IDENTITY) != 0u;
			const bool translation_zero = (flags & GS_INSTANCE_FLAG_TRANSLATION_ZERO) != 0u;
			traced_total++;
			traced_rotation_identity += rotation_identity ? 1u : 0u;
			traced_scale_identity += scale_identity ? 1u : 0u;
			traced_translation_zero += translation_zero ? 1u : 0u;
			traced_fully_identity += (rotation_identity && scale_identity && translation_zero) ? 1u : 0u;
		}
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG(vformat("[InstanceBuffer] idx=%d node=%s asset=%u lod=%u flags=0x%08X pos=(%.3f,%.3f,%.3f) scale=%.3f",
					out.size() - 1,
					String::num_uint64((uint64_t)record.node_id), record.asset_id, record.last_lod, record.flags,
					entry.translation_scale[0], entry.translation_scale[1], entry.translation_scale[2], entry.translation_scale[3]));
		}
	}

	if (log_enabled) {
		GS_LOG_RENDERER_DEBUG(vformat("[InstanceBuffer] total_instances=%d (world=%d)",
				out.size(), world->instances.size()));
	}

	if (trace_enabled) {
		GaussianSplatting::debug_trace_record_instance_buffer(out.size(), world->instances.size(), skipped_instances);
		GaussianSplatting::debug_trace_record_instance_flags(traced_total, traced_rotation_identity, traced_scale_identity,
				traced_translation_zero, traced_fully_identity);
		if (skipped_instances > 0 || out.size() != world->instances.size()) {
			GaussianSplatting::debug_trace_record_event("instance_buffer",
					vformat("build out=%d world=%d skipped=%d",
							out.size(), world->instances.size(), skipped_instances),
					skipped_instances > 0);
		}
	}
}

uint64_t GaussianSplatSceneDirector::get_instance_generation_for_renderer(const GaussianSplatRenderer *p_renderer) const {
	MutexLock lock(world_mutex);
	const SharedWorld *world = _find_world_for_renderer(p_renderer);
	if (!world) {
		return 0;
	}
	return world->instance_generation;
}

uint32_t GaussianSplatSceneDirector::get_instance_count_for_renderer(const GaussianSplatRenderer *p_renderer) const {
	MutexLock lock(world_mutex);
	const SharedWorld *world = _find_world_for_renderer(p_renderer);
	if (!world) {
		return 0;
	}
	return world->instances.size();
}

bool GaussianSplatSceneDirector::upsert_world_submission(const WorldSubmission &p_submission) {
	if (p_submission.owner_id == ObjectID() || !p_submission.scenario.is_valid()) {
		return false;
	}

	// Scaffolding-only path: update stored submission metadata without mutating renderer state.
	MutexLock lock(world_mutex);
	SharedWorld *previous_world = _find_world_for_world_submission(p_submission.owner_id);
	if (previous_world && previous_world->scenario != p_submission.scenario) {
		previous_world->world_submission = SharedWorld::WorldSubmissionRecord();
	}

	SharedWorld *world = worlds.getptr(p_submission.scenario);
	if (!world) {
		SharedWorld new_world;
		new_world.scenario = p_submission.scenario;
		worlds.insert(p_submission.scenario, new_world);
		world = worlds.getptr(p_submission.scenario);
	}

	_store_world_submission_record(world->world_submission, p_submission);
	return true;
}

bool GaussianSplatSceneDirector::submit_world_submission(const WorldSubmission &p_submission) {
	if (p_submission.owner_id == ObjectID() || !p_submission.scenario.is_valid()) {
		return false;
	}

	// Runtime world path: renderer mutation, ownership arbitration, and rollback stay centralized here.
	MutexLock lock(world_mutex);
	SharedWorld *world = _get_or_create_world_for_scenario(p_submission.scenario);
	if (!world) {
		return false;
	}

	SharedWorld *previous_world = _find_world_for_world_submission(p_submission.owner_id);
	const SharedWorld::WorldSubmissionRecord target_previous_record = world->world_submission;
	const bool same_owner = target_previous_record.active && target_previous_record.owner_id == p_submission.owner_id;
	if (target_previous_record.active && !same_owner) {
		if (_is_world_submission_owner_live(world->world_submission.owner_id)) {
			return false;
		}
	}

	const WorldSubmissionRendererConfigSnapshot target_previous_renderer_config =
			_snapshot_world_submission_renderer_config(world->renderer);
	SharedWorld::WorldSubmissionRecord candidate_record;
	_store_world_submission_record(candidate_record, p_submission);
	if (!_apply_world_submission_to_renderer(*world, candidate_record)) {
		_restore_world_submission_renderer_config(world->renderer, target_previous_renderer_config);
		if (target_previous_record.active) {
			_apply_world_submission_to_renderer(*world, target_previous_record);
		} else {
			_clear_world_submission_renderer(*world);
		}
		return false;
	}

	if (previous_world && previous_world != world) {
		_clear_world_submission_renderer(*previous_world);
		previous_world->world_submission = SharedWorld::WorldSubmissionRecord();
	}

	world->world_submission = candidate_record;
	return true;
}

void GaussianSplatSceneDirector::unregister_world_submission(ObjectID p_owner_id) {
	MutexLock lock(world_mutex);
	SharedWorld *world = _find_world_for_world_submission(p_owner_id);
	if (!world) {
		return;
	}
	world->world_submission = SharedWorld::WorldSubmissionRecord();
}

void GaussianSplatSceneDirector::release_world_submission(ObjectID p_owner_id) {
	MutexLock lock(world_mutex);
	SharedWorld *world = _find_world_for_world_submission(p_owner_id);
	if (!world) {
		return;
	}
	_clear_world_submission_renderer(*world);
	world->world_submission = SharedWorld::WorldSubmissionRecord();
}

bool GaussianSplatSceneDirector::get_world_submission(ObjectID p_owner_id, WorldSubmission *r_submission) const {
	ERR_FAIL_NULL_V(r_submission, false);

	MutexLock lock(world_mutex);
	const SharedWorld *world = _find_world_for_world_submission(p_owner_id);
	if (!world || !world->world_submission.active) {
		return false;
	}

	_copy_world_submission_record(*world, world->world_submission, r_submission);
	return true;
}

bool GaussianSplatSceneDirector::get_world_submission_for_scenario(const RID &p_scenario, WorldSubmission *r_submission) const {
	ERR_FAIL_NULL_V(r_submission, false);

	MutexLock lock(world_mutex);
	const SharedWorld *world = worlds.getptr(p_scenario);
	if (!world || !world->world_submission.active) {
		return false;
	}

	_copy_world_submission_record(*world, world->world_submission, r_submission);
	return true;
}

bool GaussianSplatSceneDirector::has_world_submission_for_renderer(const GaussianSplatRenderer *p_renderer) const {
	MutexLock lock(world_mutex);
	const SharedWorld *world = _find_world_for_renderer(p_renderer);
	return world && world->world_submission.active;
}

bool GaussianSplatSceneDirector::upsert_preview_submission(const PreviewSubmission &p_submission) {
	if (p_submission.owner_id == ObjectID()) {
		return false;
	}

	MutexLock lock(world_mutex);
	PreviewSubmissionRecord record;
	record.owner_id = p_submission.owner_id;
	record.renderer = p_submission.renderer;
	record.gaussian_data = p_submission.gaussian_data;
	record.metadata = p_submission.metadata;
	record.source_label = p_submission.source_label;
	record.has_desired_residency_hint = p_submission.has_desired_residency_hint;
	record.desired_residency_hint = p_submission.desired_residency_hint;
	preview_submissions[p_submission.owner_id] = record;
	return true;
}

void GaussianSplatSceneDirector::unregister_preview_submission(ObjectID p_owner_id) {
	MutexLock lock(world_mutex);
	preview_submissions.erase(p_owner_id);
}

bool GaussianSplatSceneDirector::get_preview_submission(ObjectID p_owner_id, PreviewSubmission *r_submission) const {
	ERR_FAIL_NULL_V(r_submission, false);

	MutexLock lock(world_mutex);
	const PreviewSubmissionRecord *record = preview_submissions.getptr(p_owner_id);
	if (!record) {
		return false;
	}

	r_submission->owner_id = record->owner_id;
	r_submission->renderer = record->renderer;
	r_submission->gaussian_data = record->gaussian_data;
	r_submission->metadata = record->metadata;
	r_submission->source_label = record->source_label;
	r_submission->has_desired_residency_hint = record->has_desired_residency_hint;
	r_submission->desired_residency_hint = record->desired_residency_hint;
	return true;
}

bool GaussianSplatSceneDirector::has_preview_submission_for_renderer(const GaussianSplatRenderer *p_renderer) const {
	MutexLock lock(world_mutex);
	return _find_preview_submission_for_renderer(p_renderer) != nullptr;
}

bool GaussianSplatSceneDirector::get_submission_residency_hint_for_renderer(const GaussianSplatRenderer *p_renderer,
		int32_t *r_hint, String *r_source) const {
	ERR_FAIL_NULL_V(r_hint, false);

	MutexLock lock(world_mutex);
	if (const PreviewSubmissionRecord *preview = _find_preview_submission_for_renderer(p_renderer)) {
		if (preview->has_desired_residency_hint) {
			*r_hint = preview->desired_residency_hint;
			if (r_source) {
				*r_source = "preview_submission";
			}
			return true;
		}
	}

	if (const SharedWorld *world = _find_world_for_renderer(p_renderer)) {
		if (world->world_submission.active && world->world_submission.has_desired_residency_hint) {
			*r_hint = world->world_submission.desired_residency_hint;
			if (r_source) {
				*r_source = "world_submission";
			}
			return true;
		}

		bool found_instance_hint = false;
		int32_t instance_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT;
		for (const InstanceRecord &record : world->instances) {
			if (!record.has_desired_residency_hint) {
				continue;
			}
			if (!found_instance_hint) {
				found_instance_hint = true;
				instance_hint = record.desired_residency_hint;
				continue;
			}
			if (instance_hint != record.desired_residency_hint) {
				if (r_source) {
					*r_source = "mixed_instance_submissions";
				}
				return false;
			}
		}
		if (found_instance_hint) {
			*r_hint = instance_hint;
			if (r_source) {
				*r_source = "instance_submission";
			}
			return true;
		}
	}

	if (r_source) {
		*r_source = "none";
	}
	return false;
}

GaussianSplatSceneDirector::SubmissionCounts GaussianSplatSceneDirector::get_submission_counts() const {
	MutexLock lock(world_mutex);

	SubmissionCounts counts;
	for (const KeyValue<RID, SharedWorld> &E : worlds) {
		counts.instance_submissions += E.value.instances.size();
		if (E.value.world_submission.active) {
			counts.world_submissions++;
		}
	}
	counts.preview_submissions = preview_submissions.size();
	return counts;
}

namespace {

static int _metadata_int(const Dictionary &p_metadata, const StringName &p_key, int p_default) {
	if (!p_metadata.has(p_key)) {
		return p_default;
	}
	const Variant value = p_metadata[p_key];
	if (value.get_type() == Variant::FLOAT) {
		return int((double)value);
	}
	return int(value);
}

static double _metadata_double(const Dictionary &p_metadata, const StringName &p_key, double p_default) {
	if (!p_metadata.has(p_key)) {
		return p_default;
	}
	const Variant value = p_metadata[p_key];
	if (value.get_type() == Variant::INT) {
		return double(int64_t(value));
	}
	return (double)value;
}

static bool _asset_requests_full_fidelity_runtime(const Ref<GaussianSplatAsset> &p_asset) {
	if (p_asset.is_null()) {
		return false;
	}
	const Dictionary import_metadata = p_asset->get_import_metadata();
	const int import_max_splats = _metadata_int(import_metadata, StringName("max_splats"), -1);
	const double density_multiplier = _metadata_double(import_metadata, StringName("density_multiplier"), 1.0);
	return import_max_splats == 0 && density_multiplier >= 0.999;
}

} // namespace

void GaussianSplatSceneDirector::collect_instance_assets_for_renderer(const GaussianSplatRenderer *p_renderer,
		LocalVector<InstanceAssetRegistration> &out, bool p_shadow_casters_only) const {
	MutexLock lock(world_mutex);
	out.clear();

	const SharedWorld *world = _find_world_for_renderer(p_renderer);
	if (!world || world->asset_records.is_empty()) {
		return;
	}

	HashSet<uint32_t> selected_asset_ids;
	selected_asset_ids.reserve(world->asset_records.size());
	for (const InstanceRecord &record : world->instances) {
		if (!record.visible) {
			continue;
		}
		if (p_shadow_casters_only && !record.casts_shadow) {
			continue;
		}
		if (record.asset_id != 0) {
			selected_asset_ids.insert(record.asset_id);
		}
	}

	out.reserve(selected_asset_ids.size());
	for (const uint32_t &asset_id : selected_asset_ids) {
		const SharedWorld::AssetRecord *record = world->asset_records.getptr(asset_id);
		if (!record || record->data.is_null()) {
			continue;
		}
		InstanceAssetRegistration entry;
		entry.asset_id = asset_id;
		entry.data = record->data;
		entry.edited_version = record->edited_version;
		entry.requests_full_fidelity_runtime = _asset_requests_full_fidelity_runtime(record->asset);
		out.push_back(entry);
	}
}





Ref<GaussianSplatRenderer> GaussianSplatSceneDirector::get_shared_renderer(World3D *p_world) {
	MutexLock lock(world_mutex);
	SharedWorld *world = _get_or_create_world(p_world);
	if (!world) {
		return Ref<GaussianSplatRenderer>();
	}
	return world->renderer;
}

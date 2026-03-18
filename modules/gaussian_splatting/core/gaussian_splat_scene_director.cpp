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
    if (singleton == this) {
        singleton = nullptr;
    }
}

void GaussianSplatSceneDirector::_bind_methods() {
}

GaussianSplatSceneDirector::SharedWorld *GaussianSplatSceneDirector::_get_or_create_world(World3D *p_world) {
	ERR_FAIL_NULL_V(p_world, nullptr);
	RID scenario = p_world->get_scenario();
	if (!scenario.is_valid()) {
		return nullptr;
	}

	SharedWorld *entry = worlds.getptr(scenario);
	if (!entry) {
		SharedWorld world;
		world.scenario = scenario;
		worlds.insert(scenario, world);
		entry = worlds.getptr(scenario);
	}

	if (entry && !entry->renderer.is_valid()) {
		GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
		RenderingDevice *device = manager ? manager->get_primary_rendering_device() : nullptr;
		if (!device) {
			static bool warned_missing_device = false;
			if (!warned_missing_device) {
				warned_missing_device = true;
				GS_LOG_RENDERER_ERROR("[GaussianSplatSceneDirector] Unable to acquire local RenderingDevice for shared renderer");
			}
			return entry;
		}

		entry->renderer = Ref<GaussianSplatRenderer>(memnew(GaussianSplatRenderer(device)));
		if (_is_scene_director_log_enabled()) {
			GS_LOG_RENDERER_DEBUG("[SceneDirector] Created shared renderer (deferred initialization)");
		}

	}

	return entry;
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


void GaussianSplatSceneDirector::register_instance(ObjectID p_node_id, const Ref<GaussianSplatAsset> &p_asset,
        const Transform3D &p_transform, float p_opacity, float p_lod_bias, uint32_t p_flags,
        float p_wind_intensity, uint32_t p_wind_mode, const Vector3 &p_wind_direction, float p_wind_frequency) {
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
		uint32_t p_flags, float p_wind_intensity, uint32_t p_wind_mode,
		const Vector3 &p_wind_direction, float p_wind_frequency) {
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
					GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%llu asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (no change)",
							(uint64_t)record.node_id, record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
				}
				continue;
			}

			if (desired_lod > static_cast<int>(current_lod)) {
				const float threshold = p_lod_config.get_distance_threshold(desired_lod);
				const float zone = use_fallback ? MAX(0.5f, 0.05f * threshold) : p_hysteresis_zone;
				if (effective_distance < threshold + zone) {
					if (log_enabled) {
						GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%llu asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (hold-up)",
								(uint64_t)record.node_id, record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
					}
					continue;
				}
			} else {
				const float threshold = p_lod_config.get_distance_threshold(static_cast<int>(current_lod));
				const float zone = use_fallback ? MAX(0.5f, 0.05f * threshold) : p_hysteresis_zone;
				if (effective_distance > threshold - zone) {
					if (log_enabled) {
						GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%llu asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (hold-down)",
								(uint64_t)record.node_id, record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
					}
					continue;
				}
			}

			record.last_lod = static_cast<uint32_t>(desired_lod);
			record.dirty = true;
			any_changed = true;
			if (log_enabled) {
				GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%llu asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u -> %u",
						(uint64_t)record.node_id, record.asset_id, distance, bias, effective_distance,
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
				GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%llu asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (no change)",
						(uint64_t)record.node_id, record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
			}
			continue;
		}

		if (desired_lod > static_cast<int>(current_lod)) {
			const float threshold = p_lod_config.get_distance_threshold(desired_lod);
			const float zone = use_fallback ? MAX(0.5f, 0.05f * threshold) : p_hysteresis_zone;
			if (effective_distance < threshold + zone) {
				if (log_enabled) {
					GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%llu asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (hold-up)",
							(uint64_t)record.node_id, record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
				}
				continue;
			}
		} else {
			const float threshold = p_lod_config.get_distance_threshold(static_cast<int>(current_lod));
			const float zone = use_fallback ? MAX(0.5f, 0.05f * threshold) : p_hysteresis_zone;
			if (effective_distance > threshold - zone) {
				if (log_enabled) {
					GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%llu asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u desired=%d (hold-down)",
							(uint64_t)record.node_id, record.asset_id, distance, bias, effective_distance, current_lod, desired_lod));
				}
				continue;
			}
		}

		record.last_lod = static_cast<uint32_t>(desired_lod);
		record.dirty = true;
		any_changed = true;
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG(vformat("[InstanceLOD] node=%llu asset=%u dist=%.3f bias=%.3f eff=%.3f lod=%u -> %u",
					(uint64_t)record.node_id, record.asset_id, distance, bias, effective_distance,
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
		LocalVector<InstanceDataGPU> &out) const {
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
			GS_LOG_RENDERER_DEBUG(vformat("[InstanceBuffer] idx=%d node=%llu asset=%u lod=%u flags=0x%08X pos=(%.3f,%.3f,%.3f) scale=%.3f",
					out.size() - 1,
					(uint64_t)record.node_id, record.asset_id, record.last_lod, record.flags,
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

void GaussianSplatSceneDirector::collect_instance_assets_for_renderer(const GaussianSplatRenderer *p_renderer,
		LocalVector<InstanceAssetRegistration> &out) const {
	MutexLock lock(world_mutex);
	out.clear();

	const SharedWorld *world = _find_world_for_renderer(p_renderer);
	if (!world || world->asset_records.is_empty()) {
		return;
	}

	out.reserve(world->asset_records.size());
	for (const KeyValue<uint32_t, SharedWorld::AssetRecord> &E : world->asset_records) {
		if (E.key == 0 || E.value.data.is_null()) {
			continue;
		}
		InstanceAssetRegistration entry;
		entry.asset_id = E.key;
		entry.data = E.value.data;
		entry.edited_version = E.value.edited_version;
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

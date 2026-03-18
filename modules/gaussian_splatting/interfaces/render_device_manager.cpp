#include "render_device_manager.h"
#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "servers/rendering_server.h"
#include "../core/gaussian_splat_manager.h"
#include "../interfaces/sync_policy.h"
#include "../logger/gs_logger.h"

// Helper functions (static - no class scope needed)
static bool _is_main_rendering_device(RenderingDevice *p_device) {
	if (!p_device) {
		return false;
	}
	if (RenderingServer *rs = RenderingServer::get_singleton()) {
		if (RenderingDevice *server_device = rs->get_rendering_device()) {
			return p_device == server_device;
		}
	}
	if (RenderingDevice *singleton_device = RenderingDevice::get_singleton()) {
		return p_device == singleton_device;
	}
	return false;
}

static RenderingDevice *_ensure_local_device(RenderingDevice *p_candidate) {
	return p_candidate;
}

static RenderingDevice *_acquire_manager_local_device() {
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		if (RenderingDevice *primary = _ensure_local_device(manager->get_primary_rendering_device())) {
			return primary;
		}
		if (RenderingDevice *shared = _ensure_local_device(manager->get_shared_submission_device())) {
			return shared;
		}
	}
	return nullptr;
}

static bool _matches_known_resource_type(RenderingDevice *p_device, const RID &p_rid, uint8_t p_type_flags) {
	constexpr uint8_t RESOURCE_TYPE_NONE = 0;
	constexpr uint8_t RESOURCE_TYPE_TEXTURE = 1 << 0;
	constexpr uint8_t RESOURCE_TYPE_FRAMEBUFFER = 1 << 1;
	constexpr uint8_t RESOURCE_TYPE_UNIFORM_SET = 1 << 2;
	constexpr uint8_t RESOURCE_TYPE_RENDER_PIPELINE = 1 << 3;
	constexpr uint8_t RESOURCE_TYPE_COMPUTE_PIPELINE = 1 << 4;
	constexpr uint8_t RESOURCE_TYPE_BUFFER = 1 << 5;

	if (!p_device || !p_rid.is_valid()) {
		return false;
	}
	if (p_type_flags == RESOURCE_TYPE_NONE) {
		return p_device->texture_is_valid(p_rid) || p_device->framebuffer_is_valid(p_rid) ||
				p_device->uniform_set_is_valid(p_rid) || p_device->render_pipeline_is_valid(p_rid) ||
				p_device->compute_pipeline_is_valid(p_rid);
	}

	if ((p_type_flags & RESOURCE_TYPE_TEXTURE) && p_device->texture_is_valid(p_rid)) {
		return true;
	}
	if ((p_type_flags & RESOURCE_TYPE_FRAMEBUFFER) && p_device->framebuffer_is_valid(p_rid)) {
		return true;
	}
	if ((p_type_flags & RESOURCE_TYPE_UNIFORM_SET) && p_device->uniform_set_is_valid(p_rid)) {
		return true;
	}
	if ((p_type_flags & RESOURCE_TYPE_RENDER_PIPELINE) && p_device->render_pipeline_is_valid(p_rid)) {
		return true;
	}
	if ((p_type_flags & RESOURCE_TYPE_COMPUTE_PIPELINE) && p_device->compute_pipeline_is_valid(p_rid)) {
		return true;
	}
	if ((p_type_flags & RESOURCE_TYPE_BUFFER) && p_device->buffer_is_valid(p_rid)) {
		return true;
	}
	return false;
}

void RenderDeviceManager::_bind_methods() {
	// Exposed for debugging/introspection if needed
}

RenderDeviceManager::RenderDeviceManager() {
}

RenderDeviceManager::~RenderDeviceManager() {
	shutdown();
}

Error RenderDeviceManager::initialize(RenderingDevice *p_primary_device) {
	main_rd = p_primary_device;

	if (!main_rd) {
		if (RenderingServer *rs = RenderingServer::get_singleton()) {
			main_rd = rs->get_rendering_device();
		}
	}

	if (!main_rd) {
		main_rd = RenderingDevice::get_singleton();
	}

	if (!main_rd) {
		main_rd = _acquire_manager_device();
	}

	if (!main_rd) {
		GS_LOG_WARN_DEFAULT("[RenderDeviceManager] No RenderingDevice available at initialization");
		return ERR_UNAVAILABLE;
	}

	local_rd = main_rd;
	submission_rd = main_rd;

	return OK;
}

void RenderDeviceManager::shutdown() {
	resource_owner_map.clear();
	resource_owner_instance_id_map.clear();
	resource_ownership_map.clear();
	resource_label_map.clear();
	resource_type_map.clear();
	texture_owner_map.clear();
	texture_trace.clear();
	cross_device_ops.clear();

	if (owns_local_rd && local_rd && local_rd != main_rd) {
		// Note: We don't free the device - Godot manages RenderingDevice lifecycle
		local_rd = nullptr;
	}

	main_rd = nullptr;
	local_rd = nullptr;
	submission_rd = nullptr;
	owns_local_rd = false;
}

DeviceContext RenderDeviceManager::get_context() const {
	DeviceContext ctx;
	ctx.main_device = main_rd;
	ctx.submission_device = submission_rd ? submission_rd : main_rd;
	ctx.resource_device = local_rd ? local_rd : main_rd;
	return ctx;
}

RenderingDevice *RenderDeviceManager::get_main_device() const {
	return main_rd;
}

RenderingDevice *RenderDeviceManager::get_submission_device() const {
	// Implements _get_submission_device logic from god class
	RenderingDevice *temp_local = nullptr;
	GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
	if (manager) {
		RenderingDevice *shared_device = _ensure_local_device(manager->get_shared_submission_device());
		if (shared_device) {
			temp_local = shared_device;
			reported_missing_submission = false;
			return temp_local;
		}
	}

	if (main_rd) {
		temp_local = main_rd;
		reported_missing_submission = false;
		return temp_local;
	}

	if (main_rd && !reported_missing_submission) {
		GS_LOG_WARN_DEFAULT("[RenderDeviceManager] Unable to acquire local RenderingDevice for submissions");
		reported_missing_submission = true;
	}

	return main_rd;
}

RenderingDevice *RenderDeviceManager::acquire_rendering_device() {
	// Implements _acquire_rendering_device logic
	if (main_rd) {
		// Also update viewport tracking
		if (RenderingDevice *singleton_device = RenderingDevice::get_singleton()) {
			// Track viewport device (mutable state in god class)
		}
		return main_rd;
	}

	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		main_rd = _ensure_local_device(manager->get_primary_rendering_device());
		if (!main_rd) {
			main_rd = _acquire_manager_local_device();
		}
	}

	if (!main_rd) {
		// Try RenderingServer render device or RenderingDevice singleton
		if (RenderingServer *rs = RenderingServer::get_singleton()) {
			main_rd = rs->get_rendering_device();
		}
	}

	if (!main_rd) {
		if (RenderingDevice *singleton_device = RenderingDevice::get_singleton()) {
			main_rd = singleton_device;
		}
	}

	if (main_rd) {
		local_rd = main_rd;
		submission_rd = main_rd;
	}

	return main_rd;
}

bool RenderDeviceManager::ensure_rendering_device(const char *p_context) {
	if (main_rd) {
		return true;
	}

	acquire_rendering_device();

	if (!main_rd) {
		if (!reported_missing_device) {
			GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] No RenderingDevice available (context: %s)",
					p_context ? p_context : "unknown"));
			reported_missing_device = true;
		}
		return false;
	}

	return true;
}

bool RenderDeviceManager::ensure_submission_device(const char *p_context) {
	if (!ensure_rendering_device(p_context)) {
		return false;
	}

	if (!submission_rd) {
		submission_rd = get_submission_device();
	}

	if (!submission_rd) {
		if (!reported_missing_submission) {
			GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] No submission device (context: %s)",
					p_context ? p_context : "unknown"));
			reported_missing_submission = true;
		}
		return false;
	}

	return true;
}

void RenderDeviceManager::track_resource(const RID &p_rid, RenderingDevice *p_device, bool p_owned, const char *p_label) {
	if (!p_rid.is_valid() || !p_device) {
		return;
	}

	const uint64_t rid_id = p_rid.get_id();
	// Do not downgrade explicit ownership when the same RID is later
	// observed through read/consumer paths (for example culling/sorting).
	if (const bool *existing_owned_ptr = resource_ownership_map.getptr(rid_id)) {
		if (*existing_owned_ptr && !p_owned) {
			p_owned = true;
		}
	}
	uint8_t type_flags = RESOURCE_TYPE_NONE;
	if (p_device->texture_is_valid(p_rid)) {
		type_flags |= RESOURCE_TYPE_TEXTURE;
	}
	if (p_device->framebuffer_is_valid(p_rid)) {
		type_flags |= RESOURCE_TYPE_FRAMEBUFFER;
	}
	if (p_device->uniform_set_is_valid(p_rid)) {
		type_flags |= RESOURCE_TYPE_UNIFORM_SET;
	}
	if (p_device->render_pipeline_is_valid(p_rid)) {
		type_flags |= RESOURCE_TYPE_RENDER_PIPELINE;
	}
	if (p_device->compute_pipeline_is_valid(p_rid)) {
		type_flags |= RESOURCE_TYPE_COMPUTE_PIPELINE;
	}
	if (p_device->buffer_is_valid(p_rid)) {
		type_flags |= RESOURCE_TYPE_BUFFER;
	}

	const uint64_t owner_instance_id = p_device->get_device_instance_id();
	if (RenderingDevice *const *existing_owner = resource_owner_map.getptr(rid_id)) {
		const uint64_t *existing_owner_id = resource_owner_instance_id_map.getptr(rid_id);
		const uint64_t resolved_existing_owner_id = existing_owner_id ? *existing_owner_id : 0;
		const uint64_t current_existing_owner_id = *existing_owner ? (*existing_owner)->get_device_instance_id() : 0;
		uint8_t existing_type_flags = RESOURCE_TYPE_NONE;
		if (const uint8_t *existing_type_flags_ptr = resource_type_map.getptr(rid_id)) {
			existing_type_flags = *existing_type_flags_ptr;
		}
		if (*existing_owner && *existing_owner != p_device && resolved_existing_owner_id != 0 &&
				current_existing_owner_id == resolved_existing_owner_id) {
			const bool existing_owner_valid = _matches_known_resource_type(*existing_owner, p_rid, existing_type_flags);
			const bool new_owner_valid = _matches_known_resource_type(p_device, p_rid, type_flags);
			if (existing_owner_valid && !new_owner_valid) {
				String existing_label;
				if (const String *label_ptr = resource_label_map.getptr(rid_id)) {
					existing_label = *label_ptr;
				}
				GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] Rejecting ownership reassignment of RID %s%s from device %s to %s: tracked owner still validates resource",
						String::num_uint64(rid_id),
						existing_label.is_empty() ? String() : vformat(" (%s)", existing_label),
						String::num_uint64(resolved_existing_owner_id),
						String::num_uint64(owner_instance_id)));
				return;
			}
		}
	}

	resource_owner_map.insert(rid_id, p_device);
	resource_owner_instance_id_map.insert(rid_id, owner_instance_id);
	resource_ownership_map.insert(rid_id, p_owned);

	if (p_label && p_label[0] != '\0') {
		resource_label_map.insert(rid_id, String(p_label));
	} else {
		resource_label_map.erase(rid_id);
	}

	if (type_flags != RESOURCE_TYPE_NONE) {
		resource_type_map.insert(rid_id, type_flags);
	} else {
		resource_type_map.erase(rid_id);
	}

	// Check if it's a texture
	if (type_flags & RESOURCE_TYPE_TEXTURE) {
		texture_owner_map.insert(rid_id, p_device);
		push_texture_trace("track", p_rid, p_device);
	} else {
		texture_owner_map.erase(rid_id);
	}
}

RenderingDevice *RenderDeviceManager::get_resource_owner(const RID &p_rid, RenderingDevice *p_fallback) const {
	if (!p_rid.is_valid()) {
		return nullptr;
	}

	if (RenderingDevice *const *found = resource_owner_map.getptr(p_rid.get_id())) {
		if (!*found) {
			return p_fallback;
		}
		if (const uint64_t *tracked_instance_id = resource_owner_instance_id_map.getptr(p_rid.get_id())) {
			if (*tracked_instance_id != 0 && (*found)->get_device_instance_id() != *tracked_instance_id) {
				GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] Resource owner instance mismatch for RID %s (expected=%s actual=%s); using fallback owner",
						String::num_uint64(p_rid.get_id()), String::num_uint64(*tracked_instance_id),
						String::num_uint64((*found)->get_device_instance_id())));
				return p_fallback;
			}
		}
		return *found;
	}

	return p_fallback;
}

void RenderDeviceManager::forget_resource(const RID &p_rid) {
	if (!p_rid.is_valid()) {
		return;
	}

	uint64_t rid_id = p_rid.get_id();

	if (RenderingDevice *const *owner = texture_owner_map.getptr(rid_id)) {
		push_texture_trace("forget", p_rid, *owner);
	}

	resource_owner_map.erase(rid_id);
	resource_owner_instance_id_map.erase(rid_id);
	resource_ownership_map.erase(rid_id);
	resource_label_map.erase(rid_id);
	resource_type_map.erase(rid_id);
	texture_owner_map.erase(rid_id);
}

void RenderDeviceManager::free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid) {
	if (!p_rid.is_valid()) {
		return;
	}

	const uint64_t rid_id = p_rid.get_id();
	String label;
	if (const String *label_ptr = resource_label_map.getptr(rid_id)) {
		label = *label_ptr;
	}

	RenderingDevice *const *tracked_owner_ptr = resource_owner_map.getptr(rid_id);
	if (!tracked_owner_ptr) {
		GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] Refusing to free untracked RID %s%s in ownership-managed path; invalidating handle only",
				String::num_uint64(rid_id), label.is_empty() ? String() : vformat(" (%s)", label)));
		p_rid = RID();
		return;
	}

	RenderingDevice *tracked_owner = *tracked_owner_ptr;
	uint64_t tracked_owner_instance_id = 0;
	if (const uint64_t *instance_id_ptr = resource_owner_instance_id_map.getptr(rid_id)) {
		tracked_owner_instance_id = *instance_id_ptr;
	}

	if (!tracked_owner) {
		GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] Tracked owner device missing for RID %s%s; invalidating handle",
				String::num_uint64(rid_id), label.is_empty() ? String() : vformat(" (%s)", label)));
		forget_resource(p_rid);
		p_rid = RID();
		return;
	}

	if (tracked_owner_instance_id != 0 && tracked_owner->get_device_instance_id() != tracked_owner_instance_id) {
		GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] Tracked owner device instance mismatch for RID %s%s (expected=%s actual=%s); refusing free",
				String::num_uint64(rid_id), label.is_empty() ? String() : vformat(" (%s)", label),
				String::num_uint64(tracked_owner_instance_id), String::num_uint64(tracked_owner->get_device_instance_id())));
		forget_resource(p_rid);
		p_rid = RID();
		return;
	}

	if (p_fallback_device && p_fallback_device != tracked_owner) {
		const uint64_t fallback_id = p_fallback_device->get_device_instance_id();
		if (tracked_owner_instance_id != 0 && fallback_id != tracked_owner_instance_id) {
			GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] Ignoring fallback device %s while freeing RID %s%s tracked to owner %s",
					String::num_uint64(fallback_id), String::num_uint64(rid_id),
					label.is_empty() ? String() : vformat(" (%s)", label), String::num_uint64(tracked_owner_instance_id)));
		}
	}

	const bool is_tracked_texture = texture_owner_map.has(rid_id);
	uint8_t type_flags = RESOURCE_TYPE_NONE;
	if (const uint8_t *flags = resource_type_map.getptr(rid_id)) {
		type_flags = *flags;
	}

	bool owned = true;
	if (const bool *flag = resource_ownership_map.getptr(rid_id)) {
		owned = *flag;
	} else {
		GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] Missing ownership flag for tracked RID %s%s; treating as non-owned",
				String::num_uint64(rid_id), label.is_empty() ? String() : vformat(" (%s)", label)));
		owned = false;
	}

	if (!owned) {
		forget_resource(p_rid);
		p_rid = RID();
		return;
	}

	RenderingDevice *device = tracked_owner;

	GaussianSplatManager::ScopedSubmissionLock submission_lock;
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		if (device) {
			RenderingDevice *submission_device = manager->acquire_submission_device(device, submission_lock);
			if (submission_device) {
				if (submission_device == device || tracked_owner_instance_id == 0 ||
						submission_device->get_device_instance_id() == tracked_owner_instance_id) {
					device = submission_device;
				} else {
					GS_LOG_WARN_DEFAULT(vformat("[RenderDeviceManager] Submission device instance mismatch while freeing RID %s%s (owner=%s submission=%s); using tracked owner",
							String::num_uint64(rid_id), label.is_empty() ? String() : vformat(" (%s)", label),
							String::num_uint64(tracked_owner_instance_id), String::num_uint64(submission_device->get_device_instance_id())));
				}
			}
		}
	}

	if (!device) {
		ERR_FAIL_MSG(vformat("[RenderDeviceManager] Unable to resolve owning RenderingDevice for owned RID %s%s",
				String::num_uint64(rid_id),
				label.is_empty() ? String() : vformat(" (%s)", label)));
	}

	const bool texture_valid = device->texture_is_valid(p_rid);
	const bool framebuffer_valid = device->framebuffer_is_valid(p_rid);
	const bool uniform_set_valid = device->uniform_set_is_valid(p_rid);
	const bool render_pipeline_valid = device->render_pipeline_is_valid(p_rid);
	const bool compute_pipeline_valid = device->compute_pipeline_is_valid(p_rid);
	const bool buffer_valid = device->buffer_is_valid(p_rid);

	if (is_tracked_texture && !texture_valid) {
		texture_owner_map.erase(rid_id);
		forget_resource(p_rid);
		p_rid = RID();
		return;
	}

	// Explicitly audited auto-free resource classes (Godot PR 103113):
	// uniform sets, compute pipelines, render pipelines, and framebuffers.
	// They are released by dependency teardown and must not be explicitly freed.
	constexpr uint8_t AUTO_FREE_TYPES = RESOURCE_TYPE_UNIFORM_SET | RESOURCE_TYPE_COMPUTE_PIPELINE |
			RESOURCE_TYPE_RENDER_PIPELINE | RESOURCE_TYPE_FRAMEBUFFER;
	if (type_flags & AUTO_FREE_TYPES) {
		forget_resource(p_rid);
		p_rid = RID();
		return;
	}

	const bool any_known_type_valid = texture_valid || framebuffer_valid || uniform_set_valid || render_pipeline_valid || compute_pipeline_valid || buffer_valid;
	const bool tracked_known_type = is_tracked_texture || ((type_flags & AUTO_FREE_TYPES) != 0) || ((type_flags & RESOURCE_TYPE_BUFFER) != 0);

	// If this RID was tracked as a known typed resource and all type checks are now
	// invalid, it was already released elsewhere.
	if (!any_known_type_valid && tracked_known_type) {
		if (!label.is_empty()) {
			GS_LOG_INFO_DEFAULT(vformat("[RenderDeviceManager] Skipping free of RID %s (%s): resource already released",
					String::num_uint64(rid_id), label));
		}
		forget_resource(p_rid);
		p_rid = RID();
		return;
	}

	push_texture_trace("free", p_rid, device);
	device->free(p_rid);

	if (is_tracked_texture) {
		texture_owner_map.erase(rid_id);
	}
	forget_resource(p_rid);
	p_rid = RID();
}

void RenderDeviceManager::track_texture(const RID &p_texture, RenderingDevice *p_device) {
	if (!p_texture.is_valid() || !p_device) {
		return;
	}

	track_resource(p_texture, p_device, true, "texture");
	push_texture_trace("track_texture", p_texture, p_device);
}

RenderingDevice *RenderDeviceManager::get_texture_owner(const RID &p_texture) const {
	if (!p_texture.is_valid()) {
		return nullptr;
	}

	if (RenderingDevice *const *found = texture_owner_map.getptr(p_texture.get_id())) {
		return *found;
	}

	// Fallback to general resource owner
	if (RenderingDevice *const *fallback = resource_owner_map.getptr(p_texture.get_id())) {
		return *fallback;
	}

	return nullptr;
}

void RenderDeviceManager::submit_and_sync(RenderingDevice *p_device) {
	if (!p_device) {
		return;
	}

	gs_device_utils::safe_submit_and_sync(p_device);
}

void RenderDeviceManager::push_texture_trace(const String &p_action, const RID &p_texture, RenderingDevice *p_device) {
	if (!p_texture.is_valid()) {
		return;
	}

	// Limit trace size to prevent unbounded growth
	if (texture_trace.size() >= MAX_TRACE_ENTRIES) {
		texture_trace.remove_at(0);
	}

	TextureTraceEntry entry;
	entry.timestamp_usec = OS::get_singleton()->get_ticks_usec();
	entry.action = p_action;
	entry.texture_rid = p_texture.get_id();
	entry.device_instance_id = p_device ? p_device->get_device_instance_id() : 0;
#ifdef DEBUG_ENABLED
	entry.device_pointer_debug = p_device ? reinterpret_cast<uint64_t>(p_device) : 0;
#endif
	texture_trace.push_back(entry);
}

void RenderDeviceManager::push_cross_device_operation(const String &p_context, RenderingDevice *p_source, RenderingDevice *p_target) {
	if (!p_source || !p_target || p_source == p_target) {
		return;
	}

	if (cross_device_ops.size() >= MAX_TRACE_ENTRIES) {
		cross_device_ops.remove_at(0);
	}

	CrossDeviceOperation op;
	op.timestamp_usec = OS::get_singleton()->get_ticks_usec();
	op.context = p_context;
	op.source_device_instance_id = p_source->get_device_instance_id();
	op.target_device_instance_id = p_target->get_device_instance_id();
#ifdef DEBUG_ENABLED
	op.source_device_pointer_debug = reinterpret_cast<uint64_t>(p_source);
	op.target_device_pointer_debug = reinterpret_cast<uint64_t>(p_target);
#endif
	cross_device_ops.push_back(op);
}

void RenderDeviceManager::clear_diagnostics() {
	texture_trace.clear();
	cross_device_ops.clear();
}

RenderingDevice *RenderDeviceManager::_acquire_manager_device() {
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		if (RenderingDevice *primary = manager->get_primary_rendering_device()) {
			return primary;
		}
		if (RenderingDevice *shared = manager->get_shared_submission_device()) {
			return shared;
		}
	}
	return nullptr;
}

bool RenderDeviceManager::_is_main_rendering_device(RenderingDevice *p_device) const {
	if (!p_device) {
		return false;
	}
	if (RenderingServer *rs = RenderingServer::get_singleton()) {
		if (RenderingDevice *server_device = rs->get_rendering_device()) {
			return p_device == server_device;
		}
	}
	if (RenderingDevice *singleton_device = RenderingDevice::get_singleton()) {
		return p_device == singleton_device;
	}
	return false;
}

RenderingDevice *RenderDeviceManager::acquire_submission_device_for(RenderingDevice *p_device,
		GaussianSplatManager::ScopedSubmissionLock &r_lock) const {
	if (!p_device) {
		return nullptr;
	}

	RenderingDevice *device = p_device;
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		RenderingDevice *acquired = manager->acquire_submission_device(device, r_lock);
		if (acquired) {
			device = acquired;
		}
	}

	return device;
}

void RenderDeviceManager::synchronize_tile_submission(RenderingDevice *p_device, const char *p_context) {
	if (!p_device) {
		return;
	}

	RenderingDevice *sync_device = p_device;
	GaussianSplatManager::ScopedSubmissionLock submission_lock;
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		RenderingDevice *acquired = manager->acquire_submission_device(sync_device, submission_lock);
		if (acquired) {
			if (sync_device != acquired) {
				push_cross_device_operation(p_context ? p_context : "tile_sync", sync_device, acquired);
			}
			sync_device = acquired;
		}
	}

	if (sync_device) {
		gs_device_utils::safe_submit_and_sync(sync_device);
	}
}

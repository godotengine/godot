/**************************************************************************/
/*  test_render_device_manager_ownership.cpp                               */
/*  Ownership/lifetime regression tests for RenderDeviceManager            */
/**************************************************************************/

#include "test_macros.h"

#include "../interfaces/render_device_manager.h"

#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"

#ifdef TESTS_ENABLED

namespace TestGaussianSplatting {

static RenderingDevice *_create_local_test_device() {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        return nullptr;
    }
    return rs->create_local_rendering_device();
}

TEST_CASE("[GaussianSplatting] RenderDeviceManager blocks untracked free path") {
    RenderingServer *rs = RenderingServer::get_singleton();
    CHECK(rs != nullptr);
    if (!rs) {
        return;
    }

    RenderingDevice *rd = _create_local_test_device();
    bool owns_rd = true;
    if (!rd) {
        rd = rs->get_rendering_device();
        owns_rd = false;
    }
    CHECK(rd != nullptr);
    if (!rd) {
        return;
    }

    Ref<RenderDeviceManager> manager;
    manager.instantiate();
    Error init_err = manager->initialize(rd);
    CHECK(init_err == OK);
    if (init_err != OK) {
        if (owns_rd) {
            memdelete(rd);
        }
        return;
    }

    RID buffer = rd->storage_buffer_create(64);
    rd->set_resource_name(buffer, "GS_Test_RDManager_Ownership_UntrackedBuffer");
    CHECK(buffer.is_valid());
    if (!buffer.is_valid()) {
        manager->shutdown();
        if (owns_rd) {
            memdelete(rd);
        }
        return;
    }

    Vector<uint8_t> update_bytes;
    update_bytes.resize(4);
    for (int i = 0; i < update_bytes.size(); i++) {
        update_bytes.write[i] = 0x5a;
    }

    CHECK(rd->buffer_update(buffer, 0, update_bytes.size(), update_bytes.ptr()) == OK);

    RID managed_buffer = buffer;
    manager->free_owned_resource(rd, managed_buffer);

    CHECK_FALSE(managed_buffer.is_valid());
    CHECK(manager->get_tracked_resource_count() == 0);
    CHECK(rd->buffer_update(buffer, 0, update_bytes.size(), update_bytes.ptr()) == OK);

    rd->free(buffer);
    manager->shutdown();

    if (owns_rd) {
        memdelete(rd);
    }
}

TEST_CASE("[GaussianSplatting] RenderDeviceManager rejects foreign re-track for typed RID") {
    RenderingDevice *owner_rd = _create_local_test_device();
    RenderingDevice *foreign_rd = _create_local_test_device();
    CHECK(owner_rd != nullptr);
    CHECK(foreign_rd != nullptr);
    if (!owner_rd || !foreign_rd) {
        if (owner_rd) {
            memdelete(owner_rd);
        }
        if (foreign_rd) {
            memdelete(foreign_rd);
        }
        return;
    }

    if (owner_rd->get_device_instance_id() == foreign_rd->get_device_instance_id()) {
        memdelete(owner_rd);
        memdelete(foreign_rd);
        return;
    }

    Ref<RenderDeviceManager> manager;
    manager.instantiate();
    Error init_err = manager->initialize(owner_rd);
    CHECK(init_err == OK);
    if (init_err != OK) {
        memdelete(owner_rd);
        memdelete(foreign_rd);
        return;
    }

    RD::TextureFormat format;
    format.width = 8;
    format.height = 8;
    format.depth = 1;
    format.array_layers = 1;
    format.mipmaps = 1;
    format.texture_type = RD::TEXTURE_TYPE_2D;
    format.samples = RD::TEXTURE_SAMPLES_1;
    format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT |
            RD::TEXTURE_USAGE_STORAGE_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

    RID owner_texture = owner_rd->texture_create(format, RD::TextureView());
    owner_rd->set_resource_name(owner_texture, "GS_Test_RDManager_Ownership_TypedTexture");
    CHECK(owner_texture.is_valid());
    if (!owner_texture.is_valid()) {
        manager->shutdown();
        memdelete(owner_rd);
        memdelete(foreign_rd);
        return;
    }

    manager->track_resource(owner_texture, owner_rd, true, "typed_texture_owner");
    manager->track_resource(owner_texture, foreign_rd, true, "typed_texture_foreign");

    RID managed_texture = owner_texture;
    manager->free_owned_resource(foreign_rd, managed_texture);

    CHECK_FALSE(managed_texture.is_valid());
    CHECK_FALSE(owner_rd->texture_is_valid(owner_texture));

    if (owner_rd->texture_is_valid(owner_texture)) {
        owner_rd->free(owner_texture);
    }

    manager->shutdown();
    memdelete(owner_rd);
    memdelete(foreign_rd);
}

TEST_CASE("[GaussianSplatting] RenderDeviceManager keeps owned RID semantics across passive re-track") {
    RenderingServer *rs = RenderingServer::get_singleton();
    CHECK(rs != nullptr);
    if (!rs) {
        return;
    }

    RenderingDevice *rd = _create_local_test_device();
    bool owns_rd = true;
    if (!rd) {
        rd = rs->get_rendering_device();
        owns_rd = false;
    }
    CHECK(rd != nullptr);
    if (!rd) {
        return;
    }

    Ref<RenderDeviceManager> manager;
    manager.instantiate();
    Error init_err = manager->initialize(rd);
    CHECK(init_err == OK);
    if (init_err != OK) {
        if (owns_rd) {
            memdelete(rd);
        }
        return;
    }

    RID buffer = rd->storage_buffer_create(64);
    rd->set_resource_name(buffer, "GS_Test_RDManager_Ownership_NoDowngrade");
    CHECK(buffer.is_valid());
    if (!buffer.is_valid()) {
        manager->shutdown();
        if (owns_rd) {
            memdelete(rd);
        }
        return;
    }

    manager->track_resource(buffer, rd, true, "owned_first");
    // Simulate a consumer pass tracking the same RID as non-owned.
    manager->track_resource(buffer, rd, false, "consumer_pass");

    RID managed_buffer = buffer;
    manager->free_owned_resource(rd, managed_buffer);

    Vector<uint8_t> update_bytes;
    update_bytes.resize(4);
    for (int i = 0; i < update_bytes.size(); i++) {
        update_bytes.write[i] = 0x7c;
    }

    CHECK_FALSE(managed_buffer.is_valid());
    CHECK(manager->get_tracked_resource_count() == 0);
    const bool still_valid = rd->buffer_update(buffer, 0, update_bytes.size(), update_bytes.ptr()) == OK;
    CHECK_FALSE(still_valid);

    if (still_valid) {
        rd->free(buffer);
    }

    manager->shutdown();
    if (owns_rd) {
        memdelete(rd);
    }
}

TEST_CASE("[GaussianSplatting] RenderDeviceManager diagnostics use stable device instance IDs") {
    RenderingDevice *source_rd = _create_local_test_device();
    RenderingDevice *target_rd = _create_local_test_device();
    CHECK(source_rd != nullptr);
    CHECK(target_rd != nullptr);
    if (!source_rd || !target_rd) {
        if (source_rd) {
            memdelete(source_rd);
        }
        if (target_rd) {
            memdelete(target_rd);
        }
        return;
    }

    if (source_rd->get_device_instance_id() == target_rd->get_device_instance_id()) {
        memdelete(source_rd);
        memdelete(target_rd);
        return;
    }

    Ref<RenderDeviceManager> manager;
    manager.instantiate();
    Error init_err = manager->initialize(source_rd);
    CHECK(init_err == OK);
    if (init_err != OK) {
        memdelete(source_rd);
        memdelete(target_rd);
        return;
    }

    RID traced_rid = source_rd->storage_buffer_create(16);
    CHECK(traced_rid.is_valid());
    if (!traced_rid.is_valid()) {
        manager->shutdown();
        memdelete(source_rd);
        memdelete(target_rd);
        return;
    }

    manager->push_texture_trace("test_trace", traced_rid, source_rd);
    const Vector<RenderDeviceManager::TextureTraceEntry> &texture_trace = manager->get_texture_trace();
    CHECK(texture_trace.size() == 1);
    if (texture_trace.size() == 1) {
        const RenderDeviceManager::TextureTraceEntry &entry = texture_trace[0];
        CHECK(entry.device_instance_id == source_rd->get_device_instance_id());
#ifdef DEBUG_ENABLED
        CHECK(entry.device_pointer_debug == reinterpret_cast<uint64_t>(source_rd));
#endif
    }

    manager->push_cross_device_operation("test_cross_device", source_rd, target_rd);
    const Vector<RenderDeviceManager::CrossDeviceOperation> &cross_ops = manager->get_cross_device_operations();
    CHECK(cross_ops.size() == 1);
    if (cross_ops.size() == 1) {
        const RenderDeviceManager::CrossDeviceOperation &op = cross_ops[0];
        CHECK(op.source_device_instance_id == source_rd->get_device_instance_id());
        CHECK(op.target_device_instance_id == target_rd->get_device_instance_id());
#ifdef DEBUG_ENABLED
        CHECK(op.source_device_pointer_debug == reinterpret_cast<uint64_t>(source_rd));
        CHECK(op.target_device_pointer_debug == reinterpret_cast<uint64_t>(target_rd));
#endif
    }

    source_rd->free(traced_rid);
    manager->shutdown();
    memdelete(source_rd);
    memdelete(target_rd);
}

} // namespace TestGaussianSplatting

#endif // TESTS_ENABLED

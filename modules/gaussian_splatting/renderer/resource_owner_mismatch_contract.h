#ifndef RESOURCE_OWNER_MISMATCH_CONTRACT_H
#define RESOURCE_OWNER_MISMATCH_CONTRACT_H

#include "core/string/ustring.h"
#include "servers/rendering/rendering_device.h"
#include <cstdint>

class ResourceOwnerMismatchContract {
public:
    struct Inputs {
        bool rid_valid = false;
        bool has_owner = false;
        uint64_t owner_instance_id = 0;
        uint64_t active_instance_id = 0;
    };

    struct Decision {
        bool mismatch_detected = false;
        bool should_attempt_release = false;
        bool should_force_invalidate_after_release = false;
    };

    // Invariants:
    // - release/force actions are only valid when mismatch_detected=true.
    // - missing owner for a live RID is treated as mismatch and force-invalidate path.
    static Decision evaluate(const Inputs &p_inputs);
    static bool validate(const Inputs &p_inputs, const Decision &p_decision, String *r_error = nullptr);

    // Cross-device resource ownership verification (ISSUE-002).
    // Returns true if p_resource is owned by p_rd (texture or buffer).
    // Returns false and prints an error if the resource belongs to a different device.
    // p_label is used for diagnostic messages.
    static bool verify_texture_device_ownership(RenderingDevice *p_rd, RID p_resource, const char *p_label = nullptr);
    static bool verify_buffer_device_ownership(RenderingDevice *p_rd, RID p_resource, const char *p_label = nullptr);

    // Validate that a stored device pointer is still safe to use for cleanup.
    // Compares the stored device_id against the device's current instance id.
    // Returns true if the pointer is safe; returns false if the device was recycled
    // or destroyed (stale pointer).
    static bool is_device_generation_valid(RenderingDevice *p_device, uint64_t p_stored_device_id);
};

#endif // RESOURCE_OWNER_MISMATCH_CONTRACT_H

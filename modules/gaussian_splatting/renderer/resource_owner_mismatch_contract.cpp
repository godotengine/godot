#include "resource_owner_mismatch_contract.h"

#include "core/error/error_macros.h"

namespace {

static void _set_error(String *r_error, const String &p_error) {
    if (r_error) {
        *r_error = p_error;
    }
}

} // namespace

ResourceOwnerMismatchContract::Decision ResourceOwnerMismatchContract::evaluate(const Inputs &p_inputs) {
    Decision decision;
    if (!p_inputs.rid_valid) {
        return decision;
    }

    const bool owner_matches = p_inputs.has_owner &&
            p_inputs.owner_instance_id != 0 &&
            p_inputs.active_instance_id != 0 &&
            p_inputs.owner_instance_id == p_inputs.active_instance_id;
    if (owner_matches) {
        return decision;
    }

    decision.mismatch_detected = true;
    decision.should_attempt_release = true;
    decision.should_force_invalidate_after_release = !p_inputs.has_owner || p_inputs.owner_instance_id == 0;
    return decision;
}

bool ResourceOwnerMismatchContract::validate(const Inputs &p_inputs, const Decision &p_decision, String *r_error) {
    if (!p_decision.mismatch_detected) {
        if (p_decision.should_attempt_release || p_decision.should_force_invalidate_after_release) {
            _set_error(r_error, "non-mismatch decision cannot request release or invalidation");
            return false;
        }
        return true;
    }

    if (!p_decision.should_attempt_release) {
        _set_error(r_error, "mismatch decision must attempt release");
        return false;
    }
    if (p_decision.should_force_invalidate_after_release && !p_inputs.rid_valid) {
        _set_error(r_error, "force-invalidate path requires a valid RID");
        return false;
    }
    return true;
}

bool ResourceOwnerMismatchContract::verify_texture_device_ownership(RenderingDevice *p_rd, RID p_resource, const char *p_label) {
    if (!p_resource.is_valid()) {
        return true; // No resource to verify.
    }
    ERR_FAIL_NULL_V_MSG(p_rd, false,
            vformat("[ResourceOwnerMismatch] Null RenderingDevice when verifying texture ownership (%s)",
                    p_label ? p_label : "unknown"));
    if (!p_rd->texture_is_valid(p_resource)) {
        ERR_FAIL_V_MSG(false,
                vformat("[ResourceOwnerMismatch] Texture not owned by target RenderingDevice (%s) - "
                        "cross-device binding would cause GPU hang",
                        p_label ? p_label : "unknown"));
    }
    return true;
}

bool ResourceOwnerMismatchContract::verify_buffer_device_ownership(RenderingDevice *p_rd, RID p_resource, const char *p_label) {
    if (!p_resource.is_valid()) {
        return true; // No resource to verify.
    }
    ERR_FAIL_NULL_V_MSG(p_rd, false,
            vformat("[ResourceOwnerMismatch] Null RenderingDevice when verifying buffer ownership (%s)",
                    p_label ? p_label : "unknown"));
    if (!p_rd->buffer_is_valid(p_resource)) {
        ERR_FAIL_V_MSG(false,
                vformat("[ResourceOwnerMismatch] Buffer not owned by target RenderingDevice (%s) - "
                        "cross-device binding would cause GPU hang",
                        p_label ? p_label : "unknown"));
    }
    return true;
}

bool ResourceOwnerMismatchContract::is_device_generation_valid(RenderingDevice *p_device, uint64_t p_stored_device_id) {
    if (!p_device) {
        return false;
    }
    if (p_stored_device_id == 0) {
        return false; // No generation was recorded; treat as invalid.
    }
    return p_device->get_device_instance_id() == p_stored_device_id;
}

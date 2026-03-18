#ifndef GS_OUTPUT_COMPOSITOR_INTERFACES_H
#define GS_OUTPUT_COMPOSITOR_INTERFACES_H

#include "core/math/vector2i.h"
#include "core/templates/rid.h"
#include "servers/rendering/rendering_device.h"

// Parameters for copying final output to render target
struct OutputCopyParams {
    RID source_texture;
    RID source_depth;
    RID destination_texture;
    RID destination_depth;
    Size2i viewport_size;
    bool composite_with_destination = false;
    bool source_is_premultiplied = false;
    bool depth_test_enabled = false;
    bool depth_is_orthogonal = false;
    float z_near = 0.0f;
    float z_far = 1.0f;
    float depth_linearize_mul = 0.0f;
    float depth_linearize_add = 1.0f;
    float depth_epsilon = 0.01f;
};

// Result from output copy operation
struct OutputCopyResult {
    bool success = false;
    Size2i source_size;
    Size2i dest_size;
    String error;
};

// Parameters for framebuffer-based copy
struct FramebufferCopyParams {
    RID source_texture;
    RID framebuffer;
    Size2i viewport_size;
    bool composite_with_destination = false;
    bool source_is_premultiplied = false;
};

// Information about a framebuffer attachment's validation status
struct AttachmentValidationInfo {
    RID original_attachment;
    RD::TextureFormat original_format;
    bool is_depth = false;
};

// Pure abstract interface for output composition operations
// Handles framebuffer management, texture copying, and viewport integration
class IOutputCompositor {
public:
    virtual ~IOutputCompositor() = default;

    // Lifecycle
    virtual Error initialize(RenderingDevice *p_device) = 0;
    virtual void shutdown() = 0;
    virtual bool is_initialized() const = 0;

    // Core output operations
    virtual OutputCopyResult copy_to_render_target(const OutputCopyParams &p_params) = 0;
    virtual bool copy_to_framebuffer(const FramebufferCopyParams &p_params) = 0;

    // Framebuffer management
    virtual RID get_cached_framebuffer(RenderingDevice *p_device, const RID &p_texture) = 0;
    virtual void clear_cached_framebuffers() = 0;
    virtual void clear_viewport_blit_resources() = 0;

    // Attachment validation
    virtual bool validate_framebuffer_attachments(RenderingDevice *p_device, const Vector<RID> &p_attachments,
            Vector<AttachmentValidationInfo> &r_infos, Size2i &r_extent, RD::TextureSamples &r_samples, String &r_error) = 0;

    // State queries
    virtual bool get_last_copy_success() const = 0;
    virtual Size2i get_last_copy_source_size() const = 0;
    virtual Size2i get_last_copy_dest_size() const = 0;

    // Final render texture management
    virtual void set_final_render_texture(const RID &p_texture) = 0;
    virtual RID get_final_render_texture() const = 0;
    virtual void set_has_valid_render(bool p_valid) = 0;
    virtual bool get_has_valid_render() const = 0;

    // Implementation info
    virtual String get_name() const = 0;
};

#endif // GS_OUTPUT_COMPOSITOR_INTERFACES_H

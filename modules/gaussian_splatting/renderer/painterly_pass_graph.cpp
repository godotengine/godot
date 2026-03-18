#include "painterly_pass_graph.h"

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"

PainterlyPassGraph::PainterlyPassGraph() {
    textures.resize(TEXTURE_COUNT);
}

void PainterlyPassGraph::set_color_format(RD::DataFormat p_format) {
    if (color_format == p_format) {
        return;
    }

    color_format = p_format;

    if (textures.size() != TEXTURE_COUNT) {
        return;
    }

    _release_texture(TEXTURE_COLOR);
    _release_texture(TEXTURE_STYLIZED);
}

void PainterlyPassGraph::setup(RenderingDevice *p_rd) {
    rd = p_rd;
    if (textures.size() != TEXTURE_COUNT) {
        textures.resize(TEXTURE_COUNT);
    }
}

void PainterlyPassGraph::reset() {
    if (textures.size() != TEXTURE_COUNT) {
        return;
    }

    for (int i = 0; i < TEXTURE_COUNT; i++) {
        _release_texture(TextureSlot(i));
    }

    passes.clear();
}

void PainterlyPassGraph::configure(const Size2i &p_render_size, float p_scale, bool p_enable_stylization, bool p_low_end_mode) {
    ERR_FAIL_NULL_MSG(rd, "PainterlyPassGraph::configure requires a valid RenderingDevice");

    if (textures.size() != TEXTURE_COUNT) {
        textures.resize(TEXTURE_COUNT);
    }

    requested_size.x = MAX(1, p_render_size.x);
    requested_size.y = MAX(1, p_render_size.y);

    low_end_mode = p_low_end_mode;

    float applied_scale = CLAMP(p_scale, 0.25f, 1.0f);
    if (low_end_mode) {
        applied_scale = MIN(applied_scale, 0.5f);
    }

    internal_scale = applied_scale;
    stylization_enabled = p_enable_stylization && !low_end_mode;

    Size2i desired_size;
    desired_size.x = MAX(1, int(Math::round(requested_size.x * internal_scale)));
    desired_size.y = MAX(1, int(Math::round(requested_size.y * internal_scale)));

    if (desired_size != internal_size) {
        internal_size = desired_size;
        for (int i = 0; i < TEXTURE_COUNT; i++) {
            _release_texture(TextureSlot(i));
        }
    }

    RD::DataFormat desired_color_format = color_format;
    _ensure_texture(TEXTURE_COLOR, desired_color_format,
            BitField<RD::TextureUsageBits>(RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT |
                    RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT));
    RD::DataFormat effective_color_format = textures[TEXTURE_COLOR].format.format;
    _ensure_texture(TEXTURE_DEPTH, RD::DATA_FORMAT_R32_SFLOAT,
            BitField<RD::TextureUsageBits>(RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT |
                    RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT));

    if (stylization_enabled) {
        _ensure_texture(TEXTURE_EDGE, RD::DATA_FORMAT_R8G8B8A8_UNORM,
                BitField<RD::TextureUsageBits>(RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT |
                        RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT));
        _ensure_texture(TEXTURE_STYLIZED, effective_color_format,
                BitField<RD::TextureUsageBits>(RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT |
                        RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT));
    } else {
        _release_texture(TEXTURE_EDGE);
        _release_texture(TEXTURE_STYLIZED);
    }

    _rebuild_passes();
}

RID PainterlyPassGraph::get_texture(TextureSlot p_slot) const {
    if (p_slot < 0 || p_slot >= textures.size()) {
        return RID();
    }
    return textures[p_slot].texture;
}

RID PainterlyPassGraph::get_shared_texture(TextureSlot p_slot) const {
    if (p_slot < 0 || p_slot >= textures.size()) {
        return RID();
    }
    const TextureInfo &info = textures[p_slot];
    return info.shared_texture.is_valid() ? info.shared_texture : info.texture;
}

RenderingDevice *PainterlyPassGraph::get_shared_texture_owner(TextureSlot p_slot) const {
    if (p_slot < 0 || p_slot >= textures.size()) {
        return nullptr;
    }
    const TextureInfo &info = textures[p_slot];
    if (info.shared_texture.is_valid() && info.is_shared_with_main) {
        return RenderingDevice::get_singleton();
    }
    return rd;
}

const PainterlyPassGraph::PassNode *PainterlyPassGraph::find_pass(PassId p_id) const {
    for (int i = 0; i < passes.size(); i++) {
        if (passes[i].id == p_id) {
            return &passes[i];
        }
    }
    return nullptr;
}

void PainterlyPassGraph::_ensure_texture(TextureSlot p_slot, RD::DataFormat p_format, BitField<RD::TextureUsageBits> p_usage) {
    ERR_FAIL_NULL_MSG(rd, "PainterlyPassGraph requires a valid RenderingDevice to allocate textures");

    if (p_slot >= textures.size()) {
        textures.resize(TEXTURE_COUNT);
    }

    TextureInfo *w = textures.ptrw();
    TextureInfo &info = w[p_slot];

    RD::DataFormat requested_format = p_format;
    if (rd && !rd->texture_is_format_supported_for_usage(requested_format, p_usage)) {
        requested_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    }

    bool usage_changed = static_cast<uint64_t>(info.usage.get_shared(p_usage)) != static_cast<uint64_t>(p_usage);
    RenderingDevice *primary_device = RenderingDevice::get_singleton();
    bool wants_primary_share = rd && primary_device && primary_device != rd;
    bool currently_shared_with_primary = info.is_shared_with_main && info.shared_texture.is_valid() && info.shared_texture != info.texture;
    bool recreate = !info.texture.is_valid() || info.size != internal_size || info.format.format != requested_format || usage_changed;

    if (!recreate) {
        if (wants_primary_share && !currently_shared_with_primary && info.texture.is_valid()) {
            RD::TextureFormat share_format = info.format;
            share_format.usage_bits = static_cast<uint32_t>(info.usage);
            RD::TextureView share_view;

            RID primary_texture = primary_device->texture_create(share_format, share_view);
            if (primary_texture.is_valid()) {
                primary_device->set_resource_name(primary_texture, "GS_PainterlyPassGraph_SharedTexture_Upgrade");
                uint64_t driver_handle = primary_device->get_driver_resource(RenderingDevice::DRIVER_RESOURCE_TEXTURE, primary_texture, 0);
                if (driver_handle != 0) {
                    RID local_reference = rd->texture_create_from_extension(share_format.texture_type, share_format.format, share_format.samples,
                            share_format.usage_bits, driver_handle, share_format.width, share_format.height, share_format.depth, share_format.array_layers, share_format.mipmaps);
                    if (local_reference.is_valid()) {
                        rd->set_resource_name(local_reference, "GS_PainterlyPassGraph_LocalTextureRef_Upgrade");
                        RID old_texture = info.texture;
                        RID old_shared = info.shared_texture;
                        bool old_shared_with_main = info.is_shared_with_main;

                        info.texture = local_reference;
                        info.shared_texture = primary_texture;
                        info.is_shared_with_main = true;
                        info.valid = true;

                        if (old_texture.is_valid() && rd && rd->texture_is_valid(old_texture)) {
                            rd->free(old_texture);
                        }
                        if (old_shared.is_valid() && old_shared != old_texture) {
                            if (old_shared_with_main) {
                                if (RenderingDevice *main_rd = RenderingDevice::get_singleton()) {
                                    if (main_rd->texture_is_valid(old_shared)) {
                                        main_rd->free(old_shared);
                                    }
                                }
                            } else if (rd && rd->texture_is_valid(old_shared)) {
                                rd->free(old_shared);
                            }
                        }

                        version++;
                        return;
                    } else if (primary_device->texture_is_valid(primary_texture)) {
                        primary_device->free(primary_texture);
                    }
                } else if (primary_device->texture_is_valid(primary_texture)) {
                    primary_device->free(primary_texture);
                }
            }
        }

        return;
    }

    bool had_texture = info.texture.is_valid() || info.shared_texture.is_valid();
    _release_texture(p_slot);

    RD::TextureFormat format;
    format.format = requested_format;
    format.width = internal_size.x;
    format.height = internal_size.y;
    format.depth = 1;
    format.array_layers = 1;
    format.mipmaps = 1;
    format.texture_type = RD::TEXTURE_TYPE_2D;
    format.samples = RD::TEXTURE_SAMPLES_1;
    format.usage_bits = static_cast<uint32_t>(p_usage);

    RD::TextureView view;
    info.shared_texture = RID();
    info.is_shared_with_main = false;
    info.texture = RID();

    bool shared_with_primary = false;

    if (wants_primary_share) {
        RID primary_texture = primary_device->texture_create(format, view);
        if (primary_texture.is_valid()) {
            primary_device->set_resource_name(primary_texture, "GS_PainterlyPassGraph_SharedTexture");
            uint64_t driver_handle = primary_device->get_driver_resource(RenderingDevice::DRIVER_RESOURCE_TEXTURE, primary_texture, 0);
            if (driver_handle != 0) {
                RID local_reference = rd->texture_create_from_extension(format.texture_type, format.format, format.samples,
                        format.usage_bits, driver_handle, format.width, format.height, format.depth, format.array_layers, format.mipmaps);
                if (local_reference.is_valid()) {
                    rd->set_resource_name(local_reference, "GS_PainterlyPassGraph_LocalTextureRef");
                    info.texture = local_reference;
                    info.shared_texture = primary_texture;
                    info.is_shared_with_main = true;
                    shared_with_primary = true;
                } else if (primary_device->texture_is_valid(primary_texture)) {
                    primary_device->free(primary_texture);
                }
            } else if (primary_device->texture_is_valid(primary_texture)) {
                primary_device->free(primary_texture);
            }
        }
    }

    if (!shared_with_primary && rd) {
        info.texture = rd->texture_create(format, view);
        if (info.texture.is_valid()) {
            rd->set_resource_name(info.texture, "GS_PainterlyPassGraph_Texture");
            info.shared_texture = info.texture;
            info.is_shared_with_main = false;
        }
    }

    info.format = format;
    info.usage = p_usage;
    info.size = internal_size;
    info.valid = info.texture.is_valid();

    if (info.valid && !had_texture) {
        version++;
    }
}

void PainterlyPassGraph::_release_texture(TextureSlot p_slot) {
    if (p_slot >= textures.size()) {
        return;
    }

    TextureInfo *w = textures.ptrw();
    TextureInfo &info = w[p_slot];

    bool released = info.texture.is_valid() || info.shared_texture.is_valid();

    if (info.texture.is_valid() && rd && rd->texture_is_valid(info.texture)) {
        rd->free(info.texture);
    }

    if (info.shared_texture.is_valid() && info.shared_texture != info.texture) {
        // If shared with main device, get fresh singleton pointer to avoid use-after-free
        if (info.is_shared_with_main) {
            if (RenderingDevice *main_rd = RenderingDevice::get_singleton()) {
                if (main_rd->texture_is_valid(info.shared_texture)) {
                    main_rd->free(info.shared_texture);
                }
            }
        } else if (rd && rd->texture_is_valid(info.shared_texture)) {
            // Otherwise it's our local device
            rd->free(info.shared_texture);
        }
    }

    info.texture = RID();
    info.shared_texture = RID();
    info.is_shared_with_main = false;
    info.valid = false;

    if (released) {
        version++;
    }
}

void PainterlyPassGraph::_rebuild_passes() {
    passes.clear();

    PassNode gbuffer;
    gbuffer.id = PASS_GBUFFER;
    gbuffer.type = PASS_TYPE_GRAPHICS;
    gbuffer.name = StringName("painterly_gbuffer");
    gbuffer.inputs.clear();
    gbuffer.outputs.clear();
    gbuffer.outputs.push_back(TEXTURE_COLOR);
    gbuffer.outputs.push_back(TEXTURE_DEPTH);
    passes.push_back(gbuffer);

    PassNode sobel;
    sobel.id = PASS_SOBEL_EDGES;
    sobel.type = PASS_TYPE_COMPUTE;
    sobel.name = StringName("painterly_sobel_edges");
    sobel.inputs.clear();
    sobel.outputs.clear();
    sobel.enabled = stylization_enabled;
    sobel.inputs.push_back(TEXTURE_COLOR);
    if (stylization_enabled) {
        sobel.outputs.push_back(TEXTURE_EDGE);
    }
    passes.push_back(sobel);

    PassNode brush;
    brush.id = PASS_BRUSH_ACCUMULATION;
    brush.type = PASS_TYPE_COMPUTE;
    brush.name = StringName("painterly_brush_accumulation");
    brush.inputs.clear();
    brush.outputs.clear();
    brush.enabled = stylization_enabled;
    if (stylization_enabled) {
        brush.inputs.push_back(TEXTURE_COLOR);
        brush.inputs.push_back(TEXTURE_EDGE);
        brush.outputs.push_back(TEXTURE_STYLIZED);
    } else {
        brush.inputs.push_back(TEXTURE_COLOR);
        brush.outputs.push_back(TEXTURE_COLOR);
    }
    passes.push_back(brush);
}

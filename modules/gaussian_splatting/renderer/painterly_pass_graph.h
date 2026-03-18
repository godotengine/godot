#ifndef PAINTERLY_PASS_GRAPH_H
#define PAINTERLY_PASS_GRAPH_H

#include "core/math/vector2i.h"
#include "core/string/string_name.h"
#include "core/templates/bit_field.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"

class PainterlyPassGraph {
public:
    enum TextureSlot {
        TEXTURE_COLOR = 0,
        TEXTURE_DEPTH,
        TEXTURE_EDGE,
        TEXTURE_STYLIZED,
        TEXTURE_COUNT
    };

    enum PassId {
        PASS_GBUFFER = 0,
        PASS_SOBEL_EDGES,
        PASS_BRUSH_ACCUMULATION
    };

    enum PassType {
        PASS_TYPE_GRAPHICS,
        PASS_TYPE_COMPUTE
    };

    struct PassNode {
        PassId id = PASS_GBUFFER;
        PassType type = PASS_TYPE_GRAPHICS;
        StringName name;
        Vector<TextureSlot> inputs;
        Vector<TextureSlot> outputs;
        bool enabled = true;
    };

    struct TextureInfo {
        Size2i size = Size2i();
        RD::TextureFormat format;
        BitField<RD::TextureUsageBits> usage = {};
        RID texture; // Local RenderingDevice handle used by compute passes
        RID shared_texture; // Primary RenderingDevice handle for viewport compositing
        bool is_shared_with_main = false; // True if shared_texture belongs to main RenderingDevice
        bool valid = false;
    };

private:
    RenderingDevice *rd = nullptr;
    Size2i requested_size = Size2i(1, 1);
    Size2i internal_size = Size2i(1, 1);
    float internal_scale = 1.0f;
    bool stylization_enabled = true;
    bool low_end_mode = false;
    RD::DataFormat color_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;

    uint64_t version = 0;

    Vector<TextureInfo> textures;
    Vector<PassNode> passes;

    void _ensure_texture(TextureSlot p_slot, RD::DataFormat p_format, BitField<RD::TextureUsageBits> p_usage);
    void _release_texture(TextureSlot p_slot);
    void _rebuild_passes();

public:
    PainterlyPassGraph();

    void setup(RenderingDevice *p_rd);
    void reset();

    void configure(const Size2i &p_render_size, float p_scale, bool p_enable_stylization, bool p_low_end_mode);
    void set_color_format(RD::DataFormat p_format);
    RD::DataFormat get_color_format() const { return color_format; }

    Size2i get_internal_size() const { return internal_size; }
    Size2i get_requested_size() const { return requested_size; }
    float get_internal_scale() const { return internal_scale; }
    bool is_stylization_enabled() const { return stylization_enabled; }
    bool is_low_end_mode() const { return low_end_mode; }
    bool is_ready() const { return rd != nullptr && textures.size() == TEXTURE_COUNT && textures[TEXTURE_COLOR].texture.is_valid(); }

    RID get_texture(TextureSlot p_slot) const;
    RID get_shared_texture(TextureSlot p_slot) const;
    RenderingDevice *get_shared_texture_owner(TextureSlot p_slot) const;
    const TextureInfo &get_texture_info(TextureSlot p_slot) const { return textures[p_slot]; }

    const Vector<PassNode> &get_passes() const { return passes; }
    const PassNode *find_pass(PassId p_id) const;

    uint64_t get_version() const { return version; }
};

#endif // PAINTERLY_PASS_GRAPH_H

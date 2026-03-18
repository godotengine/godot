#ifndef GAUSSIAN_SPLAT_WORLD_H
#define GAUSSIAN_SPLAT_WORLD_H

#include "core/io/resource.h"
#include "core/math/aabb.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

#include "gaussian_data.h"
#include "../renderer/gaussian_splat_renderer.h"

class GaussianSplatWorld : public Resource {
    GDCLASS(GaussianSplatWorld, Resource);
    RES_BASE_EXTENSION("gsplatworld");

private:
    Ref<GaussianData> gaussian_data;
    Vector<GaussianSplatRenderer::StaticChunk> static_chunks;
    AABB bounds;
    Dictionary metadata;

protected:
    static void _bind_methods();
    bool _get(const StringName &p_name, Variant &r_ret) const;
    void _get_property_list(List<PropertyInfo> *p_list) const;

public:
    void set_gaussian_data(const Ref<GaussianData> &p_data);
    Ref<GaussianData> get_gaussian_data() const { return gaussian_data; }

    void set_bounds(const AABB &p_bounds);
    AABB get_bounds() const { return bounds; }

    void set_metadata(const Dictionary &p_metadata);
    Dictionary get_metadata() const { return metadata; }

    void set_static_chunks(const Vector<GaussianSplatRenderer::StaticChunk> &p_chunks);
    const Vector<GaussianSplatRenderer::StaticChunk> &get_static_chunks() const { return static_chunks; }

    int get_chunk_count() const;
    PackedInt32Array get_chunk_sizes() const;
    Array get_chunk_aabbs() const;

    void clear();
};

#endif // GAUSSIAN_SPLAT_WORLD_H

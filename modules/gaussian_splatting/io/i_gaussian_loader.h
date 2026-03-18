#ifndef I_GAUSSIAN_LOADER_H
#define I_GAUSSIAN_LOADER_H

#include "core/io/resource_loader.h"
#include "../core/gaussian_splat_asset.h"

class IGaussianLoader : public RefCounted {
    GDCLASS(IGaussianLoader, RefCounted);

protected:
    static void _bind_methods() {}

public:
    virtual ~IGaussianLoader() {}

    // Pure virtual interface for format-specific loaders
    virtual bool can_load(const String &p_path) const = 0;
    virtual Error load(const String &p_path, Ref<GaussianSplatAsset> &r_asset) = 0;
    virtual String get_format_name() const = 0;
    virtual Vector<String> get_recognized_extensions() const = 0;
};

// Resource format loader for Godot's resource system integration
class ResourceFormatLoaderGaussianSplat : public ResourceFormatLoader {
public:
    virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "",
                               Error *r_error = nullptr, bool p_use_sub_threads = false,
                               float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;

    virtual void get_recognized_extensions(List<String> *p_extensions) const override;
    virtual bool handles_type(const String &p_type) const override;
    virtual String get_resource_type(const String &p_path) const override;
};

#endif // I_GAUSSIAN_LOADER_H

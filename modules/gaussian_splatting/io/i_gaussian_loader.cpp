#include "i_gaussian_loader.h"

#include "core/object/class_db.h"

Ref<Resource> ResourceFormatLoaderGaussianSplat::load(const String &p_path, const String &p_original_path,
                                                      Error *r_error, bool p_use_sub_threads,
                                                      float *r_progress, CacheMode p_cache_mode) {
    if (r_progress) {
        *r_progress = 0.0f;
    }

    const String extension = p_path.get_extension().to_lower();
    if (extension != "ply" && extension != "spz") {
        if (r_error) {
            *r_error = ERR_FILE_UNRECOGNIZED;
        }
        return Ref<Resource>();
    }

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();

    const Error err = asset->load_from_file(p_path);
    if (r_error) {
        *r_error = err;
    }

    if (err != OK) {
        return Ref<Resource>();
    }

    String source_path = p_original_path.is_empty() ? p_path : p_original_path;
    asset->set_source_path(source_path);

    Dictionary import_metadata = asset->get_import_metadata();
    import_metadata[StringName("resource_path")] = p_path;
    import_metadata[StringName("source_path")] = source_path;
    import_metadata[StringName("resource_loader")] = String("ResourceFormatLoaderGaussianSplat");
    import_metadata[StringName("cache_mode")] = (int)p_cache_mode;
    import_metadata[StringName("used_sub_threads")] = p_use_sub_threads;
    import_metadata[StringName("loaded_via_resource_loader")] = true;
    asset->set_import_metadata(import_metadata);

    if (r_progress) {
        *r_progress = 1.0f;
    }

    return asset;
}

void ResourceFormatLoaderGaussianSplat::get_recognized_extensions(List<String> *p_extensions) const {
    if (p_extensions == nullptr) {
        return;
    }

    p_extensions->push_back("ply");
    p_extensions->push_back("spz");
}

bool ResourceFormatLoaderGaussianSplat::handles_type(const String &p_type) const {
    if (p_type == "GaussianSplatAsset") {
        return true;
    }

    return ClassDB::is_parent_class(p_type, "GaussianSplatAsset");
}

String ResourceFormatLoaderGaussianSplat::get_resource_type(const String &p_path) const {
    String ext = p_path.get_extension().to_lower();
    if (ext == "ply" || ext == "spz") {
        return "GaussianSplatAsset";
    }
    return "";
}
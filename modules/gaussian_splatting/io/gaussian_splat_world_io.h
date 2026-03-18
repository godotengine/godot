#ifndef GAUSSIAN_SPLAT_WORLD_IO_H
#define GAUSSIAN_SPLAT_WORLD_IO_H

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"

class ResourceFormatLoaderGaussianSplatWorld : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "",
			Error *r_error = nullptr, bool p_use_sub_threads = false,
			float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
};

class ResourceFormatSaverGaussianSplatWorld : public ResourceFormatSaver {
public:
	virtual Error save(const Ref<Resource> &p_resource, const String &p_path,
			uint32_t p_flags = 0) override;
	virtual void get_recognized_extensions(const Ref<Resource> &p_resource,
			List<String> *p_extensions) const override;
	virtual bool recognize(const Ref<Resource> &p_resource) const override;
	virtual bool recognize_path(const Ref<Resource> &p_resource, const String &p_path) const override;
};

#endif // GAUSSIAN_SPLAT_WORLD_IO_H

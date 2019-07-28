#include "rendering_device.h"

RenderingDevice *RenderingDevice::singleton = NULL;

RenderingDevice *RenderingDevice::get_singleton() {
	return singleton;
}

RenderingDevice::ShaderCompileFunction RenderingDevice::compile_function = NULL;
RenderingDevice::ShaderCacheFunction RenderingDevice::cache_function = NULL;

void RenderingDevice::shader_set_compile_function(ShaderCompileFunction p_function) {
	compile_function = p_function;
}
void RenderingDevice::shader_set_cache_function(ShaderCacheFunction p_function) {
	cache_function = p_function;
}

PoolVector<uint8_t> RenderingDevice::shader_compile_from_source(ShaderStage p_stage, const String &p_source_code, ShaderLanguage p_language, String *r_error, bool p_allow_cache) {
	if (p_allow_cache && cache_function) {
		PoolVector<uint8_t> cache = cache_function(p_stage, p_source_code, p_language);
		if (cache.size()) {
			return cache;
		}
	}

	ERR_FAIL_COND_V(!compile_function, PoolVector<uint8_t>());

	return compile_function(p_stage, p_source_code, p_language, r_error);
}

RenderingDevice::RenderingDevice() {

	ShaderCompileFunction compile_function;
	ShaderCacheFunction cache_function;

	singleton = this;
}

#ifndef GAUSSIAN_SPLAT_SOURCE_PATH_H
#define GAUSSIAN_SPLAT_SOURCE_PATH_H

#include "gaussian_splat_asset.h"

namespace GaussianSplatSourcePath {

static inline String get_asset_source_path(const Ref<GaussianSplatAsset> &p_asset) {
	if (p_asset.is_null()) {
		return String();
	}

	String source_path = p_asset->get_source_path();
	if (!source_path.is_empty()) {
		return source_path;
	}

	const Dictionary metadata = p_asset->get_import_metadata();
	if (metadata.has(StringName("source_file"))) {
		return String(metadata[StringName("source_file")]);
	}
	if (metadata.has(StringName("runtime_load_source_path"))) {
		return String(metadata[StringName("runtime_load_source_path")]);
	}
	return String();
}

static inline String resolve_primary_source_path(const Ref<GaussianSplatAsset> &p_asset,
		const String &p_fallback_file_path = String()) {
	const String asset_source_path = get_asset_source_path(p_asset);
	if (!asset_source_path.is_empty()) {
		return asset_source_path;
	}
	return p_fallback_file_path;
}

} // namespace GaussianSplatSourcePath

#endif // GAUSSIAN_SPLAT_SOURCE_PATH_H

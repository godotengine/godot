#ifdef TOOLS_ENABLED

#include "gaussian_resource_preview_generator.h"

#include "core/io/resource_loader.h"
#include "core/math/math_funcs.h"
#include "core/object/object.h"
#include "scene/resources/texture.h"

#include "../core/gaussian_splat_asset.h"
#include "gaussian_thumbnail_generator.h"

void GaussianSplatAssetPreviewGenerator::_bind_methods() {}

bool GaussianSplatAssetPreviewGenerator::handles(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "GaussianSplatAsset");
}

Ref<Texture2D> GaussianSplatAssetPreviewGenerator::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<GaussianSplatAsset> asset = p_from;
	if (asset.is_null()) {
		return Ref<Texture2D>();
	}

	Ref<Texture2D> existing_thumbnail = asset->get_thumbnail();
	if (existing_thumbnail.is_valid()) {
		p_metadata[StringName("gaussian_preview_source")] = String("stored_thumbnail");
		return existing_thumbnail;
	}

	if (thumbnail_generator.is_null()) {
		thumbnail_generator.instantiate();
	}
	if (thumbnail_generator.is_null()) {
		return Ref<Texture2D>();
	}

	const int thumbnail_size = MAX(64, int(MAX(p_size.x, p_size.y)));
	p_metadata[StringName("gaussian_preview_source")] = String("generated_thumbnail");
	return thumbnail_generator->generate_thumbnail(asset, thumbnail_size, GaussianThumbnailGenerator::THUMBNAIL_STYLE_COLOR);
}

Ref<Texture2D> GaussianSplatAssetPreviewGenerator::generate_from_path(const String &p_path, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<GaussianSplatAsset> asset = ResourceLoader::load(p_path, "GaussianSplatAsset");
	if (asset.is_null()) {
		return Ref<Texture2D>();
	}
	p_metadata[StringName("source_path")] = p_path;
	return generate(asset, p_size, p_metadata);
}

bool GaussianSplatAssetPreviewGenerator::generate_small_preview_automatically() const {
	return true;
}

#endif // TOOLS_ENABLED

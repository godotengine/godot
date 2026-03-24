#ifndef GAUSSIAN_RESOURCE_PREVIEW_GENERATOR_H
#define GAUSSIAN_RESOURCE_PREVIEW_GENERATOR_H

#ifdef TOOLS_ENABLED

#include "editor/inspector/editor_resource_preview.h"

class GaussianThumbnailGenerator;

class GaussianSplatAssetPreviewGenerator : public EditorResourcePreviewGenerator {
	GDCLASS(GaussianSplatAssetPreviewGenerator, EditorResourcePreviewGenerator);

	mutable Ref<GaussianThumbnailGenerator> thumbnail_generator;

protected:
	static void _bind_methods();

public:
	virtual bool handles(const String &p_type) const override;
	virtual Ref<Texture2D> generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const override;
	virtual Ref<Texture2D> generate_from_path(const String &p_path, const Size2 &p_size, Dictionary &p_metadata) const override;
	virtual bool generate_small_preview_automatically() const override;
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_RESOURCE_PREVIEW_GENERATOR_H

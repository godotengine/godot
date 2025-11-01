/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_VIDEO_RAW_UTILS_H
#define SPA_VIDEO_RAW_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/pod/parser.h>
#include <spa/pod/builder.h>
#include <spa/param/video/raw.h>

static inline int
spa_format_video_raw_parse(const struct spa_pod *format,
			   struct spa_video_info_raw *info)
{
	info->flags = SPA_VIDEO_FLAG_NONE;
	const struct spa_pod_prop *mod_prop;
	if ((mod_prop = spa_pod_find_prop (format, NULL, SPA_FORMAT_VIDEO_modifier)) != NULL) {
		info->flags |= SPA_VIDEO_FLAG_MODIFIER;
		if ((mod_prop->flags & SPA_POD_PROP_FLAG_DONT_FIXATE) == SPA_POD_PROP_FLAG_DONT_FIXATE)
			info->flags |= SPA_VIDEO_FLAG_MODIFIER_FIXATION_REQUIRED;
	}

	return spa_pod_parse_object(format,
		SPA_TYPE_OBJECT_Format, NULL,
		SPA_FORMAT_VIDEO_format,		SPA_POD_OPT_Id(&info->format),
		SPA_FORMAT_VIDEO_modifier,		SPA_POD_OPT_Long(&info->modifier),
		SPA_FORMAT_VIDEO_size,			SPA_POD_OPT_Rectangle(&info->size),
		SPA_FORMAT_VIDEO_framerate,		SPA_POD_OPT_Fraction(&info->framerate),
		SPA_FORMAT_VIDEO_maxFramerate,		SPA_POD_OPT_Fraction(&info->max_framerate),
		SPA_FORMAT_VIDEO_views,			SPA_POD_OPT_Int(&info->views),
		SPA_FORMAT_VIDEO_interlaceMode,		SPA_POD_OPT_Id(&info->interlace_mode),
		SPA_FORMAT_VIDEO_pixelAspectRatio,	SPA_POD_OPT_Fraction(&info->pixel_aspect_ratio),
		SPA_FORMAT_VIDEO_multiviewMode,		SPA_POD_OPT_Id(&info->multiview_mode),
		SPA_FORMAT_VIDEO_multiviewFlags,	SPA_POD_OPT_Id(&info->multiview_flags),
		SPA_FORMAT_VIDEO_chromaSite,		SPA_POD_OPT_Id(&info->chroma_site),
		SPA_FORMAT_VIDEO_colorRange,		SPA_POD_OPT_Id(&info->color_range),
		SPA_FORMAT_VIDEO_colorMatrix,		SPA_POD_OPT_Id(&info->color_matrix),
		SPA_FORMAT_VIDEO_transferFunction,	SPA_POD_OPT_Id(&info->transfer_function),
		SPA_FORMAT_VIDEO_colorPrimaries,	SPA_POD_OPT_Id(&info->color_primaries));
}

static inline struct spa_pod *
spa_format_video_raw_build(struct spa_pod_builder *builder, uint32_t id,
			   const struct spa_video_info_raw *info)
{
	struct spa_pod_frame f;
	spa_pod_builder_push_object(builder, &f, SPA_TYPE_OBJECT_Format, id);
	spa_pod_builder_add(builder,
			SPA_FORMAT_mediaType,		SPA_POD_Id(SPA_MEDIA_TYPE_video),
			SPA_FORMAT_mediaSubtype,	SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
			0);
	if (info->format != SPA_VIDEO_FORMAT_UNKNOWN)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_format,	SPA_POD_Id(info->format), 0);
	if (info->size.width != 0 && info->size.height != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_size,		SPA_POD_Rectangle(&info->size), 0);
	if (info->framerate.denom != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_framerate,	SPA_POD_Fraction(&info->framerate), 0);
	if (info->modifier != 0 || info->flags & SPA_VIDEO_FLAG_MODIFIER) {
		spa_pod_builder_prop(builder,
			SPA_FORMAT_VIDEO_modifier,	SPA_POD_PROP_FLAG_MANDATORY);
		spa_pod_builder_long(builder,           info->modifier);
	}
	if (info->max_framerate.denom != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_maxFramerate,	SPA_POD_Fraction(&info->max_framerate), 0);
	if (info->views != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_views,		SPA_POD_Int(info->views), 0);
	if (info->interlace_mode != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_interlaceMode,	SPA_POD_Id(info->interlace_mode), 0);
	if (info->pixel_aspect_ratio.denom != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_pixelAspectRatio, SPA_POD_Fraction(&info->pixel_aspect_ratio), 0);
	if (info->multiview_mode != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_multiviewMode,	SPA_POD_Id(info->multiview_mode), 0);
	if (info->multiview_flags != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_multiviewFlags,SPA_POD_Id(info->multiview_flags), 0);
	if (info->chroma_site != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_chromaSite,	SPA_POD_Id(info->chroma_site), 0);
	if (info->color_range != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_colorRange,	SPA_POD_Id(info->color_range), 0);
	if (info->color_matrix != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_colorMatrix,	SPA_POD_Id(info->color_matrix), 0);
	if (info->transfer_function != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_transferFunction,SPA_POD_Id(info->transfer_function), 0);
	if (info->color_primaries != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_colorPrimaries,SPA_POD_Id(info->color_primaries), 0);
	return (struct spa_pod*)spa_pod_builder_pop(builder, &f);
}

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_VIDEO_RAW_UTILS_H */

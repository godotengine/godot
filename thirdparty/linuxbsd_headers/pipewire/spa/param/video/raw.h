/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_VIDEO_RAW_H
#define SPA_VIDEO_RAW_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/format.h>
#include <spa/param/video/chroma.h>
#include <spa/param/video/color.h>
#include <spa/param/video/multiview.h>

#define SPA_VIDEO_MAX_PLANES 4
#define SPA_VIDEO_MAX_COMPONENTS 4

/**
 * Video formats
 *
 * The components are in general described in big-endian order. There are some
 * exceptions (e.g. RGB15 and RGB16) which use the host endianness.
 *
 * Most of the formats are identical to their GStreamer equivalent. See the
 * GStreamer video formats documentation for more details:
 *
 * https://gstreamer.freedesktop.org/documentation/additional/design/mediatype-video-raw.html#formats
 */
enum spa_video_format {
	SPA_VIDEO_FORMAT_UNKNOWN,
	SPA_VIDEO_FORMAT_ENCODED,

	SPA_VIDEO_FORMAT_I420,
	SPA_VIDEO_FORMAT_YV12,
	SPA_VIDEO_FORMAT_YUY2,
	SPA_VIDEO_FORMAT_UYVY,
	SPA_VIDEO_FORMAT_AYUV,
	SPA_VIDEO_FORMAT_RGBx,
	SPA_VIDEO_FORMAT_BGRx,
	SPA_VIDEO_FORMAT_xRGB,
	SPA_VIDEO_FORMAT_xBGR,
	SPA_VIDEO_FORMAT_RGBA,
	SPA_VIDEO_FORMAT_BGRA,
	SPA_VIDEO_FORMAT_ARGB,
	SPA_VIDEO_FORMAT_ABGR,
	SPA_VIDEO_FORMAT_RGB,
	SPA_VIDEO_FORMAT_BGR,
	SPA_VIDEO_FORMAT_Y41B,
	SPA_VIDEO_FORMAT_Y42B,
	SPA_VIDEO_FORMAT_YVYU,
	SPA_VIDEO_FORMAT_Y444,
	SPA_VIDEO_FORMAT_v210,
	SPA_VIDEO_FORMAT_v216,
	SPA_VIDEO_FORMAT_NV12,
	SPA_VIDEO_FORMAT_NV21,
	SPA_VIDEO_FORMAT_GRAY8,
	SPA_VIDEO_FORMAT_GRAY16_BE,
	SPA_VIDEO_FORMAT_GRAY16_LE,
	SPA_VIDEO_FORMAT_v308,
	SPA_VIDEO_FORMAT_RGB16,
	SPA_VIDEO_FORMAT_BGR16,
	SPA_VIDEO_FORMAT_RGB15,
	SPA_VIDEO_FORMAT_BGR15,
	SPA_VIDEO_FORMAT_UYVP,
	SPA_VIDEO_FORMAT_A420,
	SPA_VIDEO_FORMAT_RGB8P,
	SPA_VIDEO_FORMAT_YUV9,
	SPA_VIDEO_FORMAT_YVU9,
	SPA_VIDEO_FORMAT_IYU1,
	SPA_VIDEO_FORMAT_ARGB64,
	SPA_VIDEO_FORMAT_AYUV64,
	SPA_VIDEO_FORMAT_r210,
	SPA_VIDEO_FORMAT_I420_10BE,
	SPA_VIDEO_FORMAT_I420_10LE,
	SPA_VIDEO_FORMAT_I422_10BE,
	SPA_VIDEO_FORMAT_I422_10LE,
	SPA_VIDEO_FORMAT_Y444_10BE,
	SPA_VIDEO_FORMAT_Y444_10LE,
	SPA_VIDEO_FORMAT_GBR,
	SPA_VIDEO_FORMAT_GBR_10BE,
	SPA_VIDEO_FORMAT_GBR_10LE,
	SPA_VIDEO_FORMAT_NV16,
	SPA_VIDEO_FORMAT_NV24,
	SPA_VIDEO_FORMAT_NV12_64Z32,
	SPA_VIDEO_FORMAT_A420_10BE,
	SPA_VIDEO_FORMAT_A420_10LE,
	SPA_VIDEO_FORMAT_A422_10BE,
	SPA_VIDEO_FORMAT_A422_10LE,
	SPA_VIDEO_FORMAT_A444_10BE,
	SPA_VIDEO_FORMAT_A444_10LE,
	SPA_VIDEO_FORMAT_NV61,
	SPA_VIDEO_FORMAT_P010_10BE,
	SPA_VIDEO_FORMAT_P010_10LE,
	SPA_VIDEO_FORMAT_IYU2,
	SPA_VIDEO_FORMAT_VYUY,
	SPA_VIDEO_FORMAT_GBRA,
	SPA_VIDEO_FORMAT_GBRA_10BE,
	SPA_VIDEO_FORMAT_GBRA_10LE,
	SPA_VIDEO_FORMAT_GBR_12BE,
	SPA_VIDEO_FORMAT_GBR_12LE,
	SPA_VIDEO_FORMAT_GBRA_12BE,
	SPA_VIDEO_FORMAT_GBRA_12LE,
	SPA_VIDEO_FORMAT_I420_12BE,
	SPA_VIDEO_FORMAT_I420_12LE,
	SPA_VIDEO_FORMAT_I422_12BE,
	SPA_VIDEO_FORMAT_I422_12LE,
	SPA_VIDEO_FORMAT_Y444_12BE,
	SPA_VIDEO_FORMAT_Y444_12LE,

	SPA_VIDEO_FORMAT_RGBA_F16,
	SPA_VIDEO_FORMAT_RGBA_F32,

	SPA_VIDEO_FORMAT_xRGB_210LE,	/**< 32-bit x:R:G:B 2:10:10:10 little endian */
	SPA_VIDEO_FORMAT_xBGR_210LE,	/**< 32-bit x:B:G:R 2:10:10:10 little endian */
	SPA_VIDEO_FORMAT_RGBx_102LE,	/**< 32-bit R:G:B:x 10:10:10:2 little endian */
	SPA_VIDEO_FORMAT_BGRx_102LE,	/**< 32-bit B:G:R:x 10:10:10:2 little endian */
	SPA_VIDEO_FORMAT_ARGB_210LE,	/**< 32-bit A:R:G:B 2:10:10:10 little endian */
	SPA_VIDEO_FORMAT_ABGR_210LE,	/**< 32-bit A:B:G:R 2:10:10:10 little endian */
	SPA_VIDEO_FORMAT_RGBA_102LE,	/**< 32-bit R:G:B:A 10:10:10:2 little endian */
	SPA_VIDEO_FORMAT_BGRA_102LE,	/**< 32-bit B:G:R:A 10:10:10:2 little endian */

	/* Aliases */
	SPA_VIDEO_FORMAT_DSP_F32 = SPA_VIDEO_FORMAT_RGBA_F32,
};

/**
 * Extra video flags
 */
enum spa_video_flags {
	SPA_VIDEO_FLAG_NONE = 0,				/**< no flags */
	SPA_VIDEO_FLAG_VARIABLE_FPS = (1 << 0),			/**< a variable fps is selected, fps_n and fps_d
								 *   denote the maximum fps of the video */
	SPA_VIDEO_FLAG_PREMULTIPLIED_ALPHA = (1 << 1),		/**< Each color has been scaled by the alpha value. */
	SPA_VIDEO_FLAG_MODIFIER = (1 << 2),			/**< use the format modifier */
	SPA_VIDEO_FLAG_MODIFIER_FIXATION_REQUIRED = (1 << 3),	/**< format modifier was not fixated yet */
};

/**
 * The possible values of the #spa_video_interlace_mode describing the interlace
 * mode of the stream.
 */
enum spa_video_interlace_mode {
	SPA_VIDEO_INTERLACE_MODE_PROGRESSIVE = 0,	/**< all frames are progressive */
	SPA_VIDEO_INTERLACE_MODE_INTERLEAVED,		/**< 2 fields are interleaved in one video frame.
							 * Extra buffer flags describe the field order. */
	SPA_VIDEO_INTERLACE_MODE_MIXED,			/**< frames contains both interlaced and progressive
							 *   video, the buffer flags describe the frame and
							 *   fields. */
	SPA_VIDEO_INTERLACE_MODE_FIELDS,		/**< 2 fields are stored in one buffer, use the
							 *   frame ID to get access to the required
							 *   field. For multiview (the 'views'
							 *   property > 1) the fields of view N can
							 *   be found at frame ID (N * 2) and (N *
							 *   2) + 1. Each field has only half the
							 *   amount of lines as noted in the height
							 *   property. This mode requires multiple
							 *   spa_data to describe the fields. */
};

/**
 */
struct spa_video_info_raw {
	enum spa_video_format format;				/**< the format */
	uint32_t flags;						/**< extra video flags */
	uint64_t modifier;					/**< format modifier
								  * only used with DMA-BUF */
	struct spa_rectangle size;				/**< the frame size of the video */
	struct spa_fraction framerate;				/**< the framerate of the video, 0/1 means variable rate */
	struct spa_fraction max_framerate;			/**< the maximum framerate of the video. This is only valid when
								     \ref framerate is 0/1 */
	uint32_t views;						/**< the number of views in this video */
	enum spa_video_interlace_mode interlace_mode;		/**< the interlace mode */
	struct spa_fraction pixel_aspect_ratio;			/**< the pixel aspect ratio */
	enum spa_video_multiview_mode multiview_mode;		/**< multiview mode */
	enum spa_video_multiview_flags multiview_flags;		/**< multiview flags */
	enum spa_video_chroma_site chroma_site;			/**< the chroma siting */
	enum spa_video_color_range color_range;			/**< the color range. This is the valid range for the samples.
								 *   It is used to convert the samples to Y'PbPr values. */
	enum spa_video_color_matrix color_matrix;		/**< the color matrix. Used to convert between Y'PbPr and
								 *   non-linear RGB (R'G'B') */
	enum spa_video_transfer_function transfer_function;	/**< the transfer function. used to convert between R'G'B' and RGB */
	enum spa_video_color_primaries color_primaries;		/**< color primaries. used to convert between R'G'B' and CIE XYZ */
};

#define SPA_VIDEO_INFO_RAW_INIT(...)	((struct spa_video_info_raw) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_VIDEO_RAW_H */

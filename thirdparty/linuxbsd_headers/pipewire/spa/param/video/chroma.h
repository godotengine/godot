/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_VIDEO_CHROMA_H
#define SPA_VIDEO_CHROMA_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

/** Various Chroma settings.
 */
enum spa_video_chroma_site {
	SPA_VIDEO_CHROMA_SITE_UNKNOWN = 0,		/**< unknown cositing */
	SPA_VIDEO_CHROMA_SITE_NONE = (1 << 0),		/**< no cositing */
	SPA_VIDEO_CHROMA_SITE_H_COSITED = (1 << 1),	/**< chroma is horizontally cosited */
	SPA_VIDEO_CHROMA_SITE_V_COSITED = (1 << 2),	/**< chroma is vertically cosited */
	SPA_VIDEO_CHROMA_SITE_ALT_LINE = (1 << 3),	/**< chroma samples are sited on alternate lines */
	/* some common chroma cositing */
	/** chroma samples cosited with luma samples */
	SPA_VIDEO_CHROMA_SITE_COSITED = (SPA_VIDEO_CHROMA_SITE_H_COSITED | SPA_VIDEO_CHROMA_SITE_V_COSITED),
	/** jpeg style cositing, also for mpeg1 and mjpeg */
	SPA_VIDEO_CHROMA_SITE_JPEG = (SPA_VIDEO_CHROMA_SITE_NONE),
	/** mpeg2 style cositing */
	SPA_VIDEO_CHROMA_SITE_MPEG2 = (SPA_VIDEO_CHROMA_SITE_H_COSITED),
	/**< DV style cositing */
	SPA_VIDEO_CHROMA_SITE_DV = (SPA_VIDEO_CHROMA_SITE_COSITED | SPA_VIDEO_CHROMA_SITE_ALT_LINE),
};

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_VIDEO_CHROMA_H */

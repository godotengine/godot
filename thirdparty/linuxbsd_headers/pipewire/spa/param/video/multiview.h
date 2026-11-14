/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_VIDEO_MULTIVIEW_H
#define SPA_VIDEO_MULTIVIEW_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

/**
 * All possible stereoscopic 3D and multiview representations.
 * In conjunction with \ref spa_video_multiview_flags, describes how
 * multiview content is being transported in the stream.
 */
enum spa_video_multiview_mode {
	/** A special value indicating no multiview information. Used in spa_video_info and other
	 * places to indicate that no specific multiview handling has been requested or provided.
	 * This value is never carried on caps. */
	SPA_VIDEO_MULTIVIEW_MODE_NONE = -1,
	SPA_VIDEO_MULTIVIEW_MODE_MONO = 0,		/**< All frames are monoscopic */
	/* Single view modes */
	SPA_VIDEO_MULTIVIEW_MODE_LEFT,			/**< All frames represent a left-eye view */
	SPA_VIDEO_MULTIVIEW_MODE_RIGHT,			/**< All frames represent a right-eye view */
	/* Stereo view modes */
	SPA_VIDEO_MULTIVIEW_MODE_SIDE_BY_SIDE,		/**< Left and right eye views are provided
							 *   in the left and right half of the frame
							 *   respectively. */
	SPA_VIDEO_MULTIVIEW_MODE_SIDE_BY_SIDE_QUINCUNX, /**< Left and right eye views are provided
							 *   in the left and right half of the
							 *   frame, but have been sampled using
							 *   quincunx method, with half-pixel offset
							 *   between the 2 views. */
	SPA_VIDEO_MULTIVIEW_MODE_COLUMN_INTERLEAVED,	/**< Alternating vertical columns of pixels
							 *   represent the left and right eye view
							 *   respectively. */
	SPA_VIDEO_MULTIVIEW_MODE_ROW_INTERLEAVED,	/**< Alternating horizontal rows of pixels
							 *   represent the left and right eye view
							 *   respectively. */
	SPA_VIDEO_MULTIVIEW_MODE_TOP_BOTTOM,		/**< The top half of the frame contains the
							 *   left eye, and the bottom half the right
							 *   eye. */
	SPA_VIDEO_MULTIVIEW_MODE_CHECKERBOARD,		/**< Pixels are arranged with alternating
							 *   pixels representing left and right eye
							 *   views in a checkerboard fashion. */
	/* Padding for new frame packing modes */

	SPA_VIDEO_MULTIVIEW_MODE_FRAME_BY_FRAME = 32,	/**< Left and right eye views are provided
							 *   in separate frames alternately. */
	/* Multiview mode(s) */
	SPA_VIDEO_MULTIVIEW_MODE_MULTIVIEW_FRAME_BY_FRAME, /**< Multipleindependent views are
							    *   provided in separate frames in
							    *   sequence. This method only applies to
							    *   raw video buffers at the moment.
							    *   Specific view identification is via
							    *   \ref spa_video_multiview_meta on raw
							    *   video buffers. */
	SPA_VIDEO_MULTIVIEW_MODE_SEPARATED,		/**< Multiple views are provided as separate
							 *   \ref spa_data framebuffers attached
							 *   to each \ref spa_buffer, described
							 *   by the \ref spa_video_multiview_meta */
	/* future expansion for annotated modes */
};

/**
 * spa_video_multiview_flags are used to indicate extra properties of a
 * stereo/multiview stream beyond the frame layout and buffer mapping
 * that is conveyed in the \ref spa_video_multiview_mode.
 */
enum spa_video_multiview_flags {
	SPA_VIDEO_MULTIVIEW_FLAGS_NONE = 0,			/**< No flags */
	SPA_VIDEO_MULTIVIEW_FLAGS_RIGHT_VIEW_FIRST = (1 << 0),	/**< For stereo streams, the normal arrangement
								 *   of left and right views is reversed */
	SPA_VIDEO_MULTIVIEW_FLAGS_LEFT_FLIPPED = (1 << 1),	/**< The left view is vertically mirrored */
	SPA_VIDEO_MULTIVIEW_FLAGS_LEFT_FLOPPED = (1 << 2),	/**< The left view is horizontally mirrored */
	SPA_VIDEO_MULTIVIEW_FLAGS_RIGHT_FLIPPED = (1 << 3),	/**< The right view is vertically mirrored */
	SPA_VIDEO_MULTIVIEW_FLAGS_RIGHT_FLOPPED = (1 << 4),	/**< The right view is horizontally mirrored */
	SPA_VIDEO_MULTIVIEW_FLAGS_HALF_ASPECT = (1 << 14),	/**< For frame-packed multiview
								 *   modes, indicates that the individual
								 *   views have been encoded with half the true
								 *   width or height and should be scaled back
								 *   up for display. This flag is used for
								 *   overriding input layout interpretation
								 *   by adjusting pixel-aspect-ratio.
								 *   For side-by-side, column interleaved or
								 *   checkerboard packings, the
								 *   pixel width will be doubled.
								 *   For row interleaved and
								 *   top-bottom encodings, pixel height will
								 *   be doubled */
	SPA_VIDEO_MULTIVIEW_FLAGS_MIXED_MONO = (1 << 15),	/**< The video stream contains both
								 *   mono and multiview portions,
								 *   signalled on each buffer by the
								 *   absence or presence of the
								 *   \ref SPA_VIDEO_BUFFER_FLAG_MULTIPLE_VIEW
								 *   buffer flag. */
};


/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_VIDEO_MULTIVIEW_H */

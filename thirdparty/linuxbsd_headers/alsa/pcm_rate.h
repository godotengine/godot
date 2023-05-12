/**
 * \file include/pcm_rate.h
 * \brief External Rate-Converter-Plugin SDK
 * \author Takashi Iwai <tiwai@suse.de>
 * \date 2006
 *
 * External Rate-Converter-Plugin SDK
 */

/*
 * ALSA external PCM rate-converter plugin SDK (draft version)
 *
 * Copyright (c) 2006 Takashi Iwai <tiwai@suse.de>
 *
 *   This library is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as
 *   published by the Free Software Foundation; either version 2.1 of
 *   the License, or (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

#ifndef __ALSA_PCM_RATE_H
#define __ALSA_PCM_RATE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Protocol version
 */
#define SND_PCM_RATE_PLUGIN_VERSION	0x010002

/** hw_params information for a single side */
typedef struct snd_pcm_rate_side_info {
	snd_pcm_format_t format;
	unsigned int rate;
	snd_pcm_uframes_t buffer_size;
	snd_pcm_uframes_t period_size;
} snd_pcm_rate_side_info_t;

/** hw_params information */
typedef struct snd_pcm_rate_info {
	struct snd_pcm_rate_side_info in;
	struct snd_pcm_rate_side_info out;
	unsigned int channels;
} snd_pcm_rate_info_t;

/** Callback table of rate-converter */
typedef struct snd_pcm_rate_ops {
	/**
	 * close the converter; optional
	 */
	void (*close)(void *obj);
	/**
	 * initialize the converter, called at hw_params
	 */
	int (*init)(void *obj, snd_pcm_rate_info_t *info);
	/**
	 * free the converter; optional
	 */
	void (*free)(void *obj);
	/**
	 * reset the converter, called at prepare; optional
	 */
	void (*reset)(void *obj);
	/**
	 * adjust the pitch, called at sw_params; optional
	 */
	int (*adjust_pitch)(void *obj, snd_pcm_rate_info_t *info);
	/**
	 * convert the data
	 */
	void (*convert)(void *obj,
			const snd_pcm_channel_area_t *dst_areas,
			snd_pcm_uframes_t dst_offset, unsigned int dst_frames,
			const snd_pcm_channel_area_t *src_areas,
			snd_pcm_uframes_t src_offset, unsigned int src_frames);
	/**
	 * convert an s16 interleaved-data array; exclusive with convert
	 */
	void (*convert_s16)(void *obj, int16_t *dst, unsigned int dst_frames,
			    const int16_t *src, unsigned int src_frames);
	/**
	 * compute the frame size for input
	 */
	snd_pcm_uframes_t (*input_frames)(void *obj, snd_pcm_uframes_t frames);
	/**
	 * compute the frame size for output
	 */
	snd_pcm_uframes_t (*output_frames)(void *obj, snd_pcm_uframes_t frames);
	/**
	 * the protocol version the plugin supports;
	 * new field since version 0x010002
	 */
	unsigned int version;
	/**
	 * return the supported min / max sample rates;
	 * new ops since version 0x010002
	 */
	int (*get_supported_rates)(void *obj, unsigned int *rate_min,
				   unsigned int *rate_max);
	/**
	 * show some status messages for verbose mode;
	 * new ops since version 0x010002
	 */
	void (*dump)(void *obj, snd_output_t *out);
} snd_pcm_rate_ops_t;

/** open function type */
typedef int (*snd_pcm_rate_open_func_t)(unsigned int version, void **objp,
					snd_pcm_rate_ops_t *opsp);

/**
 * Define the object entry for external PCM rate-converter plugins
 */
#define SND_PCM_RATE_PLUGIN_ENTRY(name) _snd_pcm_rate_##name##_open


#ifndef DOC_HIDDEN
/* old rate_ops for protocol version 0x010001 */
typedef struct snd_pcm_rate_old_ops {
	void (*close)(void *obj);
	int (*init)(void *obj, snd_pcm_rate_info_t *info);
	void (*free)(void *obj);
	void (*reset)(void *obj);
	int (*adjust_pitch)(void *obj, snd_pcm_rate_info_t *info);
	void (*convert)(void *obj,
			const snd_pcm_channel_area_t *dst_areas,
			snd_pcm_uframes_t dst_offset, unsigned int dst_frames,
			const snd_pcm_channel_area_t *src_areas,
			snd_pcm_uframes_t src_offset, unsigned int src_frames);
	void (*convert_s16)(void *obj, int16_t *dst, unsigned int dst_frames,
			    const int16_t *src, unsigned int src_frames);
	snd_pcm_uframes_t (*input_frames)(void *obj, snd_pcm_uframes_t frames);
	snd_pcm_uframes_t (*output_frames)(void *obj, snd_pcm_uframes_t frames);
} snd_pcm_rate_old_ops_t;
#endif

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_PCM_RATE_H */

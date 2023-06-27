/**
 * \file include/pcm_plugin.h
 * \brief Common PCM plugin code
 * \author Abramo Bagnara <abramo@alsa-project.org>
 * \author Jaroslav Kysela <perex@perex.cz>
 * \date 2000-2001
 *
 * Application interface library for the ALSA driver.
 * See the \ref pcm_plugins page for more details.
 *
 * \warning Using of contents of this header file might be dangerous
 *	    in the sense of compatibility reasons. The contents might be
 *	    freely changed in future.
 */
/*
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

#ifndef __ALSA_PCM_PLUGIN_H

/**
 * \defgroup PCM_Plugins PCM Plugins
 * \ingroup PCM
 * See the \ref pcm_plugins page for more details.
 * \{
 */
  
#define SND_PCM_PLUGIN_RATE_MIN 4000	/**< minimal rate for the rate plugin */
#define SND_PCM_PLUGIN_RATE_MAX 192000	/**< maximal rate for the rate plugin */

/* ROUTE_FLOAT should be set to 0 for machines without FP unit - like iPAQ */
#ifdef HAVE_SOFT_FLOAT
#define SND_PCM_PLUGIN_ROUTE_FLOAT 0	   /**< use integers for route plugin */
#else
#define SND_PCM_PLUGIN_ROUTE_FLOAT 1	   /**< use floats for route plugin */
#endif

#define SND_PCM_PLUGIN_ROUTE_RESOLUTION 16 /**< integer resolution for route plugin */

#if SND_PCM_PLUGIN_ROUTE_FLOAT
/** route ttable entry type */
typedef float snd_pcm_route_ttable_entry_t;
#define SND_PCM_PLUGIN_ROUTE_HALF 0.5	/**< half value */
#define SND_PCM_PLUGIN_ROUTE_FULL 1.0	/**< full value */
#else
/** route ttable entry type */
typedef int snd_pcm_route_ttable_entry_t;
#define SND_PCM_PLUGIN_ROUTE_HALF (SND_PCM_PLUGIN_ROUTE_RESOLUTION / 2)	/**< half value */
#define SND_PCM_PLUGIN_ROUTE_FULL SND_PCM_PLUGIN_ROUTE_RESOLUTION	/**< full value */
#endif

/*
 *  Hardware plugin
 */
int snd_pcm_hw_open(snd_pcm_t **pcmp, const char *name,
		    int card, int device, int subdevice,
		    snd_pcm_stream_t stream, int mode,
		    int mmap_emulation, int sync_ptr_ioctl);
int _snd_pcm_hw_open(snd_pcm_t **pcmp, const char *name,
		     snd_config_t *root ATTRIBUTE_UNUSED, snd_config_t *conf,
		     snd_pcm_stream_t stream, int mode);

/*
 *  Copy plugin
 */
int snd_pcm_copy_open(snd_pcm_t **pcmp, const char *name,
		      snd_pcm_t *slave, int close_slave);
int _snd_pcm_copy_open(snd_pcm_t **pcmp, const char *name,
		       snd_config_t *root, snd_config_t *conf,
                       snd_pcm_stream_t stream, int mode);
                                              
/*
 *  Linear conversion plugin
 */
int snd_pcm_linear_open(snd_pcm_t **pcmp, const char *name,
			snd_pcm_format_t sformat, snd_pcm_t *slave,
			int close_slave);
int _snd_pcm_linear_open(snd_pcm_t **pcmp, const char *name,
			 snd_config_t *root, snd_config_t *conf,
			 snd_pcm_stream_t stream, int mode);

/*
 *  Linear<->Float conversion plugin
 */
int snd_pcm_lfloat_open(snd_pcm_t **pcmp, const char *name,
			snd_pcm_format_t sformat, snd_pcm_t *slave,
			int close_slave);
int _snd_pcm_lfloat_open(snd_pcm_t **pcmp, const char *name,
			 snd_config_t *root, snd_config_t *conf,
			 snd_pcm_stream_t stream, int mode);

/*
 *  Linear<->mu-Law conversion plugin
 */
int snd_pcm_mulaw_open(snd_pcm_t **pcmp, const char *name,
		       snd_pcm_format_t sformat, snd_pcm_t *slave,
		       int close_slave);
int _snd_pcm_mulaw_open(snd_pcm_t **pcmp, const char *name,
			snd_config_t *root, snd_config_t *conf,
                        snd_pcm_stream_t stream, int mode);

/*
 *  Linear<->a-Law conversion plugin
 */
int snd_pcm_alaw_open(snd_pcm_t **pcmp, const char *name,
		      snd_pcm_format_t sformat, snd_pcm_t *slave,
		      int close_slave);
int _snd_pcm_alaw_open(snd_pcm_t **pcmp, const char *name,
		       snd_config_t *root, snd_config_t *conf,
		       snd_pcm_stream_t stream, int mode);

/*
 *  Linear<->Ima-ADPCM conversion plugin
 */
int snd_pcm_adpcm_open(snd_pcm_t **pcmp, const char *name,
		       snd_pcm_format_t sformat, snd_pcm_t *slave,
		       int close_slave);
int _snd_pcm_adpcm_open(snd_pcm_t **pcmp, const char *name,
			snd_config_t *root, snd_config_t *conf,
			snd_pcm_stream_t stream, int mode);

/*
 *  Route plugin for linear formats
 */
int snd_pcm_route_load_ttable(snd_config_t *tt, snd_pcm_route_ttable_entry_t *ttable,
			      unsigned int tt_csize, unsigned int tt_ssize,
			      unsigned int *tt_cused, unsigned int *tt_sused,
			      int schannels);
int snd_pcm_route_determine_ttable(snd_config_t *tt,
				   unsigned int *tt_csize,
				   unsigned int *tt_ssize);
int snd_pcm_route_open(snd_pcm_t **pcmp, const char *name,
		       snd_pcm_format_t sformat, int schannels,
		       snd_pcm_route_ttable_entry_t *ttable,
		       unsigned int tt_ssize,
		       unsigned int tt_cused, unsigned int tt_sused,
		       snd_pcm_t *slave, int close_slave);
int _snd_pcm_route_open(snd_pcm_t **pcmp, const char *name,
			snd_config_t *root, snd_config_t *conf,
			snd_pcm_stream_t stream, int mode);

/*
 *  Rate plugin for linear formats
 */
int snd_pcm_rate_open(snd_pcm_t **pcmp, const char *name,
		      snd_pcm_format_t sformat, unsigned int srate,
		      const snd_config_t *converter,
		      snd_pcm_t *slave, int close_slave);
int _snd_pcm_rate_open(snd_pcm_t **pcmp, const char *name,
		       snd_config_t *root, snd_config_t *conf,
		       snd_pcm_stream_t stream, int mode);

/*
 *  Hooks plugin
 */
int snd_pcm_hooks_open(snd_pcm_t **pcmp, const char *name,
		       snd_pcm_t *slave, int close_slave);
int _snd_pcm_hooks_open(snd_pcm_t **pcmp, const char *name,
			snd_config_t *root, snd_config_t *conf,
			snd_pcm_stream_t stream, int mode);

/*
 *  LADSPA plugin
 */
int snd_pcm_ladspa_open(snd_pcm_t **pcmp, const char *name,
			const char *ladspa_path,
			unsigned int channels,
			snd_config_t *ladspa_pplugins,
			snd_config_t *ladspa_cplugins,
			snd_pcm_t *slave, int close_slave);
int _snd_pcm_ladspa_open(snd_pcm_t **pcmp, const char *name,
			 snd_config_t *root, snd_config_t *conf,
			 snd_pcm_stream_t stream, int mode);

/*
 *  Jack plugin
 */
int snd_pcm_jack_open(snd_pcm_t **pcmp, const char *name,
					snd_config_t *playback_conf,
					snd_config_t *capture_conf,
		      snd_pcm_stream_t stream, int mode);
int _snd_pcm_jack_open(snd_pcm_t **pcmp, const char *name,
                       snd_config_t *root, snd_config_t *conf,
                       snd_pcm_stream_t stream, int mode);


/** \} */

#endif /* __ALSA_PCM_PLUGIN_H */

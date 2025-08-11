/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_UTILS_NAMES_H
#define SPA_UTILS_NAMES_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup spa_names Factory Names
 * SPA plugin factory names
 */

/**
 * \addtogroup spa_names
 * \{
 */

/** for factory names */
#define SPA_NAME_SUPPORT_CPU		"support.cpu"			/**< A CPU interface */
#define SPA_NAME_SUPPORT_DBUS		"support.dbus"			/**< A DBUS interface */
#define SPA_NAME_SUPPORT_LOG		"support.log"			/**< A Log interface */
#define SPA_NAME_SUPPORT_LOOP		"support.loop"			/**< A Loop/LoopControl/LoopUtils
									  *  interface */
#define SPA_NAME_SUPPORT_SYSTEM		"support.system"		/**< A System interface */

#define SPA_NAME_SUPPORT_NODE_DRIVER	"support.node.driver"		/**< A dummy driver node */

/* control mixer */
#define SPA_NAME_CONTROL_MIXER		"control.mixer"			/**< mixes control streams */

/* audio mixer */
#define SPA_NAME_AUDIO_MIXER		"audio.mixer"			/**< mixes the raw audio on N input
									  *  ports together on the output
									  *  port */
#define SPA_NAME_AUDIO_MIXER_DSP	"audio.mixer.dsp"		/**< mixes mono audio with fixed input
									  *  and output buffer sizes. supported
									  *  formats must include f32 and
									  *  optionally f64 and s24_32 */

/** audio processing */
#define SPA_NAME_AUDIO_PROCESS_FORMAT	"audio.process.format"		/**< processes raw audio from one format
									  *  to another */
#define SPA_NAME_AUDIO_PROCESS_CHANNELMIX	\
					"audio.process.channelmix"	/**< mixes raw audio channels and applies
									  *  volume change. */
#define SPA_NAME_AUDIO_PROCESS_RESAMPLE		\
					"audio.process.resample"	/**< resamples raw audio */
#define SPA_NAME_AUDIO_PROCESS_DEINTERLEAVE	\
					"audio.process.deinterleave"	/**< deinterleave raw audio channels */
#define SPA_NAME_AUDIO_PROCESS_INTERLEAVE	\
					"audio.process.interleave"	/**< interleave raw audio channels */


/** audio convert combines some of the audio processing */
#define SPA_NAME_AUDIO_CONVERT		"audio.convert"			/**< converts raw audio from one format
									  *  to another. Must include at least
									  *  format, channelmix and resample
									  *  processing */
#define SPA_NAME_AUDIO_ADAPT		"audio.adapt"			/**< combination of a node and an
									  *  audio.convert. Does clock slaving */

#define SPA_NAME_AEC				"audio.aec"				/**< Echo canceling */

/** video processing */
#define SPA_NAME_VIDEO_PROCESS_FORMAT	"video.process.format"		/**< processes raw video from one format
									  *  to another */
#define SPA_NAME_VIDEO_PROCESS_SCALE	"video.process.scale"		/**< scales raw video */

/** video convert combines some of the video processing */
#define SPA_NAME_VIDEO_CONVERT		"video.convert"			/**< converts raw video from one format
									  *  to another. Must include at least
									  *  format and scaling */
#define SPA_NAME_VIDEO_ADAPT		"video.adapt"			/**< combination of a node and a
									  *  video.convert. */
/** keys for alsa factory names */
#define SPA_NAME_API_ALSA_ENUM_UDEV	"api.alsa.enum.udev"		/**< an alsa udev Device interface */
#define SPA_NAME_API_ALSA_PCM_DEVICE	"api.alsa.pcm.device"		/**< an alsa Device interface */
#define SPA_NAME_API_ALSA_PCM_SOURCE	"api.alsa.pcm.source"		/**< an alsa Node interface for
									  *  capturing PCM */
#define SPA_NAME_API_ALSA_PCM_SINK	"api.alsa.pcm.sink"		/**< an alsa Node interface for
									  *  playback PCM */
#define SPA_NAME_API_ALSA_SEQ_DEVICE	"api.alsa.seq.device"		/**< an alsa Midi device */
#define SPA_NAME_API_ALSA_SEQ_SOURCE	"api.alsa.seq.source"		/**< an alsa Node interface for
									  *  capture of midi */
#define SPA_NAME_API_ALSA_SEQ_SINK	"api.alsa.seq.sink"		/**< an alsa Node interface for
									  *  playback of midi */
#define SPA_NAME_API_ALSA_SEQ_BRIDGE	"api.alsa.seq.bridge"		/**< an alsa Node interface for
									  *  bridging midi ports */
#define SPA_NAME_API_ALSA_ACP_DEVICE	"api.alsa.acp.device"		/**< an alsa ACP Device interface */
#define SPA_NAME_API_ALSA_COMPRESS_OFFLOAD_DEVICE	"api.alsa.compress.offload.device"	/**< an alsa Device interface for
												  * compressed audio */
#define SPA_NAME_API_ALSA_COMPRESS_OFFLOAD_SINK		"api.alsa.compress.offload.sink"	/**< an alsa Node interface for
												  * compressed audio */

/** keys for bluez5 factory names */
#define SPA_NAME_API_BLUEZ5_ENUM_DBUS	"api.bluez5.enum.dbus"		/**< a dbus Device interface */
#define SPA_NAME_API_BLUEZ5_DEVICE	"api.bluez5.device"		/**< a Device interface */
#define SPA_NAME_API_BLUEZ5_MEDIA_SINK	"api.bluez5.media.sink"		/**< a playback Node interface for A2DP/BAP profiles */
#define SPA_NAME_API_BLUEZ5_MEDIA_SOURCE	"api.bluez5.media.source"	/**< a capture Node interface for A2DP/BAP profiles */
#define SPA_NAME_API_BLUEZ5_A2DP_SINK	"api.bluez5.a2dp.sink"		/**< alias for media.sink */
#define SPA_NAME_API_BLUEZ5_A2DP_SOURCE	"api.bluez5.a2dp.source"	/**< alias for media.source */
#define SPA_NAME_API_BLUEZ5_SCO_SINK	"api.bluez5.sco.sink"		/**< a playback Node interface for HSP/HFP profiles */
#define SPA_NAME_API_BLUEZ5_SCO_SOURCE	"api.bluez5.sco.source"		/**< a capture Node interface for HSP/HFP profiles */
#define SPA_NAME_API_BLUEZ5_MIDI_ENUM	"api.bluez5.midi.enum"		/**< a dbus midi Device interface */
#define SPA_NAME_API_BLUEZ5_MIDI_NODE	"api.bluez5.midi.node"		/**< a midi Node interface */

/** keys for codec factory names */
#define SPA_NAME_API_CODEC_BLUEZ5_MEDIA	"api.codec.bluez5.media"	/**< Bluez5 Media codec plugin */

/** keys for v4l2 factory names */
#define SPA_NAME_API_V4L2_ENUM_UDEV	"api.v4l2.enum.udev"		/**< a v4l2 udev Device interface */
#define SPA_NAME_API_V4L2_DEVICE	"api.v4l2.device"		/**< a v4l2 Device interface */
#define SPA_NAME_API_V4L2_SOURCE	"api.v4l2.source"		/**< a v4l2 Node interface for
									  *  capturing */

/** keys for libcamera factory names */
#define SPA_NAME_API_LIBCAMERA_ENUM_CLIENT	"api.libcamera.enum.client"	/**< a libcamera client Device interface */
#define SPA_NAME_API_LIBCAMERA_ENUM_MANAGER	"api.libcamera.enum.manager"	/**< a libcamera manager Device interface */
#define SPA_NAME_API_LIBCAMERA_DEVICE		"api.libcamera.device"		/**< a libcamera Device interface */
#define SPA_NAME_API_LIBCAMERA_SOURCE		"api.libcamera.source"		/**< a libcamera Node interface for
									  *  capturing */

/** keys for jack factory names */
#define SPA_NAME_API_JACK_DEVICE	"api.jack.device"		/**< a jack device. This is a
									  *  client connected to a server */
#define SPA_NAME_API_JACK_SOURCE	"api.jack.source"		/**< a jack source */
#define SPA_NAME_API_JACK_SINK		"api.jack.sink"			/**< a jack sink */

/** keys for vulkan factory names */
#define SPA_NAME_API_VULKAN_COMPUTE_SOURCE	\
					"api.vulkan.compute.source"	/**< a vulkan compute source. */
#define SPA_NAME_API_VULKAN_COMPUTE_FILTER	\
					"api.vulkan.compute.filter"	/**< a vulkan compute filter. */

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_UTILS_NAMES_H */

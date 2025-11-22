/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_PROPS_H
#define SPA_PARAM_PROPS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param.h>

/** properties of SPA_TYPE_OBJECT_PropInfo */
enum spa_prop_info {
	SPA_PROP_INFO_START,
	SPA_PROP_INFO_id,		/**< associated id of the property */
	SPA_PROP_INFO_name,		/**< name of the property */
	SPA_PROP_INFO_type,		/**< type and range/enums of property */
	SPA_PROP_INFO_labels,		/**< labels of property if any, this is a
					  *  struct with pairs of values, the first one
					  *  is of the type of the property, the second
					  *  one is a string with a user readable label
					  *  for the value. */
	SPA_PROP_INFO_container,	/**< type of container if any (Id) */
	SPA_PROP_INFO_params,		/**< is part of params property (Bool) */
	SPA_PROP_INFO_description,	/**< User readable description */
};

/** predefined properties for SPA_TYPE_OBJECT_Props */
enum spa_prop {
	SPA_PROP_START,

	SPA_PROP_unknown,		/**< an unknown property */

	SPA_PROP_START_Device	= 0x100,	/**< device related properties */
	SPA_PROP_device,
	SPA_PROP_deviceName,
	SPA_PROP_deviceFd,
	SPA_PROP_card,
	SPA_PROP_cardName,

	SPA_PROP_minLatency,
	SPA_PROP_maxLatency,
	SPA_PROP_periods,
	SPA_PROP_periodSize,
	SPA_PROP_periodEvent,
	SPA_PROP_live,
	SPA_PROP_rate,
	SPA_PROP_quality,
	SPA_PROP_bluetoothAudioCodec,
	SPA_PROP_bluetoothOffloadActive,

	SPA_PROP_START_Audio	= 0x10000,	/**< audio related properties */
	SPA_PROP_waveType,
	SPA_PROP_frequency,
	SPA_PROP_volume,			/**< a volume (Float), 0.0 silence, 1.0 no attenutation */
	SPA_PROP_mute,				/**< mute (Bool) */
	SPA_PROP_patternType,
	SPA_PROP_ditherType,
	SPA_PROP_truncate,
	SPA_PROP_channelVolumes,		/**< a volume array, one (linear) volume per channel
						  * (Array of Float). 0.0 is silence, 1.0 is
						  *  without attenuation. This is the effective
						  *  volume that is applied. It can result
						  *  in a hardware volume and software volume
						  *  (see softVolumes) */
	SPA_PROP_volumeBase,			/**< a volume base (Float) */
	SPA_PROP_volumeStep,			/**< a volume step (Float) */
	SPA_PROP_channelMap,			/**< a channelmap array
						  * (Array (Id enum spa_audio_channel)) */
	SPA_PROP_monitorMute,			/**< mute (Bool) */
	SPA_PROP_monitorVolumes,		/**< a volume array, one (linear) volume per
						  *  channel (Array of Float) */
	SPA_PROP_latencyOffsetNsec,		/**< delay adjustment */
	SPA_PROP_softMute,			/**< mute (Bool) applied in software */
	SPA_PROP_softVolumes,			/**< a volume array, one (linear) volume per channel
						  * (Array of Float). 0.0 is silence, 1.0 is without
						  * attenuation. This is the volume applied in
						  * software, there might be a part applied in
						  * hardware. */

	SPA_PROP_iec958Codecs,			/**< enabled IEC958 (S/PDIF) codecs,
						  *  (Array (Id enum spa_audio_iec958_codec) */
	SPA_PROP_volumeRampSamples,		/**< Samples to ramp the volume over */
	SPA_PROP_volumeRampStepSamples,		/**< Step or incremental Samples to ramp
						  *  the volume over */
	SPA_PROP_volumeRampTime,		/**< Time in millisec to ramp the volume over */
	SPA_PROP_volumeRampStepTime,		/**< Step or incremental Time in nano seconds
						  *  to ramp the */
	SPA_PROP_volumeRampScale,		/**< the scale or graph to used to ramp the
						  *  volume */

	SPA_PROP_START_Video	= 0x20000,	/**< video related properties */
	SPA_PROP_brightness,
	SPA_PROP_contrast,
	SPA_PROP_saturation,
	SPA_PROP_hue,
	SPA_PROP_gamma,
	SPA_PROP_exposure,
	SPA_PROP_gain,
	SPA_PROP_sharpness,

	SPA_PROP_START_Other	= 0x80000,	/**< other properties */
	SPA_PROP_params,			/**< simple control params
						  *    (Struct(
						  *	  (String : key,
						  *	   Pod    : value)*)) */


	SPA_PROP_START_CUSTOM	= 0x1000000,
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_PROPS_H */

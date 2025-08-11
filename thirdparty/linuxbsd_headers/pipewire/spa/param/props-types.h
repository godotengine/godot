/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_PROPS_TYPES_H
#define SPA_PARAM_PROPS_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param-types.h>

#include <spa/param/bluetooth/type-info.h>

/** Props Param */
#define SPA_TYPE_INFO_Props			SPA_TYPE_INFO_PARAM_BASE "Props"
#define SPA_TYPE_INFO_PROPS_BASE		SPA_TYPE_INFO_Props ":"

static const struct spa_type_info spa_type_props[] = {
	{ SPA_PROP_START, SPA_TYPE_Id, SPA_TYPE_INFO_PROPS_BASE, spa_type_param, },
	{ SPA_PROP_unknown, SPA_TYPE_None, SPA_TYPE_INFO_PROPS_BASE "unknown", NULL },
	{ SPA_PROP_device, SPA_TYPE_String, SPA_TYPE_INFO_PROPS_BASE "device", NULL },
	{ SPA_PROP_deviceName, SPA_TYPE_String, SPA_TYPE_INFO_PROPS_BASE "deviceName", NULL },
	{ SPA_PROP_deviceFd, SPA_TYPE_Fd, SPA_TYPE_INFO_PROPS_BASE "deviceFd", NULL },
	{ SPA_PROP_card, SPA_TYPE_String, SPA_TYPE_INFO_PROPS_BASE "card", NULL },
	{ SPA_PROP_cardName, SPA_TYPE_String, SPA_TYPE_INFO_PROPS_BASE "cardName", NULL },
	{ SPA_PROP_minLatency, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "minLatency", NULL },
	{ SPA_PROP_maxLatency, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "maxLatency", NULL },
	{ SPA_PROP_periods, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "periods", NULL },
	{ SPA_PROP_periodSize, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "periodSize", NULL },
	{ SPA_PROP_periodEvent, SPA_TYPE_Bool, SPA_TYPE_INFO_PROPS_BASE "periodEvent", NULL },
	{ SPA_PROP_live, SPA_TYPE_Bool, SPA_TYPE_INFO_PROPS_BASE "live", NULL },
	{ SPA_PROP_rate, SPA_TYPE_Double, SPA_TYPE_INFO_PROPS_BASE "rate", NULL },
	{ SPA_PROP_quality, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "quality", NULL },
	{ SPA_PROP_bluetoothAudioCodec, SPA_TYPE_Id, SPA_TYPE_INFO_PROPS_BASE "bluetoothAudioCodec", spa_type_bluetooth_audio_codec },
	{ SPA_PROP_bluetoothOffloadActive, SPA_TYPE_Bool, SPA_TYPE_INFO_PROPS_BASE "bluetoothOffloadActive", NULL },

	{ SPA_PROP_waveType, SPA_TYPE_Id, SPA_TYPE_INFO_PROPS_BASE "waveType", NULL },
	{ SPA_PROP_frequency, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "frequency", NULL },
	{ SPA_PROP_volume, SPA_TYPE_Float, SPA_TYPE_INFO_PROPS_BASE "volume", NULL },
	{ SPA_PROP_mute, SPA_TYPE_Bool, SPA_TYPE_INFO_PROPS_BASE "mute", NULL },
	{ SPA_PROP_patternType, SPA_TYPE_Id, SPA_TYPE_INFO_PROPS_BASE "patternType", NULL },
	{ SPA_PROP_ditherType, SPA_TYPE_Id, SPA_TYPE_INFO_PROPS_BASE "ditherType", NULL },
	{ SPA_PROP_truncate, SPA_TYPE_Bool, SPA_TYPE_INFO_PROPS_BASE "truncate", NULL },
	{ SPA_PROP_channelVolumes, SPA_TYPE_Array, SPA_TYPE_INFO_PROPS_BASE "channelVolumes", spa_type_prop_float_array },
	{ SPA_PROP_volumeBase, SPA_TYPE_Float, SPA_TYPE_INFO_PROPS_BASE "volumeBase", NULL },
	{ SPA_PROP_volumeStep, SPA_TYPE_Float, SPA_TYPE_INFO_PROPS_BASE "volumeStep", NULL },
	{ SPA_PROP_channelMap, SPA_TYPE_Array, SPA_TYPE_INFO_PROPS_BASE "channelMap", spa_type_prop_channel_map },
	{ SPA_PROP_monitorMute, SPA_TYPE_Bool, SPA_TYPE_INFO_PROPS_BASE "monitorMute", NULL },
	{ SPA_PROP_monitorVolumes, SPA_TYPE_Array, SPA_TYPE_INFO_PROPS_BASE "monitorVolumes", spa_type_prop_float_array },
	{ SPA_PROP_latencyOffsetNsec, SPA_TYPE_Long, SPA_TYPE_INFO_PROPS_BASE "latencyOffsetNsec", NULL },
	{ SPA_PROP_softMute, SPA_TYPE_Bool, SPA_TYPE_INFO_PROPS_BASE "softMute", NULL },
	{ SPA_PROP_softVolumes, SPA_TYPE_Array, SPA_TYPE_INFO_PROPS_BASE "softVolumes", spa_type_prop_float_array },
	{ SPA_PROP_iec958Codecs, SPA_TYPE_Array, SPA_TYPE_INFO_PROPS_BASE "iec958Codecs", spa_type_prop_iec958_codec },
	{ SPA_PROP_volumeRampSamples, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "volumeRampSamples", NULL },
	{ SPA_PROP_volumeRampStepSamples, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "volumeRampStepSamples", NULL },
	{ SPA_PROP_volumeRampTime, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "volumeRampTime", NULL },
	{ SPA_PROP_volumeRampStepTime, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "volumeRampStepTime", NULL },
	{ SPA_PROP_volumeRampScale, SPA_TYPE_Id, SPA_TYPE_INFO_PROPS_BASE "volumeRampScale", NULL },

	{ SPA_PROP_brightness, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "brightness", NULL },
	{ SPA_PROP_contrast, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "contrast", NULL },
	{ SPA_PROP_saturation, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "saturation", NULL },
	{ SPA_PROP_hue, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "hue", NULL },
	{ SPA_PROP_gamma, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "gamma", NULL },
	{ SPA_PROP_exposure, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "exposure", NULL },
	{ SPA_PROP_gain, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "gain", NULL },
	{ SPA_PROP_sharpness, SPA_TYPE_Int, SPA_TYPE_INFO_PROPS_BASE "sharpness", NULL },

	{ SPA_PROP_params, SPA_TYPE_Struct, SPA_TYPE_INFO_PROPS_BASE "params", NULL },
	{ 0, 0, NULL, NULL },
};

/** Enum Property info */
#define SPA_TYPE_INFO_PropInfo			SPA_TYPE_INFO_PARAM_BASE "PropInfo"
#define SPA_TYPE_INFO_PROP_INFO_BASE		SPA_TYPE_INFO_PropInfo ":"

static const struct spa_type_info spa_type_prop_info[] = {
	{ SPA_PROP_INFO_START, SPA_TYPE_Id, SPA_TYPE_INFO_PROP_INFO_BASE, spa_type_param, },
	{ SPA_PROP_INFO_id, SPA_TYPE_Id, SPA_TYPE_INFO_PROP_INFO_BASE "id", spa_type_props },
	{ SPA_PROP_INFO_name, SPA_TYPE_String, SPA_TYPE_INFO_PROP_INFO_BASE "name", NULL },
	{ SPA_PROP_INFO_type, SPA_TYPE_Pod, SPA_TYPE_INFO_PROP_INFO_BASE "type", NULL },
	{ SPA_PROP_INFO_labels, SPA_TYPE_Struct, SPA_TYPE_INFO_PROP_INFO_BASE "labels", NULL },
	{ SPA_PROP_INFO_container, SPA_TYPE_Id, SPA_TYPE_INFO_PROP_INFO_BASE "container", NULL },
	{ SPA_PROP_INFO_params, SPA_TYPE_Bool, SPA_TYPE_INFO_PROP_INFO_BASE "params", NULL },
	{ SPA_PROP_INFO_description, SPA_TYPE_String, SPA_TYPE_INFO_PROP_INFO_BASE "description", NULL },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_PROPS_TYPES_H */

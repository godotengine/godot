/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_TYPES_H
#define SPA_PARAM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/props.h>
#include <spa/param/format.h>
#include <spa/buffer/type-info.h>

/* base for parameter object enumerations */
#define SPA_TYPE_INFO_ParamId		SPA_TYPE_INFO_ENUM_BASE "ParamId"
#define SPA_TYPE_INFO_PARAM_ID_BASE	SPA_TYPE_INFO_ParamId ":"

static const struct spa_type_info spa_type_param[] = {
	{ SPA_PARAM_Invalid, SPA_TYPE_None, SPA_TYPE_INFO_PARAM_ID_BASE "Invalid", NULL },
	{ SPA_PARAM_PropInfo, SPA_TYPE_OBJECT_PropInfo, SPA_TYPE_INFO_PARAM_ID_BASE "PropInfo", NULL },
	{ SPA_PARAM_Props, SPA_TYPE_OBJECT_Props, SPA_TYPE_INFO_PARAM_ID_BASE "Props", NULL },
	{ SPA_PARAM_EnumFormat, SPA_TYPE_OBJECT_Format, SPA_TYPE_INFO_PARAM_ID_BASE "EnumFormat", NULL },
	{ SPA_PARAM_Format, SPA_TYPE_OBJECT_Format, SPA_TYPE_INFO_PARAM_ID_BASE "Format", NULL },
	{ SPA_PARAM_Buffers, SPA_TYPE_OBJECT_ParamBuffers, SPA_TYPE_INFO_PARAM_ID_BASE "Buffers", NULL },
	{ SPA_PARAM_Meta, SPA_TYPE_OBJECT_ParamMeta, SPA_TYPE_INFO_PARAM_ID_BASE "Meta", NULL },
	{ SPA_PARAM_IO, SPA_TYPE_OBJECT_ParamIO, SPA_TYPE_INFO_PARAM_ID_BASE "IO", NULL },
	{ SPA_PARAM_EnumProfile, SPA_TYPE_OBJECT_ParamProfile, SPA_TYPE_INFO_PARAM_ID_BASE "EnumProfile", NULL },
	{ SPA_PARAM_Profile, SPA_TYPE_OBJECT_ParamProfile, SPA_TYPE_INFO_PARAM_ID_BASE "Profile", NULL },
	{ SPA_PARAM_EnumPortConfig, SPA_TYPE_OBJECT_ParamPortConfig, SPA_TYPE_INFO_PARAM_ID_BASE "EnumPortConfig", NULL },
	{ SPA_PARAM_PortConfig, SPA_TYPE_OBJECT_ParamPortConfig, SPA_TYPE_INFO_PARAM_ID_BASE "PortConfig", NULL },
	{ SPA_PARAM_EnumRoute, SPA_TYPE_OBJECT_ParamRoute, SPA_TYPE_INFO_PARAM_ID_BASE "EnumRoute", NULL },
	{ SPA_PARAM_Route, SPA_TYPE_OBJECT_ParamRoute, SPA_TYPE_INFO_PARAM_ID_BASE "Route", NULL },
	{ SPA_PARAM_Control, SPA_TYPE_Sequence, SPA_TYPE_INFO_PARAM_ID_BASE "Control", NULL },
	{ SPA_PARAM_Latency, SPA_TYPE_OBJECT_ParamLatency, SPA_TYPE_INFO_PARAM_ID_BASE "Latency", NULL },
	{ SPA_PARAM_ProcessLatency, SPA_TYPE_OBJECT_ParamProcessLatency, SPA_TYPE_INFO_PARAM_ID_BASE "ProcessLatency", NULL },
	{ SPA_PARAM_Tag, SPA_TYPE_OBJECT_ParamTag, SPA_TYPE_INFO_PARAM_ID_BASE "Tag", NULL },
	{ 0, 0, NULL, NULL },
};

/* base for parameter objects */
#define SPA_TYPE_INFO_Param			SPA_TYPE_INFO_OBJECT_BASE "Param"
#define SPA_TYPE_INFO_PARAM_BASE		SPA_TYPE_INFO_Param ":"

#include <spa/param/audio/type-info.h>

static const struct spa_type_info spa_type_prop_float_array[] = {
	{ SPA_PROP_START, SPA_TYPE_Float, SPA_TYPE_INFO_BASE "floatArray", NULL, },
	{ 0, 0, NULL, NULL },
};

static const struct spa_type_info spa_type_prop_channel_map[] = {
	{ SPA_PROP_START, SPA_TYPE_Id, SPA_TYPE_INFO_BASE "channelMap", spa_type_audio_channel, },
	{ 0, 0, NULL, NULL },
};

static const struct spa_type_info spa_type_prop_iec958_codec[] = {
	{ SPA_PROP_START, SPA_TYPE_Id, SPA_TYPE_INFO_BASE "iec958Codec", spa_type_audio_iec958_codec, },
	{ 0, 0, NULL, NULL },
};

#define SPA_TYPE_INFO_ParamBitorder		SPA_TYPE_INFO_ENUM_BASE "ParamBitorder"
#define SPA_TYPE_INFO_PARAM_BITORDER_BASE	SPA_TYPE_INFO_ParamBitorder ":"

static const struct spa_type_info spa_type_param_bitorder[] = {
	{ SPA_PARAM_BITORDER_unknown, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BITORDER_BASE "unknown", NULL },
	{ SPA_PARAM_BITORDER_msb, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BITORDER_BASE "msb", NULL },
	{ SPA_PARAM_BITORDER_lsb, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_BITORDER_BASE "lsb", NULL },
	{ 0, 0, NULL, NULL },
};

#define SPA_TYPE_INFO_ParamAvailability		SPA_TYPE_INFO_ENUM_BASE "ParamAvailability"
#define SPA_TYPE_INFO_PARAM_AVAILABILITY_BASE	SPA_TYPE_INFO_ParamAvailability ":"

static const struct spa_type_info spa_type_param_availability[] = {
	{ SPA_PARAM_AVAILABILITY_unknown, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_AVAILABILITY_BASE "unknown", NULL },
	{ SPA_PARAM_AVAILABILITY_no, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_AVAILABILITY_BASE "no", NULL },
	{ SPA_PARAM_AVAILABILITY_yes, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_AVAILABILITY_BASE "yes", NULL },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_TYPES_H */

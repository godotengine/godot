/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_POD_VARARG_H
#define SPA_POD_VARARG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>

#include <spa/pod/pod.h>

/**
 * \addtogroup spa_pod
 * \{
 */

#define SPA_POD_Prop(key,...)				\
	key, ##__VA_ARGS__

#define SPA_POD_Control(offset,type,...)		\
	offset, type, ##__VA_ARGS__

#define SPA_CHOICE_RANGE(def,min,max)			3,(def),(min),(max)
#define SPA_CHOICE_STEP(def,min,max,step)		4,(def),(min),(max),(step)
#define SPA_CHOICE_ENUM(n_vals,...)			(n_vals),##__VA_ARGS__
#define SPA_CHOICE_FLAGS(flags)				1, (flags)
#define SPA_CHOICE_BOOL(def)				3,(def),(def),!(def)

#define SPA_POD_Bool(val)				"b", val
#define SPA_POD_CHOICE_Bool(def)			"?eb", SPA_CHOICE_BOOL(def)

#define SPA_POD_Id(val)					"I", val
#define SPA_POD_CHOICE_ENUM_Id(n_vals,...)		"?eI", SPA_CHOICE_ENUM(n_vals, __VA_ARGS__)

#define SPA_POD_Int(val)				"i", val
#define SPA_POD_CHOICE_ENUM_Int(n_vals,...)		"?ei", SPA_CHOICE_ENUM(n_vals, __VA_ARGS__)
#define SPA_POD_CHOICE_RANGE_Int(def,min,max)		"?ri", SPA_CHOICE_RANGE(def, min, max)
#define SPA_POD_CHOICE_STEP_Int(def,min,max,step)	"?si", SPA_CHOICE_STEP(def, min, max, step)
#define SPA_POD_CHOICE_FLAGS_Int(flags)			"?fi", SPA_CHOICE_FLAGS(flags)

#define SPA_POD_Long(val)				"l", val
#define SPA_POD_CHOICE_ENUM_Long(n_vals,...)		"?el", SPA_CHOICE_ENUM(n_vals, __VA_ARGS__)
#define SPA_POD_CHOICE_RANGE_Long(def,min,max)		"?rl", SPA_CHOICE_RANGE(def, min, max)
#define SPA_POD_CHOICE_STEP_Long(def,min,max,step)	"?sl", SPA_CHOICE_STEP(def, min, max, step)
#define SPA_POD_CHOICE_FLAGS_Long(flags)		"?fl", SPA_CHOICE_FLAGS(flags)

#define SPA_POD_Float(val)				"f", val
#define SPA_POD_CHOICE_ENUM_Float(n_vals,...)		"?ef", SPA_CHOICE_ENUM(n_vals, __VA_ARGS__)
#define SPA_POD_CHOICE_RANGE_Float(def,min,max)		"?rf", SPA_CHOICE_RANGE(def, min, max)
#define SPA_POD_CHOICE_STEP_Float(def,min,max,step)	"?sf", SPA_CHOICE_STEP(def, min, max, step)

#define SPA_POD_Double(val)				"d", val
#define SPA_POD_CHOICE_ENUM_Double(n_vals,...)		"?ed", SPA_CHOICE_ENUM(n_vals, __VA_ARGS__)
#define SPA_POD_CHOICE_RANGE_Double(def,min,max)	"?rd", SPA_CHOICE_RANGE(def, min, max)
#define SPA_POD_CHOICE_STEP_Double(def,min,max,step)	"?sd", SPA_CHOICE_STEP(def, min, max, step)

#define SPA_POD_String(val)				"s",val
#define SPA_POD_Stringn(val,len)			"S",val,len

#define SPA_POD_Bytes(val,len)				"y",val,len

#define SPA_POD_Rectangle(val)				"R",val
#define SPA_POD_CHOICE_ENUM_Rectangle(n_vals,...)	"?eR", SPA_CHOICE_ENUM(n_vals, __VA_ARGS__)
#define SPA_POD_CHOICE_RANGE_Rectangle(def,min,max)	"?rR", SPA_CHOICE_RANGE((def),(min),(max))
#define SPA_POD_CHOICE_STEP_Rectangle(def,min,max,step)	"?sR", SPA_CHOICE_STEP((def),(min),(max),(step))

#define SPA_POD_Fraction(val)				"F",val
#define SPA_POD_CHOICE_ENUM_Fraction(n_vals,...)	"?eF", SPA_CHOICE_ENUM(n_vals, __VA_ARGS__)
#define SPA_POD_CHOICE_RANGE_Fraction(def,min,max)	"?rF", SPA_CHOICE_RANGE((def),(min),(max))
#define SPA_POD_CHOICE_STEP_Fraction(def,min,max,step)	"?sF", SPA_CHOICE_STEP(def, min, max, step)

#define SPA_POD_Array(csize,ctype,n_vals,vals)		"a", csize,ctype,n_vals,vals
#define SPA_POD_Pointer(type,val)			"p", type,val
#define SPA_POD_Fd(val)					"h", val
#define SPA_POD_None()					"P", NULL
#define SPA_POD_Pod(val)				"P", val
#define SPA_POD_PodObject(val)				"O", val
#define SPA_POD_PodStruct(val)				"T", val
#define SPA_POD_PodChoice(val)				"V", val

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_POD_VARARG_H */

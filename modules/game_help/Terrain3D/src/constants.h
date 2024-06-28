// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef CONSTANTS_CLASS_H
#define CONSTANTS_CLASS_H

//using namespace godot;

// Constants

//#define RS RenderingServer::get_singleton()
#include "servers/rendering_server.h"
#include "core/variant/variant_utility.h"

#define COLOR_NAN Color(NAN, NAN, NAN, NAN)
#define COLOR_BLACK Color(0.0f, 0.0f, 0.0f, 1.0f)
#define COLOR_WHITE Color(1.0f, 1.0f, 1.0f, 1.0f)
#define COLOR_ROUGHNESS Color(1.0f, 1.0f, 1.0f, 0.5f)
#define COLOR_CHECKED Color(1.f, 1.f, 1.0f, -1.0f)
#define COLOR_NORMAL Color(0.5f, 0.5f, 1.0f, 1.0f)
#define COLOR_CONTROL Color(as_float(enc_auto(true)), 0.f, 0.f, 1.0f)

// For consistency between msvc, gcc, clang

#ifndef __FLT_MAX__
#define __FLT_MAX__ FLT_MAX
#endif

// Double precision builds

#ifdef REAL_T_IS_DOUBLE
typedef PackedFloat64Array PackedRealArray;
#else
typedef PackedFloat32Array PackedRealArray;
#endif

// Set class name for logger.h

#define CLASS_NAME() const String __class__ = get_class_static() + \
		String("#") + String::num_uint64(get_instance_id()).right(4);

#define CLASS_NAME_STATIC(p_name) static inline const char *__class__ = p_name;

// Validation macros

#define VOID // a return value for void, to avoid compiler warnings

#define IS_INIT(ret)           \
	if (_terrain == nullptr) { \
		return ret;            \
	}

#define IS_INIT_MESG(mesg, ret) \
	if (_terrain == nullptr) {  \
		LOG(ERROR, mesg);       \
		return ret;             \
	}

#define IS_INIT_COND(cond, ret)        \
	if (_terrain == nullptr || cond) { \
		return ret;                    \
	}

#define IS_INIT_COND_MESG(cond, mesg, ret) \
	if (_terrain == nullptr || cond) {     \
		LOG(ERROR, mesg);                  \
		return ret;                        \
	}

#define IS_INSTANCER_INIT(ret)                                         \
	if (_terrain == nullptr || _terrain->get_instancer() == nullptr) { \
		return ret;                                                    \
	}

#define IS_INSTANCER_INIT_MESG(mesg, ret)                              \
	if (_terrain == nullptr || _terrain->get_instancer() == nullptr) { \
		LOG(ERROR, mesg);                                              \
		return ret;                                                    \
	}

#define IS_STORAGE_INIT(ret)                                        \
	if (_terrain == nullptr || _terrain->get_storage().is_null()) { \
		return ret;                                                 \
	}

#define IS_STORAGE_INIT_MESG(mesg, ret)                             \
	if (_terrain == nullptr || _terrain->get_storage().is_null()) { \
		LOG(ERROR, mesg);                                           \
		return ret;                                                 \
	}

#endif // CONSTANTS_CLASS_H
/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_TYPE_H
#define PIPEWIRE_TYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/type.h>

/** \defgroup pw_type Type info
 * Type information
 */

/**
 * \addtogroup pw_type
 * \{
 */

enum {
	PW_TYPE_FIRST = SPA_TYPE_VENDOR_PipeWire,
};

#define PW_TYPE_INFO_BASE		"PipeWire:"

#define PW_TYPE_INFO_Object		PW_TYPE_INFO_BASE "Object"
#define PW_TYPE_INFO_OBJECT_BASE	PW_TYPE_INFO_Object ":"

#define PW_TYPE_INFO_Interface		PW_TYPE_INFO_BASE "Interface"
#define PW_TYPE_INFO_INTERFACE_BASE	PW_TYPE_INFO_Interface ":"

const struct spa_type_info * pw_type_info(void);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_TYPE_H */

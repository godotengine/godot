/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_COMMAND_H
#define SPA_COMMAND_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/pod/pod.h>

/**
 * \addtogroup spa_pod
 * \{
 */

struct spa_command_body {
	struct spa_pod_object_body body;
};

struct spa_command {
	struct spa_pod		pod;
	struct spa_command_body body;
};

#define SPA_COMMAND_TYPE(cmd)		((cmd)->body.body.type)
#define SPA_COMMAND_ID(cmd,type)	(SPA_COMMAND_TYPE(cmd) == (type) ? \
						(cmd)->body.body.id : SPA_ID_INVALID)

#define SPA_COMMAND_INIT_FULL(t,size,type,id,...) ((t)			\
	{ { (size), SPA_TYPE_Object },					\
	  { { (type), (id) }, ##__VA_ARGS__ } })

#define SPA_COMMAND_INIT(type,id)					\
	SPA_COMMAND_INIT_FULL(struct spa_command,			\
			sizeof(struct spa_command_body), type, id)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_COMMAND_H */

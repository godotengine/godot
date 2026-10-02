/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_EVENT_H
#define SPA_EVENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/pod/pod.h>

/**
 * \addtogroup spa_pod
 * \{
 */

struct spa_event_body {
	struct spa_pod_object_body body;
};

struct spa_event {
	struct spa_pod pod;
	struct spa_event_body body;
};

#define SPA_EVENT_TYPE(ev)	((ev)->body.body.type)
#define SPA_EVENT_ID(ev,type)	(SPA_EVENT_TYPE(ev) == (type) ? \
					(ev)->body.body.id : SPA_ID_INVALID)

#define SPA_EVENT_INIT_FULL(t,size,type,id,...) ((t)			\
	{ { (size), SPA_TYPE_OBJECT },					\
	  { { (type), (id) }, ##__VA_ARGS__ } })			\

#define SPA_EVENT_INIT(type,id)						\
	SPA_EVENT_INIT_FULL(struct spa_event,				\
			sizeof(struct spa_event_body), type, id)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_EVENT_H */

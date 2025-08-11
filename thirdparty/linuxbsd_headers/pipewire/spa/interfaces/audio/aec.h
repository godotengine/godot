/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2021 Wim Taymans <wim.taymans@gmail.com> */
/* SPDX-License-Identifier: MIT */

#include <spa/pod/builder.h>
#include <spa/utils/dict.h>
#include <spa/utils/hook.h>
#include <spa/param/audio/raw.h>

#ifndef SPA_AUDIO_AEC_H
#define SPA_AUDIO_AEC_H

#ifdef __cplusplus
extern "C" {
#endif

#define SPA_TYPE_INTERFACE_AUDIO_AEC SPA_TYPE_INFO_INTERFACE_BASE "Audio:AEC"

#define SPA_VERSION_AUDIO_AEC   1
struct spa_audio_aec {
	struct spa_interface iface;
	const char *name;
	const struct spa_dict *info;
	const char *latency;
};

struct spa_audio_aec_info {
#define SPA_AUDIO_AEC_CHANGE_MASK_PROPS	(1u<<0)
        uint64_t change_mask;

	const struct spa_dict *props;
};

struct spa_audio_aec_events {
#define SPA_VERSION_AUDIO_AEC_EVENTS	0
        uint32_t version;       /**< version of this structure */

	/** Emitted when info changes */
	void (*info) (void *data, const struct spa_audio_aec_info *info);
};

struct spa_audio_aec_methods {
#define SPA_VERSION_AUDIO_AEC_METHODS	3
        uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct spa_audio_aec_events *events,
			void *data);

	int (*init) (void *object, const struct spa_dict *args, const struct spa_audio_info_raw *info);
	int (*run) (void *object, const float *rec[], const float *play[], float *out[], uint32_t n_samples);
	int (*set_props) (void *object, const struct spa_dict *args);
	/* since 0.3.58, version 1:1 */
	int (*activate) (void *object);
	/* since 0.3.58, version 1:1 */
	int (*deactivate) (void *object);

	/* version 1:2 */
	int (*enum_props) (void* object, int index, struct spa_pod_builder* builder);
	int (*get_params) (void* object, struct spa_pod_builder* builder);
	int (*set_params) (void *object, const struct spa_pod *args);

	/* version 1:3 */
	int (*init2) (void *object, const struct spa_dict *args,
			struct spa_audio_info_raw *play_info,
			struct spa_audio_info_raw *rec_info,
			struct spa_audio_info_raw *out_info);
};

#define spa_audio_aec_method(o,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	struct spa_audio_aec *_o = (o);					\
	spa_interface_call_res(&_o->iface,				\
			struct spa_audio_aec_methods, _res,		\
			method, (version), ##__VA_ARGS__);		\
	_res;								\
})

#define spa_audio_aec_add_listener(o,...)	spa_audio_aec_method(o, add_listener, 0, __VA_ARGS__)
#define spa_audio_aec_init(o,...)		spa_audio_aec_method(o, init, 0, __VA_ARGS__)
#define spa_audio_aec_run(o,...)		spa_audio_aec_method(o, run, 0, __VA_ARGS__)
#define spa_audio_aec_set_props(o,...)		spa_audio_aec_method(o, set_props, 0, __VA_ARGS__)
#define spa_audio_aec_activate(o)		spa_audio_aec_method(o, activate, 1)
#define spa_audio_aec_deactivate(o)		spa_audio_aec_method(o, deactivate, 1)
#define spa_audio_aec_enum_props(o,...)		spa_audio_aec_method(o, enum_props, 2, __VA_ARGS__)
#define spa_audio_aec_get_params(o,...)		spa_audio_aec_method(o, get_params, 2, __VA_ARGS__)
#define spa_audio_aec_set_params(o,...)		spa_audio_aec_method(o, set_params, 2, __VA_ARGS__)
#define spa_audio_aec_init2(o,...)		spa_audio_aec_method(o, init2, 3, __VA_ARGS__)

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_AEC_H */

/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_LATENY_H
#define SPA_PARAM_LATENY_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param.h>

/**
 * Properties for SPA_TYPE_OBJECT_ParamLatency
 *
 * The latency indicates:
 *
 * - for playback: time delay between start of a graph cycle, and the rendering of
 *   the first sample of that cycle in audio output.
 *
 * - for capture: time delay between start of a graph cycle, and the first sample
 *   of that cycle having occurred in audio input.
 *
 * For physical output/input, the latency is intended to correspond to the
 * rendering/capture of physical audio, including hardware internal rendering delay.
 *
 * The latency values are adjusted by \ref SPA_PROP_latencyOffsetNsec or
 * SPA_PARAM_ProcessLatency, if present. (e.g. for ALSA this is used to adjust for
 * the internal hardware latency).
 */
enum spa_param_latency {
	SPA_PARAM_LATENCY_START,
	SPA_PARAM_LATENCY_direction,		/**< direction, input/output (Id enum spa_direction) */
	SPA_PARAM_LATENCY_minQuantum,		/**< min latency relative to quantum (Float) */
	SPA_PARAM_LATENCY_maxQuantum,		/**< max latency relative to quantum (Float) */
	SPA_PARAM_LATENCY_minRate,		/**< min latency (Int) relative to graph rate */
	SPA_PARAM_LATENCY_maxRate,		/**< max latency (Int) relative to graph rate */
	SPA_PARAM_LATENCY_minNs,		/**< min latency (Long) in nanoseconds */
	SPA_PARAM_LATENCY_maxNs,		/**< max latency (Long) in nanoseconds */
};

/** helper structure for managing latency objects */
struct spa_latency_info {
	enum spa_direction direction;
	float min_quantum;
	float max_quantum;
	uint32_t min_rate;
	uint32_t max_rate;
	uint64_t min_ns;
	uint64_t max_ns;
};

#define SPA_LATENCY_INFO(dir,...) ((struct spa_latency_info) { .direction = (dir), ## __VA_ARGS__ })

/**
 * Properties for SPA_TYPE_OBJECT_ParamProcessLatency
 *
 * The processing latency indicates logical time delay between a sample in an input port,
 * and a corresponding sample in an output port, relative to the graph time.
 */
enum spa_param_process_latency {
	SPA_PARAM_PROCESS_LATENCY_START,
	SPA_PARAM_PROCESS_LATENCY_quantum,	/**< latency relative to quantum (Float) */
	SPA_PARAM_PROCESS_LATENCY_rate,		/**< latency (Int) relative to graph rate */
	SPA_PARAM_PROCESS_LATENCY_ns,		/**< latency (Long) in nanoseconds */
};

/** Helper structure for managing process latency objects */
struct spa_process_latency_info {
	float quantum;
	uint32_t rate;
	uint64_t ns;
};

#define SPA_PROCESS_LATENCY_INFO_INIT(...)	((struct spa_process_latency_info) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_LATENY_H */

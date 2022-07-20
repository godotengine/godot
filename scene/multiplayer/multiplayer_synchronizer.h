/*************************************************************************/
/*  multiplayer_synchronizer.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef MULTIPLAYER_SYNCHRONIZER_H
#define MULTIPLAYER_SYNCHRONIZER_H

#include "scene/main/node.h"

#include "scene/resources/scene_replication_config.h"

class MultiplayerSynchronizer : public Node {
	GDCLASS(MultiplayerSynchronizer, Node);

private:
	Ref<SceneReplicationConfig> replication_config;
	NodePath root_path = NodePath(".."); // Start with parent, like with AnimationPlayer.
	uint64_t interval_msec = 0;

	static Object *_get_prop_target(Object *p_obj, const NodePath &p_prop);
	void _start();
	void _stop();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	static Error get_state(const List<NodePath> &p_properties, Object *p_obj, Vector<Variant> &r_variant, Vector<const Variant *> &r_variant_ptrs);
	static Error set_state(const List<NodePath> &p_properties, Object *p_obj, const Vector<Variant> &p_state);

	void set_replication_interval(double p_interval);
	double get_replication_interval() const;
	uint64_t get_replication_interval_msec() const;

	void set_replication_config(Ref<SceneReplicationConfig> p_config);
	Ref<SceneReplicationConfig> get_replication_config();

	void set_root_path(const NodePath &p_path);
	NodePath get_root_path() const;
	virtual void set_multiplayer_authority(int p_peer_id, bool p_recursive = true) override;

	MultiplayerSynchronizer() {}
};

#endif // MULTIPLAYER_SYNCHRONIZER_H

#pragma once

#include "../ufbx_write.h"

#include <string.h>

namespace ufbxw {

struct string {
	string(const char *str) : m_data(str), m_size(strlen(str)) { }

	const char *data() const { return m_data; }
	size_t size() const { return m_size; }

private:
	const char *m_data;
	size_t m_size;
};

struct element;
struct node;

struct scene {
	scene(ufbxw_scene *scene) : m_scene(scene) { }

	node create_node();
	node create_node(string name);

protected:
	ufbxw_scene *m_scene;
};

struct element {
	element(ufbxw_scene *scene, ufbxw_id id) : m_scene(scene), m_id(id) { }

	ufbxw_id id() const { return m_id; }

	void delete_element() { ufbxw_delete_element(m_scene, m_id); }

	void set_name(string name) { ufbxw_set_name_len(m_scene, m_id, name.data(), name.size()); }

protected:
	ufbxw_scene *m_scene;
	ufbxw_id m_id;
};

struct node : element {
	node(ufbxw_scene *scene, ufbxw_node node) : element(scene, node.id) { }

	ufbxw_node node_id() const { return ufbxw_node{m_id}; }

	void set_translation(ufbxw_vec3 translation) { ufbxw_node_set_translation(m_scene, node_id(), translation); }
};

inline node scene::create_node() { return node{m_scene, ufbxw_create_node(m_scene)}; }
inline node scene::create_node(string name) {
	node n = create_node(name);
	n.set_name(name);
	return n;
}

inline scene create_scene() { return ufbxw_create_scene(nullptr); }
inline scene create_scene(const ufbxw_scene_opts &opts) { return ufbxw_create_scene(&opts); }

}

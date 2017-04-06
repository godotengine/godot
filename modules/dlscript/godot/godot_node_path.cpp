#include "godot_node_path.h"

#include "path_db.h"

#ifdef __cplusplus
extern "C" {
#endif

void _node_path_api_anchor() {
}

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

// @Bug ?
// Do I need to memnew_placement when returning strings?

void GDAPI godot_node_path_new(godot_node_path *p_np, const godot_string *p_from) {
	NodePath *np = (NodePath *)p_np;
	String *from = (String *)p_from;
	memnew_placement_custom(np, NodePath, NodePath(*from));
}

void GDAPI godot_node_path_copy(godot_node_path *p_np, const godot_node_path *p_from) {
	NodePath *np = (NodePath *)p_np;
	NodePath *from = (NodePath *)p_from;
	*np = *from;
}

godot_string GDAPI godot_node_path_get_name(const godot_node_path *p_np, const godot_int p_idx) {
	const NodePath *np = (const NodePath *)p_np;
	godot_string str;
	String *s = (String *)&str;
	memnew_placement(s, String);
	*s = np->get_name(p_idx);
	return str;
}

godot_int GDAPI godot_node_path_get_name_count(const godot_node_path *p_np) {
	const NodePath *np = (const NodePath *)p_np;
	return np->get_name_count();
}

godot_string GDAPI godot_node_path_get_property(const godot_node_path *p_np) {
	const NodePath *np = (const NodePath *)p_np;
	godot_string str;
	String *s = (String *)&str;
	memnew_placement(s, String);
	*s = np->get_property();
	return str;
}

godot_string GDAPI godot_node_path_get_subname(const godot_node_path *p_np, const godot_int p_idx) {
	const NodePath *np = (const NodePath *)p_np;
	godot_string str;
	String *s = (String *)&str;
	memnew_placement(s, String);
	*s = np->get_subname(p_idx);
	return str;
}

godot_int GDAPI godot_node_path_get_subname_count(const godot_node_path *p_np) {
	const NodePath *np = (const NodePath *)p_np;
	return np->get_subname_count();
}

godot_bool GDAPI godot_node_path_is_absolute(const godot_node_path *p_np) {
	const NodePath *np = (const NodePath *)p_np;
	return np->is_absolute();
}

godot_bool GDAPI godot_node_path_is_empty(const godot_node_path *p_np) {
	const NodePath *np = (const NodePath *)p_np;
	return np->is_empty();
}

godot_string GDAPI godot_node_path_as_string(const godot_node_path *p_np) {
	const NodePath *np = (const NodePath *)p_np;
	godot_string str;
	String *s = (String *)&str;
	memnew_placement(s, String);
	*s = *np;
	return str;
}

void GDAPI godot_node_path_destroy(godot_node_path *p_np) {
	((NodePath *)p_np)->~NodePath();
}

#ifdef __cplusplus
}
#endif

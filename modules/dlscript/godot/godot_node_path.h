#ifndef GODOT_DLSCRIPT_NODE_PATH_H
#define GODOT_DLSCRIPT_NODE_PATH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_NODE_PATH_TYPE_DEFINED
typedef struct godot_node_path {
	uint8_t _dont_touch_that[8];
} godot_node_path;
#endif

#include "../godot.h"

void GDAPI godot_node_path_new(godot_node_path *p_np, const godot_string *p_from);
void GDAPI godot_node_path_copy(godot_node_path *p_np, const godot_node_path *p_from);

godot_string GDAPI godot_node_path_get_name(const godot_node_path *p_np, const godot_int p_idx);
godot_int GDAPI godot_node_path_get_name_count(const godot_node_path *p_np);

godot_string GDAPI godot_node_path_get_property(const godot_node_path *p_np);
godot_string GDAPI godot_node_path_get_subname(const godot_node_path *p_np, const godot_int p_idx);
godot_int GDAPI godot_node_path_get_subname_count(const godot_node_path *p_np);

godot_bool GDAPI godot_node_path_is_absolute(const godot_node_path *p_np);
godot_bool GDAPI godot_node_path_is_empty(const godot_node_path *p_np);

godot_string GDAPI godot_node_path_as_string(const godot_node_path *p_np);

void GDAPI godot_node_path_destroy(godot_node_path *p_np);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_NODE_PATH_H

#undef UFBXWT_TEST_GROUP
#define UFBXWT_TEST_GROUP "errors"

#if UFBXWT_IMPL
static void ufbxwt_capture_error(void *user, const ufbxw_error *error)
{
	ufbxw_error *dst = (ufbxw_error*)user;
	ufbxwt_assert(dst->type == UFBXW_ERROR_NONE);
	*dst = *error;
}
#endif

UFBXWT_TEST(node_not_found)
#if UFBXWT_IMPL
{
	ufbxw_scene *scene = ufbxw_create_scene(NULL);
	ufbxwt_assert(scene);

	ufbxw_error error = { UFBXW_ERROR_NONE };
	ufbxw_set_error_callback(scene, &ufbxwt_capture_error, &error);

	ufbxw_node node = ufbxwt_create_node(scene, "Node");

	ufbxw_string name = ufbxw_get_name(scene, node.id);
	ufbxwt_assert_string(name, "Node");

	ufbxw_delete_element(scene, node.id);

	ufbxwt_assert(error.type == UFBXW_ERROR_NONE);

	ufbxw_string del_name = ufbxw_get_name(scene, node.id);
	ufbxwt_assert_string(del_name, "");

	ufbxwt_assert_error(&error, UFBXW_ERROR_ELEMENT_NOT_FOUND, "ufbxw_get_name", "element not found");

	ufbxw_free_scene(scene);
}
#endif

UFBXWT_TEST(node_wrong_type)
#if UFBXWT_IMPL
{
	ufbxw_scene *scene = ufbxw_create_scene(NULL);
	ufbxwt_assert(scene);

	ufbxw_error error = { UFBXW_ERROR_NONE };
	ufbxw_set_error_callback(scene, &ufbxwt_capture_error, &error);

	ufbxw_mesh mesh = ufbxw_create_mesh(scene);

	ufbxw_node fake_node = { mesh.id };

	ufbxwt_assert(error.type == UFBXW_ERROR_NONE);

	ufbxw_vec3 translation = { 1.0f, 2.0f, 3.0f };
	ufbxw_node_set_translation(scene, fake_node, translation);

	ufbxwt_assert_error(&error, UFBXW_ERROR_ELEMENT_WRONG_TYPE, "ufbxw_node_set_translation", "wrong type: mesh");
}
#endif

UFBXWT_TEST(error_out_of_bounds)
#if UFBXWT_IMPL
{
	ufbxw_scene *scene = ufbxw_create_scene(NULL);
	ufbxwt_assert(scene);

	ufbxw_error error = { UFBXW_ERROR_NONE };
	ufbxw_set_error_callback(scene, &ufbxwt_capture_error, &error);

	ufbxw_node parent = ufbxwt_create_node(scene, "Parent");
	ufbxw_node child = ufbxwt_create_node(scene, "Child");
	ufbxw_node_set_parent(scene, child, parent);

	ufbxwt_assert(ufbxw_node_get_num_children(scene, parent) == 1);
	ufbxwt_assert(ufbxw_node_get_num_children(scene, child) == 0);

	ufbxw_node ref_child = ufbxw_node_get_child(scene, parent, 0);
	ufbxwt_assert(ref_child.id == child.id);

	ufbxwt_assert(error.type == UFBXW_ERROR_NONE);

	ufbxw_node bad_child = ufbxw_node_get_child(scene, parent, 1);
	ufbxwt_assert(bad_child.id == 0);

	ufbxwt_assert_error(&error, UFBXW_ERROR_INDEX_OUT_OF_BOUNDS, "ufbxw_node_get_child", "index (1) out of bounds (1)");

	ufbxw_free_scene(scene);
}
#endif

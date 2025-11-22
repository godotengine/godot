#undef UFBXWT_TEST_GROUP
#define UFBXWT_TEST_GROUP "basic"

UFBXWT_TEST(create_scene)
#if UFBXWT_IMPL
{
	ufbxw_scene *scene = ufbxw_create_scene(NULL);
	ufbxwt_assert(scene);

	ufbxw_free_scene(scene);
}
#endif

UFBXWT_TEST(memory_stats)
#if UFBXWT_IMPL
{
	ufbxw_scene *scene = ufbxw_create_scene(NULL);
	ufbxwt_assert(scene);

	ufbxw_memory_stats stats = ufbxw_get_memory_stats(scene);
	ufbxwt_assert(stats.allocated_bytes > 0);
	ufbxwt_assert(stats.allocation_count > 0);

	ufbxw_free_scene(scene);
}
#endif

UFBXWT_SCENE_TEST(simple_node)
#if UFBXWT_IMPL
{
	ufbxw_node node = ufbxw_create_node(scene);
	ufbxw_set_name(scene, node.id, "Test_Node");

	ufbxw_vec3 translation = { 1.0f, 2.0f, 3.0f };
	ufbxw_node_set_translation(scene, node, translation);

	ufbxw_vec3 scaling = { 1.0f, 0.5f, 0.25f };
	ufbxw_set_vec3(scene, node.id, "Lcl Scaling", scaling);

	ufbxwt_check_error(scene);

	ufbxw_vec3 field_translation = ufbxw_node_get_translation(scene, node);
	ufbxw_vec3 prop_translation = ufbxw_get_vec3(scene, node.id, "Lcl Translation");
	ufbxwt_assert(ufbxwt_equal_vec3(field_translation, translation));
	ufbxwt_assert(ufbxwt_equal_vec3(prop_translation, translation));

	ufbxw_vec3 field_scaling = ufbxw_node_get_scaling(scene, node);
	ufbxw_vec3 prop_scaling = ufbxw_get_vec3(scene, node.id, "Lcl Scaling");
	ufbxwt_assert(ufbxwt_equal_vec3(field_scaling, scaling));
	ufbxwt_assert(ufbxwt_equal_vec3(prop_scaling, scaling));
}
#endif

UFBXWT_SCENE_CHECK(simple_node)
#if UFBXWT_IMPL
{
	ufbx_node *node = ufbx_find_node(scene, "Test_Node");
	ufbxwt_assert(node);

	ufbx_vec3 translation = { 1.0f, 2.0f, 3.0f };
	ufbxwt_assert_close_uvec3(err, node->local_transform.translation, translation);

	ufbx_vec3 scale = { 1.0f, 0.5f, 0.25f };
	ufbxwt_assert_close_uvec3(err, node->local_transform.scale, scale);
}
#endif


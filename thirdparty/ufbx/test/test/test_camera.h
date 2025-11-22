#undef UFBXWT_TEST_GROUP
#define UFBXWT_TEST_GROUP "camera"

UFBXWT_SCENE_TEST(camera_defaults)
#if UFBXWT_IMPL
{
	ufbxw_node node = ufbxwt_create_node(scene, "Node");
	ufbxw_camera camera = ufbxw_create_camera(scene, node);
}
#endif

UFBXWT_SCENE_CHECK(camera_defaults)
#if UFBXWT_IMPL
{
	ufbx_node *node = ufbx_find_node(scene, "Node");
	ufbxwt_assert(node);

	ufbx_camera *camera = node->camera;
	ufbxwt_assert(camera);
}
#endif


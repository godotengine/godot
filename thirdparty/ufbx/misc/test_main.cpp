#include "ufbx_write.hpp"

void test()
{
	ufbxw::scene scene = ufbxw::create_scene();
	ufbxw::node node = scene.create_node("pCube1");

	node.set_translation({ 1.0f, 2.0f, 3.0f });

	node.delete_element();
}


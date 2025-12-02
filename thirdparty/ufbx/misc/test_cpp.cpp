#include "../ufbx_write.h"

int main(int argc, char **argv)
{
    ufbxw_scene *scene = ufbxw_create_scene(nullptr);

    ufbxw_node node = ufbxw_create_node(scene);
    ufbxw_set_name(scene, node.id, "Test");

    ufbxw_free_scene(scene);

    return 0;
}

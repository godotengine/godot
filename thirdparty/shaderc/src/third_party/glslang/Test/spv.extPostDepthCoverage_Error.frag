#version 450

#extension GL_EXT_post_depth_coverage : enable

layout(post_depth_coverage) in; // should fail since for GL_EXT_post_depth_coverage
                                // explicit declaration of early_fragment_tests is required

void main () {
}

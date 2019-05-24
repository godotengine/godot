#version 310 es

#extension GL_ARB_post_depth_coverage : enable

precision highp float;

layout(post_depth_coverage, location = 0) in float a;  // should fail since post_depth_coverage may only
                                                       // be declared on in only (not with variable declarations)

void main () {

}

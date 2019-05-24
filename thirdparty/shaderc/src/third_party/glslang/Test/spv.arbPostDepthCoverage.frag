#version 450

#extension GL_ARB_post_depth_coverage : enable
#extension GL_EXT_post_depth_coverage : enable //according to ARB_post_depth_coverage, 
                                               //if both enabled, this one should be ignored
precision highp int;
layout(post_depth_coverage) in;

layout (location = 0) out int readSampleMaskIn;

void main () {
    readSampleMaskIn = gl_SampleMaskIn[0];
}

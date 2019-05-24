#version 450

#extension GL_NV_conservative_raster_underestimation : enable

out vec4 color;

void main() {
    color = vec4(gl_FragFullyCoveredNV, 0, 0, 0);
}

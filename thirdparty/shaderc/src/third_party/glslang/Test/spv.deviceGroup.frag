#version 450

#extension GL_EXT_device_group : enable

out vec4 color;

void main() {
    color = vec4(gl_DeviceIndex, 0, 0, 0);
}

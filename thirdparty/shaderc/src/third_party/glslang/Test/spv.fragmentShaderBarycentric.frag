#version 450
#extension GL_NV_fragment_shader_barycentric : require

layout(location = 0) pervertexNV in vertices {
    float attrib;
    } v[];   
      
layout(location = 1) out float value;
      
void main () {
    value = (gl_BaryCoordNV.x * v[0].attrib +
             gl_BaryCoordNV.y * v[1].attrib +
             gl_BaryCoordNV.z * v[2].attrib);

}

#version 140

uniform sampler1D       texSampler1D;
uniform sampler2D       texSampler2D;
uniform sampler3D       texSampler3D;
uniform samplerCube	    texSamplerCube;
uniform sampler1DShadow shadowSampler1D;
uniform sampler2DShadow shadowSampler2D;

in vec2 coords2D;

void main()
{  
    float lod		 = 3.0;
    float coords1D   = 1.789;
    vec3  coords3D   = vec3(1.789, 2.718, 3.453);
    vec4  coords4D   = vec4(1.789, 2.718, 3.453, 2.0);
    vec4  color      = vec4(0.0, 0.0, 0.0, 0.0);

    color += textureLod(texSampler1D, coords1D, lod);
    color += textureProjLod(texSampler1D, coords2D, lod);
    color += textureProjLod(texSampler1D, coords4D, lod);
    
    color += textureLod     (texSampler2D, coords2D, lod);
    color += textureProjLod (texSampler2D, coords3D, lod);
    color += textureProjLod (texSampler2D, coords4D, lod);

    color += textureLod     (texSampler3D, coords3D, lod);
    color += textureProjLod (texSampler3D, coords4D, lod);
    
    color += textureLod (texSamplerCube, coords3D, lod);

    color += textureLod    (shadowSampler1D, coords3D, lod);
    color += textureLod    (shadowSampler2D, coords3D, lod);
    color += textureProjLod(shadowSampler1D, coords4D, lod);
    color += textureProjLod(shadowSampler2D, coords4D, lod);

    gl_Position = color;
}

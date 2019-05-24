#version 450

#extension GL_NV_shader_texture_footprint : require


layout (location = 0) in vec2 P2;
layout (location = 2) in vec3 P3;
layout (location = 3) in flat int granularity;
layout (location = 4) in float lodClamp;
layout (location = 5) in float lod;
layout (location = 6) in vec2 dx;
layout (location = 8) in vec2 dy;
layout (location = 9) in float bias;

uniform sampler2D sample2D;
uniform sampler3D sample3D;

buffer result2D {
    bool ret2D;
    uvec2 anchor2D;
    uvec2 offset2D;
    uvec2 mask2D;
    uint lod2D;
    uint granularity2D;
};

buffer result3D {
    bool ret3D;
    uvec3 anchor3D;
    uvec3 offset3D;
    uvec2 mask3D;
    uint lod3D;
    uint granularity3D;
};

void main() {
 gl_TextureFootprint2DNV fp2D;
 gl_TextureFootprint3DNV fp3D;
 
 ret2D = textureFootprintNV(sample2D, P2, granularity, true, fp2D);
 anchor2D = fp2D.anchor;
 offset2D = fp2D.offset;
 mask2D = fp2D.mask;
 lod2D = fp2D.lod;
 granularity2D = fp2D.granularity;
 
 ret2D = textureFootprintNV(sample2D, P2, granularity, true, fp2D, bias);
 anchor2D += fp2D.anchor;
 offset2D += fp2D.offset;
 mask2D += fp2D.mask;
 lod2D += fp2D.lod;
 granularity2D += fp2D.granularity;
 
 ret2D = textureFootprintClampNV(sample2D, P2, lodClamp, granularity, true, fp2D);
 anchor2D += fp2D.anchor;
 offset2D += fp2D.offset;
 mask2D += fp2D.mask;
 lod2D += fp2D.lod;
 granularity2D += fp2D.granularity;
 
 ret2D = textureFootprintClampNV(sample2D, P2, lodClamp, granularity, true, fp2D, bias);
 anchor2D += fp2D.anchor;
 offset2D += fp2D.offset;
 mask2D += fp2D.mask;
 lod2D += fp2D.lod;
 granularity2D += fp2D.granularity;
 
 ret2D = textureFootprintLodNV(sample2D, P2, lod, granularity, true, fp2D);
 anchor2D += fp2D.anchor;
 offset2D += fp2D.offset;
 mask2D += fp2D.mask;
 lod2D += fp2D.lod;
 granularity2D += fp2D.granularity;
 
 ret2D = textureFootprintGradNV(sample2D, P2, dx, dy, granularity, true, fp2D);
 anchor2D += fp2D.anchor;
 offset2D += fp2D.offset;
 mask2D += fp2D.mask;
 lod2D += fp2D.lod;
 granularity2D += fp2D.granularity;
 
 ret2D = textureFootprintGradClampNV(sample2D, P2, dx, dy, lodClamp, granularity, true, fp2D);
 anchor2D += fp2D.anchor;
 offset2D += fp2D.offset;
 mask2D += fp2D.mask;
 lod2D += fp2D.lod;
 granularity2D += fp2D.granularity;
 
 ret3D = textureFootprintNV(sample3D, P3, granularity, true, fp3D);
 anchor3D = fp3D.anchor;
 offset3D = fp3D.offset;
 mask3D = fp3D.mask;
 lod3D = fp3D.lod;
 granularity3D = fp3D.granularity;
 
 ret3D = textureFootprintNV(sample3D, P3, granularity, true, fp3D, bias);
 anchor3D += fp3D.anchor;
 offset3D += fp3D.offset;
 mask3D += fp3D.mask;
 lod3D += fp3D.lod;
 granularity3D += fp3D.granularity;
 
 ret3D = textureFootprintClampNV(sample3D, P3, lodClamp, granularity, true, fp3D);
 anchor3D += fp3D.anchor;
 offset3D += fp3D.offset;
 mask3D += fp3D.mask;
 lod3D += fp3D.lod;
 granularity3D += fp3D.granularity;
 
 ret3D = textureFootprintClampNV(sample3D, P3, lodClamp, granularity, true, fp3D, bias);
 anchor3D += fp3D.anchor;
 offset3D += fp3D.offset;
 mask3D += fp3D.mask;
 lod3D += fp3D.lod;
 granularity3D += fp3D.granularity;
 
 ret3D = textureFootprintLodNV(sample3D, P3, lod, granularity, true, fp3D);
 anchor3D += fp3D.anchor;
 offset3D += fp3D.offset;
 mask3D += fp3D.mask;
 lod3D += fp3D.lod;
 granularity3D += fp3D.granularity;
}
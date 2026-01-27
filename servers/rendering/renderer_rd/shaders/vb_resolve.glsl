#[compute]
#version 450
#VERSION_DEFINES

// ——— Workgroup ———
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// ——— I/O (imaginile trebuie pe set=0 ca în C++) ———
layout(rg32ui,  set = 0, binding = 0) uniform uimage2D vb_vis_image;
layout(rgba16f, set = 0, binding = 1) uniform image2D  out_color_image;

// opționale:
layout(rg16f,   set = 0, binding = 2) uniform image2D  vb_aux_image;
layout(r32f,    set = 0, binding = 3) uniform image2D  vb_depth_image;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    uvec2 packed = imageLoad(vb_vis_image, px).xy;

    // de test: scriem o culoare ca să verificăm că totul e legat corect
    vec4 c = vec4(
        float(packed.x & 255u) / 255.0,
        float((packed.x >> 8) & 255u) / 255.0,
        float((packed.y) & 255u) / 255.0,
        1.0
    );
    imageStore(out_color_image, px, c);
}


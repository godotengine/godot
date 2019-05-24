#version 150 core

layout(triangles_adjacency) in;
layout(max_vertices = 30) out;
layout(stream = 3, triangle_strip) out;

in fromVertex {
    in vec3 color;
} fromV[];

out toFragment {
    out vec3 color;
} toF;

out fromVertex {
    vec3 color;
};

void main()
{
    color = fromV[0].color;
    //?? gl_ClipDistance[3] = gl_in[1].gl_ClipDistance[2];
    gl_Position = gl_in[0].gl_Position;
    gl_PointSize = gl_in[3].gl_PointSize;
    gl_PrimitiveID = gl_PrimitiveIDIn;
    gl_Layer = 2;

    EmitVertex();

    color = 2 * fromV[0].color;
    gl_Position = 2.0 * gl_in[0].gl_Position;
    gl_PointSize = 2.0 * gl_in[3].gl_PointSize;
    gl_PrimitiveID = gl_PrimitiveIDIn + 1;
    gl_Layer = 3;

    EmitVertex();

    EndPrimitive();
}

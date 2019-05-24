#version 450 core

layout(vertices = 4) out;

layout(location=1) patch out vec4 patchOut;

struct S {
    float sMem1;  // should not see a patch decoration
    float sMem2;  // should not see a patch decoration
};

layout(location = 12) patch out TheBlock {
    highp float bMem1;  // should not see a location decoration
    highp float bMem2;
    S s;                // should see a patch decoration
} tcBlock[2];

void main()
{
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}

layout(location = 2) patch out SingleBlock {
    highp float bMem1;  // should not see a location decoration
    highp float bMem2;
    S s;                // should see a patch decoration
} singleBlock;

layout(location = 20) patch out bn {
                        vec4 v1; // location 20
  layout(location = 24) vec4 v2; // location 24
                        vec4 v3; // location 25
};
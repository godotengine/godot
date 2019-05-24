#version 450

buffer block {
    float m0;
    vec3 m4;
    //////
    float m16;
    layout(offset=20) vec3 m20;
    /////
    vec3 m32;
    /////
    vec2 m48;
    vec2 m56;
    ////
    float m64;
    vec2 m68;
    float m76;
    //////
    float m80;
    layout(offset=88) vec2 m88;
    //////
    vec2 m96;
    ///////
    dvec2 m112;
};

void main() {}
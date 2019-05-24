struct VertexOut {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD;
};
VertexOut r0() {
    const float f = 2.0;
    return (VertexOut)f;
}
VertexOut r1() {
    const float f = 2.0;
    return (VertexOut)(f + 1.0);
}
VertexOut r2() {
    const float f = 2.0;
    return (VertexOut)(sin(f));
}
VertexOut r3() {
    float f = 2.0;
    return (VertexOut)f;
}
VertexOut r4() {
    float f = 2.0;
    return (VertexOut)(f + 1.0);
}
VertexOut r5() {
    float f = 2.0;
    return (VertexOut)(sin(f));
}
VertexOut main() {
    VertexOut v0 = r0();
    VertexOut v1 = r1();
    VertexOut v2 = r2();
    VertexOut v3 = r3();
    VertexOut v4 = r4();
    VertexOut v5 = r5();
    return (VertexOut)1;
}

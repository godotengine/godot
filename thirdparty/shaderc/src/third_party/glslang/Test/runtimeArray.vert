#version 450 core

buffer bn {
    int a[];
    float b[];
} buf;

uniform un {
    int a[];
    float b[];
} ubuf;

buffer bna {
    int a[];
    float b[];
} bufa[4];

uniform una {
    int a[];
    float b[];
} ubufa[4];

buffer abn {
    int aba[];
    float abb[];
};

uniform aun {
    int aua[];
    float aub[];
};

layout(binding=1)                             uniform samplerBuffer       uniformTexelBufferDyn[];
layout(binding=2, r32f)                       uniform imageBuffer         storageTexelBufferDyn[];
layout(binding=3)                             uniform uname { float a; }  uniformBuffer[];
layout(binding=4)                             buffer  bname { float b; }  storageBuffer[];
layout(binding=5)                             uniform sampler2D           sampledImage[];
layout(binding=6, r32f)                       uniform image2D             storageImage[];
layout(binding=8)                             uniform samplerBuffer       uniformTexelBuffer[];
layout(binding=9, r32f)                       uniform imageBuffer         storageTexelBuffer[];

int i;

void main()
{
    ubuf.a[3];
    ubuf.b[3];
    buf.a[3];
    buf.b[3];

    ubufa[3].a[3];
    ubufa[3].b[3];
    bufa[3].a[3];
    bufa[3].b[3];

    aua[3];
    aub[3];
    aba[3];
    abb[3];

    ubuf.a[i];             // ERROR
    ubuf.b[i];             // ERROR
    buf.a[i];              // ERROR
    buf.b[i];

    ubuf.a.length();       // ERROR
    ubuf.b.length();       // ERROR
    buf.a.length();        // ERROR
    buf.b.length();

    ubufa[1].a[i];         // ERROR
    ubufa[1].b[i];         // ERROR
    bufa[1].a[i];          // ERROR
    bufa[1].b[i];

    ubufa[1].a.length();   // ERROR
    ubufa[1].b.length();   // ERROR
    bufa[1].a.length();    // ERROR
    bufa[1].b.length();

    aua[i];                // ERROR
    aub[i];                // ERROR
    aba[i];                // ERROR
    abb[i];

    aua.length();          // ERROR
    aub.length();          // ERROR
    aba.length();          // ERROR
    abb.length();

    uniformTexelBufferDyn[1];
    storageTexelBufferDyn[1];
    uniformBuffer[1];
    storageBuffer[1];
    sampledImage[1];
    storageImage[1];
    uniformTexelBuffer[1];
    storageTexelBuffer[1];

    uniformTexelBufferDyn[i];  // ERROR, need extension
    storageTexelBufferDyn[i];  // ERROR, need extension
    uniformBuffer[i];          // ERROR, need extension
    storageBuffer[i];          // ERROR, need extension
    sampledImage[i];           // ERROR, need extension
    storageImage[i];           // ERROR, need extension
    uniformTexelBuffer[i];     // ERROR, need extension
    storageTexelBuffer[i];     // ERROR, need extension

    float local[] = ubuf.b;    // ERROR, can initialize with runtime-sized array
}

#version 450 core

out gl_PerVertex {
    float gl_CullDistance[3];
};

void main()
{
    gl_CullDistance[2] = 4.5;
}

out bool outb;         // ERROR
out sampler2D outo;    // ERROR
out float outa[4];
out float outaa[4][2];
struct S { float f; };
out S outs;
out S[4] outasa;
out S outsa[4];
struct SA { float f[4]; };
out SA outSA;
struct SS { float f; S s; };
out SS outSS;

layout(binding = 0) uniform atomic_uint aui;
uint ui;

void foo()
{
    SS::f;
    atomicCounterAdd(aui, ui);           // ERROR, need 4.6
    atomicCounterSubtract(aui, ui);      // ERROR, need 4.6
    atomicCounterMin(aui, ui);           // ERROR, need 4.6
    atomicCounterMax(aui, ui);           // ERROR, need 4.6
    atomicCounterAnd(aui, ui);           // ERROR, need 4.6
    atomicCounterOr(aui, ui);            // ERROR, need 4.6
    atomicCounterXor(aui, ui);           // ERROR, need 4.6
    atomicCounterExchange(aui, ui);      // ERROR, need 4.6
    atomicCounterCompSwap(aui, ui, ui);  // ERROR, need 4.6

    int a = gl_BaseVertex + gl_BaseInstance + gl_DrawID; // ERROR, need 4.6

    bool b1;
    anyInvocation(b1);        // ERROR, need 4.6
    allInvocations(b1);       // ERROR, need 4.6
    allInvocationsEqual(b1);  // ERROR, need 4.6
}
; // ERROR: no extraneous semicolons

layout(location = 0) uniform locBlock {        // ERROR, no location uniform block
    int a;
};

layout(location = 0) buffer locBuffBlock {     // ERROR, no location on buffer block
    int b;
};

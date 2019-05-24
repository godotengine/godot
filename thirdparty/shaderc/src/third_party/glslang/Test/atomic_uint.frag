#version 420 core

layout(binding = 0) uniform atomic_uint counter;

uint func(atomic_uint c)
{
    return atomicCounterIncrement(c);
}

uint func2(out atomic_uint c) // ERROR
{
    return counter;           // ERROR, type mismatch
    return atomicCounter(counter);
}

void main()
{
     atomic_uint non_uniform_counter; // ERROR
     uint val = atomicCounter(counter);
     atomicCounterDecrement(counter);
}

layout(binding = 1, offset = 3) uniform atomic_uint countArr[4];
uniform int i;

void opac()
{
    counter + counter;  // ERROR
    -counter;           // ERROR
    int a[3];
    a[counter];         // ERROR
    countArr[2];
    countArr[i];
    counter = 4;        // ERROR
}

in atomic_uint acin;    // ERROR
atomic_uint acg;        // ERROR
uniform atomic_uint;
uniform atomic_uint aNoBind;                          // ERROR, no binding
layout(binding=0, offset=32) uniform atomic_uint aOffset;
layout(binding=0, offset=4) uniform atomic_uint;
layout(binding=0) uniform atomic_uint bar3;           // offset is 4
layout(binding=0) uniform atomic_uint ac[3];          // offset = 8
layout(binding=0) uniform atomic_uint ad;             // offset = 20
layout(offset=8) uniform atomic_uint bar4;            // ERROR, no binding
layout(binding = 0, offset = 12) uniform atomic_uint overlap;  // ERROR, overlapping offsets
layout(binding = 20) uniform atomic_uint bigBind;     // ERROR, binding too big

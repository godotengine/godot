#version 460 core

struct S {
    float f;
    vec4 v;
};

in S s;

void main()
{
    interpolateAtCentroid(s.v);
    bool b1;
    b1 = anyInvocation(b1);
    b1 = allInvocations(b1);
    b1 = allInvocationsEqual(b1);
}

void attExtBad()
{
    // ERRORs, not enabled
    [[dependency_length(1+3)]] for (int i = 0; i < 8; ++i) { }
    [[flatten]]                if (true) { } else { }
}

#extension GL_EXT_control_flow_attributes : enable

void attExt()
{
    [[dependency_length(-3)]] do {  } while(true); // ERROR, not positive
    [[dependency_length(0)]] do {  } while(true);  // ERROR, not positive
}

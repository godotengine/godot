#version 450

layout(constant_id = 200) const float scf1 = 1.0;
layout(constant_id = 201) const bool scbt = true;
layout(constant_id = 202) const int sci2 = 2;

void main()
{
    bool(scf1);   // not a spec-const
    bool(scbt);   // spec-const
    bool(sci2);   // spec-const

    float(scf1);   // not a spec-const
    float(scbt);   // not a spec-const
    float(sci2);   // not a spec-const

    int(scf1);   // not a spec-const
    int(scbt);   // spec-const
    int(sci2);   // spec-const

    scf1 * scf1;   // not a spec-const
    scbt || scbt;  // spec-const
    sci2 * sci2;   // spec-const
    scf1 + sci2;   // implicit conversion not a spec-const

    -scf1;     // not a spec-const
    !scbt;     // spec-const
    -sci2;     // spec-const

    scf1 > scf1;   // not a spec-const
    sci2 > sci2;   // spec-const

    scf1 != scf1;  // not a spec-const
    scbt != scbt;  // spec-const
    sci2 != sci2;  // spec-const

    ivec2(sci2, sci2);   // spec-const
    ivec2[2](ivec2(sci2, sci2), ivec2(sci2, sci2)); // not a spec-const

    vec2(scf1, scf1);   // not spec-const
    vec2[2](vec2(scf1, scf1), vec2(scf1, scf1)); // not a spec-const
}

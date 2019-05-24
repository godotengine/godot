
// array size from initializer
static float g_array [ ] = { 1, 2, 3, 4, 5 };

// Unused: array size from initializer
static float g_array_unused [ ] = { 1, 2, 3, 4, 5, 6, 7 };

// Test initializer sizing for arrayed structs
static struct mystruct {
    int i;
    float f;
} g_mystruct[] = {
    { 1, 2.0 },
    { 3, 4.0 },
};

struct PS_OUTPUT { float4 color : SV_Target0; };

// INVALID: implicit size requires an initializer expression.
// uniform float bad[];

// INVALID: function parameters cannot be implicitly sized
// void BadFunction(int a[]) { }

void main(out PS_OUTPUT ps_output)
{
    // local array sized from initializers
    float l_array[] = { 1, 2, 3 };
    int idx;

    ps_output.color = g_array[0] + g_array[4] + l_array[1] + g_mystruct[0].f + g_array[idx];
}

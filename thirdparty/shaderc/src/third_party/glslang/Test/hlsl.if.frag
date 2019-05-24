float4 PixelShaderFunction(float4 input) : COLOR0
{
    if (all(input == input))
        return input;

    if (all(input == input))
        return input;
    else
        return -input;

    if (all(input == input))
        ;

    if (all(input == input))
        ;
    else
        ;

    [flatten] if (all(input == input)) {
        return input;
    }

    if (all(input == input)) {
        return input;
    } else {
        return -input;
    }

	int ii;
	if (float ii = input.z)
	    ++ii;
	++ii;
    if (float(ii) == 1.0)
        ++ii;
}

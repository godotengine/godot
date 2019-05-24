
float PixelShaderFunction()
{
    // TODO: All of the below should fail, although presently the first failure
    // aborts compilation and the rest are skipped.  Having a separate test for
    // each would be cumbersome.

    vector<void, 2>    r00;  // cannot declare vectors of voids
    matrix<void, 2, 3> r01;  // cannot declare matrices of voids

    vector<float, 2, 3> r02;  // too many parameters to vector
    matrix<float, 2>    r03;  // not enough parameters to matrix

    int three = 3;
    vector<void, three> r04; // size must be a literal constant integer
    matrix<void, three, three> r05; // size must be a literal constant integer

    vector<vector<int, 3>, 3> r06;  // type must be a simple scalar
    vector<float3, 3> r07;          // type must be a simple scalar

    return 0.0;
}


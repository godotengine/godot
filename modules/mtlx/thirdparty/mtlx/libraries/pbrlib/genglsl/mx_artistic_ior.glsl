void mx_artistic_ior(vec3 reflectivity, vec3 edge_color, out vec3 ior, out vec3 extinction)
{
    // "Artist Friendly Metallic Fresnel", Ole Gulbrandsen, 2014
    // http://jcgt.org/published/0003/04/03/paper.pdf

    vec3 r = clamp(reflectivity, 0.0, 0.99);
    vec3 r_sqrt = sqrt(r);
    vec3 n_min = (1.0 - r) / (1.0 + r);
    vec3 n_max = (1.0 + r_sqrt) / (1.0 - r_sqrt);
    ior = mix(n_max, n_min, edge_color);

    vec3 np1 = ior + 1.0;
    vec3 nm1 = ior - 1.0;
    vec3 k2 = (np1*np1 * r - nm1*nm1) / (1.0 - r);
    k2 = max(k2, 0.0);
    extinction = sqrt(k2);
}

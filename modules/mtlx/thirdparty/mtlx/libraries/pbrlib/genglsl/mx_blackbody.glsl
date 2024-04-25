/// XYZ to Rec.709 RGB colorspace conversion
const mat3 XYZ_to_RGB = mat3( 3.2406, -0.9689, 0.0557,
                             -1.5372, 1.8758, -0.2040,
                             -0.4986, 0.0415, 1.0570);

void mx_blackbody(float temperatureKelvin, out vec3 colorValue)
{
    float xc, yc;
    float t, t2, t3, xc2, xc3;

    // if value outside valid range of approximation clamp to accepted temperature range
    temperatureKelvin = clamp(temperatureKelvin, 1667.0, 25000.0);

    t = 1000.0 / temperatureKelvin;
    t2 = t * t;
    t3 = t * t * t;

    // Cubic spline approximation for Kelvin temperature to sRGB conversion
    // (https://en.wikipedia.org/wiki/Planckian_locus#Approximation)
    if (temperatureKelvin < 4000.0) {  // 1667K <= temperatureKelvin < 4000K
      xc = -0.2661239 * t3 - 0.2343580 * t2 + 0.8776956 * t + 0.179910;
    }
    else {  // 4000K <= temperatureKelvin <= 25000K
      xc = -3.0258469 * t3 + 2.1070379 * t2 + 0.2226347 * t + 0.240390;
    }
    xc2 = xc * xc;
    xc3 = xc * xc * xc;

    if (temperatureKelvin < 2222.0) {  // 1667K <= temperatureKelvin < 2222K
      yc = -1.1063814 * xc3 - 1.34811020 * xc2 + 2.18555832 * xc - 0.20219683;
    }
    else if (temperatureKelvin < 4000.0) {  // 2222K <= temperatureKelvin < 4000K
      yc = -0.9549476 * xc3 - 1.37418593 * xc2 + 2.09137015 * xc - 0.16748867;
    }
    else {  // 4000K <= temperatureKelvin <= 25000K
      yc = 3.0817580 * xc3 - 5.87338670 * xc2 + 3.75112997 * xc - 0.37001483;
    }

    if (yc <= 0.0) {  // avoid division by zero
      colorValue = vec3(1.0);
      return;
    }

    vec3 XYZ = vec3(xc / yc, 1.0, (1.0 - xc - yc) / yc);

    colorValue = XYZ_to_RGB * XYZ;
    colorValue = max(colorValue, vec3(0.0));
}

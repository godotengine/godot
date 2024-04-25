/*
Color transform functions.

These funcions are modified versions of the color operators found in Open Shading Language:
github.com/imageworks/OpenShadingLanguage/blob/master/src/liboslexec/opcolor.cpp

It contains the subset of color operators needed to implement the MaterialX
standard library. The modifications are for conversions from C++ to GLSL.

Original copyright notice:
------------------------------------------------------------------------
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
------------------------------------------------------------------------
*/

vec3 mx_hsvtorgb(vec3 hsv)
{
    // Reference for this technique: Foley & van Dam
    float h = hsv.x; float s = hsv.y; float v = hsv.z;
    if (s < 0.0001f) {
      return vec3 (v, v, v);
    } else {
        h = 6.0f * (h - floor(h));  // expand to [0..6)
        int hi = int(trunc(h));
        float f = h - float(hi);
        float p = v * (1.0f-s);
        float q = v * (1.0f-s*f);
        float t = v * (1.0f-s*(1.0f-f));
        if (hi == 0)
            return vec3 (v, t, p);
        else if (hi == 1)
            return vec3 (q, v, p);
        else if (hi == 2)
            return vec3 (p, v, t);
        else if (hi == 3)
            return vec3 (p, q, v);
        else if (hi == 4)
            return vec3 (t, p, v);
        return vec3 (v, p, q);
    }
}


vec3 mx_rgbtohsv(vec3 c)
{
    // See Foley & van Dam
    float r = c.x; float g = c.y; float b = c.z;
    float mincomp = min (r, min(g, b));
    float maxcomp = max (r, max(g, b));
    float delta = maxcomp - mincomp;  // chroma
    float h, s, v;
    v = maxcomp;
    if (maxcomp > 0.0f)
        s = delta / maxcomp;
    else s = 0.0f;
    if (s <= 0.0f)
        h = 0.0f;
    else {
        if      (r >= maxcomp) h = (g-b) / delta;
        else if (g >= maxcomp) h = 2.0f + (b-r) / delta;
        else                   h = 4.0f + (r-g) / delta;
        h *= (1.0f/6.0f);
        if (h < 0.0f)
            h += 1.0f;
    }
    return vec3(h, s, v);
}

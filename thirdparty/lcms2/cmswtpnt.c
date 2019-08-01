//---------------------------------------------------------------------------------
//
//  Little Color Management System
//  Copyright (c) 1998-2017 Marti Maria Saguer
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//---------------------------------------------------------------------------------
//

#include "lcms2_internal.h"


// D50 - Widely used
const cmsCIEXYZ* CMSEXPORT cmsD50_XYZ(void)
{
    static cmsCIEXYZ D50XYZ = {cmsD50X, cmsD50Y, cmsD50Z};

    return &D50XYZ;
}

const cmsCIExyY* CMSEXPORT cmsD50_xyY(void)
{
    static cmsCIExyY D50xyY;

    cmsXYZ2xyY(&D50xyY, cmsD50_XYZ());

    return &D50xyY;
}

// Obtains WhitePoint from Temperature
cmsBool  CMSEXPORT cmsWhitePointFromTemp(cmsCIExyY* WhitePoint, cmsFloat64Number TempK)
{
    cmsFloat64Number x, y;
    cmsFloat64Number T, T2, T3;
    // cmsFloat64Number M1, M2;

    _cmsAssert(WhitePoint != NULL);

    T = TempK;
    T2 = T*T;            // Square
    T3 = T2*T;           // Cube

    // For correlated color temperature (T) between 4000K and 7000K:

    if (T >= 4000. && T <= 7000.)
    {
        x = -4.6070*(1E9/T3) + 2.9678*(1E6/T2) + 0.09911*(1E3/T) + 0.244063;
    }
    else
        // or for correlated color temperature (T) between 7000K and 25000K:

        if (T > 7000.0 && T <= 25000.0)
        {
            x = -2.0064*(1E9/T3) + 1.9018*(1E6/T2) + 0.24748*(1E3/T) + 0.237040;
        }
        else {
            cmsSignalError(0, cmsERROR_RANGE, "cmsWhitePointFromTemp: invalid temp");
            return FALSE;
        }

        // Obtain y(x)
        y = -3.000*(x*x) + 2.870*x - 0.275;

        // wave factors (not used, but here for futures extensions)

        // M1 = (-1.3515 - 1.7703*x + 5.9114 *y)/(0.0241 + 0.2562*x - 0.7341*y);
        // M2 = (0.0300 - 31.4424*x + 30.0717*y)/(0.0241 + 0.2562*x - 0.7341*y);

        WhitePoint -> x = x;
        WhitePoint -> y = y;
        WhitePoint -> Y = 1.0;

        return TRUE;
}



typedef struct {

    cmsFloat64Number mirek;  // temp (in microreciprocal kelvin)
    cmsFloat64Number ut;     // u coord of intersection w/ blackbody locus
    cmsFloat64Number vt;     // v coord of intersection w/ blackbody locus
    cmsFloat64Number tt;     // slope of ISOTEMPERATURE. line

    } ISOTEMPERATURE;

static const ISOTEMPERATURE isotempdata[] = {
//  {Mirek, Ut,       Vt,      Tt      }
    {0,     0.18006,  0.26352,  -0.24341},
    {10,    0.18066,  0.26589,  -0.25479},
    {20,    0.18133,  0.26846,  -0.26876},
    {30,    0.18208,  0.27119,  -0.28539},
    {40,    0.18293,  0.27407,  -0.30470},
    {50,    0.18388,  0.27709,  -0.32675},
    {60,    0.18494,  0.28021,  -0.35156},
    {70,    0.18611,  0.28342,  -0.37915},
    {80,    0.18740,  0.28668,  -0.40955},
    {90,    0.18880,  0.28997,  -0.44278},
    {100,   0.19032,  0.29326,  -0.47888},
    {125,   0.19462,  0.30141,  -0.58204},
    {150,   0.19962,  0.30921,  -0.70471},
    {175,   0.20525,  0.31647,  -0.84901},
    {200,   0.21142,  0.32312,  -1.0182 },
    {225,   0.21807,  0.32909,  -1.2168 },
    {250,   0.22511,  0.33439,  -1.4512 },
    {275,   0.23247,  0.33904,  -1.7298 },
    {300,   0.24010,  0.34308,  -2.0637 },
    {325,   0.24702,  0.34655,  -2.4681 },
    {350,   0.25591,  0.34951,  -2.9641 },
    {375,   0.26400,  0.35200,  -3.5814 },
    {400,   0.27218,  0.35407,  -4.3633 },
    {425,   0.28039,  0.35577,  -5.3762 },
    {450,   0.28863,  0.35714,  -6.7262 },
    {475,   0.29685,  0.35823,  -8.5955 },
    {500,   0.30505,  0.35907,  -11.324 },
    {525,   0.31320,  0.35968,  -15.628 },
    {550,   0.32129,  0.36011,  -23.325 },
    {575,   0.32931,  0.36038,  -40.770 },
    {600,   0.33724,  0.36051,  -116.45  }
};

#define NISO sizeof(isotempdata)/sizeof(ISOTEMPERATURE)


// Robertson's method
cmsBool  CMSEXPORT cmsTempFromWhitePoint(cmsFloat64Number* TempK, const cmsCIExyY* WhitePoint)
{
    cmsUInt32Number j;
    cmsFloat64Number us,vs;
    cmsFloat64Number uj,vj,tj,di,dj,mi,mj;
    cmsFloat64Number xs, ys;

    _cmsAssert(WhitePoint != NULL);
    _cmsAssert(TempK != NULL);

    di = mi = 0;
    xs = WhitePoint -> x;
    ys = WhitePoint -> y;

    // convert (x,y) to CIE 1960 (u,WhitePoint)

    us = (2*xs) / (-xs + 6*ys + 1.5);
    vs = (3*ys) / (-xs + 6*ys + 1.5);


    for (j=0; j < NISO; j++) {

        uj = isotempdata[j].ut;
        vj = isotempdata[j].vt;
        tj = isotempdata[j].tt;
        mj = isotempdata[j].mirek;

        dj = ((vs - vj) - tj * (us - uj)) / sqrt(1.0 + tj * tj);

        if ((j != 0) && (di/dj < 0.0)) {

            // Found a match
            *TempK = 1000000.0 / (mi + (di / (di - dj)) * (mj - mi));
            return TRUE;
        }

        di = dj;
        mi = mj;
    }

    // Not found
    return FALSE;
}


// Compute chromatic adaptation matrix using Chad as cone matrix

static
cmsBool ComputeChromaticAdaptation(cmsMAT3* Conversion,
                                const cmsCIEXYZ* SourceWhitePoint,
                                const cmsCIEXYZ* DestWhitePoint,
                                const cmsMAT3* Chad)

{

    cmsMAT3 Chad_Inv;
    cmsVEC3 ConeSourceXYZ, ConeSourceRGB;
    cmsVEC3 ConeDestXYZ, ConeDestRGB;
    cmsMAT3 Cone, Tmp;


    Tmp = *Chad;
    if (!_cmsMAT3inverse(&Tmp, &Chad_Inv)) return FALSE;

    _cmsVEC3init(&ConeSourceXYZ, SourceWhitePoint -> X,
                             SourceWhitePoint -> Y,
                             SourceWhitePoint -> Z);

    _cmsVEC3init(&ConeDestXYZ,   DestWhitePoint -> X,
                             DestWhitePoint -> Y,
                             DestWhitePoint -> Z);

    _cmsMAT3eval(&ConeSourceRGB, Chad, &ConeSourceXYZ);
    _cmsMAT3eval(&ConeDestRGB,   Chad, &ConeDestXYZ);

    // Build matrix
    _cmsVEC3init(&Cone.v[0], ConeDestRGB.n[0]/ConeSourceRGB.n[0],    0.0,  0.0);
    _cmsVEC3init(&Cone.v[1], 0.0,   ConeDestRGB.n[1]/ConeSourceRGB.n[1],   0.0);
    _cmsVEC3init(&Cone.v[2], 0.0,   0.0,   ConeDestRGB.n[2]/ConeSourceRGB.n[2]);


    // Normalize
    _cmsMAT3per(&Tmp, &Cone, Chad);
    _cmsMAT3per(Conversion, &Chad_Inv, &Tmp);

    return TRUE;
}

// Returns the final chrmatic adaptation from illuminant FromIll to Illuminant ToIll
// The cone matrix can be specified in ConeMatrix. If NULL, Bradford is assumed
cmsBool  _cmsAdaptationMatrix(cmsMAT3* r, const cmsMAT3* ConeMatrix, const cmsCIEXYZ* FromIll, const cmsCIEXYZ* ToIll)
{
    cmsMAT3 LamRigg   = {{ // Bradford matrix
        {{  0.8951,  0.2664, -0.1614 }},
        {{ -0.7502,  1.7135,  0.0367 }},
        {{  0.0389, -0.0685,  1.0296 }}
    }};

    if (ConeMatrix == NULL)
        ConeMatrix = &LamRigg;

    return ComputeChromaticAdaptation(r, FromIll, ToIll, ConeMatrix);
}

// Same as anterior, but assuming D50 destination. White point is given in xyY
static
cmsBool _cmsAdaptMatrixToD50(cmsMAT3* r, const cmsCIExyY* SourceWhitePt)
{
    cmsCIEXYZ Dn;
    cmsMAT3 Bradford;
    cmsMAT3 Tmp;

    cmsxyY2XYZ(&Dn, SourceWhitePt);

    if (!_cmsAdaptationMatrix(&Bradford, NULL, &Dn, cmsD50_XYZ())) return FALSE;

    Tmp = *r;
    _cmsMAT3per(r, &Bradford, &Tmp);

    return TRUE;
}

// Build a White point, primary chromas transfer matrix from RGB to CIE XYZ
// This is just an approximation, I am not handling all the non-linear
// aspects of the RGB to XYZ process, and assumming that the gamma correction
// has transitive property in the transformation chain.
//
// the alghoritm:
//
//            - First I build the absolute conversion matrix using
//              primaries in XYZ. This matrix is next inverted
//            - Then I eval the source white point across this matrix
//              obtaining the coeficients of the transformation
//            - Then, I apply these coeficients to the original matrix
//
cmsBool _cmsBuildRGB2XYZtransferMatrix(cmsMAT3* r, const cmsCIExyY* WhitePt, const cmsCIExyYTRIPLE* Primrs)
{
    cmsVEC3 WhitePoint, Coef;
    cmsMAT3 Result, Primaries;
    cmsFloat64Number xn, yn;
    cmsFloat64Number xr, yr;
    cmsFloat64Number xg, yg;
    cmsFloat64Number xb, yb;

    xn = WhitePt -> x;
    yn = WhitePt -> y;
    xr = Primrs -> Red.x;
    yr = Primrs -> Red.y;
    xg = Primrs -> Green.x;
    yg = Primrs -> Green.y;
    xb = Primrs -> Blue.x;
    yb = Primrs -> Blue.y;

    // Build Primaries matrix
    _cmsVEC3init(&Primaries.v[0], xr,        xg,         xb);
    _cmsVEC3init(&Primaries.v[1], yr,        yg,         yb);
    _cmsVEC3init(&Primaries.v[2], (1-xr-yr), (1-xg-yg),  (1-xb-yb));


    // Result = Primaries ^ (-1) inverse matrix
    if (!_cmsMAT3inverse(&Primaries, &Result))
        return FALSE;


    _cmsVEC3init(&WhitePoint, xn/yn, 1.0, (1.0-xn-yn)/yn);

    // Across inverse primaries ...
    _cmsMAT3eval(&Coef, &Result, &WhitePoint);

    // Give us the Coefs, then I build transformation matrix
    _cmsVEC3init(&r -> v[0], Coef.n[VX]*xr,          Coef.n[VY]*xg,          Coef.n[VZ]*xb);
    _cmsVEC3init(&r -> v[1], Coef.n[VX]*yr,          Coef.n[VY]*yg,          Coef.n[VZ]*yb);
    _cmsVEC3init(&r -> v[2], Coef.n[VX]*(1.0-xr-yr), Coef.n[VY]*(1.0-xg-yg), Coef.n[VZ]*(1.0-xb-yb));


    return _cmsAdaptMatrixToD50(r, WhitePt);

}


// Adapts a color to a given illuminant. Original color is expected to have
// a SourceWhitePt white point.
cmsBool CMSEXPORT cmsAdaptToIlluminant(cmsCIEXYZ* Result,
                                       const cmsCIEXYZ* SourceWhitePt,
                                       const cmsCIEXYZ* Illuminant,
                                       const cmsCIEXYZ* Value)
{
    cmsMAT3 Bradford;
    cmsVEC3 In, Out;

    _cmsAssert(Result != NULL);
    _cmsAssert(SourceWhitePt != NULL);
    _cmsAssert(Illuminant != NULL);
    _cmsAssert(Value != NULL);

    if (!_cmsAdaptationMatrix(&Bradford, NULL, SourceWhitePt, Illuminant)) return FALSE;

    _cmsVEC3init(&In, Value -> X, Value -> Y, Value -> Z);
    _cmsMAT3eval(&Out, &Bradford, &In);

    Result -> X = Out.n[0];
    Result -> Y = Out.n[1];
    Result -> Z = Out.n[2];

    return TRUE;
}



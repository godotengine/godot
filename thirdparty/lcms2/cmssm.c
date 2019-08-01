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


// ------------------------------------------------------------------------

// Gamut boundary description by using Jan Morovic's Segment maxima method
// Many thanks to Jan for allowing me to use his algorithm.

// r = C*
// alpha = Hab
// theta = L*

#define SECTORS 16      // number of divisions in alpha and theta

// Spherical coordinates
typedef struct {

    cmsFloat64Number r;
    cmsFloat64Number alpha;
    cmsFloat64Number theta;

} cmsSpherical;

typedef  enum {
        GP_EMPTY,
        GP_SPECIFIED,
        GP_MODELED

    } GDBPointType;


typedef struct {

    GDBPointType Type;
    cmsSpherical p;         // Keep also alpha & theta of maximum

} cmsGDBPoint;


typedef struct {

    cmsContext ContextID;
    cmsGDBPoint Gamut[SECTORS][SECTORS];

} cmsGDB;


// A line using the parametric form
// P = a + t*u
typedef struct {

    cmsVEC3 a;
    cmsVEC3 u;

} cmsLine;


// A plane using the parametric form
// Q = b + r*v + s*w
typedef struct {

    cmsVEC3 b;
    cmsVEC3 v;
    cmsVEC3 w;

} cmsPlane;



// --------------------------------------------------------------------------------------------

// ATAN2() which always returns degree positive numbers

static
cmsFloat64Number _cmsAtan2(cmsFloat64Number y, cmsFloat64Number x)
{
    cmsFloat64Number a;

    // Deal with undefined case
    if (x == 0.0 && y == 0.0) return 0;

    a = (atan2(y, x) * 180.0) / M_PI;

    while (a < 0) {
        a += 360;
    }

    return a;
}

// Convert to spherical coordinates
static
void ToSpherical(cmsSpherical* sp, const cmsVEC3* v)
{

    cmsFloat64Number L, a, b;

    L = v ->n[VX];
    a = v ->n[VY];
    b = v ->n[VZ];

    sp ->r = sqrt( L*L + a*a + b*b );

   if (sp ->r == 0) {
        sp ->alpha = sp ->theta = 0;
        return;
    }

    sp ->alpha = _cmsAtan2(a, b);
    sp ->theta = _cmsAtan2(sqrt(a*a + b*b), L);
}


// Convert to cartesian from spherical
static
void ToCartesian(cmsVEC3* v, const cmsSpherical* sp)
{
    cmsFloat64Number sin_alpha;
    cmsFloat64Number cos_alpha;
    cmsFloat64Number sin_theta;
    cmsFloat64Number cos_theta;
    cmsFloat64Number L, a, b;

    sin_alpha = sin((M_PI * sp ->alpha) / 180.0);
    cos_alpha = cos((M_PI * sp ->alpha) / 180.0);
    sin_theta = sin((M_PI * sp ->theta) / 180.0);
    cos_theta = cos((M_PI * sp ->theta) / 180.0);

    a = sp ->r * sin_theta * sin_alpha;
    b = sp ->r * sin_theta * cos_alpha;
    L = sp ->r * cos_theta;

    v ->n[VX] = L;
    v ->n[VY] = a;
    v ->n[VZ] = b;
}


// Quantize sector of a spherical coordinate. Saturate 360, 180 to last sector
// The limits are the centers of each sector, so
static
void QuantizeToSector(const cmsSpherical* sp, int* alpha, int* theta)
{
    *alpha = (int) floor(((sp->alpha * (SECTORS)) / 360.0) );
    *theta = (int) floor(((sp->theta * (SECTORS)) / 180.0) );

    if (*alpha >= SECTORS)
        *alpha = SECTORS-1;
    if (*theta >= SECTORS)
        *theta = SECTORS-1;
}


// Line determined by 2 points
static
void LineOf2Points(cmsLine* line, cmsVEC3* a, cmsVEC3* b)
{

    _cmsVEC3init(&line ->a, a ->n[VX], a ->n[VY], a ->n[VZ]);
    _cmsVEC3init(&line ->u, b ->n[VX] - a ->n[VX],
                            b ->n[VY] - a ->n[VY],
                            b ->n[VZ] - a ->n[VZ]);
}


// Evaluate parametric line
static
void GetPointOfLine(cmsVEC3* p, const cmsLine* line, cmsFloat64Number t)
{
    p ->n[VX] = line ->a.n[VX] + t * line->u.n[VX];
    p ->n[VY] = line ->a.n[VY] + t * line->u.n[VY];
    p ->n[VZ] = line ->a.n[VZ] + t * line->u.n[VZ];
}



/*
    Closest point in sector line1 to sector line2 (both are defined as 0 <=t <= 1)
    http://softsurfer.com/Archive/algorithm_0106/algorithm_0106.htm

    Copyright 2001, softSurfer (www.softsurfer.com)
    This code may be freely used and modified for any purpose
    providing that this copyright notice is included with it.
    SoftSurfer makes no warranty for this code, and cannot be held
    liable for any real or imagined damage resulting from its use.
    Users of this code must verify correctness for their application.

*/

static
cmsBool ClosestLineToLine(cmsVEC3* r, const cmsLine* line1, const cmsLine* line2)
{
    cmsFloat64Number a, b, c, d, e, D;
    cmsFloat64Number sc, sN, sD;
    //cmsFloat64Number tc; // left for future use
    cmsFloat64Number tN, tD;
    cmsVEC3 w0;

    _cmsVEC3minus(&w0, &line1 ->a, &line2 ->a);

    a  = _cmsVEC3dot(&line1 ->u, &line1 ->u);
    b  = _cmsVEC3dot(&line1 ->u, &line2 ->u);
    c  = _cmsVEC3dot(&line2 ->u, &line2 ->u);
    d  = _cmsVEC3dot(&line1 ->u, &w0);
    e  = _cmsVEC3dot(&line2 ->u, &w0);

    D  = a*c - b * b;      // Denominator
    sD = tD = D;           // default sD = D >= 0

    if (D <  MATRIX_DET_TOLERANCE) {   // the lines are almost parallel

        sN = 0.0;        // force using point P0 on segment S1
        sD = 1.0;        // to prevent possible division by 0.0 later
        tN = e;
        tD = c;
    }
    else {                // get the closest points on the infinite lines

        sN = (b*e - c*d);
        tN = (a*e - b*d);

        if (sN < 0.0) {       // sc < 0 => the s=0 edge is visible

            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if (sN > sD) {   // sc > 1 => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {           // tc < 0 => the t=0 edge is visible

        tN = 0.0;
        // recompute sc for this edge
        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    }
    else if (tN > tD) {      // tc > 1 => the t=1 edge is visible

        tN = tD;

        // recompute sc for this edge
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d + b);
            sD = a;
        }
    }
    // finally do the division to get sc and tc
    sc = (fabs(sN) < MATRIX_DET_TOLERANCE ? 0.0 : sN / sD);
    //tc = (fabs(tN) < MATRIX_DET_TOLERANCE ? 0.0 : tN / tD); // left for future use.

    GetPointOfLine(r, line1, sc);
    return TRUE;
}



// ------------------------------------------------------------------ Wrapper


// Allocate & free structure
cmsHANDLE  CMSEXPORT cmsGBDAlloc(cmsContext ContextID)
{
    cmsGDB* gbd = (cmsGDB*) _cmsMallocZero(ContextID, sizeof(cmsGDB));
    if (gbd == NULL) return NULL;

    gbd -> ContextID = ContextID;

    return (cmsHANDLE) gbd;
}


void CMSEXPORT cmsGBDFree(cmsHANDLE hGBD)
{
    cmsGDB* gbd = (cmsGDB*) hGBD;
    if (hGBD != NULL)
        _cmsFree(gbd->ContextID, (void*) gbd);
}


// Auxiliary to retrieve a pointer to the segmentr containing the Lab value
static
cmsGDBPoint* GetPoint(cmsGDB* gbd, const cmsCIELab* Lab, cmsSpherical* sp)
{
    cmsVEC3 v;
    int alpha, theta;

    // Housekeeping
    _cmsAssert(gbd != NULL);
    _cmsAssert(Lab != NULL);
    _cmsAssert(sp != NULL);

    // Center L* by subtracting half of its domain, that's 50
    _cmsVEC3init(&v, Lab ->L - 50.0, Lab ->a, Lab ->b);

    // Convert to spherical coordinates
    ToSpherical(sp, &v);

    if (sp ->r < 0 || sp ->alpha < 0 || sp->theta < 0) {
         cmsSignalError(gbd ->ContextID, cmsERROR_RANGE, "spherical value out of range");
         return NULL;
    }

    // On which sector it falls?
    QuantizeToSector(sp, &alpha, &theta);

    if (alpha < 0 || theta < 0 || alpha >= SECTORS || theta >= SECTORS) {
         cmsSignalError(gbd ->ContextID, cmsERROR_RANGE, " quadrant out of range");
         return NULL;
    }

    // Get pointer to the sector
    return &gbd ->Gamut[theta][alpha];
}

// Add a point to gamut descriptor. Point to add is in Lab color space.
// GBD is centered on a=b=0 and L*=50
cmsBool CMSEXPORT cmsGDBAddPoint(cmsHANDLE hGBD, const cmsCIELab* Lab)
{
    cmsGDB* gbd = (cmsGDB*) hGBD;
    cmsGDBPoint* ptr;
    cmsSpherical sp;


    // Get pointer to the sector
    ptr = GetPoint(gbd, Lab, &sp);
    if (ptr == NULL) return FALSE;

    // If no samples at this sector, add it
    if (ptr ->Type == GP_EMPTY) {

        ptr -> Type = GP_SPECIFIED;
        ptr -> p    = sp;
    }
    else {


        // Substitute only if radius is greater
        if (sp.r > ptr -> p.r) {

                ptr -> Type = GP_SPECIFIED;
                ptr -> p    = sp;
        }
    }

    return TRUE;
}

// Check if a given point falls inside gamut
cmsBool CMSEXPORT cmsGDBCheckPoint(cmsHANDLE hGBD, const cmsCIELab* Lab)
{
    cmsGDB* gbd = (cmsGDB*) hGBD;
    cmsGDBPoint* ptr;
    cmsSpherical sp;

    // Get pointer to the sector
    ptr = GetPoint(gbd, Lab, &sp);
    if (ptr == NULL) return FALSE;

    // If no samples at this sector, return no data
    if (ptr ->Type == GP_EMPTY) return FALSE;

    // In gamut only if radius is greater

    return (sp.r <= ptr -> p.r);
}

// -----------------------------------------------------------------------------------------------------------------------

// Find near sectors. The list of sectors found is returned on Close[].
// The function returns the number of sectors as well.

// 24   9  10  11  12
// 23   8   1   2  13
// 22   7   *   3  14
// 21   6   5   4  15
// 20  19  18  17  16
//
// Those are the relative movements
// {-2,-2}, {-1, -2}, {0, -2}, {+1, -2}, {+2,  -2},
// {-2,-1}, {-1, -1}, {0, -1}, {+1, -1}, {+2,  -1},
// {-2, 0}, {-1,  0}, {0,  0}, {+1,  0}, {+2,   0},
// {-2,+1}, {-1, +1}, {0, +1}, {+1,  +1}, {+2,  +1},
// {-2,+2}, {-1, +2}, {0, +2}, {+1,  +2}, {+2,  +2}};


static
const struct _spiral {

    int AdvX, AdvY;

    } Spiral[] = { {0,  -1}, {+1, -1}, {+1,  0}, {+1, +1}, {0,  +1}, {-1, +1},
                   {-1,  0}, {-1, -1}, {-1, -2}, {0,  -2}, {+1, -2}, {+2, -2},
                   {+2, -1}, {+2,  0}, {+2, +1}, {+2, +2}, {+1, +2}, {0,  +2},
                   {-1, +2}, {-2, +2}, {-2, +1}, {-2, 0},  {-2, -1}, {-2, -2} };

#define NSTEPS (sizeof(Spiral) / sizeof(struct _spiral))

static
int FindNearSectors(cmsGDB* gbd, int alpha, int theta, cmsGDBPoint* Close[])
{
    int nSectors = 0;
    int a, t;
    cmsUInt32Number i;
    cmsGDBPoint* pt;

    for (i=0; i < NSTEPS; i++) {

        a = alpha + Spiral[i].AdvX;
        t = theta + Spiral[i].AdvY;

        // Cycle at the end
        a %= SECTORS;
        t %= SECTORS;

        // Cycle at the begin
        if (a < 0) a = SECTORS + a;
        if (t < 0) t = SECTORS + t;

        pt = &gbd ->Gamut[t][a];

        if (pt -> Type != GP_EMPTY) {

            Close[nSectors++] = pt;
        }
    }

    return nSectors;
}


// Interpolate a missing sector. Method identifies whatever this is top, bottom or mid
static
cmsBool InterpolateMissingSector(cmsGDB* gbd, int alpha, int theta)
{
    cmsSpherical sp;
    cmsVEC3 Lab;
    cmsVEC3 Centre;
    cmsLine ray;
    int nCloseSectors;
    cmsGDBPoint* Close[NSTEPS + 1];
    cmsSpherical closel, templ;
    cmsLine edge;
    int k, m;

    // Is that point already specified?
    if (gbd ->Gamut[theta][alpha].Type != GP_EMPTY) return TRUE;

    // Fill close points
    nCloseSectors = FindNearSectors(gbd, alpha, theta, Close);


    // Find a central point on the sector
    sp.alpha = (cmsFloat64Number) ((alpha + 0.5) * 360.0) / (SECTORS);
    sp.theta = (cmsFloat64Number) ((theta + 0.5) * 180.0) / (SECTORS);
    sp.r     = 50.0;

    // Convert to Cartesian
    ToCartesian(&Lab, &sp);

    // Create a ray line from centre to this point
    _cmsVEC3init(&Centre, 50.0, 0, 0);
    LineOf2Points(&ray, &Lab, &Centre);

    // For all close sectors
    closel.r = 0.0;
    closel.alpha = 0;
    closel.theta = 0;

    for (k=0; k < nCloseSectors; k++) {

        for(m = k+1; m < nCloseSectors; m++) {

            cmsVEC3 temp, a1, a2;

            // A line from sector to sector
            ToCartesian(&a1, &Close[k]->p);
            ToCartesian(&a2, &Close[m]->p);

            LineOf2Points(&edge, &a1, &a2);

            // Find a line
            ClosestLineToLine(&temp, &ray, &edge);

            // Convert to spherical
            ToSpherical(&templ, &temp);


            if ( templ.r > closel.r &&
                 templ.theta >= (theta*180.0/SECTORS) &&
                 templ.theta <= ((theta+1)*180.0/SECTORS) &&
                 templ.alpha >= (alpha*360.0/SECTORS) &&
                 templ.alpha <= ((alpha+1)*360.0/SECTORS)) {

                closel = templ;
            }
        }
    }

    gbd ->Gamut[theta][alpha].p = closel;
    gbd ->Gamut[theta][alpha].Type = GP_MODELED;

    return TRUE;

}


// Interpolate missing parts. The algorithm fist computes slices at
// theta=0 and theta=Max.
cmsBool CMSEXPORT cmsGDBCompute(cmsHANDLE hGBD, cmsUInt32Number dwFlags)
{
    int alpha, theta;
    cmsGDB* gbd = (cmsGDB*) hGBD;

    _cmsAssert(hGBD != NULL);

    // Interpolate black
    for (alpha = 0; alpha < SECTORS; alpha++) {

        if (!InterpolateMissingSector(gbd, alpha, 0)) return FALSE;
    }

    // Interpolate white
    for (alpha = 0; alpha < SECTORS; alpha++) {

        if (!InterpolateMissingSector(gbd, alpha, SECTORS-1)) return FALSE;
    }


    // Interpolate Mid
    for (theta = 1; theta < SECTORS; theta++) {
        for (alpha = 0; alpha < SECTORS; alpha++) {

            if (!InterpolateMissingSector(gbd, alpha, theta)) return FALSE;
        }
    }

    // Done
    return TRUE;

    cmsUNUSED_PARAMETER(dwFlags);
}




// --------------------------------------------------------------------------------------------------------

// Great for debug, but not suitable for real use

#if 0
cmsBool cmsGBDdumpVRML(cmsHANDLE hGBD, const char* fname)
{
    FILE* fp;
    int   i, j;
    cmsGDB* gbd = (cmsGDB*) hGBD;
    cmsGDBPoint* pt;

    fp = fopen (fname, "wt");
    if (fp == NULL)
        return FALSE;

    fprintf (fp, "#VRML V2.0 utf8\n");

    // set the viewing orientation and distance
    fprintf (fp, "DEF CamTest Group {\n");
    fprintf (fp, "\tchildren [\n");
    fprintf (fp, "\t\tDEF Cameras Group {\n");
    fprintf (fp, "\t\t\tchildren [\n");
    fprintf (fp, "\t\t\t\tDEF DefaultView Viewpoint {\n");
    fprintf (fp, "\t\t\t\t\tposition 0 0 340\n");
    fprintf (fp, "\t\t\t\t\torientation 0 0 1 0\n");
    fprintf (fp, "\t\t\t\t\tdescription \"default view\"\n");
    fprintf (fp, "\t\t\t\t}\n");
    fprintf (fp, "\t\t\t]\n");
    fprintf (fp, "\t\t},\n");
    fprintf (fp, "\t]\n");
    fprintf (fp, "}\n");

    // Output the background stuff
    fprintf (fp, "Background {\n");
    fprintf (fp, "\tskyColor [\n");
    fprintf (fp, "\t\t.5 .5 .5\n");
    fprintf (fp, "\t]\n");
    fprintf (fp, "}\n");

    // Output the shape stuff
    fprintf (fp, "Transform {\n");
    fprintf (fp, "\tscale .3 .3 .3\n");
    fprintf (fp, "\tchildren [\n");

    // Draw the axes as a shape:
    fprintf (fp, "\t\tShape {\n");
    fprintf (fp, "\t\t\tappearance Appearance {\n");
    fprintf (fp, "\t\t\t\tmaterial Material {\n");
    fprintf (fp, "\t\t\t\t\tdiffuseColor 0 0.8 0\n");
    fprintf (fp, "\t\t\t\t\temissiveColor 1.0 1.0 1.0\n");
    fprintf (fp, "\t\t\t\t\tshininess 0.8\n");
    fprintf (fp, "\t\t\t\t}\n");
    fprintf (fp, "\t\t\t}\n");
    fprintf (fp, "\t\t\tgeometry IndexedLineSet {\n");
    fprintf (fp, "\t\t\t\tcoord Coordinate {\n");
    fprintf (fp, "\t\t\t\t\tpoint [\n");
    fprintf (fp, "\t\t\t\t\t0.0 0.0 0.0,\n");
    fprintf (fp, "\t\t\t\t\t%f 0.0 0.0,\n",  255.0);
    fprintf (fp, "\t\t\t\t\t0.0 %f 0.0,\n",  255.0);
    fprintf (fp, "\t\t\t\t\t0.0 0.0 %f]\n",  255.0);
    fprintf (fp, "\t\t\t\t}\n");
    fprintf (fp, "\t\t\t\tcoordIndex [\n");
    fprintf (fp, "\t\t\t\t\t0, 1, -1\n");
    fprintf (fp, "\t\t\t\t\t0, 2, -1\n");
    fprintf (fp, "\t\t\t\t\t0, 3, -1]\n");
    fprintf (fp, "\t\t\t}\n");
    fprintf (fp, "\t\t}\n");


    fprintf (fp, "\t\tShape {\n");
    fprintf (fp, "\t\t\tappearance Appearance {\n");
    fprintf (fp, "\t\t\t\tmaterial Material {\n");
    fprintf (fp, "\t\t\t\t\tdiffuseColor 0 0.8 0\n");
    fprintf (fp, "\t\t\t\t\temissiveColor 1 1 1\n");
    fprintf (fp, "\t\t\t\t\tshininess 0.8\n");
    fprintf (fp, "\t\t\t\t}\n");
    fprintf (fp, "\t\t\t}\n");
    fprintf (fp, "\t\t\tgeometry PointSet {\n");

    // fill in the points here
    fprintf (fp, "\t\t\t\tcoord Coordinate {\n");
    fprintf (fp, "\t\t\t\t\tpoint [\n");

    // We need to transverse all gamut hull.
    for (i=0; i < SECTORS; i++)
        for (j=0; j < SECTORS; j++) {

            cmsVEC3 v;

            pt = &gbd ->Gamut[i][j];
            ToCartesian(&v, &pt ->p);

            fprintf (fp, "\t\t\t\t\t%g %g %g", v.n[0]+50, v.n[1], v.n[2]);

            if ((j == SECTORS - 1) && (i == SECTORS - 1))
                fprintf (fp, "]\n");
            else
                fprintf (fp, ",\n");

        }

        fprintf (fp, "\t\t\t\t}\n");



    // fill in the face colors
    fprintf (fp, "\t\t\t\tcolor Color {\n");
    fprintf (fp, "\t\t\t\t\tcolor [\n");

    for (i=0; i < SECTORS; i++)
        for (j=0; j < SECTORS; j++) {

           cmsVEC3 v;

            pt = &gbd ->Gamut[i][j];


            ToCartesian(&v, &pt ->p);


        if (pt ->Type == GP_EMPTY)
            fprintf (fp, "\t\t\t\t\t%g %g %g", 0.0, 0.0, 0.0);
        else
            if (pt ->Type == GP_MODELED)
                fprintf (fp, "\t\t\t\t\t%g %g %g", 1.0, .5, .5);
            else {
                fprintf (fp, "\t\t\t\t\t%g %g %g", 1.0, 1.0, 1.0);

            }

        if ((j == SECTORS - 1) && (i == SECTORS - 1))
                fprintf (fp, "]\n");
            else
                fprintf (fp, ",\n");
    }
    fprintf (fp, "\t\t\t}\n");


    fprintf (fp, "\t\t\t}\n");
    fprintf (fp, "\t\t}\n");
    fprintf (fp, "\t]\n");
    fprintf (fp, "}\n");

    fclose (fp);

    return TRUE;
}
#endif


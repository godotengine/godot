// This code is in the public domain -- castano@gmail.com

#include "nvmesh.h" // pch

#include "AtlasPacker.h"
#include "nvmesh/halfedge/Vertex.h"
#include "nvmesh/halfedge/Face.h"
#include "nvmesh/param/Atlas.h"
#include "nvmesh/param/Util.h"
#include "nvmesh/raster/Raster.h"

#include "nvmath/Vector.inl"
#include "nvmath/ConvexHull.h"
#include "nvmath/Color.h"
#include "nvmath/ftoi.h"

#include "nvcore/StrLib.h" // debug
#include "nvcore/StdStream.h" // fileOpen

#include <float.h> // FLT_MAX
#include <limits.h> // UINT_MAX

using namespace nv;

#define DEBUG_OUTPUT 0

#if DEBUG_OUTPUT

#include "nvimage/ImageIO.h"

namespace
{
    const uint TGA_TYPE_GREY = 3;
    const uint TGA_TYPE_RGB = 2;
    const uint TGA_ORIGIN_UPPER = 0x20;

#pragma pack(push, 1)
    struct TgaHeader {
	    uint8	id_length;
	    uint8	colormap_type;
	    uint8	image_type;
	    uint16	colormap_index;
	    uint16	colormap_length;
	    uint8	colormap_size;
	    uint16	x_origin;
	    uint16	y_origin;
	    uint16	width;
	    uint16	height;
	    uint8	pixel_size;
	    uint8	flags;

	    enum { Size = 18 };		//const static int SIZE = 18;
    };
#pragma pack(pop)

    static void outputDebugBitmap(const char * fileName, const BitMap & bitmap, int w, int h)
    {
        FILE * fp = fileOpen(fileName, "wb");
        if (fp == NULL) return;
        
        nvStaticCheck(sizeof(TgaHeader) == TgaHeader::Size);
	    TgaHeader tga;
	    tga.id_length = 0;
	    tga.colormap_type = 0;
	    tga.image_type = TGA_TYPE_GREY;

	    tga.colormap_index = 0;
	    tga.colormap_length = 0;
	    tga.colormap_size = 0;

	    tga.x_origin = 0;
	    tga.y_origin = 0;
        tga.width = w;
        tga.height = h;
	    tga.pixel_size = 8;
	    tga.flags = TGA_ORIGIN_UPPER;

        fwrite(&tga, sizeof(TgaHeader), 1, fp);

		for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                uint8 color = bitmap.bitAt(i, j) ? 0xFF : 0x0;
                fwrite(&color, 1, 1, fp);
            }
		}

        fclose(fp);
    }

    static void outputDebugImage(const char * fileName, const Image & bitmap, int w, int h)
    {
        FILE * fp = fileOpen(fileName, "wb");
        if (fp == NULL) return;
        
        nvStaticCheck(sizeof(TgaHeader) == TgaHeader::Size);
	    TgaHeader tga;
	    tga.id_length = 0;
	    tga.colormap_type = 0;
        tga.image_type = TGA_TYPE_RGB;

	    tga.colormap_index = 0;
	    tga.colormap_length = 0;
	    tga.colormap_size = 0;

	    tga.x_origin = 0;
	    tga.y_origin = 0;
        tga.width = w;
        tga.height = h;
	    tga.pixel_size = 24;
	    tga.flags = TGA_ORIGIN_UPPER;

        fwrite(&tga, sizeof(TgaHeader), 1, fp);

		for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                Color32 color = bitmap.pixel(i, j);
                fwrite(&color.r, 1, 1, fp);
                fwrite(&color.g, 1, 1, fp);
                fwrite(&color.b, 1, 1, fp);
            }
		}

        fclose(fp);
    }
}

#endif // DEBUG_OUTPUT

inline int align(int x, int a) {
    //return a * ((x + a - 1) / a);
    //return (x + a - 1) & -a;
    return (x + a - 1) & ~(a - 1);
}

inline bool isAligned(int x, int a) {
    return (x & (a - 1)) == 0;
}



AtlasPacker::AtlasPacker(Atlas * atlas) : m_atlas(atlas), m_bitmap(256, 256)
{
    m_width = 0;
    m_height = 0;

    // -- GODOT start --
    //m_debug_bitmap.allocate(256, 256);
    //m_debug_bitmap.fill(Color32(0,0,0,0));
    // -- GODOT end --
}

AtlasPacker::~AtlasPacker()
{
}

// This should compute convex hull and use rotating calipers to find the best box. Currently it uses a brute force method.
static bool computeBoundingBox(Chart * chart, Vector2 * majorAxis, Vector2 * minorAxis, Vector2 * minCorner, Vector2 * maxCorner)
{
    // Compute list of boundary points.
    Array<Vector2> points(16);

    HalfEdge::Mesh * mesh = chart->chartMesh();
    const uint vertexCount = mesh->vertexCount();

    for (uint i = 0; i < vertexCount; i++) {
        HalfEdge::Vertex * vertex = mesh->vertexAt(i);
        if (vertex->isBoundary()) {
            points.append(vertex->tex);
        }
    }

    // This is not valid anymore. The chart mesh may have multiple boundaries!
    /*const HalfEdge::Vertex * vertex = findBoundaryVertex(chart->chartMesh());

    // Traverse boundary.
    const HalfEdge::Edge * const firstEdge = vertex->edge();
    const HalfEdge::Edge * edge = firstEdge;
    do {
        vertex = edge->vertex();

        nvDebugCheck (vertex->isBoundary());
        points.append(vertex->tex);

        edge = edge->next();
    } while (edge != firstEdge);*/

#if 1
    Array<Vector2> hull;
    if (points.size()==0) {
        return false;    
    }
    
    convexHull(points, hull, 0.00001f);

    // @@ Ideally I should use rotating calipers to find the best box. Using brute force for now.

    float best_area = FLT_MAX;
    Vector2 best_min;
    Vector2 best_max;
    Vector2 best_axis;

    const uint hullCount = hull.count();
    for (uint i = 0, j = hullCount-1; i < hullCount; j = i, i++) {

        if (equal(hull[i], hull[j])) {
            continue;
        }

        Vector2 axis = normalize(hull[i] - hull[j], 0.0f);
        nvDebugCheck(isFinite(axis));

        // Compute bounding box.
        Vector2 box_min(FLT_MAX, FLT_MAX);
        Vector2 box_max(-FLT_MAX, -FLT_MAX);

        for (uint v = 0; v < hullCount; v++) {

           Vector2 point = hull[v];

           float x = dot(axis, point);
           if (x < box_min.x) box_min.x = x;
           if (x > box_max.x) box_max.x = x;

           float y = dot(Vector2(-axis.y, axis.x), point);
           if (y < box_min.y) box_min.y = y;
           if (y > box_max.y) box_max.y = y;
        }
    
        // Compute box area.
        float area = (box_max.x - box_min.x) * (box_max.y - box_min.y);

        if (area < best_area) {
            best_area = area;
            best_min = box_min;
            best_max = box_max;
            best_axis = axis;
        }
    }

    // Make sure the box contains all the input points since the convex hull is not 100% accurate.
    /*const uint pointCount = points.count();
    for (uint v = 0; v < pointCount; v++) {

        Vector2 point = points[v];

        float x = dot(best_axis, point);
        if (x < best_min.x) best_min.x = x;

        float y = dot(Vector2(-best_axis.y, best_axis.x), point);
        if (y < best_min.y) best_min.y = y;
    }*/

    // Consider all points, not only boundary points, in case the input chart is malformed.
    for (uint i = 0; i < vertexCount; i++) {
        HalfEdge::Vertex * vertex = mesh->vertexAt(i);
        Vector2 point = vertex->tex;

        float x = dot(best_axis, point);
        if (x < best_min.x) best_min.x = x;
        if (x > best_max.x) best_max.x = x;

        float y = dot(Vector2(-best_axis.y, best_axis.x), point);
        if (y < best_min.y) best_min.y = y;
        if (y > best_max.y) best_max.y = y;
    }

    *majorAxis = best_axis;
    *minorAxis = Vector2(-best_axis.y, best_axis.x);
    *minCorner = best_min;
    *maxCorner = best_max;

#else
    // Approximate implementation: try 16 different directions and keep the best.

    const uint N = 16;
    Vector2 axis[N];

    float minAngle = 0;
    float maxAngle = PI / 2;

    int best;
    Vector2 mins[N];
    Vector2 maxs[N];

    const int iterationCount = 1;
    for (int j = 0; j < iterationCount; j++)
    {
        // Init predefined directions.
        for (int i = 0; i < N; i++)
        {
            float angle = lerp(minAngle, maxAngle, float(i)/N);
            axis[i].set(cosf(angle), sinf(angle));
        }

        // Compute box for each direction.
        for (int i = 0; i < N; i++)
        {
            mins[i].set(FLT_MAX, FLT_MAX);
            maxs[i].set(-FLT_MAX, -FLT_MAX);
        }

        for (uint p = 0; p < points.count(); p++)
        {
            Vector2 point = points[p];

            for (int i = 0; i < N; i++)
            {
               float x = dot(axis[i], point);
               if (x < mins[i].x) mins[i].x = x;
               if (x > maxs[i].x) maxs[i].x = x;

               float y = dot(Vector2(-axis[i].y, axis[i].x), point);
               if (y < mins[i].y) mins[i].y = y;
               if (y > maxs[i].y) maxs[i].y = y;
            }
        }

        // Find box with minimum area.
        best = -1;
        int second_best = -1;
        float best_area = FLT_MAX;
        float second_best_area = FLT_MAX;
        
        for (int i = 0; i < N; i++)
        {
            float area = (maxs[i].x - mins[i].x) * (maxs[i].y - mins[i].y);

            if (area < best_area)
            {
                second_best_area = best_area;
                second_best = best;

                best_area = area;
                best = i;
            }
            else if (area < second_best_area)
            {
                second_best_area = area;
                second_best = i;
            }
        }
        nvDebugCheck(best != -1);
        nvDebugCheck(second_best != -1);
        nvDebugCheck(best != second_best);

        if (j != iterationCount-1)
        {
            // Handle wrap-around during the first iteration.
            if (j == 0) {
                if (best == 0 && second_best == N-1) best = N;
                if (best == N-1 && second_best == 0) second_best = N;
            }

            if (best < second_best) swap(best, second_best);

            // Update angles.
            float deltaAngle = (maxAngle - minAngle) / N;
            maxAngle = minAngle + (best - 0.5f)  * deltaAngle;
            minAngle = minAngle + (second_best + 0.5f) * deltaAngle;
        }
    }

    // Compute major and minor axis, and origin.
    *majorAxis = axis[best];
    *minorAxis = Vector2(-axis[best].y, axis[best].x);
    *origin = mins[best];

    // @@ If the parameterization is invalid, we could have an interior vertex outside the boundary.
    // @@ In that case the returned bounding box would be incorrect. Compute updated bounds here.
    /*for (uint p = 0; p < points.count(); p++)
    {
        Vector2 point = points[p];

        for (int i = 0; i < N; i++)
        {
           float x = dot(*majorAxis, point);
           float y = dot(*minorAxis, point);
        }
    }*/
#endif

    return true;
}


void AtlasPacker::packCharts(int quality, float texelsPerUnit, bool blockAligned, bool conservative)
{
    const uint chartCount = m_atlas->chartCount();
    if (chartCount == 0) return;

    Array<float> chartOrderArray;
    chartOrderArray.resize(chartCount);

    Array<Vector2> chartExtents;
    chartExtents.resize(chartCount);
    
    float meshArea = 0;
    for (uint c = 0; c < chartCount; c++)
    {
        Chart * chart = m_atlas->chartAt(c);
        
        if (!chart->isVertexMapped() && !chart->isDisk()) {
            chartOrderArray[c] = 0;

            // Skip non-disks.
            continue;
        }

        Vector2 extents(0.0f);

        if (chart->isVertexMapped()) {
            // Let's assume vertex maps are arranged in a rectangle.
            //HalfEdge::Mesh * mesh = chart->chartMesh();

            // Arrange vertices in a rectangle.
            extents.x = float(chart->vertexMapWidth);
            extents.y = float(chart->vertexMapHeight);
        }
        else {
            // Compute surface area to sort charts.
            float chartArea = chart->computeSurfaceArea();
            meshArea += chartArea;
            //chartOrderArray[c] = chartArea;

            // Compute chart scale
            float parametricArea = fabs(chart->computeParametricArea());    // @@ There doesn't seem to be anything preventing parametric area to be negative.
            if (parametricArea < NV_EPSILON) {
                // When the parametric area is too small we use a rough approximation to prevent divisions by very small numbers.
                Vector2 bounds = chart->computeParametricBounds();
                parametricArea = bounds.x * bounds.y;
            }
            float scale = (chartArea / parametricArea) * texelsPerUnit;
            if (parametricArea == 0) // < NV_EPSILON)
            {
                scale = 0;
            }
            nvCheck(isFinite(scale));

            // Compute bounding box of chart.
            Vector2 majorAxis, minorAxis, origin, end;
            if (!computeBoundingBox(chart, &majorAxis, &minorAxis, &origin, &end)) {
                m_atlas->setFailed();
                return;
            }

            nvCheck(isFinite(majorAxis) && isFinite(minorAxis) && isFinite(origin));
            
            // Sort charts by perimeter. @@ This is sometimes producing somewhat unexpected results. Is this right?
            //chartOrderArray[c] = ((end.x - origin.x) + (end.y - origin.y)) * scale;

            // Translate, rotate and scale vertices. Compute extents.
            HalfEdge::Mesh * mesh = chart->chartMesh();
            const uint vertexCount = mesh->vertexCount();
            for (uint i = 0; i < vertexCount; i++)
            {
                HalfEdge::Vertex * vertex = mesh->vertexAt(i);

                //Vector2 t = vertex->tex - origin;
                Vector2 tmp;
                tmp.x = dot(vertex->tex, majorAxis);
                tmp.y = dot(vertex->tex, minorAxis);
                tmp -= origin;
                tmp *= scale;
                if (tmp.x < 0 || tmp.y < 0) {
                    nvDebug("tmp: %f %f\n", tmp.x, tmp.y);
                    nvDebug("scale: %f\n", scale);
                    nvDebug("origin: %f %f\n", origin.x, origin.y);
                    nvDebug("majorAxis: %f %f\n", majorAxis.x, majorAxis.y);
                    nvDebug("minorAxis: %f %f\n", minorAxis.x, minorAxis.y);
                    nvDebugBreak();
                }
                //nvCheck(tmp.x >= 0 && tmp.y >= 0);

                vertex->tex = tmp;

				nvCheck(isFinite(vertex->tex.x) && isFinite(vertex->tex.y));

                extents = max(extents, tmp);
            }
            nvDebugCheck(extents.x >= 0 && extents.y >= 0);

            // Limit chart size.
            if (extents.x > 1024 || extents.y > 1024) {
                float limit = max(extents.x, extents.y);

                scale = 1024 / (limit + 1);

                for (uint i = 0; i < vertexCount; i++)
                {
                    HalfEdge::Vertex * vertex = mesh->vertexAt(i);
                    vertex->tex *= scale;
                }

                extents *= scale;

                nvDebugCheck(extents.x <= 1024 && extents.y <= 1024);
            }


            // Scale the charts to use the entire texel area available. So, if the width is 0.1 we could scale it to 1 without increasing the lightmap usage and making a better 
            // use of it. In many cases this also improves the look of the seams, since vertices on the chart boundaries have more chances of being aligned with the texel centers.

            float scale_x = 1.0f;
            float scale_y = 1.0f;

            float divide_x = 1.0f;
            float divide_y = 1.0f;

            if (extents.x > 0) {
                int cw = ftoi_ceil(extents.x);

                if (blockAligned) {
                    // Align all chart extents to 4x4 blocks, but taking padding into account.
                    if (conservative) {
                        cw = align(cw + 2, 4) - 2;
                    }
                    else {
                        cw = align(cw + 1, 4) - 1;
                    }
                }

                scale_x = (float(cw) - NV_EPSILON);
                divide_x = extents.x;
                extents.x = float(cw);
            }

            if (extents.y > 0) {
                int ch = ftoi_ceil(extents.y);

                if (blockAligned) {
                    // Align all chart extents to 4x4 blocks, but taking padding into account.
                    if (conservative) {
                        ch = align(ch + 2, 4) - 2;
                    }
                    else {
                        ch = align(ch + 1, 4) - 1;
                    }
                }

                scale_y = (float(ch) - NV_EPSILON);
                divide_y = extents.y;
                extents.y = float(ch);
            }

            for (uint v = 0; v < vertexCount; v++) {
                HalfEdge::Vertex * vertex = mesh->vertexAt(v);

                vertex->tex.x /= divide_x;
                vertex->tex.y /= divide_y;
                vertex->tex.x *= scale_x;
                vertex->tex.y *= scale_y;

				nvCheck(isFinite(vertex->tex.x) && isFinite(vertex->tex.y));
            }
        }

        chartExtents[c] = extents;

        // Sort charts by perimeter.
        chartOrderArray[c] = extents.x + extents.y;
    }

    // @@ We can try to improve compression of small charts by sorting them by proximity like we do with vertex samples.
    // @@ How to do that? One idea: compute chart centroid, insert into grid, compute morton index of the cell, sort based on morton index.
    // @@ We would sort by morton index, first, then quantize the chart sizes, so that all small charts have the same size, and sort by size preserving the morton order.

    //nvDebug("Sorting charts.\n");

    // Sort charts by area.
    m_radix.sort(chartOrderArray);
    const uint32 * ranks = m_radix.ranks();

    // Estimate size of the map based on the mesh surface area and given texel scale.
    float texelCount = meshArea * square(texelsPerUnit) / 0.75f; // Assume 75% utilization.
    if (texelCount < 1) texelCount = 1;
    uint approximateExtent = nextPowerOfTwo(uint(sqrtf(texelCount)));

    //nvDebug("Init bitmap.\n");

    // @@ Pack all charts smaller than a texel into a compact rectangle.
    // @@ Start considering only 1x1 charts. Extend to 1xn charts later.

    /*for (uint i = 0; i < chartCount; i++)
    {
        uint c = ranks[chartCount - i - 1]; // largest chart first

        Chart * chart = m_atlas->chartAt(c);

        if (!chart->isDisk()) continue;

        if (iceil(chartExtents[c].x) == 1 && iceil(chartExtents[c].x) == 1) {
            // @@ Add to 
        }
    }*/



    // Init bit map.
    m_bitmap.clearAll();
    if (approximateExtent > m_bitmap.width()) {
        m_bitmap.resize(approximateExtent, approximateExtent, false);
        // -- GODOT start --
        //m_debug_bitmap.resize(approximateExtent, approximateExtent);
        //m_debug_bitmap.fill(Color32(0,0,0,0));
        // -- GODOT end --
    }

    
    int w = 0;
    int h = 0;

#if 1
    // Add sorted charts to bitmap.
    for (uint i = 0; i < chartCount; i++)
    {
        uint c = ranks[chartCount - i - 1]; // largest chart first

        Chart * chart = m_atlas->chartAt(c);

        if (!chart->isVertexMapped() && !chart->isDisk()) continue;

        //float scale_x = 1;
        //float scale_y = 1;

        BitMap chart_bitmap;

        if (chart->isVertexMapped()) {
            // Init all bits to 1.
            chart_bitmap.resize(ftoi_ceil(chartExtents[c].x), ftoi_ceil(chartExtents[c].y), /*initValue=*/true);

            // @@ Another alternative would be to try to map each vertex to a different texel trying to fill all the available unused texels.
        }
        else {
            // @@ Add special cases for dot and line charts. @@ Lightmap rasterizer also needs to handle these special cases.
            // @@ We could also have a special case for chart quads. If the quad surface <= 4 texels, align vertices with texel centers and do not add padding. May be very useful for foliage.

            // @@ In general we could reduce the padding of all charts by one texel by using a rasterizer that takes into account the 2-texel footprint of the tent bilinear filter. For example,
            // if we have a chart that is less than 1 texel wide currently we add one texel to the left and one texel to the right creating a 3-texel-wide bitmap. However, if we know that the 
            // chart is only 1 texel wide we could align it so that it only touches the footprint of two texels:

            //      |   |      <- Touches texels 0, 1 and 2.
            //    |   |        <- Only touches texels 0 and 1.
            // \   \ / \ /   /
            //  \   X   X   /
            //   \ / \ / \ /
            //    V   V   V
            //    0   1   2

            if (conservative) {
                // Init all bits to 0.
                chart_bitmap.resize(ftoi_ceil(chartExtents[c].x) + 2, ftoi_ceil(chartExtents[c].y) + 2, /*initValue=*/false);  // + 2 to add padding on both sides.

                // Rasterize chart and dilate.
                drawChartBitmapDilate(chart, &chart_bitmap, /*padding=*/1);
            }
            else {
                // Init all bits to 0.
                chart_bitmap.resize(ftoi_ceil(chartExtents[c].x) + 1, ftoi_ceil(chartExtents[c].y) + 1, /*initValue=*/false);  // Add half a texels on each side.

                // Rasterize chart and dilate.
                drawChartBitmap(chart, &chart_bitmap, Vector2(1), Vector2(0.5));
            }
        }

        int best_x, best_y;
        int best_cw, best_ch;   // Includes padding now.
        int best_r;
        findChartLocation(quality, &chart_bitmap, chartExtents[c], w, h, &best_x, &best_y, &best_cw, &best_ch, &best_r);
        
        /*if (w < best_x + best_cw || h < best_y + best_ch)
        {
            nvDebug("Resize extents to (%d, %d).\n", best_x + best_cw, best_y + best_ch);
        }*/

        // Update parametric extents.
        w = max(w, best_x + best_cw);
        h = max(h, best_y + best_ch);
        
        w = align(w, 4);
        h = align(h, 4);

        // Resize bitmap if necessary.
        if (uint(w) > m_bitmap.width() || uint(h) > m_bitmap.height())
        {
            //nvDebug("Resize bitmap (%d, %d).\n", nextPowerOfTwo(w), nextPowerOfTwo(h));
            m_bitmap.resize(nextPowerOfTwo(U32(w)), nextPowerOfTwo(U32(h)), false);
            // -- GODOT start --
            //m_debug_bitmap.resize(nextPowerOfTwo(U32(w)), nextPowerOfTwo(U32(h)));
            // -- GODOT end --
        }

        //nvDebug("Add chart at (%d, %d).\n", best_x, best_y);

        addChart(&chart_bitmap, w, h, best_x, best_y, best_r, /*debugOutput=*/NULL);

        // -- GODOT start --
        // IC: Output chart again to debug bitmap.
        /*if (chart->isVertexMapped()) {
            addChart(&chart_bitmap, w, h, best_x, best_y, best_r, &m_debug_bitmap);
        }
        else {
            addChart(chart, w, h, best_x, best_y, best_r, &m_debug_bitmap);
        }*/
        // -- GODOT end --

        //float best_angle = 2 * PI * best_r;

        // Translate and rotate chart texture coordinates.
        HalfEdge::Mesh * mesh = chart->chartMesh();
        const uint vertexCount = mesh->vertexCount();
        for (uint v = 0; v < vertexCount; v++)
        {
            HalfEdge::Vertex * vertex = mesh->vertexAt(v);

            Vector2 t = vertex->tex;
            if (best_r) swap(t.x, t.y);
            //vertex->tex.x = best_x + t.x * cosf(best_angle) - t.y * sinf(best_angle);
            //vertex->tex.y = best_y + t.x * sinf(best_angle) + t.y * cosf(best_angle);

            vertex->tex.x = best_x + t.x + 0.5f;
            vertex->tex.y = best_y + t.y + 0.5f;

            nvCheck(vertex->tex.x >= 0 && vertex->tex.y >= 0);
			nvCheck(isFinite(vertex->tex.x) && isFinite(vertex->tex.y));
        }

#if DEBUG_OUTPUT && 0
        StringBuilder fileName;
        fileName.format("debug_packer_%d.tga", i);
        //outputDebugBitmap(fileName.str(), m_bitmap, w, h);
        outputDebugImage(fileName.str(), m_debug_bitmap, w, h);
#endif
    }

#else // 0

    // Add sorted charts to bitmap.
    for (uint i = 0; i < chartCount; i++)
    {
        uint c = ranks[chartCount - i - 1]; // largest chart first

        Chart * chart = m_atlas->chartAt(c);

        if (!chart->isDisk()) continue;

        Vector2 scale(1, 1);

#if 0 // old method.
        //m_padding_x = 2*padding;
        //m_padding_y = 2*padding;
#else
        //m_padding_x = 0; //padding;
        //m_padding_y = 0; //padding;
#endif

        int bw = ftoi_ceil(chartExtents[c].x + 1);
        int bh = ftoi_ceil(chartExtents[c].y + 1);

        if (chartExtents[c].x < 1.0f) {
            scale.x = 0.01f; // @@ Ideally we would like to scale it to 0, but then our rasterizer would not touch any pixels.
            bw = 1;
        }
        if (chartExtents[c].y < 1.0f) {
            scale.y = 0.01f;
            bh = 1;
        }

        //BitMap chart_bitmap(iceil(chartExtents[c].x) + 1 + m_padding_x * 2, iceil(chartExtents[c].y) + 1 + m_padding_y * 2);
        //BitMap chart_bitmap(ftoi_ceil(chartExtents[c].x/2)*2, ftoi_ceil(chartExtents[c].y/2)*2);
        BitMap chart_bitmap(bw, bh);
        chart_bitmap.clearAll();
        
        Vector2 offset;
        offset.x = 0; // (chart_bitmap.width() - chartExtents[c].x) * 0.5f;
        offset.y = 0; // (chart_bitmap.height() - chartExtents[c].y) * 0.5f;

        drawChartBitmap(chart, &chart_bitmap, scale, offset);

        int best_x, best_y;
        int best_cw, best_ch;
        int best_r;
        findChartLocation(quality, &chart_bitmap, chartExtents[c], w, h, &best_x, &best_y, &best_cw, &best_ch, &best_r);

        /*if (w < best_x + best_cw || h < best_y + best_ch)
        {
            nvDebug("Resize extents to (%d, %d).\n", best_x + best_cw, best_y + best_ch);
        }*/

        // Update parametric extents.
        w = max(w, best_x + best_cw);
        h = max(h, best_y + best_ch);

        // Resize bitmap if necessary.
        if (uint(w) > m_bitmap.width() || uint(h) > m_bitmap.height())
        {
            //nvDebug("Resize bitmap (%d, %d).\n", nextPowerOfTwo(w), nextPowerOfTwo(h));
            m_bitmap.resize(nextPowerOfTwo(w), nextPowerOfTwo(h), false);
            m_debug_bitmap.resize(nextPowerOfTwo(w), nextPowerOfTwo(h));
        }

        //nvDebug("Add chart at (%d, %d).\n", best_x, best_y);

#if 0 // old method.
#if _DEBUG
        checkCanAddChart(chart, w, h, best_x, best_y, best_r);
#endif

        // Add chart.
        addChart(chart, w, h, best_x, best_y, best_r);
#else
        // Add chart reusing its bitmap.
        addChart(&chart_bitmap, w, h, best_x, best_y, best_r);
#endif

        //float best_angle = 2 * PI * best_r;

        // Translate and rotate chart texture coordinates.
        HalfEdge::Mesh * mesh = chart->chartMesh();
        const uint vertexCount = mesh->vertexCount();
        for (uint v = 0; v < vertexCount; v++)
        {
            HalfEdge::Vertex * vertex = mesh->vertexAt(v);

            Vector2 t = vertex->tex * scale + offset;
            if (best_r) swap(t.x, t.y);
            //vertex->tex.x = best_x + t.x * cosf(best_angle) - t.y * sinf(best_angle);
            //vertex->tex.y = best_y + t.x * sinf(best_angle) + t.y * cosf(best_angle);
            vertex->tex.x = best_x + t.x + 0.5f;
            vertex->tex.y = best_y + t.y + 0.5f;

            nvCheck(vertex->tex.x >= 0 && vertex->tex.y >= 0);
        }

#if DEBUG_OUTPUT && 0
        StringBuilder fileName;
        fileName.format("debug_packer_%d.tga", i);
        //outputDebugBitmap(fileName.str(), m_bitmap, w, h);
        outputDebugImage(fileName.str(), m_debug_bitmap, w, h);
#endif
    }

#endif // 0

    //w -= padding - 1; // Leave one pixel border!
    //h -= padding - 1;

    m_width = max(0, w);
    m_height = max(0, h);

    nvCheck(isAligned(m_width, 4));
    nvCheck(isAligned(m_height, 4));

    // -- GODOT start --
    //m_debug_bitmap.resize(m_width, m_height);
    //m_debug_bitmap.setFormat(Image::Format_ARGB);
    // -- GODOT end --

#if DEBUG_OUTPUT
    //outputDebugBitmap("debug_packer_final.tga", m_bitmap, w, h);
    //outputDebugImage("debug_packer_final.tga", m_debug_bitmap, w, h);
    ImageIO::save("debug_packer_final.tga", &m_debug_bitmap);
#endif
}


// IC: Brute force is slow, and random may take too much time to converge. We start inserting large charts in a small atlas. Using brute force is lame, because most of the space 
// is occupied at this point. At the end we have many small charts and a large atlas with sparse holes. Finding those holes randomly is slow. A better approach would be to 
// start stacking large charts as if they were tetris pieces. Once charts get small try to place them randomly. It may be interesting to try a intermediate strategy, first try 
// along one axis and then try exhaustively along that axis.
void AtlasPacker::findChartLocation(int quality, const BitMap * bitmap, Vector2::Arg extents, int w, int h, int * best_x, int * best_y, int * best_w, int * best_h, int * best_r)
{
    int attempts = 256;
    if (quality == 1) attempts = 4096;
    if (quality == 2) attempts = 2048;
    if (quality == 3) attempts = 1024;
    if (quality == 4) attempts = 512;

    if (quality == 0 || w*h < attempts)
    {
        findChartLocation_bruteForce(bitmap, extents, w, h, best_x, best_y, best_w, best_h, best_r);
    }
    else
    {
        findChartLocation_random(bitmap, extents, w, h, best_x, best_y, best_w, best_h, best_r, attempts);
    }
}

#define BLOCK_SIZE 4

void AtlasPacker::findChartLocation_bruteForce(const BitMap * bitmap, Vector2::Arg extents, int w, int h, int * best_x, int * best_y, int * best_w, int * best_h, int * best_r)
{
    int best_metric = INT_MAX;

    // Try two different orientations.
    for (int r = 0; r < 2; r++)
    {
        int cw = bitmap->width();
        int ch = bitmap->height();
        if (r & 1) swap(cw, ch);

        for (int y = 0; y <= h + 1; y += BLOCK_SIZE) // + 1 to extend atlas in case atlas full.
        {
            for (int x = 0; x <= w + 1; x += BLOCK_SIZE) // + 1 not really necessary here.
            {
                // Early out.
                int area = max(w, x+cw) * max(h, y+ch);
                //int perimeter = max(w, x+cw) + max(h, y+ch);
                int extents = max(max(w, x+cw), max(h, y+ch));

                int metric = extents*extents + area;

                if (metric > best_metric) {
                    continue;
                }
                if (metric == best_metric && max(x, y) >= max(*best_x, *best_y)) {
                    // If metric is the same, pick the one closest to the origin.
                    continue;
                }

                if (canAddChart(bitmap, w, h, x, y, r))
                {
                    best_metric = metric;
                    *best_x = x;
                    *best_y = y;
                    *best_w = cw;
                    *best_h = ch;
                    *best_r = r;

                    if (area == w*h)
                    {
                        // Chart is completely inside, do not look at any other location.
                        goto done;
                    }
                }
            }
        }
    }

done:
    nvDebugCheck (best_metric != INT_MAX);
}


void AtlasPacker::findChartLocation_random(const BitMap * bitmap, Vector2::Arg extents, int w, int h, int * best_x, int * best_y, int * best_w, int * best_h, int * best_r, int minTrialCount)
{
    int best_metric = INT_MAX;

    for (int i = 0; i < minTrialCount || best_metric == INT_MAX; i++)
    {
        int r = m_rand.getRange(1);
        int x = m_rand.getRange(w + 1); // + 1 to extend atlas in case atlas full. We may want to use a higher number to increase probability of extending atlas.
        int y = m_rand.getRange(h + 1); // + 1 to extend atlas in case atlas full.

        x = align(x, BLOCK_SIZE);
        y = align(y, BLOCK_SIZE);

        int cw = bitmap->width();
        int ch = bitmap->height();
        if (r & 1) swap(cw, ch);

        // Early out.
        int area = max(w, x+cw) * max(h, y+ch);
        //int perimeter = max(w, x+cw) + max(h, y+ch);
        int extents = max(max(w, x+cw), max(h, y+ch));

        int metric = extents*extents + area;

        if (metric > best_metric) {
            continue;
        }
        if (metric == best_metric && min(x, y) > min(*best_x, *best_y)) {
            // If metric is the same, pick the one closest to the origin.
            continue;
        }

        if (canAddChart(bitmap, w, h, x, y, r))
        {
            best_metric = metric;
            *best_x = x;
            *best_y = y;
            *best_w = cw;
            *best_h = ch;
            *best_r = r;

            if (area == w*h)
            {
                // Chart is completely inside, do not look at any other location.
                break;
            }
        }
    }
}


void AtlasPacker::drawChartBitmapDilate(const Chart * chart, BitMap * bitmap, int padding)
{
    const int w = bitmap->width();
    const int h = bitmap->height();
    const Vector2 extents = Vector2(float(w), float(h));
    
    // Rasterize chart faces, check that all bits are not set.
    const uint faceCount = chart->faceCount();
    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = chart->chartMesh()->faceAt(f);
        
        Vector2 vertices[4];

        uint edgeCount = 0;
        for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            if (edgeCount < 4)
            {
                vertices[edgeCount] = it.vertex()->tex + Vector2(0.5) + Vector2(float(padding), float(padding));
            }
            edgeCount++;
        }

        if (edgeCount == 3)
        {
            Raster::drawTriangle(Raster::Mode_Antialiased, extents, true, vertices, AtlasPacker::setBitsCallback, bitmap);
        }
        else
        {
            Raster::drawQuad(Raster::Mode_Antialiased, extents, true, vertices, AtlasPacker::setBitsCallback, bitmap);
        }
    }

    // Expand chart by padding pixels. (dilation)
    BitMap tmp(w, h);
    for (int i = 0; i < padding; i++) {
        tmp.clearAll();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                bool b = bitmap->bitAt(x, y);
                if (!b) {
                    if (x > 0) {
                        b |= bitmap->bitAt(x - 1, y);
                        if (y > 0) b |= bitmap->bitAt(x - 1, y - 1);
                        if (y < h-1) b |= bitmap->bitAt(x - 1, y + 1);
                    }
                    if (y > 0) b |= bitmap->bitAt(x, y - 1);
                    if (y < h-1) b |= bitmap->bitAt(x, y + 1);
                    if (x < w-1) {
                        b |= bitmap->bitAt(x + 1, y);
                        if (y > 0) b |= bitmap->bitAt(x + 1, y - 1);
                        if (y < h-1) b |= bitmap->bitAt(x + 1, y + 1);
                    }
                }
                if (b) tmp.setBitAt(x, y);
            }
        }

        swap(tmp, *bitmap);
    }
}


void AtlasPacker::drawChartBitmap(const Chart * chart, BitMap * bitmap, const Vector2 & scale, const Vector2 & offset)
{
    const int w = bitmap->width();
    const int h = bitmap->height();
    const Vector2 extents = Vector2(float(w), float(h));
    
    static const Vector2 pad[4] = {
        Vector2(-0.5, -0.5),
        Vector2(0.5, -0.5),
        Vector2(-0.5, 0.5),
        Vector2(0.5, 0.5)
    };
    /*static const Vector2 pad[4] = {
        Vector2(-1, -1),
        Vector2(1, -1),
        Vector2(-1, 1),
        Vector2(1, 1)
    };*/

    // Rasterize 4 times to add proper padding.
    for (int i = 0; i < 4; i++) {

        // Rasterize chart faces, check that all bits are not set.
        const uint faceCount = chart->chartMesh()->faceCount();
        for (uint f = 0; f < faceCount; f++)
        {
            const HalfEdge::Face * face = chart->chartMesh()->faceAt(f);
            
            Vector2 vertices[4];

            uint edgeCount = 0;
            for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
            {
                if (edgeCount < 4)
                {
                    vertices[edgeCount] = it.vertex()->tex * scale + offset + pad[i];
                    nvCheck(ftoi_ceil(vertices[edgeCount].x) >= 0);
                    nvCheck(ftoi_ceil(vertices[edgeCount].y) >= 0);
                    nvCheck(ftoi_ceil(vertices[edgeCount].x) <= w);
                    nvCheck(ftoi_ceil(vertices[edgeCount].y) <= h);
                }
                edgeCount++;
            }

            if (edgeCount == 3)
            {
                Raster::drawTriangle(Raster::Mode_Antialiased, extents, /*enableScissors=*/true, vertices, AtlasPacker::setBitsCallback, bitmap);
            }
            else
            {
                Raster::drawQuad(Raster::Mode_Antialiased, extents, /*enableScissors=*/true, vertices, AtlasPacker::setBitsCallback, bitmap);
            }
        }
    }

    // @@ This only allows us to expand the size in texel intervals.
    /*if (m_padding_x != 0 && m_padding_y != 0)*/ {

        // Expand chart by padding pixels. (dilation)
        BitMap tmp(w, h);
        //for (int i = 0; i < 1; i++) {
            tmp.clearAll();

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    bool b = bitmap->bitAt(x, y);
                    if (!b) {
                        if (x > 0) {
                            b |= bitmap->bitAt(x - 1, y);
                            if (y > 0) b |= bitmap->bitAt(x - 1, y - 1);
                            if (y < h-1) b |= bitmap->bitAt(x - 1, y + 1);
                        }
                        if (y > 0) b |= bitmap->bitAt(x, y - 1);
                        if (y < h-1) b |= bitmap->bitAt(x, y + 1);
                        if (x < w-1) {
                            b |= bitmap->bitAt(x + 1, y);
                            if (y > 0) b |= bitmap->bitAt(x + 1, y - 1);
                            if (y < h-1) b |= bitmap->bitAt(x + 1, y + 1);
                        }
                    }
                    if (b) tmp.setBitAt(x, y);
                }
            }

            swap(tmp, *bitmap);
        //}
    }
}

bool AtlasPacker::canAddChart(const BitMap * bitmap, int atlas_w, int atlas_h, int offset_x, int offset_y, int r)
{
    nvDebugCheck(r == 0 || r == 1);

    // Check whether the two bitmaps overlap.

    const int w = bitmap->width();
    const int h = bitmap->height();
    
    if (r == 0) {
        for (int y = 0; y < h; y++) {
            int yy = y + offset_y;
            if (yy >= 0) {
                for (int x = 0; x < w; x++) {
                    int xx = x + offset_x;
                    if (xx >= 0) {
                        if (bitmap->bitAt(x, y)) {
                            if (xx < atlas_w && yy < atlas_h) {
                                if (m_bitmap.bitAt(xx, yy)) return false;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (r == 1) {
        for (int y = 0; y < h; y++) {
            int xx = y + offset_x;
            if (xx >= 0) {
                for (int x = 0; x < w; x++) {
                    int yy = x + offset_y;
                    if (yy >= 0) {
                        if (bitmap->bitAt(x, y)) {
                            if (xx < atlas_w && yy < atlas_h) {
                                if (m_bitmap.bitAt(xx, yy)) return false;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return true;
}

#if 0
void AtlasPacker::checkCanAddChart(const Chart * chart, int w, int h, int x, int y, int r)
{
    nvDebugCheck(r == 0 || r == 1);
    Vector2 extents = Vector2(float(w), float(h));
    Vector2 offset = Vector2(float(x), float(y));

    // Rasterize chart faces, set bits.
    const uint faceCount = chart->faceCount();
    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = chart->chartMesh()->faceAt(f);
        
        Vector2 vertices[4];

        uint edgeCount = 0;
        for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            if (edgeCount < 4)
            {
                Vector2 t = it.vertex()->tex;
                if (r == 1) swap(t.x, t.y);
                vertices[edgeCount] = t + offset;
            }
            edgeCount++;
        }

        if (edgeCount == 3)
        {
            Raster::drawTriangle(Raster::Mode_Antialiased, extents, /*enableScissors=*/true, vertices, AtlasPacker::checkBitsCallback, &m_bitmap);
        }
        else
        {
            Raster::drawQuad(Raster::Mode_Antialiased, extents, /*enableScissors=*/true, vertices, AtlasPacker::checkBitsCallback, &m_bitmap);
        }
    }
}
#endif // 0


static Color32 chartColor = Color32(0);
static void selectRandomColor(MTRand & rand) {
    // Pick random color for this chart. @@ Select random hue, but fixed saturation/luminance?
    chartColor.r = 128 + rand.getRange(127);
    chartColor.g = 128 + rand.getRange(127);
    chartColor.b = 128 + rand.getRange(127);
    chartColor.a = 255;
}
static bool debugDrawCallback(void * param, int x, int y, Vector3::Arg, Vector3::Arg, Vector3::Arg, float area)
{
    Image * image = (Image *)param;

    if (area > 0.0) {
        Color32 c = image->pixel(x, y);
        c.r = chartColor.r;
        c.g = chartColor.g;
        c.b = chartColor.b;
        c.a += U8(ftoi_round(0.5f * area * 255));
        image->pixel(x, y) = c;
    }

    return true;
}

void AtlasPacker::addChart(const Chart * chart, int w, int h, int x, int y, int r, Image * debugOutput)
{
    nvDebugCheck(r == 0 || r == 1);

    nvDebugCheck(debugOutput != NULL);
    selectRandomColor(m_rand);

    Vector2 extents = Vector2(float(w), float(h));
    Vector2 offset = Vector2(float(x), float(y)) + Vector2(0.5);

    // Rasterize chart faces, set bits.
    const uint faceCount = chart->faceCount();
    for (uint f = 0; f < faceCount; f++)
    {
        const HalfEdge::Face * face = chart->chartMesh()->faceAt(f);
        
        Vector2 vertices[4];

        uint edgeCount = 0;
        for (HalfEdge::Face::ConstEdgeIterator it(face->edges()); !it.isDone(); it.advance())
        {
            if (edgeCount < 4)
            {
                Vector2 t = it.vertex()->tex;
                if (r == 1) swap(t.x, t.y);
                vertices[edgeCount] = t + offset;
            }
            edgeCount++;
        }

        if (edgeCount == 3)
        {
            Raster::drawTriangle(Raster::Mode_Antialiased, extents, /*enableScissors=*/true, vertices, debugDrawCallback, debugOutput);
        }
        else
        {
            Raster::drawQuad(Raster::Mode_Antialiased, extents, /*enableScissors=*/true, vertices, debugDrawCallback, debugOutput);
        }
    }
}


void AtlasPacker::addChart(const BitMap * bitmap, int atlas_w, int atlas_h, int offset_x, int offset_y, int r, Image * debugOutput)
{
    nvDebugCheck(r == 0 || r == 1);

    // Check whether the two bitmaps overlap.

    const int w = bitmap->width();
    const int h = bitmap->height();

    if (debugOutput != NULL) {
        selectRandomColor(m_rand);
    }

    if (r == 0) {
        for (int y = 0; y < h; y++) {
            int yy = y + offset_y;
            if (yy >= 0) {
                for (int x = 0; x < w; x++) {
                    int xx = x + offset_x;
                    if (xx >= 0) {
                        if (bitmap->bitAt(x, y)) {
                            if (xx < atlas_w && yy < atlas_h) {
                                if (debugOutput) debugOutput->pixel(xx, yy) = chartColor;
                                else {
                                    nvDebugCheck(m_bitmap.bitAt(xx, yy) == false);
                                    m_bitmap.setBitAt(xx, yy);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else if (r == 1) {
        for (int y = 0; y < h; y++) {
            int xx = y + offset_x;
            if (xx >= 0) {
                for (int x = 0; x < w; x++) {
                    int yy = x + offset_y;
                    if (yy >= 0) {
                        if (bitmap->bitAt(x, y)) {
                            if (xx < atlas_w && yy < atlas_h) {
                                if (debugOutput) debugOutput->pixel(xx, yy) = chartColor;
                                else {
                                    nvDebugCheck(m_bitmap.bitAt(xx, yy) == false);
                                    m_bitmap.setBitAt(xx, yy);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



/*static*/ bool AtlasPacker::checkBitsCallback(void * param, int x, int y, Vector3::Arg, Vector3::Arg, Vector3::Arg, float)
{
    BitMap * bitmap = (BitMap * )param;

    nvDebugCheck(bitmap->bitAt(x, y) == false);

    return true;
}

/*static*/ bool AtlasPacker::setBitsCallback(void * param, int x, int y, Vector3::Arg, Vector3::Arg, Vector3::Arg, float area)
{
    BitMap * bitmap = (BitMap * )param;

    if (area > 0.0) {
        bitmap->setBitAt(x, y);
    }

    return true;
}



float AtlasPacker::computeAtlasUtilization() const {
    const uint w = m_width;
    const uint h = m_height;
    nvDebugCheck(w <= m_bitmap.width());
    nvDebugCheck(h <= m_bitmap.height());

    uint count = 0;
    for (uint y = 0; y < h; y++) {
        for (uint x = 0; x < w; x++) {
            count += m_bitmap.bitAt(x, y);
        }
    }

    return float(count) / (w * h);
}

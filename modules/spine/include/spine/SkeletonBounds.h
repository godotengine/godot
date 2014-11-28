/******************************************************************************
 * Spine Runtimes Software License
 * Version 2.1
 * 
 * Copyright (c) 2013, Esoteric Software
 * All rights reserved.
 * 
 * You are granted a perpetual, non-exclusive, non-sublicensable and
 * non-transferable license to install, execute and perform the Spine Runtimes
 * Software (the "Software") solely for internal use. Without the written
 * permission of Esoteric Software (typically granted by licensing Spine), you
 * may not (a) modify, translate, adapt or otherwise create derivative works,
 * improvements of the Software or develop new applications using the Software
 * or (b) remove, delete, alter or obscure any trademarks or any copyright,
 * trademark, patent or other intellectual property or proprietary rights
 * notices on or in the Software, including any copy thereof. Redistributions
 * in binary or source form must include this license and terms.
 * 
 * THIS SOFTWARE IS PROVIDED BY ESOTERIC SOFTWARE "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL ESOTERIC SOFTARE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#ifndef SPINE_SKELETONBOUNDS_H_
#define SPINE_SKELETONBOUNDS_H_

#include <spine/BoundingBoxAttachment.h>
#include <spine/Skeleton.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spPolygon {
	float* const vertices;
	int count;
	int capacity;
} spPolygon;

spPolygon* spPolygon_create (int capacity);
void spPolygon_dispose (spPolygon* self);

int/*bool*/spPolygon_containsPoint (spPolygon* polygon, float x, float y);
int/*bool*/spPolygon_intersectsSegment (spPolygon* polygon, float x1, float y1, float x2, float y2);

#ifdef SPINE_SHORT_NAMES
typedef spPolygon Polygon;
#define Polygon_create(...) spPolygon_create(__VA_ARGS__)
#define Polygon_dispose(...) spPolygon_dispose(__VA_ARGS__)
#define Polygon_containsPoint(...) spPolygon_containsPoint(__VA_ARGS__)
#define Polygon_intersectsSegment(...) spPolygon_intersectsSegment(__VA_ARGS__)
#endif

/**/

typedef struct spSkeletonBounds {
	int count;
	spBoundingBoxAttachment** boundingBoxes;
	spPolygon** polygons;

	float minX, minY, maxX, maxY;
} spSkeletonBounds;

spSkeletonBounds* spSkeletonBounds_create ();
void spSkeletonBounds_dispose (spSkeletonBounds* self);
void spSkeletonBounds_update (spSkeletonBounds* self, spSkeleton* skeleton, int/*bool*/updateAabb);

/** Returns true if the axis aligned bounding box contains the point. */
int/*bool*/spSkeletonBounds_aabbContainsPoint (spSkeletonBounds* self, float x, float y);

/** Returns true if the axis aligned bounding box intersects the line segment. */
int/*bool*/spSkeletonBounds_aabbIntersectsSegment (spSkeletonBounds* self, float x1, float y1, float x2, float y2);

/** Returns true if the axis aligned bounding box intersects the axis aligned bounding box of the specified bounds. */
int/*bool*/spSkeletonBounds_aabbIntersectsSkeleton (spSkeletonBounds* self, spSkeletonBounds* bounds);

/** Returns the first bounding box attachment that contains the point, or null. When doing many checks, it is usually more
 * efficient to only call this method if spSkeletonBounds_aabbContainsPoint returns true. */
spBoundingBoxAttachment* spSkeletonBounds_containsPoint (spSkeletonBounds* self, float x, float y);

/** Returns the first bounding box attachment that contains the line segment, or null. When doing many checks, it is usually
 * more efficient to only call this method if spSkeletonBounds_aabbIntersectsSegment returns true. */
spBoundingBoxAttachment* spSkeletonBounds_intersectsSegment (spSkeletonBounds* self, float x1, float y1, float x2, float y2);

/** Returns the polygon for the specified bounding box, or null. */
spPolygon* spSkeletonBounds_getPolygon (spSkeletonBounds* self, spBoundingBoxAttachment* boundingBox);

#ifdef SPINE_SHORT_NAMES
typedef spSkeletonBounds SkeletonBounds;
#define SkeletonBounds_create(...) spSkeletonBounds_create(__VA_ARGS__)
#define SkeletonBounds_dispose(...) spSkeletonBounds_dispose(__VA_ARGS__)
#define SkeletonBounds_update(...) spSkeletonBounds_update(__VA_ARGS__)
#define SkeletonBounds_aabbContainsPoint(...) spSkeletonBounds_aabbContainsPoint(__VA_ARGS__)
#define SkeletonBounds_aabbIntersectsSegment(...) spSkeletonBounds_aabbIntersectsSegment(__VA_ARGS__)
#define SkeletonBounds_aabbIntersectsSkeleton(...) spSkeletonBounds_aabbIntersectsSkeleton(__VA_ARGS__)
#define SkeletonBounds_containsPoint(...) spSkeletonBounds_containsPoint(__VA_ARGS__)
#define SkeletonBounds_intersectsSegment(...) spSkeletonBounds_intersectsSegment(__VA_ARGS__)
#define SkeletonBounds_getPolygon(...) spSkeletonBounds_getPolygon(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_SKELETONBOUNDS_H_ */

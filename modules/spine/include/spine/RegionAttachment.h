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

#ifndef SPINE_REGIONATTACHMENT_H_
#define SPINE_REGIONATTACHMENT_H_

#include <spine/Attachment.h>
#include <spine/Atlas.h>
#include <spine/Slot.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	SP_VERTEX_X1 = 0, SP_VERTEX_Y1, SP_VERTEX_X2, SP_VERTEX_Y2, SP_VERTEX_X3, SP_VERTEX_Y3, SP_VERTEX_X4, SP_VERTEX_Y4
} spVertexIndex;

typedef struct spRegionAttachment {
	spAttachment super;
	const char* path;
	float x, y, scaleX, scaleY, rotation, width, height;
	float r, g, b, a;

	void* rendererObject;
	int regionOffsetX, regionOffsetY; /* Pixels stripped from the bottom left, unrotated. */
	int regionWidth, regionHeight; /* Unrotated, stripped pixel size. */
	int regionOriginalWidth, regionOriginalHeight; /* Unrotated, unstripped pixel size. */

	float offset[8];
	float uvs[8];
} spRegionAttachment;

spRegionAttachment* spRegionAttachment_create (const char* name);
void spRegionAttachment_setUVs (spRegionAttachment* self, float u, float v, float u2, float v2, int/*bool*/rotate);
void spRegionAttachment_updateOffset (spRegionAttachment* self);
void spRegionAttachment_computeWorldVertices (spRegionAttachment* self, spBone* bone, float* vertices);

#ifdef SPINE_SHORT_NAMES
typedef spVertexIndex VertexIndex;
#define VERTEX_X1 SP_VERTEX_X1
#define VERTEX_Y1 SP_VERTEX_Y1
#define VERTEX_X2 SP_VERTEX_X2
#define VERTEX_Y2 SP_VERTEX_Y2
#define VERTEX_X3 SP_VERTEX_X3
#define VERTEX_Y3 SP_VERTEX_Y3
#define VERTEX_X4 SP_VERTEX_X4
#define VERTEX_Y4 SP_VERTEX_Y4
typedef spRegionAttachment RegionAttachment;
#define RegionAttachment_create(...) spRegionAttachment_create(__VA_ARGS__)
#define RegionAttachment_setUVs(...) spRegionAttachment_setUVs(__VA_ARGS__)
#define RegionAttachment_updateOffset(...) spRegionAttachment_updateOffset(__VA_ARGS__)
#define RegionAttachment_computeWorldVertices(...) spRegionAttachment_computeWorldVertices(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_REGIONATTACHMENT_H_ */

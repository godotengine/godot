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

#ifndef SPINE_SKINNEDMESHATTACHMENT_H_
#define SPINE_SKINNEDMESHATTACHMENT_H_

#include <spine/Attachment.h>
#include <spine/Slot.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spSkinnedMeshAttachment {
	spAttachment super;
	const char* path;

	int bonesCount;
	int* bones;

	int weightsCount;
	float* weights;

	int trianglesCount;
	int* triangles;

	int uvsCount;
	float* regionUVs;
	float* uvs;
	int hullLength;

	float r, g, b, a;

	void* rendererObject;
	int regionOffsetX, regionOffsetY; /* Pixels stripped from the bottom left, unrotated. */
	int regionWidth, regionHeight; /* Unrotated, stripped pixel size. */
	int regionOriginalWidth, regionOriginalHeight; /* Unrotated, unstripped pixel size. */
	float regionU, regionV, regionU2, regionV2;
	int/*bool*/regionRotate;

	/* Nonessential. */
	int edgesCount;
	int* edges;
	float width, height;
} spSkinnedMeshAttachment;

spSkinnedMeshAttachment* spSkinnedMeshAttachment_create (const char* name);
void spSkinnedMeshAttachment_updateUVs (spSkinnedMeshAttachment* self);
void spSkinnedMeshAttachment_computeWorldVertices (spSkinnedMeshAttachment* self, spSlot* slot, float* worldVertices);

#ifdef SPINE_SHORT_NAMES
typedef spSkinnedMeshAttachment SkinnedMeshAttachment;
#define SkinnedMeshAttachment_create(...) spSkinnedMeshAttachment_create(__VA_ARGS__)
#define SkinnedMeshAttachment_updateUVs(...) spSkinnedMeshAttachment_updateUVs(__VA_ARGS__)
#define SkinnedMeshAttachment_computeWorldVertices(...) spSkinnedMeshAttachment_computeWorldVertices(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_SKINNEDMESHATTACHMENT_H_ */

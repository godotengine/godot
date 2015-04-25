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

#ifndef SPINE_BONE_H_
#define SPINE_BONE_H_

#include <spine/BoneData.h>

#ifdef __cplusplus
extern "C" {
#endif

struct spSkeleton;

typedef struct spBone spBone;
struct spBone {
	spBoneData* const data;
	struct spSkeleton* const skeleton;
	spBone* const parent;
	float x, y;
	float rotation, rotationIK;
	float scaleX, scaleY;
	int/*bool*/flipX, flipY;

	float const m00, m01, worldX; /* a b x */
	float const m10, m11, worldY; /* c d y */
	float const worldRotation;
	float const worldScaleX, worldScaleY;
	int/*bool*/const worldFlipX, worldFlipY;
};

void spBone_setYDown (int/*bool*/yDown);

/* @param parent May be 0. */
spBone* spBone_create (spBoneData* data, struct spSkeleton* skeleton, spBone* parent);
void spBone_dispose (spBone* self);

void spBone_setToSetupPose (spBone* self);

void spBone_updateWorldTransform (spBone* self);

void spBone_worldToLocal (spBone* self, float worldX, float worldY, float* localX, float* localY);
void spBone_localToWorld (spBone* self, float localX, float localY, float* worldX, float* worldY);

#ifdef SPINE_SHORT_NAMES
typedef spBone Bone;
#define Bone_setYDown(...) spBone_setYDown(__VA_ARGS__)
#define Bone_create(...) spBone_create(__VA_ARGS__)
#define Bone_dispose(...) spBone_dispose(__VA_ARGS__)
#define Bone_setToSetupPose(...) spBone_setToSetupPose(__VA_ARGS__)
#define Bone_updateWorldTransform(...) spBone_updateWorldTransform(__VA_ARGS__)
#define Bone_worldToLocal(...) spBone_worldToLocal(__VA_ARGS__)
#define Bone_localToWorld(...) spBone_localToWorld(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_BONE_H_ */

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

#include <spine/MeshAttachment.h>
#include <spine/extension.h>

void _spMeshAttachment_dispose (spAttachment* attachment) {
	spMeshAttachment* self = SUB_CAST(spMeshAttachment, attachment);
	_spAttachment_deinit(attachment);
	FREE(self->path);
	FREE(self->vertices);
	FREE(self->regionUVs);
	FREE(self->uvs);
	FREE(self->triangles);
	FREE(self->edges);
	FREE(self);
}

spMeshAttachment* spMeshAttachment_create (const char* name) {
	spMeshAttachment* self = NEW(spMeshAttachment);
	self->r = 1;
	self->g = 1;
	self->b = 1;
	self->a = 1;
	_spAttachment_init(SUPER(self), name, SP_ATTACHMENT_MESH, _spMeshAttachment_dispose);
	return self;
}

void spMeshAttachment_updateUVs (spMeshAttachment* self) {
	int i;
	float width = self->regionU2 - self->regionU, height = self->regionV2 - self->regionV;
	FREE(self->uvs);
	self->uvs = MALLOC(float, self->verticesCount);
	if (self->regionRotate) {
		for (i = 0; i < self->verticesCount; i += 2) {
			self->uvs[i] = self->regionU + self->regionUVs[i + 1] * width;
			self->uvs[i + 1] = self->regionV + height - self->regionUVs[i] * height;
		}
	} else {
		for (i = 0; i < self->verticesCount; i += 2) {
			self->uvs[i] = self->regionU + self->regionUVs[i] * width;
			self->uvs[i + 1] = self->regionV + self->regionUVs[i + 1] * height;
		}
	}
}

void spMeshAttachment_computeWorldVertices (spMeshAttachment* self, spSlot* slot, float* worldVertices) {
	int i;
	float* vertices = self->vertices;
	const spBone* bone = slot->bone;
	float x = bone->skeleton->x + bone->worldX, y = bone->skeleton->y + bone->worldY;
	if (slot->attachmentVerticesCount == self->verticesCount) vertices = slot->attachmentVertices;
	for (i = 0; i < self->verticesCount; i += 2) {
		const float vx = vertices[i], vy = vertices[i + 1];
		worldVertices[i] = vx * bone->m00 + vy * bone->m01 + x;
		worldVertices[i + 1] = vx * bone->m10 + vy * bone->m11 + y;
	}
}

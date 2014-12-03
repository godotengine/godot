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

#include <spine/SkeletonJson.h>
#include <stdio.h>
#include "Json.h"
#include <spine/extension.h>
#include <spine/AtlasAttachmentLoader.h>

typedef struct {
	spSkeletonJson super;
	int ownsLoader;
} _spSkeletonJson;

spSkeletonJson* spSkeletonJson_createWithLoader (spAttachmentLoader* attachmentLoader) {
	spSkeletonJson* self = SUPER(NEW(_spSkeletonJson));
	self->scale = 1;
	self->attachmentLoader = attachmentLoader;
	return self;
}

spSkeletonJson* spSkeletonJson_create (spAtlas* atlas) {
	spAtlasAttachmentLoader* attachmentLoader = spAtlasAttachmentLoader_create(atlas);
	spSkeletonJson* self = spSkeletonJson_createWithLoader(SUPER(attachmentLoader));
	SUB_CAST(_spSkeletonJson, self)->ownsLoader = 1;
	return self;
}

void spSkeletonJson_dispose (spSkeletonJson* self) {
	if (SUB_CAST(_spSkeletonJson, self)->ownsLoader) spAttachmentLoader_dispose(self->attachmentLoader);
	FREE(self->error);
	FREE(self);
}

void _spSkeletonJson_setError (spSkeletonJson* self, Json* root, const char* value1, const char* value2) {
	char message[256];
	int length;
	FREE(self->error);
	strcpy(message, value1);
	length = (int)strlen(value1);
	if (value2) strncat(message + length, value2, 256 - length);
	MALLOC_STR(self->error, message);
	if (root) Json_dispose(root);
}

static float toColor (const char* value, int index) {
	char digits[3];
	char *error;
	int color;

	if (strlen(value) != 8) return -1;
	value += index * 2;

	digits[0] = *value;
	digits[1] = *(value + 1);
	digits[2] = '\0';
	color = (int)strtoul(digits, &error, 16);
	if (*error != 0) return -1;
	return color / (float)255;
}

static void readCurve (spCurveTimeline* timeline, int frameIndex, Json* frame) {
	Json* curve = Json_getItem(frame, "curve");
	if (!curve) return;
	if (curve->type == Json_String && strcmp(curve->valueString, "stepped") == 0)
		spCurveTimeline_setStepped(timeline, frameIndex);
	else if (curve->type == Json_Array) {
		Json* child0 = curve->child;
		Json* child1 = child0->next;
		Json* child2 = child1->next;
		Json* child3 = child2->next;
		spCurveTimeline_setCurve(timeline, frameIndex, child0->valueFloat, child1->valueFloat, child2->valueFloat,
				child3->valueFloat);
	}
}

static spAnimation* _spSkeletonJson_readAnimation (spSkeletonJson* self, Json* root, spSkeletonData *skeletonData) {
	int i;
	spAnimation* animation;
	Json* frame;
	float duration;
	int timelinesCount = 0;

	Json* bones = Json_getItem(root, "bones");
	Json* slots = Json_getItem(root, "slots");
	Json* ik = Json_getItem(root, "ik");
	Json* ffd = Json_getItem(root, "ffd");
	Json* drawOrder = Json_getItem(root, "drawOrder");
	Json* events = Json_getItem(root, "events");
	Json* flipX = Json_getItem(root, "flipx");
	Json* flipY = Json_getItem(root, "flipy");
	Json *boneMap, *slotMap, *ikMap, *ffdMap;
	if (!drawOrder) drawOrder = Json_getItem(root, "draworder");

	for (boneMap = bones ? bones->child : 0; boneMap; boneMap = boneMap->next)
		timelinesCount += boneMap->size;
	for (slotMap = slots ? slots->child : 0; slotMap; slotMap = slotMap->next)
		timelinesCount += slotMap->size;
	timelinesCount += ik ? ik->size : 0;
	for (ffdMap = ffd ? ffd->child : 0; ffdMap; ffdMap = ffdMap->next)
		for (slotMap = ffdMap->child; slotMap; slotMap = slotMap->next)
			timelinesCount += slotMap->size;
	if (drawOrder) ++timelinesCount;
	if (events) ++timelinesCount;
	if (flipX) ++timelinesCount;
	if (flipY) ++timelinesCount;

	animation = spAnimation_create(root->name, timelinesCount);
	animation->timelinesCount = 0;
	skeletonData->animations[skeletonData->animationsCount++] = animation;

	/* Slot timelines. */
	for (slotMap = slots ? slots->child : 0; slotMap; slotMap = slotMap->next) {
		Json *timelineArray;

		int slotIndex = spSkeletonData_findSlotIndex(skeletonData, slotMap->name);
		if (slotIndex == -1) {
			spAnimation_dispose(animation);
			_spSkeletonJson_setError(self, root, "Slot not found: ", slotMap->name);
			return 0;
		}

		for (timelineArray = slotMap->child; timelineArray; timelineArray = timelineArray->next) {
			if (strcmp(timelineArray->name, "color") == 0) {
				spColorTimeline *timeline = spColorTimeline_create(timelineArray->size);
				timeline->slotIndex = slotIndex;
				for (frame = timelineArray->child, i = 0; frame; frame = frame->next, ++i) {
					const char* s = Json_getString(frame, "color", 0);
					spColorTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0), toColor(s, 0), toColor(s, 1), toColor(s, 2),
							toColor(s, 3));
					readCurve(SUPER(timeline), i, frame);
				}
				animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
				duration = timeline->frames[timelineArray->size * 5 - 5];
				if (duration > animation->duration) animation->duration = duration;

			} else if (strcmp(timelineArray->name, "attachment") == 0) {
				spAttachmentTimeline *timeline = spAttachmentTimeline_create(timelineArray->size);
				timeline->slotIndex = slotIndex;
				for (frame = timelineArray->child, i = 0; frame; frame = frame->next, ++i) {
					Json* name = Json_getItem(frame, "name");
					spAttachmentTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0),
							name->type == Json_NULL ? 0 : name->valueString);
				}
				animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
				duration = timeline->frames[timelineArray->size - 1];
				if (duration > animation->duration) animation->duration = duration;

			} else {
				spAnimation_dispose(animation);
				_spSkeletonJson_setError(self, 0, "Invalid timeline type for a slot: ", timelineArray->name);
				return 0;
			}
		}
	}

	/* Bone timelines. */
	for (boneMap = bones ? bones->child : 0; boneMap; boneMap = boneMap->next) {
		Json *timelineArray;

		int boneIndex = spSkeletonData_findBoneIndex(skeletonData, boneMap->name);
		if (boneIndex == -1) {
			spAnimation_dispose(animation);
			_spSkeletonJson_setError(self, root, "Bone not found: ", boneMap->name);
			return 0;
		}

		for (timelineArray = boneMap->child; timelineArray; timelineArray = timelineArray->next) {
			if (strcmp(timelineArray->name, "rotate") == 0) {
				spRotateTimeline *timeline = spRotateTimeline_create(timelineArray->size);
				timeline->boneIndex = boneIndex;
				for (frame = timelineArray->child, i = 0; frame; frame = frame->next, ++i) {
					spRotateTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0), Json_getFloat(frame, "angle", 0));
					readCurve(SUPER(timeline), i, frame);
				}
				animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
				duration = timeline->frames[timelineArray->size * 2 - 2];
				if (duration > animation->duration) animation->duration = duration;

			} else {
				int isScale = strcmp(timelineArray->name, "scale") == 0;
				if (isScale || strcmp(timelineArray->name, "translate") == 0) {
					float scale = isScale ? 1 : self->scale;
					spTranslateTimeline *timeline =
							isScale ? spScaleTimeline_create(timelineArray->size) : spTranslateTimeline_create(timelineArray->size);
					timeline->boneIndex = boneIndex;
					for (frame = timelineArray->child, i = 0; frame; frame = frame->next, ++i) {
						spTranslateTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0), Json_getFloat(frame, "x", 0) * scale,
								Json_getFloat(frame, "y", 0) * scale);
						readCurve(SUPER(timeline), i, frame);
					}
					animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
					duration = timeline->frames[timelineArray->size * 3 - 3];
					if (duration > animation->duration) animation->duration = duration;
				} else if (strcmp(timelineArray->name, "flipX") == 0 || strcmp(timelineArray->name, "flipY") == 0) {
					int x = strcmp(timelineArray->name, "flipX") == 0;
					const char* field = x ? "x" : "y";
					spFlipTimeline *timeline = spFlipTimeline_create(timelineArray->size, x);
					timeline->boneIndex = boneIndex;
					for (frame = timelineArray->child, i = 0; frame; frame = frame->next, ++i)
						spFlipTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0), Json_getInt(frame, field, 0));
					animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
					duration = timeline->frames[timelineArray->size * 2 - 2];
					if (duration > animation->duration) animation->duration = duration;

				} else {
					spAnimation_dispose(animation);
					_spSkeletonJson_setError(self, 0, "Invalid timeline type for a bone: ", timelineArray->name);
					return 0;
				}
			}
		}
	}

	/* IK timelines. */
	for (ikMap = ik ? ik->child : 0; ikMap; ikMap = ikMap->next) {
		spIkConstraintData* ikConstraint = spSkeletonData_findIkConstraint(skeletonData, ikMap->name);
		spIkConstraintTimeline* timeline = spIkConstraintTimeline_create(ikMap->size);
		for (i = 0; i < skeletonData->ikConstraintsCount; ++i) {
			if (ikConstraint == skeletonData->ikConstraints[i]) {
				timeline->ikConstraintIndex = i;
				break;
			}
		}
		for (frame = ikMap->child, i = 0; frame; frame = frame->next, ++i) {
			spIkConstraintTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0), Json_getFloat(frame, "mix", 0),
					Json_getInt(frame, "bendPositive", 1) ? 1 : -1);
			readCurve(SUPER(timeline), i, frame);
		}
		animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
		duration = timeline->frames[ikMap->size * 3 - 3];
		if (duration > animation->duration) animation->duration = duration;
	}

	/* FFD timelines. */
	for (ffdMap = ffd ? ffd->child : 0; ffdMap; ffdMap = ffdMap->next) {
		spSkin* skin = spSkeletonData_findSkin(skeletonData, ffdMap->name);
		for (slotMap = ffdMap->child; slotMap; slotMap = slotMap->next) {
			int slotIndex = spSkeletonData_findSlotIndex(skeletonData, slotMap->name);
			Json* timelineArray;
			for (timelineArray = slotMap->child; timelineArray; timelineArray = timelineArray->next) {
				Json* frame;
				int verticesCount = 0;
				float* tempVertices;
				spFFDTimeline *timeline;

				spAttachment* attachment = spSkin_getAttachment(skin, slotIndex, timelineArray->name);
				if (!attachment) {
					spAnimation_dispose(animation);
					_spSkeletonJson_setError(self, 0, "Attachment not found: ", timelineArray->name);
					return 0;
				}
				if (attachment->type == SP_ATTACHMENT_MESH)
					verticesCount = SUB_CAST(spMeshAttachment, attachment)->verticesCount;
				else if (attachment->type == SP_ATTACHMENT_SKINNED_MESH)
					verticesCount = SUB_CAST(spSkinnedMeshAttachment, attachment)->weightsCount / 3 * 2;

				timeline = spFFDTimeline_create(timelineArray->size, verticesCount);
				timeline->slotIndex = slotIndex;
				timeline->attachment = attachment;

				tempVertices = MALLOC(float, verticesCount);
				for (frame = timelineArray->child, i = 0; frame; frame = frame->next, ++i) {
					Json* vertices = Json_getItem(frame, "vertices");
					float* frameVertices;
					if (!vertices) {
						if (attachment->type == SP_ATTACHMENT_MESH)
							frameVertices = SUB_CAST(spMeshAttachment, attachment)->vertices;
						else {
							frameVertices = tempVertices;
							memset(frameVertices, 0, sizeof(float) * verticesCount);
						}
					} else {
						int v, start = Json_getInt(frame, "offset", 0);
						Json* vertex;
						frameVertices = tempVertices;
						memset(frameVertices, 0, sizeof(float) * start);
						if (self->scale == 1) {
							for (vertex = vertices->child, v = start; vertex; vertex = vertex->next, ++v)
								frameVertices[v] = vertex->valueFloat;
						} else {
							for (vertex = vertices->child, v = start; vertex; vertex = vertex->next, ++v)
								frameVertices[v] = vertex->valueFloat * self->scale;
						}
						memset(frameVertices + v, 0, sizeof(float) * (verticesCount - v));
						if (attachment->type == SP_ATTACHMENT_MESH) {
							float* meshVertices = SUB_CAST(spMeshAttachment, attachment)->vertices;
							for (v = 0; v < verticesCount; ++v)
								frameVertices[v] += meshVertices[v];
						}
					}
					spFFDTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0), frameVertices);
					readCurve(SUPER(timeline), i, frame);
				}
				FREE(tempVertices);

				animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
				duration = timeline->frames[timelineArray->size - 1];
				if (duration > animation->duration) animation->duration = duration;
			}
		}
	}

	/* Draw order timeline. */
	if (drawOrder) {
		spDrawOrderTimeline* timeline = spDrawOrderTimeline_create(drawOrder->size, skeletonData->slotsCount);
		for (frame = drawOrder->child, i = 0; frame; frame = frame->next, ++i) {
			int ii;
			int* drawOrder = 0;
			Json* offsets = Json_getItem(frame, "offsets");
			if (offsets) {
				Json* offsetMap;
				int* unchanged = MALLOC(int, skeletonData->slotsCount - offsets->size);
				int originalIndex = 0, unchangedIndex = 0;

				drawOrder = MALLOC(int, skeletonData->slotsCount);
				for (ii = skeletonData->slotsCount - 1; ii >= 0; --ii)
					drawOrder[ii] = -1;

				for (offsetMap = offsets->child; offsetMap; offsetMap = offsetMap->next) {
					int slotIndex = spSkeletonData_findSlotIndex(skeletonData, Json_getString(offsetMap, "slot", 0));
					if (slotIndex == -1) {
						spAnimation_dispose(animation);
						_spSkeletonJson_setError(self, 0, "Slot not found: ", Json_getString(offsetMap, "slot", 0));
						return 0;
					}
					/* Collect unchanged items. */
					while (originalIndex != slotIndex)
						unchanged[unchangedIndex++] = originalIndex++;
					/* Set changed items. */
					drawOrder[originalIndex + Json_getInt(offsetMap, "offset", 0)] = originalIndex;
					originalIndex++;
				}
				/* Collect remaining unchanged items. */
				while (originalIndex < skeletonData->slotsCount)
					unchanged[unchangedIndex++] = originalIndex++;
				/* Fill in unchanged items. */
				for (ii = skeletonData->slotsCount - 1; ii >= 0; ii--)
					if (drawOrder[ii] == -1) drawOrder[ii] = unchanged[--unchangedIndex];
				FREE(unchanged);
			}
			spDrawOrderTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0), drawOrder);
			FREE(drawOrder);
		}
		animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
		duration = timeline->frames[drawOrder->size - 1];
		if (duration > animation->duration) animation->duration = duration;
	}

	/* Event timeline. */
	if (events) {
		Json* frame;

		spEventTimeline* timeline = spEventTimeline_create(events->size);
		for (frame = events->child, i = 0; frame; frame = frame->next, ++i) {
			spEvent* event;
			const char* stringValue;
			spEventData* eventData = spSkeletonData_findEvent(skeletonData, Json_getString(frame, "name", 0));
			if (!eventData) {
				spAnimation_dispose(animation);
				_spSkeletonJson_setError(self, 0, "Event not found: ", Json_getString(frame, "name", 0));
				return 0;
			}
			event = spEvent_create(eventData);
			event->intValue = Json_getInt(frame, "int", eventData->intValue);
			event->floatValue = Json_getFloat(frame, "float", eventData->floatValue);
			stringValue = Json_getString(frame, "string", eventData->stringValue);
			if (stringValue) MALLOC_STR(event->stringValue, stringValue);
			spEventTimeline_setFrame(timeline, i, Json_getFloat(frame, "time", 0), event);
		}
		animation->timelines[animation->timelinesCount++] = SUPER_CAST(spTimeline, timeline);
		duration = timeline->frames[events->size - 1];
		if (duration > animation->duration) animation->duration = duration;
	}

	return animation;
}

spSkeletonData* spSkeletonJson_readSkeletonDataFile (spSkeletonJson* self, const char* path) {
	int length;
	spSkeletonData* skeletonData;
	const char* json = _spUtil_readFile(path, &length);
	if (!json) {
		_spSkeletonJson_setError(self, 0, "Unable to read skeleton file: ", path);
		return 0;
	}
	skeletonData = spSkeletonJson_readSkeletonData(self, json);
	FREE(json);
	return skeletonData;
}

spSkeletonData* spSkeletonJson_readSkeletonData (spSkeletonJson* self, const char* json) {
	int i, ii;
	spSkeletonData* skeletonData;
	Json *root, *skeleton, *bones, *boneMap, *ik, *slots, *skins, *animations, *events;

	FREE(self->error);
	CONST_CAST(char*, self->error) = 0;

	root = Json_create(json);
	if (!root) {
		_spSkeletonJson_setError(self, 0, "Invalid skeleton JSON: ", Json_getError());
		return 0;
	}

	skeletonData = spSkeletonData_create();

	skeleton = Json_getItem(root, "skeleton");
	if (skeleton) {
		MALLOC_STR(skeletonData->hash, Json_getString(skeleton, "hash", 0));
		MALLOC_STR(skeletonData->version,  Json_getString(skeleton, "spine", 0));
		skeletonData->width = Json_getFloat(skeleton, "width", 0);
		skeletonData->height = Json_getFloat(skeleton, "height", 0);
	}

	/* Bones. */
	bones = Json_getItem(root, "bones");
	skeletonData->bones = MALLOC(spBoneData*, bones->size);
	for (boneMap = bones->child, i = 0; boneMap; boneMap = boneMap->next, ++i) {
		spBoneData* boneData;

		spBoneData* parent = 0;
		const char* parentName = Json_getString(boneMap, "parent", 0);
		if (parentName) {
			parent = spSkeletonData_findBone(skeletonData, parentName);
			if (!parent) {
				spSkeletonData_dispose(skeletonData);
				_spSkeletonJson_setError(self, root, "Parent bone not found: ", parentName);
				return 0;
			}
		}

		boneData = spBoneData_create(Json_getString(boneMap, "name", 0), parent);
		boneData->length = Json_getFloat(boneMap, "length", 0) * self->scale;
		boneData->x = Json_getFloat(boneMap, "x", 0) * self->scale;
		boneData->y = Json_getFloat(boneMap, "y", 0) * self->scale;
		boneData->rotation = Json_getFloat(boneMap, "rotation", 0);
		boneData->scaleX = Json_getFloat(boneMap, "scaleX", 1);
		boneData->scaleY = Json_getFloat(boneMap, "scaleY", 1);
		boneData->inheritScale = Json_getInt(boneMap, "inheritScale", 1);
		boneData->inheritRotation = Json_getInt(boneMap, "inheritRotation", 1);
		boneData->flipX = Json_getInt(boneMap, "flipX", 0);
		boneData->flipY = Json_getInt(boneMap, "flipY", 0);

		skeletonData->bones[i] = boneData;
		skeletonData->bonesCount++;
	}

	/* IK constraints. */
	ik = Json_getItem(root, "ik");
	if (ik) {
		Json *ikMap;
		skeletonData->ikConstraintsCount = ik->size;
		skeletonData->ikConstraints = MALLOC(spIkConstraintData*, ik->size);
		for (ikMap = ik->child, i = 0; ikMap; ikMap = ikMap->next, ++i) {
			const char* targetName;

			spIkConstraintData* ikConstraintData = spIkConstraintData_create(Json_getString(ikMap, "name", 0));
			boneMap = Json_getItem(ikMap, "bones");
			ikConstraintData->bonesCount = boneMap->size;
			ikConstraintData->bones = MALLOC(spBoneData*, boneMap->size);
			for (boneMap = boneMap->child, ii = 0; boneMap; boneMap = boneMap->next, ++ii) {
				ikConstraintData->bones[ii] = spSkeletonData_findBone(skeletonData, boneMap->valueString);
				if (!ikConstraintData->bones[ii]) {
					spSkeletonData_dispose(skeletonData);
					_spSkeletonJson_setError(self, root, "IK bone not found: ", boneMap->valueString);
					return 0;
				}
			}

			targetName = Json_getString(ikMap, "target", 0);
			ikConstraintData->target = spSkeletonData_findBone(skeletonData, targetName);
			if (!ikConstraintData->target) {
				spSkeletonData_dispose(skeletonData);
				_spSkeletonJson_setError(self, root, "Target bone not found: ", boneMap->name);
				return 0;
			}

			ikConstraintData->bendDirection = Json_getInt(ikMap, "bendPositive", 1) ? 1 : -1;
			ikConstraintData->mix = Json_getFloat(ikMap, "mix", 1);

			skeletonData->ikConstraints[i] = ikConstraintData;
		}
	}

	/* Slots. */
	slots = Json_getItem(root, "slots");
	if (slots) {
		Json *slotMap;
		skeletonData->slotsCount = slots->size;
		skeletonData->slots = MALLOC(spSlotData*, slots->size);
		for (slotMap = slots->child, i = 0; slotMap; slotMap = slotMap->next, ++i) {
			spSlotData* slotData;
			const char* color;
			Json *attachmentItem;

			const char* boneName = Json_getString(slotMap, "bone", 0);
			spBoneData* boneData = spSkeletonData_findBone(skeletonData, boneName);
			if (!boneData) {
				spSkeletonData_dispose(skeletonData);
				_spSkeletonJson_setError(self, root, "Slot bone not found: ", boneName);
				return 0;
			}

			slotData = spSlotData_create(Json_getString(slotMap, "name", 0), boneData);

			color = Json_getString(slotMap, "color", 0);
			if (color) {
				slotData->r = toColor(color, 0);
				slotData->g = toColor(color, 1);
				slotData->b = toColor(color, 2);
				slotData->a = toColor(color, 3);
			}

			attachmentItem = Json_getItem(slotMap, "attachment");
			if (attachmentItem) spSlotData_setAttachmentName(slotData, attachmentItem->valueString);

			slotData->additiveBlending = Json_getInt(slotMap, "additive", 0);

			skeletonData->slots[i] = slotData;
		}
	}

	/* Skins. */
	skins = Json_getItem(root, "skins");
	if (skins) {
		Json *slotMap;
		skeletonData->skinsCount = skins->size;
		skeletonData->skins = MALLOC(spSkin*, skins->size);
		for (slotMap = skins->child, i = 0; slotMap; slotMap = slotMap->next, ++i) {
			Json *attachmentsMap;
			spSkin *skin = spSkin_create(slotMap->name);

			skeletonData->skins[i] = skin;
			if (strcmp(slotMap->name, "default") == 0) skeletonData->defaultSkin = skin;

			for (attachmentsMap = slotMap->child; attachmentsMap; attachmentsMap = attachmentsMap->next) {
				int slotIndex = spSkeletonData_findSlotIndex(skeletonData, attachmentsMap->name);
				Json *attachmentMap;

				for (attachmentMap = attachmentsMap->child; attachmentMap; attachmentMap = attachmentMap->next) {
					spAttachment* attachment;
					const char* skinAttachmentName = attachmentMap->name;
					const char* attachmentName = Json_getString(attachmentMap, "name", skinAttachmentName);
					const char* path = Json_getString(attachmentMap, "path", attachmentName);
					const char* color;
					int i;
					Json* entry;

					const char* typeString = Json_getString(attachmentMap, "type", "region");
					spAttachmentType type;
					if (strcmp(typeString, "region") == 0)
						type = SP_ATTACHMENT_REGION;
					else if (strcmp(typeString, "mesh") == 0)
						type = SP_ATTACHMENT_MESH;
					else if (strcmp(typeString, "skinnedmesh") == 0)
						type = SP_ATTACHMENT_SKINNED_MESH;
					else if (strcmp(typeString, "boundingbox") == 0)
						type = SP_ATTACHMENT_BOUNDING_BOX;
					else {
						spSkeletonData_dispose(skeletonData);
						_spSkeletonJson_setError(self, root, "Unknown attachment type: ", typeString);
						return 0;
					}

					attachment = spAttachmentLoader_newAttachment(self->attachmentLoader, skin, type, attachmentName, path);
					if (!attachment) {
						if (self->attachmentLoader->error1) {
							spSkeletonData_dispose(skeletonData);
							_spSkeletonJson_setError(self, root, self->attachmentLoader->error1, self->attachmentLoader->error2);
							return 0;
						}
						continue;
					}

					switch (attachment->type) {
					case SP_ATTACHMENT_REGION: {
						spRegionAttachment* region = SUB_CAST(spRegionAttachment, attachment);
						if (path) MALLOC_STR(region->path, path);
						region->x = Json_getFloat(attachmentMap, "x", 0) * self->scale;
						region->y = Json_getFloat(attachmentMap, "y", 0) * self->scale;
						region->scaleX = Json_getFloat(attachmentMap, "scaleX", 1);
						region->scaleY = Json_getFloat(attachmentMap, "scaleY", 1);
						region->rotation = Json_getFloat(attachmentMap, "rotation", 0);
						region->width = Json_getFloat(attachmentMap, "width", 32) * self->scale;
						region->height = Json_getFloat(attachmentMap, "height", 32) * self->scale;

						color = Json_getString(attachmentMap, "color", 0);
						if (color) {
							region->r = toColor(color, 0);
							region->g = toColor(color, 1);
							region->b = toColor(color, 2);
							region->a = toColor(color, 3);
						}

						spRegionAttachment_updateOffset(region);
						break;
					}
					case SP_ATTACHMENT_MESH: {
						spMeshAttachment* mesh = SUB_CAST(spMeshAttachment, attachment);

						MALLOC_STR(mesh->path, path);

						entry = Json_getItem(attachmentMap, "vertices");
						mesh->verticesCount = entry->size;
						mesh->vertices = MALLOC(float, entry->size);
						for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
							mesh->vertices[i] = entry->valueFloat * self->scale;

						entry = Json_getItem(attachmentMap, "triangles");
						mesh->trianglesCount = entry->size;
						mesh->triangles = MALLOC(int, entry->size);
						for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
							mesh->triangles[i] = entry->valueInt;

						entry = Json_getItem(attachmentMap, "uvs");
						mesh->regionUVs = MALLOC(float, entry->size);
						for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
							mesh->regionUVs[i] = entry->valueFloat;

						spMeshAttachment_updateUVs(mesh);

						color = Json_getString(attachmentMap, "color", 0);
						if (color) {
							mesh->r = toColor(color, 0);
							mesh->g = toColor(color, 1);
							mesh->b = toColor(color, 2);
							mesh->a = toColor(color, 3);
						}

						mesh->hullLength = Json_getInt(attachmentMap, "hull", 0);

						entry = Json_getItem(attachmentMap, "edges");
						if (entry) {
							mesh->edgesCount = entry->size;
							mesh->edges = MALLOC(int, entry->size);
							for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
								mesh->edges[i] = entry->valueInt;
						}

						mesh->width = Json_getFloat(attachmentMap, "width", 32) * self->scale;
						mesh->height = Json_getFloat(attachmentMap, "height", 32) * self->scale;
						break;
					}
					case SP_ATTACHMENT_SKINNED_MESH: {
						spSkinnedMeshAttachment* mesh = SUB_CAST(spSkinnedMeshAttachment, attachment);
						int verticesCount, b, w, nn;
						float* vertices;

						MALLOC_STR(mesh->path, path);

						entry = Json_getItem(attachmentMap, "uvs");
						mesh->uvsCount = entry->size;
						mesh->regionUVs = MALLOC(float, entry->size);
						for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
							mesh->regionUVs[i] = entry->valueFloat;

						entry = Json_getItem(attachmentMap, "vertices");
						verticesCount = entry->size;
						vertices = MALLOC(float, entry->size);
						for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
							vertices[i] = entry->valueFloat;

						for (i = 0; i < verticesCount;) {
							int bonesCount = (int)vertices[i];
							mesh->bonesCount += bonesCount + 1;
							mesh->weightsCount += bonesCount * 3;
							i += 1 + bonesCount * 4;
						}
						mesh->bones = MALLOC(int, mesh->bonesCount);
						mesh->weights = MALLOC(float, mesh->weightsCount);

						for (i = 0, b = 0, w = 0; i < verticesCount;) {
							int bonesCount = (int)vertices[i++];
							mesh->bones[b++] = bonesCount;
							for (nn = i + bonesCount * 4; i < nn; i += 4, ++b, w += 3) {
								mesh->bones[b] = (int)vertices[i];
								mesh->weights[w] = vertices[i + 1] * self->scale;
								mesh->weights[w + 1] = vertices[i + 2] * self->scale;
								mesh->weights[w + 2] = vertices[i + 3];
							}
						}

						FREE(vertices);

						entry = Json_getItem(attachmentMap, "triangles");
						mesh->trianglesCount = entry->size;
						mesh->triangles = MALLOC(int, entry->size);
						for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
							mesh->triangles[i] = entry->valueInt;

						spSkinnedMeshAttachment_updateUVs(mesh);

						color = Json_getString(attachmentMap, "color", 0);
						if (color) {
							mesh->r = toColor(color, 0);
							mesh->g = toColor(color, 1);
							mesh->b = toColor(color, 2);
							mesh->a = toColor(color, 3);
						}

						mesh->hullLength = Json_getInt(attachmentMap, "hull", 0);

						entry = Json_getItem(attachmentMap, "edges");
						if (entry) {
							mesh->edgesCount = entry->size;
							mesh->edges = MALLOC(int, entry->size);
							for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
								mesh->edges[i] = entry->valueInt;
						}

						mesh->width = Json_getFloat(attachmentMap, "width", 32) * self->scale;
						mesh->height = Json_getFloat(attachmentMap, "height", 32) * self->scale;
						break;
					}
					case SP_ATTACHMENT_BOUNDING_BOX: {
						spBoundingBoxAttachment* box = SUB_CAST(spBoundingBoxAttachment, attachment);
						entry = Json_getItem(attachmentMap, "vertices");
						box->verticesCount = entry->size;
						box->vertices = MALLOC(float, entry->size);
						for (entry = entry->child, i = 0; entry; entry = entry->next, ++i)
							box->vertices[i] = entry->valueFloat * self->scale;
						break;
					}
					}

					spSkin_addAttachment(skin, slotIndex, skinAttachmentName, attachment);
				}
			}
		}
	}

	/* Events. */
	events = Json_getItem(root, "events");
	if (events) {
		Json *eventMap;
		const char* stringValue;
		skeletonData->eventsCount = events->size;
		skeletonData->events = MALLOC(spEventData*, events->size);
		for (eventMap = events->child, i = 0; eventMap; eventMap = eventMap->next, ++i) {
			spEventData* eventData = spEventData_create(eventMap->name);
			eventData->intValue = Json_getInt(eventMap, "int", 0);
			eventData->floatValue = Json_getFloat(eventMap, "float", 0);
			stringValue = Json_getString(eventMap, "string", 0);
			if (stringValue) MALLOC_STR(eventData->stringValue, stringValue);
			skeletonData->events[i] = eventData;
		}
	}

	/* Animations. */
	animations = Json_getItem(root, "animations");
	if (animations) {
		Json *animationMap;
		skeletonData->animations = MALLOC(spAnimation*, animations->size);
		for (animationMap = animations->child; animationMap; animationMap = animationMap->next)
			_spSkeletonJson_readAnimation(self, animationMap, skeletonData);
	}

	Json_dispose(root);
	return skeletonData;
}

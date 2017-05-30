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

#ifndef SPINE_ATLAS_H_
#define SPINE_ATLAS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spAtlas spAtlas;

typedef enum {
	SP_ATLAS_ALPHA,
	SP_ATLAS_INTENSITY,
	SP_ATLAS_LUMINANCE_ALPHA,
	SP_ATLAS_RGB565,
	SP_ATLAS_RGBA4444,
	SP_ATLAS_RGB888,
	SP_ATLAS_RGBA8888
} spAtlasFormat;

typedef enum {
	SP_ATLAS_NEAREST,
	SP_ATLAS_LINEAR,
	SP_ATLAS_MIPMAP,
	SP_ATLAS_MIPMAP_NEAREST_NEAREST,
	SP_ATLAS_MIPMAP_LINEAR_NEAREST,
	SP_ATLAS_MIPMAP_NEAREST_LINEAR,
	SP_ATLAS_MIPMAP_LINEAR_LINEAR
} spAtlasFilter;

typedef enum {
	SP_ATLAS_MIRROREDREPEAT, SP_ATLAS_CLAMPTOEDGE, SP_ATLAS_REPEAT
} spAtlasWrap;

typedef struct spAtlasPage spAtlasPage;
struct spAtlasPage {
	const spAtlas* atlas;
	const char* name;
	spAtlasFormat format;
	spAtlasFilter minFilter, magFilter;
	spAtlasWrap uWrap, vWrap;

	void* rendererObject;
	int width, height;

	spAtlasPage* next;
};

spAtlasPage* spAtlasPage_create (spAtlas* atlas, const char* name);
void spAtlasPage_dispose (spAtlasPage* self);

#ifdef SPINE_SHORT_NAMES
typedef spAtlasFormat AtlasFormat;
#define ATLAS_ALPHA SP_ATLAS_ALPHA
#define ATLAS_INTENSITY SP_ATLAS_INTENSITY
#define ATLAS_LUMINANCE_ALPHA SP_ATLAS_LUMINANCE_ALPHA
#define ATLAS_RGB565 SP_ATLAS_RGB565
#define ATLAS_RGBA4444 SP_ATLAS_RGBA4444
#define ATLAS_RGB888 SP_ATLAS_RGB888
#define ATLAS_RGBA8888 SP_ATLAS_RGBA8888
typedef spAtlasFilter AtlasFilter;
#define ATLAS_NEAREST SP_ATLAS_NEAREST
#define ATLAS_LINEAR SP_ATLAS_LINEAR
#define ATLAS_MIPMAP SP_ATLAS_MIPMAP
#define ATLAS_MIPMAP_NEAREST_NEAREST SP_ATLAS_MIPMAP_NEAREST_NEAREST
#define ATLAS_MIPMAP_LINEAR_NEAREST SP_ATLAS_MIPMAP_LINEAR_NEAREST
#define ATLAS_MIPMAP_NEAREST_LINEAR SP_ATLAS_MIPMAP_NEAREST_LINEAR
#define ATLAS_MIPMAP_LINEAR_LINEAR SP_ATLAS_MIPMAP_LINEAR_LINEAR
typedef spAtlasWrap AtlasWrap;
#define ATLAS_MIRROREDREPEAT SP_ATLAS_MIRROREDREPEAT
#define ATLAS_CLAMPTOEDGE SP_ATLAS_CLAMPTOEDGE
#define ATLAS_REPEAT SP_ATLAS_REPEAT
typedef spAtlasPage AtlasPage;
#define AtlasPage_create(...) spAtlasPage_create(__VA_ARGS__)
#define AtlasPage_dispose(...) spAtlasPage_dispose(__VA_ARGS__)
#endif

/**/

typedef struct spAtlasRegion spAtlasRegion;
struct spAtlasRegion {
	const char* name;
	int x, y, width, height;
	float u, v, u2, v2;
	int offsetX, offsetY;
	int originalWidth, originalHeight;
	int index;
	int/*bool*/rotate;
	int/*bool*/flip;
	int* splits;
	int* pads;

	spAtlasPage* page;

	spAtlasRegion* next;
};

spAtlasRegion* spAtlasRegion_create ();
void spAtlasRegion_dispose (spAtlasRegion* self);

#ifdef SPINE_SHORT_NAMES
typedef spAtlasRegion AtlasRegion;
#define AtlasRegion_create(...) spAtlasRegion_create(__VA_ARGS__)
#define AtlasRegion_dispose(...) spAtlasRegion_dispose(__VA_ARGS__)
#endif

/**/

struct spAtlas {
	spAtlasPage* pages;
	spAtlasRegion* regions;

	void* rendererObject;
};

/* Image files referenced in the atlas file will be prefixed with dir. */
spAtlas* spAtlas_create (const char* data, int length, const char* dir, void* rendererObject);
/* Image files referenced in the atlas file will be prefixed with the directory containing the atlas file. */
spAtlas* spAtlas_createFromFile (const char* path, void* rendererObject);
void spAtlas_dispose (spAtlas* atlas);

/* Returns 0 if the region was not found. */
spAtlasRegion* spAtlas_findRegion (const spAtlas* self, const char* name);

#ifdef SPINE_SHORT_NAMES
typedef spAtlas Atlas;
#define Atlas_create(...) spAtlas_create(__VA_ARGS__)
#define Atlas_createFromFile(...) spAtlas_createFromFile(__VA_ARGS__)
#define Atlas_dispose(...) spAtlas_dispose(__VA_ARGS__)
#define Atlas_findRegion(...) spAtlas_findRegion(__VA_ARGS__)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPINE_ATLAS_H_ */

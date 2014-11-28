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

#include <spine/Atlas.h>
#include <ctype.h>
#include <spine/extension.h>

spAtlasPage* spAtlasPage_create (spAtlas* atlas, const char* name) {
	spAtlasPage* self = NEW(spAtlasPage);
	CONST_CAST(spAtlas*, self->atlas) = atlas;
	MALLOC_STR(self->name, name);
	return self;
}

void spAtlasPage_dispose (spAtlasPage* self) {
	_spAtlasPage_disposeTexture(self);
	FREE(self->name);
	FREE(self);
}

/**/

spAtlasRegion* spAtlasRegion_create () {
	return NEW(spAtlasRegion);
}

void spAtlasRegion_dispose (spAtlasRegion* self) {
	FREE(self->name);
	FREE(self->splits);
	FREE(self->pads);
	FREE(self);
}

/**/

typedef struct {
	const char* begin;
	const char* end;
} Str;

static void trim (Str* str) {
	while (isspace(*str->begin) && str->begin < str->end)
		(str->begin)++;
	if (str->begin == str->end) return;
	str->end--;
	while (isspace(*str->end) && str->end >= str->begin)
		str->end--;
	str->end++;
}

/* Tokenize string without modification. Returns 0 on failure. */
static int readLine (const char* begin, const char* end, Str* str) {
	static const char* nextStart;
	if (begin) {
		nextStart = begin;
		return 1;
	}
	if (nextStart == end) return 0;
	str->begin = nextStart;

	/* Find next delimiter. */
	while (nextStart != end && *nextStart != '\n')
		nextStart++;

	str->end = nextStart;
	trim(str);

	if (nextStart != end) nextStart++;
	return 1;
}

/* Moves str->begin past the first occurence of c. Returns 0 on failure. */
static int beginPast (Str* str, char c) {
	const char* begin = str->begin;
	while (1) {
		char lastSkippedChar = *begin;
		if (begin == str->end) return 0;
		begin++;
		if (lastSkippedChar == c) break;
	}
	str->begin = begin;
	return 1;
}

/* Returns 0 on failure. */
static int readValue (const char* end, Str* str) {
	readLine(0, end, str);
	if (!beginPast(str, ':')) return 0;
	trim(str);
	return 1;
}

/* Returns the number of tuple values read (1, 2, 4, or 0 for failure). */
static int readTuple (const char* end, Str tuple[]) {
	int i;
	Str str;
	readLine(0, end, &str);
	if (!beginPast(&str, ':')) return 0;

	for (i = 0; i < 3; ++i) {
		tuple[i].begin = str.begin;
		if (!beginPast(&str, ',')) break;
		tuple[i].end = str.begin - 2;
		trim(&tuple[i]);
	}
	tuple[i].begin = str.begin;
	tuple[i].end = str.end;
	trim(&tuple[i]);
	return i + 1;
}

static char* mallocString (Str* str) {
	int length = (int)(str->end - str->begin);
	char* string = MALLOC(char, length + 1);
	memcpy(string, str->begin, length);
	string[length] = '\0';
	return string;
}

static int indexOf (const char** array, int count, Str* str) {
	int length = (int)(str->end - str->begin);
	int i;
	for (i = count - 1; i >= 0; i--)
		if (strncmp(array[i], str->begin, length) == 0) return i;
	return -1;
}

static int equals (Str* str, const char* other) {
	return strncmp(other, str->begin, str->end - str->begin) == 0;
}

static int toInt (Str* str) {
	return (int)strtol(str->begin, (char**)&str->end, 10);
}

static spAtlas* abortAtlas (spAtlas* self) {
	spAtlas_dispose(self);
	return 0;
}

static const char* formatNames[] = {"Alpha", "Intensity", "LuminanceAlpha", "RGB565", "RGBA4444", "RGB888", "RGBA8888"};
static const char* textureFilterNames[] = {"Nearest", "Linear", "MipMap", "MipMapNearestNearest", "MipMapLinearNearest",
		"MipMapNearestLinear", "MipMapLinearLinear"};

spAtlas* spAtlas_create (const char* begin, int length, const char* dir, void* rendererObject) {
	spAtlas* self;

	int count;
	const char* end = begin + length;
	int dirLength = (int)strlen(dir);
	int needsSlash = dirLength > 0 && dir[dirLength - 1] != '/' && dir[dirLength - 1] != '\\';

	spAtlasPage *page = 0;
	spAtlasPage *lastPage = 0;
	spAtlasRegion *lastRegion = 0;
	Str str;
	Str tuple[4];

	self = NEW(spAtlas);
	self->rendererObject = rendererObject;

	readLine(begin, 0, 0);
	while (readLine(0, end, &str)) {
		if (str.end - str.begin == 0) {
			page = 0;
		} else if (!page) {
			char* name = mallocString(&str);
			char* path = MALLOC(char, dirLength + needsSlash + strlen(name) + 1);
			memcpy(path, dir, dirLength);
			if (needsSlash) path[dirLength] = '/';
			strcpy(path + dirLength + needsSlash, name);

			page = spAtlasPage_create(self, name);
			FREE(name);
			if (lastPage)
				lastPage->next = page;
			else
				self->pages = page;
			lastPage = page;

			switch (readTuple(end, tuple)) {
			case 0:
				return abortAtlas(self);
			case 2:  /* size is only optional for an atlas packed with an old TexturePacker. */
				page->width = toInt(tuple);
				page->height = toInt(tuple + 1);
				if (!readTuple(end, tuple)) return abortAtlas(self);
			}
			page->format = (spAtlasFormat)indexOf(formatNames, 7, tuple);

			if (!readTuple(end, tuple)) return abortAtlas(self);
			page->minFilter = (spAtlasFilter)indexOf(textureFilterNames, 7, tuple);
			page->magFilter = (spAtlasFilter)indexOf(textureFilterNames, 7, tuple + 1);

			if (!readValue(end, &str)) return abortAtlas(self);
			if (!equals(&str, "none")) {
				page->uWrap = *str.begin == 'x' ? SP_ATLAS_REPEAT : (*str.begin == 'y' ? SP_ATLAS_CLAMPTOEDGE : SP_ATLAS_REPEAT);
				page->vWrap = *str.begin == 'x' ? SP_ATLAS_CLAMPTOEDGE : (*str.begin == 'y' ? SP_ATLAS_REPEAT : SP_ATLAS_REPEAT);
			}

			_spAtlasPage_createTexture(page, path);
			FREE(path);
		} else {
			spAtlasRegion *region = spAtlasRegion_create();
			if (lastRegion)
				lastRegion->next = region;
			else
				self->regions = region;
			lastRegion = region;

			region->page = page;
			region->name = mallocString(&str);

			if (!readValue(end, &str)) return abortAtlas(self);
			region->rotate = equals(&str, "true");

			if (readTuple(end, tuple) != 2) return abortAtlas(self);
			region->x = toInt(tuple);
			region->y = toInt(tuple + 1);

			if (readTuple(end, tuple) != 2) return abortAtlas(self);
			region->width = toInt(tuple);
			region->height = toInt(tuple + 1);

			region->u = region->x / (float)page->width;
			region->v = region->y / (float)page->height;
			if (region->rotate) {
				region->u2 = (region->x + region->height) / (float)page->width;
				region->v2 = (region->y + region->width) / (float)page->height;
			} else {
				region->u2 = (region->x + region->width) / (float)page->width;
				region->v2 = (region->y + region->height) / (float)page->height;
			}

			if (!(count = readTuple(end, tuple))) return abortAtlas(self);
			if (count == 4) { /* split is optional */
				region->splits = MALLOC(int, 4);
				region->splits[0] = toInt(tuple);
				region->splits[1] = toInt(tuple + 1);
				region->splits[2] = toInt(tuple + 2);
				region->splits[3] = toInt(tuple + 3);

				if (!(count = readTuple(end, tuple))) return abortAtlas(self);
				if (count == 4) { /* pad is optional, but only present with splits */
					region->pads = MALLOC(int, 4);
					region->pads[0] = toInt(tuple);
					region->pads[1] = toInt(tuple + 1);
					region->pads[2] = toInt(tuple + 2);
					region->pads[3] = toInt(tuple + 3);

					if (!readTuple(end, tuple)) return abortAtlas(self);
				}
			}

			region->originalWidth = toInt(tuple);
			region->originalHeight = toInt(tuple + 1);

			readTuple(end, tuple);
			region->offsetX = toInt(tuple);
			region->offsetY = toInt(tuple + 1);

			if (!readValue(end, &str)) return abortAtlas(self);
			region->index = toInt(&str);
		}
	}

	return self;
}

spAtlas* spAtlas_createFromFile (const char* path, void* rendererObject) {
	int dirLength;
	char *dir;
	int length;
	const char* data;

	spAtlas* atlas = 0;

	/* Get directory from atlas path. */
	const char* lastForwardSlash = strrchr(path, '/');
	const char* lastBackwardSlash = strrchr(path, '\\');
	const char* lastSlash = lastForwardSlash > lastBackwardSlash ? lastForwardSlash : lastBackwardSlash;
	if (lastSlash == path) lastSlash++; /* Never drop starting slash. */
	dirLength = (int)(lastSlash ? lastSlash - path : 0);
	dir = MALLOC(char, dirLength + 1);
	memcpy(dir, path, dirLength);
	dir[dirLength] = '\0';

	data = _spUtil_readFile(path, &length);
	if (data) atlas = spAtlas_create(data, length, dir, rendererObject);

	FREE(data);
	FREE(dir);
	return atlas;
}

void spAtlas_dispose (spAtlas* self) {
	spAtlasRegion* region, *nextRegion;
	spAtlasPage* page = self->pages;
	while (page) {
		spAtlasPage* nextPage = page->next;
		spAtlasPage_dispose(page);
		page = nextPage;
	}

	region = self->regions;
	while (region) {
		nextRegion = region->next;
		spAtlasRegion_dispose(region);
		region = nextRegion;
	}

	FREE(self);
}

spAtlasRegion* spAtlas_findRegion (const spAtlas* self, const char* name) {
	spAtlasRegion* region = self->regions;
	while (region) {
		if (strcmp(region->name, name) == 0) return region;
		region = region->next;
	}
	return 0;
}

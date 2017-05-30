/*************************************************************************/
/*  register_types.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifdef MODULE_SPINE_ENABLED

#include "object_type_db.h"
#include "core/globals.h"
#include "register_types.h"

#include <spine/extension.h>
#include <spine/spine.h>
#include "spine.h"

#include "core/os/file_access.h"
#include "core/io/resource_loader.h"
#include "scene/resources/texture.h"

typedef Ref<Texture> TextureRef;

void _spAtlasPage_createTexture(spAtlasPage* self, const char* path) {

	TextureRef *ref = memnew(TextureRef);
	*ref = ResourceLoader::load(path, "Texture");
	ERR_FAIL_COND(ref->is_null());
	self->rendererObject = ref;
	self->width = (*ref)->get_width();
	self->height = (*ref)->get_height();
}

void _spAtlasPage_disposeTexture(spAtlasPage* self) {

	TextureRef *ref = static_cast<TextureRef *>(self->rendererObject);
	memdelete(ref);
}

char* _spUtil_readFile(const char* p_path, int* p_length) {

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V(!f, NULL);

	*p_length = f->get_len();
	char *data = (char *)_malloc(*p_length, __FILE__, __LINE__);
	ERR_FAIL_COND_V(data == NULL, NULL);
	f->get_buffer((uint8_t *)data, *p_length);

	memdelete(f);
	return data;
}

static void *spine_malloc(size_t p_size) {

	return memalloc(p_size);
}

static void spine_free(void *ptr) {

	if (ptr == NULL)
		return;
	memfree(ptr);
}

class ResourceFormatLoaderSpine : public ResourceFormatLoader {
public:

	virtual RES load(const String &p_path, const String& p_original_path = "") {

		Spine::SpineResource *res = memnew(Spine::SpineResource);
		Ref<Spine::SpineResource> ref(res);

		String p_atlas = p_path.replace(".json", ".atlas");
		res->atlas = spAtlas_createFromFile(p_atlas.utf8().get_data(), 0);
		ERR_FAIL_COND_V(res->atlas == NULL, RES());
		spSkeletonJson *json = spSkeletonJson_create(res->atlas);
		ERR_FAIL_COND_V(json == NULL, RES());
		json->scale = 1;

		res->data = spSkeletonJson_readSkeletonDataFile(json, p_path.utf8().get_data());
		ERR_EXPLAIN(json->error ? json->error : "");
		ERR_FAIL_COND_V(res->data == NULL, RES());
		spSkeletonJson_dispose(json);

		res->set_path(p_path);
		return ref;
	}

	virtual void get_recognized_extensions(List<String> *p_extensions) const {

		p_extensions->push_back("json");
	}

	virtual bool handles_type(const String& p_type) const {

		return p_type=="SpineResource";
	}

	virtual String get_resource_type(const String &p_path) const {

		String el = p_path.extension().to_lower();
		if (el=="json")
			return "SpineResource";
		return "";
	}
};

static ResourceFormatLoaderSpine *resource_loader_spine = NULL;

void register_spine_types() {

	ObjectTypeDB::register_type<Spine>();
	ObjectTypeDB::register_type<Spine::SpineResource>();
	resource_loader_spine = memnew( ResourceFormatLoaderSpine );
	ResourceLoader::add_resource_format_loader(resource_loader_spine);

	_setMalloc(spine_malloc);
	_setFree(spine_free);
}

void unregister_spine_types() {

	if (resource_loader_spine)
		memdelete(resource_loader_spine);

}

#else

void register_spine_types() {}
void unregister_spine_types() {}

#endif // MODULE_SPINE_ENABLED

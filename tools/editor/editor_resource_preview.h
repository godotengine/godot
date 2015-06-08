#ifndef EDITORRESOURCEPREVIEW_H
#define EDITORRESOURCEPREVIEW_H

#include "scene/main/node.h"
#include "os/semaphore.h"
#include "os/thread.h"
#include "scene/resources/texture.h"

/* make previews for:
*packdscene
*wav
*image
*mesh
-font
*script
*material
-shader
-shader graph?
-navigation mesh
-collision?
-occluder polygon
-navigation polygon
-tileset
-curve and curve2D
*/


class EditorResourcePreviewGenerator : public Reference {

	OBJ_TYPE(EditorResourcePreviewGenerator,Reference );

public:

	virtual bool handles(const String& p_type) const=0;
	virtual Ref<Texture> generate(const RES& p_from)=0;
	virtual Ref<Texture> generate_from_path(const String& p_path);

	EditorResourcePreviewGenerator();
};


class EditorResourcePreview : public Node {

	OBJ_TYPE(EditorResourcePreview,Node);


	static EditorResourcePreview* singleton;

	struct QueueItem {
		String path;
		ObjectID id;
		StringName function;
		Variant userdata;
	};

	List<QueueItem> queue;

	Mutex *preview_mutex;
	Semaphore *preview_sem;
	Thread *thread;
	bool exit;

	struct Item {
		Ref<Texture> preview;
		int order;
	};

	int order;

	Map<String,Item> cache;

	void _preview_ready(const String& p_str,const Ref<Texture>& p_texture, ObjectID id, const StringName &p_func, const Variant &p_ud);
	Ref<Texture> _generate_preview(const QueueItem& p_item, const String &cache_base);

	static void _thread_func(void *ud);
	void _thread();

	Vector<Ref<EditorResourcePreviewGenerator> > preview_generators;
protected:

	static void _bind_methods();
public:

	static EditorResourcePreview* get_singleton();

	//callback funtion is callback(String p_path,Ref<Texture> preview,Variant udata) preview null if could not load
	void queue_resource_preview(const String& p_path, Object* p_receiver, const StringName& p_receiver_func, const Variant& p_userdata);

	void add_preview_generator(const Ref<EditorResourcePreviewGenerator>& p_generator);

	EditorResourcePreview();
	~EditorResourcePreview();
};

#endif // EDITORRESOURCEPREVIEW_H

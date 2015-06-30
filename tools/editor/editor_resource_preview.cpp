#include "editor_resource_preview.h"
#include "editor_settings.h"
#include "os/file_access.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "globals.h"


Ref<Texture> EditorResourcePreviewGenerator::generate_from_path(const String& p_path) {

	RES res = ResourceLoader::load(p_path);
	if (!res.is_valid())
		return res;
	return generate(res);
}

EditorResourcePreviewGenerator::EditorResourcePreviewGenerator() {


}


EditorResourcePreview* EditorResourcePreview::singleton=NULL;


void EditorResourcePreview::_thread_func(void *ud) {

	EditorResourcePreview *erp=(EditorResourcePreview*)ud;
	erp->_thread();

}


void EditorResourcePreview::_preview_ready(const String& p_str,const Ref<Texture>& p_texture,ObjectID id,const StringName& p_func,const Variant& p_ud) {

	//print_line("preview is ready");
	preview_mutex->lock();

	Item item;
	item.order=order++;
	item.preview=p_texture;
	cache[p_str]=item;

	Object *recv = ObjectDB::get_instance(id);
	if (recv) {
		recv->call_deferred(p_func,p_str,p_texture,p_ud);
	}

	preview_mutex->unlock();
}

Ref<Texture> EditorResourcePreview::_generate_preview(const QueueItem& p_item,const String& cache_base) {

	String type = ResourceLoader::get_resource_type(p_item.path);
	//print_line("resource type is: "+type);

	if (type=="")
		return Ref<Texture>(); //could not guess type

	Ref<Texture> generated;

	for(int i=0;i<preview_generators.size();i++) {
		if (!preview_generators[i]->handles(type))
			continue;
		generated = preview_generators[i]->generate_from_path(p_item.path);

		break;
	}

	if (generated.is_valid()) {
		//print_line("was generated");
		int thumbnail_size = EditorSettings::get_singleton()->get("file_dialog/thumbnail_size");
		//wow it generated a preview... save cache
		ResourceSaver::save(cache_base+".png",generated);
		FileAccess *f=FileAccess::open(cache_base+".txt",FileAccess::WRITE);
		f->store_line(itos(thumbnail_size));
		f->store_line(itos(FileAccess::get_modified_time(p_item.path)));
		f->store_line(FileAccess::get_md5(p_item.path));
		memdelete(f);
	} else {
		//print_line("was not generated");

	}

	return generated;
}

void EditorResourcePreview::_thread() {

	//print_line("begin thread");
	while(!exit) {

		//print_line("wait for semaphore");
		preview_sem->wait();
		preview_mutex->lock();

		//print_line("blue team go");

		if (queue.size()) {



			QueueItem item = queue.front()->get();
			queue.pop_front();
			preview_mutex->unlock();

			Ref<Texture> texture;

			//print_line("pop from queue "+item.path);

			uint64_t modtime = FileAccess::get_modified_time(item.path);
			int thumbnail_size = EditorSettings::get_singleton()->get("file_dialog/thumbnail_size");

			if (cache.has(item.path)) {
				//already has it because someone loaded it, just let it know it's ready
				call_deferred("_preview_ready",item.path,cache[item.path].preview,item.id,item.function,item.userdata);

			} else {


				String temp_path=EditorSettings::get_singleton()->get_settings_path().plus_file("tmp");
				String cache_base = Globals::get_singleton()->globalize_path(item.path).md5_text();
				cache_base = temp_path.plus_file("resthumb-"+cache_base);

				//does not have it, try to load a cached thumbnail

				String file = cache_base+".txt";
				//print_line("cachetxt at "+file);
				FileAccess *f=FileAccess::open(file,FileAccess::READ);
				if (!f) {

					//print_line("generate because not cached");

					//generate
					texture=_generate_preview(item,cache_base);
				} else {

					int tsize = f->get_line().to_int64();
					uint64_t last_modtime = f->get_line().to_int64();

					bool cache_valid = true;

					if (tsize!=thumbnail_size) {
						cache_valid=false;
						memdelete(f);
					} else if (last_modtime!=modtime) {

						String last_md5 = f->get_line();
						String md5 = FileAccess::get_md5(item.path);
						memdelete(f);

						if (last_md5!=md5) {

							cache_valid=false;
						} else {
							//update modified time

							f=FileAccess::open(file,FileAccess::WRITE);
							f->store_line(itos(modtime));
							f->store_line(md5);
							memdelete(f);
						}
					} else {
						memdelete(f);
					}

					if (cache_valid) {

						texture = ResourceLoader::load(cache_base+".png","ImageTexture",true);
						if (!texture.is_valid()) {
							//well fuck
							cache_valid=false;
						}
					}

					if (!cache_valid) {

						texture=_generate_preview(item,cache_base);
					}

				}

				//print_line("notify of preview ready");
				call_deferred("_preview_ready",item.path,texture,item.id,item.function,item.userdata);

			}

		} else {
			preview_mutex->unlock();
		}

	}
}




void EditorResourcePreview::queue_resource_preview(const String& p_path, Object* p_receiver, const StringName& p_receiver_func, const Variant& p_userdata) {

	ERR_FAIL_NULL(p_receiver);
	preview_mutex->lock();
	if (cache.has(p_path)) {
		cache[p_path].order=order++;
		p_receiver->call_deferred(p_receiver_func,p_path,cache[p_path].preview,p_userdata);
		preview_mutex->unlock();
		return;

	}

	//print_line("send to thread "+p_path);
	QueueItem item;
	item.function=p_receiver_func;
	item.id=p_receiver->get_instance_ID();
	item.path=p_path;
	item.userdata=p_userdata;

	queue.push_back(item);
	preview_mutex->unlock();
	preview_sem->post();

}

void EditorResourcePreview::add_preview_generator(const Ref<EditorResourcePreviewGenerator>& p_generator) {

	preview_generators.push_back(p_generator);
}

EditorResourcePreview* EditorResourcePreview::get_singleton() {

	return singleton;
}

void EditorResourcePreview::_bind_methods() {

	ObjectTypeDB::bind_method("_preview_ready",&EditorResourcePreview::_preview_ready);
}

EditorResourcePreview::EditorResourcePreview() {
	singleton=this;
	preview_mutex = Mutex::create();
	preview_sem  = Semaphore::create();
	order=0;
	exit=false;

	thread = Thread::create(_thread_func,this);
}


EditorResourcePreview::~EditorResourcePreview()
{

	exit=true;
	preview_sem->post();
	Thread::wait_to_finish(thread);
	memdelete(thread);
	memdelete(preview_mutex);
	memdelete(preview_sem);


}


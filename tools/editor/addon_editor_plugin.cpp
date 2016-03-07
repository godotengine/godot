#include "addon_editor_plugin.h"
#include "editor_node.h"





void EditorAssetLibraryItem::configure(const String& p_title,int p_asset_id,const String& p_category,int p_category_id,const String& p_author,int p_author_id,int p_rating,const String& p_cost) {

	title->set_text(p_title);
	asset_id=p_asset_id;
	category->set_text(p_category);
	category_id=p_category_id;
	author->set_text(p_author);
	author_id=p_author_id;
	price->set_text(p_cost);

	for(int i=0;i<5;i++) {
		if (i>2)
			stars[i]->set_texture(get_icon("RatingNoStar","EditorIcons"));
		else
			stars[i]->set_texture(get_icon("RatingStar","EditorIcons"));
	}


}

void EditorAssetLibraryItem::set_image(int p_type,int p_index,const Ref<Texture>& p_image) {

	ERR_FAIL_COND(p_type!=EditorAddonLibrary::IMAGE_QUEUE_ICON);
	ERR_FAIL_COND(p_index!=0);

	icon->set_normal_texture(p_image);
}

void EditorAssetLibraryItem::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		icon->set_normal_texture(get_icon("GodotAssetDefault","EditorIcons"));
		category->add_color_override("font_color", Color(0.5,0.5,0.5) );
		author->add_color_override("font_color", Color(0.5,0.5,0.5) );

	}
}

void EditorAssetLibraryItem::_asset_clicked() {

	emit_signal("asset_selected",asset_id);
}

void EditorAssetLibraryItem::_category_clicked(){

	emit_signal("category_selected",category_id);
}
void EditorAssetLibraryItem::_author_clicked(){

	emit_signal("author_selected",author_id);

}

void EditorAssetLibraryItem::_bind_methods() {

	ObjectTypeDB::bind_method("set_image",&EditorAssetLibraryItem::set_image);
	ObjectTypeDB::bind_method("_asset_clicked",&EditorAssetLibraryItem::_asset_clicked);
	ObjectTypeDB::bind_method("_category_clicked",&EditorAssetLibraryItem::_category_clicked);
	ObjectTypeDB::bind_method("_author_clicked",&EditorAssetLibraryItem::_author_clicked);
	ADD_SIGNAL( MethodInfo("asset_selected"));
	ADD_SIGNAL( MethodInfo("category_selected"));
	ADD_SIGNAL( MethodInfo("author_selected"));


}

EditorAssetLibraryItem::EditorAssetLibraryItem() {

	Ref<StyleBoxEmpty> border;
	border.instance();
	/*border->set_default_margin(MARGIN_LEFT,5);
	border->set_default_margin(MARGIN_RIGHT,5);
	border->set_default_margin(MARGIN_BOTTOM,5);
	border->set_default_margin(MARGIN_TOP,5);*/
	add_style_override("panel",border);

	HBoxContainer *hb = memnew( HBoxContainer );
	add_child(hb);

	icon = memnew( TextureButton );
	icon->set_default_cursor_shape(CURSOR_POINTING_HAND);
	icon->connect("pressed",this,"_asset_clicked");

	hb->add_child(icon);

	VBoxContainer *vb = memnew( VBoxContainer );

	hb->add_child(vb);
	vb->set_h_size_flags(SIZE_EXPAND_FILL);

	title = memnew( LinkButton );
	title->set_text("My Awesome Addon");
	title->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	title->connect("pressed",this,"_asset_clicked");
	vb->add_child(title);


	category = memnew( LinkButton );
	category->set_text("Editor Tools");
	category->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	title->connect("pressed",this,"_category_clicked");
	vb->add_child(category);

	author = memnew( LinkButton );
	author->set_text("Johny Tolengo");
	author->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	title->connect("pressed",this,"_author_clicked");
	vb->add_child(author);

	HBoxContainer *rating_hb = memnew( HBoxContainer );
	vb->add_child(rating_hb);

	for(int i=0;i<5;i++) {
		stars[i]=memnew(TextureFrame);
		rating_hb->add_child(stars[i]);
	}
	price = memnew( Label );
	price->set_text("Free");
	vb->add_child(price);

	set_custom_minimum_size(Size2(250,100));
	set_h_size_flags(SIZE_EXPAND_FILL);

	set_stop_mouse(false);
}

//////////////////////////////////////////////////////////////////////////////


void EditorAddonLibraryItemDescription::set_image(int p_type,int p_index,const Ref<Texture>& p_image) {

	switch(p_type) {

		case EditorAddonLibrary::IMAGE_QUEUE_ICON: {

			item->call("set_image",p_type,p_index,p_image);
		} break;
		case EditorAddonLibrary::IMAGE_QUEUE_THUMBNAIL: {

			for(int i=0;i<preview_images.size();i++) {
				if (preview_images[i].id==p_index) {
					preview_images[i].button->set_icon(p_image);
				}
			}
			//item->call("set_image",p_type,p_index,p_image);
		} break;
		case EditorAddonLibrary::IMAGE_QUEUE_SCREENSHOT: {

			for(int i=0;i<preview_images.size();i++) {
				if (preview_images[i].id==p_index && preview_images[i].button->is_pressed()) {
					preview->set_texture(p_image);
				}
			}
			//item->call("set_image",p_type,p_index,p_image);
		} break;
	}
}

void EditorAddonLibraryItemDescription::_bind_methods() {
	ObjectTypeDB::bind_method(_MD("set_image"),&EditorAddonLibraryItemDescription::set_image);
}

void EditorAddonLibraryItemDescription::configure(const String& p_title,int p_asset_id,const String& p_category,int p_category_id,const String& p_author,int p_author_id,int p_rating,const String& p_cost,const String& p_description) {

	item->configure(p_title,p_asset_id,p_category,p_category_id,p_author,p_author_id,p_rating,p_cost);
	description->parse_bbcode(p_description);
	set_title(p_title);
}

void EditorAddonLibraryItemDescription::add_preview(int p_id, bool p_video,const String& p_url){

	Preview preview;
	preview.id=p_id;
	preview.video_link=p_url;
	preview.button = memnew( Button );
	preview.button->set_flat(true);
	preview.button->set_icon(get_icon("ThumbnailWait","EditorIcons"));
	preview.button->set_toggle_mode(true);
	preview_hb->add_child(preview.button);
	if (preview_images.size()==0)
		preview.button->set_pressed(true);
	preview_images.push_back(preview);
}

EditorAddonLibraryItemDescription::EditorAddonLibraryItemDescription() {

	VBoxContainer *vbox = memnew( VBoxContainer );
	add_child(vbox);
	set_child_rect(vbox);


	HBoxContainer *hbox = memnew( HBoxContainer);
	vbox->add_child(hbox);
	vbox->add_constant_override("separation",15);
	VBoxContainer *desc_vbox = memnew( VBoxContainer );
	hbox->add_child(desc_vbox);
	hbox->add_constant_override("separation",15);

	item = memnew( EditorAssetLibraryItem );

	desc_vbox->add_child(item);
	desc_vbox->set_custom_minimum_size(Size2(300,0));


	PanelContainer * desc_bg = memnew( PanelContainer );
	desc_vbox->add_child(desc_bg);
	desc_bg->set_v_size_flags(SIZE_EXPAND_FILL);

	description = memnew( RichTextLabel );
	//desc_vbox->add_child(description);
	desc_bg->add_child(description);
	desc_bg->add_style_override("panel",get_stylebox("normal","TextEdit"));

	preview = memnew( TextureFrame );
	preview->set_custom_minimum_size(Size2(640,345));
	hbox->add_child(preview);

	PanelContainer * previews_bg = memnew( PanelContainer );
	vbox->add_child(previews_bg);
	previews_bg->set_custom_minimum_size(Size2(0,85));
	previews_bg->add_style_override("panel",get_stylebox("normal","TextEdit"));

	previews = memnew( ScrollContainer );
	previews_bg->add_child(previews);
	previews->set_enable_v_scroll(false);
	previews->set_enable_h_scroll(true);
	preview_hb = memnew( HBoxContainer );
	preview_hb->set_v_size_flags(SIZE_EXPAND_FILL);

	previews->add_child(preview_hb);
	get_ok()->set_text("Install");
	get_cancel()->set_text("Close");



}


////////////////////////////////////////////////////////////////////////////////
void EditorAddonLibrary::_notification(int p_what) {

	if (p_what==NOTIFICATION_READY) {
		_api_request("api/configure");
	}

	if (p_what==NOTIFICATION_PROCESS) {


	}

}


const char* EditorAddonLibrary::sort_key[SORT_MAX]={
	"rating",
	"downloads",
	"name",
	"cost",
	"updated"
};

const char* EditorAddonLibrary::sort_text[SORT_MAX]={
	"Rating",
	"Downloads",
	"Name",
	"Cost",
	"Updated"
};


void EditorAddonLibrary::_select_author(int p_id) {

	//opemn author window
}

void EditorAddonLibrary::_select_category(int p_id){

	for(int i=0;i<categories->get_item_count();i++) {

		if (i==0)
			continue;
		int id = categories->get_item_metadata(i);
		if (id==p_id) {
			categories->select(i);
			_search();
			break;
		}
	}
}
void EditorAddonLibrary::_select_asset(int p_id){

	_api_request("api/asset","?id="+itos(p_id));

	/*
	if (description) {
		memdelete(description);
	}


	description = memnew( EditorAddonLibraryItemDescription );
	add_child(description);
	description->popup_centered_minsize();*/
}



void EditorAddonLibrary::_image_request_completed(int p_status, int p_code, const StringArray& headers, const ByteArray& p_data,int p_queue_id) {

	ERR_FAIL_COND( !image_queue.has(p_queue_id) );

	if (p_status==HTTPRequest::RESULT_SUCCESS) {


		print_line("GOT IMAGE YAY!");
		Object *obj = ObjectDB::get_instance(image_queue[p_queue_id].target);

		if (obj) {
			int len=p_data.size();
			ByteArray::Read r=p_data.read();

			Image image(r.ptr(),len);
			if (!image.empty()) {
				Ref<ImageTexture> tex;
				tex.instance();
				tex->create_from_image(image);

				obj->call("set_image",image_queue[p_queue_id].image_type,image_queue[p_queue_id].image_index,tex);
			}
		}
	} else {
		WARN_PRINTS("Error getting PNG file for asset id "+itos(image_queue[p_queue_id].asset_id));
	}

	image_queue[p_queue_id].request->queue_delete();;
	image_queue.erase(p_queue_id);

	_update_image_queue();

}

void EditorAddonLibrary::_update_image_queue() {

	int max_images=2;
	int current_images=0;

	List<int> to_delete;
	for (Map<int,ImageQueue>::Element *E=image_queue.front();E;E=E->next()) {
		if (!E->get().active && current_images<max_images) {

			String api;
			switch(E->get().image_type) {
				case IMAGE_QUEUE_ICON: api="api/icon/icon.png"; break;
				case IMAGE_QUEUE_SCREENSHOT: api="api/screenshot/screenshot.png"; break;
				case IMAGE_QUEUE_THUMBNAIL: api="api/thumbnail/thumbnail.png"; break;
			}

			print_line("REQUEST ICON FOR: "+itos(E->get().asset_id));
			Error err = E->get().request->request(host+"/"+api+"?asset_id="+itos(E->get().asset_id)+"&index="+itos(E->get().image_index));
			if (err!=OK) {
				to_delete.push_back(E->key());
			} else {
				E->get().active=true;
			}
			current_images++;
		} else if (E->get().active) {
			current_images++;
		}
	}

	while(to_delete.size()) {
		image_queue[to_delete.front()->get()].request->queue_delete();
		image_queue.erase(to_delete.front()->get());
		to_delete.pop_front();
	}
}

void EditorAddonLibrary::_request_image(ObjectID p_for,int p_asset_id,ImageType p_type,int p_image_index) {


	ImageQueue iq;
	iq.asset_id=p_asset_id;
	iq.image_index=p_image_index;
	iq.image_type=p_type;
	iq.request = memnew( HTTPRequest );

	iq.target=p_for;
	iq.queue_id=++last_queue_id;
	iq.active=false;

	iq.request->connect("request_completed",this,"_image_request_completed",varray(iq.queue_id));

	image_queue[iq.queue_id]=iq;

	add_child(iq.request);

	_update_image_queue();


}


void EditorAddonLibrary::_search(int p_page) {

	String args;

	args=String()+"?sort="+sort_key[sort->get_selected()];

	if (categories->get_selected()>0) {

		args+="&category="+itos(categories->get_item_metadata(categories->get_selected()));
	}

	if (filter->get_text()!=String()) {
		args+="&filter="+filter->get_text().http_escape();
	}

	if (p_page>0) {
		args+="&page="+itos(p_page);
	}

	_api_request("api/search",args);
}

HBoxContainer* EditorAddonLibrary::_make_pages(int p_page,int p_max_page,int p_page_len,int p_total_items,int p_current_items) {

	HBoxContainer * hbc = memnew( HBoxContainer );

	//do the mario
	int from = p_page-5;
	if (from<0)
		from=0;
	int to = from+10;
	if (to>p_max_page)
		to=p_max_page;

	Color gray = Color(0.65,0.65,0.65);

	hbc->add_spacer();
	hbc->add_constant_override("separation",10);

	LinkButton *first = memnew( LinkButton );
	first->set_text("first");
	first->add_color_override("font_color", gray );
	first->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	first->connect("pressed",this,"_search",varray(0));
	hbc->add_child(first);

	if (p_page>0) {
		LinkButton *prev = memnew( LinkButton );
		prev->set_text("prev");
		prev->add_color_override("font_color", gray );
		prev->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
		prev->connect("pressed",this,"_search",varray(p_page-1));
		hbc->add_child(prev);
	}

	for(int i=from;i<=to;i++) {

		if (i==p_page) {

			Label *current = memnew(Label);
			current->set_text(itos(i));
			hbc->add_child(current);
		} else {

			LinkButton *current = memnew( LinkButton );
			current->add_color_override("font_color", gray );
			current->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
			current->set_text(itos(i));
			current->connect("pressed",this,"_search",varray(i));

			hbc->add_child(current);

		}
	}

	if (p_page<p_max_page) {
		LinkButton *next = memnew( LinkButton );
		next->set_text("next");
		next->add_color_override("font_color", gray );
		next->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
		next->connect("pressed",this,"_search",varray(p_page+1));

		hbc->add_child(next);
	}
	LinkButton *last = memnew( LinkButton );
	last->set_text("last");
	last->add_color_override("font_color", gray );
	last->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	hbc->add_child(last);
	last->connect("pressed",this,"_search",varray(p_max_page));

	Label *totals = memnew( Label );
	totals->set_text("( "+itos(from*p_page_len)+" - "+itos(from*p_page_len+p_current_items-1)+" / "+itos(p_total_items)+" )");
	hbc->add_child(totals);

	hbc->add_spacer();

	return hbc;
}


void EditorAddonLibrary::_api_request(const String& p_request,const String& p_arguments) {

	if (requesting!=REQUESTING_NONE) {
		request->cancel_request();
	}

	current_request=p_request;
	request->request(host+"/"+p_request+p_arguments);
}



void EditorAddonLibrary::_http_request_completed(int p_status, int p_code, const StringArray& headers, const ByteArray& p_data) {


	String str;

	{
		int datalen=p_data.size();
		ByteArray::Read r = p_data.read();
		str.parse_utf8((const char*)r.ptr(),datalen);
	}

	print_line("response: "+itos(p_status)+" code: "+itos(p_code));
	Dictionary d;
	d.parse_json(str);

	print_line(Variant(d).get_construct_string());

	if (current_request=="api/configure") {

		categories->clear();
		categories->add_item("All");
		categories->set_item_metadata(0,0);
		if (d.has("categories")) {
			Array clist = d["categories"];
			for(int i=0;i<clist.size();i++) {
				Dictionary cat = clist[i];
				if (!cat.has("name") || !cat.has("id"))
					continue;
				String name=cat["name"];
				int id=cat["id"];
				categories->add_item(name);
				categories->set_item_metadata( categories->get_item_count() -1, id);
			}
		}

		_search();
	} else if (current_request=="api/search") {

		if (asset_items) {
			memdelete(asset_items);
		}

		if (asset_top_page) {
			memdelete(asset_top_page);
		}

		if (asset_bottom_page) {
			memdelete(asset_bottom_page);
		}

		int page=0;
		int pages=1;
		int page_len=10;
		int total_items=1;
		Array result;


		if (d.has("page")) {
			page=d["page"];
		}
		if (d.has("pages")) {
			pages=d["pages"];
		}
		if (d.has("page_length")) {
			page_len=d["page_length"];
		}
		if (d.has("total")) {
			total_items=d["total"];
		}
		if (d.has("result")) {
			result=d["result"];
		}

		asset_top_page = _make_pages(page,pages,page_len,total_items,result.size());
		library_vb->add_child(asset_top_page);

		asset_items = memnew( GridContainer );
		asset_items->set_columns(2);
		asset_items->add_constant_override("hseparation",10);
		asset_items->add_constant_override("vseparation",10);

		library_vb->add_child(asset_items);

		asset_bottom_page = _make_pages(page,pages,page_len,total_items,result.size());
		library_vb->add_child(asset_bottom_page);

		for(int i=0;i<result.size();i++) {

			Dictionary r = result[i];

			ERR_CONTINUE(!r.has("title"));
			ERR_CONTINUE(!r.has("asset_id"));
			ERR_CONTINUE(!r.has("author"));
			ERR_CONTINUE(!r.has("author_id"));
			ERR_CONTINUE(!r.has("category"));
			ERR_CONTINUE(!r.has("category_id"));
			ERR_CONTINUE(!r.has("rating"));
			ERR_CONTINUE(!r.has("cost"));


			EditorAssetLibraryItem *item = memnew( EditorAssetLibraryItem );
			asset_items->add_child(item);
			item->configure(r["title"],r["asset_id"],r["category"],r["category_id"],r["author"],r["author_id"],r["rating"],r["cost"]);
			item->connect("asset_selected",this,"_select_asset");
			item->connect("author_selected",this,"_select_author");
			item->connect("category_selected",this,"_category_selected");

			_request_image(item->get_instance_ID(),r["asset_id"],IMAGE_QUEUE_ICON,0);
		}
	} else if (current_request=="api/asset") {

		ERR_FAIL_COND(!d.has("info"));

		Dictionary r = d["info"];

		ERR_FAIL_COND(!r.has("title"));
		ERR_FAIL_COND(!r.has("asset_id"));
		ERR_FAIL_COND(!r.has("author"));
		ERR_FAIL_COND(!r.has("author_id"));
		ERR_FAIL_COND(!r.has("category"));
		ERR_FAIL_COND(!r.has("category_id"));
		ERR_FAIL_COND(!r.has("rating"));
		ERR_FAIL_COND(!r.has("cost"));
		ERR_FAIL_COND(!r.has("description"));

		if (description) {
			memdelete(description);
		}

		description = memnew( EditorAddonLibraryItemDescription );
		add_child(description);
		description->popup_centered_minsize();

		description->configure(r["title"],r["asset_id"],r["category"],r["category_id"],r["author"],r["author_id"],r["rating"],r["cost"],r["description"]);
		/*item->connect("asset_selected",this,"_select_asset");
		item->connect("author_selected",this,"_select_author");
		item->connect("category_selected",this,"_category_selected");*/

		_request_image(description->get_instance_ID(),r["asset_id"],IMAGE_QUEUE_ICON,0);

		if (d.has("previews")) {
			Array previews = d["previews"];

			for(int i=0;i<previews.size();i++) {


				Dictionary p=previews[i];

				ERR_CONTINUE(!p.has("id"));

				bool is_video=p.has("type") && String(p["type"])=="video";
				String video_url;
				if (is_video && p.has("link")) {
					video_url="link";
				}

				int id=p["id"];

				description->add_preview(id,is_video,video_url);

				_request_image(description->get_instance_ID(),r["asset_id"],IMAGE_QUEUE_THUMBNAIL,id);
				if (i==0) {
					_request_image(description->get_instance_ID(),r["asset_id"],IMAGE_QUEUE_SCREENSHOT,id);
				}

			}
		}
	}

}

void EditorAddonLibrary::_bind_methods() {

	ObjectTypeDB::bind_method("_http_request_completed",&EditorAddonLibrary::_http_request_completed);
	ObjectTypeDB::bind_method("_select_asset",&EditorAddonLibrary::_select_asset);
	ObjectTypeDB::bind_method("_select_author",&EditorAddonLibrary::_select_author);
	ObjectTypeDB::bind_method("_select_category",&EditorAddonLibrary::_select_category);
	ObjectTypeDB::bind_method("_image_request_completed",&EditorAddonLibrary::_image_request_completed);
	ObjectTypeDB::bind_method("_search",&EditorAddonLibrary::_search,DEFVAL(0));

}

EditorAddonLibrary::EditorAddonLibrary() {

	tabs = memnew( TabContainer );
	tabs->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(tabs);

	installed = memnew( EditorPluginSettings );
	installed->set_name("Installed");
	tabs->add_child(installed);

	Ref<StyleBoxEmpty> border;
	border.instance();
	border->set_default_margin(MARGIN_LEFT,15);
	border->set_default_margin(MARGIN_RIGHT,15);
	border->set_default_margin(MARGIN_BOTTOM,15);
	border->set_default_margin(MARGIN_TOP,15);

	PanelContainer *margin_panel = memnew( PanelContainer );

	margin_panel->set_name("Online");
	margin_panel->add_style_override("panel",border);
	tabs->add_child(margin_panel);

	VBoxContainer *library_main = memnew( VBoxContainer );

	margin_panel->add_child(library_main);


	HBoxContainer *search_hb = memnew( HBoxContainer );

	library_main->add_child(search_hb);
	library_main->add_constant_override("separation",20);

	search_hb->add_child( memnew( Label("Search: ")));
	filter =memnew( LineEdit );
	search_hb->add_child(filter);
	filter->set_h_size_flags(SIZE_EXPAND_FILL);
	filter->connect("text_entered",this,"_search");
	search = memnew( Button("Search"));
	search->connect("pressed",this,"_search");
	search_hb->add_child(search);
	library_vb->add_child(search_hb);

	HBoxContainer *search_hb2 = memnew( HBoxContainer );
	library_main->add_child(search_hb2);

	search_hb2->add_child( memnew( Label("Sort: ")));
	sort = memnew( OptionButton );
	for(int i=0;i<SORT_MAX;i++) {
		sort->add_item(sort_text[i]);
	}

	search_hb2->add_child(sort);

	sort->set_h_size_flags(SIZE_EXPAND_FILL);

	reverse = memnew( CheckBox);
	reverse->set_text("Reverse");
	search_hb2->add_child(reverse);

	search_hb2->add_child(memnew(VSeparator));

	//search_hb2->add_spacer();

	search_hb2->add_child( memnew( Label("Category: ")));
	categories = memnew( OptionButton );
	categories->add_item("All");
	search_hb2->add_child(categories);
	categories->set_h_size_flags(SIZE_EXPAND_FILL);
	//search_hb2->add_spacer();

	search_hb2->add_child(memnew(VSeparator));

	search_hb2->add_child( memnew( Label("Site: ")));
	repository = memnew( OptionButton );

	repository->add_item("Godot");
	search_hb2->add_child(repository);
	repository->set_h_size_flags(SIZE_EXPAND_FILL);

	/////////

	PanelContainer * library_scroll_bg = memnew( PanelContainer );
	library_main->add_child(library_scroll_bg);
	library_scroll_bg->add_style_override("panel",get_stylebox("normal","TextEdit"));
	library_scroll_bg->set_v_size_flags(SIZE_EXPAND_FILL);

	library_scroll = memnew( ScrollContainer );
	library_scroll->set_enable_v_scroll(true);
	library_scroll->set_enable_h_scroll(false);

	library_scroll_bg->add_child(library_scroll);


	Ref<StyleBoxEmpty> border2;
	border2.instance();
	border2->set_default_margin(MARGIN_LEFT,15);
	border2->set_default_margin(MARGIN_RIGHT,35);
	border2->set_default_margin(MARGIN_BOTTOM,15);
	border2->set_default_margin(MARGIN_TOP,15);


	PanelContainer * library_vb_border = memnew( PanelContainer );
	library_scroll->add_child(library_vb_border);
	library_vb_border->add_style_override("panel",border2);
	library_vb_border->set_h_size_flags(SIZE_EXPAND_FILL);
	library_vb_border->set_stop_mouse(false);



	library_vb = memnew( VBoxContainer );
	library_vb->set_h_size_flags(SIZE_EXPAND_FILL);

	library_vb_border->add_child(library_vb);
//	margin_panel->set_stop_mouse(false);

	asset_top_page = memnew( HBoxContainer );
	library_vb->add_child(asset_top_page);

	asset_items = memnew( GridContainer );
	asset_items->set_columns(2);
	asset_items->add_constant_override("hseparation",10);
	asset_items->add_constant_override("vseparation",10);

	library_vb->add_child(asset_items);

	asset_bottom_page = memnew( HBoxContainer );
	library_vb->add_child(asset_bottom_page);

	request = memnew( HTTPRequest );
	add_child(request);
	request->connect("request_completed",this,"_http_request_completed");

	last_queue_id=0;

	library_vb->add_constant_override("separation",20);

	description = NULL;

	host="http://localhost:8000";
}


///////


void AddonEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		addon_library->show();
	} else {

		addon_library->hide();
	}

}

AddonEditorPlugin::AddonEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	addon_library = memnew( EditorAddonLibrary );
	addon_library->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor->get_viewport()->add_child(addon_library);
	addon_library->set_area_as_parent_rect();
	addon_library->hide();

}

AddonEditorPlugin::~AddonEditorPlugin() {

}

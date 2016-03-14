#include "addon_editor_plugin.h"
#include "editor_node.h"
#include "editor_settings.h"




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
			icon=p_image;
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
	ObjectTypeDB::bind_method(_MD("_link_click"),&EditorAddonLibraryItemDescription::_link_click);

}

void EditorAddonLibraryItemDescription::_link_click(const String& p_url) {

	ERR_FAIL_COND(!p_url.begins_with("http"));
	OS::get_singleton()->shell_open(p_url);
}

void EditorAddonLibraryItemDescription::configure(const String& p_title,int p_asset_id,const String& p_category,int p_category_id,const String& p_author,int p_author_id,int p_rating,const String& p_cost,const String& p_version,const String& p_description,const String& p_download_url,const String& p_browse_url) {

	asset_id=p_asset_id;
	title=p_title;
	download_url=p_download_url;
	item->configure(p_title,p_asset_id,p_category,p_category_id,p_author,p_author_id,p_rating,p_cost);
	description->clear();
	description->add_text("Version: "+p_version+"\n");
	description->add_text("Contents: ");
	description->push_meta(p_browse_url);
	description->add_text("View Files");
	description->pop();
	description->add_text("\nDescription:\n\n");
	description->append_bbcode(p_description);
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
	description->connect("meta_clicked",this,"_link_click");
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
///////////////////////////////////////////////////////////////////////////////////

void EditorAddonLibraryItemDownload::_http_download_completed(int p_status, int p_code, const StringArray& headers, const ByteArray& p_data) {


	String error_text;

	switch(p_status) {

		case HTTPRequest::RESULT_CANT_RESOLVE: {
			error_text=("Can't resolve hostname: "+host);
			status->set_text("Can't resolve.");
		} break;
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH: {
			error_text=("Connection error, please try again.");
			status->set_text("Can't connect.");
		} break;
		case HTTPRequest::RESULT_SSL_HANDSHAKE_ERROR:
		case HTTPRequest::RESULT_CANT_CONNECT: {
			error_text=("Can't connect to host: "+host);
			status->set_text("Can't connect.");
		} break;
		case HTTPRequest::RESULT_NO_RESPONSE: {
			error_text=("No response from host: "+host);
			status->set_text("No response.");
		} break;
		case HTTPRequest::RESULT_REQUEST_FAILED: {
			error_text=("Request failed, return code: "+itos(p_code));
			status->set_text("Req. Failed.");
		} break;
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: {
			error_text=("Request failed, too many redirects");
			status->set_text("Redirect Loop.");
		} break;
		default: {
			if (p_code!=200) {
				error_text=("Request failed, return code: "+itos(p_code));
				status->set_text("Failed: "+itos(p_code));
			} else {

				//all good
			}
		} break;

	}

	if (error_text!=String()) {
		download_error->set_text("Asset Download Error:\n"+error_text);
		download_error->popup_centered_minsize();
		return;

	}

	progress->set_max( download->get_body_size() );
	progress->set_val(download->get_downloaded_bytes());

	print_line("max: "+itos(download->get_body_size())+" bytes: "+itos(download->get_downloaded_bytes()));
	install->set_disabled(false);

	status->set_text("Success!");
	set_process(false);
}


void EditorAddonLibraryItemDownload::configure(const String& p_title,int p_asset_id,const Ref<Texture>& p_preview, const String& p_download_url) {

	title->set_text(p_title);
	icon->set_texture(p_preview);
	asset_id=p_asset_id;
	if (!p_preview.is_valid())
		icon->set_texture(get_icon("GodotAssetDefault","EditorIcons"));

	host=p_download_url;
	set_process(true);
	download->set_download_file(EditorSettings::get_singleton()->get_settings_path().plus_file("tmp").plus_file("tmp_asset_"+itos(p_asset_id))+".zip");
	Error err = download->request(p_download_url);
	ERR_FAIL_COND(err!=OK);
	asset_installer->connect("confirmed",this,"_close");
	dismiss->set_normal_texture(get_icon("Close","EditorIcons"));


}


void EditorAddonLibraryItemDownload::_notification(int p_what) {

	if (p_what==NOTIFICATION_PROCESS) {

		progress->set_max( download->get_body_size() );
		progress->set_val(download->get_downloaded_bytes());

		int cstatus = download->get_http_client_status();
		if (cstatus!=prev_status) {
			switch(cstatus) {

				case HTTPClient::STATUS_RESOLVING: {
					status->set_text("Resolving..");
				} break;
				case HTTPClient::STATUS_CONNECTING: {
					status->set_text("Connecting..");
				} break;
				case HTTPClient::STATUS_REQUESTING: {
					status->set_text("Requesting..");
				} break;
				case HTTPClient::STATUS_BODY: {
					status->set_text("Downloading..");
				} break;
				default: {}
			}
			prev_status=cstatus;
		}

	}
}
void EditorAddonLibraryItemDownload::_close() {

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->remove(download->get_download_file()); //clean up removed file
	memdelete(da);
	queue_delete();
}

void EditorAddonLibraryItemDownload::_install() {

	String file = download->get_download_file();
	asset_installer->open(file,1);
}

void EditorAddonLibraryItemDownload::_bind_methods() {

	ObjectTypeDB::bind_method("_http_download_completed",&EditorAddonLibraryItemDownload::_http_download_completed);
	ObjectTypeDB::bind_method("_install",&EditorAddonLibraryItemDownload::_install);
	ObjectTypeDB::bind_method("_close",&EditorAddonLibraryItemDownload::_close);

}

EditorAddonLibraryItemDownload::EditorAddonLibraryItemDownload() {

	HBoxContainer *hb = memnew( HBoxContainer);
	add_child(hb);
	icon = memnew( TextureFrame );
	hb->add_child(icon);

	VBoxContainer *vb = memnew( VBoxContainer );
	hb->add_child(vb);
	vb->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *title_hb = memnew( HBoxContainer);
	vb->add_child(title_hb);
	title = memnew( Label );
	title_hb->add_child(title);
	title->set_h_size_flags(SIZE_EXPAND_FILL);

	dismiss = memnew( TextureButton );
	dismiss->connect("pressed",this,"_close");
	title_hb->add_child(dismiss);

	title->set_clip_text(true);

	vb->add_spacer();

	status = memnew (Label("Idle"));
	vb->add_child(status);
	status->add_color_override("font_color", Color(0.5,0.5,0.5) );
	progress = memnew( ProgressBar );
	vb->add_child(progress);



	HBoxContainer *hb2 = memnew( HBoxContainer );
	vb->add_child(hb2);
	hb2->add_spacer();

	install = memnew( Button );
	install->set_text("Install");
	install->set_disabled(true);
	install->connect("pressed",this,"_install");

	hb2->add_child(install);
	set_custom_minimum_size(Size2(250,0));

	download = memnew( HTTPRequest );
	add_child(download);
	download->connect("request_completed",this,"_http_download_completed");

	download_error = memnew( AcceptDialog );
	add_child(download_error);
	download_error->set_title("Download Error");

	asset_installer = memnew( EditorAssetInstaller );
	add_child(asset_installer);

	prev_status=-1;


}



////////////////////////////////////////////////////////////////////////////////
void EditorAddonLibrary::_notification(int p_what) {

	if (p_what==NOTIFICATION_READY) {
		TextureFrame *tf = memnew(TextureFrame);
		tf->set_texture(get_icon("Error","EditorIcons"));
		error_hb->add_child(tf);
		error_label->raise();

		_api_request("api/configure");
	}

	if (p_what==NOTIFICATION_PROCESS) {

		HTTPClient::Status s = request->get_http_client_status();
		bool visible = s!=HTTPClient::STATUS_DISCONNECTED;

		if (visible != !load_status->is_hidden()) {
			load_status->set_hidden(!visible);
		}

		if (visible) {
			switch(s) {

				case HTTPClient::STATUS_RESOLVING: {
					load_status->set_val(0.1);
				} break;
				case HTTPClient::STATUS_CONNECTING: {
					load_status->set_val(0.2);
				} break;
				case HTTPClient::STATUS_REQUESTING: {
					load_status->set_val(0.3);
				} break;
				case HTTPClient::STATUS_BODY: {
					load_status->set_val(0.4);
				} break;
				default: {}

			}
		}

		bool no_downloads = downloads_hb->get_child_count()==0;
		if (no_downloads != downloads_scroll->is_hidden()) {
			downloads_scroll->set_hidden(no_downloads);
		}
	}

}


void EditorAddonLibrary::_install_asset() {

	ERR_FAIL_COND(!description);

	for(int i=0;i<downloads_hb->get_child_count();i++) {

		EditorAddonLibraryItemDownload *d  = downloads_hb->get_child(i)->cast_to<EditorAddonLibraryItemDownload>();
		if (d && d->get_asset_id() == description->get_asset_id()) {

			EditorNode::get_singleton()->show_warning("Download for this asset is already in progress!");
			return;
		}
	}


	EditorAddonLibraryItemDownload * download = memnew( EditorAddonLibraryItemDownload );
	downloads_hb->add_child(download);
	download->configure(description->get_title(),description->get_asset_id(),description->get_preview_icon(),description->get_download_url());

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

	error_hb->hide();
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

	bool error_abort=true;

	switch(p_status) {

		case HTTPRequest::RESULT_CANT_RESOLVE: {
			error_label->set_text("Can't resolve hostname: "+host);
		} break;
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH: {
			error_label->set_text("Connection error, please try again.");
		} break;
		case HTTPRequest::RESULT_SSL_HANDSHAKE_ERROR:
		case HTTPRequest::RESULT_CANT_CONNECT: {
			error_label->set_text("Can't connect to host: "+host);
		} break;
		case HTTPRequest::RESULT_NO_RESPONSE: {
			error_label->set_text("No response from host: "+host);
		} break;
		case HTTPRequest::RESULT_REQUEST_FAILED: {
			error_label->set_text("Request failed, return code: "+itos(p_code));
		} break;
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: {
			error_label->set_text("Request failed, too many redirects");

		} break;
		default: {
			if (p_code!=200) {
				error_label->set_text("Request failed, return code: "+itos(p_code));
			} else {

				error_abort=false;
			}
		} break;

	}


	if (error_abort) {
		error_hb->show();
		return;
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

		r["download_url"]="https://github.com/reduz/godot-test-addon/archive/master.zip";
		r["browse_url"]="https://github.com/reduz/godot-test-addon";
		r["version"]="1.1";

		ERR_FAIL_COND(!r.has("title"));
		ERR_FAIL_COND(!r.has("asset_id"));
		ERR_FAIL_COND(!r.has("author"));
		ERR_FAIL_COND(!r.has("author_id"));
		ERR_FAIL_COND(!r.has("version"));
		ERR_FAIL_COND(!r.has("category"));
		ERR_FAIL_COND(!r.has("category_id"));
		ERR_FAIL_COND(!r.has("rating"));
		ERR_FAIL_COND(!r.has("cost"));
		ERR_FAIL_COND(!r.has("description"));
		ERR_FAIL_COND(!r.has("download_url"));
		ERR_FAIL_COND(!r.has("browse_url"));


		if (description) {
			memdelete(description);
		}

		description = memnew( EditorAddonLibraryItemDescription );
		add_child(description);
		description->popup_centered_minsize();
		description->connect("confirmed",this,"_install_asset");

		description->configure(r["title"],r["asset_id"],r["category"],r["category_id"],r["author"],r["author_id"],r["rating"],r["cost"],r["version"],r["description"],r["download_url"],r["browse_url"]);
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


void EditorAddonLibrary::_asset_file_selected(const String& p_file) {

	if (asset_installer) {
		memdelete( asset_installer );
		asset_installer=NULL;
	}

	asset_installer = memnew( EditorAssetInstaller );
	add_child(asset_installer);
	asset_installer->open(p_file);


}

void EditorAddonLibrary::_asset_open() {

	asset_open->popup_centered_ratio();
}

void EditorAddonLibrary::_manage_plugins() {

	ProjectSettings::get_singleton()->popup_project_settings();
	ProjectSettings::get_singleton()->set_plugins_page();
}



void EditorAddonLibrary::_bind_methods() {

	ObjectTypeDB::bind_method("_http_request_completed",&EditorAddonLibrary::_http_request_completed);
	ObjectTypeDB::bind_method("_select_asset",&EditorAddonLibrary::_select_asset);
	ObjectTypeDB::bind_method("_select_author",&EditorAddonLibrary::_select_author);
	ObjectTypeDB::bind_method("_select_category",&EditorAddonLibrary::_select_category);
	ObjectTypeDB::bind_method("_image_request_completed",&EditorAddonLibrary::_image_request_completed);
	ObjectTypeDB::bind_method("_search",&EditorAddonLibrary::_search,DEFVAL(0));
	ObjectTypeDB::bind_method("_install_asset",&EditorAddonLibrary::_install_asset);
	ObjectTypeDB::bind_method("_manage_plugins",&EditorAddonLibrary::_manage_plugins);
	ObjectTypeDB::bind_method("_asset_open",&EditorAddonLibrary::_asset_open);
	ObjectTypeDB::bind_method("_asset_file_selected",&EditorAddonLibrary::_asset_file_selected);

}

EditorAddonLibrary::EditorAddonLibrary() {


	Ref<StyleBoxEmpty> border;
	border.instance();
	border->set_default_margin(MARGIN_LEFT,15);
	border->set_default_margin(MARGIN_RIGHT,15);
	border->set_default_margin(MARGIN_BOTTOM,5);
	border->set_default_margin(MARGIN_TOP,5);

	add_style_override("panel",border);

	VBoxContainer *library_main = memnew( VBoxContainer );

	add_child(library_main);

	HBoxContainer *search_hb = memnew( HBoxContainer );

	library_main->add_child(search_hb);
	library_main->add_constant_override("separation",10);



	search_hb->add_child( memnew( Label("Search: ")));
	filter =memnew( LineEdit );
	search_hb->add_child(filter);
	filter->set_h_size_flags(SIZE_EXPAND_FILL);
	filter->connect("text_entered",this,"_search");
	search = memnew( Button("Search"));
	search->connect("pressed",this,"_search");
	search_hb->add_child(search);

	search_hb->add_child(memnew( VSeparator ));

	Button * open_asset = memnew( Button );
	open_asset->set_text("Import");
	search_hb->add_child(open_asset);
	open_asset->connect("pressed",this,"_asset_open");

	Button * plugins = memnew( Button );
	plugins->set_text("Plugins");
	search_hb->add_child(plugins);
	plugins->connect("pressed",this,"_manage_plugins");


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

	load_status = memnew( ProgressBar );
	load_status->set_min(0);
	load_status->set_max(1);
	load_status->set_step(0.001);
	library_main->add_child(load_status);

	error_hb = memnew( HBoxContainer );
	library_main->add_child(error_hb);
	error_label = memnew( Label );
	error_label->add_color_override("color",Color(1,0.4,0.3));
	error_hb->add_child(error_label);

	description = NULL;

	//host="http://localhost:8000";
	host="http://godotengine.org/addonlib";
	set_process(true);

	downloads_scroll = memnew( ScrollContainer );
	downloads_scroll->set_enable_h_scroll(true);
	downloads_scroll->set_enable_v_scroll(false);
	library_main->add_child(downloads_scroll);
	downloads_hb = memnew( HBoxContainer );
	downloads_scroll->add_child(downloads_hb);

	asset_open = memnew( EditorFileDialog );

	asset_open->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	asset_open->add_filter("*.zip ; Assets ZIP File");
	asset_open->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	add_child(asset_open);
	asset_open->connect("file_selected",this,"_asset_file_selected");

	asset_installer=NULL;

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

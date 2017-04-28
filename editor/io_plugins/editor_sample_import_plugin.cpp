/*************************************************************************/
/*  editor_sample_import_plugin.cpp                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_sample_import_plugin.h"

#include "editor/editor_dir_dialog.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/property_editor.h"
#include "io/marshalls.h"
#include "io/resource_saver.h"
#include "os/file_access.h"

#if 0

class _EditorSampleImportOptions : public Object {

	GDCLASS(_EditorSampleImportOptions,Object);
public:

	enum CompressMode {
		COMPRESS_MODE_DISABLED,
		COMPRESS_MODE_RAM,
		COMPRESS_MODE_DISK
	};

	enum CompressBitrate {
		COMPRESS_64,
		COMPRESS_96,
		COMPRESS_128,
		COMPRESS_192
	};

	bool force_8_bit;
	bool force_mono;
	bool force_rate;
	float force_rate_hz;

	bool edit_trim;
	bool edit_normalize;
	bool edit_loop;

	CompressMode compress_mode;
	CompressBitrate compress_bitrate;


	bool _set(const StringName& p_name, const Variant& p_value) {

		String n = p_name;
		if (n=="force/8_bit")
			force_8_bit=p_value;
		else if (n=="force/mono")
			force_mono=p_value;
		else if (n=="force/max_rate")
			force_rate=p_value;
		else if (n=="force/max_rate_hz")
			force_rate_hz=p_value;
		else if (n=="edit/trim")
			edit_trim=p_value;
		else if (n=="edit/normalize")
			edit_normalize=p_value;
		else if (n=="edit/loop")
			edit_loop=p_value;
		else if (n=="compress/mode")
			compress_mode=CompressMode(int(p_value));
		else if (n=="compress/bitrate")
			compress_bitrate=CompressBitrate(int(p_value));
		else
			return false;

		return true;

	}

	bool _get(const StringName& p_name,Variant &r_ret) const{

		String n = p_name;
		if (n=="force/8_bit")
			r_ret=force_8_bit;
		else if (n=="force/mono")
			r_ret=force_mono;
		else if (n=="force/max_rate")
			r_ret=force_rate;
		else if (n=="force/max_rate_hz")
			r_ret=force_rate_hz;
		else if (n=="edit/trim")
			r_ret=edit_trim;
		else if (n=="edit/normalize")
			r_ret=edit_normalize;
		else if (n=="edit/loop")
			r_ret=edit_loop;
		else if (n=="compress/mode")
			r_ret=compress_mode;
		else if (n=="compress/bitrate")
			r_ret=compress_bitrate;
		else
			return false;

		return true;

	}
	void _get_property_list( List<PropertyInfo> *p_list) const{

		p_list->push_back(PropertyInfo(Variant::BOOL,"force/8_bit"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"force/mono"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"force/max_rate"));
		p_list->push_back(PropertyInfo(Variant::REAL,"force/max_rate_hz",PROPERTY_HINT_EXP_RANGE,"11025,192000,1"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"edit/trim"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"edit/normalize"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"edit/loop"));
		p_list->push_back(PropertyInfo(Variant::INT,"compress/mode",PROPERTY_HINT_ENUM,"Disabled,RAM (Ima-ADPCM)"));
		//p_list->push_back(PropertyInfo(Variant::INT,"compress/bitrate",PROPERTY_HINT_ENUM,"64,96,128,192"));


	}


	static void _bind_methods() {


		ADD_SIGNAL( MethodInfo("changed"));
	}


	_EditorSampleImportOptions() {

		force_8_bit=false;
		force_mono=false;
		force_rate=true;
		force_rate_hz=44100;

		edit_trim=true;
		edit_normalize=true;
		edit_loop=false;

		compress_mode=COMPRESS_MODE_RAM;
		compress_bitrate=COMPRESS_128;
	}


};

class EditorSampleImportDialog : public ConfirmationDialog {

	GDCLASS(EditorSampleImportDialog,ConfirmationDialog);

	EditorSampleImportPlugin *plugin;

	LineEdit *import_path;
	LineEdit *save_path;
	EditorFileDialog *file_select;
	EditorDirDialog *save_select;
	ConfirmationDialog *error_dialog;
	PropertyEditor *option_editor;

	_EditorSampleImportOptions *options;


public:

	void _choose_files(const Vector<String>& p_path) {

		String files;
		for(int i=0;i<p_path.size();i++) {

			if (i>0)
				files+=",";
			files+=p_path[i];
		}
		/*
		if (p_path.size()) {
			String srctex=p_path[0];
			String ipath = EditorImportDB::get_singleton()->find_source_path(srctex);

			if (ipath!="")
				save_path->set_text(ipath.get_base_dir());
		}*/
		import_path->set_text(files);

	}
	void _choose_save_dir(const String& p_path) {

		save_path->set_text(p_path);
	}

	void _browse() {

		file_select->popup_centered_ratio();
	}

	void _browse_target() {

		save_select->popup_centered_ratio();

	}


	void popup_import(const String& p_path) {

		popup_centered(Size2(400,400)*EDSCALE);
		if (p_path!="") {

			Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_path);
			ERR_FAIL_COND(!rimd.is_valid());

			save_path->set_text(p_path.get_base_dir());
			List<String> opts;
			rimd->get_options(&opts);
			for(List<String>::Element *E=opts.front();E;E=E->next()) {

				options->_set(E->get(),rimd->get_option(E->get()));
			}

			String src = "";
			for(int i=0;i<rimd->get_source_count();i++) {
				if (i>0)
					src+=",";
				src+=EditorImportPlugin::expand_source_path(rimd->get_source_path(i));
			}
			import_path->set_text(src);
		}
	}


	void _import() {

		Vector<String> samples = import_path->get_text().split(",");

		if (samples.size()==0) {
			error_dialog->set_text(TTR("No samples to import!"));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
		}

		if (save_path->get_text().strip_edges()=="") {
			error_dialog->set_text(TTR("Target path is empty."));
			error_dialog->popup_centered_minsize();
			return;
		}

		if (!save_path->get_text().begins_with("res://")) {
			error_dialog->set_text(TTR("Target path must be a complete resource path."));
			error_dialog->popup_centered_minsize();
			return;
		}

		if (!DirAccess::exists(save_path->get_text())) {
			error_dialog->set_text(TTR("Target path must exist."));
			error_dialog->popup_centered_minsize();
			return;
		}

		for(int i=0;i<samples.size();i++) {

			Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );

			List<PropertyInfo> pl;
			options->_get_property_list(&pl);
			for(List<PropertyInfo>::Element *E=pl.front();E;E=E->next()) {

				Variant v;
				String opt=E->get().name;
				options->_get(opt,v);
				imd->set_option(opt,v);

			}

			imd->add_source(EditorImportPlugin::validate_source_path(samples[i]));

			String dst = save_path->get_text();
			if (dst=="") {
				error_dialog->set_text(TTR("Save path is empty!"));
				error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			}

			dst = dst.plus_file(samples[i].get_file().get_basename()+".smp");

			plugin->import(dst,imd);
		}

		hide();

	}


	void _notification(int p_what) {


		if (p_what==NOTIFICATION_ENTER_TREE) {

			option_editor->edit(options);
		}
	}

	static void _bind_methods() {


		ClassDB::bind_method("_choose_files",&EditorSampleImportDialog::_choose_files);
		ClassDB::bind_method("_choose_save_dir",&EditorSampleImportDialog::_choose_save_dir);
		ClassDB::bind_method("_import",&EditorSampleImportDialog::_import);
		ClassDB::bind_method("_browse",&EditorSampleImportDialog::_browse);
		ClassDB::bind_method("_browse_target",&EditorSampleImportDialog::_browse_target);
		//ADD_SIGNAL( MethodInfo("imported",PropertyInfo(Variant::OBJECT,"scene")) );
	}

	EditorSampleImportDialog(EditorSampleImportPlugin *p_plugin) {

		plugin=p_plugin;


		set_title(TTR("Import Audio Samples"));

		VBoxContainer *vbc = memnew( VBoxContainer );
		add_child(vbc);
		//set_child_rect(vbc);


		HBoxContainer *hbc = memnew( HBoxContainer );
		vbc->add_margin_child(TTR("Source Sample(s):"),hbc);

		import_path = memnew( LineEdit );
		import_path->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(import_path);

		Button * import_choose = memnew( Button );
		import_choose->set_text(" .. ");
		hbc->add_child(import_choose);

		import_choose->connect("pressed", this,"_browse");

		hbc = memnew( HBoxContainer );
		vbc->add_margin_child(TTR("Target Path:"),hbc);

		save_path = memnew( LineEdit );
		save_path->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(save_path);

		Button * save_choose = memnew( Button );
		save_choose->set_text(" .. ");
		hbc->add_child(save_choose);

		save_choose->connect("pressed", this,"_browse_target");

		file_select = memnew(EditorFileDialog);
		file_select->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		add_child(file_select);
		file_select->set_mode(EditorFileDialog::MODE_OPEN_FILES);
		file_select->connect("files_selected", this,"_choose_files");
		file_select->add_filter("*.wav ; MS Waveform");
		save_select = memnew(	EditorDirDialog );
		add_child(save_select);

		//save_select->set_mode(EditorFileDialog::MODE_OPEN_DIR);
		save_select->connect("dir_selected", this,"_choose_save_dir");

		get_ok()->connect("pressed", this,"_import");
		get_ok()->set_text(TTR("Import"));


		error_dialog = memnew ( ConfirmationDialog );
		add_child(error_dialog);
		error_dialog->get_ok()->set_text(TTR("Accept"));
		//error_dialog->get_cancel()->hide();

		set_hide_on_ok(false);
		options = memnew( _EditorSampleImportOptions );

		option_editor = memnew( PropertyEditor );
		option_editor->hide_top_label();
		vbc->add_margin_child(TTR("Options:"),option_editor,true);
	}

	~EditorSampleImportDialog() {
		memdelete(options);
	}

};


String EditorSampleImportPlugin::get_name() const {

	return "sample";
}
String EditorSampleImportPlugin::get_visible_name() const{

	return TTR("Audio Sample");
}
void EditorSampleImportPlugin::import_dialog(const String& p_from){

	dialog->popup_import(p_from);
}
Error EditorSampleImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from){

	ERR_FAIL_COND_V(p_from->get_source_count()!=1,ERR_INVALID_PARAMETER);

	Ref<ResourceImportMetadata> from=p_from;

	String src_path=EditorImportPlugin::expand_source_path(from->get_source_path(0));
	Ref<Sample> smp = ResourceLoader::load(src_path);
	ERR_FAIL_COND_V(smp.is_null(),ERR_CANT_OPEN);


	float rate = smp->get_mix_rate();
	bool is16 = smp->get_format()==Sample::FORMAT_PCM16;
	int chans = smp->is_stereo()?2:1;
	int len = smp->get_length();
	Sample::LoopFormat loop= smp->get_loop_format();
	int loop_beg = smp->get_loop_begin();
	int loop_end = smp->get_loop_end();

	print_line("Input Sample: ");
	print_line("\tlen: "+itos(len));
	print_line("\tchans: "+itos(chans));
	print_line("\t16bits: "+itos(is16));
	print_line("\trate: "+itos(rate));
	print_line("\tloop: "+itos(loop));
	print_line("\tloop begin: "+itos(loop_beg));
	print_line("\tloop end: "+itos(loop_end));
	Vector<float> data;
	data.resize(len*chans);

	{
		PoolVector<uint8_t> src_data = smp->get_data();
		PoolVector<uint8_t>::Read sr = src_data.read();


		for(int i=0;i<len*chans;i++) {

			float s=0;
			if (is16) {

				int16_t i16 = decode_uint16(&sr[i*2]);
				s=i16/32767.0;
			} else {

				int8_t i8 = sr[i];
				s=i8/127.0;
			}
			data[i]=s;
		}
	}

	//apply frequency limit

	bool limit_rate = from->get_option("force/max_rate");
	int limit_rate_hz = from->get_option("force/max_rate_hz");
	if (limit_rate && rate > limit_rate_hz) {
		//resampleeee!!!
		int new_data_len = len * limit_rate_hz / rate;
		Vector<float> new_data;
		new_data.resize( new_data_len * chans );
		for(int c=0;c<chans;c++) {

			for(int i=0;i<new_data_len;i++) {

				//simple cubic interpolation should be enough.
				float pos = float(i) * len / new_data_len;
				float mu = pos-Math::floor(pos);
				int ipos = int(Math::floor(pos));

				float y0=data[MAX(0,ipos-1)*chans+c];
				float y1=data[ipos*chans+c];
				float y2=data[MIN(len-1,ipos+1)*chans+c];
				float y3=data[MIN(len-1,ipos+2)*chans+c];

				float mu2 = mu*mu;
				float a0 = y3 - y2 - y0 + y1;
				float a1 = y0 - y1 - a0;
				float a2 = y2 - y0;
				float a3 = y1;

				float res=(a0*mu*mu2+a1*mu2+a2*mu+a3);

				new_data[i*chans+c]=res;
			}
		}

		if (loop) {

			loop_beg=loop_beg*new_data_len/len;
			loop_end=loop_end*new_data_len/len;
		}
		data=new_data;
		rate=limit_rate_hz;
		len=new_data_len;
	}


	bool normalize = from->get_option("edit/normalize");

	if (normalize) {

		float max=0;
		for(int i=0;i<data.size();i++) {

			float amp = Math::abs(data[i]);
			if (amp>max)
				max=amp;
		}

		if (max>0) {

			float mult=1.0/max;
			for(int i=0;i<data.size();i++) {

				data[i]*=mult;
			}

		}
	}

	bool trim = from->get_option("edit/trim");

	if (trim && !loop) {

		int first=0;
		int last=(len*chans)-1;
		bool found=false;
		float limit = Math::db2linear((float)-30);
		for(int i=0;i<data.size();i++) {
			float amp = Math::abs(data[i]);

			if (!found && amp > limit) {
				first=i;
				found=true;
			}

			if (found && amp > limit) {
				last=i;
			}
		}

		first/=chans;
		last/=chans;

		if (first<last) {

			Vector<float> new_data;
			new_data.resize((last-first+1)*chans);
			for(int i=first*chans;i<=last*chans;i++) {
				new_data[i-first*chans]=data[i];
			}

			data=new_data;
			len=data.size()/chans;
		}

	}

	bool make_loop = from->get_option("edit/loop");

	if (make_loop && !loop) {

		loop=Sample::LOOP_FORWARD;
		loop_beg=0;
		loop_end=len;
	}

	int compression = from->get_option("compress/mode");
	bool force_mono = from->get_option("force/mono");


	if (force_mono && chans==2) {

		Vector<float> new_data;
		new_data.resize(data.size()/2);
		for(int i=0;i<len;i++) {
			new_data[i]=(data[i*2+0]+data[i*2+1])/2.0;
		}

		data=new_data;
		chans=1;
	}

	bool force_8_bit = from->get_option("force/8_bit");
	if (force_8_bit) {

		is16=false;
	}


	PoolVector<uint8_t> dst_data;
	Sample::Format dst_format;

	if ( compression == _EditorSampleImportOptions::COMPRESS_MODE_RAM) {

		dst_format=Sample::FORMAT_IMA_ADPCM;
		if (chans==1) {
			_compress_ima_adpcm(data,dst_data);
		} else {

			print_line("INTERLEAAVE!");



			//byte interleave
			Vector<float> left;
			Vector<float> right;

			int tlen = data.size()/2;
			left.resize(tlen);
			right.resize(tlen);

			for(int i=0;i<tlen;i++) {
				left[i]=data[i*2+0];
				right[i]=data[i*2+1];
			}

			PoolVector<uint8_t> bleft;
			PoolVector<uint8_t> bright;

			_compress_ima_adpcm(left,bleft);
			_compress_ima_adpcm(right,bright);

			int dl = bleft.size();
			dst_data.resize( dl *2 );

			PoolVector<uint8_t>::Write w=dst_data.write();
			PoolVector<uint8_t>::Read rl=bleft.read();
			PoolVector<uint8_t>::Read rr=bright.read();

			for(int i=0;i<dl;i++) {
				w[i*2+0]=rl[i];
				w[i*2+1]=rr[i];
			}
		}

		//print_line("compressing ima-adpcm, resulting buffersize is "+itos(dst_data.size())+" from "+itos(data.size()));

	} else {

		dst_format=is16?Sample::FORMAT_PCM16:Sample::FORMAT_PCM8;
		dst_data.resize( data.size() * (is16?2:1));
		{
			PoolVector<uint8_t>::Write w = dst_data.write();

			int ds=data.size();
			for(int i=0;i<ds;i++) {

				if (is16) {
					int16_t v = CLAMP(data[i]*32767,-32768,32767);
					encode_uint16(v,&w[i*2]);
				} else {
					int8_t v = CLAMP(data[i]*127,-128,127);
					w[i]=v;
				}
			}
		}
	}


	Ref<Sample> target;

	if (ResourceCache::has(p_path)) {

		target = Ref<Sample>( ResourceCache::get(p_path)->cast_to<Sample>() );
	} else {

		target = smp;
	}

	target->create(dst_format,chans==2?true:false,len);
	target->set_data(dst_data);
	target->set_mix_rate(rate);
	target->set_loop_format(loop);
	target->set_loop_begin(loop_beg);
	target->set_loop_end(loop_end);

	from->set_source_md5(0,FileAccess::get_md5(src_path));
	from->set_editor(get_name());
	target->set_import_metadata(from);


	Error err = ResourceSaver::save(p_path,smp);

	return err;

}

void EditorSampleImportPlugin::_compress_ima_adpcm(const Vector<float>& p_data,PoolVector<uint8_t>& dst_data) {


	/*p_sample_data->data = (void*)malloc(len);
	xm_s8 *dataptr=(xm_s8*)p_sample_data->data;*/

	static const int16_t _ima_adpcm_step_table[89] = {
		7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
		19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
		50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
		130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
		337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
		876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
		2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
		5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
		15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
	};

	static const int8_t _ima_adpcm_index_table[16] = {
		-1, -1, -1, -1, 2, 4, 6, 8,
		-1, -1, -1, -1, 2, 4, 6, 8
	};


	int datalen = p_data.size();
	int datamax=datalen;
	if (datalen&1)
		datalen++;

	dst_data.resize(datalen/2+4);
	PoolVector<uint8_t>::Write w = dst_data.write();


	int i,step_idx=0,prev=0;
	uint8_t *out = w.ptr();
	//int16_t xm_prev=0;
	const float *in=p_data.ptr();


	/* initial value is zero */
	*(out++) =0;
	*(out++) =0;
	/* Table index initial value */
	*(out++) =0;
	/* unused */
	*(out++) =0;

	for (i=0;i<datalen;i++) {
		int step,diff,vpdiff,mask;
		uint8_t nibble;
		int16_t xm_sample;

		if (i>=datamax)
			xm_sample=0;
		else {


			xm_sample=CLAMP(in[i]*32767.0,-32768,32767);
			/*
			if (xm_sample==32767 || xm_sample==-32768)
				printf("clippy!\n",xm_sample);
			*/
		}

		//xm_sample=xm_sample+xm_prev;
		//xm_prev=xm_sample;

		diff = (int)xm_sample - prev ;

		nibble=0 ;
		step =  _ima_adpcm_step_table[ step_idx ];
		vpdiff = step >> 3 ;
		if (diff < 0) {
			nibble=8;
			diff=-diff ;
		}
		mask = 4 ;
		while (mask) {

			if (diff >= step) {

				nibble |= mask;
				diff -= step;
				vpdiff += step;
			}

			step >>= 1 ;
			mask >>= 1 ;
		};

		if (nibble&8)
			prev-=vpdiff ;
		else
			prev+=vpdiff ;

		if (prev > 32767) {
			//printf("%i,xms %i, prev %i,diff %i, vpdiff %i, clip up %i\n",i,xm_sample,prev,diff,vpdiff,prev);
			prev=32767;
		} else if (prev < -32768) {
			//printf("%i,xms %i, prev %i,diff %i, vpdiff %i, clip down %i\n",i,xm_sample,prev,diff,vpdiff,prev);
			prev = -32768 ;
		}

		step_idx += _ima_adpcm_index_table[nibble];
		if (step_idx< 0)
			step_idx= 0 ;
		else if (step_idx> 88)
			step_idx= 88 ;


		if (i&1) {
			*out|=nibble<<4;
			out++;
		} else {
			*out=nibble;
		}
		/*dataptr[i]=prev>>8;*/
	}

}


EditorSampleImportPlugin* EditorSampleImportPlugin::singleton=NULL;


void EditorSampleImportPlugin::import_from_drop(const Vector<String>& p_drop, const String &p_dest_path) {


	Vector<String> files;
	for(int i=0;i<p_drop.size();i++) {
		String ext = p_drop[i].get_extension().to_lower();

		if (ext=="wav") {

			files.push_back(p_drop[i]);
		}
	}

	if (files.size()) {
		import_dialog();
		dialog->_choose_files(files);
		dialog->_choose_save_dir(p_dest_path);
	}
}

void EditorSampleImportPlugin::reimport_multiple_files(const Vector<String>& p_list) {

	if (p_list.size()==0)
		return;

	Vector<String> sources;
	for(int i=0;i<p_list.size();i++) {
		int idx;
		EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->find_file(p_list[i],&idx);
		if (efsd) {
			for(int j=0;j<efsd->get_source_count(idx);j++) {
				String file = expand_source_path(efsd->get_source_file(idx,j));
				if (sources.find(file)==-1) {
					sources.push_back(file);
				}

			}
		}
	}

	if (sources.size()) {

		dialog->popup_import(p_list[0]);
		dialog->_choose_files(sources);
		dialog->_choose_save_dir(p_list[0].get_base_dir());
	}
}

bool EditorSampleImportPlugin::can_reimport_multiple_files() const {

	return true;
}

EditorSampleImportPlugin::EditorSampleImportPlugin(EditorNode* p_editor) {

	singleton=this;
	dialog = memnew( EditorSampleImportDialog(this));
	p_editor->get_gui_base()->add_child(dialog);
}

Vector<uint8_t> EditorSampleExportPlugin::custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform) {



	if (EditorImportExport::get_singleton()->sample_get_action()==EditorImportExport::SAMPLE_ACTION_NONE || p_path.get_extension().to_lower()!="wav") {

		return Vector<uint8_t>();
	}

	Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );

	imd->add_source(EditorImportPlugin::validate_source_path(p_path));

	imd->set_option("force/8_bit",false);
	imd->set_option("force/mono",false);
	imd->set_option("force/max_rate",true);
	imd->set_option("force/max_rate_hz",EditorImportExport::get_singleton()->sample_get_max_hz());
	imd->set_option("edit/trim",EditorImportExport::get_singleton()->sample_get_trim());
	imd->set_option("edit/normalize",false);
	imd->set_option("edit/loop",false);
	imd->set_option("compress/mode",1);

	String savepath = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp/smpconv.smp");
	Error err = EditorSampleImportPlugin::singleton->import(savepath,imd);


	ERR_FAIL_COND_V(err!=OK,Vector<uint8_t>());

	p_path=p_path.get_basename()+".converted.smp";
	return FileAccess::get_file_as_array(savepath);

}



EditorSampleExportPlugin::EditorSampleExportPlugin() {

}

#endif

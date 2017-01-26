#include "resource_import.h"
#include "variant_parser.h"

Error ResourceFormatImporter::_get_path_and_type(const String& p_path, PathAndType &r_path_and_type) const {

	Error err;
	FileAccess *f= FileAccess::open(p_path+".import",FileAccess::READ,&err);

	if (!f)
		return err;

	VariantParser::StreamFile stream;
	stream.f=f;

	String assign;
	Variant value;
	VariantParser::Tag next_tag;

	int lines=0;
	String error_text;
	while(true) {

		assign=Variant();
		next_tag.fields.clear();
		next_tag.name=String();

		err = VariantParser::parse_tag_assign_eof(&stream,lines,error_text,next_tag,assign,value,NULL,true);
		if (err==ERR_FILE_EOF) {
			memdelete(f);
			return OK;
		}
		else if (err!=OK) {
			ERR_PRINTS("ResourceFormatImporter::load - "+p_path+":"+itos(lines)+" error: "+error_text);
			memdelete(f);
			return err;
		}

		if (assign!=String()) {
			if (assign=="path") {
				r_path_and_type.path=value;
			} else if (assign=="type") {
				r_path_and_type.type=value;
			}

		} else if (next_tag.name!="remap") {
			break;
		}
	}

	memdelete(f);

	if (r_path_and_type.path==String() || r_path_and_type.type==String()) {
		return ERR_FILE_CORRUPT;
	}
	return OK;

}


RES ResourceFormatImporter::load(const String &p_path,const String& p_original_path,Error *r_error) {

	PathAndType pat;
	Error err = _get_path_and_type(p_path,pat);

	if (err!=OK) {

		if (r_error)
			*r_error=err;

		return RES();
	}


	return ResourceLoader::load(pat.path,pat.type,false,r_error);

}

void ResourceFormatImporter::get_recognized_extensions(List<String> *p_extensions) const{

	Set<String> found;

	for (Set< Ref<ResourceImporter> >::Element *E=importers.front();E;E=E->next()) {
		List<String> local_exts;
		E->get()->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F=local_exts.front();F;F=F->next()) {
			if (!found.has(F->get())) {
				p_extensions->push_back(F->get());
				found.insert(F->get());
			}
		}
	}
}

void ResourceFormatImporter::get_recognized_extensions_for_type(const String& p_type,List<String> *p_extensions) const{

	Set<String> found;

	for (Set< Ref<ResourceImporter> >::Element *E=importers.front();E;E=E->next()) {
		if (!ClassDB::is_parent_class(p_type,E->get()->get_resource_type()))
			continue;

		List<String> local_exts;
		E->get()->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F=local_exts.front();F;F=F->next()) {
			if (!found.has(F->get())) {
				p_extensions->push_back(F->get());
				found.insert(F->get());
			}
		}
	}
}

bool ResourceFormatImporter::recognize_path(const String& p_path,const String& p_for_type) const{

	return (p_path.get_extension().to_lower()=="import");
}

bool ResourceFormatImporter::handles_type(const String& p_type) const {

	for (Set< Ref<ResourceImporter> >::Element *E=importers.front();E;E=E->next()) {
		if (ClassDB::is_parent_class(p_type,E->get()->get_resource_type()))
			return true;

	}

	return true;
}

String ResourceFormatImporter::get_resource_type(const String &p_path) const {

	PathAndType pat;
	Error err = _get_path_and_type(p_path,pat);

	if (err!=OK) {

		return "";
	}

	return pat.type;
}

void ResourceFormatImporter::get_dependencies(const String& p_path,List<String> *p_dependencies,bool p_add_types){

	PathAndType pat;
	Error err = _get_path_and_type(p_path,pat);

	if (err!=OK) {

		return;
	}

	return ResourceLoader::get_dependencies(pat.path,p_dependencies,p_add_types);
}


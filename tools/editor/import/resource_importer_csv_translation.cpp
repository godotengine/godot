
#include "resource_importer_csv_translation.h"
#include "os/file_access.h"
#include "translation.h"
#include "io/resource_saver.h"
#include "compressed_translation.h"

String ResourceImporterCSVTranslation::get_importer_name() const {

	return "csv_translation";
}

String ResourceImporterCSVTranslation::get_visible_name() const{

	return "CSV Translation";
}
void ResourceImporterCSVTranslation::get_recognized_extensions(List<String> *p_extensions) const{

	p_extensions->push_back("csv");
}

String ResourceImporterCSVTranslation::get_save_extension() const {
	return ""; //does not save a single resoure
}

String ResourceImporterCSVTranslation::get_resource_type() const{

	return "StreamCSVTranslation";
}

bool ResourceImporterCSVTranslation::get_option_visibility(const String& p_option,const Map<StringName,Variant>& p_options) const {

	return true;
}

int ResourceImporterCSVTranslation::get_preset_count() const {
	return 0;
}
String ResourceImporterCSVTranslation::get_preset_name(int p_idx) const {

	return "";
}


void ResourceImporterCSVTranslation::get_import_options(List<ImportOption> *r_options,int p_preset) const {


	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"compress"),true));

}



Error ResourceImporterCSVTranslation::import(const String& p_source_file, const String& p_save_path, const Map<StringName,Variant>& p_options, List<String>* r_platform_variants, List<String> *r_gen_files) {


	bool compress = p_options["compress"];
	FileAccessRef f = FileAccess::open(p_source_file,FileAccess::READ);

	ERR_FAIL_COND_V( !f, ERR_INVALID_PARAMETER );

	Vector<String> line = f->get_csv_line();
	if (line.size()<=1) {
		return ERR_PARSE_ERROR;
	}

	Vector<String> locales;
	Vector<Ref<Translation> > translations;

	for(int i=1;i<line.size();i++) {

		String locale = line[i];
		if (!TranslationServer::is_locale_valid(locale)) {
			return ERR_PARSE_ERROR;
		}

		locales.push_back(locale);
		Ref<Translation> translation;
		translation.instance();
		translation->set_locale(locale);
		translations.push_back(translation);
	}

	line = f->get_csv_line();

	while(line.size()==locales.size()+1) {

		String key = line[0];
		if (key!="") {

			for(int i=1;i<line.size();i++) {
				translations[i-1]->add_message(key,line[i]);
			}
		}

		line = f->get_csv_line();
	}


	for(int i=0;i<translations.size();i++) {
		Ref<Translation> xlt = translations[i];

		if (compress) {
			Ref<PHashTranslation> cxl = memnew( PHashTranslation );
			cxl->generate( xlt );
			xlt=cxl;
		}

		String save_path = p_source_file.get_basename()+"."+translations[i]->get_locale()+".xl";

		ResourceSaver::save(save_path,xlt);
		if (r_gen_files) {
			r_gen_files->push_back(save_path);
		}
	}



	return OK;

}

ResourceImporterCSVTranslation::ResourceImporterCSVTranslation()
{

}

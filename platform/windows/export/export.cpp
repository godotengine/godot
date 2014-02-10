#include "export.h"
#include "platform/windows/logo.h"
#include "tools/editor/editor_import_export.h"

void register_windows_exporter() {

	Image img(_windows_logo);
	Ref<ImageTexture> logo = memnew( ImageTexture );
	logo->create_from_image(img);

	{
		Ref<EditorExportPlatformPC> exporter = Ref<EditorExportPlatformPC>( memnew(EditorExportPlatformPC) );
		exporter->set_binary_extension("exe");
		exporter->set_release_binary32("windows_32_release.exe");
		exporter->set_debug_binary32("windows_32_debug.exe");
		exporter->set_release_binary64("windows_64_release.exe");
		exporter->set_debug_binary64("windows_64_debug.exe");
		exporter->set_name("Windows Desktop");
		exporter->set_logo(logo);
		EditorImportExport::get_singleton()->add_export_platform(exporter);
	}


}

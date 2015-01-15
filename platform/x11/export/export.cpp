#include "export.h"
#include "platform/x11/logo.h"
#include "tools/editor/editor_import_export.h"
#include "scene/resources/texture.h"

void register_x11_exporter() {

	Image img(_x11_logo);
	Ref<ImageTexture> logo = memnew( ImageTexture );
	logo->create_from_image(img);

	{
		Ref<EditorExportPlatformPC> exporter = Ref<EditorExportPlatformPC>( memnew(EditorExportPlatformPC) );
		exporter->set_binary_extension("");
		exporter->set_release_binary32("linux_x11_32_release");
		exporter->set_debug_binary32("linux_x11_32_debug");
		exporter->set_release_binary64("linux_x11_64_release");
		exporter->set_debug_binary64("linux_x11_64_debug");
		exporter->set_name("Linux X11");
		exporter->set_logo(logo);
		EditorImportExport::get_singleton()->add_export_platform(exporter);
	}

}

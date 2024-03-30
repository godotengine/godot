#include "CSV_EditorImportPlugin.h"
#include "../logic/data_table_manager.h"


void CSV_EditorImportPlugin::on_table_loaded()
{
    DataTableManager::get_singleton()->reload();
}
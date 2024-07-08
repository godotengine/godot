#pragma once

#include "body_main.h"
#include "CSV_EditorImportPlugin.h"

class CharacterManager : public Object
{
    GDCLASS(CharacterManager, Object);

    public:

    // 創建一個身體

    Ref<CSVData> csv_data;

};
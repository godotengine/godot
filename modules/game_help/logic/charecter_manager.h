#pragma once

#include "body_main.h"
#include "CSV_EditorImportPlugin.h"

class CharacterManager : public Object
{
    GDCLASS(CharacterManager, Object);

    public:

    // 創建一個身體
    class CharacterBodyMain* create_body(int id,const Ref<CharacterController>& controller);

    Ref<CSVData> csv_data;

};
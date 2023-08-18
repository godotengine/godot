/**************************************************************************/
/*  gdscript_preprocessor.h                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GDSCRIPT_PREPROCESSOR_H
#define GDSCRIPT_PREPROCESSOR_H

#include "gdscript.h"

struct DataIf;
struct DataEndIf;

struct DataIf {
    String feature;
    int line;
    int column;
    int ident_level;
    DataEndIf* matching_endif;
};

struct DataEndIf {
    int line;
    int ident_level;
    DataIf* matching_if;
};



class GDScriptPreprocessor {

    public:
    struct ParserError {
        String message = "";
        int line = 0, column = 0;
    };
    static const StringName PREP_IF;
    static const StringName PREP_ENDIF;

    ParserError read_source(const String &p_source_code, String &p_new_source_code);
    GDScriptPreprocessor();
    ~GDScriptPreprocessor();

    private:
    LocalVector<DataIf> data_if;
    LocalVector<DataEndIf> data_endif;
    List<String> features;
    
    bool find_preprocessor_if(const String &p_text, DataIf &p_data);
    bool find_preprocessor_endif(const String &p_text, DataEndIf &p_data);
    bool match(const String &p_search, const String &p_target, int p_at_index);
    bool fast_check(const char &p_first_letter,const String &p_text, int &p_index, int &p_ident_level, char &p_c);
    bool check(const StringName &p_PREP, int& p_index, const String &p_text, char &p_c);
    bool is_active_feature(const String &p_feature);
    ParserError validate();

};

#endif
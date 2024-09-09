//
//  std_io.cpp
//  std_io
//
//  Created by Oded Streigold on 4/18/20.
//

#include "std_io.h"
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#pragma warning (disable : 4996)
#include <iostream>
#include <queue>
#include <locale>
#include <codecvt>
int STD_IO::start_process(String path_godot_str)
{
    std::u32string u32str(path_godot_str.get_data());
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> utf32conv;
    std::string path_string = utf32conv.to_bytes(u32str);
    
    if(this->process != nullptr)
    {
        stop_process();
    }
    
    this->process = new Process(path_string, "",
        [this](const char *bytes, size_t n)
        {
            //cout << "Output from stdout: " << string(bytes, n);
            std::string output_str(bytes, n);
        
            String output_godot_str( output_str.c_str() );
            Variant v(output_godot_str);
        
            Variant** args = new Variant*[1];
            args[0] = {&v};
        
            Callable::CallError err;
            this->godot_output_callback.callp( (const Variant**)args, 1,returnval,err);
        
            delete[] args;
        },
        [this](const char *bytes, size_t n)
        {
            //cout << "Output from stderr: " << string(bytes, n);
        }, true);
    
    return 0;
}

void STD_IO::set_stdout_callback(Callable callback )
{
      this->godot_output_callback = callback ;
}


std::string STD_IO::godot_string_std_string(String godot_string)
{
    std::u32string u32str(godot_string.get_data());
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> utf32conv;
    std::string std_string = utf32conv.to_bytes(u32str);
    return std_string;
}


int STD_IO::send_command(String command){
    if(this->process == nullptr)
        return -1;
    std::string std_str_command = godot_string_std_string(command);
    this->process->write(std_str_command + "\n");
    return 0;
    
}

int STD_IO::stop_process(){
    if(this->process == nullptr)
        return -1;
    this->process->kill();
    delete(this->process);
    this->process = nullptr;
    return 0;
}

void STD_IO::_bind_methods() {
    
    ClassDB::bind_method(D_METHOD("set_stdout_callback", "callback"), &STD_IO::set_stdout_callback);
    ClassDB::bind_method(D_METHOD("start_process", "path_godot_str"), &STD_IO::start_process);
    ClassDB::bind_method(D_METHOD("send_command", "command"), &STD_IO::send_command);
    ClassDB::bind_method(D_METHOD("stop_process"), &STD_IO::stop_process);
}

STD_IO::STD_IO() {
}
STD_IO::~STD_IO() {
}
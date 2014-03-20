/*************************************************************************/
/*  lua_script.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include <stdlib.h>
#include <string.h>
#include "lua_script.h"
#include "globals.h"
#include "global_constants.h"
//#include "gd_compiler.h"
#include "os/file_access.h"

//////////////////////////////
//         INSTANCE         //
//////////////////////////////

typedef struct {
    const char *type;
    Variant::Type vt;
} BulitinTypes;

static BulitinTypes vtypes[] = {
    // math types
    { "Vector2", Variant::VECTOR2 },
    { "Rect2", Variant::RECT2 },
    { "Vector3", Variant::VECTOR3 },
    { "Matrix32", Variant::MATRIX32 },
    { "Plane", Variant::PLANE },
    { "Quat", Variant::QUAT },
    { "AABB", Variant::_AABB },
    { "Matrix3", Variant::MATRIX3 },
    { "Transform", Variant::TRANSFORM },
    // misc types
    { "Color", Variant::COLOR },
    { "Image", Variant::IMAGE },
    { "NodePath", Variant::NODE_PATH },
    { "RID", Variant::_RID },
    { "Object", Variant::OBJECT },
    { "InputEvent", Variant::INPUT_EVENT },
    { "Dictionary", Variant::DICTIONARY },
    { "Array", Variant::ARRAY },
    { "RawArray", Variant::RAW_ARRAY },
    { "IntArray", Variant::INT_ARRAY },
    { "FloatArray", Variant::REAL_ARRAY },
    { "StringArray", Variant::STRING_ARRAY },
    { "Vector2Array", Variant::VECTOR2_ARRAY },
    { "Vector3Array", Variant::VECTOR3_ARRAY },
    { "ColorArray", Variant::COLOR_ARRAY },
};

int LuaInstance::l_bultins_caller_wrapper(lua_State *L)
{
    const char *key = luaL_checkstring(L, lua_upvalueindex(1));
    int top = lua_gettop(L);

    void *ptr = luaL_checkudata(L, 1, "Variant");
    Variant* var = *((Variant **) ptr);

    if(top == 1)
    {
        Variant::CallError err;
        Variant ret = var->call(key, NULL, 0, err);
        if(err.error == Variant::CallError::CALL_OK)
        {
            l_push_variant(L, ret);
            return 1;
        }
    }
    else
    {
        Variant *vars = memnew_arr(Variant, top - 1);
        Variant *args[128];
        for(int idx = 2; idx <= top; idx++)
        {
            Variant& var = vars[idx - 2];
            args[idx - 2] = &var;
            l_get_variant(L, idx, var);
        }
        Variant::CallError err;
        Variant ret = var->call(key, (const Variant**) (args), top - 1, err);
        memdelete_arr(vars);

        if(err.error == Variant::CallError::CALL_OK)
        {
            l_push_variant(L, ret);
            return 1;
        }
    }
    return 0;
}

int LuaInstance::l_bultins_wrapper(lua_State *L)
{
    LUA_MULTITHREAD_GUARD();

    int type = lua_tointeger(L, lua_upvalueindex(1));
    int top = lua_gettop(L);

    if(top == 0)
    {
        Variant::CallError err;
        Variant ret = Variant::construct((Variant::Type) type, NULL, 0, err);
        if(err.error == Variant::CallError::CALL_OK)
        {
            l_push_variant(L, ret);
            return 1;
        }
    }
    else
    {
        Variant *vars = memnew_arr(Variant, top);
        Variant *args[128];
        for(int idx = 1; idx <= top; idx++)
        {
            Variant& var = vars[idx - 1];
            args[idx - 1] = &var;
            l_get_variant(L, idx, var);
        }
        Variant::CallError err;
        Variant ret = Variant::construct((Variant::Type) type, (const Variant**) (args), top, err);
        memdelete_arr(vars);

        if(err.error == Variant::CallError::CALL_OK)
        {
            l_push_variant(L, ret);
            return 1;
        }
    }
    return 0;
}

bool LuaInstance::l_register_bultins_ctors(lua_State *L)
{
    for(int idx = 0; idx < (sizeof(vtypes) / sizeof(vtypes[0])); idx++)
    {
        BulitinTypes& t = vtypes[idx];
        lua_pushinteger(L, t.vt);
        lua_pushcclosure(L, l_bultins_wrapper, 1);
        lua_setglobal(L, t.type);
    }
    return true;
}

int LuaInstance::meta_bultins__gc(lua_State *L)
{
    LUA_MULTITHREAD_GUARD();

    void *ptr = luaL_checkudata(L, 1, "Variant");
    Variant* var = *((Variant **) ptr);
    memdelete(var);

    lua_pushnil(L);
    lua_setmetatable(L, 1);

    return 0;
}

int LuaInstance::meta_bultins__evaluate(lua_State *L)
{
    LUA_MULTITHREAD_GUARD();

    Variant::Operator op = (Variant::Operator) lua_tointeger(L, lua_upvalueindex(1));

    void *ptr1 = luaL_checkudata(L, 1, "Variant");
    Variant* var1 = *((Variant **) ptr1);

    Variant var2;
    l_get_variant(L, 2, var2);

    Variant ret;
    bool valid = false;
    Variant::evaluate(op, *var1, var2, ret, valid);
    if(valid)
    {
        l_push_variant(L, ret);
        return 1;
    }
    return 0;
}

int LuaInstance::meta_bultins__tostring(lua_State *L)
{
    LUA_MULTITHREAD_GUARD();

    void *ptr = luaL_checkudata(L, 1, "Variant");
    Variant* var = *((Variant **) ptr);

    char buf[4096];
    sprintf(buf, "%s[%s]", var->get_type_name(var->get_type()).utf8().get_data(), (var->operator String()).utf8().get_data());;
    lua_pushstring(L, buf);

    return 1;
}

int LuaInstance::meta_bultins__index(lua_State *L)
{
    LUA_MULTITHREAD_GUARD();

    void *ptr = luaL_checkudata(L, 1, "Variant");
    Variant* var = *((Variant **) ptr);

    Variant key;
    l_get_variant(L, 2, key);

    bool valid = false;
    Variant value = var->get(key, &valid);
    if(valid)
    {
        l_push_variant(L, value);
        return 1;    
    }

    if(lua_type(L, 2) == LUA_TSTRING)
    {
        lua_pushvalue(L, 2);
        lua_pushcclosure(L, l_bultins_caller_wrapper, 1);
        return 1;
    }
    return 0;
}

int LuaInstance::meta_bultins__newindex(lua_State *L)
{
    LUA_MULTITHREAD_GUARD();

    void *ptr = luaL_checkudata(L, 1, "Variant");
    Variant* var = *((Variant **) ptr);

    Variant key, value;
    l_get_variant(L, 2, key);
    l_get_variant(L, 3, value);

    bool valid = false;
    var->set(key, value, &valid);
    if(!valid)
        luaL_error(L, "Unable to set field: '%s'", ((String) key).utf8().get_data());

    return 0;
}

int LuaInstance::l_push_bulltins_type(lua_State *L, const Variant& var)
{
    LUA_MULTITHREAD_GUARD();

    void *ptr = lua_newuserdata(L, sizeof(Variant*));
    *((Variant **) ptr) = memnew(Variant);
    **((Variant **) ptr) = var;
    luaL_getmetatable(L, "Variant");
    lua_setmetatable(L, -2);

    return 1;
}

void LuaInstance::l_get_variant(lua_State *L, int idx, Variant& var)
{
    LUA_MULTITHREAD_GUARD();

    switch(lua_type(L, idx))
    {
    case LUA_TNONE:
    case LUA_TNIL:
    case LUA_TTHREAD:
    case LUA_TLIGHTUSERDATA:
    case LUA_TFUNCTION:
        var = Variant();
        break;

    case LUA_TTABLE:
        break;

    case LUA_TBOOLEAN:
        var = (lua_toboolean(L, idx) != 0);
        break;

    case LUA_TNUMBER:
        var = lua_tonumber(L, idx);
        break;

    case LUA_TSTRING:
        var = lua_tostring(L, idx);
        break;

    case LUA_TUSERDATA:
        {
            void *p = lua_touserdata(L, idx);
            if(p != NULL)
            {
                if(lua_getmetatable(L, idx))
                {
                    lua_getfield(L, LUA_REGISTRYINDEX, "GdObject");
                    if(lua_rawequal(L, -1, -2))
                    {
                        lua_pop(L, 2);
                        var = **((Variant **) p);
                        return;
                    }
                    lua_pop(L, 1);

                    lua_getfield(L, LUA_REGISTRYINDEX, "Variant");
                    if(lua_rawequal(L, -1, -2))
                    {
                        lua_pop(L, 2);
                        var = **((Variant **) p);
                        return;
                    }
                    lua_pop(L, 1);
                }
                lua_pop(L, 1);
            }
        }
        break;
    }
}

void LuaInstance::l_push_variant(lua_State *L, const Variant& var)
{
    LUA_MULTITHREAD_GUARD();

    switch(var.get_type())
    {
    case Variant::NIL:
        lua_pushnil(L);
        break;

    case Variant::BOOL:
        lua_pushboolean(L, ((bool) var) ? 1 : 0);
        break;

    case Variant::INT:
        lua_pushinteger(L, (int) var);
        break;

    case Variant::REAL:
        lua_pushnumber(L, (double) var);
        break;

    case Variant::STRING:
        lua_pushstring(L, ((String) var).utf8().get_data());
        break;

    case Variant::OBJECT:
        {
            Object *obj = var;
            if(obj == NULL)
            {
                lua_pushnil(L);
                break;
            }

            ScriptInstance *sci = obj->get_script_instance();
            if(sci != NULL)
            {
                LuaInstance *inst = dynamic_cast<LuaInstance *>(sci);
                if(inst != NULL)
                {
                    lua_rawgeti(L, LUA_REGISTRYINDEX, inst->ref);
                    if(lua_istable(L, -1))
                    {
                        lua_pushstring(L, ".c_instance");
                        lua_rawget(L, -2);
                        if(!lua_isnil(L, -1))
                        {
                            lua_remove(L, -2);
                            break;
                        }
                        lua_pop(L, 2);
                    }
                    lua_pop(L, 1);
                }
            }
            void *ptr = lua_newuserdata(L, sizeof(obj));
            *((Variant **) ptr)= memnew(Variant);
            **((Variant **) ptr) = var;
            //Reference *ref = dynamic_cast<Reference *>(obj);
            //if(ref != NULL)
            //    ref->reference();
            //lua_pushlightuserdata(L, obj);
            luaL_getmetatable(L, "GdObject");
            lua_setmetatable(L, -2);
        }
        break;

    case Variant::VECTOR2:
    case Variant::RECT2:
    case Variant::VECTOR3:
    case Variant::MATRIX32:
    case Variant::PLANE:
    case Variant::QUAT:
    case Variant::_AABB:
    case Variant::MATRIX3:
    case Variant::TRANSFORM:

    case Variant::COLOR:
    case Variant::IMAGE:
    case Variant::NODE_PATH:
    case Variant::_RID:
    case Variant::INPUT_EVENT:
    case Variant::DICTIONARY:
    case Variant::ARRAY:
    case Variant::RAW_ARRAY:
    case Variant::INT_ARRAY:
    case Variant::REAL_ARRAY:
    case Variant::STRING_ARRAY:
    case Variant::VECTOR2_ARRAY:
    case Variant::VECTOR3_ARRAY:
    case Variant::COLOR_ARRAY:
        l_push_bulltins_type(L, var);
        break;
    }
}

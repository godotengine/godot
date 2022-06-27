/****************************************************************************
 *
 * ftmodapi.h
 *
 *   FreeType modules public interface (specification).
 *
 * Copyright (C) 1996-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTMODAPI_H_
#define FTMODAPI_H_


#include <freetype/freetype.h>

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   module_management
   *
   * @title:
   *   Module Management
   *
   * @abstract:
   *   How to add, upgrade, remove, and control modules from FreeType.
   *
   * @description:
   *   The definitions below are used to manage modules within FreeType.
   *   Internal and external modules can be added, upgraded, and removed at
   *   runtime.  For example, an alternative renderer or proprietary font
   *   driver can be registered and prioritized.  Additionally, some module
   *   properties can also be controlled.
   *
   *   Here is a list of existing values of the `module_name` field in the
   *   @FT_Module_Class structure.
   *
   *   ```
   *     autofitter
   *     bdf
   *     cff
   *     gxvalid
   *     otvalid
   *     pcf
   *     pfr
   *     psaux
   *     pshinter
   *     psnames
   *     raster1
   *     sfnt
   *     smooth
   *     truetype
   *     type1
   *     type42
   *     t1cid
   *     winfonts
   *   ```
   *
   *   Note that the FreeType Cache sub-system is not a FreeType module.
   *
   * @order:
   *   FT_Module
   *   FT_Module_Constructor
   *   FT_Module_Destructor
   *   FT_Module_Requester
   *   FT_Module_Class
   *
   *   FT_Add_Module
   *   FT_Get_Module
   *   FT_Remove_Module
   *   FT_Add_Default_Modules
   *
   *   FT_FACE_DRIVER_NAME
   *   FT_Property_Set
   *   FT_Property_Get
   *   FT_Set_Default_Properties
   *
   *   FT_New_Library
   *   FT_Done_Library
   *   FT_Reference_Library
   *
   *   FT_Renderer
   *   FT_Renderer_Class
   *
   *   FT_Get_Renderer
   *   FT_Set_Renderer
   *
   *   FT_Set_Debug_Hook
   *
   */


  /* module bit flags */
#define FT_MODULE_FONT_DRIVER         1  /* this module is a font driver  */
#define FT_MODULE_RENDERER            2  /* this module is a renderer     */
#define FT_MODULE_HINTER              4  /* this module is a glyph hinter */
#define FT_MODULE_STYLER              8  /* this module is a styler       */

#define FT_MODULE_DRIVER_SCALABLE      0x100  /* the driver supports      */
                                              /* scalable fonts           */
#define FT_MODULE_DRIVER_NO_OUTLINES   0x200  /* the driver does not      */
                                              /* support vector outlines  */
#define FT_MODULE_DRIVER_HAS_HINTER    0x400  /* the driver provides its  */
                                              /* own hinter               */
#define FT_MODULE_DRIVER_HINTS_LIGHTLY 0x800  /* the driver's hinter      */
                                              /* produces LIGHT hints     */


  /* deprecated values */
#define ft_module_font_driver         FT_MODULE_FONT_DRIVER
#define ft_module_renderer            FT_MODULE_RENDERER
#define ft_module_hinter              FT_MODULE_HINTER
#define ft_module_styler              FT_MODULE_STYLER

#define ft_module_driver_scalable       FT_MODULE_DRIVER_SCALABLE
#define ft_module_driver_no_outlines    FT_MODULE_DRIVER_NO_OUTLINES
#define ft_module_driver_has_hinter     FT_MODULE_DRIVER_HAS_HINTER
#define ft_module_driver_hints_lightly  FT_MODULE_DRIVER_HINTS_LIGHTLY


  typedef FT_Pointer  FT_Module_Interface;


  /**************************************************************************
   *
   * @functype:
   *   FT_Module_Constructor
   *
   * @description:
   *   A function used to initialize (not create) a new module object.
   *
   * @input:
   *   module ::
   *     The module to initialize.
   */
  typedef FT_Error
  (*FT_Module_Constructor)( FT_Module  module );


  /**************************************************************************
   *
   * @functype:
   *   FT_Module_Destructor
   *
   * @description:
   *   A function used to finalize (not destroy) a given module object.
   *
   * @input:
   *   module ::
   *     The module to finalize.
   */
  typedef void
  (*FT_Module_Destructor)( FT_Module  module );


  /**************************************************************************
   *
   * @functype:
   *   FT_Module_Requester
   *
   * @description:
   *   A function used to query a given module for a specific interface.
   *
   * @input:
   *   module ::
   *     The module to be searched.
   *
   *   name ::
   *     The name of the interface in the module.
   */
  typedef FT_Module_Interface
  (*FT_Module_Requester)( FT_Module    module,
                          const char*  name );


  /**************************************************************************
   *
   * @struct:
   *   FT_Module_Class
   *
   * @description:
   *   The module class descriptor.  While being a public structure necessary
   *   for FreeType's module bookkeeping, most of the fields are essentially
   *   internal, not to be used directly by an application.
   *
   * @fields:
   *   module_flags ::
   *     Bit flags describing the module.
   *
   *   module_size ::
   *     The size of one module object/instance in bytes.
   *
   *   module_name ::
   *     The name of the module.
   *
   *   module_version ::
   *     The version, as a 16.16 fixed number (major.minor).
   *
   *   module_requires ::
   *     The version of FreeType this module requires, as a 16.16 fixed
   *     number (major.minor).  Starts at version 2.0, i.e., 0x20000.
   *
   *   module_interface ::
   *     A typeless pointer to a structure (which varies between different
   *     modules) that holds the module's interface functions.  This is
   *     essentially what `get_interface` returns.
   *
   *   module_init ::
   *     The initializing function.
   *
   *   module_done ::
   *     The finalizing function.
   *
   *   get_interface ::
   *     The interface requesting function.
   */
  typedef struct  FT_Module_Class_
  {
    FT_ULong               module_flags;
    FT_Long                module_size;
    const FT_String*       module_name;
    FT_Fixed               module_version;
    FT_Fixed               module_requires;

    const void*            module_interface;

    FT_Module_Constructor  module_init;
    FT_Module_Destructor   module_done;
    FT_Module_Requester    get_interface;

  } FT_Module_Class;


  /**************************************************************************
   *
   * @function:
   *   FT_Add_Module
   *
   * @description:
   *   Add a new module to a given library instance.
   *
   * @inout:
   *   library ::
   *     A handle to the library object.
   *
   * @input:
   *   clazz ::
   *     A pointer to class descriptor for the module.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   An error will be returned if a module already exists by that name, or
   *   if the module requires a version of FreeType that is too great.
   */
  FT_EXPORT( FT_Error )
  FT_Add_Module( FT_Library              library,
                 const FT_Module_Class*  clazz );


  /**************************************************************************
   *
   * @function:
   *   FT_Get_Module
   *
   * @description:
   *   Find a module by its name.
   *
   * @input:
   *   library ::
   *     A handle to the library object.
   *
   *   module_name ::
   *     The module's name (as an ASCII string).
   *
   * @return:
   *   A module handle.  0~if none was found.
   *
   * @note:
   *   FreeType's internal modules aren't documented very well, and you
   *   should look up the source code for details.
   */
  FT_EXPORT( FT_Module )
  FT_Get_Module( FT_Library   library,
                 const char*  module_name );


  /**************************************************************************
   *
   * @function:
   *   FT_Remove_Module
   *
   * @description:
   *   Remove a given module from a library instance.
   *
   * @inout:
   *   library ::
   *     A handle to a library object.
   *
   * @input:
   *   module ::
   *     A handle to a module object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The module object is destroyed by the function in case of success.
   */
  FT_EXPORT( FT_Error )
  FT_Remove_Module( FT_Library  library,
                    FT_Module   module );


  /**************************************************************************
   *
   * @macro:
   *   FT_FACE_DRIVER_NAME
   *
   * @description:
   *   A macro that retrieves the name of a font driver from a face object.
   *
   * @note:
   *   The font driver name is a valid `module_name` for @FT_Property_Set
   *   and @FT_Property_Get.  This is not the same as @FT_Get_Font_Format.
   *
   * @since:
   *   2.11
   *
   */
#define FT_FACE_DRIVER_NAME( face )                                     \
          ( ( *FT_REINTERPRET_CAST( FT_Module_Class**,                  \
                                    ( face )->driver ) )->module_name )


  /**************************************************************************
   *
   * @function:
   *    FT_Property_Set
   *
   * @description:
   *    Set a property for a given module.
   *
   * @input:
   *    library ::
   *      A handle to the library the module is part of.
   *
   *    module_name ::
   *      The module name.
   *
   *    property_name ::
   *      The property name.  Properties are described in section
   *      @properties.
   *
   *      Note that only a few modules have properties.
   *
   *    value ::
   *      A generic pointer to a variable or structure that gives the new
   *      value of the property.  The exact definition of `value` is
   *      dependent on the property; see section @properties.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *    If `module_name` isn't a valid module name, or `property_name`
   *    doesn't specify a valid property, or if `value` doesn't represent a
   *    valid value for the given property, an error is returned.
   *
   *    The following example sets property 'bar' (a simple integer) in
   *    module 'foo' to value~1.
   *
   *    ```
   *      FT_UInt  bar;
   *
   *
   *      bar = 1;
   *      FT_Property_Set( library, "foo", "bar", &bar );
   *    ```
   *
   *    Note that the FreeType Cache sub-system doesn't recognize module
   *    property changes.  To avoid glyph lookup confusion within the cache
   *    you should call @FTC_Manager_Reset to completely flush the cache if a
   *    module property gets changed after @FTC_Manager_New has been called.
   *
   *    It is not possible to set properties of the FreeType Cache sub-system
   *    itself with FT_Property_Set; use @FTC_Property_Set instead.
   *
   * @since:
   *   2.4.11
   *
   */
  FT_EXPORT( FT_Error )
  FT_Property_Set( FT_Library        library,
                   const FT_String*  module_name,
                   const FT_String*  property_name,
                   const void*       value );


  /**************************************************************************
   *
   * @function:
   *    FT_Property_Get
   *
   * @description:
   *    Get a module's property value.
   *
   * @input:
   *    library ::
   *      A handle to the library the module is part of.
   *
   *    module_name ::
   *      The module name.
   *
   *    property_name ::
   *      The property name.  Properties are described in section
   *      @properties.
   *
   * @inout:
   *    value ::
   *      A generic pointer to a variable or structure that gives the value
   *      of the property.  The exact definition of `value` is dependent on
   *      the property; see section @properties.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *    If `module_name` isn't a valid module name, or `property_name`
   *    doesn't specify a valid property, or if `value` doesn't represent a
   *    valid value for the given property, an error is returned.
   *
   *    The following example gets property 'baz' (a range) in module 'foo'.
   *
   *    ```
   *      typedef  range_
   *      {
   *        FT_Int32  min;
   *        FT_Int32  max;
   *
   *      } range;
   *
   *      range  baz;
   *
   *
   *      FT_Property_Get( library, "foo", "baz", &baz );
   *    ```
   *
   *    It is not possible to retrieve properties of the FreeType Cache
   *    sub-system with FT_Property_Get; use @FTC_Property_Get instead.
   *
   * @since:
   *   2.4.11
   *
   */
  FT_EXPORT( FT_Error )
  FT_Property_Get( FT_Library        library,
                   const FT_String*  module_name,
                   const FT_String*  property_name,
                   void*             value );


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Default_Properties
   *
   * @description:
   *   If compilation option `FT_CONFIG_OPTION_ENVIRONMENT_PROPERTIES` is
   *   set, this function reads the `FREETYPE_PROPERTIES` environment
   *   variable to control driver properties.  See section @properties for
   *   more.
   *
   *   If the compilation option is not set, this function does nothing.
   *
   *   `FREETYPE_PROPERTIES` has the following syntax form (broken here into
   *   multiple lines for better readability).
   *
   *   ```
   *     <optional whitespace>
   *     <module-name1> ':'
   *     <property-name1> '=' <property-value1>
   *     <whitespace>
   *     <module-name2> ':'
   *     <property-name2> '=' <property-value2>
   *     ...
   *   ```
   *
   *   Example:
   *
   *   ```
   *     FREETYPE_PROPERTIES=truetype:interpreter-version=35 \
   *                         cff:no-stem-darkening=0
   *   ```
   *
   * @inout:
   *   library ::
   *     A handle to a new library object.
   *
   * @since:
   *   2.8
   */
  FT_EXPORT( void )
  FT_Set_Default_Properties( FT_Library  library );


  /**************************************************************************
   *
   * @function:
   *   FT_Reference_Library
   *
   * @description:
   *   A counter gets initialized to~1 at the time an @FT_Library structure
   *   is created.  This function increments the counter.  @FT_Done_Library
   *   then only destroys a library if the counter is~1, otherwise it simply
   *   decrements the counter.
   *
   *   This function helps in managing life-cycles of structures that
   *   reference @FT_Library objects.
   *
   * @input:
   *   library ::
   *     A handle to a target library object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @since:
   *   2.4.2
   */
  FT_EXPORT( FT_Error )
  FT_Reference_Library( FT_Library  library );


  /**************************************************************************
   *
   * @function:
   *   FT_New_Library
   *
   * @description:
   *   This function is used to create a new FreeType library instance from a
   *   given memory object.  It is thus possible to use libraries with
   *   distinct memory allocators within the same program.  Note, however,
   *   that the used @FT_Memory structure is expected to remain valid for the
   *   life of the @FT_Library object.
   *
   *   Normally, you would call this function (followed by a call to
   *   @FT_Add_Default_Modules or a series of calls to @FT_Add_Module, and a
   *   call to @FT_Set_Default_Properties) instead of @FT_Init_FreeType to
   *   initialize the FreeType library.
   *
   *   Don't use @FT_Done_FreeType but @FT_Done_Library to destroy a library
   *   instance.
   *
   * @input:
   *   memory ::
   *     A handle to the original memory object.
   *
   * @output:
   *   alibrary ::
   *     A pointer to handle of a new library object.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   See the discussion of reference counters in the description of
   *   @FT_Reference_Library.
   */
  FT_EXPORT( FT_Error )
  FT_New_Library( FT_Memory    memory,
                  FT_Library  *alibrary );


  /**************************************************************************
   *
   * @function:
   *   FT_Done_Library
   *
   * @description:
   *   Discard a given library object.  This closes all drivers and discards
   *   all resource objects.
   *
   * @input:
   *   library ::
   *     A handle to the target library.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   See the discussion of reference counters in the description of
   *   @FT_Reference_Library.
   */
  FT_EXPORT( FT_Error )
  FT_Done_Library( FT_Library  library );


  /**************************************************************************
   *
   * @functype:
   *   FT_DebugHook_Func
   *
   * @description:
   *   A drop-in replacement (or rather a wrapper) for the bytecode or
   *   charstring interpreter's main loop function.
   *
   *   Its job is essentially
   *
   *   - to activate debug mode to enforce single-stepping,
   *
   *   - to call the main loop function to interpret the next opcode, and
   *
   *   - to show the changed context to the user.
   *
   *   An example for such a main loop function is `TT_RunIns` (declared in
   *   FreeType's internal header file `src/truetype/ttinterp.h`).
   *
   *   Have a look at the source code of the `ttdebug` FreeType demo program
   *   for an example of a drop-in replacement.
   *
   * @inout:
   *   arg ::
   *     A typeless pointer, to be cast to the main loop function's data
   *     structure (which depends on the font module).  For TrueType fonts
   *     it is bytecode interpreter's execution context, `TT_ExecContext`,
   *     which is declared in FreeType's internal header file `tttypes.h`.
   */
  typedef FT_Error
  (*FT_DebugHook_Func)( void*  arg );


  /**************************************************************************
   *
   * @enum:
   *   FT_DEBUG_HOOK_XXX
   *
   * @description:
   *   A list of named debug hook indices.
   *
   * @values:
   *   FT_DEBUG_HOOK_TRUETYPE::
   *     This hook index identifies the TrueType bytecode debugger.
   */
#define FT_DEBUG_HOOK_TRUETYPE  0


  /**************************************************************************
   *
   * @function:
   *   FT_Set_Debug_Hook
   *
   * @description:
   *   Set a debug hook function for debugging the interpreter of a font
   *   format.
   *
   *   While this is a public API function, an application needs access to
   *   FreeType's internal header files to do something useful.
   *
   *   Have a look at the source code of the `ttdebug` FreeType demo program
   *   for an example of its usage.
   *
   * @inout:
   *   library ::
   *     A handle to the library object.
   *
   * @input:
   *   hook_index ::
   *     The index of the debug hook.  You should use defined enumeration
   *     macros like @FT_DEBUG_HOOK_TRUETYPE.
   *
   *   debug_hook ::
   *     The function used to debug the interpreter.
   *
   * @note:
   *   Currently, four debug hook slots are available, but only one (for the
   *   TrueType interpreter) is defined.
   */
  FT_EXPORT( void )
  FT_Set_Debug_Hook( FT_Library         library,
                     FT_UInt            hook_index,
                     FT_DebugHook_Func  debug_hook );


  /**************************************************************************
   *
   * @function:
   *   FT_Add_Default_Modules
   *
   * @description:
   *   Add the set of default drivers to a given library object.  This is
   *   only useful when you create a library object with @FT_New_Library
   *   (usually to plug a custom memory manager).
   *
   * @inout:
   *   library ::
   *     A handle to a new library object.
   */
  FT_EXPORT( void )
  FT_Add_Default_Modules( FT_Library  library );



  /**************************************************************************
   *
   * @section:
   *   truetype_engine
   *
   * @title:
   *   The TrueType Engine
   *
   * @abstract:
   *   TrueType bytecode support.
   *
   * @description:
   *   This section contains a function used to query the level of TrueType
   *   bytecode support compiled in this version of the library.
   *
   */


  /**************************************************************************
   *
   * @enum:
   *    FT_TrueTypeEngineType
   *
   * @description:
   *    A list of values describing which kind of TrueType bytecode engine is
   *    implemented in a given FT_Library instance.  It is used by the
   *    @FT_Get_TrueType_Engine_Type function.
   *
   * @values:
   *    FT_TRUETYPE_ENGINE_TYPE_NONE ::
   *      The library doesn't implement any kind of bytecode interpreter.
   *
   *    FT_TRUETYPE_ENGINE_TYPE_UNPATENTED ::
   *      Deprecated and removed.
   *
   *    FT_TRUETYPE_ENGINE_TYPE_PATENTED ::
   *      The library implements a bytecode interpreter that covers the full
   *      instruction set of the TrueType virtual machine (this was governed
   *      by patents until May 2010, hence the name).
   *
   * @since:
   *    2.2
   *
   */
  typedef enum  FT_TrueTypeEngineType_
  {
    FT_TRUETYPE_ENGINE_TYPE_NONE = 0,
    FT_TRUETYPE_ENGINE_TYPE_UNPATENTED,
    FT_TRUETYPE_ENGINE_TYPE_PATENTED

  } FT_TrueTypeEngineType;


  /**************************************************************************
   *
   * @function:
   *    FT_Get_TrueType_Engine_Type
   *
   * @description:
   *    Return an @FT_TrueTypeEngineType value to indicate which level of the
   *    TrueType virtual machine a given library instance supports.
   *
   * @input:
   *    library ::
   *      A library instance.
   *
   * @return:
   *    A value indicating which level is supported.
   *
   * @since:
   *    2.2
   *
   */
  FT_EXPORT( FT_TrueTypeEngineType )
  FT_Get_TrueType_Engine_Type( FT_Library  library );

  /* */


FT_END_HEADER

#endif /* FTMODAPI_H_ */


/* END */

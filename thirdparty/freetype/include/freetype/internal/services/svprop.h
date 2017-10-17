/***************************************************************************/
/*                                                                         */
/*  svprop.h                                                               */
/*                                                                         */
/*    The FreeType property service (specification).                       */
/*                                                                         */
/*  Copyright 2012-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef SVPROP_H_
#define SVPROP_H_


FT_BEGIN_HEADER


#define FT_SERVICE_ID_PROPERTIES  "properties"


  typedef FT_Error
  (*FT_Properties_SetFunc)( FT_Module    module,
                            const char*  property_name,
                            const void*  value,
                            FT_Bool      value_is_string );

  typedef FT_Error
  (*FT_Properties_GetFunc)( FT_Module    module,
                            const char*  property_name,
                            void*        value );


  FT_DEFINE_SERVICE( Properties )
  {
    FT_Properties_SetFunc  set_property;
    FT_Properties_GetFunc  get_property;
  };


#ifndef FT_CONFIG_OPTION_PIC

#define FT_DEFINE_SERVICE_PROPERTIESREC( class_,          \
                                         set_property_,   \
                                         get_property_ )  \
  static const FT_Service_PropertiesRec  class_ =         \
  {                                                       \
    set_property_,                                        \
    get_property_                                         \
  };

#else /* FT_CONFIG_OPTION_PIC */

#define FT_DEFINE_SERVICE_PROPERTIESREC( class_,                \
                                         set_property_,         \
                                         get_property_ )        \
  void                                                          \
  FT_Init_Class_ ## class_( FT_Service_PropertiesRec*  clazz )  \
  {                                                             \
    clazz->set_property = set_property_;                        \
    clazz->get_property = get_property_;                        \
  }

#endif /* FT_CONFIG_OPTION_PIC */

  /* */


FT_END_HEADER


#endif /* SVPROP_H_ */


/* END */

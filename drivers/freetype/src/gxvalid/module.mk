#
# FreeType 2 gxvalid module definition
#

# Copyright 2004, 2005, 2006
#   by suzuki toshiya, Masatake YAMATO, Red Hat K.K.,
#   David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += GXVALID_MODULE

define GXVALID_MODULE
$(OPEN_DRIVER) FT_Module_Class, gxv_module_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)gxvalid   $(ECHO_DRIVER_DESC)TrueTypeGX/AAT validation module$(ECHO_DRIVER_DONE)
endef

# EOF

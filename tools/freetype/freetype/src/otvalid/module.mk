#
# FreeType 2 otvalid module definition
#


# Copyright 2004, 2006 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += OTVALID_MODULE

define OTVALID_MODULE
$(OPEN_DRIVER) FT_Module_Class, otv_module_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)otvalid   $(ECHO_DRIVER_DESC)OpenType validation module$(ECHO_DRIVER_DONE)
endef

# EOF

#
# FreeType 2 PSHinter module definition
#


# Copyright (C) 1996-2019 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += PSHINTER_MODULE

define PSHINTER_MODULE
$(OPEN_DRIVER) FT_Module_Class, pshinter_module_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)pshinter  $(ECHO_DRIVER_DESC)Postscript hinter module$(ECHO_DRIVER_DONE)
endef

# EOF

#
# FreeType 2 PSaux module definition
#


# Copyright 1996-2018 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += PSAUX_MODULE

define PSAUX_MODULE
$(OPEN_DRIVER) FT_Module_Class, psaux_module_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)psaux     $(ECHO_DRIVER_DESC)Postscript Type 1 & Type 2 helper module$(ECHO_DRIVER_DONE)
endef

# EOF

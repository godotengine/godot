#
# FreeType 2 PSnames module definition
#


# Copyright 1996-2018 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += PSNAMES_MODULE

define PSNAMES_MODULE
$(OPEN_DRIVER) FT_Module_Class, psnames_module_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)psnames   $(ECHO_DRIVER_DESC)Postscript & Unicode Glyph name handling$(ECHO_DRIVER_DONE)
endef

# EOF

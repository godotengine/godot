if(NOT SKBUILD)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MaterialX" DESTINATION "python" MESSAGE_NEVER)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Scripts" DESTINATION "python" MESSAGE_NEVER)
endif()

if(SKBUILD)
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Scripts/"
    DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}/_scripts"
    PATTERN "README.md" EXCLUDE
  )
endif()

if(MATERIALX_PYTHON_OCIO_DIR)
  if(NOT EXISTS "${MATERIALX_PYTHON_OCIO_DIR}/config.ocio")
    message(WARNING "No file named config.ocio was found in the given OCIO directory.")
  endif()
  install(DIRECTORY "${MATERIALX_PYTHON_OCIO_DIR}/" DESTINATION "${MATERIALX_PYTHON_FOLDER_NAME}/config/" MESSAGE_NEVER)
endif()

if(MATERIALX_INSTALL_PYTHON AND PYTHON_EXECUTABLE AND NOT SKBUILD)
  set(SETUP_PY "${CMAKE_INSTALL_PREFIX}/python/setup.py")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in" "${SETUP_PY}")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install clean --all)")
endif()

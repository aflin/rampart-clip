cmake_minimum_required(VERSION 3.15)
project(rampart_clip C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Common compile flags
add_compile_options(-Wall -g -O3)

include(${CMAKE_CURRENT_SOURCE_DIR}/extern/extern.cmake)

find_program(RAMPART_EXECUTABLE rampart)
if(NOT RAMPART_EXECUTABLE)
  message(FATAL_ERROR "rampart not found in PATH")
endif()

# Get the install path of rampart
execute_process(
  COMMAND ${RAMPART_EXECUTABLE} -c "console.log(process.installPath)"
  RESULT_VARIABLE _res
  OUTPUT_VARIABLE RP_PATH
  ERROR_VARIABLE  _err
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT _res EQUAL 0)
  message(FATAL_ERROR "rampart get installPath failed: ${_err}")
endif()

if(NOT EXISTS "${RP_PATH}/include/rampart.h")
  message(FATAL_ERROR "rampart.h not found in ${RP_PATH}/include")
endif()

include_directories(
  ${RP_PATH}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/clip.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/clip.cpp/ggml/include
)

set(CLIP_LIBS
    clip
    ggml
    -lpthread
    -lm
)

add_library(rampart-clip SHARED
    rampart-clip.c
)

if(APPLE)

  target_link_options(rampart-clip PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-clip.so"
  )

  # Accelerate is linked by clip.cpp/ggml automatically.
  # Metal frameworks needed when GGML_METAL is enabled (Apple Silicon).
  if(GGML_METAL)
    target_link_libraries(rampart-clip PRIVATE
      "-framework Foundation"
      "-framework Metal"
      "-framework MetalKit"
      "-framework MetalPerformanceShaders"
    )
  endif()

else()

  # LINUX
  target_link_options(rampart-clip PRIVATE
    "-Wl,-soname,rampart-clip.so"
  )

  # C++17 filesystem check for older Linux
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++11")

endif()

add_dependencies(rampart-clip
  clip
  ggml
)

set_target_properties(
    rampart-clip
    PROPERTIES SUFFIX ".so"
    PREFIX ""
    OUTPUT_NAME "rampart-clip"
)

target_link_libraries(rampart-clip PRIVATE ${CLIP_LIBS} stdc++)

# After build: copy test scripts and symlink test_images into build dir
file(GLOB TEST_SCRIPTS ${CMAKE_SOURCE_DIR}/*.js)
add_custom_command(TARGET rampart-clip POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SCRIPTS} ${CMAKE_BINARY_DIR}/
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_SOURCE_DIR}/test_images
    ${CMAKE_BINARY_DIR}/test_images
  COMMENT "Copying scripts and linking test_images to build dir"
)

install(TARGETS rampart-clip
  DESTINATION "${RP_PATH}/modules"
)

install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-clip.so\")")

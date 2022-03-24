#cmake_minimum_required(VERSION 3.16)
#project(example DESCRIPTION "examples how to use LogHelper lib")

if (NOT TARGET LogHelper)
    find_package(LogHelper REQUIRED)
endif()
set(CMAKE_BUILD_TYPE Release)

add_executable(example_c example_log.c)
include_directories(example ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(example_c LogHelper)
add_test(NAME ex_c COMMAND example_c)

if (WITH_CXX_BACKEND)
    add_executable(example_cxx example_log.cpp)
    add_test(NAME ex_cxx COMMAND example_cxx)
    target_link_libraries(example_cxx LogHelper)
endif ()

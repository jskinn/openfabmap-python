cmake_minimum_required(VERSION 2.8)
project(OPENFABMAP_PYTHON_BINDING)

# Options
set(PYTHON_VERSION "3.4" CACHE STRING "Python version.")
#set(PY_INSTALL_DIR "Install directory for python modules" /usr/local/lib/python${PYTHON_VERSION}/dist-packages)
if (NOT PY_INSTALL_DIR)
    set(PY_INSTALL_DIR /usr/local/lib/python${PYTHON_VERSION}/dist-packages)
endif()

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})


#LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#Set the openFABMAP directory path
#TODO: Convert this to a proper FindOpenFABMAP.cmake
set(OPEN_FABMAP_INCLUDE_DIR "/opt/openFABMAP/include" CACHE STRING "Path to parent folder of openFABMAP headers")
set(OPEN_FABMAP_LIB "/opt/openFABMAP/build/lib/libopenFABMAP.so" CACHE STRING "Path to openFABMAP library")

# I'm using a custom boost library, because the default one uses python 2 -_-
set(BOOST_ROOT "/opt/boost_1_62_0_python3/")
set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION})

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
#find_package(Boost 1.45.0 REQUIRED)
find_package(Boost 1.45.0 REQUIRED COMPONENTS python)
find_package(OpenCV 2.4.3 REQUIRED)

# Macro for opencv version
if(NOT (OpenCV_VERSION VERSION_LESS "2.4.0"))
    add_definitions(-DOPENCV2P4)
endif()

# RPATH fiddling for the library, so that we can find our alternate boost
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
${PROJECT_SOURCE_DIR}/src
${OPEN_FABMAP_INCLUDE_DIR}
${Boost_INCLUDE_DIRS}
${PYTHON_INCLUDE_DIRS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(TARGET_MODULE_NAME openFABMAP)
add_library(${TARGET_MODULE_NAME} SHARED
src/detectorsAndExtractors.cpp
src/FabMapVocabluary.cpp
src/ChowLiuTree.cpp
src/openFABMAPPython.cpp
src/PythonBindings.cpp
)
set_target_properties(${TARGET_MODULE_NAME} PROPERTIES PREFIX "")

MESSAGE(${Boost_INCLUDE_DIRS})

target_link_libraries(${TARGET_MODULE_NAME} 
${OPEN_FABMAP_LIB}
${OpenCV_LIBS}
${Boost_LIBRARIES}
${PYTHON_LIBRARIES}
)
install(TARGETS ${TARGET_MODULE_NAME} DESTINATION ${PY_INSTALL_DIR})


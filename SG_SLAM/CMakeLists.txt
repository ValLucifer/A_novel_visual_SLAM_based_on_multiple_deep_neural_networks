find_package(PythonInterp REQUIRED)

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(SG_SLAM)

IF(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3 -fopenmp -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wall -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

########## dependencies ##########
set(Torch_DIR "../Thirdparty/pytorch/libtorch/share/cmake/Torch")  # libtorch
set(OpenCV_DIR "/usr/local/share/OpenCV")  # OpenCV

# Eigen
# include_directories("/usr/local/include/eigen3")
include_directories("/usr/include/eigen3")

# Sophus
include_directories( ${PROJECT_SOURCE_DIR}/../Thirdparty/Sophus )

# libtorch
find_package(Torch REQUIRED)
# include_directories( ${Torch_INCLUDE_DIRS} )
# MESSAGE("Torch: " ${Torch_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV 3.4.13 QUIET)
include_directories( ${OpenCV_INCLUDE_DIRS} )
MESSAGE("OpenCV: ${OpenCV_INCLUDE_DIRS}")

# glog
# Google glog——一个基于程序级记录日志信息的c++库
find_package( Glog REQUIRED )
include_directories( ${GLOG_INCLUDE_DIRS} )

# gflags
# GFlags是Google开源的一套命令行参数处理的开源库，包括C++的版本和python 版本
# google好用的gFlags——解析命令行参数（支持c++和python）
find_package( GFlags REQUIRED )
include_directories( ${GFLAGS_INCLUDE_DIRS} )

# fmt
# Sophus 的依赖项 (Sophus现在默认启用 fmt 管理字符输出)
# fmt：现代的 C++ 字符串格式化库，实现了 C++20 的特征
# https://www.jianshu.com/p/fdca0fde50ac
find_package(fmt)

# caffe
include_directories( ${PROJECT_SOURCE_DIR}/../Thirdparty/caffe/build/install/include )

# Pangolin
find_package( Pangolin REQUIRED )
include_directories( ${Pangolin_INCLUDE_DIRS} )

# G2O
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )
SET(G2O_LIBS g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension  )

# csparse # FindCSparse.cmake from g2o
find_package( CSparse REQUIRED )
include_directories( ${CSPARSE_INCLUDE_DIR} )

set( THIRD_PARTY_LIBS 
        ${TORCH_LIBRARIES} 
        ${OpenCV_LIBS}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        fmt::fmt
        ${PROJECT_SOURCE_DIR}/../Thirdparty/caffe/build/install/lib/libcaffe.so
        pthread 
        # ${Pangolin_LIBRARIES} GL GLU GLEW glut
        ${Pangolin_LIBRARIES} 
        ${G2O_LIBS} )

########## source and test ##########
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)
add_subdirectory(${PROJECT_SOURCE_DIR}/app)

get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
MESSAGE("INCLUDE_DIRECTORIES: ${dirs}" )

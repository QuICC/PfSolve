cmake_minimum_required(VERSION 3.10)
project(PfSolve_gen)
#set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
#set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
set(VKFFT_BACKEND 1 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL")

if(${VKFFT_BACKEND} EQUAL 1)
	option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" OFF)
else()
	option(build_VkFFT_cuFFT_benchmark "Build VkFFT cuFFT benchmark" OFF)
endif()

if(${VKFFT_BACKEND} EQUAL 2)
	option(build_VkFFT_rocFFT_benchmark "Build VkFFT rocFFT benchmark" ON)
else()
	option(build_VkFFT_rocFFT_benchmark "Build VkFFT rocFFT benchmark" OFF)
endif()

option(PfSolve_use_FP128 "Perform all CPU calulations in __float128" OFF)

option(build_VkFFT_FFTW_precision "Build VkFFT FFTW precision comparison" OFF)
if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
add_executable(${PROJECT_NAME} PfSolve.cpp
		benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_0_benchmark_VkFFT_single.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_1_benchmark_VkFFT_double.cpp
		benchmark_scripts/vkFFT_scripts/src/sample_2.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND})
if(${VKFFT_BACKEND} EQUAL 0)
	find_package(Vulkan REQUIRED)
elseif(${VKFFT_BACKEND} EQUAL 1)
	find_package(CUDA 9.0 REQUIRED)
	enable_language(CUDA)
if (MSVC)
else()
		set_source_files_properties(PfSolve.cpp PROPERTIES LANGUAGE CUDA)
endif()
	set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)
	target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
	-DVKFFT_BACKEND=${VKFFT_BACKEND}
	-gencode arch=compute_60,code=compute_60
	-gencode arch=compute_70,code=compute_70 
	-gencode arch=compute_75,code=compute_75 
	-gencode arch=compute_80,code=compute_80 
	-gencode arch=compute_86,code=compute_86>")
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
elseif(${VKFFT_BACKEND} EQUAL 2)
	list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
	find_package(hip)
    	find_library(HIP_hiprtc_LIBRARY hiprtc
      		PATHS
      		ENV ROCM_PATH
       		ENV HIP_PATH
      		/opt/rocm
      		PATH_SUFFIXES lib
       		NO_DEFAULT_PATH)
	add_library(hiprt UNKNOWN IMPORTED)
	set_target_properties(hiprt PROPERTIES
		INTERFACE_LINK_LIBRARIES ${HIP_hiprtc_LIBRARY}
	)
elseif(${VKFFT_BACKEND} EQUAL 3)
	#find_package(OpenCL REQUIRED)
	set(OpenCL_LIB_DIR "C:/Program Files (x86)/Intel/oneAPI/compiler/2021.1-beta09/windows")
	set(OpenCL_INCLUDE_DIR "C:/Program Files (x86)/Intel/oneAPI/compiler/2021.1-beta09/windows/include/sycl")
	find_library(
		OpenCL_LIB
		NAMES "OpenCL"
		PATHS ${OpenCL_LIB_DIR}
		PATH_SUFFIXES "lib" "lib64"
		NO_DEFAULT_PATH
	  )
	find_path(
		OpenCL_INCLUDES
		NAMES "CL/opencl.h"
		PATHS ${OpenCL_INCLUDE_DIR}
		PATH_SUFFIXES "include" 
		NO_DEFAULT_PATH
	  )
	target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDES})
endif()
target_compile_definitions(${PROJECT_NAME} PUBLIC -DVK_API_VERSION=11)#10 - Vulkan 1.0, 11 - Vulkan 1.1, 12 - Vulkan 1.2 
if(${VKFFT_BACKEND} EQUAL 0)
	target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glslang-master/glslang/Include/)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/glslang-master)
endif()
if(PfSolve_use_FP128)
	target_compile_definitions(${PROJECT_NAME} PUBLIC -DPFSOLVE_FP128)
	target_link_libraries(${PROJECT_NAME} PUBLIC quadmath)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/PfSolve/)
add_library(PfSolve INTERFACE)
target_compile_definitions(PfSolve INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/vkFFT_scripts/include/)

if(${VKFFT_BACKEND} EQUAL 0)
	target_link_libraries(${PROJECT_NAME} PUBLIC SPIRV glslang Vulkan::Vulkan PfSolve half)
elseif(${VKFFT_BACKEND} EQUAL 1)
	find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
	target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} cuda ${CUDA_NVRTC_LIB} ${CUDA_cusparse_LIBRARY} PfSolve)
	add_definitions(-DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}")
	target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
elseif(${VKFFT_BACKEND} EQUAL 2)
	target_link_libraries(${PROJECT_NAME} PUBLIC hip::host PfSolve)
elseif(${VKFFT_BACKEND} EQUAL 3)
	target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL PfSolve)
endif()

if(build_VkFFT_FFTW_precision)
	add_definitions(-DUSE_FFTW)
	set(FFTW3_LIB_DIR "/usr/lib/x86_64-linux-gnu/")
	set(FFTW3_INCLUDE_DIR "/usr/include/")
	find_library(
		FFTW_LIB
		NAMES "libfftw3-3" "fftw3"
		PATHS ${FFTW3_LIB_DIR}
		PATH_SUFFIXES "lib" "lib64"
		NO_DEFAULT_PATH
	  )
	find_path(
		FFTW_INCLUDES
		NAMES "fftw3.h"
		PATHS ${FFTW3_INCLUDE_DIR}
		PATH_SUFFIXES "include"
		NO_DEFAULT_PATH
	  )
	
	target_link_libraries (${PROJECT_NAME} PUBLIC ${FFTW_LIB})
	target_include_directories(${PROJECT_NAME} PUBLIC ${FFTW_INCLUDES})
endif()
	
if(build_VkFFT_cuFFT_benchmark)
	add_definitions(-DUSE_cuSparse)
	find_package(CUDA 9.0 REQUIRED)
	enable_language(CUDA)	
	if(build_VkFFT_FFTW_precision)
		add_library(cuFFT_scripts STATIC
		benchmark_scripts/cuFFT_scripts/src/user_benchmark_cuFFT.cu
		benchmark_scripts/cuFFT_scripts/src/sample_0_benchmark_cuFFT_single.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1_benchmark_cuFFT_double.cu
		benchmark_scripts/cuFFT_scripts/src/sample_2_benchmark_cuFFT_half.cu
		benchmark_scripts/cuFFT_scripts/src/sample_3_benchmark_cuFFT_single_3d.cu
		benchmark_scripts/cuFFT_scripts/src/sample_6_benchmark_cuFFT_single_r2c.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1000_benchmark_cuFFT_single_2_4096.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1001_benchmark_cuFFT_double_2_4096.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1003_benchmark_cuFFT_single_3d_2_512.cu
		benchmark_scripts/cuFFT_scripts/src/precision_cuFFT_single.cu
		benchmark_scripts/cuFFT_scripts/src/precision_cuFFT_r2c.cu
		benchmark_scripts/cuFFT_scripts/src/precision_cuFFT_double.cu
		benchmark_scripts/cuFFT_scripts/src/precision_cuFFT_half.cu)
	else()
		add_library(cuFFT_scripts STATIC
		benchmark_scripts/cuFFT_scripts/src/user_benchmark_cuFFT.cu
		benchmark_scripts/cuFFT_scripts/src/sample_0_benchmark_cuFFT_single.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1_benchmark_cuFFT_double.cu
		benchmark_scripts/cuFFT_scripts/src/sample_2_benchmark_cuFFT_half.cu
		benchmark_scripts/cuFFT_scripts/src/sample_3_benchmark_cuFFT_single_3d.cu
		benchmark_scripts/cuFFT_scripts/src/sample_6_benchmark_cuFFT_single_r2c.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1000_benchmark_cuFFT_single_2_4096.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1001_benchmark_cuFFT_double_2_4096.cu
		benchmark_scripts/cuFFT_scripts/src/sample_1003_benchmark_cuFFT_single_3d_2_512.cu)
	endif()
	set_property(TARGET cuFFT_scripts PROPERTY CUDA_ARCHITECTURES 35 60 70 75 80 86)
	CUDA_ADD_CUFFT_TO_TARGET(cuFFT_scripts)
	target_compile_options(cuFFT_scripts PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
	-gencode arch=compute_35,code=compute_35
	-gencode arch=compute_60,code=compute_60
	-gencode arch=compute_70,code=compute_70 
	-gencode arch=compute_75,code=compute_75 
	-gencode arch=compute_80,code=compute_80 
	-gencode arch=compute_86,code=compute_86>")
	target_include_directories(cuFFT_scripts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/cuFFT_scripts/include)
	set_target_properties(cuFFT_scripts PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(cuFFT_scripts PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	target_link_libraries(${PROJECT_NAME} PUBLIC cuFFT_scripts)
endif()
if(build_VkFFT_rocFFT_benchmark)
	add_definitions(-DUSE_rocFFT)
	list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
	find_package(hip)
    	find_package(rocfft)

	if(build_VkFFT_FFTW_precision)
		add_library(rocFFT_scripts STATIC
		benchmark_scripts/rocFFT_scripts/src/user_benchmark_rocFFT.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_0_benchmark_rocFFT_single.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1_benchmark_rocFFT_double.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_3_benchmark_rocFFT_single_3d.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_6_benchmark_rocFFT_single_r2c.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1000_benchmark_rocFFT_single_2_4096.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1001_benchmark_rocFFT_double_2_4096.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1003_benchmark_rocFFT_single_3d_2_512.cpp
		benchmark_scripts/rocFFT_scripts/src/precision_rocFFT_single.cpp
		benchmark_scripts/rocFFT_scripts/src/precision_rocFFT_r2c.cpp
		benchmark_scripts/rocFFT_scripts/src/precision_rocFFT_double.cpp)
	else()
		add_library(rocFFT_scripts STATIC
		benchmark_scripts/rocFFT_scripts/src/user_benchmark_rocFFT.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_0_benchmark_rocFFT_single.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1_benchmark_rocFFT_double.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_3_benchmark_rocFFT_single_3d.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_6_benchmark_rocFFT_single_r2c.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1000_benchmark_rocFFT_single_2_4096.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1001_benchmark_rocFFT_double_2_4096.cpp
		benchmark_scripts/rocFFT_scripts/src/sample_1003_benchmark_rocFFT_single_3d_2_512.cpp)
	endif()
	target_include_directories(rocFFT_scripts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/rocFFT_scripts/include)
	target_link_libraries(rocFFT_scripts PRIVATE hip::host roc::rocfft)
	target_link_libraries(${PROJECT_NAME} PUBLIC rocFFT_scripts)
endif()

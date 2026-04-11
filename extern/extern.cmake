set(EXTERN_DIR ${CMAKE_SOURCE_DIR}/extern)

# ---- CPU feature detection for clip.cpp / ggml ----
# Detect actual CPU features to avoid illegal instructions at runtime.

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")

    # x86: use -march=native for baseline optimizations (SSE4 etc).
    # The CLIP_AVX/AVX2/etc flags are detected separately below to
    # avoid explicit -mavx on CPUs that don't support it.
    set(CLIP_NATIVE ON CACHE BOOL "" FORCE)

    if(APPLE)
        # macOS x86: use sysctl
        execute_process(COMMAND sysctl -n machdep.cpu.features OUTPUT_VARIABLE _CPU_FEATURES OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        execute_process(COMMAND sysctl -n machdep.cpu.leaf7_features OUTPUT_VARIABLE _CPU_LEAF7 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(TOUPPER "${_CPU_FEATURES} ${_CPU_LEAF7}" _CPU_ALL)
    else()
        # Linux: read /proc/cpuinfo
        execute_process(COMMAND grep -m1 "^flags" /proc/cpuinfo OUTPUT_VARIABLE _CPU_ALL OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()

    # AVX
    if(_CPU_ALL MATCHES "avx" AND NOT _CPU_ALL MATCHES "^$")
        message(STATUS "clip.cpp: CPU has AVX")
        set(CLIP_AVX ON CACHE BOOL "" FORCE)
    else()
        message(STATUS "clip.cpp: CPU lacks AVX")
        set(CLIP_AVX OFF CACHE BOOL "" FORCE)
    endif()

    # AVX2
    if(_CPU_ALL MATCHES "avx2")
        message(STATUS "clip.cpp: CPU has AVX2")
        set(CLIP_AVX2 ON CACHE BOOL "" FORCE)
    else()
        message(STATUS "clip.cpp: CPU lacks AVX2")
        set(CLIP_AVX2 OFF CACHE BOOL "" FORCE)
    endif()

    # FMA
    if(_CPU_ALL MATCHES "fma")
        message(STATUS "clip.cpp: CPU has FMA")
        set(CLIP_FMA ON CACHE BOOL "" FORCE)
    else()
        message(STATUS "clip.cpp: CPU lacks FMA")
        set(CLIP_FMA OFF CACHE BOOL "" FORCE)
    endif()

    # F16C
    if(_CPU_ALL MATCHES "f16c")
        message(STATUS "clip.cpp: CPU has F16C")
        set(CLIP_F16C ON CACHE BOOL "" FORCE)
    else()
        message(STATUS "clip.cpp: CPU lacks F16C")
        set(CLIP_F16C OFF CACHE BOOL "" FORCE)
    endif()

    # AVX512
    if(_CPU_ALL MATCHES "avx512f")
        message(STATUS "clip.cpp: CPU has AVX512")
        set(CLIP_AVX512 ON CACHE BOOL "" FORCE)
    else()
        message(STATUS "clip.cpp: CPU lacks AVX512")
        set(CLIP_AVX512 OFF CACHE BOOL "" FORCE)
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64|aarch64)")

    # ARM: -march=native is safe (NEON is mandatory on aarch64)
    set(CLIP_NATIVE ON CACHE BOOL "" FORCE)

    # Disable x86 SIMD flags
    set(CLIP_AVX OFF CACHE BOOL "" FORCE)
    set(CLIP_AVX2 OFF CACHE BOOL "" FORCE)
    set(CLIP_FMA OFF CACHE BOOL "" FORCE)
    set(CLIP_F16C OFF CACHE BOOL "" FORCE)
    set(CLIP_AVX512 OFF CACHE BOOL "" FORCE)

    # Apple Silicon: enable Metal GPU acceleration via ggml
    if(APPLE)
        set(GGML_METAL ON CACHE BOOL "" FORCE)
        message(STATUS "clip.cpp: Apple Silicon — enabling Metal and Accelerate")
    endif()

else()
    # Other architectures
    set(CLIP_NATIVE OFF CACHE BOOL "" FORCE)
    set(CLIP_AVX OFF CACHE BOOL "" FORCE)
    set(CLIP_AVX2 OFF CACHE BOOL "" FORCE)
    set(CLIP_FMA OFF CACHE BOOL "" FORCE)
    set(CLIP_F16C OFF CACHE BOOL "" FORCE)
    set(CLIP_AVX512 OFF CACHE BOOL "" FORCE)
endif()

# ---- CLIP.CPP build config ----
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CLIP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLIP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLIP_BUILD_IMAGE_SEARCH OFF CACHE BOOL "" FORCE)
set(CLIP_LTO OFF CACHE BOOL "" FORCE)

add_subdirectory(${EXTERN_DIR}/clip.cpp ${CMAKE_BINARY_DIR}/extern/clip.cpp EXCLUDE_FROM_ALL)

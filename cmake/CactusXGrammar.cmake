include_guard(GLOBAL)

function(cactus_require_xgrammar_source xgrammar_root)
    if(NOT EXISTS "${xgrammar_root}/CMakeLists.txt")
        message(FATAL_ERROR
            "XGrammar source checkout not found at ${xgrammar_root}. "
            "Initialize the submodule before configuring this build."
        )
    endif()

    if(NOT EXISTS "${xgrammar_root}/3rdparty/dlpack/include/dlpack/dlpack.h")
        message(FATAL_ERROR
            "XGrammar nested submodules are missing under ${xgrammar_root}. "
            "Initialize submodules recursively before configuring this build."
        )
    endif()
endfunction()

function(cactus_add_xgrammar_subdirectory xgrammar_root binary_dir)
    cactus_require_xgrammar_source("${xgrammar_root}")

    if(NOT TARGET xgrammar)
        file(WRITE "${CMAKE_BINARY_DIR}/config.cmake" [=[
set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)
set(XGRAMMAR_BUILD_CXX_TESTS OFF)
set(XGRAMMAR_ENABLE_CPPTRACE OFF)
set(XGRAMMAR_ENABLE_COVERAGE OFF)
set(XGRAMMAR_ENABLE_INTERNAL_CHECK OFF)
]=])
        set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
        set(XGRAMMAR_BUILD_CXX_TESTS OFF CACHE BOOL "" FORCE)
        set(XGRAMMAR_ENABLE_CPPTRACE OFF CACHE BOOL "" FORCE)
        set(XGRAMMAR_ENABLE_COVERAGE OFF CACHE BOOL "" FORCE)
        set(XGRAMMAR_ENABLE_INTERNAL_CHECK OFF CACHE BOOL "" FORCE)
        add_subdirectory("${xgrammar_root}" "${binary_dir}" EXCLUDE_FROM_ALL)

        if(APPLE AND CMAKE_GENERATOR STREQUAL "Xcode")
            target_compile_options(xgrammar PRIVATE
                -Wno-shorten-64-to-32
                -Wno-error=shorten-64-to-32
            )
        endif()
    endif()
endfunction()

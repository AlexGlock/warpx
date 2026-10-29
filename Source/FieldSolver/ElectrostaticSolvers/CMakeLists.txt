foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        EffectivePotentialES.cpp
        ElectrostaticSolver.cpp
        LabFrameExplicitES.cpp
        PoissonBoundaryHandler.cpp
        RelativisticExplicitES.cpp
        SparseLUPoissonSolver.cpp
    )
    if(WarpX_SPARSE_LU)
        target_link_libraries(lib_${SD} PRIVATE PETSC)
        target_compile_definitions(lib_${SD} PUBLIC WARPX_USE_SPARSE_LU)
    endif()
endforeach()

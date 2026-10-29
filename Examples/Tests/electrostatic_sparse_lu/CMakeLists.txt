if(NOT WarpX_SPARSE_LU)
    return()
endif()

# Exercise the actual backend independently of particle initialization. Include
# the production source so these short operator tests need only AMReX and PETSc/MUMPS.
foreach(D IN ITEMS 2 3)
    if(NOT D IN_LIST WarpX_DIMS)
        continue()
    endif()
    set(test_target test_${D}d_sparse_lu)
    add_executable(${test_target}
        test_solver.cpp
        ${WarpX_SOURCE_DIR}/Source/FieldSolver/ElectrostaticSolvers/SparseLUPoissonSolver.cpp)
    target_include_directories(${test_target} PRIVATE
        ${WarpX_SOURCE_DIR}/Source/FieldSolver/ElectrostaticSolvers)
    target_link_libraries(${test_target} PRIVATE AMReX::amrex_${D}d PETSC)
    target_compile_features(${test_target} PRIVATE cxx_std_20)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(${test_target})
    endif()
    target_compile_definitions(${test_target} PRIVATE WARPX_USE_SPARSE_LU)
    if(D EQUAL 2)
        target_compile_definitions(${test_target} PRIVATE WARPX_DIM_XZ)
    else()
        target_compile_definitions(${test_target} PRIVATE WARPX_DIM_3D)
    endif()
    add_test(NAME ${test_target} COMMAND ${test_target})
    add_test(NAME ${test_target}_external_petsc
        COMMAND ${test_target} test.external_petsc=1)
    if(WarpX_MPI)
        add_test(NAME ${test_target}_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_target}> ${MPIEXEC_POSTFLAGS})
    endif()
    if(WarpX_EB)
        add_warpx_test(
            test_${D}d_sparse_lu_eb
            ${D}
            1
            inputs_${D}d
            "analysis.py"
            OFF
            OFF
        )
    endif()
endforeach()

add_test(NAME test_sparse_lu_picmi_inputs
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_picmi.py)
set_tests_properties(test_sparse_lu_picmi_inputs PROPERTIES
    ENVIRONMENT "PYTHONPATH=${WarpX_SOURCE_DIR}/Python:$ENV{PYTHONPATH}")

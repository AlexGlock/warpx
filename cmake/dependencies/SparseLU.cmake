if(NOT WarpX_PETSC OR NOT WarpX_MPI)
    message(FATAL_ERROR
        "WarpX_SPARSE_LU requires WarpX_PETSC=ON and WarpX_MPI=ON. "
        "Select a real-valued PETSc installation built with MUMPS support.")
endif()

# AMReX's PETSc finder supplies this imported target. An external AMReX
# installation may not have loaded it yet.
if(NOT TARGET PETSC)
    find_package(PETSc REQUIRED)
endif()
if(NOT TARGET MPI::MPI_CXX)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES PETSC MPI::MPI_CXX)
# Recheck when a user switches cluster modules or PETSC_DIR in an existing build.
unset(WarpX_PETSC_HAS_MUMPS CACHE)
check_cxx_source_compiles([=[
    #include <petscksp.h>
    #if !defined(PETSC_HAVE_MUMPS)
    #error PETSc must be built with MUMPS
    #endif
    #if defined(PETSC_USE_COMPLEX)
    #error sparse_lu requires real-valued PETSc
    #endif
    int main() {
        PetscBool initialized = PETSC_FALSE;
        return PetscInitialized(&initialized);
    }
]=] WarpX_PETSC_HAS_MUMPS)
cmake_pop_check_state()
if(NOT WarpX_PETSC_HAS_MUMPS)
    message(FATAL_ERROR
        "WarpX_SPARSE_LU needs a linkable, real-valued PETSc built with MUMPS. "
        "Load a matching PETSc/MUMPS module stack that exports PETSC_HOME, or pass "
        "-DPETSC_DIR=/path/to/petsc (plus -DPETSC_ARCH=architecture for an in-tree "
        "PETSc build). Loading standalone MUMPS does not add MUMPS "
        "support to an existing PETSc installation. See the CMake configure log.")
endif()

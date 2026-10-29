/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPARSE_LU_POISSON_SOLVER_H_
#define WARPX_SPARSE_LU_POISSON_SOLVER_H_

#include <AMReX_MLEBNodeFDLaplacian.H>

#include <memory>

/** Single-level Cartesian Dirichlet Poisson solve with cached PETSc/MUMPS LU.
 *
 * The AMReX nodal FD operator supplies the EB stencil, RHS scaling and gradients.
 * Only free nodes enter the matrix. Distributed fields are gathered to rank zero
 * for factorization/substitution, then scattered back. Geometry must remain fixed.
 * An OpenMP-enabled MUMPS/BLAS stack uses the job's thread configuration; this
 * backend does not alter the thread count used by other WarpX operations.
 */
class SparseLUPoissonSolver
{
public:
    SparseLUPoissonSolver (
        amrex::Geometry const& geom, amrex::BoxArray const& grids,
        amrex::DistributionMapping const& dmap,
        amrex::FabFactory<amrex::FArrayBox> const* factory, int verbosity);
    ~SparseLUPoissonSolver ();

    SparseLUPoissonSolver (SparseLUPoissonSolver const&) = delete;
    SparseLUPoissonSolver& operator= (SparseLUPoissonSolver const&) = delete;

    /** Set EB potentials through this operator before each solve; these do not
     * change the cached matrix. Do not change its geometry, BC types or sigma. */
    amrex::MLEBNodeFDLaplacian& linop ();

    /** Solve Laplacian(phi) = rhs. phi supplies domain Dirichlet values.
     * rhs is unscaled by the EB row weights and is never modified. */
    void Solve (amrex::MultiFab const& rhs, amrex::MultiFab& phi);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif

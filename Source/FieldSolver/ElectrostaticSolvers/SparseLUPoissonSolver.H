/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPARSE_LU_POISSON_SOLVER_H_
#define WARPX_SPARSE_LU_POISSON_SOLVER_H_

#if defined(WARPX_DIM_1D_Z)
#include <AMReX_MLNodeLaplacian.H>
using SparseLUPoissonLinOp = amrex::MLNodeLaplacian;
#else
#include <AMReX_MLEBNodeFDLaplacian.H>
using SparseLUPoissonLinOp = amrex::MLEBNodeFDLaplacian;
#endif

#include <memory>

/** Single-level Cartesian 1D/2D/3D Poisson solve with cached PETSc/MUMPS LU.
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
    enum class BoundaryType { Dirichlet, Periodic, Open };
    using BoundaryArray = amrex::Array<BoundaryType, AMREX_SPACEDIM>;

    SparseLUPoissonSolver (
        amrex::Geometry const& geom, amrex::BoxArray const& grids,
        amrex::DistributionMapping const& dmap,
        amrex::FabFactory<amrex::FArrayBox> const* factory, int verbosity,
        BoundaryArray const& lo = {}, BoundaryArray const& hi = {},
        amrex::Real open_reference_radius = 0);
    ~SparseLUPoissonSolver ();

    SparseLUPoissonSolver (SparseLUPoissonSolver const&) = delete;
    SparseLUPoissonSolver& operator= (SparseLUPoissonSolver const&) = delete;

    /** Set EB potentials through this operator before each solve; these do not
     * change the cached matrix. Do not change its geometry, BC types or sigma. */
    SparseLUPoissonLinOp& linop ();

    /** Solve Laplacian(phi) = rhs. phi supplies domain Dirichlet values.
     * rhs is unscaled by the EB row weights and is never modified.
     * Open faces impose radial far-field decay, centered on the first Dirichlet
     * wall (coordinate order, low before high), or the domain center if all open.
     * All-open 2D uses a zero-potential reference radius, defaulting to three
     * times the domain diagonal when open_reference_radius is zero. */
    void Solve (amrex::MultiFab const& rhs, amrex::MultiFab& phi);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif

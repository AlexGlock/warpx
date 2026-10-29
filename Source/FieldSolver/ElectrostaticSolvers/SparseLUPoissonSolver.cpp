/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#if defined(WARPX_USE_SPARSE_LU) && (defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D))

#include "SparseLUPoissonSolver.H"

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace
{
    void checkPetsc (PetscErrorCode error)
    {
        if (error != 0) {
            amrex::Abort("sparse_lu: PETSc/MUMPS error " + std::to_string(error)
                + "; see PETSc's diagnostic above");
        }
    }

    void initializePetsc ()
    {
        // Native WarpX initializes PETSc at startup. PICMI may instead arrive
        // here without PETSc, or with a PETSc runtime owned by petsc4py.
        // All simulation ranks must participate in initialization, even though
        // the actual LU objects below live on rank zero's PETSC_COMM_SELF.
        PetscBool initialized = PETSC_FALSE;
        PetscBool finalized = PETSC_FALSE;
        checkPetsc(PetscInitialized(&initialized));
        checkPetsc(PetscFinalized(&finalized));
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!finalized,
            "sparse_lu cannot restart PETSc after PetscFinalize");
        if (!initialized) {
            PETSC_COMM_WORLD = amrex::ParallelDescriptor::Communicator();
            checkPetsc(PetscInitializeNoArguments());
            // Finalize only a runtime we own, after the WarpX solver is destroyed
            // and before AMReX shuts down MPI. Do not finalize petsc4py's runtime.
            amrex::ExecOnFinalize([] {
                PetscBool already_finalized = PETSC_FALSE;
                checkPetsc(PetscFinalized(&already_finalized));
                if (!already_finalized) { checkPetsc(PetscFinalize()); }
            });
        }
    }

    // Every point in a radius-one stencil has a different color. This permits
    // recovering its coefficients in 3^D operator applications, rather than N.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int color (int i, int j, int k)
    {
        amrex::ignore_unused(k);
        return AMREX_D_TERM((i % 3 + 3) % 3,
                         + 3 * ((j % 3 + 3) % 3),
                         + 9 * ((k % 3 + 3) % 3));
    }
}

struct SparseLUPoissonSolver::Impl
{
    amrex::Geometry m_geom;
    amrex::Box m_domain;
    amrex::MLEBNodeFDLaplacian m_linop;
    amrex::MultiFab m_active, m_probe, m_applied, m_rhs, m_lift, m_global;
    std::vector<PetscInt> m_ids;
    Mat m_matrix = nullptr;
    KSP m_ksp = nullptr;
    Vec m_b = nullptr;
    Vec m_x = nullptr;

    ~Impl ()
    {
        // These are null on non-root ranks and for a domain with no free nodes.
        if (m_ksp) { checkPetsc(KSPDestroy(&m_ksp)); }
        if (m_b) { checkPetsc(VecDestroy(&m_b)); }
        if (m_x) { checkPetsc(VecDestroy(&m_x)); }
        if (m_matrix) { checkPetsc(MatDestroy(&m_matrix)); }
    }

    void Gather (amrex::MultiFab const& field)
    {
        m_global.ParallelCopy(field, 0, 0, 1);
        amrex::Gpu::streamSynchronize(); // host accesses the pinned gather buffer
    }

    Impl (amrex::Geometry const& geom, amrex::BoxArray const& grids,
          amrex::DistributionMapping const& dmap,
          amrex::FabFactory<amrex::FArrayBox> const* factory, int verbosity)
        : m_geom(geom), m_domain(amrex::surroundingNodes(geom.Domain()))
    {
        initializePetsc();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!geom.isAnyPeriodic(),
            "sparse_lu currently requires Dirichlet domain boundaries");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_domain.numPts() <= std::numeric_limits<PetscInt>::max(),
            "sparse_lu node numbering exceeds the supported integer range");
        amrex::LPInfo info;
        info.setMaxCoarseningLevel(0);
#ifdef AMREX_USE_EB
        auto const* eb_factory = dynamic_cast<amrex::EBFArrayBoxFactory const*>(factory);
        if (eb_factory) {
            m_linop.define({geom}, {grids}, {dmap}, info, {eb_factory});
        } else
#else
        amrex::ignore_unused(factory);
#endif
        {
            m_linop.define({geom}, {grids}, {dmap}, info);
        }
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc;
        bc.fill(amrex::LinOpBCType::Dirichlet);
        m_linop.setDomainBC(bc, bc);
        m_linop.prepareForSolve();

        auto const nodes = amrex::convert(grids, amrex::IntVect::TheNodeVector());
        m_active.define(nodes, dmap, 1, 0);
        m_probe.define(nodes, dmap, 1, 1);
        m_applied.define(nodes, dmap, 1, 0);
        m_rhs.define(nodes, dmap, 1, 0);
        m_lift.define(nodes, dmap, 1, 1);
        m_global.define(amrex::BoxArray(m_domain),
            amrex::DistributionMapping(amrex::Vector<int>{0}), 1, 0,
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        m_active.setVal(amrex::Real(1));
        m_linop.setDirichletNodesToZero(0, 0, m_active);
        Gather(m_active);

        PetscInt n = 0;
        if (amrex::ParallelDescriptor::MyProc() == 0) {
            m_ids.resize(m_domain.numPts(), -1);
            auto const a = m_global[0].const_array();
            for (amrex::BoxIterator it(m_domain); it.ok(); ++it) {
                if (a(it()) != amrex::Real(0)) { m_ids[m_domain.index(it())] = n++; }
            }
            if (n > 0) {
                checkPetsc(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n,
                    std::min<PetscInt>(n, 2 * AMREX_SPACEDIM + 1), nullptr, &m_matrix));
                checkPetsc(MatSetOption(m_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
                PetscBool available = PETSC_FALSE;
                checkPetsc(MatGetFactorAvailable(m_matrix, MATSOLVERMUMPS, MAT_FACTOR_LU,
                    &available));
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(available,
                    "sparse_lu requires PETSc built with MUMPS LU support");
            }
        }

        PetscInt nentries = 0;
        constexpr int ncolors = AMREX_D_TERM(3, *3, *3);
        for (int c = 0; c < ncolors; ++c) {
            m_probe.setVal(amrex::Real(0));
            for (amrex::MFIter mfi(m_probe); mfi.isValid(); ++mfi) {
                auto const p = m_probe.array(mfi);
                auto const a = m_active.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        p(i,j,k) = (color(i,j,k) == c) ? a(i,j,k) : amrex::Real(0);
                    });
            }
            m_linop.apply(0, 0, m_applied, m_probe,
                amrex::MLLinOp::BCMode::Homogeneous, amrex::MLLinOp::StateMode::Correction);
            Gather(m_applied);
            if (amrex::ParallelDescriptor::MyProc() == 0) {
                auto const a = m_global[0].const_array();
                for (amrex::BoxIterator it(m_domain); it.ok(); ++it) {
                    PetscInt const row = m_ids[m_domain.index(it())];
                    if (row < 0) { continue; }
                    for (int s = 0; s < 2 * AMREX_SPACEDIM + 1; ++s) {
                        auto q = it();
                        if (s > 0) { q[(s-1)/2] += (s % 2) ? -1 : 1; }
                        if (!m_domain.contains(q)) { continue; }
                        PetscInt const col = m_ids[m_domain.index(q)];
                        auto const qv = amrex::GpuArray<int,3>{AMREX_D_DECL(q[0], q[1], q[2])};
                        if (col >= 0 && color(qv[0], qv[1], qv[2]) == c) {
                            checkPetsc(MatSetValue(m_matrix, row, col,
                                static_cast<PetscScalar>(a(it())), INSERT_VALUES));
                            ++nentries;
                        }
                    }
                }
            }
        }
        if (amrex::ParallelDescriptor::MyProc() == 0 && n > 0) {
            checkPetsc(MatAssemblyBegin(m_matrix, MAT_FINAL_ASSEMBLY));
            checkPetsc(MatAssemblyEnd(m_matrix, MAT_FINAL_ASSEMBLY));
            checkPetsc(MatCreateVecs(m_matrix, &m_x, &m_b));
            checkPetsc(KSPCreate(PETSC_COMM_SELF, &m_ksp));
            checkPetsc(KSPSetOptionsPrefix(m_ksp, "warpx_sparse_lu_"));
            checkPetsc(KSPSetOperators(m_ksp, m_matrix, m_matrix));
            checkPetsc(KSPSetType(m_ksp, KSPPREONLY));
            PC pc = nullptr;
            checkPetsc(KSPGetPC(m_ksp, &pc));
            checkPetsc(PCSetType(pc, PCLU));
            checkPetsc(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
            // MUMPS controls can be supplied through PETSC_OPTIONS, prefixed
            // with -warpx_sparse_lu_. Keep this backend a cached direct solve.
            checkPetsc(KSPSetFromOptions(m_ksp));
            KSPType ksp_type = nullptr;
            PCType pc_type = nullptr;
            checkPetsc(KSPGetType(m_ksp, &ksp_type));
            checkPetsc(PCGetType(pc, &pc_type));
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::strcmp(ksp_type, KSPPREONLY) == 0 && std::strcmp(pc_type, PCLU) == 0,
                "sparse_lu requires KSPPREONLY and PCLU; do not override these PETSc options");
            MatSolverType solver_type = nullptr;
            checkPetsc(PCFactorGetMatSolverType(pc, &solver_type));
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                solver_type && std::strcmp(solver_type, MATSOLVERMUMPS) == 0,
                "sparse_lu requires the MUMPS factorization backend");
            checkPetsc(KSPSetErrorIfNotConverged(m_ksp, PETSC_TRUE));
            checkPetsc(KSPSetReusePreconditioner(m_ksp, PETSC_TRUE));
            checkPetsc(KSPSetUp(m_ksp)); // symbolic analysis and numeric factorization, once
            if (verbosity > 0) {
                amrex::Print() << "sparse_lu: cached MUMPS factorization of " << n
                               << " free nodes (" << nentries << " matrix entries)\n";
            }
        }
    }

    void Solve (amrex::MultiFab const& rhs, amrex::MultiFab& phi)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            phi.boxArray() == m_active.boxArray() &&
            phi.DistributionMap() == m_active.DistributionMap() &&
            rhs.boxArray() == m_active.boxArray() &&
            rhs.DistributionMap() == m_active.DistributionMap(),
            "sparse_lu requires a fixed grid and distribution (disable load balancing)");
        m_lift.setVal(amrex::Real(0));
        for (amrex::MFIter mfi(m_lift); mfi.isValid(); ++mfi) {
            auto const l = m_lift.array(mfi);
            auto const a = m_active.const_array(mfi);
            auto const p = phi.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    l(i,j,k) = (a(i,j,k) == amrex::Real(0)) ? p(i,j,k) : amrex::Real(0);
                });
        }
        amrex::MultiFab::Copy(m_rhs, rhs, 0, 0, 1, 0);
        m_rhs.OverrideSync(m_geom.periodicity());
        // Match MLMG's cut-edge row scaling. The original charge stays untouched.
        amrex::ignore_unused(m_linop.scaleRHS(0, &m_rhs));
        // b - L(lift), including the inhomogeneous embedded-boundary contribution.
        m_linop.solutionResidual(0, m_applied, m_lift, m_rhs);
        Gather(m_applied);
        if (amrex::ParallelDescriptor::MyProc() == 0) {
            auto const a = m_global[0].array();
            PetscScalar const* x = nullptr;
            if (m_ksp) {
                PetscScalar* b = nullptr;
                checkPetsc(VecGetArray(m_b, &b));
                for (amrex::BoxIterator it(m_domain); it.ok(); ++it) {
                    PetscInt const row = m_ids[m_domain.index(it())];
                    if (row >= 0) { b[row] = static_cast<PetscScalar>(a(it())); }
                }
                checkPetsc(VecRestoreArray(m_b, &b));
                checkPetsc(KSPSolve(m_ksp, m_b, m_x));
                KSPConvergedReason reason;
                checkPetsc(KSPGetConvergedReason(m_ksp, &reason));
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(reason > 0,
                    "sparse_lu: MUMPS solve failed");
                checkPetsc(VecGetArrayRead(m_x, &x));
            }
            for (amrex::BoxIterator it(m_domain); it.ok(); ++it) {
                PetscInt const row = m_ids[m_domain.index(it())];
                amrex::Real const value = (row >= 0)
                    ? static_cast<amrex::Real>(PetscRealPart(x[row])) : amrex::Real(0);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(value),
                    "sparse_lu: MUMPS returned a non-finite potential");
                a(it()) = value;
            }
            if (m_ksp) { checkPetsc(VecRestoreArrayRead(m_x, &x)); }
        }
        phi.ParallelCopy(m_global, 0, 0, 1);
        amrex::MultiFab::Add(phi, m_lift, 0, 0, 1, 0);
        phi.FillBoundary(m_geom.periodicity());
    }
};

SparseLUPoissonSolver::SparseLUPoissonSolver (
    amrex::Geometry const& geom, amrex::BoxArray const& grids,
    amrex::DistributionMapping const& dmap,
    amrex::FabFactory<amrex::FArrayBox> const* factory, int verbosity)
    : m_impl(std::make_unique<Impl>(geom, grids, dmap, factory, verbosity))
{}

SparseLUPoissonSolver::~SparseLUPoissonSolver () = default;

amrex::MLEBNodeFDLaplacian& SparseLUPoissonSolver::linop ()
{
    return m_impl->m_linop;
}

void SparseLUPoissonSolver::Solve (amrex::MultiFab const& rhs, amrex::MultiFab& phi)
{
    BL_PROFILE("SparseLUPoissonSolver::Solve");
    m_impl->Solve(rhs, phi);
}

#endif

/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "SparseLUPoissonSolver.H"

#include <AMReX.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EB2_GeometryShop.H>
#include <AMReX_EB2_IF_Sphere.H>
#include <AMReX_EBFabFactory.H>
#endif

#include <petscsys.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
    void test (bool with_eb)
    {
        using amrex::Real;
        // Multiple boxes exercise shared nodal values, ghosts and MPI transfers.
        amrex::Box domain(amrex::IntVect(0), amrex::IntVect(15));
        amrex::RealBox real_box({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
        amrex::Array<int, AMREX_SPACEDIM> periodic{};
        amrex::Geometry geom(domain, real_box, 0, periodic);
        amrex::BoxArray grids(domain);
        grids.maxSize(8);
        amrex::DistributionMapping dmap(grids);
        std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>> factory;
#ifdef AMREX_USE_EB
        if (with_eb) {
            amrex::EB2::SphereIF sphere(0.19, {AMREX_D_DECL(0.5,0.5,0.5)}, false);
            amrex::EB2::Build(amrex::EB2::makeShop(sphere), geom, 0, 0);
            factory = amrex::makeEBFabFactory(geom, grids, dmap, {2,2,2}, amrex::EBSupport::full);
        }
#endif
        auto const nodes = amrex::convert(grids, amrex::IntVect::TheNodeVector());
        amrex::MultiFab phi(nodes, dmap, 1, 1), reference(nodes, dmap, 1, 1);
        amrex::MultiFab rhs(nodes, dmap, 1, 0), error(nodes, dmap, 1, 0);
        SparseLUPoissonSolver direct(geom, grids, dmap, factory.get(), 1);
        amrex::MLEBNodeFDLaplacian reference_op;
        amrex::LPInfo info;
#ifdef AMREX_USE_EB
        if (with_eb) {
            reference_op.define({geom}, {grids}, {dmap}, info,
                {static_cast<amrex::EBFArrayBoxFactory const*>(factory.get())});
        } else
#endif
        {
            reference_op.define({geom}, {grids}, {dmap}, info);
        }
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc;
        bc.fill(amrex::LinOpBCType::Dirichlet);
        reference_op.setDomainBC(bc, bc);
        Real const eps = std::numeric_limits<Real>::epsilon();
        Real const tolerance = std::max(Real(1.e-8), Real(500)*eps);
        auto const dx = geom.CellSizeArray();

        // Reuse one LU factorization for different charges and nonzero BCs.
        for (int step = 1; step <= 3; ++step) {
#ifdef AMREX_USE_EB
            if (with_eb) {
                if (step == 2) {
                    auto potential = [=] AMREX_GPU_HOST_DEVICE
                        (AMREX_D_DECL(Real x, Real y, Real z)) noexcept {
                            amrex::ignore_unused(AMREX_D_DECL(x,y,z));
                            return step * (Real(1.7) + x);
                        };
                    direct.linop().setEBDirichlet(std::numeric_limits<Real>::lowest());
                    reference_op.setEBDirichlet(std::numeric_limits<Real>::lowest());
                    direct.linop().setEBDirichlet(potential);
                    reference_op.setEBDirichlet(potential);
                } else {
                    direct.linop().setEBDirichlet(Real(1.7)*step);
                    reference_op.setEBDirichlet(Real(1.7)*step);
                }
            }
#endif
            phi.setVal(Real(0));
            for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
                auto const p = phi.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        amrex::ignore_unused(k);
                        p(i,j,k) = step * (Real(1) + AMREX_D_TERM(
                            i*i*dx[0]*dx[0], + j*j*dx[1]*dx[1], + k*k*dx[2]*dx[2]));
                    });
            }
            amrex::MultiFab::Copy(reference, phi, 0, 0, 1, 1);
            amrex::MultiFab::Copy(error, phi, 0, 0, 1, 0); // analytic quadratic
            rhs.setVal(Real(2*AMREX_SPACEDIM*step));
            direct.Solve(rhs, phi);
            AMREX_ALWAYS_ASSERT(rhs.min(0) == Real(2*AMREX_SPACEDIM*step));
            AMREX_ALWAYS_ASSERT(rhs.max(0) == Real(2*AMREX_SPACEDIM*step));
            if (!with_eb) {
                amrex::MultiFab::Subtract(error, phi, 0, 0, 1, 0);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error.norm0() < tolerance,
                    "sparse LU failed the manufactured quadratic solution");
            }
            amrex::MLMG mlmg(reference_op);
            mlmg.setVerbose(0);
            mlmg.solve({&reference}, {&rhs}, std::max(Real(1.e-12), Real(10)*eps), Real(0));
            amrex::MultiFab::Copy(error, reference, 0, 0, 1, 0);
            amrex::MultiFab::Subtract(error, phi, 0, 0, 1, 0);
            // Covered nodes do not enter either linear system.
            direct.linop().setDirichletNodesToZero(0, 0, error);
            amrex::Print() << "step " << step << ", EB " << with_eb
                           << ": potential difference " << error.norm0() << '\n';
            AMREX_ALWAYS_ASSERT(error.norm0() < tolerance);
            amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> grad, grad_ref;
            amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> gp, rp;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                auto type = amrex::IntVect::TheNodeVector();
                type[d] = 0;
                auto ba = amrex::convert(grids, type);
                grad[d] = std::make_unique<amrex::MultiFab>(ba, dmap, 1, 0);
                grad_ref[d] = std::make_unique<amrex::MultiFab>(ba, dmap, 1, 0);
                gp[d] = grad[d].get();
                rp[d] = grad_ref[d].get();
            }
            direct.linop().compGrad(0, gp, phi, amrex::MLLinOp::Location::FaceCenter);
            mlmg.getGradSolution({rp});
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::MultiFab::Subtract(*gp[d], *rp[d], 0, 0, 1, 0);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(gp[d]->norm0() < Real(100)*tolerance,
                    "sparse LU and MLMG gradients differ");
            }
        }
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    bool external_petsc = false;
    amrex::ParmParse("test").query("external_petsc", external_petsc);
    if (external_petsc) {
        PETSC_COMM_WORLD = amrex::ParallelDescriptor::Communicator();
        AMREX_ALWAYS_ASSERT(PetscInitializeNoArguments() == 0);
    }
    {
        test(false);
#ifdef AMREX_USE_EB
        test(true);
#endif
    }
    if (external_petsc) {
        PetscBool finalized = PETSC_FALSE;
        AMREX_ALWAYS_ASSERT(PetscFinalized(&finalized) == 0);
        AMREX_ALWAYS_ASSERT(!finalized); // backend must not finalize a runtime it did not create
        AMREX_ALWAYS_ASSERT(PetscFinalize() == 0);
    }
    amrex::Finalize();
    PetscBool finalized = PETSC_FALSE;
    AMREX_ALWAYS_ASSERT(PetscFinalized(&finalized) == 0);
    AMREX_ALWAYS_ASSERT(finalized);
}

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
        SparseLUPoissonLinOp reference_op;
        amrex::LPInfo info;
#ifdef AMREX_USE_EB
        if (with_eb) {
            reference_op.define({geom}, {grids}, {dmap}, info,
                {static_cast<amrex::EBFArrayBoxFactory const*>(factory.get())});
        } else
#endif
        {
#if AMREX_SPACEDIM == 1
            reference_op.define({geom}, {grids}, {dmap}, info,
                amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, Real(1));
#else
            reference_op.define({geom}, {grids}, {dmap}, info);
#endif
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
#if AMREX_SPACEDIM > 1
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
#endif
        }
    }
    void test_boundaries (std::string const& mode, int n, bool with_eb)
    {
        using amrex::Real;
        using BC = SparseLUPoissonSolver::BoundaryType;
        SparseLUPoissonSolver::BoundaryArray lo{}, hi{};
        amrex::Array<int, AMREX_SPACEDIM> periodic{};
        amrex::GpuArray<int, AMREX_SPACEDIM> periodic_fun{}, linear_fun{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (mode == "all_periodic" ||
                (mode == "periodic" && d < AMREX_SPACEDIM-1) ||
                (mode == "periodic_open" && d == 0)) {
                lo[d] = hi[d] = BC::Periodic;
                periodic[d] = periodic_fun[d] = 1;
            } else if (mode == "all_open" ||
                       (mode == "open" && d < AMREX_SPACEDIM-1) ||
                       (mode == "periodic_open" && d > 0)) {
                lo[d] = hi[d] = BC::Open;
                linear_fun[d] = 1;
            } else if (mode == "one_sided_open" && d == 0) {
                lo[d] = BC::Open;
                linear_fun[d] = 1;
            }
        }
        // Shifted indices and cycle lengths 1, 2, 3k, 3k+1, 3k+2 exercise
        // periodic aliasing and coloring across MPI/box boundaries.
        amrex::IntVect const origin = with_eb ? amrex::IntVect(0) :
            amrex::IntVect(AMREX_D_DECL(2,-3,4));
        amrex::IntVect size(8);
        for (int d = 0; d < AMREX_SPACEDIM-1; ++d) { size[d] = n; }
        amrex::Box domain(origin, origin + size - amrex::IntVect(1));
        amrex::RealBox real_box({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
        amrex::Geometry geom(domain, real_box, 0, periodic);
        amrex::BoxArray grids(domain);
        grids.maxSize(4);
        amrex::DistributionMapping dmap(grids);
        std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>> factory;
#ifdef AMREX_USE_EB
        if (with_eb) {
            amrex::EB2::SphereIF sphere(0.16, {AMREX_D_DECL(0.5,0.5,0.5)}, false);
            amrex::EB2::Build(amrex::EB2::makeShop(sphere), geom, 0, 0);
            factory = amrex::makeEBFabFactory(geom, grids, dmap, {2,2,2}, amrex::EBSupport::full);
        }
#else
        amrex::ignore_unused(with_eb);
#endif
        auto const nodes = amrex::convert(grids, amrex::IntVect::TheNodeVector());
        amrex::MultiFab phi(nodes, dmap, 1, 1), exact(nodes, dmap, 1, 1);
        amrex::MultiFab rhs(nodes, dmap, 1, 0), weights(nodes, dmap, 1, 0);
        SparseLUPoissonSolver direct(geom, grids, dmap, factory.get(), 0, lo, hi);
        auto const dx = geom.CellSizeArray();
        Real const pi = std::acos(Real(-1));
        Real const tolerance = std::max(Real(1.e-8),
            Real(2000)*std::numeric_limits<Real>::epsilon());
        for (int step = 1; step <= 2; ++step) {
            auto potential = [=] AMREX_GPU_HOST_DEVICE
                (AMREX_D_DECL(Real x, Real y, Real z)) noexcept {
                amrex::GpuArray<Real, AMREX_SPACEDIM> const pos{AMREX_D_DECL(x,y,z)};
                Real value = 1;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    value += periodic_fun[d] ? Real(0.1)*std::cos(2*pi*pos[d]) :
                        (linear_fun[d] ? pos[d] : pos[d]*pos[d]);
                }
                return step * value;
            };
#ifdef AMREX_USE_EB
            if (with_eb) {
                direct.linop().setEBDirichlet(std::numeric_limits<Real>::lowest());
                direct.linop().setEBDirichlet(potential);
            }
#endif
            for (amrex::MFIter mfi(exact); mfi.isValid(); ++mfi) {
                auto const p = exact.array(mfi);
                auto const b = rhs.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        amrex::IntVect const q(AMREX_D_DECL(i,j,k));
                        amrex::GpuArray<Real, AMREX_SPACEDIM> pos{};
                        Real lap = 0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            pos[d] = (q[d]-origin[d])*dx[d];
                            if (periodic_fun[d]) {
                                lap += Real(0.2)*(std::cos(2*pi*dx[d])-1) /
                                    (dx[d]*dx[d]) * std::cos(2*pi*pos[d]);
                            } else if (!linear_fun[d]) { lap += 2; }
                        }
                        p(i,j,k) = potential(AMREX_D_DECL(pos[0], pos[1], pos[2]));
                        b(i,j,k) = step * lap;
                    });
            }
            if (with_eb) {
                // Manufacture the EB RHS from an exact field, then undo the
                // row weights that Solve applies. Open rows ignore this RHS.
                weights.setVal(Real(0));
                direct.linop().solutionResidual(0, rhs, exact, weights);
                rhs.mult(Real(-1));
                weights.setVal(Real(1));
                amrex::ignore_unused(direct.linop().scaleRHS(0, &weights));
                for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                    auto const b = rhs.array(mfi);
                    auto const w = weights.const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            if (w(i,j,k) != Real(0)) { b(i,j,k) /= w(i,j,k); }
                        });
                }
            }
            amrex::MultiFab::Copy(phi, exact, 0, 0, 1, 0);
            direct.Solve(rhs, phi);
            amrex::MultiFab::Subtract(phi, exact, 0, 0, 1, 0);
            direct.linop().setDirichletNodesToZero(0, 0, phi);
            amrex::Print() << mode << ", n=" << n << ", EB=" << with_eb
                           << ", step=" << step << ": error " << phi.norm0() << '\n';
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(phi.norm0() < tolerance,
                "sparse LU boundary test failed");
        }
    }
    // Check open rows (including corners), the interior Poisson residual, fixed
    // walls, and linearity across solves that share the cached factorization.
    // An all-open sphere/cylinder also has a known exterior potential.
    amrex::Real test_radial (int n, int wall, bool with_eb,
                            amrex::Real reference_radius = 0)
    {
        using amrex::Real;
        using BC = SparseLUPoissonSolver::BoundaryType;
        SparseLUPoissonSolver::BoundaryArray lo, hi;
        lo.fill(BC::Open);
        hi.fill(BC::Open);
        if (wall >= 0) { (wall % 2 == 0 ? lo : hi)[wall/2] = BC::Dirichlet; }
        amrex::IntVect const origin = with_eb ? amrex::IntVect(0) :
            amrex::IntVect(AMREX_D_DECL(2,-3,4));
        amrex::Box domain(origin, origin + amrex::IntVect(n-1));
        amrex::RealBox real_box({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
        if (!with_eb) {
            real_box = amrex::RealBox({AMREX_D_DECL(-1.,2.,-3.)},
                                     {AMREX_D_DECL(1.,3.,0.)});
        }
        amrex::Array<int, AMREX_SPACEDIM> periodic{};
        amrex::Geometry geom(domain, real_box, 0, periodic);
        amrex::BoxArray grids(domain);
        grids.maxSize(8);
        amrex::DistributionMapping dmap(grids);
        std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>> factory;
        Real constexpr sphere_radius = Real(0.2);
#ifdef AMREX_USE_EB
        if (with_eb) {
            amrex::EB2::SphereIF sphere(sphere_radius, {AMREX_D_DECL(0.5,0.5,0.5)}, false);
            amrex::EB2::Build(amrex::EB2::makeShop(sphere), geom, 0, 0);
            factory = amrex::makeEBFabFactory(geom, grids, dmap, {2,2,2}, amrex::EBSupport::full);
        }
#endif
        auto const nodes = amrex::convert(grids, amrex::IntVect::TheNodeVector());
        amrex::MultiFab phi(nodes, dmap, 1, 1), first(nodes, dmap, 1, 0);
        amrex::MultiFab rhs(nodes, dmap, 1, 0), residual(nodes, dmap, 1, 0);
        amrex::MultiFab scaled_rhs(nodes, dmap, 1, 0), error(nodes, dmap, 1, 0);
        SparseLUPoissonSolver direct(geom, grids, dmap, factory.get(), 0, lo, hi,
                                    reference_radius);
        auto const dx = geom.CellSizeArray();
        auto const problo = geom.ProbLoArray();
        amrex::GpuArray<Real, AMREX_SPACEDIM> center{};
        Real diagonal_squared = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            center[d] = geom.ProbLo(d) + Real(0.5)*geom.ProbLength(d);
            diagonal_squared += geom.ProbLength(d)*geom.ProbLength(d);
        }
        if (wall >= 0) {
            center[wall/2] = wall % 2 == 0 ? geom.ProbLo(wall/2) : geom.ProbHi(wall/2);
        }
        if (reference_radius == 0) { reference_radius = 3*std::sqrt(diagonal_squared); }
        bool const logarithmic = wall < 0 && AMREX_SPACEDIM == 2;
        Real const power = wall < 0 ? 1 : AMREX_SPACEDIM-1;
        Real const tolerance = std::max(Real(1.e-8),
            Real(10000)*std::numeric_limits<Real>::epsilon());
        Real analytic_error = 0;
        for (int step = 1; step <= 2; ++step) {
#ifdef AMREX_USE_EB
            if (with_eb) { direct.linop().setEBDirichlet(Real(step)); }
#endif
            phi.setVal(Real(step)); // Nonzero prescribed wall values exercise lifting.
            rhs.setVal(with_eb ? Real(0) : Real(-step));
            direct.Solve(rhs, phi);
            AMREX_ALWAYS_ASSERT(rhs.min(0) == (with_eb ? Real(0) : Real(-step)));
            AMREX_ALWAYS_ASSERT(rhs.max(0) == (with_eb ? Real(0) : Real(-step)));
            amrex::MultiFab::Copy(scaled_rhs, rhs, 0, 0, 1, 0);
            amrex::ignore_unused(direct.linop().scaleRHS(0, &scaled_rhs));
            direct.linop().solutionResidual(0, residual, phi, scaled_rhs);
            for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
                auto const p = phi.const_array(mfi);
                auto const r = residual.array(mfi);
                auto const e = error.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        amrex::IntVect const q(AMREX_D_DECL(i,j,k));
                        bool boundary = false;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            boundary |= q[d] == origin[d] || q[d] == origin[d]+n;
                        }
                        bool const fixed = wall >= 0 && q[wall/2] ==
                            origin[wall/2] + (wall % 2 == 0 ? 0 : n);
                        Real radius_squared = 0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            Real const x = problo[d] + (q[d]-origin[d])*dx[d] - center[d];
                            radius_squared += x*x;
                        }
                        if (fixed) { r(i,j,k) = p(i,j,k)-step; }
                        else if (boundary) {
                            Real value = p(i,j,k) * (logarithmic ?
                                1/std::log(reference_radius/std::sqrt(radius_squared)) : power);
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                Real const x = problo[d]+(q[d]-origin[d])*dx[d]-center[d];
                                if (x != Real(0)) {
                                    auto neighbor = q;
                                    neighbor[d] += x > 0 ? -1 : 1;
                                    value += std::abs(x)/dx[d] * (p(i,j,k)-p(neighbor));
                                }
                            }
                            r(i,j,k) = value;
                        }
                        e(i,j,k) = 0;
                        if (with_eb && wall < 0 && radius_squared > sphere_radius*sphere_radius) {
                            Real const radius = std::sqrt(radius_squared);
                            Real const exact = logarithmic ?
                                std::log(reference_radius/radius) /
                                    std::log(reference_radius/sphere_radius) : sphere_radius/radius;
                            e(i,j,k) = p(i,j,k)/step-exact;
                        }
                    });
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(residual.norm0() < tolerance,
                "radial boundary or interior Poisson equation failed");
            analytic_error = error.norm0();
            if (step == 1) { amrex::MultiFab::Copy(first, phi, 0, 0, 1, 0); }
            else {
                amrex::MultiFab::LinComb(error, Real(1), phi, 0, Real(-2), first, 0, 0, 1, 0);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error.norm0() < tolerance,
                    "cached radial solve is not linear in the RHS and fixed potentials");
            }
        }
        amrex::Print() << "radial n=" << n << ", wall=" << wall << ", EB=" << with_eb
                       << ": analytic error " << analytic_error << '\n';
        return analytic_error;
    }

}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    std::string boundary_failure;
    amrex::ParmParse("test").query("boundary_failure", boundary_failure);
    if (boundary_failure == "periodic") { test_boundaries("all_periodic", 8, false); }
    if (boundary_failure == "periodic_open") { test_boundaries("periodic_open", 8, false); }
    if (boundary_failure == "all_open_1d") { test_boundaries("all_open", 8, false); }
    if (boundary_failure == "invalid_reference") { test_radial(8, -1, false, 0.1); }
    if (boundary_failure == "unused_reference") { test_radial(8, 0, false, 10); }
#ifdef AMREX_USE_EB
    if (boundary_failure == "eb_near_open") { test_radial(2, -1, true); }
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(boundary_failure.empty(),
        "invalid boundary test unexpectedly returned");
    bool external_petsc = false;
    amrex::ParmParse("test").query("external_petsc", external_petsc);
    if (external_petsc) {
        PETSC_COMM_WORLD = amrex::ParallelDescriptor::Communicator();
        AMREX_ALWAYS_ASSERT(PetscInitializeNoArguments() == 0);
    }
    {
        test(false);
#if AMREX_SPACEDIM > 1
        for (int n : {1, 2, 5, 6, 7}) { test_boundaries("periodic", n, false); }
        for (int wall = -1; wall < 2*AMREX_SPACEDIM; ++wall) {
            test_radial(8, wall, false);
        }
        test_radial(1, -1, false);
#else
        test_radial(8, 0, false);
        test_radial(8, 1, false);
        test_radial(1, 0, false);
#endif
        test_radial(2, 0, false);
#if AMREX_SPACEDIM == 2
        test_radial(8, -1, false, 10);
#endif
#ifdef AMREX_USE_EB
        test(true);
        test_boundaries("periodic", 8, true);
        test_boundaries("all_periodic", 8, true);
        test_radial(8, 0, true);
        auto const coarse = test_radial(12, -1, true);
        auto const fine = test_radial(24, -1, true);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fine < coarse && fine < amrex::Real(0.1),
            "radial sphere/cylinder potential failed to converge to the exterior solution");
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

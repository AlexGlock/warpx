/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#if defined(WARPX_USE_SPARSE_LU) && \
    (defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D))

#include "SparseLUPoissonSolver.H"

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <petscksp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
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
#ifdef AMREX_USE_MPI
            PETSC_COMM_WORLD = amrex::ParallelDescriptor::Communicator();
#endif
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

    // Color the radius-one stencil, including the periodic seam. The last two
    // periodic nodes get separate colors when a cycle cannot use three colors.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int color (amrex::IntVect q, amrex::Box const& domain,
               amrex::GpuArray<int, AMREX_SPACEDIM> const& periodic)
    {
        int result = 0, stride = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            int const n = domain.length(d);
            int x = q[d] - domain.smallEnd(d);
            bool const extra = periodic[d] && n % 3 != 0;
            if (periodic[d]) { x = (x % n + n) % n; }
            int const c = (extra && x >= n-2) ? 3 + x - (n-2) : x % 3;
            result += stride * c;
            stride *= extra ? 5 : 3;
        }
        return result;
    }
}

struct SparseLUPoissonSolver::Impl
{
    amrex::Geometry m_geom;
    amrex::Box m_domain;
    BoundaryArray m_lo, m_hi;
    amrex::GpuArray<int, AMREX_SPACEDIM> m_periodic{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_open_center{};
    amrex::Real m_open_decay_power = 1;
    amrex::Real m_open_reference_radius = 0;
    bool m_open_logarithmic = false;
    struct OpenRow {
        PetscInt m_row;
        std::array<amrex::IntVect, AMREX_SPACEDIM+1> m_nodes;
        std::array<amrex::Real, AMREX_SPACEDIM+1> m_weights;
        std::array<amrex::Real, AMREX_SPACEDIM> m_cut_weights{};
        std::array<amrex::RealVect, AMREX_SPACEDIM> m_cut_positions{};
    };
    std::vector<OpenRow> m_open_rows;
    struct FaceValue {
        PetscInt row;
        amrex::IntVect low, high;
        amrex::Real weight;
        bool embedded;
        amrex::RealVect offset;
    };
    std::vector<FaceValue> m_face_values;
    std::map<PetscInt, PetscInt> m_face_ids;
    KSP m_face_solver = nullptr;
    Vec m_face_rhs = nullptr, m_background = nullptr;
    std::vector<PetscScalar> m_open_rhs;
    bool m_fixed_robin = false;
    bool m_robin_rhs_ready = false;
    SparseLUPoissonLinOp m_linop;
    amrex::MultiFab m_active, m_probe, m_applied, m_rhs, m_lift, m_global;
    std::vector<PetscInt> m_ids;
    Mat m_matrix = nullptr;
    KSP m_ksp = nullptr;
    Vec m_b = nullptr;
    Vec m_x = nullptr;

    // A single radial equation applies at open faces, edges and corners.
    // Dirichlet nodes take precedence through the AMReX active-node mask.
    bool IsOpenNode (amrex::IntVect const& q) const
    {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if ((q[d] == m_domain.smallEnd(d) && m_lo[d] == BoundaryType::Open) ||
                (q[d] == m_domain.bigEnd(d) && m_hi[d] == BoundaryType::Open)) {
                return true;
            }
        }
        return false;
    }

    bool Wrap (amrex::IntVect& q) const
    {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (m_periodic[d]) {
                int const n = m_domain.length(d);
                q[d] = m_domain.smallEnd(d) +
                    ((q[d] - m_domain.smallEnd(d)) % n + n) % n;
            }
        }
        return m_domain.contains(q);
    }

    void DefineBackground (std::map<PetscInt, std::array<amrex::Real, 4>> const& geometry)
    {
#ifdef WARPX_DIM_3D
        using amrex::Real;
        auto section = m_domain;
        section.setBig(1, section.smallEnd(1));
        auto opposite = [&](amrex::IntVect q) {
            q[1] = m_domain.bigEnd(1);
            return q;
        };
        auto id = [&](amrex::IntVect q) { return m_ids[m_domain.index(q)]; };
        auto embedded = [&](amrex::IntVect q) {
            return !geometry.empty() && geometry.at(m_domain.index(q))[3] >= 0;
        };
        PetscInt count = 0;
        for (amrex::BoxIterator it(section); it.ok(); ++it) {
            auto const q = it();
            auto const hi = opposite(q);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE((id(q) < 0) == (id(hi) < 0) &&
                embedded(q) == embedded(hi), "Robin faces have different electrode footprints");
            if (id(q) >= 0) {
                m_face_ids[id(q)] = count;
                m_face_ids[id(hi)] = count++;
            }
        }
        if (count == 0) { return; }
        Mat matrix;
        checkPetsc(MatCreateSeqAIJ(PETSC_COMM_SELF, count, count,
                                   std::min<PetscInt>(count, 5), nullptr, &matrix));
        for (amrex::BoxIterator it(section); it.ok(); ++it) {
            auto const q = it();
            if (id(q) < 0) { continue; }
            auto const row = m_face_ids.at(id(q));
            for (int d : {0, 2}) {
                std::array<amrex::IntVect, 2> neighbor{q, q};
                std::array<Real, 2> distance{};
                for (int s = 0; s < 2; ++s) {
                    auto& p = neighbor[s];
                    p[d] += s ? 1 : -1;
                    AMREX_ALWAYS_ASSERT(Wrap(p));
                    Real fraction = 1;
                    if (embedded(p)) {
                        auto const edge = s ? q : p;
                        auto length = [&](amrex::IntVect e) {
                            Real const c = geometry.at(m_domain.index(e))[d];
                            return std::abs(c) == 1 ? Real(1) : 1 - 2*std::abs(c);
                        };
                        fraction = length(edge);
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fraction > 0 &&
                            std::abs(fraction-length(opposite(edge))) <
                            128*std::numeric_limits<Real>::epsilon(),
                            "Robin faces have different cut edges");
                    }
                    distance[s] = fraction*m_geom.CellSize(d);
                }
                // Nonuniform three-point second derivative at a cut edge.
                for (int s = 0; s < 2; ++s) {
                    Real const w = 2 / ((distance[0]+distance[1])*distance[s]);
                    checkPetsc(MatSetValue(matrix, row, row, w, ADD_VALUES));
                    auto const p = neighbor[s];
                    if (id(p) >= 0) {
                        checkPetsc(MatSetValue(matrix, row, m_face_ids.at(id(p)), -w, ADD_VALUES));
                    } else {
                        bool const cut = embedded(p);
                        amrex::RealVect offset(0.0);
                        if (cut) { offset[d] = s ? distance[s] : -distance[s]; }
                        auto const base = cut ? q : p;
                        m_face_values.push_back({row, base, opposite(base), w, cut, offset});
                    }
                }
            }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_face_values.empty(),
            "Robin cross-section requires a fixed-potential wall or embedded electrode");
        checkPetsc(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY));
        checkPetsc(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY));
        checkPetsc(MatCreateVecs(matrix, &m_background, &m_face_rhs));
        checkPetsc(KSPCreate(PETSC_COMM_SELF, &m_face_solver));
        checkPetsc(KSPSetOperators(m_face_solver, matrix, matrix));
        checkPetsc(KSPSetType(m_face_solver, KSPPREONLY));
        PC pc;
        checkPetsc(KSPGetPC(m_face_solver, &pc));
        checkPetsc(PCSetType(pc, PCLU));
        checkPetsc(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
        checkPetsc(KSPSetErrorIfNotConverged(m_face_solver, PETSC_TRUE));
        checkPetsc(MatDestroy(&matrix));
#else
        amrex::ignore_unused(geometry);
#endif
    }

    void UpdateBackground (amrex::Array4<amrex::Real const> lift, EBPotential const& eb)
    {
        PetscInt count;
        checkPetsc(VecGetSize(m_face_rhs, &count));
        std::vector<PetscScalar> rhs(count, 0);
        for (auto const& term : m_face_values) {
            auto value = [&](amrex::IntVect q) {
                if (!term.embedded) { return lift(q); }
                AMREX_ALWAYS_ASSERT(eb);
                auto pos = term.offset;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    pos[d] += m_geom.ProbLo(d) +
                        (q[d]-m_geom.Domain().smallEnd(d))*m_geom.CellSize(d);
                }
                return eb(pos);
            };
            auto const lo = value(term.low);
            auto const hi = value(term.high);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(lo-hi) <=
                128*std::numeric_limits<amrex::Real>::epsilon()*
                std::max({amrex::Real(1), std::abs(lo), std::abs(hi)}),
                "Robin faces have different prescribed potentials");
            rhs[term.row] += term.weight * lo;
        }
        PetscScalar* b;
        checkPetsc(VecGetArray(m_face_rhs, &b));
        std::copy(rhs.begin(), rhs.end(), b);
        checkPetsc(VecRestoreArray(m_face_rhs, &b));
        checkPetsc(KSPSolve(m_face_solver, m_face_rhs, m_background));
    }

    ~Impl ()
    {
        if (m_face_solver) { checkPetsc(KSPDestroy(&m_face_solver)); }
        if (m_face_rhs) { checkPetsc(VecDestroy(&m_face_rhs)); }
        if (m_background) { checkPetsc(VecDestroy(&m_background)); }
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
          amrex::FabFactory<amrex::FArrayBox> const* factory, int verbosity,
          BoundaryArray const& lo, BoundaryArray const& hi, amrex::Real open_reference_radius)
        : m_geom(geom), m_domain(amrex::surroundingNodes(geom.Domain())),
          m_lo(lo), m_hi(hi), m_open_reference_radius(open_reference_radius)
    {
        initializePetsc();
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc, hibc;
        bool has_open = false;
        int reference_face = -1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            has_open |= lo[d] == BoundaryType::Open || hi[d] == BoundaryType::Open;
            m_open_center[d] = geom.ProbLo(d) + amrex::Real(0.5)*geom.ProbLength(d);
            if (reference_face < 0) {
                if (lo[d] == BoundaryType::Dirichlet) { reference_face = 2*d; }
                else if (hi[d] == BoundaryType::Dirichlet) { reference_face = 2*d+1; }
            }
            m_periodic[d] = geom.isPeriodic(d);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                (lo[d] == BoundaryType::Periodic) == bool(m_periodic[d]) &&
                (hi[d] == BoundaryType::Periodic) == bool(m_periodic[d]),
                "sparse_lu: periodic boundaries must be paired and match the geometry");
            if (m_periodic[d]) { m_domain.growHi(d, -1); }
            auto const convert = [](BoundaryType bc) {
                if (bc == BoundaryType::Periodic) { return amrex::LinOpBCType::Periodic; }
                // Neumann keeps open nodes active in the AMReX mask. Their
                // equations are replaced below by explicit radial-decay rows.
                if (bc == BoundaryType::Open) { return amrex::LinOpBCType::Neumann; }
                return amrex::LinOpBCType::Dirichlet;
            };
            lobc[d] = convert(lo[d]);
            hibc[d] = convert(hi[d]);
        }
        amrex::Real decay_length = 0;
        bool const robin = amrex::ParmParse("warpx").query("decay_length", decay_length);
        m_fixed_robin = robin;
        if (robin) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3 &&
                lo[1] == BoundaryType::Open && hi[1] == BoundaryType::Open &&
                std::count(lo.begin(), lo.end(), BoundaryType::Open) == 1 &&
                std::count(hi.begin(), hi.end(), BoundaryType::Open) == 1,
                "warpx.decay_length requires paired open y faces only in Cartesian 3D");
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!robin || (amrex::Math::isfinite(decay_length) &&
            decay_length > 0 && has_open && open_reference_radius == 0),
            "warpx.decay_length requires a finite positive length, open faces and no radial radius");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(robin || !has_open || !geom.isAnyPeriodic(),
            "sparse_lu: radial open boundaries cannot be combined with periodic boundaries");
#if AMREX_SPACEDIM == 1
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!has_open || reference_face >= 0,
            "sparse_lu: Cartesian 1D open boundaries require at least one Dirichlet face");
#endif
        if (reference_face >= 0) {
            int const d = reference_face/2;
            m_open_center[d] = reference_face % 2 ? geom.ProbHi(d) : geom.ProbLo(d);
            // A grounded infinite plane and its image charges have a dipolar far field.
            m_open_decay_power = AMREX_SPACEDIM-1;
        }
        // All-open 3D uses monopole decay. In 2D the monopole is logarithmic:
        // phi = A log(Rref/r), with A determined by charge, including induced EB charge.
        m_open_logarithmic = !robin && has_open && reference_face < 0 && AMREX_SPACEDIM == 2;
        if (m_open_logarithmic) {
            amrex::Real radius_squared = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                radius_squared += amrex::Real(0.25)*geom.ProbLength(d)*geom.ProbLength(d);
            }
            if (m_open_reference_radius == amrex::Real(0)) {
                m_open_reference_radius = 6 * std::sqrt(radius_squared);
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::Math::isfinite(m_open_reference_radius) &&
                m_open_reference_radius > std::sqrt(radius_squared),
                "sparse_lu: all-open 2D requires warpx.sparse_lu_open_reference_radius "
                "larger than the domain half-diagonal; this sets phi=0 at a finite radius "
                "for the logarithmic far field");
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_open_reference_radius == amrex::Real(0),
                "sparse_lu_open_reference_radius is only used for all-open Cartesian 2D");
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_domain.numPts() <= std::numeric_limits<PetscInt>::max(),
            "sparse_lu node numbering exceeds the supported integer range");
        amrex::LPInfo info;
        info.setMaxCoarseningLevel(0);
#ifdef AMREX_USE_EB
        auto const* eb_factory = dynamic_cast<amrex::EBFArrayBoxFactory const*>(factory);
        if (eb_factory && !robin) {
            // Radial differences use an uncut boundary cell layer. Do not apply
            // a regular-grid boundary equation through an embedded conductor.
            auto const& flags = eb_factory->getMultiEBCellFlagFab();
            int clear = 1;
            for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi) {
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    for (int side = 0; side < 2; ++side) {
                        if ((side == 0 ? lo[d] : hi[d]) != BoundaryType::Open) { continue; }
                        auto strip = geom.Domain();
                        if (side == 0) { strip.setBig(d, strip.smallEnd(d)); }
                        else { strip.setSmall(d, strip.bigEnd(d)); }
                        auto const bx = strip & mfi.validbox();
                        if (bx.ok() && flags[mfi].getType(bx) != amrex::FabType::regular) {
                            clear = 0;
                        }
                    }
                }
            }
            amrex::ParallelDescriptor::ReduceIntMin(clear);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(clear,
                "sparse_lu: the cell layer next to open faces must be free of EB");
        }
        if (eb_factory) {
            m_linop.define({geom}, {grids}, {dmap}, info, {eb_factory});
        } else
#else
        amrex::ignore_unused(factory);
#endif
        {
#if AMREX_SPACEDIM == 1
            m_linop.define({geom}, {grids}, {dmap}, info,
                amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{},
                amrex::Real(1));
#else
            m_linop.define({geom}, {grids}, {dmap}, info);
#endif
        }
        m_linop.setDomainBC(lobc, hibc);
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
        std::vector<amrex::Real> row_sum(n, 0), row_norm(n, 0);
        int ncolors = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            ncolors *= (m_periodic[d] && m_domain.length(d) % 3 != 0) ? 5 : 3;
        }
        auto const domain = m_domain;
        auto const periodic = m_periodic;
        for (int c = 0; c < ncolors; ++c) {
            m_probe.setVal(amrex::Real(0));
            for (amrex::MFIter mfi(m_probe); mfi.isValid(); ++mfi) {
                auto const p = m_probe.array(mfi);
                auto const a = m_active.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        amrex::IntVect const q(AMREX_D_DECL(i,j,k));
                        p(i,j,k) = (color(q, domain, periodic) == c)
                            ? a(i,j,k) : amrex::Real(0);
                    });
            }
            m_linop.apply(0, 0, m_applied, m_probe,
                amrex::MLLinOp::BCMode::Homogeneous, amrex::MLLinOp::StateMode::Correction);
            Gather(m_applied);
            if (amrex::ParallelDescriptor::MyProc() == 0) {
                auto const a = m_global[0].const_array();
                for (amrex::BoxIterator it(m_domain); it.ok(); ++it) {
                    PetscInt const row = m_ids[m_domain.index(it())];
                    if (row < 0 || IsOpenNode(it())) { continue; }
                    // On cycles of one or two cells several stencil offsets
                    // name the same unknown. The probe already sums them.
                    std::array<PetscInt, 2 * AMREX_SPACEDIM + 1> seen{};
                    int nseen = 0;
                    for (int s = 0; s < 2 * AMREX_SPACEDIM + 1; ++s) {
                        auto q = it();
                        if (s > 0) { q[(s-1)/2] += (s % 2) ? -1 : 1; }
                        if (!Wrap(q)) { continue; }
                        PetscInt const col = m_ids[m_domain.index(q)];
                        if (col >= 0 && color(q, domain, periodic) == c &&
                            std::find(seen.begin(), seen.begin()+nseen, col) ==
                                seen.begin()+nseen) {
                            seen[nseen++] = col;
                            checkPetsc(MatSetValue(m_matrix, row, col,
                                static_cast<PetscScalar>(a(it())), INSERT_VALUES));
                            row_sum[row] += a(it());
                            row_norm[row] += std::abs(a(it()));
                            ++nentries;
                        }
                    }
                }
            }
        }
        // Cache the vacuum fraction of each inward edge ending on a conductor.
        // Tangential edge centroids and level-set signs on the two y faces.
        std::map<PetscInt, std::array<amrex::Real, 4>> face_geometry;
#if defined(WARPX_DIM_3D) && defined(AMREX_USE_EB)
        if (robin && eb_factory) {
            Gather(eb_factory->getLevelSet());
            auto save_face = [&](int component) {
                if (amrex::ParallelDescriptor::MyProc() != 0) { return; }
                auto const a = m_global[0].const_array();
                for (amrex::BoxIterator it(domain); it.ok(); ++it) {
                    if (IsOpenNode(it())) {
                        face_geometry[domain.index(it())][component] = a(it());
                    }
                }
            };
            save_face(3);
            for (int d : {0, 2}) {
                auto const& edges = *eb_factory->getEdgeCent()[d];
                m_probe.setVal(1);
                for (amrex::MFIter mfi(m_probe); mfi.isValid(); ++mfi) {
                    if (!edges.ok(mfi)) { continue; }
                    auto const ec = edges.const_array(mfi);
                    auto const a = m_probe.array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            if (ec.contains(i,j,k)) { a(i,j,k) = ec(i,j,k); }
                        });
                }
                Gather(m_probe);
                save_face(d);
            }
        }
#endif
        std::map<std::pair<PetscInt, int>, amrex::Real> cut_fractions;
#ifdef AMREX_USE_EB
        if (robin && eb_factory) {
            auto const& flags = eb_factory->getMultiEBCellFlagFab();
            auto const& centroids = eb_factory->getEdgeCent();
            auto const& level_set = eb_factory->getLevelSet();
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (lo[d] != BoundaryType::Open && hi[d] != BoundaryType::Open) { continue; }
                m_probe.setVal(0);
                for (amrex::MFIter mfi(m_probe); mfi.isValid(); ++mfi) {
                    auto const type = flags[mfi].getType();
                    if (type == amrex::FabType::covered) { continue; }
                    bool const cut = type != amrex::FabType::regular;
                    auto const ec = cut ? centroids[d]->const_array(mfi) :
                        amrex::Array4<amrex::Real const>{};
                    auto const ls = level_set.const_array(mfi);
                    auto const fraction = m_probe.array(mfi);
                    for (int side = 0; side < 2; ++side) {
                        if ((side ? hi[d] : lo[d]) != BoundaryType::Open) { continue; }
                        auto face = domain;
                        int const end = side ? domain.bigEnd(d) : domain.smallEnd(d);
                        face.setSmall(d, end);
                        face.setBig(d, end);
                        amrex::ParallelFor(face & mfi.validbox(),
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                                amrex::ignore_unused(j, k); // AMREX_D_DECL drops inactive axes.
                                amrex::IntVect q(AMREX_D_DECL(i,j,k));
                                auto inward = q;
                                inward[d] += side ? -1 : 1;
                                if (ls(q) < 0 && ls(inward) >= 0) {
                                    auto const centroid = cut ? ec(side ? inward : q) : 1;
                                    fraction(q) = std::abs(centroid) == 1 ? 1 :
                                        1 - 2 * std::abs(centroid);
                                }
                            });
                    }
                }
                Gather(m_probe);
                if (amrex::ParallelDescriptor::MyProc() == 0) {
                    auto const f = m_global[0].const_array();
                    for (amrex::BoxIterator it(m_domain); it.ok(); ++it) {
                        if (f(it()) > 0) {
                            cut_fractions[{m_domain.index(it()), d}] = f(it());
                        }
                    }
                }
            }
        }
#endif
        if (amrex::ParallelDescriptor::MyProc() == 0 && n > 0) {
            if (robin) { DefineBackground(face_geometry); }
            for (amrex::BoxIterator it(m_domain); it.ok(); ++it) {
                PetscInt const row = m_ids[m_domain.index(it())];
                if (row < 0 || !IsOpenNode(it())) { continue; }
                OpenRow boundary_row{row, {}, {}};
                boundary_row.m_nodes.fill(it());
                boundary_row.m_weights[0] = robin ? 0 : m_open_decay_power;
                if (m_open_logarithmic) {
                    amrex::Real r2 = 0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        amrex::Real const r = geom.ProbLo(d) +
                            (it()[d]-geom.Domain().smallEnd(d))*geom.CellSize(d) - m_open_center[d];
                        r2 += r*r;
                    }
                    boundary_row.m_weights[0] =
                        1 / std::log(m_open_reference_radius/std::sqrt(r2));
                }
                // (x-x0).grad(phi) + beta*phi = 0. Use first-order, one-sided
                // derivatives toward x0 in every coordinate, including tangents.
                // In 2D/3D, beta > 0 makes the row strictly diagonally dominant.
                // The beta=0 closure in 1D is anchored by its Dirichlet wall.
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    amrex::Real r = geom.ProbLo(d) +
                        (it()[d]-geom.Domain().smallEnd(d))*geom.CellSize(d) - m_open_center[d];
                    if (robin) {
                        bool const lo_open = m_lo[d] == BoundaryType::Open &&
                            it()[d] == m_domain.smallEnd(d);
                        bool const hi_open = m_hi[d] == BoundaryType::Open &&
                            it()[d] == m_domain.bigEnd(d);
                        if (!lo_open && !hi_open) { continue; }
                        // Sum phi + R*dphi/dn = 0 over incident open faces at edges.
                        r = lo_open ? -decay_length : decay_length;
                        boundary_row.m_weights[0] += 1;
                    }
                    if (r == amrex::Real(0)) { continue; }
                    amrex::Real const weight = std::abs(r)*geom.InvCellSize(d);
                    boundary_row.m_nodes[d+1][d] += r > 0 ? -1 : 1;
                    auto const cut = cut_fractions.find({m_domain.index(it()), d});
                    if (cut != cut_fractions.end()) {
                        auto const w = weight / cut->second;
                        boundary_row.m_weights[0] += w;
                        boundary_row.m_cut_weights[d] = w;
                        auto& pos = boundary_row.m_cut_positions[d];
                        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
                            pos[axis] = geom.ProbLo(axis) + geom.CellSize(axis) *
                                (it()[axis] - geom.Domain().smallEnd(axis));
                        }
                        pos[d] += (r > 0 ? -1 : 1) * cut->second * geom.CellSize(d);
                        continue;
                    }
                    AMREX_ALWAYS_ASSERT(m_domain.contains(boundary_row.m_nodes[d+1]));
                    boundary_row.m_weights[d+1] = -weight;
                    boundary_row.m_weights[0] += weight;
                }
                m_open_rows.push_back(boundary_row);
                for (int s = 0; s <= AMREX_SPACEDIM; ++s) {
                    auto const& q = boundary_row.m_nodes[s];
                    PetscInt const col = m_ids[m_domain.index(q)];
                    amrex::Real const value = boundary_row.m_weights[s];
                    if (value == amrex::Real(0)) { continue; }
                    if (col >= 0) {
                        checkPetsc(MatSetValue(m_matrix, row, col, value, INSERT_VALUES));
                        row_sum[row] += value;
                        row_norm[row] += std::abs(value);
                        ++nentries;
                    }
                }
            }
            bool anchored = false;
            for (PetscInt row = 0; row < n; ++row) {
                anchored |= std::abs(row_sum[row]) >
                    100 * std::numeric_limits<amrex::Real>::epsilon() * row_norm[row];
            }
            // Radial decay anchors open domains; fully periodic domains still
            // need fixed-potential embedded conductors. Do not add artificial gauges.
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(anchored,
                "sparse_lu requires fixed-potential domain or embedded boundaries; "
                "unanchored periodic Poisson problems are not supported");
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
            auto const error = KSPSetUp(m_ksp); // factorize once
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error == 0,
                "sparse_lu: MUMPS factorization failed; check PETSc diagnostics and that "
                "the geometry and boundary conditions define a nonsingular problem");
            if (verbosity > 0) {
                amrex::Print() << "sparse_lu: cached MUMPS factorization of " << n
                               << " free nodes (" << nentries << " matrix entries)\n";
                if (robin) {
                    amrex::Print() << "sparse_lu: normal Robin decay length = "
                                   << decay_length << '\n';
                } else if (has_open) {
                    amrex::Print() << "sparse_lu: radial open center =";
                    for (auto x : m_open_center) { amrex::Print() << " " << x; }
                    if (m_open_logarithmic) {
                        amrex::Print() << ", logarithmic reference radius = "
                                       << m_open_reference_radius << '\n';
                    } else {
                        amrex::Print() << ", decay power = " << m_open_decay_power << '\n';
                    }
                }
            }
        }
    }

    void Solve (amrex::MultiFab const& rhs, amrex::MultiFab& phi, EBPotential const& eb)
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
        // Gather collectively even on ranks that own no open rows.
        bool has_open = false;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            has_open |= m_lo[d] == BoundaryType::Open || m_hi[d] == BoundaryType::Open;
        }
        if (has_open && !m_robin_rhs_ready) {
            Gather(m_lift);
            if (amrex::ParallelDescriptor::MyProc() == 0) {
                m_open_rhs.clear();
                m_open_rhs.reserve(m_open_rows.size());
                auto const lift = m_global[0].const_array();
                PetscScalar const* background = nullptr;
                if (m_face_solver) {
                    UpdateBackground(lift, eb);
                    checkPetsc(VecGetArrayRead(m_background, &background));
                }
                for (auto const& row : m_open_rows) {
                    amrex::Real value = background ?
                        PetscRealPart(background[m_face_ids.at(row.m_row)]) : 0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        if (row.m_cut_weights[d] > 0) {
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(eb, "Robin cut edge needs EB potential");
                            value += row.m_cut_weights[d] * eb(row.m_cut_positions[d]);
                        }
                    }
                    for (int s = 0; s <= AMREX_SPACEDIM; ++s) {
                        value -= row.m_weights[s] * lift(row.m_nodes[s]);
                    }
                    m_open_rhs.push_back(static_cast<PetscScalar>(value));
                }
                if (background) { checkPetsc(VecRestoreArrayRead(m_background, &background)); }
            }
            // All ranks must skip the collective lift gather on subsequent Robin solves.
            m_robin_rhs_ready = m_fixed_robin;
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
                for (std::size_t i = 0; i < m_open_rows.size(); ++i) {
                    b[m_open_rows[i].m_row] = m_open_rhs[i];
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
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::Math::isfinite(value),
                    "sparse_lu: MUMPS returned a non-finite potential");
                a(it()) = value;
            }
            if (m_ksp) { checkPetsc(VecRestoreArrayRead(m_x, &x)); }
        }
        phi.ParallelCopy(m_global, 0, 0, 1, 0, 0, m_geom.periodicity());
        amrex::MultiFab::Add(phi, m_lift, 0, 0, 1, 0);
        phi.FillBoundary(m_geom.periodicity());
    }
};

SparseLUPoissonSolver::SparseLUPoissonSolver (
    amrex::Geometry const& geom, amrex::BoxArray const& grids,
    amrex::DistributionMapping const& dmap,
    amrex::FabFactory<amrex::FArrayBox> const* factory, int verbosity,
    BoundaryArray const& lo, BoundaryArray const& hi, amrex::Real open_reference_radius)
    : m_impl(std::make_unique<Impl>(geom, grids, dmap, factory, verbosity, lo, hi,
                                  open_reference_radius))
{}

SparseLUPoissonSolver::~SparseLUPoissonSolver () = default;

SparseLUPoissonLinOp& SparseLUPoissonSolver::linop ()
{
    return m_impl->m_linop;
}

void SparseLUPoissonSolver::Solve (amrex::MultiFab const& rhs, amrex::MultiFab& phi,
                                   EBPotential const& eb)
{
    BL_PROFILE("SparseLUPoissonSolver::Solve");
    m_impl->Solve(rhs, phi, eb);
}

#endif

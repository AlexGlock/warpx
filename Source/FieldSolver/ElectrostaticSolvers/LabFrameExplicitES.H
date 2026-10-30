/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_LABFRAMEEXPLICITES_H_
#define WARPX_LABFRAMEEXPLICITES_H_

#include "ElectrostaticSolver.H"

#if defined(WARPX_USE_SPARSE_LU) && \
    (defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D))
#include "SparseLUPoissonSolver.H"
#endif

class LabFrameExplicitES final : public ElectrostaticSolver
{
public:

    explicit LabFrameExplicitES (int nlevs_max) : ElectrostaticSolver (nlevs_max) {
        ReadParameters();
    }

    void InitData () override;

    void ComputeSpaceChargeField (
        ablastr::fields::MultiFabRegister& fields,
        MultiParticleContainer& mpc,
        MultiFluidContainer* mfl,
        int max_level) override;

    void computePhiTriDiagonal (
        const ablastr::fields::MultiLevelScalarField& rho,
        const ablastr::fields::MultiLevelScalarField& phi
    );

    void computePhiTriDiagonal_periodic (
        const ablastr::fields::MultiLevelScalarField& rho,
        const ablastr::fields::MultiLevelScalarField& phi
    );

private:
#if defined(WARPX_USE_SPARSE_LU) && \
    (defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D))
    std::unique_ptr<SparseLUPoissonSolver> m_sparse_lu;
    std::unique_ptr<amrex::MultiFab> m_sparse_lu_rhs;
    void ComputePhiSparseLU (
        amrex::MultiFab const& rho, amrex::MultiFab& phi,
        ablastr::fields::MultiLevelVectorField const& efield);
#endif

};

#endif  // WARPX_LABFRAMEEXPLICITES_H_

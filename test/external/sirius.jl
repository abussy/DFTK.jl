@testitem "Test SIRIUS closed-shell" tags=[:dont_test_windows] setup=[TestCases] begin

    # If not set before using SIRIUS, library uses all available threads and performs badly
    ENV["OMP_NUM_THREADS"] = "1"
    using DFTK
    using SIRIUS
    silicon = TestCases.silicon
    
    ### Testing a closed-shell calculation with SIRIUS
    Si = ElementPsp(:Si, psp=load_psp(silicon.psp_upf))
    atoms = [Si, Si]
    positions = [ones(3)/8.2, -ones(3)/8]

    Ecut = 12.0
    kgrid = [2, 2, 2]
    temp = 0.01
    tol = 1.0e-8
    maxiter = 20

    model = model_DFT(silicon.lattice, atoms, positions;
                      functionals=LDA(), temperature=temp)

    #DFTK reference
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)
    scfres = self_consistent_field(basis, tol=tol; mixing=SimpleMixing(), maxiter=maxiter)
    etot_dftk = scfres.energies.total
    forces_dftk = compute_forces(scfres)
    stress_dftk = compute_stresses_cart(scfres)

    #SIRIUS results
    basis = SiriusBasis(model; Ecut=Ecut, kgrid=kgrid)
    scfres = self_consistent_field(basis, tol=tol; mixing=SimpleMixing(), maxiter=maxiter)
    etot_sirius = scfres.energies.total
    forces_sirius = compute_forces(scfres)
    stress_sirius = compute_stresses_cart(scfres)

    @testset "SIRIUS silicon LDA" begin
        @test abs(etot_dftk - etot_sirius) < 1.0e-6
        fdiff = forces_dftk-forces_sirius
        @test maximum(abs.(vcat([fd for fd in fdiff]...))) < 1.0e-7
        @test maximum(abs.(Array(stress_dftk-stress_sirius))) < 1.0e-7
    end
end

@testitem "Test SIRIUS open-shell" tags=[:dont_test_windows] setup=[TestCases] begin

    # If not set before using SIRIUS, library uses all available threads and performs badly
    ENV["OMP_NUM_THREADS"] = "1"
    using DFTK
    using SIRIUS
    using LazyArtifacts
    iron_bcc = TestCases.iron_bcc
    
    ### Testing an open-shell calculation with SIRIUS
    oncv_pbe_family  = artifact"sg15_2022.02.06_upf"
    Fe = ElementPsp(:Fe, psp=load_psp(joinpath(oncv_pbe_family, "Fe_ONCV_PBE-1.2.upf")))
    atoms = [Fe]

    Ecut = 16.0
    kgrid = [3, 3, 3]
    temp = 0.01
    tol = 1.0e-8
    maxiter = 30
    magnetic_moments = [4.0]

    model = model_DFT(iron_bcc.lattice, atoms, iron_bcc.positions; functionals=PBE(), 
                      magnetic_moments=magnetic_moments, temperature=temp, 
                      smearing=Smearing.Gaussian())

    #DFTK reference
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)
    ρ0 = guess_density(basis, magnetic_moments)
    scfres = self_consistent_field(basis, tol=tol; mixing=SimpleMixing(), maxiter=maxiter, ρ=ρ0)
    etot_dftk = scfres.energies.total
    forces_dftk = compute_forces(scfres)

    #SIRIUS results
    basis = SiriusBasis(model; Ecut=Ecut, kgrid=kgrid)
    ρ0 = guess_density(basis, magnetic_moments)
    scfres = self_consistent_field(basis, tol=tol; mixing=SimpleMixing(), maxiter=maxiter, ρ=ρ0)
    etot_sirius = scfres.energies.total
    forces_sirius = compute_forces(scfres)

    @testset "SIRIUS iron-bcc PBE" begin
        @test abs(etot_dftk - etot_sirius) < 1.0e-5
        fdiff = forces_dftk-forces_sirius
        @test maximum(abs.(vcat([fd for fd in fdiff]...))) < 1.0e-6
    end
end
#### Using a CallBack  
using DifferentialEquations, Plots, LinearAlgebra, Distributions, OffsetArrays, Random

function Valve(R, deltaP)
    q = 0.0
    if (-deltaP) < 0.0 
        q =  deltaP/R
    else
        q = 0.0
    end
    return q

end

function ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    global tr
    #tᵢ = rem(t + (1 - Eshift) * τ, τ)
    tᵢ = t - tr
    Eₚ = (tᵢ <= τₑₛ) * (1 - cos(tᵢ / τₑₛ * pi)) / 2 +
         (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * (1 + cos((tᵢ - τₑₛ) / (τₑₚ - τₑₛ) * pi)) / 2 +
         (tᵢ <= τₑₚ) * 0

    E = Eₘᵢₙ + (Eₘₐₓ - Eₘᵢₙ) * Eₚ

    return E
end



function DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    global tr
    #tᵢ = rem(t + (1 - Eshift) * τ, τ)
    tᵢ = t - tr
    DEₚ = (tᵢ <= τₑₛ) * pi / τₑₛ * sin(tᵢ / τₑₛ * pi) / 2 +
          (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * pi / (τₑₚ - τₑₛ) * sin((τₑₛ - tᵢ) / (τₑₚ - τₑₛ) * pi) / 2
    (tᵢ <= τₑₚ) * 0
    DE = (Eₘₐₓ - Eₘᵢₙ) * DEₚ

    return DE
end

#Shi timing parameters
Eshift = 0.0
Eₘᵢₙ = 0.03
τₑₛ = 0.3
τₑₚ = 0.45 
Eₘₐₓ = 1.5
Rmv = 0.06



function NIK!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u 
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = p
    # pressures (more readable names)
# the differential equations
    du[1] = (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) + pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # 1 Left Ventricle
    du[2] = (Qav - Qs ) / Csa #Systemic arteries     
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # volume
    du[5]    = Valve(Zao, (pLV - psa)) - Qav # AV 
    du[6]   = Valve(Rmv, (psv - pLV)) - Qmv # MV
    du[7]     = (du[2] - du[3]) / Rs # Systemic flow
    nothing 
end
##
M = [1.  0  0  0  0  0  0
     0  1.  0  0  0  0  0
     0  0  1.  0  0  0  0
     0  0  0  1.  0  0  0
     0  0  0  0  0  0  0
     0  0  0  0  0  0  0 
     0  0  0  0  0  0  1. ]

Nik_ODE = ODEFunction(NIK!,mass_matrix=M)

u0 = [8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0]

p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

c = 16 # number of cycles needed plus 1 
function HRV(c)
    t_τL = zeros(c) 
    t_τL[1] = rand(Uniform(0.8,1.1),1)[1]
    for i in 1:c-1
        t_τL[i+1] = t_τL[i] + rand(Uniform(0.8,1.1),1)[1]
    end 
    return t_τL
end 

t_τL = HRV(c)


τ :: Float64 = t_τL[1]
tr = 0.0
tspan = (0, 15)
prob = ODEProblem(Nik_ODE, u0, tspan, p)


function condition(u,t,integrator)
    global τ
    integrator.t - tr > τ
end

n :: Int = 0
# need counter 
function affect!(integrator)
    global n
    global tr
    n = n + 1
    τ_new = t_τL[n+1] - t_τL[n]
    #print(τ_new)
    global τ = τ_new
    tr = t_τL[n]
end


save_positions = (false,false)

cb = DiscreteCallback(condition, affect!, save_positions=save_positions)

@time sol = solve(prob, Rodas5P(autodiff = false), adaptive = false, dt = 0.002, reltol = 1e-8, abstol = 1e-8, callback = cb)

plot(sol, label = ["P_LV" "P_SA" "P_SV" "V_LV" "Q_av" "Q_mv" "Q_s"], tspan = (10*τ, 13*τ))

plot(sol, idxs = [1,2,3], tspan = (10*τ, 13*τ))
plot(sol, idxs = 4, tspan = (10*τ, 13*τ))

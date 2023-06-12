using MPI
using ADIOS2
using LinearAlgebra
using Printf
using Statistics
using ImplicitGlobalGrid
using ParallelStencil
using CUDA

@views d_xa(A) = A[2:end  , :     , :     ] .- A[1:end-1, :     , :     ];
@views d_xi(A) = A[2:end  ,2:end-1,2:end-1] .- A[1:end-1,2:end-1,2:end-1];
@views d_ya(A) = A[ :     ,2:end  , :     ] .- A[ :     ,1:end-1, :     ];
@views d_yi(A) = A[2:end-1,2:end  ,2:end-1] .- A[2:end-1,1:end-1,2:end-1];
@views  inn(A) = A[2:end-1,2:end-1,2:end-1]

# Physics
lam        = 1.0                 # Thermal conductivity
cp_min     = 1.0                 # Minimal heat capacity
lx, ly     = 10.0, 10.0          # Length of computational domain in dimension x and y

nx, ny     = 128, 128
nt         = 10000
me, dims   = init_global_grid(nx, ny, 0)
#nx_g       = dims[1]*(nx-2) + 2
#ny_g       = dims[2]*(ny-2) + 2
nx_v = (nx-2)*dims[1]
ny_v = (ny-2)*dims[2]
dx         = lx/(nx_g()-1)
dy         = ly/(ny_g()-1)

T     = CUDA.zeros(Float64, nx,   ny)
Cp    = CUDA.zeros(Float64, nx,   ny)
dTedt = CUDA.zeros(Float64, nx-2, ny-2)
qTx   = CUDA.zeros(Float64, nx-1, ny-2)
qTy   = CUDA.zeros(Float64, nx-2, ny-1)

x0    = coords[1]*(nx-2)*dx
y0    = coords[2]*(ny-2)*dy

T_v  = zeros(nx_v, ny_v)


Cp .= cp_min .+ CuArray([5*exp(-((x_g(ix,dx,Cp)-lx/1.5))^2-((y_g(iy,dy,Cp)-ly/2))^2 +
                         5*exp(-((x_g(ix,dx,Cp)-lx/3.0))^2-((y_g(iy,dy,Cp)-ly/2))^2 for ix=1:size(T,1), iy=1:size(T,2)])
T  .= CuArray([100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2 +
                50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2 for ix=1:size(T,1), iy=1:size(T,2)])
# ADIOS2
# (size and start of the local and global problem)
#nxy_nohalo   = [nx-2, ny-2]
#nxy_g_nohalo = [nx_g-2, ny_g-2]
#start        = [coords[1]*nxy_nohalo[1], coords[2]*nxy_nohalo[2]]
T_nohalo     = zeros(nx-2, ny-2)                                  # Preallocate array for writing temperature
# (intialize ADIOS2, io, engine and define the variable temperature)
adios = ADIOS2.adios_init_mpi(joinpath(pwd(),"adios2.xml"), comm) # Use the configurations defined in "adios2.xml"...
io = ADIOS2.declare_io(adios, "IO")                               # ... in the section "writerIO"
T_id = define_variable(io, "temperature", eltype(T))              # Define the variable "temperature"
engine = ADIOS2.open(io, "diffusion2D.bp", mode_write)            # Open the file/stream "diffusion2D.bp"

# Time loop
nsteps = 50
dt     = min(dx,dy)^2*cp_min/lam/4.1
global t      = 0
tic    = time()

for it in 1:nt
    if it % (nt/nsteps) == 0                                     # Write data only nsteps times
        T_nohalo[:] = T[2:end-1, 2:end-1]                        # Copy data removing the halo
        gather!(T_nohalo, T_v)
        begin_step(engine)                                       # Begin ADIOS2 write step
        put!(engine, T_id, T_nohalo)                             # Add T (without halo) to variables for writing
        end_step(engine)                                         # End ADIOS2 write step (includes normally the actual writing of data)
        println("Time step " * string(it) * "...")
    end
    qx    .= -lam.*d_xi(T)./dx;             # Fourier's law of heat conduction: q_x   = -λ δT/δx
    qy    .= -lam.*d_yi(T)./dy;               # ...                               q_y   = -λ δT/δy
    dTedt .= 1.0./inn(Cp).*(-d_xa(qx)./dx .- d_ya(qy)./dy                                             - δq_y/dy)
    T[2:end-1,2:end-1] = T[2:end-1,2:end-1] + dt*dTedt           # Update of temperature             T_new = T_old + δT/δt
    global t            = t + dt                                 # Elapsed physical time
    update_halo!(T)                     # Update the halo of T
end

close(engine)

@printf("\ntime: %.8f\n", time() - tic)
@printf("Min. temperature: %2.2e\n", minimum(T))
@printf("Max. temperature: %2.2e\n", maximum(T))

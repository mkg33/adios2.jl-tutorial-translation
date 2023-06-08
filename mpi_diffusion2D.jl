using MPI
#using Plots
using ADIOS2
using LinearAlgebra
using Printf
using Statistics


function update_halo(A, neighbors_x, neighbors_y)
    if neighbors_x[1] >= 0
        sendbuf = copy(A[2:end-1,:])
        recvbuf = zeros(size(A,2))
        MPI.Sendrecv!(sendbuf, neighbors_x[1], 1, recvbuf, neighbors_x[1], 0)
        A[end,:] = recvbuf
    end
    if neighbors_x[2] >= 0
        sendbuf = copy(A[2:end-1,:])
        recvbuf = zeros(size(A,2))
        MPI.Sendrecv!(sendbuf, neighbors_x[2], 0, recvbuf, neighbors_x[2], 1)
        A[1,:] = recvbuf
    end
    if neighbors_y[1] >= 0
        sendbuf = copy(A[:,2:end-1])
        recvbuf = zeros(size(A,1))
        MPI.Sendrecv!(sendbuf, neighbors_y[1], 3, recvbuf, neighbors_y[1], 2)
        A[:,end] = recvbuf
    end
    if neighbors_y[2] >= 0
        sendbuf = copy(A[:,2:end-1])
        recvbuf = zeros(size(A,1))
        MPI.Sendrecv!(sendbuf, neighbors_y[2], 2, recvbuf, neighbors_y[2], 3)
        A[:,1] = recvbuf
    end
end

# MPI
MPI.Init()
nprocs      = MPI.Comm_size(MPI.COMM_WORLD)
dims        = MPI.Dims_create!(nprocs, [0,0])
comm        = MPI.Cart_create(MPI.COMM_WORLD, dims)
me          = MPI.Comm_rank(comm)
coords      = MPI.Cart_coords(comm)
neighbors_x = MPI.Cart_shift(comm, 0, 1)
neighbors_y = MPI.Cart_shift(comm, 1, 1)

# Physics
lam        = 1.0                 # Thermal conductivity
cp_min     = 1.0                 # Minimal heat capacity
lx, ly     = 10.0, 10.0          # Length of computational domain in dimension x and y

nx, ny     = 128, 128
nt         = 10000
nx_g       = dims[1]*(nx-2) + 2
ny_g       = dims[2]*(ny-2) + 2
dx         = lx/(nx_g-1)
dy         = ly/(ny_g-1)

T     = zeros(nx,   ny)
Cp    = zeros(nx,   ny)
dTedt = zeros(nx-2, ny-2)
qTx   = zeros(nx-1, ny-2)
qTy   = zeros(nx-2, ny-1)

x0    = coords[1]*(nx-2)*dx
y0    = coords[2]*(ny-2)*dy


Cp[:] = cp_min .+ reshape([  5*exp(-((x0 + ix*dx - lx/1.5)/1.0)^2 - ((y0 + iy*dy - ly/1.5)/1.0)^2) +
                                            5*exp(-((x0 + ix*dx - lx/1.5)/1.0)^2 - ((y0 + iy*dy - ly/3.0)/1.0)^2) for ix in 1:nx, iy in 1:ny], (nx,ny))


T[:]  =          reshape([100*exp(-((x0 + ix*dx - lx/3.0)/2.0)^2 - ((y0 + iy*dy - ly/2.0)/2.0)^2) +
                                          50*exp(-((x0 + ix*dx - lx/1.5)/2.0)^2 - ((y0 + iy*dy - ly/2.0)/2.0)^2) for ix in 1:nx, iy in 1:ny], (nx,ny))

# ADIOS2
# (size and start of the local and global problem)
nxy_nohalo   = [nx-2, ny-2]
nxy_g_nohalo = [nx_g-2, ny_g-2]
start        = [coords[1]*nxy_nohalo[1], coords[2]*nxy_nohalo[2]]
T_nohalo     = zeros(nx-2, ny-2)                               # Preallocate array for writing temperature
# (intialize ADIOS2, io, engine and define the variable temperature)
adios = ADIOS2.adios_init_mpi(joinpath(pwd(),"adios2.xml"), comm) # Use the configurations defined in "adios2.xml"...
io = ADIOS2.declare_io(adios, "IO") # ... in the section "writerIO"
T_id = define_variable(io, "temperature", eltype(T))

#print(T_id)

#T_id   = ADIOS2.define_variable(io, "temperature", typeof(T), nxy_g_nohalo, start, nxy_nohalo, true)  # Define the variable "temperature"
engine = ADIOS2.open(io, "diffusion2D.bp", mode_write) # Open the file/stream "diffusion2D.bp"

# Time loop
nsteps = 50
dt     = min(dx,dy)^2*cp_min/lam/4.1
global t      = 0
tic    = time()

for it in 1:nt
    if it % (nt/nsteps) == 0
                          # Write data only nsteps times


        T_nohalo[:] = T[2:end-1, 2:end-1]
                               # Copy data removing the halo
        begin_step(engine)                                       # Begin ADIOS2 write step
        put!(engine, T_id, T_nohalo)                               # Add T (without halo) to variables for writing
        end_step(engine)                                         # End ADIOS2 write step (includes normally the actual writing of data)
        println("Time step " * string(it) * "...")
    end
    qTx[:]       = -lam*diff(T[:,2:end-1],dims=1)/dx             # Fourier's law of heat conduction: q_x   = -λ δT/δx
    qTy[:]       = -lam*diff(T[2:end-1,:],dims=2)/dy             # ...                               q_y   = -λ δT/δy
    dTedt[:]     = 1.0./Cp[2:end-1,2:end-1].*(-diff(qTx,dims=1)
                                       /dx - diff(qTy,dims=2)/dy)   # Conservation of energy:           δT/δt = 1/cₚ(-δq_x/δx
                                          #                                               - δq_y/dy)
    T[2:end-1,2:end-1] = T[2:end-1,2:end-1] + dt*dTedt           # Update of temperature             T_new = T_old + δT/δt
    global t            = t + dt                                        # Elapsed physical time
    update_halo(T, neighbors_x, neighbors_y)                     # Update the halo of T
end

close(engine)
MPI.Finalize()

@printf("\ntime: %.8f\n", time() - tic)
@printf("Min. temperature: %2.2e\n", minimum(T))
@printf("Max. temperature: %2.2e\n", maximum(T))

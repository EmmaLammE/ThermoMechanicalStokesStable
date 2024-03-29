
# import Pkg;
using HDF5
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2)
using Statistics, LinearAlgebra, CUDA, Interpolations
using Plots, Printf, MAT, CSV, DataFrames, Base
using Distributed,SharedArrays

import ParallelStencil: INDICES
const ix,iy=INDICES[1],INDICES[2]
macro sum_IBM_steny(A) esc(:(($A[$ix,$iy]+$A[$ix,$iy+1]+$A[$ix,$iy+2]+$A[$ix,$iy+3]))) end
macro sum_IBM_stenx(A,idx) esc(:(sum($A[$idx[($ix+1)]]))) end




@parallel function compute_timesteps!(dτVx::Data.Array,dτVy::Data.Array,dτPt::Data.Array,dτT::Data.Array,Rho::Data.Array,Mus::Data.Array,Kappa::Data.Array,Vsc::Data.Number,Ptsc::Data.Number,min_dxy2::Data.Number,max_nxy::Int,cp::Data.Number,dt::Data.Number,eta_b::Data.Number)
    @all(dτVx) = Vsc*min_dxy2/@av_xi(Mus)/4.1/(1+eta_b)
    @all(dτVy) = Vsc*min_dxy2/@av_yi(Mus)/4.1/(1+eta_b)
    @all(dτPt) = Ptsc*4.1*@all(Mus)/max_nxy*(1+eta_b)
    @all(dτT)  = ((min_dxy2/@all(Kappa)*@all(Rho)*cp/4.1).^(-1)+1/dt).^(-1)
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, dτPt::Data.Array, Rho::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)/@all(Rho)
    return
end

@parallel function compute_E_τ!(∇V::Data.Array,Exx::Data.Array,Eyy::Data.Array,Exy::Data.Array,Exyn::Data.Array,τxx::Data.Array,τyy::Data.Array,τxy::Data.Array,Vx::Data.Array,Vy::Data.Array,Mus::Data.Array,Mus_s::Data.Array,dx::Data.Number,dy::Data.Number)
    @all(Exx) = @d_xa(Vx)/dx
    @all(Eyy) = @d_ya(Vy)/dy 
    @all(Exy) = 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @inn(Exyn) = @av(Exy)
    @all(Mus_s) = @av(Mus)
    @all(τxx) = 2.0*@all(Mus)*@all(Exx) # - 1.0/3.0*@all(∇V))
    @all(τyy) = 2.0*@all(Mus)*@all(Eyy) # - 1.0/3.0*@all(∇V))
    @all(τxy) = 2.0*@all(Mus_s)*@all(Exy)
    return
end

@parallel function test!(∇V::Data.Array,Exx::Data.Array,Eyy::Data.Array,Exy::Data.Array,Exyn::Data.Array,τxx::Data.Array,τyy::Data.Array,τxy::Data.Array,Vx::Data.Array,Vy::Data.Array,Mus::Data.Array,Mus_s::Data.Array,dx::Data.Number,dy::Data.Number)
    @all(Exy) =(@d_yi(Vx))
    return
end

@parallel function compute_Mus_s!(Mus::Data.Array,Mus_s::Data.Array)
    @all(Mus_s) = @av(Mus)
    return
end

@parallel function compute_viscosity!(∇V::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array)
    # @all(τxx) = 2.0*@all(Mus)*(@d_xa(Vx)/dx) # - 1.0/3.0*@all(∇V))
    # @all(τyy) = 2.0*@all(Mus)*(@d_ya(Vy)/dy) # - 1.0/3.0*@all(∇V))
    # @all(τxy) = 2.0*@av(Mus)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, Pt::Data.Array, Rogx::Data.Array, Rogy::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dampX::Data.Number, dampY::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(Rx)    = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx + @all(Rogx)
    @all(Ry)    = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @all(Rogy)
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    return
end

@parallel function compute_Hs!(Exx::Data.Array,Eyy::Data.Array,Exyn::Data.Array,EII2::Data.Array,Hs::Data.Array,Mus::Data.Array)
    @all(EII2) = 0.5*((@all(Exx)).^2+(@all(Eyy)).^2)+(@all(Exyn)).^2+1e-20
    @all(Hs) = 4.0*@all(Mus)*@all(EII2)
    return
end

@parallel function compute_qT!(Tt::Data.Array,qxT::Data.Array,qyT::Data.Array,Kappa_qTx::Data.Array,Kappa_qTy::Data.Array,dx::Data.Number,dy::Data.Number)
    @inn_x(qxT) = -@all(Kappa_qTx)*@d_xa(Tt)/dx
    @inn_y(qyT) = -@all(Kappa_qTy)*@d_ya(Tt)/dy
    return
end

@parallel function compute_weaken_factor!(H::Data.Array,Tt::Data.Array,T_melt::Data.Number)
    @all(H) = 1-tanh(-(@all(Tt)-T_melt-0.15)./0.5)
    return
end

@parallel function compute_dT!(Tt::Data.Array,T_o::Data.Array,qxT::Data.Array,qyT::Data.Array,dTdt::Data.Array,dTdt_o::Data.Array,T_res::Data.Array,Hs::Data.Array,H::Data.Array,Rho::Data.Array,cp::Data.Number,ρ::Data.Number,CN::Data.Number,dt::Data.Number,dx::Data.Number,dy::Data.Number)
    @all(dTdt) = 1/@all(Rho)/cp*(-@d_xa(qxT)/dx-@d_ya(qyT)/dy+@all(Hs)*(1.0-@all(H)))
    @all(T_res) = -(@all(Tt)-@all(T_o))/dt+(1-CN)*@all(dTdt)+CN*@all(dTdt_o)
    return
end

@parallel function compute_T!(Tt::Data.Array,T_res::Data.Array,dτT::Data.Array)
    @all(Tt) = @all(Tt)+@all(dτT)*@all(T_res)
    return
end

@parallel function compute_dTdx_dTdy!(Tt::Data.Array,dTdx::Data.Array,dTdy::Data.Array,dx::Data.Number,dy::Data.Number)
    @inn_x(dTdx) = @d_xa(Tt)/dx/2
    @inn_y(dTdy) = @d_ya(Tt)/dy/2
    return
end

@parallel function advect_temp!(Tt::Data.Array,Vxin::Data.Array,Vyin::Data.Array,dTdx::Data.Array,dTdy::Data.Array,dt::Data.Number)
    @inn(Tt) = @inn(Tt) - (@inn(Vxin)*@inn(dTdx)*dt + @inn(Vyin)*@inn(dTdy)*dt)
    return
end

function print_debug(τxx,τyy,τxy,Pt,Rogx,Rogy,Rx,Ry)
    print("\nτxx[2,1:5]: ",τxx[2,1:5],"\n τyy[2,1:5]: ",τyy[2,1:5],
    "\n τxy[2,1:5]: ",τxy[2,1:5],"\n Pt[2,1:5]: ",Pt[2,1:5],
    "\n Rogx[2,1:5]: ",Rogx[2,1:5],"\n Rogy[2,1:5]: ",Rogy[2,1:5],
    "\n Rx[2,1:5]: ", Rx[2,1:5],"\n Ry[2,1:5]: ", Ry[2,1:5],
    "\n @d_ya(τxy)/dy: ", τxy[2,3:7]-τxy[2,2:6],
    "\n @av_xi(Rogx): ", (Rogx[2,3:7]+Rogx[2,2:6])*0.5,"\n\n "
    )
end

@parallel_indices (ix,iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix,iy) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

## Rheology models:
@parallel function compute_glen!(EII2::Data.Array,Tt::Data.Array,Mus::Data.Array,Mus_phy::Data.Array,Musit::Data.Array,H::Data.Array,LS_H::Data.Array,a0::Data.Number,npow::Data.Number,mpow::Data.Number,Q0::Data.Number,R::Data.Number,T_surf::Data.Number,μs0::Data.Number,rele::Data.Number,eta_air::Data.Number)
    @all(Mus_phy) = 0.5*(a0^(-1/npow)).*(@all(EII2).^(mpow)).*(exp.(Q0./(npow*R*(T_surf+@all(Tt)))))
    @all(Musit) = ((@all(Mus_phy)).^(-1)+1/μs0).^(-1)
    @all(Musit) = @all(Musit)*(1-@all(H))
    @all(Mus) = exp.(rele*log.(@all(Musit))+(1-rele)*log.(@all(Mus)))
    @all(Mus) = eta_air + (@all(Mus) - eta_air)*@all(LS_H)
    return
end



###################### IBM functions ########################
#~~~~~~~~~~~~~~~~ parallel ~~~~~~~~~~~~~~~~~~~~#
@parallel function take_av(Vx::Data.Array,Vy::Data.Array,Vxin::Data.Array,Vyin::Data.Array)
    @all(Vxin)=@av_xa(Vx)
    @all(Vyin)=@av_ya(Vy)
    return
end

@parallel function np_compute_weigted_u_euler2lag!(u::Data.Array,v::Data.Array,IBM_delta::Data.Array,IBM_fx::Data.Array,IBM_fy::Data.Array)
    @all(IBM_fx)=@all(u)*@all(IBM_delta)
    @all(IBM_fy)=@all(v)*@all(IBM_delta)
    return
end

@parallel function sum_IBM!(IBM_fx::Data.Array,IBM_fy::Data.Array,IBM_fxTemp::Data.Array,IBM_fyTemp::Data.Array)
    @all(IBM_fxTemp)=@sum_IBM_steny(IBM_fx)
    @all(IBM_fyTemp)=@sum_IBM_steny(IBM_fy)
   return
end

@parallel function IBM_velocity_correction!(IBM_vx::Data.Array,IBM_vy::Data.Array,Vx::Data.Array,Vy::Data.Array)
    @inn(Vx)=@inn(Vx)+@inn(IBM_vx)
    @inn(Vy)=@inn(Vy)+@inn(IBM_vy)
    return
end

@parallel function par_desired_T!(IBM_Ttd::Data.Array,IBM_dTdxLag::Data.Array,IBM_dTdyLag::Data.Array,IBM_lagNormalX::Data.Array,IBM_lagNormalY::Data.Array,q_geo::Data.Number)
    @all(IBM_dTdxLag)=@all(IBM_dTdxLag)*@all(IBM_lagNormalX)
    @all(IBM_dTdyLag)=@all(IBM_dTdyLag)*@all(IBM_lagNormalY)
    @all(IBM_Ttd)=2*q_geo-@all(IBM_dTdxLag)-@all(IBM_dTdyLag)
    return
end

@parallel function IBM_temperature_correction!(IBM_T_correction::Data.Array,Tt::Data.Array,dτT::Data.Array)
    @all(Tt)=@all(Tt)+@all(IBM_T_correction)*@all(dτT)
    return
end

@parallel function compute_desired_V!(IBM_fxd::Data.Array,IBM_fyd::Data.Array,IBM_fxLag::Data.Array,IBM_fyLag::Data.Array,ud::Data.Number,vd::Data.Number)
    # IBM_fxd,IBM_fyd=(ud.-IBM_fxLag)./phi2,(vd.-IBM_fyLag)./phi2
    @all(IBM_fxd) = ud - @all(IBM_fxLag)*2.0
    @all(IBM_fyd) = vd - @all(IBM_fyLag)*2.0
    return
end

# norm2_vx, norm2_vy, norm2_p = sqrt(sum((Vxa.-Vxin).^2*dx*dy)), sqrt(sum((Vya.-Vyin).^2*dx*dy)), sqrt(sum((Pa.-Pt).^2)*dx*dy)
# norm_vx, norm_vy, norm_p,Vxin,Vyin,Pt,Vxa,Vya,Pa,dx,dy
@parallel function compute_norm2!(norm_vx::Data.Array,norm_vy::Data.Array,norm_p::Data.Array,Vxin::Data.Array,Vyin::Data.Array,Pt::Data.Array,Vxa::Data.Array,Vya::Data.Array,Pa::Data.Array,dx::Data.Number,dy::Data.Number)
    @all(norm_vx) = (@all(Vxa)-@all(Vxin))*(@all(Vxa)-@all(Vxin))*dx*dy
    @all(norm_vy) = (@all(Vya)-@all(Vyin))*(@all(Vya)-@all(Vyin))*dx*dy
    @all(norm_p) = (@all(Pa)-@all(Pt))*(@all(Pa)-@all(Pt))*dx*dy
    return
end

@parallel_indices (ix,iy) function Exyn_2sides!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix,iy) function BC_periodic!(Vx::Data.Array)
    Vx[1,iy]   = Vx[end-1,iy]
    Vx[end,iy] = Vx[2,iy]
    return
end

@parallel_indices (ix,iy) function BC_vx_inlet!(Vx::Data.Array,Vx_in_ice::Data.Array)
    Vx[1,iy] = 2*Vx_in_ice[iy] - Vx[2,iy]
    return
end

@parallel_indices (ix,iy) function BC_vy_inlet!(Vy::Data.Array,Vy_in_ice::Data.Array)
    Vy[1,iy] = Vy_in_ice[iy]
    return
end

@parallel_indices (ix,iy) function BC_vx_outlet!(Vx::Data.Array)
    Vx[end,iy] = Vx[end-1,iy]
    return
end

@parallel_indices (ix,iy) function BC_vy_outlet!(Vy::Data.Array)
    Vy[end,iy] = Vy[end-1,iy]
    return
end

@parallel_indices (ix,iy) function BC_Tt_outlet!(Tt::Data.Array)
    Tt[end,iy] = Tt[end-1,iy]
    return
end

@parallel_indices (ix,iy) function BC_vx_bot!(Vx::Data.Array)
    Vx[ix,1] = 0.0
    return
end

@parallel_indices (ix,iy) function BC_vy_bot!(Vy::Data.Array)
    Vy[ix,1] = -Vy[ix,2]
    return
end

function par_getU_euler2lag(IBM_idxx,IBM_idxy,Vxin,Vyin,Vx_euler2lag,Vy_euler2lag,IBM_sten)
    numLag=size(IBM_idxx,1)
    arrx = SharedArray{Float64}(IBM_sten*numLag,IBM_sten)
    arry = SharedArray{Float64}(IBM_sten*numLag,IBM_sten)
    Vxin_h,Vyin_h=zeros(size(Vxin)),zeros(size(Vyin))
    copyto!(Vxin_h, Vxin);copyto!(Vyin_h, Vyin)
    # time0 = Base.time();
    @sync @distributed for i=1:numLag
        arrx[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vxin_h[IBM_idxx[i,:],IBM_idxy[i,:]]
        arry[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vyin_h[IBM_idxx[i,:],IBM_idxy[i,:]]
    end
    # partime = Base.time()-time0
    # print("\n\nPar time get U: ",partime)
    return Data.Array(arrx), Data.Array(arry)

    # time0 = Base.time();
    # for i=1:numLag
    #     Vx_euler2lag[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vxin[IBM_idxx[i,:],IBM_idxy[i,:]]
    #     Vy_euler2lag[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vyin[IBM_idxx[i,:],IBM_idxy[i,:]]
    # end
    # partime = Base.time()-time0
    # print("\nSer time get U: ",partime)
    # return Vx_euler2lag, Vy_euler2lag 
end

function par_getT_euler2lag(IBM_idxx,IBM_idxy,dTdx,dTdy,IBM_sten)
    numLag=size(IBM_idxx,1)
    arrTx = SharedArray{Float64}(IBM_sten*numLag,IBM_sten)
    arrTy = SharedArray{Float64}(IBM_sten*numLag,IBM_sten)
    dTdx_h,dTdy_h=zeros(size(dTdx)),zeros(size(dTdy))
    copyto!(dTdx_h, dTdx);copyto!(dTdy_h, dTdy)
    @sync @distributed for i=1:numLag
        arrTx[(i-1)*IBM_sten+1:i*IBM_sten,:]=dTdx_h[IBM_idxx[i,:],IBM_idxy[i,:]]
        arrTy[(i-1)*IBM_sten+1:i*IBM_sten,:]=dTdy_h[IBM_idxx[i,:],IBM_idxy[i,:]]
    end
    return Data.Array(arrTx), Data.Array(arrTy)
end

function par_getUT_euler2lag(IBM_idxx,IBM_idxy,Vxin,Vyin,Tt,Vx_euler2lag,Vy_euler2lag,Tt_euler2lag,IBM_sten)
    numLag=size(IBM_idxx,1)
    arrx = SharedArray{Float64}(IBM_sten*numLag,IBM_sten)
    arry = SharedArray{Float64}(IBM_sten*numLag,IBM_sten)
    arrT = SharedArray{Float64}(IBM_sten*numLag,IBM_sten)
    Vxin_h,Vyin_h,T_h=zeros(size(Vxin)),zeros(size(Vyin)),zeros(size(Tt))
    copyto!(Vxin_h, Vxin);copyto!(Vyin_h, Vyin);copyto!(T_h, Tt)
    # time0 = Base.time();
    @sync @distributed for i=1:numLag
        arrx[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vxin_h[IBM_idxx[i,:],IBM_idxy[i,:]]
        arry[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vyin_h[IBM_idxx[i,:],IBM_idxy[i,:]]
        arrT[(i-1)*IBM_sten+1:i*IBM_sten,:]=T_h[IBM_idxx[i,:],IBM_idxy[i,:]]
    end
    # partime = Base.time()-time0
    # print("\n\nPar time get U: ",partime)
    return Data.Array(arrx), Data.Array(arry),Data.Array(arrT)

    # time0 = Base.time();
    # for i=1:numLag
    #     Vx_euler2lag[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vxin[IBM_idxx[i,:],IBM_idxy[i,:]]
    #     Vy_euler2lag[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vyin[IBM_idxx[i,:],IBM_idxy[i,:]]
    # end
    # partime = Base.time()-time0
    # print("\nSer time get U: ",partime)
    # return Vx_euler2lag, Vy_euler2lag 
end

function par_sumEuler2Lag(IBM_fxTemp,IBM_fyTemp,IBM_fxLag,IBM_fyLag,IBM_sten,numLag)
    arrx = SharedVector{Float64}(numLag)
    arry = SharedVector{Float64}(numLag)
    IBM_fxTemp_h,IBM_fyTemp_h=zeros(size(IBM_fxTemp)),zeros(size(IBM_fyTemp))
    copyto!(IBM_fxTemp_h, IBM_fxTemp);copyto!(IBM_fyTemp_h, IBM_fyTemp)

    # time0 = Base.time();
    @sync @distributed for i=1:numLag
        arrx[i] = sum(IBM_fxTemp_h[Int((i-1)*IBM_sten+1):1:Int(i*IBM_sten)])
        arry[i] = sum(IBM_fyTemp_h[Int((i-1)*IBM_sten+1):1:Int(i*IBM_sten)])
    end
    return Data.Array(arrx), Data.Array(arry) 
    # partime = Base.time()-time0
    # print("\nPar time: ",partime)
    # print("\nIBM_fxLag[4], IBM_fxLag[5]: ",arrx[4],", ", arry[5])

    # time0 = Base.time();
    # for i=1:numLag
    #     IBM_fxLag[i] = sum(IBM_fxTemp[Int((i-1)*IBM_sten+1):1:Int(i*IBM_sten)])
    #     IBM_fyLag[i] = sum(IBM_fyTemp[Int((i-1)*IBM_sten+1):1:Int(i*IBM_sten)])
    # end
    # partime = Base.time()-time0
    # print("\nSer time: ",partime)
    # print("\nIBM_fxLag[4], IBM_fxLag[5]: ",IBM_fxLag[4],", ", IBM_fyLag[5])
    # exit()
    # return IBM_fxLag,IBM_fyLag
end

function par_lag2euler(IBM_idxx,IBM_idxy,IBM_delta,IBM_fxd,IBM_fyd,IBM_vx_correction,IBM_vy_correction,IBM_sten)
    IBM_fx_correction_h = SharedArray{Float64}(size(IBM_vy_correction,1),size(IBM_vx_correction,2))
    IBM_fy_correction_h = SharedArray{Float64}(size(IBM_vy_correction,1),size(IBM_vx_correction,2))
    IBM_fxd_h,IBM_fyd_h,IBM_delta_h=zeros(size(IBM_fxd)),zeros(size(IBM_fyd)),zeros(size(IBM_delta))
    copyto!(IBM_fxd_h, IBM_fxd);copyto!(IBM_fyd_h, IBM_fyd);copyto!(IBM_delta_h, IBM_delta)
    numLag=size(IBM_idxx,1)
    # time0 = Base.time();
    # tempx,tempy=0.0,0.0
    @sync @distributed for i=1:numLag
        IBM_fx_correction_h[IBM_idxx[i,:],IBM_idxy[i,:]] = IBM_fx_correction_h[IBM_idxx[i,:],IBM_idxy[i,:]] .+ IBM_fxd_h[i]*IBM_delta_h[(i-1)*IBM_sten+1:i*IBM_sten,:]
        IBM_fy_correction_h[IBM_idxx[i,:],IBM_idxy[i,:]] = IBM_fy_correction_h[IBM_idxx[i,:],IBM_idxy[i,:]] .+ IBM_fyd_h[i]*IBM_delta_h[(i-1)*IBM_sten+1:i*IBM_sten,:]
        # tempx = IBM_fx_correction[IBM_idxx[i,:],IBM_idxy[i,:]]
        # tempy = IBM_fy_correction[IBM_idxx[i,:],IBM_idxy[i,:]]
    end
    # partime = Base.time()-time0
    IBM_vx_correction[1,:]=(3*IBM_fx_correction_h[1,:]-IBM_fx_correction_h[2,:])./2
    IBM_vx_correction[end,:]=(3*IBM_fx_correction_h[end,:]-IBM_fx_correction_h[end-1,:])./2
    IBM_vx_correction[2:end-1,:]=(IBM_fx_correction_h[1:end-1,:]+IBM_fx_correction_h[2:end,:])./2
    IBM_vy_correction[:,1]=(3*IBM_fy_correction_h[:,1]-IBM_fy_correction_h[:,2])./2
    IBM_vy_correction[:,end]=(3*IBM_fy_correction_h[:,end]-IBM_fy_correction_h[:,end-1])./2
    IBM_vy_correction[:,2:end-1]=(IBM_fy_correction_h[:,1:end-1]+IBM_fy_correction_h[:,2:end])./2

    # print("\nPar time: ",partime)
    # print("\nvx_cor[1,5:10], vy_cor[1,5:10]: ",IBM_vx_correction[1,5:10],", ",IBM_vy_correction[1,5:10])
    return Data.Array(IBM_vx_correction),Data.Array(IBM_vy_correction)
end

function par_lag2eulerT(IBM_idxx,IBM_idxy,IBM_delta,IBM_Ttd,IBM_T_correction,IBM_sten)
    IBM_T_correction_h = SharedArray{Float64}(size(IBM_T_correction,1),size(IBM_T_correction,2))
    IBM_Ttd_h,IBM_delta_h=zeros(size(IBM_Ttd)),zeros(size(IBM_delta))
    copyto!(IBM_Ttd_h, IBM_Ttd);copyto!(IBM_delta_h, IBM_delta)
    numLag=size(IBM_idxx,1)
    @sync @distributed for i=1:numLag
        IBM_T_correction_h[IBM_idxx[i,:],IBM_idxy[i,:]] = IBM_T_correction_h[IBM_idxx[i,:],IBM_idxy[i,:]] .+ IBM_Ttd_h[i]*IBM_delta_h[(i-1)*IBM_sten+1:i*IBM_sten,:]
    end
    return Data.Array(IBM_T_correction_h)
end

# need to be parallelized
function getU_euler2lag(IBM_idxx,IBM_idxy,Vxin,Vyin,Vx_euler2lag,Vy_euler2lag,IBM_sten)
    numLag=size(IBM_idxx,1)
    for i=1:numLag
        Vx_euler2lag[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vxin[IBM_idxx[i,:],IBM_idxy[i,:]]
        Vy_euler2lag[(i-1)*IBM_sten+1:i*IBM_sten,:]=Vyin[IBM_idxx[i,:],IBM_idxy[i,:]]
    end
    return Vx_euler2lag, Vy_euler2lag 
end

function sumEuler2Lag(IBM_fxTemp,IBM_fyTemp,IBM_fxLag,IBM_fyLag,IBM_sten,numLag)
    for i=1:numLag
        IBM_fxLag[i] = sum(IBM_fxTemp_h[Int((i-1)*IBM_sten+1):1:Int(i*IBM_sten)])
        IBM_fyLag[i] = sum(IBM_fyTemp_h[Int((i-1)*IBM_sten+1):1:Int(i*IBM_sten)])
    end
    return IBM_fxLag,IBM_fyLag
end

function lag2euler(IBM_lagIdx,IBM_lagDelta,IBM_fxd,IBM_fyd,IBM_vx_correction,IBM_vy_correction,iter)
    IBM_fx_correction=zeros(size(IBM_vy_correction,1),size(IBM_vx_correction,2))
    IBM_fy_correction=zeros(size(IBM_vy_correction,1),size(IBM_vx_correction,2))
    time0 = Base.time();
    for i=1:size(IBM_lagIdx,1)    
        for j=1:size(IBM_lagIdx,2)
            if length(IBM_lagIdx[i,j])>0
                tempx=0.0;tempy=0.0;
                aa=IBM_fxd[IBM_lagIdx[i,j]]; bb=IBM_fyd[IBM_lagIdx[i,j]]; cc=IBM_lagDelta[i,j]
                for ii=1:size(IBM_lagDelta[i,j],1)
                    for jj=1:size(IBM_lagDelta[i,j],2)
                        tempx = tempx+aa[ii,jj]*cc[ii,jj]
                        tempy = tempy+bb[ii,jj]*cc[ii,jj]
                    end
                end
                IBM_fx_correction[i,j]=tempx;IBM_fy_correction[i,j]=tempy;
            end
        end
    end
    partime = Base.time()-time0
    IBM_vx_correction[1,:]=(3*IBM_fx_correction[1,:]-IBM_fx_correction[2,:])./2
    IBM_vx_correction[end,:]=(3*IBM_fx_correction[end,:]-IBM_fx_correction[end-1,:])./2
    IBM_vx_correction[2:end-1,:]=(IBM_fx_correction[1:end-1,:]+IBM_fx_correction[2:end,:])./2
    IBM_vy_correction[:,1]=(3*IBM_fy_correction[:,1]-IBM_fy_correction[:,2])./2
    IBM_vy_correction[:,end]=(3*IBM_fy_correction[:,end]-IBM_fy_correction[:,end-1])./2
    IBM_vy_correction[:,2:end-1]=(IBM_fy_correction[:,1:end-1]+IBM_fy_correction[:,2:end])./2
    print("\nSerial time: ",partime)
    print("\nvx_cor[1,5:10], vy_cor[1,5:10]: ",IBM_vx_correction[1,5:10],", ",IBM_vy_correction[1,5:10])
    return IBM_vx_correction,IBM_vy_correction
end

#~~~~~~~~~~~~~~~~ level set ~~~~~~~~~~~~~~~~~~~~#
function ls_heavenside(phi,dy)
    LS_H = zeros(size(phi))
    LS_H[phi.>dy] .= 1.0
    LS_H[abs.(phi) .< dy] .= 0.5 * (1 .+ phi[abs.(phi) .< dy] ./ dy .+ sin.(PI .* phi[abs.(phi) .< dy]) ./ dy ./ PI)
    return LS_H
end

function LS_getNormal(LS_phi_vx,LS_phi_vy,smooth_over,dy)
    LS_H_vx = ls_heavenside(LS_phi_vx,smooth_over*dy)
    LS_H_vy = ls_heavenside(LS_phi_vy,smooth_over*dy)
    LS_nx, LS_ny = diff(LS_phi_vx,dims=1), diff(LS_phi_vy,dims=2)
    LS_n = sqrt.(LS_nx.^2 .+ LS_ny.^2)
    LS_nx, LS_ny = LS_nx./LS_n, LS_ny./LS_n
    return LS_nx, LS_ny
end

function LS_init(etan,kappa_air,kappa_ice,eta_air,rho_ice,rho_air,Y2,Y2_vx,Y2_vy,ly_air,ly_ice,dy,smooth_over)
    num_air_idx = Int64(round(ly_air/dy)) 
    # heaviside for 511x127
    LS_phi = ly_ice .- Y2
    LS_H = ls_heavenside(LS_phi,smooth_over*dy)
    # calculate normal of the interface, vx: 512x127, vy: 511x128, dont need so far
    LS_phi_vx, LS_phi_vy = ly_ice .- Y2_vx, ly_ice .- Y2_vy
    LS_nx, LS_ny = LS_getNormal(LS_phi_vx,LS_phi_vy,smooth_over,dy)

    Kappa = kappa_air .+ (kappa_ice - kappa_air)*LS_H;
    etan  = eta_air   .+ (etan .- eta_air).*LS_H;
    Rho   = rho_air   .+ (rho_ice - rho_air)*LS_H;

    Kappa_qTx = 0.5*(Kappa[1:end-1,:]+Kappa[2:end,:])
    Kappa_qTy = 0.5*(Kappa[:,1:end-1]+Kappa[:,2:end])
    return LS_phi, LS_H, LS_nx, LS_ny, Kappa, Kappa_qTx, Kappa_qTy , etan, Rho
end

function advect_phi(phi_n,LS_nx,LS_ny,Vxin,Vyin,dx,dy,dt)
    phi_np1 = phi_n; phi_np2 = phi_n;
    Dxp = Vxin.>0.0; Dxm = Vxin.<=0; Dxp = Dxp[2:end-1,2:end-1]; Dxm = Dxm[2:end-1,2:end-1];
    Dyp = Vyin.>0.0; Dym = Vyin.<=0; Dyp = Dyp[2:end-1,2:end-1]; Dym = Dym[2:end-1,2:end-1];

    dphidx = ((phi_n[2:end-1,2:end-1] .- phi_n[1:end-2,2:end-1]).*Dxp .+ (phi_n[3:end,2:end-1] .- phi_n[2:end-1,2:end-1]).*Dxm)/dx;
    dphidz = ((phi_n[2:end-1,2:end-1] .- phi_n[2:end-1,1:end-2]).*Dyp .+ (phi_n[2:end-1,3:end] .- phi_n[2:end-1,2:end-1]).*Dym)/dy;
    
    phi_np1[2:end-1,2:end-1] .= phi_n[2:end-1,2:end-1] .+ (Vxin[2:end-1,2:end-1].*(LS_nx[2:end-1,2:end-1]).*dphidx .+ Vyin[2:end-1,2:end-1].*(LS_ny[2:end-1,2:end-1]).*dphidz);
    phi_np1[2:end-1,end] .= phi_np1[2:end-1,end-1];

    dphidx1 = ((phi_np1[2:end-1,2:end-1] .- phi_np1[1:end-2,2:end-1]).*Dxp .+ (phi_np1[3:end,2:end-1] .- phi_np1[2:end-1,2:end-1]).*Dxm)/dx;
    dphidz1 = ((phi_np1[2:end-1,2:end-1] .- phi_np1[2:end-1,1:end-2]).*Dyp .+ (phi_np1[2:end-1,3:end] .- phi_np1[2:end-1,2:end-1]).*Dym)/dy;
    
    phi_np2[2:end-1,2:end-1] .= phi_np1[2:end-1,2:end-1] .+ (Vxin[2:end-1,2:end-1].*(LS_nx[2:end-1,2:end-1]).*dphidx1 .+ Vyin[2:end-1,2:end-1].*(LS_ny[2:end-1,2:end-1]).*dphidz1);
    phi_np2[2:end-1,end] .= phi_np2[2:end-1,end-1];
    phi_np1 = (phi_n + phi_np2)/2;
    return Array(phi_np1)
end

#~~~~~~~~~~~~~~~~ initialization ~~~~~~~~~~~~~~~~~~~~#
function getObjShape(X,rc)
    IBM_lagXTemp = X
    center,r=0,rc
    theta=range(0,stop=2*pi,length=200)
    IBM_lagXTemp=center.+r.*sin.(theta)
    IBM_lagYTemp=center.+r.*cos.(theta)
    return IBM_lagXTemp,IBM_lagYTemp 
end

function readBedInputs(bed_input_name)
    data_df = CSV.read(bed_input_name,DataFrame)
    IBM_lagXTemp = data_df[!,"x"]
    IBM_lagYTemp = data_df[!,"bed"]
    IBM_lagXTemp = reshape(IBM_lagXTemp, :, 1)
    IBM_lagYTemp = reshape(IBM_lagYTemp, :, 1)
    return IBM_lagXTemp,IBM_lagYTemp
end

function getLagPoints(IBM_lagXTemp,IBM_lagYTemp,dx,dy,lx)
    # compute total Lag points needed
    segX=IBM_lagXTemp[2:end]-IBM_lagXTemp[1:end-1]
    segY=IBM_lagYTemp[2:end]-IBM_lagYTemp[1:end-1]
    seg=sqrt.(segX.^2+segY.^2)
    segLength=sum(seg)
    numLag=floor.(segLength/sqrt(dx*dx))
    # define Lag points
    interp_linear_extrap = linear_interpolation(vec(IBM_lagXTemp),vec(IBM_lagYTemp),extrapolation_bc=Line())
    IBM_lagX = range(0,stop=lx,length=Int64(numLag))
    IBM_lagY = interp_linear_extrap(IBM_lagX)
    IBM_lagX = collect(IBM_lagX)
    IBM_lagY = collect(IBM_lagY)
    return IBM_lagX,IBM_lagY
end

function getBedNormalDir(IBM_lagX,IBM_lagY)
    IBM_lagNormalX,IBM_lagNormalY = zeros(size(IBM_lagX)),zeros(size(IBM_lagX))
    segX=IBM_lagX[2:end]-IBM_lagX[1:end-1]
    segY=IBM_lagY[2:end]-IBM_lagY[1:end-1]
    seg=sqrt.(segX.^2+segY.^2)
    IBM_lagNormalX[2:end] = abs.(segY)./seg
    IBM_lagNormalY[2:end] = abs.(segX)./seg
    IBM_lagNormalX[1], IBM_lagNormalY[1] = 0.0, 1.0
    # print("IBM_lagNormalX ",IBM_lagNormalX[1:30],"\n")
    # print("IBM_lagNormalY ",IBM_lagNormalY[1:30],"\n")
    return Data.Array(IBM_lagNormalX),Data.Array(IBM_lagNormalY)
end

function getDeltaIdx(IBM_deltaIdx,IBM_lagX,IBM_lagY,nx,ny,X,Y,dx,dy,PERIO_BC)
    
    x_idx_start,y_idx_start=1,1
    # define the matrix location around each lag point
    for lagCount=1:length(IBM_lagX)
        x_idx_start,y_idx_start=x_idx_start-5,y_idx_start-5
        x_idx_start,y_idx_start=Int64((x_idx_start.+abs.(x_idx_start))./2+1),Int64((y_idx_start.+abs.(y_idx_start))./2+1) # lower bound
        x_idx_end,y_idx_end=max(x_idx_start+10,nx-1),max(y_idx_start+10,ny-1)
        lagX,lagY=IBM_lagX[lagCount]-dx,IBM_lagY[lagCount]-dy
        # get the x lower bonud index of the current lag point
        for xcount=x_idx_start:x_idx_end
            if (X[xcount]<=lagX && X[xcount+1] >= lagX)
	            IBM_deltaIdx[lagCount,1]=Int64(xcount)
	        end
            if (PERIO_BC==1)
                if (lagX<=X[1]-dx)
                    IBM_deltaIdx[lagCount,1]=Int64(nx-1)
                elseif(lagX<=X[1])
                    IBM_deltaIdx[lagCount,1]=Int64(nx)
                end
                if (lagX>=X[nx]); IBM_deltaIdx[lagCount,1]=nx; end
            else
                if ((lagX<=X[1]) || (lagX<=X[1]-dx));IBM_deltaIdx[lagCount,1]=1; end
                if (lagX>=X[nx]-3*dx); IBM_deltaIdx[lagCount,1]=Int64(nx-3); end
            end
        end
        # get the z lower bound index of the current lag point
        for ycount=y_idx_start:y_idx_end
            if (Y[ycount]<=lagY && Y[ycount+1]>=lagY)
                IBM_deltaIdx[lagCount,2]=Int64(ycount)
            end
            if (lagY<=Y[1]); IBM_deltaIdx[lagCount,2]=Int64(1); end
            if (lagY>=Y[ny]); IBM_deltaIdx[lagCount,2]=Int64(ny); end
	    end
        # print("\nlag  ",lagCount,", IBM_deltaIdx[lagCount,1] ",IBM_deltaIdx[lagCount,1])
    end
    return IBM_deltaIdx
end

function getDeltaMatrix(IBM_idxx,IBM_idxy,IBM_delta,IBM_deltaIdx,IBM_lagX,IBM_lagY,IBM_sten,nx,ny,lx,ly,X,Y,dx,dy,numLag)
    
    IBM_deltaIdxx,IBM_deltaIdxy=zeros(Int64,numLag,IBM_sten),zeros(Int64,numLag,IBM_sten)
    IBM_deltaMat=zeros(IBM_sten,IBM_sten,numLag)
    # IBM_delta,IBM_idxx,IBM_idxy=@zeros(numLag*IBM_sten,IBM_sten),@zeros(Int64,numLag*IBM_sten,IBM_sten),zeros(Int64,numLag*IBM_sten,IBM_sten)
    for lagCount=1:numLag
	    phi_x,phi_y=zeros(1,IBM_sten),zeros(1,IBM_sten)
        x_corner,y_corner=Int64(IBM_deltaIdx[lagCount,1]),Int64(IBM_deltaIdx[lagCount,2]) # top left corner
	    xlag,ylag=IBM_lagX[lagCount],IBM_lagY[lagCount]
        x_corner_end=Int64(mod(x_corner+3,nx)); if(x_corner_end==0);x_corner_end=Int64(nx); end
        y_corner_end=Int64(mod(y_corner+3,ny)); if(y_corner_end==0);y_corner_end=Int64(ny); end
        if (x_corner<x_corner_end)
            IBM_deltaIdxx[lagCount,:]= x_corner:x_corner_end
        else
            IBM_deltaIdxx[lagCount,:]=[x_corner:nx;1:x_corner_end]
        end
        if (y_corner<y_corner_end)
    	    IBM_deltaIdxy[lagCount,:]= y_corner:y_corner_end
        else
    	    IBM_deltaIdxy[lagCount,:]=[y_corner:ny;1:y_corner_end]
        end
        rx,ry=(X[x_corner:x_corner_end].-xlag)./dx,(Y[y_corner:y_corner_end].-ylag)./dy
        if (lagCount>numLag/2)
            if(x_corner>x_corner_end);rx=([X[x_corner:nx];X[1:x_corner_end].+lx].-xlag)./dx;end
            if(y_corner>y_corner_end);ry=([Y[y_corner:ny];Y[1:y_corner_end].+ly].-ylag)./dy;end
        else
    	    if(x_corner>x_corner_end);rx=([X[x_corner-1:nx-1].-lx;X[1:x_corner_end]].-xlag)./dx;end
            if(y_corner>y_corner_end);ry=([Y[y_corner:ny].-lx;Y[1:y_corner_end]].-ylag)./dy;end
        end
        # get phi in each direction, 4 points stencil, see Peskin eq 6.27
        for stenCount=1:IBM_sten
            rx_cur,ry_cur=rx[stenCount],ry[stenCount]
            rx_cur_abs,ry_cur_abs=abs(rx_cur),abs(ry_cur)
            # equ 6.27 Peskin 2002
            # x
            if (rx_cur_abs>=2)
            phi_x[stenCount]=0
            elseif (rx_cur_abs>=1 && rx_cur_abs<=2)
            phi_x[stenCount]=(5-2*rx_cur_abs-sqrt(-7+12*rx_cur_abs-4*rx_cur^2))/8
            elseif (rx_cur_abs<=1)
            phi_x[stenCount]=(3-2*rx_cur_abs+sqrt(1+4*rx_cur_abs-4*rx_cur^2))/8
            end
            # y
            if (ry_cur_abs>=2)
            phi_y[stenCount]=0
            elseif (ry_cur_abs>=1 && ry_cur_abs<=2)
            phi_y[stenCount]=(5-2*ry_cur_abs-sqrt(-7+12*ry_cur_abs-4*ry_cur^2))/8
            elseif (ry_cur_abs<=1)
                phi_y[stenCount]=(3-2*ry_cur_abs+sqrt(1+4*ry_cur_abs-4*ry_cur^2))/8
            end
        end
	    # ensure conservation
        phi_yn=copy(phi_y)      
        if (abs(sum(phi_x'*phi_y)-1)>1e-4)
            phi_yn[1]=phi_y[1]/sum(phi_x'*phi_y)
            phi_yn[2]=phi_y[2]/sum(phi_x'*phi_y)
            phi_yn[3]=phi_y[3]/sum(phi_x'*phi_y)
        end
        # combine x and y, eq 4 and eq 5, Kempe 2012
        IBM_deltaMat[:,:,lagCount]=phi_x'*phi_yn
        # reshape the delta matrix for parallel
        IBM_delta[(lagCount-1)*IBM_sten+1:lagCount*IBM_sten,:]=phi_x'*phi_yn
        IBM_idxx[(lagCount-1)*IBM_sten+1:lagCount*IBM_sten,:]=[IBM_deltaIdxx[lagCount,:] IBM_deltaIdxx[lagCount,:] IBM_deltaIdxx[lagCount,:] IBM_deltaIdxx[lagCount,:]]
        IBM_idxy[(lagCount-1)*IBM_sten+1:lagCount*IBM_sten,:]=[IBM_deltaIdxy[lagCount,:] IBM_deltaIdxy[lagCount,:] IBM_deltaIdxy[lagCount,:] IBM_deltaIdxy[lagCount,:]]'
        # print("\nlag ",lagCount,", IBM_idxx ", IBM_idxx[(lagCount-1)*IBM_sten+1:lagCount*IBM_sten,:])
    end 
    # exit()
    return IBM_deltaIdxx,IBM_deltaIdxy,IBM_delta
end

function correct_IC_velo(Vx,Vx_tar,IBM_lagX,IBM_lagY,dy)
    # print(",,,length IBM_lagX ",length(IBM_lagX))
    for i=1:min(size(Vx,1),length(length(IBM_lagX)))
        elevation_start_idx = Int64(round(IBM_lagY[i]/dy))+1
        Vx[i,elevation_start_idx:end] = Vx[i,1:end-elevation_start_idx+1]
        Vx[i,1:elevation_start_idx-1] .= Vx_tar
    end
    return Vx
end

function getlagIdxforEulerloop(IBM_lagIdx,IBM_lagDelta,IBM_idxx,IBM_idxy,IBM_delta,idxy_max,nx)
    # print("IBM_lagY max:",maximum((IBM_idxy)),", size of IBM_lagIdx: ",size(IBM_delta,1),", ",size(IBM_delta,2)," \n")
    for i=1:nx, j=1:idxy_max
	IBM_lagIdx[i,j],IBM_lagDelta[i,j]=[],[]
    end
    for i=1:size(IBM_idxx,1)
	for j=1:size(IBM_idxx,2)
	    for k=1:size(IBM_idxy,2)
            push!(IBM_lagIdx[IBM_idxx[i,j],IBM_idxy[i,k]],i)
            # print("i: ",i,", j: ",j,", k: ",k, ", IBM_idxx[i,j]: ",IBM_idxx[i,j],", IBM_idxy[i,k]: ",IBM_idxy[i,k],", (i-1)*size(IBM_idxy,2)+j: ",(i-1)*size(IBM_idxy,2)+j,"\n")
            push!(IBM_lagDelta[IBM_idxx[i,j],IBM_idxy[i,k]],IBM_delta[(i-1)*size(IBM_idxy,2)+j,k])
	    end
	end
    end
    return IBM_lagIdx,IBM_lagDelta
end

function compute_inflow_velo(Y,μsU,μsL,ly,ρg,alpha)
    # constant etan
    # Vx_in_ice = ρg*sind(alpha)./μsi.*((Y)*(ly)-(Y).^2*0.5);
    # etan_in_ice = μsi*ones(1,size(Y,1))
    # linear etan
    k_etan = (μsU-μsL)/ly
    Vx_in_ice = ρg*sind(alpha)/k_etan^2*(μsU*log.(μsL.+k_etan*Y).-k_etan*Y).-ρg*sind(alpha)/k_etan^2*μsU*log(μsL)
    etan_in_ice = μsL .+ (μsU-μsL)*Y/ly
    # heaviside etan
    # a = 3.9e14; c = 520; k = 4.2e-3; A = ρg*sind(alpha)
    # etan_in_ice = a./((1).+(exp.(-2*k*(Y.-c))))
    # Vx_in_ice = 0.3*A/a*((ly*Y.-Y.^2*0.5).-(0.5*ly/k.*exp.(-2*k*(Y.-c))).+((0.25/k^2).+(0.25*Y/k)).*exp.(k*(2*c.-2*Y)).+
    # (0.5*ly/k*exp(2*k*c)-0.25/k^2*exp(2*k*c)));
    return Vx_in_ice, etan_in_ice
end

function copyFromHost2Device(Vxin,Vyin,Vx,Vy,Pt,Tt,Hs,τxy,Vx_in_ice,Mus,X2,Y2,Vxin_h,Vyin_h,Vx_h,Vy_h,Pt_h,T_h,Hs_h,τxy_h,Vx_in_ice_h,Mus_h,X2_h,Y2_h)
    copyto!(Vxin_h, Vxin)
    copyto!(Vyin_h, Vyin)
    copyto!(Vx_h,   Vx)
    copyto!(Vy_h,   Vy)
    copyto!(Pt_h, Pt)
    copyto!(T_h, Tt)
    copyto!(Hs_h, Hs)
    copyto!(τxy_h, τxy)
    copyto!(Vx_in_ice_h, Vx_in_ice)
    copyto!(Mus_h, Mus)
    copyto!(X2_h, X2)
    copyto!(Y2_h, Y2)
    return Vxin_h,Vyin_h,Vx,Vy,Pt_h,T_h, Hs_h,τxy_h,Vx_in_ice_h,Mus_h,X2_h,Y2_h
end

function saveResults2CSV(Vxin_h,Vyin_h,Vx_h,Vy_h,Pt_h,T_h,Hs_h,τxy_h,Vx_in_ice_h,Mus_h,IBM_lagX,IBM_lagY,X2_h,Y2_h,dx,dy,time_step,filepath,filename)
    df_length=maximum(length.([vec(Vxin_h),vec(Vx_h),vec(Vy_h),vec(IBM_lagX)]))
    results_df = DataFrame(Vxin=[vec(Vxin_h);fill(NaN,df_length-length(vec(Vxin_h)))])
    # print(results_df)
    results_df = DataFrame(Vxin=[vec(Vxin_h);fill(NaN,df_length-length(vec(Vxin_h)))],
                           Vyin=[vec(Vyin_h);fill(NaN,df_length-length(vec(Vyin_h)))],
                           Vx=[vec(Vx_h);fill(NaN,df_length-length(vec(Vx_h)))],
                           Vy=[vec(Vy_h);fill(NaN,df_length-length(vec(Vy_h)))],
                           Pt=[vec(Pt_h);fill(NaN,df_length-length(vec(Pt_h)))],
                           T=[vec(T_h);fill(NaN,df_length-length(vec(T_h)))],
                           Hs=[vec(Hs_h);fill(NaN,df_length-length(vec(Hs_h)))],
                           txy=[vec(τxy_h);fill(NaN,df_length-length(vec(τxy_h)))],
                           Vx_in_ice=[vec(Vx_in_ice_h);fill(NaN,df_length-length(vec(Vx_in_ice_h)))],
                           etan=[vec(Mus_h);fill(NaN,df_length-length(vec(Mus_h)))],
                           X2=[vec(X2_h);fill(NaN,df_length-length(vec(X2_h)))],
                           Y2=[vec(Y2_h);fill(NaN,df_length-length(vec(Y2_h)))],
                           IBM_lagX=[vec(IBM_lagX);fill(NaN,df_length-length(vec(IBM_lagX)))],
                           IBM_lagY=[vec(IBM_lagY);fill(NaN,df_length-length(vec(IBM_lagY)))])
    CSV.write(filepath*filename*string(time_step)*".csv",results_df)
end

##################################################


##################################################
@views function Stokes2D()
    # performance
    global totaltime0 = Base.time()
    num_gpus = Int(length(devices()))
    global S2Y = 3600.0*24*365.25
    global S2D = 3600.0*24
    global PI = 3.141592653579385626
    # Settings
    PERIO_BC = 0    # periodic BC
    INCLU_IBM = 0
    VIS_TYPE = "GLEN"  # could be: LIN_VIS, GLEN, GOLD
    ADVECT = 1
    # Physics
    lx, ly_ice,ly_air    = 4800.0, 800.0, 50.0  # domain extends
    ly = ly_ice+ly_air
    μsL,μsU,μsi,μs0= 5.8e12,5.8e14,2e14,1e30  # lower and upper bound of initial viscosity
    ρ	      = 900.0	  # ice density
    g         = 9.8
    ρg        = ρ*g       
    alpha     = 1.2	      # tilt angle
    final_time = 100*S2Y  # final physical time in seconds
    dt        = 60*S2D    # time step
    T_base    = -0.1        # define initial basal temperature in degrees
    T_melt    = -0.1
    T_surf    = -26.0       # define initial surface temperature in degrees
    T_K       = 273.0
    q_geo     = 0.05       # geothermal heat flux, k/m*W/(m*k) = w/m^2, original # 0.05
    kappa_ice   = 2.51       # thermal conductivity at -10 degrees, W/mK
    kappa_air   = 0.1 #22e-3      # thermal conductivity at -26 degrees, W/mK
    rho_air   = 40 #1.293      # kg/m^3
    eta_air   = 5e11        # Pa S
    cp        = 2096.9     # specific heat at -10 degrees, J/KgK
    a0        = 13e-13     # pre factor, original 8.75e-13, needs to be tuned
    Q0        = 6e4        # activation energy
    R         = 8.314      # gas constant
    d         = 5.0e3      # grain size
    # Numerics
    iterMax   = 1e5         # maximum number of pseudo-transient iterations
    nout      = 1000        # error checking frequency
    Vdmp      = 2.0         # damping paramter for the momentum equations
    Vsc       = 1.0         # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 1.0         # relaxation paramter for the pressure equation pseudo-timestep limiter
    ε         = 1e-6        # nonlinear absolute tolerence
    ε_rel     = 1e-10
    nx, ny    = 511,127    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    CN        = 0.5
    rele      = 2e-2       # relaxition
    CN        = 0.5
    npow      = 3.0         # exponent
    eta_b     = 0.5
    Vx_max    = 1e4/S2Y
    Vy_max    = 1e3/S2Y
    # Derived numerics
    dx, dy    = lx/(nx-1), ly/(ny-1) # cell sizes
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(nx,ny)
    dampX     = 1.0-Vdmp/nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/ny # damping term for the y-momentum equation
    num_time_step = Int64(round(final_time/dt))
    T_base    = T_base+T_K
    T_surf    = T_surf+T_K
    T_melt    = T_melt+T_K-T_surf
    mpow      = -(1-1/npow)/2
    # Array allocations
    Pt        = @zeros(nx  ,ny  )
    dτPt      = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Exx       = @zeros(nx  ,ny  )
    Eyy       = @zeros(nx  ,ny  )
    Exy       = @zeros(nx-1,ny-1)
    Exyn      = @zeros(nx  ,ny  )
    EII2      = @zeros(nx  ,ny  )
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    dVxdτ     = @zeros(nx-1,ny-2)
    dVydτ     = @zeros(nx-2,ny-1)
    dτVx      = @zeros(nx-1,ny-2)
    dτVy      = @zeros(nx-2,ny-1)
    Tt        = @zeros(nx,ny)
    H         = @zeros(nx,ny)
    dτT       = @zeros(nx,ny)
    qxT       = @zeros(nx+1,ny  )
    qyT       = @zeros(nx  ,ny+1)
    dTdt      = @zeros(nx  ,ny  )
    T_res     = @zeros(nx  ,ny  )
    Hs        = @zeros(nx,ny)
    Mus_phy   = @zeros(nx  ,ny  )
    Musit     = @zeros(nx  ,ny  )
    mech      = @zeros(nx  ,ny  )
    dTdx      = @zeros(nx  ,ny  )
    dTdy      = @zeros(nx  ,ny  )
    Vxt1      = zeros(1,ny)
    Vxt2      = zeros(1,ny)
    Vyt1      = zeros(1,ny+1)
    Vyt2      = zeros(1,ny+1)
    # file inputs and outputs   
    bed_input_name= "./../bed_inputs/sine_amp50_num10.csv"
    filepath= "./../results/"
    filename= "sin_LS_"*VIS_TYPE*"_alpha_"*string(alpha)*"_reso_"*string(nx)*"x"*string(ny)*"_"
    
    ########### IBM  ##########
    # ~~ parameters
    IBM_sten    = Int(4)
    IBM_us      = 0
    IBM_s       = 0
    ud,vd	    = 0.0,0.0
    phi2	    = 1.0 
    ###########################
    # Initial conditions
    Radc      =  @zeros(nx  ,ny  )
    Rogx      =  ρg*sind(alpha)*ones(nx-1,ny-2)
    Rogy      =  ρg*cosd(alpha)*ones(nx-2,ny-1)
    Mu        =  range(μsL,μsU,ny)
    Mus       =  ones(nx,1)*Mu'
    Mus       = Data.Array(Mus)
    Mus_s     = @zeros(nx-1,ny-1)
    Rogx      = Data.Array(Rogx)
    Rogy      = Data.Array(Rogy)

    X, Y    = 0:dx:lx, 0:dy:ly # 511, 127
    Xv, Yv  = -dx/2:dx:lx+dx/2, -dy/2:dy:(ly+dy/2) # 512, 128
    X2, Y2  = Array(X)*ones(1,size(Y,1)), ones(size(X,1),1)*Array(Y)' # both 511x127
    X2_vx, Y2_vx= Array(Xv)*ones(1,size(Y,1)), ones(size(Xv,1),1)*Array(Y)' # 512x127
    X2_vy, Y2_vy= Array(X)*ones(1,size(Yv,1)), ones(size(X,1),1)*Array(Yv)' # 511x128
    
    ########### IBM setup ##########
    # ~~ immersed shape
    IBM_lagXTemp,IBM_lagYTemp=readBedInputs(bed_input_name);
    IBM_lagX,IBM_lagY=getLagPoints(IBM_lagXTemp,IBM_lagYTemp,dx,dy,lx)
    IBM_lagNormalX,IBM_lagNormalY = getBedNormalDir(IBM_lagX,IBM_lagY)
    # q_geoX,q_geoY = IBM_lagNormalX*q_geo,IBM_lagNormalY*q_geo
    IBM_ud,IBM_wd=IBM_us,IBM_us 
    # ~~ compute delta functions
    IBM_deltaIdx=@zeros(length(IBM_lagX),2)
    numLag=length(IBM_lagX)
    IBM_delta,IBM_idxx,IBM_idxy=@zeros(numLag*IBM_sten,IBM_sten),zeros(Int64,numLag*IBM_sten,IBM_sten),zeros(Int64,numLag*IBM_sten,IBM_sten)
    CUDA.@allowscalar IBM_deltaIdx=getDeltaIdx(IBM_deltaIdx,IBM_lagX,IBM_lagY,nx,ny,X,Y,dx,dy,PERIO_BC)
    CUDA.@allowscalar IBM_idxx,IBM_idxy,IBM_delta=getDeltaMatrix(IBM_idxx,IBM_idxy,IBM_delta,IBM_deltaIdx,IBM_lagX,IBM_lagY,IBM_sten,nx,ny,lx,ly,X,Y,dx,dy,numLag)
    idxy_max=maximum(IBM_idxy)
    IBM_lagIdx=Array{Array{Int64,1}}(undef,nx,idxy_max)
    IBM_lagDelta=Array{Array{Float64,1}}(undef,nx,idxy_max)
    CUDA.@allowscalar IBM_lagIdx,IBM_lagDelta=getlagIdxforEulerloop(IBM_lagIdx,IBM_lagDelta,IBM_idxx,IBM_idxy,IBM_delta,idxy_max,nx)
    IBM_sum_idx=collect(1:IBM_sten:size(IBM_delta,1))
    IBM_sum_idx=Data.Array(IBM_sum_idx)
    # ~~ allocation
    IBM_fx,IBM_fy=@zeros(nx+1,ny), @zeros(nx,ny+1)
    IBM_qxT,IBM_qyT=@zeros(nx+1,ny), @zeros(nx,ny+1)
    Vx_euler2lag=@zeros(IBM_sten*numLag,IBM_sten)
    Vy_euler2lag=@zeros(IBM_sten*numLag,IBM_sten)
    dTdx_euler2lag=@zeros(IBM_sten*numLag,IBM_sten)
    dTdy_euler2lag=@zeros(IBM_sten*numLag,IBM_sten)
    IBM_fx=@zeros(IBM_sten*numLag,IBM_sten)
    IBM_fy=@zeros(IBM_sten*numLag,IBM_sten)
    IBM_dTdx=@zeros(IBM_sten*numLag,IBM_sten)
    IBM_dTdy=@zeros(IBM_sten*numLag,IBM_sten)
    IBM_fxTemp=@zeros(IBM_sten*numLag,1)
    IBM_fyTemp=@zeros(IBM_sten*numLag,1)
    IBM_dTdxTemp=@zeros(IBM_sten*numLag,1)
    IBM_dTdyTemp=@zeros(IBM_sten*numLag,1)
    IBM_fxLag=@zeros(numLag,1)
    IBM_fyLag=@zeros(numLag,1)
    IBM_dTdxLag=@zeros(numLag,1)
    IBM_dTdyLag=@zeros(numLag,1)
    IBM_fxd=@zeros(numLag,1)
    IBM_fyd=@zeros(numLag,1)
    IBM_Ttd=@zeros(numLag,1)
    IBM_vx_correction,IBM_vy_correction,IBM_T_correction=@zeros(nx+1,ny),@zeros(nx,ny+1),@zeros(nx,ny)
    Vxin,Vyin=@zeros(nx,ny),@zeros(nx,ny)
    Rxin,Ryin=@zeros(nx,ny),@zeros(nx,ny)
    norm_vx,norm_vy,norm_p=@zeros(nx,ny),@zeros(nx,ny),@zeros(nx,ny)
    Vxin_h,Vyin_h,Pt_h,T_h = zeros(size(Vxin)),zeros(size(Vyin)),zeros(size(Pt)),zeros(size(Tt))
    Hs_h, τxy_h = zeros(size(Hs)), zeros(size(τxy))
    Vx_h, Vy_h = zeros(size(Vx)),zeros(size(Vy))
    X2_h,Y2_h = zeros(size(X2)), zeros(size(Y2))
    IBM_lagX_h, IBM_lagY_h = zeros(size(IBM_lagX)), zeros(size(IBM_lagY))
    Mus_h = zeros(size(Mus))
    IBM_T_correction_h=zeros(size(IBM_T_correction))
    ###########################


    ########### initial inflow condition: velo, pressure, temp, rheo  ##########
    # inflow pressure set up
    Pt_in = ρg*cosd(alpha)*(ly*ones(size(Y))-Y);
    Pt = ones(size(Pt,1),1)*Pt_in'
    Pt = Data.Array(Pt)
    # inflow temperature and geothermal heat flux set up
    T_init_lin = range(T_base-T_surf,0.0,ny);T_init_lin=Array(T_init_lin)   # from T_base to T_surf in Kelvin or in degrees
    Tt = ones(size(Tt,1),1)*T_init_lin'
    elevation_start_idx = Int64(round(IBM_lagY[1]/dy))+1
    T_o = copy(Tt)

    # inflow velo set up
    Vx_in_ice, etan_in_ice = compute_inflow_velo(Y,μsU,μsL,ly,ρg,alpha)
    Vx_in_ice[elevation_start_idx:end] = Vx_in_ice[1:end-elevation_start_idx+1]
    Vx_in_ice[1:elevation_start_idx-1] .= 0.0
    Vx = ones(size(Vx,1),1)*Vx_in_ice'
    Mus = ones(size(Mus,1),1)*etan_in_ice'
    # CUDA.@allowscalar Vx= correct_IC_velo(Vx,0.0,IBM_lagX,IBM_lagY,dy); 
    
    ########### LS setup ##########
    smooth_over = 5
    # level set setup
    LS_phi, LS_H, LS_nx, LS_ny, Kappa, Kappa_qTx, Kappa_qTy, Mus, Rho = LS_init(Mus,kappa_air,kappa_ice,eta_air,
                                                   ρ,rho_air,Y2,Y2_vx,Y2_vy,ly_air,
                                                   ly_ice,dy,smooth_over);
    
    Rogx = 0.5*(Rho[1:end-1,2:end-1]+Rho[2:end,2:end-1])*g*sind(alpha)
    Rogy = 0.5*(Rho[2:end-1,1:end-1]+Rho[2:end-1,2:end])*g*cosd(alpha)
    Rho = Data.Array(Rho)
    Rogx = Data.Array(Rogx)
    Rogy = Data.Array(Rogy)
    Kappa_qTx = Data.Array(Kappa_qTx)
    Kappa_qTy = Data.Array(Kappa_qTy)
    Kappa = Data.Array(Kappa)
    LS_phi = Data.Array(LS_phi)
    LS_H = Data.Array(LS_H)
    LS_nx = Data.Array(LS_nx)
    LS_ny = Data.Array(LS_ny)
    Vx = Data.Array(Vx)
    Mus = Data.Array(Mus)
    Vx_in_ice = Data.Array(Vx_in_ice)
    T_init_lin = Data.Array(T_init_lin)
    Vy_in_ice = zeros(size(Vx_in_ice,1)+1)
    Vy_in_ice = Data.Array(Vy_in_ice)
    Vx_in_ice_h = zeros(size(Vx_in_ice))

    Tt = Data.Array(Tt)
    T_o = Data.Array(T_o)
    dTdt = @zeros(nx,ny)
    dTdt_o = @zeros(nx,ny)
    nouse = @zeros(nx,ny)
    CUDA.@allowscalar Tt[Tt.>T_melt].=T_melt
     
    ###########################
    ########### print put  ##########
    print("\n----------------------------------------------------\n")
    print("                 SINE example                  \n")
    print("simulation parameters:\n")
    print("    Lx: ", lx, ", Ly: ",ly,"\n    Nx: ",nx,", Ny: ",ny,"\n    dx: ",dx,", dy: ",dy,", aspect ratio: ",dx/dy,"\n")
    print("    num of Lag points: ", numLag,"\n")
    print("    Simulation time: ", final_time/S2Y," years, dt: ",dt/S2D," days, total num of time steps: ",num_time_step," \n")
    print("    tilting angle alpha: ", alpha,"\n")
    CUDA.@allowscalar print("    starting surface temp: ", T_surf," K, bed temp: ",T_base," K, dT: ", (Tt[5,2]-Tt[5,1])," K\n")
    print("    rheology type: ", VIS_TYPE,", n = ",npow,"\n")
    print("    starting viscosity at bed: ", μsL,", at surf: ",μsU,"\n")
    print("    # of GPU using: ", num_gpus,"\n")
    print("    # of threads using: ", Threads.nthreads(),"\n")
    print("    bed data input: ", bed_input_name,"\n")
    print("    results will be saved as: ", filepath*filename,"\n")
    print("----------------------------------------------------\n")
    #################################
    Tt = Data.Array(Tt)
    # Pysical time loop
    @parallel compute_qT!(Tt,qxT,qyT,Kappa_qTx,Kappa_qTy,dx,dy)
    @parallel (1:size(qyT,1), 1:size(qyT,2)) bc_y!(qyT)
    @parallel (1:size(qxT,1), 1:size(qxT,2)) bc_x!(qxT)
    @parallel compute_weaken_factor!(H,Tt,T_melt)
    CUDA.@allowscalar H[H.>1.0] .= 1.0
    @parallel compute_dT!(Tt,T_o,qxT,qyT,dTdt,dTdt_o,T_res,Hs,H,Rho,cp,ρ,CN,dt,dx,dy)
    global time_step = 0;
    for time_step = 1:num_time_step
        T_o,dTdt_o = copy(Tt),copy(dTdt)
        @parallel compute_timesteps!(dτVx,dτVy,dτPt,dτT,Rho,Mus,Kappa,Vsc,Ptsc,min_dxy2,max_nxy,cp,dt,eta_b)
        err=2*ε; err_old=20*ε; err_rel = abs(err-err_old)
        iter=1; global niter=0; err_evo1=[]; err_evo2=[]
        global itertime0 = Base.time();

        # Pseudo time loop
        while (err > ε || err_rel>ε_rel) && iter <= iterMax
            if (iter==1)  global runtime0 = Base.time(); end
            norm2_vx, norm2_vy, norm2_p = 0.0,0.0,0.0
            @parallel compute_P!(∇V, Pt, Vx, Vy, dτPt, Rho, dx, dy)
            @parallel compute_E_τ!(∇V,Exx,Eyy,Exy,Exyn,τxx,τyy,τxy,Vx,Vy,Mus,Mus_s,dx,dy)
            @parallel (1:size(Exyn,1), 1:size(Exyn,2)) Exyn_2sides!(Exyn)
            @parallel (1:size(Exyn,1), 1:size(Exyn,2)) bc_y!(Exyn)
            @parallel compute_Hs!(Exx,Eyy,Exyn,EII2,Hs,Mus)
            @parallel compute_glen!(EII2,Tt,Mus,Mus_phy,Musit,H,LS_H,a0,npow,mpow,Q0,R,T_surf,μs0,rele,eta_air)
            @parallel compute_dV!(Rx,Ry,dVxdτ,dVydτ,Pt,Rogx,Rogy,τxx,τyy,τxy,dampX,dampY,dx,dy)
            @parallel compute_qT!(Tt,qxT,qyT,Kappa_qTx,Kappa_qTy,dx,dy) 
            @parallel (1:size(qyT,1), 1:size(qyT,2)) bc_y!(qyT)
            @parallel (1:size(qxT,1), 1:size(qxT,2)) bc_x!(qxT)
            @parallel compute_dT!(Tt,T_o,qxT,qyT,dTdt,dTdt_o,T_res,Hs,H,Rho,cp,ρ,CN,dt,dx,dy)
            @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
            @parallel compute_T!(Tt,T_res,dτT)
            @parallel take_av(Vx,Vy,Vxin,Vyin)
            # ~~~~~~~~~~~~~~~ IBM ~~~~~~~~~~~~~~~~~~~ #
            if (INCLU_IBM==1)
                CUDA.@allowscalar Vx_euler2lag,Vy_euler2lag=par_getU_euler2lag(IBM_idxx,IBM_idxy,Vxin,Vyin,Vx_euler2lag,Vy_euler2lag,IBM_sten)
                @parallel np_compute_weigted_u_euler2lag!(Vx_euler2lag,Vy_euler2lag,IBM_delta,IBM_fx,IBM_fy)
                @parallel sum_IBM!(IBM_fx,IBM_fy,IBM_fxTemp,IBM_fyTemp)
                CUDA.@allowscalar IBM_fxLag,IBM_fyLag=par_sumEuler2Lag(IBM_fxTemp,IBM_fyTemp,IBM_fxLag,IBM_fyLag,IBM_sten,numLag)	
                @parallel compute_desired_V!(IBM_fxd,IBM_fyd,IBM_fxLag,IBM_fyLag,ud,vd)
                CUDA.@allowscalar IBM_vx_correction,IBM_vy_correction=par_lag2euler(IBM_idxx,IBM_idxy,IBM_delta,IBM_fxd,IBM_fyd,IBM_vx_correction,IBM_vy_correction,IBM_sten)
                @parallel IBM_velocity_correction!(IBM_vx_correction,IBM_vy_correction,Vx,Vy)
                @parallel take_av(Vx,Vy,Vxin,Vyin)
            end
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            # ~~~~~~~~~~~~~~~ BC ~~~~~~~~~~~~~~~~~~~ #
            # inlet
            @parallel (1:size(Vx,1), 1:size(Vx,2)) BC_vx_inlet!(Vx,Vx_in_ice)
            @parallel (1:size(Vy,1), 1:size(Vy,2)) BC_vy_inlet!(Vy,Vy_in_ice)
            @parallel (1:size(Vx,1), 1:size(Vx,2)) BC_vy_inlet!(Tt,T_init_lin)
            # # outlet
            @parallel (1:size(Vx,1), 1:size(Vx,2)) BC_vx_outlet!(Vx)
            @parallel (1:size(Vy,1), 1:size(Vy,2)) BC_vy_outlet!(Vy)
            @parallel (1:size(Tt,1), 1:size(Tt,2)) BC_Tt_outlet!(Tt)
            # if(iter==1)
            #     @parallel (1:size(Vx,1), 1:size(Vx,2)) BC_vx_outlet!(Vx)
            #     @parallel (1:size(Vy,1), 1:size(Vy,2)) BC_vy_outlet!(Vy)
            #     CUDA.@allowscalar Vxt2 = Vx[end-2,:]
            #     CUDA.@allowscalar Vyt2 = Vy[end-2,:]
            # elseif(iter==2)
            #     @parallel (1:size(Vx,1), 1:size(Vx,2)) BC_vx_outlet!(Vx)
            #     @parallel (1:size(Vy,1), 1:size(Vy,2)) BC_vy_outlet!(Vy)
            #     CUDA.@allowscalar Vxt1 = Vx[end-1,:]
            #     CUDA.@allowscalar Vyt1 = Vy[end-1,:]
            # elseif(iter>2)
            #     CUDA.@allowscalar Vx[end,:] .= 2*Vxt1 .- Vxt2
            #     CUDA.@allowscalar Vy[end,:] .= 2*Vyt1 .- Vyt2
            #     CUDA.@allowscalar Vxt2 = Vxt1
            #     CUDA.@allowscalar Vxt1 = Vx[end-1,:]
            #     CUDA.@allowscalar Vyt2 = Vyt1
            #     CUDA.@allowscalar Vyt1 = Vy[end-1,:]
            # end
            # @parallel (1:size(Vx,1), 1:size(Vx,2)) BC_periodic!(Vx)
            # @parallel (1:size(Vy,1), 1:size(Vy,2)) BC_periodic!(Vy)
            # @parallel (1:size(Tt,1), 1:size(Tt,2)) BC_periodic!(Tt)
            # bottom
            @parallel (1:size(Vx,1), 1:size(Vx,2)) BC_vx_bot!(Vx)
            @parallel (1:size(Vy,1), 1:size(Vy,2)) BC_vy_bot!(Vy)
            CUDA.@allowscalar Tt[:,1] .= T_base-T_surf
            # top 
            CUDA.@allowscalar Vy[2:end-1,end] = Vy[2:end-1,end-1]+0.5*dy./Mus[2:end-1,end].* (Pt[2:end-1,end] .+ 1.0/3.0*(-Pt[2:end-1,end-1] .+ 2.0*Mus[2:end-1,end-1].*(Vy[2:end-1,end-1] .- Vy[2:end-1,end-2])/dy));
            @parallel compute_E_τ!(∇V,Exx,Eyy,Exy,Exyn,τxx,τyy,τxy,Vx,Vy,Mus,Mus_s,dx,dy)
            CUDA.@allowscalar Vx[2:end-1,end] = Vx[2:end-1,end-1]-dy/dx*(Vy[2:end,end-1]-Vy[1:end-1,end-1])+dy/3*τxy[:,end-1]./Mus_s[:,end]
            CUDA.@allowscalar Tt[:,end] .= 0.0
            @parallel compute_E_τ!(∇V,Exx,Eyy,Exy,Exyn,τxx,τyy,τxy,Vx,Vy,Mus,Mus_s,dx,dy)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            if mod(iter,nout)==0
                global mean_Rx, mean_Ry, mean_∇V
                mean_Rx = mean(abs.(Rx[17:end,:])); mean_Ry = mean(abs.(Ry[17:end,:])); mean_∇V = mean(abs.(∇V[17:end,:]))
                err = maximum([mean_Rx, mean_Ry, mean_∇V])
                err_rel = abs(err-err_old);
                # push!(err_evo1, err); push!(err_evo2,iter)
                itertime = Base.time()-itertime0
                @printf("Time step=%d,iter steps=%d, err=%1.3e, err_rel=%1.3e, [norm_Vx=%1.3e, norm_Vy=%1.3e, norm_P=%1.3e, time used=%1.3e min, remain=%1.3e min] \n",
                 time_step,iter, err, err_rel, maximum(Vx)*S2Y,maximum(Vy)*S2Y, mean_∇V,itertime/60,itertime/60*(iterMax-iter)/nout)
                err_old=err;
                itertime0 = Base.time()
            end
            iter+=1; niter+=1
        end
        # compute temperature correction
        @parallel compute_dTdx_dTdy!(Tt,dTdx,dTdy,dx,dy)
        @parallel (1:size(dTdx,1), 1:size(dTdx,2)) bc_x!(dTdx)
        @parallel (1:size(dTdy,1), 1:size(dTdy,2)) bc_y!(dTdy)
        CUDA.@allowscalar dTdx_euler2lag,dTdy_euler2lag=par_getT_euler2lag(IBM_idxx,IBM_idxy,dTdx,dTdy,IBM_sten)
        @parallel np_compute_weigted_u_euler2lag!(dTdx_euler2lag,dTdy_euler2lag,IBM_delta,IBM_dTdx,IBM_dTdy)
        @parallel sum_IBM!(IBM_dTdx,IBM_dTdy,IBM_dTdxTemp,IBM_dTdyTemp)
        CUDA.@allowscalar IBM_dTdxLag,IBM_dTdyLag=par_sumEuler2Lag(IBM_dTdxTemp,IBM_dTdyTemp,IBM_dTdxLag,IBM_dTdyLag,IBM_sten,numLag)	
        @parallel par_desired_T!(IBM_Ttd,IBM_dTdxLag,IBM_dTdyLag,IBM_lagNormalX,IBM_lagNormalY,q_geo)
        CUDA.@allowscalar IBM_T_correction=par_lag2eulerT(IBM_idxx,IBM_idxy,IBM_delta,IBM_Ttd/ρ/cp,IBM_T_correction,IBM_sten)
        @parallel IBM_temperature_correction!(IBM_T_correction,Tt,dτT)
        CUDA.@allowscalar Tt[Tt.>T_melt].=T_melt
        if(ADVECT==1)
            @parallel advect_temp!(Tt,Vxin,Vyin,dTdx,dTdy,dt)
            @parallel (1:size(Vx,1), 1:size(Vx,2)) BC_vy_inlet!(Tt,T_init_lin)
            @parallel (1:size(Tt,1), 1:size(Tt,2)) BC_Tt_outlet!(Tt)
            CUDA.@allowscalar Tt[:,1] .= T_base-T_surf
            CUDA.@allowscalar Tt[:,end] .= 0.0
            LS_phi = advect_phi(LS_phi,LS_nx,LS_ny,Vxin,Vyin,dx,dy,dt) #
            LS_H = ls_heavenside(LS_phi,smooth_over*dy)
            LS_phi = Data.Array(LS_phi)
            LS_H = Data.Array(LS_H)
            Kappa = kappa_air .+ (kappa_ice - kappa_air)*LS_H;
            Rho   = rho_air   .+ (ρ - rho_air)*LS_H;
            # exit()
        end
        

        # save snapshot data
        if mod(time_step,100)==1
        CUDA.@allowscalar Vxin_h,Vyin_h,Vx_h,Vy_h,Pt_h,T_h,Hs_h,τxy_h,Vx_in_ice_h,Mus_h,X2_h,Y2_h=copyFromHost2Device(Vxin,Vyin,Vx,Vy,Pt,Tt,Hs,τxy,Vx_in_ice,Mus,X2,Y2,Vxin_h,Vyin_h,Vx_h,Vy_h,Pt_h,T_h,Hs_h,τxy_h,Vx_in_ice_h,Mus_h,X2_h,Y2_h)
        CUDA.@allowscalar saveResults2CSV(Vxin_h,Vyin_h,Vx_h,Vy_h,Pt_h,T_h,Hs_h,τxy_h,Vx_in_ice_h,Mus_h,IBM_lagX,IBM_lagY,X2_h,Y2_h,dx,dy,time_step,filepath,filename)
        print("\n Finished step ",time_step,". Results save to ",filepath*filename,"\n")
        end
    end



    # Performance
    runtime    = Base.time() - runtime0
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    runtime_it = runtime/(niter-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/runtime_it                       # Effective memory throughput [GB/s]
    totaltime = Base.time() - totaltime0  # in unit of sec
    @printf("\nTime step = %d,iter steps = %d, err = %2.3e, run time = %1.3e min (@ T_eff = %1.2f GB/s), total time = %1.3e min \n",time_step, niter, err, runtime/60, round(T_eff, sigdigits=2),totaltime/60)

    # save data
    CUDA.@allowscalar Vxin_h,Vyin_h,Vx_h,Vy_h,Pt_h,T_h,Hs_h,τxy_h,Vx_in_ice_h,Mus_h,X2_h,Y2_h=copyFromHost2Device(Vxin,Vyin,Vx,Vy,Pt,Tt,Hs,τxy,Vx_in_ice,Mus,X2,Y2,Vxin_h,Vyin_h,Vx_h,Vy_h,Pt_h,T_h,Hs_h,τxy_h,Vx_in_ice_h,Mus_h,X2_h,Y2_h)
    CUDA.@allowscalar saveResults2CSV(Vxin_h,Vyin_h,Vx_h,Vy_h,Pt_h,T_h,Hs_h,τxy_h,Vx_in_ice_h,Mus_h,IBM_lagX,IBM_lagY,X2_h,Y2_h,dx,dy,time_step,filepath,filename)
    print("\n Finished simulation. Results save to ",filepath*filename)
    return
end

Stokes2D()

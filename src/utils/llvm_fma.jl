############# Partition nq into batches with W lanes #############
#TODO Give manual control over the lane count and vectorization strategy for the LLVM FMA operations based on the
# user's CPU. On M3 MAX W=4 gave some benefit to performance, while larger W tank performance. For a CPU that 
# had the 512 bit AVX-512 registers W=8 gave the best benefit to performance with Beyn (experimental)

# Used for Beyn's method where we simultaneously need nq matrices, therefore nq*O(N^2) evaluations of horner recurrences
# So what is most optimal is to batch the matrices into groups divisible by lane count and then do W-lane FMA in each block
# this way we can get W evaluations of matrices's [i,j] entry in that block more efficiently

@inline function partition_nq_W_lanes(nq::UInt32,W::UInt32) # unsigned Int32 is perfectly sufficient
    # this should run just after taylor degree tuning logically as p Taylor degree is determined
    return divrem(nq,W) # number of full batches, number of leftover nodes
end

############# Helpers for conversions LLVM <-> ComplexF64 #############

#TODO Refactor LLVM_Tuple into LLVMLanes. Annoying so just do it later
const LLVM_Tuple=NTuple{2,VecElement{Float64}} # to represent a 2-lane number (real and imaginary parts)
const LLVMLanes{W}=NTuple{W,VecElement{Float64}} # to represent a W-lane number (W / 2 complex numbers). This is for arbitrary-width vectorized operations using LLVM intrinsics if the register width allows it

@inline function llvm_lanes(z::ComplexF64)::LLVM_Tuple # wrapper for a 2 lane number (real and imaginary parts of z) 
    return (VecElement(real(z)),VecElement(imag(z)))
end

@inline function from_llvm_lanes(x::LLVM_Tuple)::ComplexF64 # generate back the ComplexF64 number from LLVM lanes
    return ComplexF64(x[1].value,x[2].value)
end

############# Calls to LLVM FMA intrinsics #############

# declare: LLVM signature of FMA (W dependant)
# define: body of function starts
# @entry: names the wrapper that Base.llvmcall selects with "entry".
# %a,b,c: input vectors to the FMA operation (a * b + c) of dimension W (W/2 complex numbers)
# %r: holds the result vector, which ret returns to Julia.

@inline function _llvm_fma(a::LLVM_Tuple,b::LLVM_Tuple,c::LLVM_Tuple)::LLVM_Tuple
    return Base.llvmcall((
        """
        declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)
        define <2 x double> @entry(<2 x double> %a, <2 x double> %b, <2 x double> %c) alwaysinline {
          %r = call <2 x double> @llvm.fma.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
          ret <2 x double> %r
        }
        """, "entry"
    ),LLVM_Tuple,Tuple{LLVM_Tuple,LLVM_Tuple,LLVM_Tuple},a,b,c)
end

@generated function _llvm_fma(a::LLVMLanes{W},b::LLVMLanes{W},c::LLVMLanes{W}) where {W}
    lanes="<$W x double>"
    intrinsic="llvm.fma.v$(W)f64"
    irreducible_rep="""
        declare $lanes @$intrinsic($lanes, $lanes, $lanes)
        define $lanes @entry($lanes %a, $lanes %b, $lanes %c) alwaysinline {
          %r = call $lanes @$intrinsic($lanes %a, $lanes %b, $lanes %c)
          ret $lanes %r
        }
        """
    return :(Base.llvmcall(($irreducible_rep,"entry"),LLVMLanes{W},Tuple{LLVMLanes{W},LLVMLanes{W},LLVMLanes{W}},a,b,c))
end

############# HIGH LEVEL #############

@inline function _llvm_fma(a::Float64,w::ComplexF64,u::ComplexF64)::ComplexF64
    # scalar-complex mutliply with complex scalar addition: a * w + u.
    return from_llvm_lanes(_llvm_fma(llvm_lanes(ComplexF64(a,a)),llvm_lanes(w),llvm_lanes(u)))
end

@inline _llvm_fma(w::ComplexF64,a::Float64,u::ComplexF64)::ComplexF64=_llvm_fma(a,w,u)

@inline function _llvm_fma(z::ComplexF64,w::ComplexF64,u::ComplexF64)::ComplexF64
    #TODO Uses 2 vector fmas, maybe on some cpus it is faster to do 4 scalar fmas???
    z_re,z_im=reim(z)
    w_re,w_im=reim(w)
    # implement it as llvm_fma on the LLVM lanes of the complex numbers as 2 complex FMAs (faster than 4 scalar FMAs)
    # = z * w + u = llvm_fma(z_re, w, llvm_fma(z_im, i*w, u)).
    # first do the inner llvm_fma(z_im, i*w, u), so z_im needs to be copied into both lanes of the LLVM tuple
    # also expand i*w as ComplexF64(-w_im, w_re)
    z_im_prod=_llvm_fma((VecElement(z_im),VecElement(z_im)),llvm_lanes(ComplexF64(-w_im,w_re)),llvm_lanes(u))
    # then do outer llvm_fma(z_re, w, z_im_prod), where w now can be simply represented as its LLVM lanes since no i multiplication
    result=_llvm_fma(llvm_lanes(ComplexF64(z_re,z_re)),llvm_lanes(w),z_im_prod)
    return from_llvm_lanes(result)
end
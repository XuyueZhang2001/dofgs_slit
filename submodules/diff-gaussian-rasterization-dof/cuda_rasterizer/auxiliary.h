/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)


// defocus blur kernel coefficients
__device__ const float CoC_coeff = 0.36067376022224085f; // 1 / (2ln4)

// ---------------------------------------------------------------------------
// Anisotropic defocus model for a centered, rotatable slit (rectangular)
// aperture.
//
// For a point at view-space depth z and focus distance F, the defocus amount is
//     d  = |1/z - 1/F|
// The half-lengths of the projected rectangular PSF along the long and short
// slit axes are
//     rL = 0.5 * KL * d ,   rW = 0.5 * KW * d
// The PSF is approximated by a Gaussian whose variance along each axis is
//     sL = coeff * rL^2 ,   sW = coeff * rW^2
// (coeff = 1/3 matches the variance of a uniform box of full length 2r;
//  coeff = 1/(2 ln 4) with KL == KW reproduces the original isotropic DOF-GS.)
//
// With the slit long axis u = (cos t, sin t) in image coordinates
// (x to the right, y downwards, t measured from +x towards +y), the added
// screen-space covariance is
//     dSigma = sL * u u^T + sW * v v^T ,  v = (-sin t, cos t)
// i.e.
//     da = sL cos^2 t + sW sin^2 t
//     db = (sL - sW) sin t cos t
//     dc = sL sin^2 t + sW cos^2 t
//
// Because the aperture is centered on the optical axis, the PSF is symmetric
// under a 180-degree rotation, so points in front of and behind the focal
// plane produce the same PSF shape and the PSF center does not shift.
// ---------------------------------------------------------------------------
struct SlitCoC
{
	float D;      // signed defocus 1/z - 1/F
	float d;      // |D|
	float rL, rW; // half-lengths along the long / short slit axis (pixels)
	float sL, sW; // Gaussian variances along the long / short slit axis
	float cs, sn; // cos(theta), sin(theta)
	float da, db, dc; // increments added to cov2D (xx, xy, yy)
};

__forceinline__ __device__ SlitCoC computeSlitCoC(
	float z, float F, float KL, float KW, float theta, float coeff)
{
	SlitCoC s;
	s.D = 1.0f / max(z, 0.0000001f) - 1.0f / F;
	s.d = fabsf(s.D);
	s.rL = 0.5f * KL * s.d;
	s.rW = 0.5f * KW * s.d;
	s.sL = coeff * s.rL * s.rL;
	s.sW = coeff * s.rW * s.rW;
	sincosf(theta, &s.sn, &s.cs);
	const float cs2 = s.cs * s.cs;
	const float sn2 = s.sn * s.sn;
	s.da = s.sL * cs2 + s.sW * sn2;
	s.db = (s.sL - s.sW) * s.sn * s.cs;
	s.dc = s.sL * sn2 + s.sW * cs2;
	return s;
}

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x4Transpose(const float4& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3] * p.w,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7] * p.w,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11] * p.w,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view,
	float& dist)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);
	dist = glm::length(glm::vec3(p_view.x, p_view.y, p_view.z));
	if (p_view.z <= 0.2f)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// In debug mode, also query cudaGetLastError(): launch-time failures (e.g.
// "no kernel image is available", "too many resources requested") are not
// returned by cudaDeviceSynchronize().
#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaGetLastError(); \
if (ret == cudaSuccess) ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
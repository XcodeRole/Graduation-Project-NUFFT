#include "common.h"
#include <vector>
#include <string>
#include <algorithm>

TDEF(fftw)
TDEF(nufft)
TDEF(pwj)

/* Constructor */
NUFFT3D::NUFFT3D(int N, int OF, float *wx, float *wy, float *wz, int P, int prechopX, int prechopY, int prechopZ, int postchopX, int postchopY, int postchopZ, int offsetX, int offsetY, int offsetZ, int W, int L)
{

	// Assignments
	this->N = N;
	this->OF = OF;
	N2 = N * OF;
	this->wx = wx;
	this->wy = wy;
	this->wz = wz;
	this->P = P;
	this->prechopX = prechopX;
	this->prechopY = prechopY;
	this->prechopZ = prechopZ;
	this->postchopX = postchopX;
	this->postchopY = postchopY;
	this->postchopZ = postchopZ;
	this->offsetX = offsetX;
	this->offsetY = offsetY;
	this->offsetZ = offsetZ;
	this->W = W;
	this->L = L;

	int DIMS[3] = {N2, N2, N2};
	f = (complex<float> *)memalign(16, N2 * N2 * N2 * sizeof(complex<float>));
	fwdPlan = fftwf_plan_dft(3, DIMS, reinterpret_cast<fftwf_complex *>(f), reinterpret_cast<fftwf_complex *>(f), FFTW_FORWARD, FFTW_ESTIMATE);
	adjPlan = fftwf_plan_dft(3, DIMS, reinterpret_cast<fftwf_complex *>(f), reinterpret_cast<fftwf_complex *>(f), FFTW_BACKWARD, FFTW_ESTIMATE);
	buildLUT();
	getScalingFunction();
}

/* Destructor */
NUFFT3D::~NUFFT3D()
{
	fftwf_destroy_plan(fwdPlan);
	fftwf_destroy_plan(adjPlan);
	free(f);
	f = NULL;
}

/* Initialize multithreaded FFTW (p threads) */
void NUFFT3D::init(int nThreads)
{
	fftwf_init_threads();
	fftwf_plan_with_nthreads(nThreads);
}

/* Forward NUFFT transform */
void NUFFT3D::fwd(complex<float> *u, complex<float> *raw, float *kx, float *ky, float *kz)
{
	// Apodization and zero-padding
	int startX = N * (OF - 1) / 2;
	int startY = N * (OF - 1) / 2;
	int startZ = N * (OF - 1) / 2;
	TSTART(nufft);
	TSTART(fftw);
#pragma omp parallel for schedule(auto)
	for (int i = 0; i < N2 * N2 * N2; i++)
	{
		f[i] = 0;
	}
	TEND(fftw);
	TPRINT(fftw, "  Init_F FWD");
	TSTART(fftw);
#pragma omp parallel for schedule(auto)//collapse(3) schedule(guided)
	for (int x = 0; x < N; x++)
	{
		for (int y = 0; y < N; y++)
		{
			for (int z = 0; z < N; z++)
			{
				f[(x + startX + offsetX) * N2 * N2 + (y + startY + offsetY) * N2 + (z + startZ + offsetZ)] = u[x * N * N + y * N + z] / q[x * N * N + y * N + z];
			}
		}
	}
	TEND(fftw);
	TPRINT(fftw, "  Roundoff_Corr FWD");

	// (Oversampled) FFT
	TSTART(fftw);
	chop3D(f, N2, N2, N2, prechopX, prechopY, prechopZ);
	TEND(fftw);
	TPRINT(fftw, "  PreChop FWD")
	TSTART(fftw);
	fftwf_execute(fwdPlan);
	TEND(fftw);
	TPRINT(fftw, "  FFTW FWD");
	TSTART(fftw)
	chop3D(f, N2, N2, N2, postchopX, postchopY, postchopZ);
	TEND(fftw);
	TPRINT(fftw, "  PostChop FWD");

	// Pull from grid
	TSTART(fftw);
	int Radius = 2 * W + 1;
#pragma omp parallel for schedule(auto)
#pragma unroll_and_jam(16)
	for (int p = 0; p < P; p++)
	{
		int kx2[Radius];
		int ky2[Radius];
		int kz2[Radius];
		float winX[Radius];
		float winY[Radius];
		float winZ[Radius];

		// Form x interpolation kernel
		// float kx = N2 * (wx[p] + 0.5);
		int x1 = (int)ceil(kx[p] - W);
		int x2 = (int)floor(kx[p] + W);
		int lx = x2 - x1 + 1;
		for (int nx = 0; nx < lx; nx++)
		{
			int kxx=nx+x1;
			kx2[nx] =(kxx>0&&kxx<N2)?kxx:mod(kxx, N2); //Group by this
			winX[nx] = LUT[(int)round(((L - 1) / W) * abs(kxx - kx[p]))];
			// kx2[nx] = mod(nx + x1, N2);
			// winX[nx] = LUT[(int)round(((L - 1) / W) * abs(nx + x1 - kx[p]))];
		}

		// Form y interpolation kernel
		// float ky = N2 * (wy[p] + 0.5);
		int y1 = (int)ceil(ky[p] - W);
		int y2 = (int)floor(ky[p] + W);
		int ly = y2 - y1 + 1;
		for (int ny = 0; ny < ly; ny++)
		{
			int kyy=ny+y1;
			ky2[ny] = (kyy>0&&kyy<N2)?kyy:mod(kyy, N2);
			winY[ny] = LUT[(int)round(((L - 1) / W) * abs(kyy - ky[p]))];
			// ky2[ny] = mod(ny + y1, N2);
			// winY[ny] = LUT[(int)round(((L - 1) / W) * abs(ny + y1 - ky[p]))];
		}

		// Form z interpolation kernel
		// float kz = N2 * (wz[p] + 0.5);
		int z1 = (int)ceil(kz[p] - W);
		int z2 = (int)floor(kz[p] + W);
		int lz = z2 - z1 + 1;
		for (int nz = 0; nz < lz; nz++)
		{
			int kzz=nz+z1;
			kz2[nz] = (kzz>0&&kzz<N2)?kzz:mod(kzz, N2);
			winZ[nz] = LUT[(int)round(((L - 1) / W) * abs(kzz - kz[p]))];
			// kz2[nz] = mod(nz + z1, N2);
			// winZ[nz] = LUT[(int)round(((L - 1) / W) * abs(nz + z1 - kz[p]))];
		}

		int nx,ny,nz;
		complex<float> tmp=0;
		int dim1,dim2;
		__m512 avx_f,avx_winXY,avx_winZ,avx_tmp,avx_mid;
		avx_tmp=_mm512_set1_ps(0);
		for (nx = 0; nx < 2*W; nx++)
		{
			dim1=kx2[nx] * N2 * N2;
			for (ny=0; ny < 2*W; ny++)
			{
				dim2=dim1+ky2[ny] * N2;
				float winXY=winX[nx] * winY[ny];
				complex<float> array_f[]={f[dim2+kz2[0]],f[dim2+kz2[1]],f[dim2+kz2[2]],f[dim2+kz2[3]],f[dim2+kz2[4]],f[dim2+kz2[5]],f[dim2+kz2[6]],f[dim2+kz2[7]]};
				avx_f=(__m512)_mm512_load_pd(array_f);
				avx_winXY=_mm512_set1_ps(winXY);
				avx_winZ=_mm512_set_ps(winZ[7],winZ[7],winZ[6],winZ[6],winZ[5],winZ[5],winZ[4],winZ[4],winZ[3],winZ[3],winZ[2],winZ[2],winZ[1],winZ[1],winZ[0],winZ[0]);
				avx_winZ=_mm512_mul_ps(avx_winZ,avx_winXY);
				avx_tmp=_mm512_fmadd_ps(avx_winZ,avx_f,avx_tmp);
			}
		}
		for(int i=0;i<8;i++){
			tmp+=complex<float>(avx_tmp[2*i],avx_tmp[2*i+1]);
		}

		for (; nx < lx; nx++)
		{
			for (; ny < ly; ny++)
			{
				float winXY=winX[nx] * winY[ny];
				for (; nz < lz; nz++)
				{
					tmp += f[kx2[nx] * N2 * N2 + ky2[ny] * N2 + kz2[nz]] * winXY* winZ[nz];
				}
			}
		}
		raw[p]=tmp;
		// Interpolation
		// complex<float> tmp = 0;
		// int dim3, dim2;
		// float winXY;
		
		// for (int nx = 0; nx < lx; nx++)
		// {
		// 	dim3 = kx2[nx] * N2 * N2;
		// 	for (int ny = 0; ny < ly; ny++)
		// 	{
		// 		dim2 = dim3 + ky2[ny] * N2;
		// 		winXY = winX[nx] * winY[ny];
		// 		//#pragma omp simd
		// 		for (int nz = 0; nz < lz; nz++)
		// 		{
		// 			tmp += f[dim2 + kz2[nz]] * winXY * winZ[nz];
		// 		}
		// 	}
		// }
		// raw[p] = tmp;

	}

	TEND(fftw);
	TPRINT(fftw, "  Convolution FWD");
	TEND(nufft);
	TPRINT(nufft, "NUFFT FWD");
}

/* Adjoint NUFFT transform */
void NUFFT3D::adj(complex<float> *raw, complex<float> *u, vector<vector<vector<int>>> &groupGray, vector<vector<int>> &Orphan, float *kx, float *ky, float *kz)
{
	TSTART(nufft);

	// Push to grid
	TSTART(fftw);
#pragma omp parallel for schedule(auto)
	for (int i = 0; i < N2 * N2 * N2; i++)
	{
		f[i] = 0;
	}
	TEND(fftw);
	TPRINT(fftw, "  Init_F ADJ");
	TSTART(fftw)

	//分组完成后的运算
	TSTART(pwj);
#pragma unroll_and_jam(8)
	for (int g = 0; g < 8; g++)
	{
#pragma omp parallel for schedule(monotonic:dynamic, 16)//
		for (int i = 0; i < groupGray[g].size(); i++)
		{
			#pragma unroll_and_jam(8)
			for (int j = 0; j < groupGray[g][i].size(); j++)
			{
				int kx2[2 * W + 1];
				int ky2[2 * W + 1];
				int kz2[2 * W + 1];
				float winX[2 * W + 1];
				float winY[2 * W + 1];
				float winZ[2 * W + 1];
				int index = groupGray[g][i][j];

				// Form x interpolation kernel
				int x1 = (int)ceil(kx[index] - W);
				int x2 = (int)floor(kx[index] + W);
				int lx = x2 - x1 + 1; 
				for (int nx = 0; nx < lx; nx++)
				{								//Points scattered by one point
					int kxx=nx+x1;
					kx2[nx] =(kxx>0&&kxx<N2)?kxx:mod(kxx, N2); //Group by this
					winX[nx] = LUT[(int)round(((L - 1) / W) * abs(kxx - kx[index]))];
				}

				// Form y interpolation kernel
				int y1 = (int)ceil(ky[index] - W);
				int y2 = (int)floor(ky[index] + W);
				int ly = y2 - y1 + 1;
				for (int ny = 0; ny < ly; ny++)
				{
					int kyy=ny+y1;
					ky2[ny] = (kyy>0&&kyy<N2)?kyy:mod(kyy, N2);
					winY[ny] = LUT[(int)round(((L - 1) / W) * abs(kyy - ky[index]))];
				}

				// Form z interpolation kernel
				int z1 = (int)ceil(kz[index] - W);
				int z2 = (int)floor(kz[index] + W);
				int lz = z2 - z1 + 1;
				for (int nz = 0; nz < lz; nz++)
				{
					int kzz=nz+z1;
					kz2[nz] = (kzz>0&&kzz<N2)?kzz:mod(kzz, N2);
					winZ[nz] = LUT[(int)round(((L - 1) / W) * abs(kzz - kz[index]))];
				}

				int dim3, dim2;
				float winXY;
				// Interpolation
				for (int nx = 0; nx < lx; nx++)
				{
					dim3 = kx2[nx] * N2 * N2;
					for (int ny = 0; ny < ly; ny++)
					{
						dim2 = dim3 + ky2[ny] * N2;
						winXY = winX[nx] * winY[ny];
						//#pragma omp simd
						for (int nz = 0; nz < lz; nz++)
						{
							f[dim2 + kz2[nz]] += raw[index] * winXY * winZ[nz];
						}
					}
				}
			}
		} //End Group
	}	  //End GrayCode

	TEND(fftw);
	TPRINT(fftw, "  Convolution ADJ");
	// (Oversampled) FFT
	TSTART(fftw);
	chop3D(f, N2, N2, N2, postchopX, postchopY, postchopZ);
	TEND(fftw);
	TPRINT(fftw, "  PostChop ADJ");
	TSTART(fftw);
	fftwf_execute(adjPlan);
	TEND(fftw);
	TPRINT(fftw, "  FFTW ADJ");
	TSTART(fftw);
	chop3D(f, N2, N2, N2, prechopX, prechopY, prechopZ);
	TEND(fftw);
	TPRINT(fftw, "  PreChop ADJ");
	// Deapodize and truncate
	int startX = N * (OF - 1) / 2;
	int startY = N * (OF - 1) / 2;
	int startZ = N * (OF - 1) / 2;
	TSTART(fftw);
	for (int i = 0; i < N * N * N; i++)
	{
		u[i] = 0;
	}
	TEND(fftw);
	TPRINT(fftw, "  Init_U ADJ");
	TSTART(fftw)
	for (int x = 0; x < N; x++)
	{
		for (int y = 0; y < N; y++)
		{
			for (int z = 0; z < N; z++)
			{
				u[x * N * N + y * N + z] = f[(x + startX + offsetX) * N2 * N2 + (y + startY + offsetY) * N2 + (z + startZ + offsetZ)] / q[x * N * N + y * N + z];
			}
		}
	}
	TEND(fftw);
	TPRINT(fftw, "  Roundoff_Corr ADJ");
	TEND(nufft);
	TPRINT(nufft, "NUFFT ADJ")
	return;
}

/* Internal lookup table generation function for interpolation kernel (Kaiser-Bessel) */
void NUFFT3D::buildLUT()
{
	LUT = new float[L];
	float *d = linspace<float>(0, W, L);
	float pi = boost::math::constants::pi<float>();
	float alpha = pi * sqrt(((2 * (float)W / OF) * (OF - 0.5)) * ((2 * (float)W / OF) * (OF - 0.5)) - 0.8);
	for (int l = 0; l < L; l++)
	{
		LUT[l] = boost::math::cyl_bessel_i(0, alpha * sqrt(1 - (d[l] * d[l]) / (W * W))) / boost::math::cyl_bessel_i(0, alpha);
	}
}

/* Internal scaling generation function */
void NUFFT3D::getScalingFunction()
{

	float dx, dy, dz;
	float s = 0;

	// Create a volume with a copy of the interpolation kernel centered at the origin, then normalize
	for (int i = 0; i < N2 * N2 * N2; i++)
	{
		f[i] = 0;
	}
	for (int x = N2 / 2 - W; x <= N2 / 2 + W; x++)
	{
		dx = abs(((float)x - N2 / 2) / W);
		for (int y = N2 / 2 - W; y <= N2 / 2 + W; y++)
		{
			dy = abs(((float)y - N2 / 2) / W);
			for (int z = N2 / 2 - W; z <= N2 / 2 + W; z++)
			{
				dz = abs(((float)z - N2 / 2) / W);
				f[x * N2 * N2 + y * N2 + z] = complex<float>(LUT[(int)round((L - 1) * dx)] * LUT[(int)round((L - 1) * dy)] * LUT[(int)round((L - 1) * dz)], 0);
				s = s + norm(f[x * N2 * N2 + y * N2 + z]);
			}
		}
	}
	s = sqrt(s);
	for (int x = N2 / 2 - W; x <= N2 / 2 + W; x++)
	{
		for (int y = N2 / 2 - W; y <= N2 / 2 + W; y++)
		{
			for (int z = N2 / 2 - W; z <= N2 / 2 + W; z++)
			{
				f[x * N2 * N2 + y * N2 + z] = f[x * N2 * N2 + y * N2 + z] / s;
			}
		}
	}

	// (Oversampled) FFT
	chop3D(f, N2, N2, N2, postchopX, postchopY, postchopZ);
	fftwf_execute(adjPlan);
	chop3D(f, N2, N2, N2, prechopX, prechopY, prechopZ);

	// Truncate and keep only the real component (presuming Fourier domain symmetry)
	q = new float[N * N * N];
	int startX = N * (OF - 1) / 2;
	int startY = N * (OF - 1) / 2;
	int startZ = N * (OF - 1) / 2;
	for (int i = 0; i < N * N * N; i++)
	{
		q[i] = 0;
	}
	for (int x = 0; x < N; x++)
	{
		for (int y = 0; y < N; y++)
		{
			for (int z = 0; z < N; z++)
			{
				q[x * N * N + y * N + z] = real(f[(x + startX + offsetX) * N2 * N2 + (y + startY + offsetY) * N2 + (z + startZ + offsetZ)]);
			}
		}
	}

	return;
}

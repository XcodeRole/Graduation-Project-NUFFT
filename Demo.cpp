#include "common.h"
#include <vector>

chrono::high_resolution_clock::time_point fwd_t0, fwd_t1, adj_t0, adj_t1, wall_t0, wall_t1,group_t0,group_t1;

using namespace std;

int main(int argc, char **argv)
{
	char rawfn[256], coordfn[256], ksfn[256], reffn[256], outfn[256];
	char base[256];
	int K, S;
	int numIters = 20;

	int numThreads = 1;
	if (argc < 2)
	{
		printf("Usages: %s <basename> [numThreads] [numIterations]\n", argv[0]);
	}
	strncpy(base, argv[1], 255);
	snprintf(ksfn, 256, "%s.%s", base, "KS");
	snprintf(rawfn, 256, "%s.%s", base, "raw");
	snprintf(coordfn, 256, "%s.%s", base, "coord");

	if (argc > 2)
	{
		numThreads = atoi(argv[2]);
		if (numThreads > 0)
		{
			omp_set_num_threads(numThreads);
		}
	}
	numThreads = omp_get_max_threads();
	printf("Using %d Threads\n", numThreads);
	if (argc > 3)
	{
		numIters = atoi(argv[3]);
		if (numIters > 20)
			numIters = 20;
		if (numIters < 1)
			printf("Number of interarions must be > 0.\n");
	}
	/* ****************** */
	/* *** Load files *** */
	/* ********************************************************************* */

	fstream in;

	// Read raw data file
	in.open(ksfn, ios::in);
	if (!in)
	{
		printf("Can't open file \"%s\"\n", ksfn);
		exit(1);
	}
	in >> K >> S;
	in.close();
	printf("K = %d , S = %d, TotalSamples = %d\n", K, S, K * S);
	float *temp = new float[K * S * 2];

	in.open(rawfn, ios::in | ios::binary);
	if (!in)
	{
		printf("Can't open file \"%s\"\n", rawfn);
		exit(1);
	}
	in.seekg(0, ios::beg);
	in.read(reinterpret_cast<char *>(temp), K * S * 2 * sizeof(float));
	complex<float> *raw = new complex<float>[K * S];
	for (int s = 0; s < S; s++)
	{
		for (int k = 0; k < K; k++)
		{
			raw[s * K + k] = complex<float>(temp[2 * (s * K + k)], temp[2 * (s * K + k) + 1]);
		}
	}
	delete[] temp;
	temp = NULL;
	in.close();

	// Read sampling file
	float *temp2 = new float[K * S * 3];
	in.open(coordfn, ios::in | ios::binary);
	if (!in)
	{
		printf("Can't open file \"%s\"\n", coordfn);
		exit(1);
	}
	in.seekg(0, ios::beg);
	in.read(reinterpret_cast<char *>(temp2), K * S * 3 * sizeof(float));
	float *wx = new float[K * S];
	float *wy = new float[K * S];
	float *wz = new float[K * S];
	for (int s = 0; s < S; s++)
	{
		for (int k = 0; k < K; k++)
		{
			wx[s * K + k] = (0.1) * temp2[0 * S * K + s * K + k];
			wy[s * K + k] = (0.1) * temp2[1 * S * K + s * K + k];
			wz[s * K + k] = (0.1) * temp2[2 * S * K + s * K + k];
		}
	}
	delete[] temp2;
	temp2 = NULL;
	in.close();

	/* ********************************* */
	/* *** Initialize NUFFT Operator *** */
	/* ********************************************************************* */

	// Multithreaded initialization
	NUFFT3D::init(numThreads);

	// Initialize NUFFT operator specifics
	int N = 220;
	int W = 4;
	int OF = 2;
	int L = 25000;
	int P = K * S;
	int prechopX = 1;
	int postchopX = 1;
	int prechopY = 1;
	int postchopY = 1;
	int prechopZ = 1;
	int postchopZ = 1;
	int offsetX = 0;
	int offsetY = 0;
	int offsetZ = -35;
	NUFFT3D *nufft = new NUFFT3D(N, OF, wx, wy, wz, P, prechopX, prechopY, prechopZ, postchopX, postchopY, postchopZ, offsetX, offsetY, offsetZ, W, L);

	/* ******************************** */
	/* *** Begin CG Iterative Recon *** */
	/* ********************************************************************* */

	float lambda = 1e-10;
	int maxCGIter = numIters;
	float epsilon = 1e-30;

	// Standard CG algorithm for solving (A'A+lambdaI)u = A'f;
	complex<float> alpha, den;
	float beta, delta_old, delta;
		complex<float> *u = new complex<float>[N*N*N];
	complex<float> *r = new complex<float>[N*N*N];
	complex<float> *p = new complex<float>[N*N*N];
	complex<float> *Ap = new complex<float>[N*N*N];
	complex<float> *z = new complex<float>[P];

	wall_t0 = Clock::now();
	adj_t0 = Clock::now();

	//myChang start----------------------------------------------------------
	int N2 = N * OF;
	
	float *kx = (float *)memalign(64, P * sizeof(float));
	float *ky = (float *)memalign(64, P * sizeof(float));
	float *kz = (float *)memalign(64, P * sizeof(float));
#pragma omp parallel for
	for (int p = 0; p < P; p++)
	{
		kx[p] = N2 * (wx[p] + 0.5);
		ky[p] = N2 * (wy[p] + 0.5);
		kz[p] = N2 * (wz[p] + 0.5);
	}
	// int myArrayLen = (int)floor(N2 * 1.0 / Radius); //向下取整，超过myArrayLen的坐标直接对MyArrayLen取模

	int partition[3][N2];
	int numPartition[3];
	int minWid = 2 * W + 1;
	int avg = P / numThreads;
	// cout << "------------------avg = " << avg << endl;
	//Chop and Partition
	// cout << "----------Chop and Partition--------------" << endl;
    
    int hist[3][N2];
    #pragma omp parallel for
    for(int i=0; i<N2; i++){
        hist[0][i]=0;
        hist[1][i]=0;
        hist[2][i]=0;
    }
    for(int p = 0; p < P; p++){
        hist[0][(int)kx[p]]++; 
        hist[1][(int)ky[p]]++; 
        hist[2][(int)kz[p]]++; 
    }
    for(int i=1; i<N2; i++){
        hist[0][i]+=hist[0][i-1];
        hist[1][i]+=hist[1][i-1];
        hist[2][i]+=hist[2][i-1];
    }
#pragma unroll_and_jam(3)
	for (int d = 0; d < 3; d++)
	{
		int i = 0, start = 0, end;
		partition[d][0] = 0;
		while (start < N2)
		{
			end = start + minWid;
			while (hist[d][end] - hist[d][start] < avg)
			{
				end++;
				if (end >= N2)
				{
					end = N2;
					break;
				}
			}
			i++;
			partition[d][i] = end;
			start = end;
		}
		numPartition[d] = i; //用于之后进行奇偶性判断等操作
	}
    // group_t0 = Clock::now();
    // cout << "Hist time = \t\t" << chrono::duration_cast<chrono::microseconds>(group_t0 - adj_t0).count() / 1e6 << "\t\t sec" << endl;
	//Group
	// cout << "------------------Group--------------------" << endl;
	vector<vector<int>> tempGroup;		   //声明用于存储分组的向量
	vector<vector<vector<int>>> groupGray; //声明用于存储分组根据格雷码分组的向量
	vector<vector<int>> Orphan;			   //声明用于存储三个维度孤儿的向量
	int  tempGroup_len = numPartition[0] * numPartition[1] * numPartition[2];
	tempGroup.resize(numPartition[0] * numPartition[1] * numPartition[2]);
	groupGray.resize(8);
	Orphan.resize(3);

	for (int p = 0; p < P; p++) {
		//长度为奇，且坐标位于最外层，则将其剥出去
		int tx = (int)ceil(kx[p] + W + 2); //lx - 1 + x1 = x2;最远距离
		int ty = (int)ceil(ky[p] + W + 2);
		int tz = (int)ceil(kz[p] + W + 2);
		int xloc = -1, yloc = -1, zloc = -1;
		if (numPartition[0] % 2 == 1 && tx >= N2)
		{
			xloc = 0;
		}
		else if (numPartition[1] % 2 == 1 && tx >= N2)
		{
			yloc = 0;
		}
		else if (numPartition[2] % 2 == 1 && tz >= N2)
		{
			zloc = 0;
		}
		for (int i = 0; i < numPartition[0]; i++) {
			if (kx[p] < partition[0][i + 1])  {
				xloc = i;
				break;
			}
		}
		for (int i = 0; i < numPartition[1]; i++) {
			if (ky[p] < partition[1][i + 1]) {
				yloc = i;
				break;
			}
		}
		for (int i = 0; i < numPartition[2]; i++) {
			if (kz[p] < partition[2][i + 1]) {
				zloc = i;
				break;
			}
		}
		int index  = xloc * numPartition[1] * numPartition[2] + yloc * numPartition[2] + zloc;
		//if(tempGroup_len <= index) cout<<index<<"fuck-";
		tempGroup[index].emplace_back(p);
	}

	//Group by GrayCode
	// cout << "------------------Group by GrayCode--------------------" << endl;
	//group_t0 = Clock::now();
	for (int x = 0; x < numPartition[0]; x++)
	{
		for (int y = 0; y < numPartition[1]; y++)
		{
			for (int z = 0; z < numPartition[2]; z++)
			{
				string grayCode;
				// grayCode = to_string(fmod((int)kx[p], 2)) + to_string(fmod((int)ky[p], 2)) + to_string(fmod((int)kz[p], 2));
				grayCode = to_string(mod(x, 2)) + to_string(mod(y, 2)) + to_string(mod(z, 2));
				// cout << "grayCode = " << grayCode << "   " << endl;
				int index = x * numPartition[1] * numPartition[2] + y * numPartition[2] + z;
				if (grayCode == "000")
				{
					groupGray[0].emplace_back(tempGroup[index]);
				}
				else if (grayCode == "001")
				{
					groupGray[1].emplace_back(tempGroup[index]);
				}
				else if (grayCode == "011")
				{
					groupGray[2].emplace_back(tempGroup[index]);
				}
				else if (grayCode == "010")
				{
					groupGray[3].emplace_back(tempGroup[index]);
				}
				else if (grayCode == "110")
				{
					groupGray[4].emplace_back(tempGroup[index]);
				}
				else if (grayCode == "111")
				{
					groupGray[5].emplace_back(tempGroup[index]);
				}
				else if (grayCode == "101")
				{
					groupGray[6].emplace_back(tempGroup[index]);
				}
				else if (grayCode == "100")
				{
					groupGray[7].emplace_back(tempGroup[index]);
				}
			}
		}
	}
    group_t1 = Clock::now();
    cout << "Group time = \t\t" << chrono::duration_cast<chrono::microseconds>(group_t1 - adj_t0).count() / 1e6 << "\t\t sec" << endl;
	cout << "=================================================" << endl;

	nufft->adj(raw, r, groupGray, Orphan, kx, ky, kz);
	adj_t1 = Clock::now();
	cout << "ADJ time = \t\t" << chrono::duration_cast<chrono::microseconds>(adj_t1 - adj_t0).count() / 1e6 << "\t\t sec" << endl;
	cout << "=================================================" << endl;
	for (int i = 0; i < N * N * N; i++)
	{
		u[i] = 0;
		p[i] = r[i];
	}
	delta_old = 0;
	for (int i = 0; i < N * N * N; i++)
		delta_old += norm(r[i]);
	for (int iter = 0; iter < maxCGIter; iter++)
	{
		cout << "Iteration " << iter + 1 << " :" << endl;
		fwd_t0 = Clock::now();
		nufft->fwd(p, z, kx, ky, kz);
		fwd_t1 = Clock::now();
		cout << "FWD time = \t\t" << chrono::duration_cast<chrono::microseconds>(fwd_t1 - fwd_t0).count() / 1e6 << "\t\t sec" << endl;
		cout << "-------------------------------------------------" << endl;
		adj_t0 = Clock::now();
		nufft->adj(z, Ap, groupGray, Orphan, kx, ky, kz);
		adj_t1 = Clock::now();
		cout << "ADJ time = \t\t" << chrono::duration_cast<chrono::microseconds>(adj_t1 - adj_t0).count() / 1e6 << "\t\t sec" << endl;
		for (int i = 0; i < N * N * N; i++)
			Ap[i] += lambda * p[i];
		den = epsilon;
		for (int i = 0; i < N * N * N; i++)
			den += conj(p[i]) * Ap[i];
		alpha = delta_old / den;
		for (int i = 0; i < N * N * N; i++)
		{
			u[i] = u[i] + alpha * p[i];
			r[i] = r[i] - alpha * Ap[i];
		}
		delta = 0;
		for (int i = 0; i < N * N * N; i++)
			delta += norm(r[i]);
		beta = delta / (delta_old + epsilon);
		delta_old = delta;
		for (int i = 0; i < N * N * N; i++)
			p[i] = r[i] + beta * p[i];
		cout << "Iteration " << iter + 1 << " out of " << maxCGIter << endl;
		cout << "=================================================" << endl;
	}
	wall_t1 = Clock::now();
	cout << "Total Computation Time:" << endl
		 << "\t\t" << chrono::duration_cast<chrono::microseconds>(wall_t1 - wall_t0).count() / 1e6 << "\t sec" << endl;
	delete nufft;

	/* ******************** */
	/* *** Save results *** */
	/* ********************************************************************* */
	snprintf(outfn, 256, "%s.%s.%d", base, "bin", maxCGIter);
	fstream out;
	out.open(outfn, ios::out | ios::binary);
	out.seekp(0, ios::beg);
	out.write(reinterpret_cast<char *>(u), N * N * N * sizeof(complex<float>));
	out.close();

	return 0;
}

//============================================================================
// Name        : RecursiveLU.cpp
// Author      : Tomohiro Suzuki
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <mkl.h>

using namespace std;

// Generate random matrix
void Gen_rand_mat(const int m, const int n, double *A)
{
	srand(20191225);

//	#pragma omp parallel for
	for (int i=0; i<m*n; i++)
		A[i] = 1.0 - 2*(double)rand() / RAND_MAX;
}

// Show matrix
void Show_mat(const int m, const int n, double *A)
{
//	cout.setf(ios::scientific);
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++)
			cout << A[i + j*m] << ", ";
//		cout << showpos << setprecision(4) << A[i + j*m] << ", ";
		cout << endl;
	}
	cout << endl;
}

// Show vector
void Show_vec(const int m, int *a)
{
	for (int i=0; i<m; i++)
		cout << a[i] << ", ";
	cout << endl << endl;;
}

int dgetrfr(const int m, const int n, double *A, const int lda, int *piv)
{
	cout << "n = " << n << endl;
	int info = -1;

	if (n == 1) // Single column, recursion stops
	{
		int idx = cblas_idamax(m,A,1);      // find maximum
		*piv = idx+1;

		double tmp = *(A+idx);
		if (tmp == 0.0)
			return 0;
		else
		{
			cblas_dscal(m,1.0/tmp,A,1);
			*(A+idx) = *A;
			*A = tmp;
		}
	}
	else
	{
		int nleft = n / 2;
		int nright = n - nleft;

		info = dgetrfr(m,nleft,A,lda,piv);           // recursive call to factor left half
		if (info != -1)
			return info;

		// pivoting forward
		assert(0 == LAPACKE_dlaswp(LAPACK_COL_MAJOR, nright, A+(nleft*lda), lda, 1, nleft, piv, 1));

		// triangular solve
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
				nleft, nright, 1.0, A, lda, A+(nleft*lda), lda);

		// Schur's complement
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				m-nleft, nright, nleft, -1.0, A+(nleft), lda, A+(nleft*lda), lda, 1.0, A+(nleft+nleft*lda), lda);

		info = dgetrfr(m-nleft,nright,A+(nleft+nleft*lda),lda,piv+nleft); // recursive call to factor right half
		if (info != -1)
		{
			info += nleft;
			return info;
		}

		for (int k=nleft; k<m; k++)
			piv[k] += nleft;

		// pivoting backward
		assert(0 == LAPACKE_dlaswp(LAPACK_COL_MAJOR, nleft, A, lda, nleft+1, n, piv, 1));
	}

	return info;
}

// Debug mode
#define DEBUG

// Trace mode
//#define TRACE

#ifdef TRACE
extern void trace_cpu_start();
extern void trace_cpu_stop(const char *color);
extern void trace_label(const char *color, const char *label);
#endif

int main(const int argc, const char **argv)
{
	// Usage "a.out [size of matrix: m n ]"
	assert(argc > 2);

	const int m = atoi(argv[1]);     // # rows
	const int n = atoi(argv[2]);     // # columns
	assert(m >= n);

	double *A = new double [m*n];    // Original matrix
	int *piv = new int [m];          // permutation vector
	const int lda = m;

//	Gen_rand_mat(m,n,A);             // Randomize elements of orig. matrix

	A[0] = 2.0; A[1] = 4.0; A[2] = 1.0;
	A[3] = 1.0; A[4] = 2.0; A[5] = 3.0;
	A[6] = 3.0; A[7] = 5.0; A[8] = 1.0;

	Show_mat(m,n,A);
	Show_vec(m,piv);

	////////// Debug mode //////////
	#ifdef DEBUG
	double *OA = new double[m*n];
	cblas_dcopy(m*n, A, 1, OA, 1);
	double *U = new double[n*n];
	#endif
	////////// Debug mode //////////

	double timer = omp_get_wtime();   // Timer start

	int info = dgetrfr(m,n,A,lda,piv);

	timer = omp_get_wtime() - timer;   // Timer stop

	Show_mat(m,n,A);
	Show_vec(m,piv);

	cout << "m = " << m << ", n = " << n << ", time = " << timer << endl;
	cout << "Info = " << info << endl;

	////////// Debug mode //////////
	#ifdef DEBUG
	// Upper triangular matrix
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++)
			U[i+j*n] = (j<i) ? 0.0 : A[i+j*m];

	// Unit lower triangular matrix
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			if (i==j)
				A[i+j*m] = 1.0;
			else if (j>i)
				A[i+j*m] = 0.0;
		}

	// Apply interchanges to original matrix A
	assert(0 == LAPACKE_dlaswp(MKL_COL_MAJOR, n, OA, m, 1, n, piv, 1));

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
			m, n, n, -1.0, A, m, U, n, 1.0, OA, m);

	cout << "Debug mode: \n";
	cout << "|| PA - LU ||_2 = " << cblas_dnrm2(m*n, OA, 1) << endl;

	delete [] OA;
	delete [] U;
	#endif
	////////// Debug mode //////////

	delete [] A;
	delete [] piv;

	return EXIT_SUCCESS;
}

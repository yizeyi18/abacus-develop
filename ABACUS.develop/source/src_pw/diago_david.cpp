#include "diago_david.h"
#include "diago_cg.h"
#include "../src_pw/global.h"

Diago_David::Diago_David()
{
    test_david = 2;
    // 1: check which function is called and which step is executed
    // 2: check the eigenvalues of the result of each iteration
    // 3: check the eigenvalues and errors of the last result
    // default: no check
}

Diago_David::~Diago_David() 
{
}


void Diago_David::diag
(
    ComplexMatrix &psi,
    double* en,
    const int& npw,
    const int& nband,
    const double* precondition,
    const int order,
    const double& eps,
    const int& maxiter,
    int& notconv,
    double& avg_iter
)
{
    if (test_david==1) TITLE("Diago_David","diag");
    timer::tick("Diago_David", "diag");

    assert( order > 1 );
    assert( order*nband < npw );

    int nbase_x = order * nband ;				// maximum dimension of the reduced basis set

    ComplexMatrix basis( nbase_x, npw );		// the reduced basis set
    ComplexMatrix hp( nbase_x, npw );			// the product of H and psi in the reduced basis set
    ComplexMatrix sp( nbase_x, npw );			// the Product of S and psi in the reduced basis set

    ComplexMatrix hc( nbase_x, nbase_x );		// Hamiltonian on the reduced basis
    ComplexMatrix sc( nbase_x, nbase_x );		// Overlap on the reduced basis
    ComplexMatrix vc( nbase_x, nbase_x );		// Eigenvectors of hc
    double* e = new double[nbase_x];			// the lowest N eigenvalues of hc
    assert(e != 0) ;

    complex<double>* psi_m = new complex<double>[npw] ;
    assert(psi_m != 0) ;
    complex<double>* hpsi = new complex<double>[npw] ;
    assert(hpsi != 0) ;
    complex<double>* spsi = new complex<double>[npw] ;
    assert(spsi != 0) ;
    complex<double>* ppsi = new complex<double>[npw] ;
    assert(ppsi != 0) ;
    complex<double>* respsi = new complex<double>[npw] ;
    assert(respsi != 0) ;

    bool* convflag = new bool[nband] ;	// convflag[m] = true if the m th band is convergent
    assert(convflag != 0) ;
    int* unconv = new int[nband] ;		// unconv[m] store the number of the m th unconvergent band
    assert(unconv != 0) ;

    int nbase = 0;						// the dimension of the reduced basis set
    notconv = nband;					// the number of the unconvergent bands
    ZEROS( convflag, nband );
    for ( int m = 0 ; m < nband; m++ ) unconv[m] = m;

    timer::tick("Diago_David","first");
    // orthogonalise the initial trial psi(0~nband-1)
    for (int m = 0; m < nband; m++)
    {
        // psi_m = psi(m)
        for ( int ig = 0; ig < npw; ig++ ) psi_m[ig] = psi(m,ig);

        this->SchmitOrth(npw, nband, m, basis, psi_m, spsi);

        hm.h_1psi(npw, psi_m, hpsi, spsi);

        // basis(m) = psi_m, hp(m) = H |psi_m>, sp(m) = S |psi_m>
        for ( int ig = 0; ig < npw; ig++ )
        {
            basis(m,ig) = psi_m[ig];
            hp(m,ig) = hpsi[ig];
            sp(m,ig) = spsi[ig];
        }
    }

    hc.zero_out();
    sc.zero_out();

    this->cal_elem( npw, nbase, notconv, basis, hp, sp, hc, sc );

    this->diag_zhegvx( nbase, nband, hc, sc, nbase_x, e, vc );

    for ( int m = 0; m < nband; m++ ) en[m] = e[m];

    timer::tick("Diago_David","first");

    int dav_iter = 0;
    do
    {
        dav_iter++;

        this->cal_grad( npw, nbase, notconv, basis, hp, sp, vc, unconv, precondition, e, hpsi, spsi, ppsi, respsi );

        this->cal_elem( npw, nbase, notconv, basis, hp, sp, hc, sc );

        this->diag_zhegvx( nbase, nband, hc, sc, nbase_x, e, vc );

        // check convergence and update eigenvalues
        timer::tick("Diago_David","check_update");

        notconv = 0;
        for ( int m = 0 ; m < nband; m++ )
        {
            convflag[m] = ( abs( e[m] - en[m] ) < eps );

            if ( !convflag[m] ) {
                unconv[notconv] = m;
                notconv++;
            }

            en[m] = e[m];
        }

        timer::tick("Diago_David","check_update");
        /*
        		// test_david==2 cout info of each iteration
        		if( test_david==2 )
        		{
        			cout << "iter = " << dav_iter << " notconv = " << notconv << endl;
        			out.printr1_d( "energy", en, nband );
        		}
        */
        if ( !notconv || ( nbase + notconv > nbase_x) || (dav_iter == maxiter) )
        {
            timer::tick("Diago_David","last");

            // updata eigenvectors of Hamiltonian
            psi.zero_out();
            for ( int m = 0; m < nband; m++ )
            {
                for ( int j = 0; j < nbase; j++ )
                {
                    for ( int ig = 0; ig < npw; ig++ ) psi(m,ig) += vc(j,m) * basis(j,ig);
                }
            }

            if ( !notconv || (dav_iter == maxiter) )
            {
                // overall convergence or last iteration: exit the iteration

                timer::tick("Diago_David","last");
                break;
            }
            else
            {
                // if the dimension of the reduced basis set is becoming too large,
                // then replace the first N (=nband) basis vectors with the current
                // estimate of the eigenvectors and set the basis dimension to N;

                this->refresh( npw, nband, nbase, en, psi, basis, hp, sp, hc, sc, vc );
                timer::tick("Diago_David","last");
            }

        }// end of if

    } while (1);

    avg_iter = static_cast<double>(dav_iter);
    /*
    	// if( test_david==3 ) cout info of davidson diag
    	if( test_david==3 )
    	{
    		cout << "hamilt davidson diag output " << endl ;
    		cout << "dav_iter = " << dav_iter <<" notconv = " << notconv << endl;
    		this->cal_err( npw, nband, nbase, vc, hp, basis, en, respsi );
        	cout << "hamilt davidson diag output  end " << endl ;
    	}
    */
    delete[] e;
    delete[] psi_m;
    delete[] hpsi;
    delete[] spsi;
    delete[] ppsi;
    delete[] respsi;
    delete[] convflag;
    delete[] unconv;

    timer::tick("Diago_David", "diag");
    return;
}

void Diago_David::cal_grad
(
    const int& npw,
    const int& nbase,	// current dimension of the reduced basis
    const int& notconv,
    ComplexMatrix &basis,
    ComplexMatrix &hp,
    ComplexMatrix &sp,
    const ComplexMatrix &vc,
    const int* unconv,
    const double* precondition,
    const double* e,
    complex<double>* hpsi,
    complex<double>* spsi,
    complex<double>* ppsi,
    complex<double>* respsi
)
{
    if ( test_david ==1 ) TITLE("DIAGO_DAVID","cal_grad");
    timer::tick("Diago_David","cal_grad");

    // expand the reduced basis set with the new basis vectors P|R(psi)>...
    // in which psi are the last eigenvectors
    // we define |R(psi)> as (H-ES)*|Psi>, E = <psi|H|psi>/<psi|S|psi>
    for ( int m = 0; m < notconv; m++ )
    {
        ZEROS( respsi, npw );
        for ( int i = 0; i < nbase; i++ )
        {
            for ( int ig = 0; ig < npw; ig++ )
            {
                respsi[ig] += vc( i, unconv[m] ) * ( hp(i,ig) - e[ unconv[m] ] * sp(i,ig) ) ;
            }
        }

        for ( int ig = 0; ig < npw; ig++ ) ppsi[ig] = respsi[ig] / precondition[ig] ;
/*
		double ppsi_norm = Diago_CG::ddot_real( npw, ppsi, ppsi);
		double rpsi_norm = Diago_CG::ddot_real( npw, respsi, respsi);
		assert( rpsi_norm > 0.0 );
		assert( ppsi_norm > 0.0 );
*/
        this->SchmitOrth(npw, nbase+notconv, nbase+m, basis, ppsi, spsi);

        hm.h_1psi(npw, ppsi, hpsi, spsi);

        for ( int ig = 0; ig < npw; ig++ )
        {
            basis(nbase+m,ig) = ppsi[ig];
            hp(nbase+m,ig) = hpsi[ig];
            sp(nbase+m,ig) = spsi[ig];
        }
    }

    timer::tick("Diago_David","cal_grad");
    return;
}

void Diago_David::cal_elem
(
    const int& npw,
    int& nbase,			// current dimension of the reduced basis
    const int& notconv,	// number of newly added basis vectors
    const ComplexMatrix &basis,
    const ComplexMatrix &hp,
    const ComplexMatrix &sp,
    ComplexMatrix &hc,
    ComplexMatrix &sc
)
{
    if ( test_david ==1 ) TITLE("DIAGO_DAVID","cal_elem");
    timer::tick("Diago_David","cal_elem");

    // updat the reduced Hamiltonian
    int offset_h = nbase * hc.nr ;
    int offset_s = nbase * sc.nr ;
//	ZEROS( hc.c+offset_h, notconv*hc.nr );
//	ZEROS( sc.c+offset_s, notconv*sc.nr );

    for ( int i = nbase; i < nbase+notconv; i++ )
    {
        for ( int j = 0; j <= i; j++ )
        {
            for ( int ig = 0; ig < npw; ig++ )
            {
                hc(i,j) += conj( basis(i,ig) ) * hp(j,ig);
                sc(i,j) += conj( basis(i,ig) ) * sp(j,ig);
            }
            //	hc(j,i) = Diago_CG::ddot( npw, basis, j, hp, i );
            //	sc(j,i) = Diago_CG::ddot( npw, basis, j, sp, i );
        }
    }

    Parallel_Reduce::reduce_complex_double_pool( hc.c+offset_h, notconv*hc.nr );
    Parallel_Reduce::reduce_complex_double_pool( sc.c+offset_s, notconv*sc.nr );
    /*
    	for( int i = nbase; i < nbase+notconv; i++ )
    	{
    		for( int j = 0; j <i; j++ )
    		{
    			hc(j,i) = conj( hc(i,j) );
    			sc(j,i) = conj( sc(i,j) );
    		}
    	}
    */
    nbase += notconv;
    timer::tick("Diago_David","cal_elem");
    return;
}

//==============================================================================
// optimize diag_zhegvx().

// 09-05-09 wangjp
// fixed a bug in diag_zhegvx().
// modify the dimension of h and s as (n,n) and copy the leading N*N
// part of hc & sc into h & s

// 09-05-10 wangjp
// As the complexmatrixs will be copied again in the subroutine ZHEGVX(...  ),
// i.e ZHEGVX(...) will not destroy the input complexmatrixs,
// we needn't creat another two complexmatrixs in diag_zhegvx().
//==============================================================================
void Diago_David::diag_zhegvx
(
    const int& n,
    const int& m,
    const ComplexMatrix &hc,
    const ComplexMatrix &sc,
    const int& ldh,
    double* e,
    ComplexMatrix &vc
)
{
//	TITLE("DIAGO_DAVID","diag_zhegvx");
    timer::tick("Diago_David","diag_zhegvx");
    assert( ldh >= max(1,n) );
    int lwork ;
    int info = 0;
    int mm = m;
    string name1 = "ZHETRD";
    string name2 = "L";

    int nb = LapackConnector::ilaenv(1, name1.c_str(), name2.c_str(), n, -1, -1, -1);
    if (nb < 1)
    {
        nb = max(1, n);
    }
    if (nb == 1 || nb >= n)
    {
        lwork = 2 * n - 1;
    }
    else
    {
        lwork = (nb + 1) * n;
    }
    complex<double> *work = new complex<double>[2*lwork]();
    assert(work != 0);
    double *rwork = new double[7*n];
    assert(rwork != 0);
    int *iwork = new int [5*n];
    assert(iwork != 0);
    int *ifail = new int[n];
    assert(ifail != 0);

    ZEROS(work,2*lwork);
    ZEROS(rwork,7*n);
    ZEROS(iwork,5*n);
    ZEROS(ifail,n);

	//WARNING_QUIT("divid","open zhegvx!");
	
	LapackConnector::zhegvx(1, 'V', 'I', 'L', n, hc, n, sc, n,
           0.0, 0.0, 1, m, 0.0, mm, e, vc, n,
           work, lwork, rwork, iwork, ifail, info);
/*
		complex<double> vc_norm = 0.0;
		for(int ib =0; ib < n; ib++)
		{
			vc_norm += conj(vc(ib, 0)) * vc(ib, 0);
		}
		assert( vc_norm.real() > 0.0);
*/
    delete[] work;
    delete[] rwork;
    delete[] iwork;
    delete[] ifail;
    timer::tick("Diago_David","diag_zhegvx");
    return;
}

void Diago_David::refresh
(
    const int& npw,
    const int& nband,
    int& nbase,
    const double* en,
    const ComplexMatrix &psi,
    ComplexMatrix &basis,
    ComplexMatrix &hp,
    ComplexMatrix &sp,
    ComplexMatrix &hc,
    ComplexMatrix &sc,
    ComplexMatrix &vc
)
{
    if ( test_david==1 ) TITLE("Diago_David","refresh");
    timer::tick("Diago_David","refresh");

    // update hp,sp
    basis.zero_out();
    for ( int m = 0; m < nband; m++ )
    {
        for ( int j = 0; j < nbase; j++ )
        {
            for ( int ig = 0; ig < npw; ig++ )
            {
                basis(m, ig) += vc(j,m) * hp(j,ig);
                basis(m+nband, ig) +=vc(j,m) * sp(j,ig);
            }
        }
    }

    for ( int m = 0; m < nband; m++ )
    {
        for (int ig = 0; ig < npw; ig++)
        {
            hp(m,ig) = basis(m, ig);
            sp(m,ig) = basis(m+nband, ig);
        }
    }

    // update basis
    basis.zero_out();
    for ( int m = 0; m < nband; m++ )
    {
        for ( int ig = 0; ig < npw; ig++ ) basis(m,ig) = psi(m,ig);
    }

    // updata the reduced Hamiltonian
    nbase = nband;
    hc.zero_out();
    sc.zero_out();
    for ( int i = 0; i < nbase; i++ )
    {
        hc(i,i) = en[i];
        sc(i,i) = ONE;
        vc(i,i) = ONE;
    }

    timer::tick("Diago_David","refresh");
    return;
}

void Diago_David::cal_err
(
    const int& npw,
    const int& nband,
    const int& nbase,
    const ComplexMatrix &vc,
    const ComplexMatrix &hp,
    const ComplexMatrix &basis,
    const double* en,
    complex<double>* respsi
)
{
    timer::tick("Diago_David","cal_err");
    double *err = new double[nband];
    assert(err != 0);

    for ( int m = 0; m < nband; m++ )
    {
        ZEROS( respsi, npw );
        for ( int j = 0; j < nbase; j++ )
        {
            for ( int ig=0; ig<npw; ig++ )
            {
                respsi[ig] +=  vc(j,m)*( hp(j,ig) - en[m] * basis(j,ig) );
            }
        }

        err[m] = Diago_CG::ddot_real( npw, respsi, respsi );
        err[m] = sqrt( err[m] );
    }

    cout << "i       eigenvalues       err (||*||)   " << endl ;

    for (int i = 0; i < nband ; i++)
    {
        cout << i << "\t\t"  << en[i]  << "\t\t" << err[i] << endl ;
    }

    delete[] err;
    timer::tick("Diago_David","cal_err");
    return;
}

void Diago_David::SchmitOrth
(
    const int& npw,
    const int n_band,
    const int m,
    const ComplexMatrix &psi,
    complex<double>* psi_m,
    complex<double>* spsi
)
{
//	if(test_david == 1) TITLE("Diago_David","SchmitOrth");
    timer::tick("Diago_David","SchmitOrth");

    // orthogonalize starting eigenfunction to those already calculated
    // psi_m orthogonalize to psi(0) ~ psi(m-1)
    // Attention, the orthogonalize here read as
    // psi(m) -> psi(m) - \sum_{i < m} \langle psi(i)|S|psi(m) \rangle psi(i)
    // so the orthogonalize is performed about S.

    assert(psi.nr >= n_band);
    assert(m >= 0);
    assert(m < n_band);

    hm.s_1psi(npw, psi_m, spsi);

    complex<double>* lagrange = new complex<double>[m+1];
    ZEROS( lagrange, m+1 );

    for (int j = 0; j < m; j++)
    {
        for (int ig = 0; ig < npw; ig++)
        {
            lagrange[j] += conj( psi(j,ig) ) * spsi[ig] ;
        }
        //	lagrange[j] = Diago_CG::ddot( npw, psi, j, spsi );
    }
    for (int ig = 0; ig <npw; ig++) lagrange[m] += conj( psi_m[ig] ) * spsi[ig];
//	lagrange[m] = Diago_CG::ddot( npw, psi_m, spsi );

    Parallel_Reduce::reduce_complex_double_pool( lagrange, m+1 );

//	out.printr1_d("lagrange", lagrange, m+1 );

    double psi_norm = lagrange[m].real();
    assert(psi_norm > 0.0);
//	cout << "m = " << m << endl;

    for (int j = 0; j < m; j++)
    {
        for (int ig = 0;ig < npw; ig++)
        {
            psi_m[ig] -= lagrange[j] * psi(j, ig);
        }
        psi_norm -= ( conj(lagrange[j]) * lagrange[j] ).real();
    }

    assert(psi_norm > 0.0);

    psi_norm = sqrt(psi_norm);

    if (psi_norm < 1.0e-12 ) {
        cout << "Diago_David::SchmitOrth:aborted for psi_norm <1.0e-12" << endl;
        cout << "n_band = " << n_band << endl;
        cout << "m = " << m << endl;
        exit(0);
    }
    else {
        for (int i = 0; i < npw; i++)
        {
            psi_m[i] /= psi_norm;
        }
    }

    hm.s_1psi(npw, psi_m, spsi);

    delete[] lagrange;
    timer::tick("Diago_David","SchmitOrth");
    return;
}	//end of SmchmitOrth

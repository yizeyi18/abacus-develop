#include "tools.h"
#include "global.h"
#include "potential.h"
#include "xc_functional.h"
#include "xc_gga_pw.h"
#include "efield.h"
#include "math.h"
#include "potential_libxc.h"
// new
#include "H_Hartree_pw.h"
#include "H_XC_pw.h"

potential::potential()
{
    vltot = new double[1];
    vrs1 = new double[1];
    this->out_potential = 0;
}

potential::~potential()
{
    delete[] vltot;
    delete[] vrs1;
}

void potential::allocate(const int nrxx)
{
    TITLE("potential","allocate");
    assert(nrxx>0);

    delete[] this->vltot;
    this->vltot = new double[nrxx];
    Memory::record("potential","vltot",nrxx,"double");

    this->vr.create(NSPIN,nrxx);
    this->vrs.create(NSPIN,nrxx);
    Memory::record("potential","vr",NSPIN*nrxx,"double");
    Memory::record("potential","vrs",NSPIN*nrxx,"double");

    delete[] this->vrs1;
    this->vrs1 = new double[nrxx];
    Memory::record("potential","vrs1",nrxx,"double");

    this->vnew.create(NSPIN,nrxx);
    Memory::record("potential","vnew",NSPIN*nrxx,"double");

    return;
}

//----------------------------------------------------------
//  Initializes the self consistent potential 
//----------------------------------------------------------
void potential::init_pot(const int &istep)
{
    TITLE("potential","init_pot");
    timer::tick("potential","init_pot");

    assert(istep>=0);

    vrs.zero_out();

    // mohan fix bug 2011-07-08
    // the vltot should and must be zero here.
    ZEROS(this->vltot, pw.nrxx);

	//-------------------------------------------------------------------
	// put the local pseudopotential + electric field (if any) in vltot
	//-------------------------------------------------------------------
	this->set_local(this->vltot);

	// mohan fix bug 2011-07-07
	// set pseudopotentials.
	int nspin0=NSPIN;//zhengdy-soc, pauli matrix, just index 0 has vlocal term.

	if(NSPIN==4) nspin0=1;

	//-------------------------------------------------------------------
	// put the local pseudopotential + electric field (if any) in vltot
	// vrs saves the total potential.
	//-------------------------------------------------------------------
	for(int is=0; is<nspin0; ++is)
	{
		for(int ir=0; ir<pw.nrxx; ++ir)
		{
			this->vrs(is,ir) = this->vltot[ir];	
		}
	}

	// core correction potential.
	CHR.set_rho_core( pw.strucFac );

	//--------------------------------------------------------------------
	// the other local potentials need charge density, so here you need to 
	// decide how to obtain the charge density from ionic step 0.
	//--------------------------------------------------------------------
    if(istep==0)
    {
        OUT(ofs_running,"start_pot",start_pot);

        cout << " START POTENTIAL      : " << start_pot << endl;
        if (this->start_pot == "atomic")//mohan add 2007-10-17
        {
            start_from_atomic:
            CHR.atomic_rho(NSPIN, CHR.rho);
        }
        else if (this->start_pot == "file")
        {
            ofs_running << " try to start potential from file : ";
            for(int is=0; is<NSPIN; is++)
            {
                stringstream ssc;
                ssc << global_out_dir << "SPIN" << is + 1 << "_CHG";
                ofs_running << ssc.str() << endl;
                // mohan update 2012-02-10
                if(CHR.read_rho( is, ssc.str(), CHR.rho[is] )) 
                {
                    ofs_running << " Read in the charge density: " << ssc.str() << endl;
				}
				else if(is>0 && NSPIN==4)
				{
					// read only spin (up+down)
					if(PRENSPIN == 1)
					{
						ofs_running << " Didn't read in the charge density but autoset it for spin " <<is+1<< endl;
						for(int ir=0;ir<pw.nrxx;ir++)
						{
							CHR.rho[is][ir] = 0.0;
						}
					}
					// 
					else if(PRENSPIN == 2)
					{//read up and down , then rearrange them.
						if(is==1) 
						{
							WARNING_QUIT("potential::init_pot","Incomplete charge density file!");
						}
						else if(is==2) 
						{
							ofs_running << " Didn't read in the charge density but would rearrange it later. "<< endl;
						}
						else if(is==3)
						{
							ofs_running << " rearrange charge density " << endl;
							for(int ir=0;ir<pw.nrxx;ir++)
							{
								CHR.rho[3][ir] = CHR.rho[0][ir] - CHR.rho[1][ir];
								CHR.rho[0][ir] = CHR.rho[0][ir] + CHR.rho[1][ir];
								CHR.rho[1][ir] = 0.0;
								CHR.rho[2][ir] = 0.0;
							}
						}
					}
					else
					{
						WARNING_QUIT("potential::init_pot","Incomplete charge density file!");
					}
				}
				else
                {
                    ofs_running << " Start charge density from atomic charge density." << endl;
                    goto start_from_atomic;
                }
            }
        }
        else
        {
            WARNING_QUIT("potential::init_pot","start_pot is wrong!");
        }
		
		// Peize Lin add 2020.04.04
		if(restart.info_load.load_charge && !restart.info_load.load_charge_finish)
		{
			for(int is=0; is<NSPIN; ++is)
			{
				restart.load_disk("charge", is);
			}
			restart.info_load.load_charge_finish = true;
		}
    }
    else
    {
        // the extrapolation part moves to ions.cpp.
    }

	// renormalize the charge density
    CHR.renormalize_rho();

    this->v_of_rho(CHR.rho, vr);

    //----------------------------------------------------------
    // Define the total local potential (external+scf) in DFT
	// Define TDDFT potential, by Fuxiang He
    //----------------------------------------------------------
    if(vext == 0) 
	{
		this->set_vrs();
	}
    else 
	{
		this->set_vrs_tddft(istep);
	}

    //figure::picture(this->vrs1,pw.ncx,pw.ncy,pw.ncz);
    timer::tick("potential","init_pot");
    return;
}


//==========================================================
// This routine computes the local potential in real space
//==========================================================
void potential::set_local(double* vl_pseudo)const
{
    TITLE("potential","set_local");
    timer::tick("potential","set_local");

    complex<double> *vg = new complex<double>[pw.ngmc];

    ZEROS( vg, pw.ngmc );

    for (int it=0; it<ucell.ntype; it++)
    {
        for (int ig=0; ig<pw.ngmc; ig++)
        {
            vg[ig] += ppcell.vloc(it, pw.ig2ngg[ig]) * pw.strucFac(it,ig);
        }
    }

    UFFT.ToRealSpace(vg, vl_pseudo); 

    delete[] vg;

    if(EFIELD && !DIPOLE)
    {
        Efield EFID;
        // in fact, CHR.rho is not used here.
        // if charge correction due to Efield is considered,
        // the structure here need to be updated.

        static bool first = true;
        if(first)
        {
            cout << " ADD THE EFIELD (V/A) : " << Efield::eamp*51.44 << endl;
            first = false;
        }
        EFID.add_efield(CHR.rho[0], vl_pseudo);	
    }

    //ofs_running <<" set local pseudopotential done." << endl;
    timer::tick("potential","set_local");
    return;
}

//==========================================================
// This routine computes the Hartree and Exchange and Correlation
// potential and energies which corresponds to a given charge density
// The XC potential is computed in real space, while the
// Hartree potential is computed in reciprocal space.
//==========================================================
void potential::v_of_rho
(
    double **rho_in,
    matrix &v_in
)
{
    TITLE("potential","v_of_rho");
    v_in.zero_out();

    timer::tick("potential","v_of_rho",'E');

//----------------------------------------------------------
//  calculate the exchange-correlation potential
//----------------------------------------------------------
	
	#ifdef USE_LIBXC
    Potential_Libxc::v_xc(rho_in, H_XC_pw::etxc, H_XC_pw::vtxc, v_in);
	#else
    H_XC_pw::v_xc(pw.nrxx, pw.ncxyz, ucell.omega, rho_in, CHR.rho_core, v_in);
	#endif

//----------------------------------------------------------
//  calculate the Hartree potential
//----------------------------------------------------------
	H_Hartree_pw::v_hartree(ucell, pw, UFFT, NSPIN, v_in, rho_in);

    // mohan add 2011-06-20
    if(EFIELD && DIPOLE)
    {
        Efield EFID;
        for (int is = 0;is < NSPIN;is++)
        {
            EFID.add_efield(rho_in[is], &v_in.c[is*pw.nrxx]);
        }
    }
    timer::tick("potential","v_of_rho",'E');
    return;
} //end subroutine v_of_rho



//==========================================================
// set the total local potential vrs on the real space grid 
// used in h_psi, adding the (spin dependent) scf (H+xc)
// part and the sum of all the local pseudopotential
// contributions.
//==========================================================
void potential::set_vrs(void)
{
    TITLE("potential","set_vrs");
    timer::tick("potential","set_vrs");

    for (int is = 0;is < NSPIN;is++)
    {
        //=================================================================
        // define the total local potential (external + scf) for each spin
        //=================================================================
		if(NSPIN==4&&is>0)
		{
			for (int i = 0;i < pw.nrxx;i++)
			{
				this->vrs(is, i) = this->vr(is, i);
			}
		}
		else        
		{
			for (int i = 0;i < pw.nrxx;i++)
	        {
	            this->vrs(is, i) = this->vltot[i] + this->vr(is, i);
			}
		}
    }


    timer::tick("potential","set_vrs");
    return;
}


// ----------------------------------------------------------------------
void potential::newd(void)
{
    if (test_potential) TITLE("potential","newd");

    // distringuish non-local pseudopotential in REAL or RECIPROCAL space.
    // if in real space, call new_r
    // if in reciprocal space, call new_g

    // new g:
    //----------------------------------------------------------------------
    //  This routine computes the integral of the effective potential with
    //  the Q function and adds it to the bare ionic D term which is used
    //  to compute the non-local term in the US scheme.

	// no ultrasoft potentials: use bare coefficients for projectors
	// if( spin_orbital) ....
	// else if(noncolin) ....
	for (int iat=0; iat<ucell.nat; iat++)
	{
		const int it = ucell.iat2it[iat];
		const int nht = ucell.atoms[it].nh;
		// nht: number of beta functions per atom type
		for (int is = 0; is < NSPIN; is++)
		{
			for (int ih=0; ih<nht; ih++)
			{
				for (int jh=ih; jh<nht; jh++)
				{
					if(LSPINORB)
					{
						ppcell.deeq_nc(is , iat , ih , jh)= ppcell.dvan_so(is , it , ih , jh);
						ppcell.deeq_nc(is , iat , jh , ih)= ppcell.dvan_so(is , it , jh , ih);
					}
					else if( NSPIN==4 )
					{
						if(is==0)
						{
							ppcell.deeq_nc(is, iat, ih, jh) = ppcell.dvan(it, ih, jh);
							ppcell.deeq_nc(is, iat, jh, ih) = ppcell.dvan(it, ih, jh);
						}
						else if(is==1)
						{
							ppcell.deeq_nc(is, iat, ih, jh) = complex<double>(0.0 , 0.0);
							ppcell.deeq_nc(is, iat, jh, ih) = complex<double>(0.0 , 0.0);
						}
						else if(is==2)
						{
							ppcell.deeq_nc(is, iat, ih, jh) = complex<double>(0.0 , 0.0);
							ppcell.deeq_nc(is, iat, jh, ih) = complex<double>(0.0 , 0.0);
						}
						else if(is==3)
						{
							ppcell.deeq_nc(is, iat, ih, jh) = ppcell.dvan(it, ih, jh);
							ppcell.deeq_nc(is, iat, jh, ih) = ppcell.dvan(it, ih, jh);
						}
					}
					else{
						ppcell.deeq(is, iat, ih, jh) = ppcell.dvan(it, ih, jh);
						ppcell.deeq(is, iat, jh, ih) = ppcell.dvan(it, ih, jh);
					}
				}
			}
		}
	}
	return;
} // end subroutine newd


//==========================================================
// this function aims to add external time-dependent potential 
// (eg: linear potential) used in tddft
// fuxiang add in 2017-05
//==========================================================
void potential::set_vrs_tddft(const int istep)
{
    TITLE("potential","set_vrs_tddft");
    timer::tick("potential","set_vrs_tddft");

    for (int is = 0;is < NSPIN;is++)
    {
        //====================================================
        // add external linear potential, fuxiang add in 2017/05
        //====================================================

        const int timescale = 1;  // get the time that vext influences;
        if (istep >= timescale)
        {
            for (int i = 0;i < pw.nrxx;i++)
            {
                this->vrs(is, i) = this->vltot[i] + this->vr(is, i);
            }
            cout << "vext = 0! " << endl;
        }
        else
        {
            this->vextold = new double[pw.nrxx];
            this->vext = new double[pw.nrxx];
            const int yz = pw.ncy*pw.nczp;
            int index, i, j, k;

            for(int ir=0; ir<pw.nrxx; ++ir)
            {
                index = ir;
                i     = index / yz; // get the z, z is the fastest
                index = index - yz * i;// get (x,y)
                j     = index / pw.nczp;// get y
                k     = index - pw.nczp*j + pw.nczp_start;// get x

                if(vext_dire == 1)
                {
                    if (k<pw.ncx*0.05) this->vextold[ir] = (0.019447*k/pw.ncx-0.001069585)*ucell.lat0;
                    else if (k>=pw.ncx*0.05 && k<pw.ncx*0.95) this->vextold[ir] = -0.0019447*k/pw.ncx*ucell.lat0;
                    else if (k>=pw.ncx*0.95) this->vextold[ir] = (0.019447*(1.0*k/pw.ncx-1)-0.001069585)*ucell.lat0;
                }
                else if(vext_dire == 2)
                {
                    if (j<pw.ncx*0.05) this->vextold[ir] = (0.019447*j/pw.ncx-0.001069585)*ucell.lat0;
                    else if (j>=pw.ncx*0.05 && j<pw.ncx*0.95)	this->vextold[ir] = -0.0019447*j/pw.ncx*ucell.lat0;
                    else if (j>=pw.ncx*0.95) this->vextold[ir] = (0.019447*(1.0*j/pw.ncx-1)-0.001069585)*ucell.lat0;
                }
                else if(vext_dire == 3)
                {
                    if (i<pw.ncx*0.05) this->vextold[ir] = (0.019447*i/pw.ncx-0.001069585)*ucell.lat0;
                    else if (i>=pw.ncx*0.05 && i<pw.ncx*0.95) this->vextold[ir] = -0.0019447*i/pw.ncx*ucell.lat0;
                    else if (i>=pw.ncx*0.95) this->vextold[ir] = (0.019447*(1.0*i/pw.ncx-1)-0.001069585)*ucell.lat0;
                }

                // Gauss
/*
                const double w = 22.13;    // eV
                const double sigmasquare = 6836;
                const double timecenter = 700;
                const double timenow = (istep-timecenter)*INPUT.md_dt*41.34;
                this->vext[ir] = this->vextold[ir]*cos(w/27.2116*timenow)*exp(-timenow*timenow*0.5/(sigmasquare))*0.25;  //0.1 is modified in 2018/1/12
*/

                //HHG of H atom
/*
                if(istep < 1875)
                {
                    this->vext[ir] = this->vextold[ir]*2.74*istep/1875*cos(0.0588*istep*INPUT.md_dt*41.34);	// 2.75 is equal to E0;
                }
                else if(istep < 5625)
                {
                    this->vext[ir] = this->vextold[ir]*2.74*cos(0.0588*istep*INPUT.md_dt*41.34);
                }
                else if(istep < 7500)
                {
                    this->vext[ir] = this->vextold[ir]*2.74*(7500-istep)/1875*cos(0.0588*istep*INPUT.md_dt*41.34);
                }
*/

                //HHG of H2

                //const double timenow = (istep)*INPUT.md_dt*41.34;
                //this->vext[ir] = this->vextold[ir]*2.74*cos(0.856*timenow)*sin(0.0214*timenow)*sin(0.0214*timenow);
                //this->vext[ir] = this->vextold[ir]*2.74*cos(0.856*timenow)*sin(0.0214*timenow)*sin(0.0214*timenow)*0.01944;
                //this->vext[ir] = this->vextold[ir]*2.74*cos(0.0428*timenow)*sin(0.00107*timenow)*sin(0.00107*timenow);

                this->vrs(is,ir) = this->vltot[ir] + this->vr(is, ir) + this->vext[ir];

                //cout << "x: " << k <<"	" << "y: " << j <<"	"<< "z: "<< i <<"	"<< "ir: " << ir << endl;
                //cout << "vext: " << this->vext[ir] << endl;
                //cout << "vrs: " << vrs(is,ir) <<endl;
            }
            cout << "vext is existed!" << endl;

            delete[] this->vextold;
            delete[] this->vext;
        }
    }


    timer::tick("potential","set_vrs_tddft");
    return;
} //end subroutine set_vrs_tddft

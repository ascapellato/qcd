/* qcd_smearing.c
 *
 * gauss-smearing of fermion fields
 *
 * Tomasz Korzec 2008
 * stout smearing
 **********************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <qcd.h>
 

 

/* perform 1 iteration of gaussian smearing 
 * on time-slice t with
 * parameter alpha.
 *
 * v-> (v+alpha*Hv) / (1+6*alpha)
 */
int qcd_gaussIteration3d(qcd_vector *v, qcd_gaugeField *u, qcd_real_8 alpha, qcd_uint_4 t)
{
  qcd_real_8 nrm = 1.0/(1.0 + alpha*6.0);
  if(!v->initialized)
    {
      fprintf(stderr,"Error in qcd_gaussIteration3d! Vector mus be properly initialized\n");
      return(1);
    }

  qcd_vector v2;
  if(qcd_initVector(&v2, v->geo))
    {
      fprintf(stderr,"process %i: Error in qcd_gaussIteration3d! Could not initialize vector", v->geo->myid);
      return(1);
    }   

  //start communication (4d comm is in principle too much, but hey...
  qcd_communicateVectorPMGaugeP(v,u);

  //smear inner points:   
  qcd_zeroVector(&v2);

  if(v->geo->lL[1]>2 && v->geo->lL[2]>2 && v->geo->lL[3]>2 
     && t>=v->geo->Pos[0]*v->geo->lL[0] && t<(v->geo->Pos[0]+1)*v->geo->lL[0])
    { 
      int tt=t-v->geo->Pos[0]*v->geo->lL[0];
      for(int z=1;z<v->geo->lL[3]-1;z++)
	for(int y=1;y<v->geo->lL[2]-1;y++)
	  for(int x=1;x<v->geo->lL[1]-1;x++)
	    {
	      int i=qcd_LEXIC(tt,x,y,z,v->geo->lL);
	      for(int nu=1;nu<4;nu++)
		{
		  qcd_real_8 upsi[24];
		  qcd_real_8 udaggerpsi[24];

		  qcd_real_8 *uu  = (qcd_real_8*) &(u->D[i][nu][0][0].re);
		  qcd_real_8 *psi = (qcd_real_8*) &(v->D[v->geo->plus[i][nu]][0][0].re);
		  qcd_APPLY_U(uu,upsi,psi);

		  uu  = (qcd_real_8*) &(u->D[v->geo->minus[i][nu]][nu][0][0].re);
		  psi = (qcd_real_8*) &(v->D[v->geo->minus[i][nu]][0][0].re); 
		  qcd_APPLY_U_DAGGER(uu,udaggerpsi,psi);
	    
		  qcd_real_8 *total = (qcd_real_8*) &(v2.D[i][0][0].re);
		  qcd_SUM_UP_HOPP(total,upsi,udaggerpsi);
		}
	    }//end inner-point loop
    }//end inner-points condition

  qcd_waitall(v->geo);

  //now boundary points
  if(t>=v->geo->Pos[0]*v->geo->lL[0] && t<(v->geo->Pos[0]+1)*v->geo->lL[0])
    for(int j=0; j<v->geo->edge0Points; j++)
      {
	int tt=t-v->geo->Pos[0]*v->geo->lL[0];
	int i=v->geo->edge0[j]*v->geo->lL[0]+tt; // works only with present lexic
	for(int nu=1;nu<4;nu++)
	  {
	    qcd_real_8 upsi[24];
	    qcd_real_8 udaggerpsi[24];

	    qcd_real_8 *uu  = (qcd_real_8*) &(u->D[i][nu][0][0].re);
	    qcd_real_8 *psi = (qcd_real_8*) &(v->D[v->geo->plus[i][nu]][0][0].re);
	    qcd_APPLY_U(uu,upsi,psi);
	 
	    uu  = (qcd_real_8*) &(u->D[v->geo->minus[i][nu]][nu][0][0].re);
	    psi = (qcd_real_8*) &(v->D[v->geo->minus[i][nu]][0][0].re); 
	    qcd_APPLY_U_DAGGER(uu,udaggerpsi,psi);
	 
	    qcd_real_8 *total = (qcd_real_8*) &(v2.D[i][0][0].re);
	    qcd_SUM_UP_HOPP(total,upsi,udaggerpsi);
	  }           
      }//end boundaries loop                  
   
  qcd_scaleVector3d(&v2,alpha,t);
  qcd_addVector(v,v,&v2);
  qcd_scaleVector3d(v,nrm,t); 

  qcd_destroyVector(&v2);
  return 0;
} 


/* perform 1 iteration of gaussian smearing 
 * on all time-slices with
 * parameter alpha.
 *
 * v-> (v+alpha*Hv) / (1+6*alpha)
 */
int qcd_gaussIteration3dAll(qcd_vector *v, qcd_gaugeField *u, qcd_real_8 alpha, qcd_uint_2 gaugeCom)
{
   qcd_real_8 nrm = 1.0/(1.0 + alpha*6.0);

   if(!v->initialized)
     {
       fprintf(stderr,"Error in qcd_gaussIteration3dAll! Vector mus be properly initialized\n");
       return(1);
     }

   qcd_vector v2;
   if(qcd_initVector(&v2, v->geo))
     {
       fprintf(stderr,"process %i: Error in qcd_gaussIteration3dAll! Could not initialize vector", v->geo->myid);
       return(1);
     }   
   
   if(gaugeCom)
     qcd_communicateVectorPMGaugeP(v,u);
   else
     qcd_communicateVectorPM(v);
   
   //smear inner points:   
   qcd_zeroVector(&v2);

   if(v->geo->lL[1]>2 && v->geo->lL[2]>2 && v->geo->lL[3]>2)
     {
       for(int z=1;z<v->geo->lL[3]-1;z++)
	 for(int y=1;y<v->geo->lL[2]-1;y++)
	   for(int x=1;x<v->geo->lL[1]-1;x++) {
#pragma omp parallel for
	     for(int tt=0;tt<v->geo->lL[0];tt++)
	       {
		 int i=qcd_LEXIC(tt,x,y,z,v->geo->lL);
		 for(int nu=1;nu<4;nu++)
		   {
		     qcd_real_8 upsi[24];
		     qcd_real_8 udaggerpsi[24];
		     qcd_real_8 *uu  = (qcd_real_8*) &(u->D[i][nu][0][0].re);
		     qcd_real_8 *psi = (qcd_real_8*) &(v->D[v->geo->plus[i][nu]][0][0].re);
		     qcd_APPLY_U(uu,upsi,psi);
		     
		     uu  = (qcd_real_8*) &(u->D[v->geo->minus[i][nu]][nu][0][0].re);
		     psi = (qcd_real_8*) &(v->D[v->geo->minus[i][nu]][0][0].re); 
		     qcd_APPLY_U_DAGGER(uu,udaggerpsi,psi);
		     
		     qcd_real_8 *total = (qcd_real_8*) &(v2.D[i][0][0].re);
		     qcd_SUM_UP_HOPP(total,upsi,udaggerpsi);
		   }
	       }//end inner-point loop
	   }
     }//end inner-points condition
   qcd_waitall(v->geo);
   
   //now boundary points
   for(int j=0; j<v->geo->edge0Points; j++)
#pragma omp parallel for
     for(int tt=0; tt<v->geo->lL[0]; tt++)
       {
	 int i=v->geo->edge0[j]*v->geo->lL[0]+tt; // works only with present lexic
	 for(int nu=1;nu<4;nu++)
	   {
	     qcd_real_8 upsi[24];
	     qcd_real_8 udaggerpsi[24];
	     qcd_real_8 *uu  = (qcd_real_8*) &(u->D[i][nu][0][0].re);
	     qcd_real_8 *psi = (qcd_real_8*) &(v->D[v->geo->plus[i][nu]][0][0].re);
	     qcd_APPLY_U(uu,upsi,psi);
	     
	     uu  = (qcd_real_8*) &(u->D[v->geo->minus[i][nu]][nu][0][0].re);
	     psi = (qcd_real_8*) &(v->D[v->geo->minus[i][nu]][0][0].re); 
	     qcd_APPLY_U_DAGGER(uu,udaggerpsi,psi);
	     
	     qcd_real_8 *total = (qcd_real_8*) &(v2.D[i][0][0].re);
	     qcd_SUM_UP_HOPP(total,upsi,udaggerpsi);
	   }           
       }//end boundaries loop                  

   qcd_scaleVector(&v2,alpha);
   qcd_addVector(v,v,&v2);
   qcd_scaleVector(v,nrm); 

   qcd_destroyVector(&v2);
   return 0;
}//end qcd_gaussIteration3dAll




/* perform 1 iteration of 3d APE-smearing 
 * with parameter alpha.
 *
 * u -> SU3-projection( u + alpha * sum spatial staples)
 */
int qcd_apeSmear3d(qcd_gaugeField *apeu, qcd_gaugeField *u, qcd_real_8 alpha)
{
   qcd_propagator edge;
   qcd_complex_16 stapleForward[3][3];
   qcd_complex_16 stapleBackward[3][3];
   qcd_uint_2 mu,nu,c1,c2;
   qcd_uint_4 l;
   qcd_complex_16 tmp[3][3];

   qcd_initPropagator(&edge,u->geo); // store edges in a propagator-structure. 

   /* since staples need next-to-nearest neighbors like U(x+mu-nu), this is done in 2 steps
      a) communicate U & calculate edges U_mu(x)U_nu(x+mu)
      b) communicate edges and put them together to staples.
   */
   
   qcd_communicateGaugePM(u);
   qcd_zeroGaugeField(apeu);   
   qcd_waitall(u->geo);

   for(mu=1; mu<4; mu++)
   for(nu=1;nu<4; nu++)
   if(mu!=nu)   
   for(l=0;l<u->geo->lV;l++)
   {
      qcd_MUL3x3(edge.D[l][mu][nu], u->D[l][mu], u->D[u->geo->plus[l][mu]][nu]);
   }
   
   qcd_communicatePropagatorP(&edge);

   //the forward staple doesn't need the edges
   for(mu=1; mu<4; mu++)
   for(nu=1;nu<4; nu++)
   if(mu!=nu)
   for(l=0;l<u->geo->lV;l++)
   {
      qcd_MUL3x3(tmp, u->D[l][nu],u->D[u->geo->plus[l][nu]][mu]);
      qcd_MULADJOINT3x3(stapleForward, tmp, u->D[u->geo->plus[l][mu]][nu]);
      for(c1=0;c1<3;c1++)
      for(c2=0;c2<3;c2++)
         apeu->D[l][mu][c1][c2] = qcd_CADD(apeu->D[l][mu][c1][c2],stapleForward[c1][c2]);
   }   
   
   qcd_waitall(u->geo);
   
   for(mu=1; mu<4; mu++)
   for(nu=1;nu<4; nu++)
   if(mu!=nu)
   for(l=0;l<u->geo->lV;l++)
   {
      qcd_ADJOINTMUL3x3(stapleBackward, u->D[u->geo->minus[l][nu]][nu], edge.D[u->geo->minus[l][nu]][mu][nu]);
      for(c1=0;c1<3;c1++)
      for(c2=0;c2<3;c2++)
         apeu->D[l][mu][c1][c2] = qcd_CADD(apeu->D[l][mu][c1][c2],stapleBackward[c1][c2]);
   }
   
   
   qcd_scaleGaugeField(apeu,alpha);
   qcd_addGaugeField(apeu,u,apeu);
   
   qcd_projectSU33d(apeu);
   
   qcd_destroyPropagator(&edge);
   return(0);
}//end qcd_apeSmear3d

/* Routines for Stout-smearing */

/* ausiliar function for the exp */
double xi0(qcd_real_8 w)
{
  qcd_real_8 tmp,tmp1;
  if(fabs(w)<0.05)
    {
      tmp=1.0-w*w/42.0;
      tmp1=1.0-w*w/20.0*tmp;
      tmp=1.0-w*w/6.0*tmp1;
      return tmp;
    }
  else
    return sin(w)/w;
}
/* exponential */
void Exp(qcd_complex_16 M[][3])
{
  qcd_complex_16 c0,c1;
  qcd_real_8 c0max,u,w,theta,Xi0;
  qcd_complex_16 f0,f1,f2;
  qcd_complex_16 h0,h1,h2;
  
  qcd_complex_16 M2[3][3],one[3][3],sum[3][3],sum1[3][3];

  qcd_complex_16 C,C1;
  qcd_complex_16 tmp,tmp1;
  qcd_real_8 factor;
  int sign;

  c0=qcd_CADD(qcd_CADD( qcd_CMUL(M[0][0],qcd_CMUL(M[1][1],M[2][2])),
			qcd_CMUL(M[0][1],qcd_CMUL(M[1][2],M[2][0]))),
	      qcd_CMUL(M[0][2],qcd_CMUL(M[1][0],M[2][1]))); //det of M (M is equivalent to Q)

  if(c0.re<0){
    sign=1;
    c0.re=- c0.re;
  }

  qcd_mul3x3(M2,M,M); //M2=M*M

  c1=qcd_trace3x3(M2); 
  c0max=2.0*pow(c1.re/(2*3.0),1.5);
  theta=acos(c0.re/c0max);
  u=sqrt(c1.re/3.0)*cos(theta/3.0);
  w=sqrt(c1.re)*sin(theta/3.0);

  Xi0=xi0(w);  //call the function xi0

  C=(qcd_complex_16){cos(2*u),sin(2*u)}; //exp(2iu)
  C1=(qcd_complex_16){cos(u),-sin(u)};   //exp(-iu)
  tmp=(qcd_complex_16){u*u-w*w,0.0};
  tmp1=(qcd_complex_16){8*u*u*cos(w),2*u*(3*u*u+w*w)*Xi0};
  h0=qcd_CADD(qcd_CMUL(tmp,C),qcd_CMUL(C1,tmp1)); //final h0,eq.(30)
  
  tmp=(qcd_complex_16){2*u,0.0};
  tmp1=(qcd_complex_16){2*u*cos(w),-(3*u*u-w*w)*Xi0};
  h1=qcd_CSUB(qcd_CMUL(tmp,C),qcd_CMUL(C1,tmp1)); //final h1, eq.(31)

  tmp=(qcd_complex_16){cos(w),3*u*Xi0};
  h2=qcd_CSUB(C,qcd_CMUL(C1,tmp)); //final h2, eq.(32)

  factor=9*u*u-w*w;  //eq.(29)

  if(sign==0) //i.e. c0.re>0
    {
      f0=qcd_CSCALE(h0,1.0/factor); //f0=h0/temp, Eq.(29)
      f1=qcd_CSCALE(h1,1.0/factor); //f1=h1/temp
      f2=qcd_CSCALE(h2,1.0/factor); //f2=h2/temp
    }
  else
    {
      f0=qcd_CONJ(qcd_CSCALE(h0,1.0/factor)); // Eq.(34)
      f1=qcd_CONJ(qcd_CSCALE(h1,-1.0/factor));
      f2=qcd_CONJ(qcd_CSCALE(h2,1.0/factor));
    }
  qcd_unit3x3(one); //I
  qcd_cScale3x3(one,f0); //f0*I and store in one
  qcd_cScale3x3(M,f1);   //f1*M and store in M (M is Q)
  qcd_cScale3x3(M2,f2);  //f2*M^2
  
  qcd_add3x3(sum,one,M); //sum=one+M
  qcd_add3x3(sum1,sum,M2); //sum1=sum+M2=f0*I+f1*Q+f2*Q^2 -> exp(iQ), eq.(19)

  return;
}
//Exp

void qcd_stoutsmearing(qcd_gaugeField *u_out, qcd_gaugeField *u, qcd_real_8 rho)
{
  qcd_uint_2 mu, nu, c1, c2;
  qcd_uint_4 l;
  qcd_complex_16 u0[3][3], u1[3][3], stapl[3][3];
  qcd_complex_16 plaq[3][3];
  qcd_complex_16 trace,factor_rho,factor;

  qcd_propagator edge; //for staple backward
  
  qcd_initPropagator(&edge,u->geo); // store edges in a propagator-structure.
  qcd_communicateGaugePM(u);
  qcd_waitall(u->geo);

  factor_rho=(qcd_complex_16) {rho/2.0,0.0};
  for(mu=1;mu<4;mu++) {
    for(nu=1;nu<4;nu++)
      if(mu!=nu)
	for(l=0; l<u->geo->lV; l++)
	  {
	    qcd_MUL3x3(edge.D[l][mu][nu], u->D[l][mu], u->D[u->geo->plus[l][mu]][nu]);
	  } //end l
  }//end mu
  qcd_communicatePropagatorP(&edge);
  qcd_waitall(u->geo);

  for(mu=1;mu<4;mu++) {
    for(l=0; l<u->geo->lV; l++)
      {
	for(c1=0;c1<3;c1++)
	  for(c2=0;c2<3;c2++)
	    stapl[c1][c2]=(qcd_complex_16) {0.0,0.0}; //initialize to zero
	for(nu=1;nu<4;nu++)
	  if(mu!=nu)
	    {
	      qcd_ADJOINTMUL3x3(u1, u->D[u->geo->minus[l][nu]][nu], edge.D[u->geo->minus[l][nu]][mu][nu]); //u1=(u^+)*edge

	      for(c1=0;c1<3;c1++)
		for(c2=0;c2<3;c2++)
		  stapl[c1][c2]=qcd_CADD(stapl[c1][c2],u1[c1][c2]); //sum all u1 to form the STAPLE

	      qcd_MUL3x3(u0,u->D[l][nu],u->D[u->geo->plus[l][nu]][mu]);
	      qcd_MULADJOINT3x3(u1, u0, u->D[u->geo->plus[l][mu]][nu]); //u1=(u0)* (u^+)

	      for(c1=0;c1<3;c1++)
		for(c2=0;c2<3;c2++)
		  stapl[c1][c2]=qcd_CADD(stapl[c1][c2],u1[c1][c2]); //sum all u1 to form the STAPLE
	    }
	/* so far we have the staple. Multiply with u^+ to form plaquette */
	qcd_MULADJOINT3x3(plaq, stapl, u->D[l][mu]); //plaq= staple * u^+

	/* Anti-hermitize */
	for(c1=0;c1<3;c1++)
	  for(c2=0;c2<3;c2++)
	    plaq[c1][c2]= qcd_CSUB(plaq[c1][c2],qcd_CONJ(plaq[c2][c1])); //plaq= plaq - plaq^+
	
	trace=qcd_trace3x3(plaq); //calculate the trace of plaq
	factor=(qcd_complex_16) {(1.0/3)*trace.re,(1.0/3)*trace.im}; //factor= 1/3*trace(plaq)
	
	for(c1=0;c1<3;c1++)
	  for(c2=0;c2<3;c2++)
	    plaq[c1][c2]= qcd_CSUB(plaq[c1][c2],factor); //plaq= plaq -1.3* trace(plaq) 
	
	//scale by \rho/2
	for(c1=0;c1<3;c1++)
	  for(c2=0;c2<3;c2++)
	    u_out->D[l][mu][c1][c2]=qcd_CMUL(factor_rho,plaq[c1][c2]); //Eq.(2), plaq*rho/2 = i*Q

	qcd_cScale3x3(u_out->D[l][mu],(qcd_complex_16) {0.0,-1.0}); // Q

      }//end l
  }//end mu ; in the end I define u_out->[l][mu], i.e. i*Q_mu(x)

  for(mu=1;mu<4;mu++)
    for(l=0; l<u->geo->lV; l++)
      {
	qcd_copy3x3(u0,u_out->D[l][mu]); //u0=u_out
	// exponentiate
	Exp(u0); //exp(iQ)
	qcd_MUL3x3(u_out->D[l][mu],u0,u->D[l][mu]);
      }
  return;

}//end of stout smearing

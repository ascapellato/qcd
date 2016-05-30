/* 
 * multigrid
 * creates smeared sources
 * create forward and sequential propagators
 * APE-smeared and momentum smearing  gauge fields
 * three point functions 
 *
 * Simone Bacchio 2016
 * Aurora Scapellato 2016
 ****************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <qcd.h>
#include <mg4qcd.h> 
#include "projectors_gamma.h"

#undef PION
#define NUCLEON
#undef DO_MOM_0_ONCE
#undef THREEP_SKIP_MOM_0
#define SKIP_MOM_ZERO


#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
  __typeof__ (b) _b = (b); \
  _a > _b ? _a : _b; })

int main(int argc,char* argv[])
{
   char*  params = NULL;
   char   gauge_name[qcd_MAX_STRING_LENGTH];
   char   param_name[qcd_MAX_STRING_LENGTH];

   qcd_int_4   x_src[4], lx_src[4], nsmear, nsmearAPE, mom[4], mom_min, mom_max, mom_loop, dir_loop, mom_step;
   qcd_real_8   alpha,alphaAPE,plaq;
   int   params_len;   

   qcd_int_4 t_start, t_stop, t, lt, x, y, z;
   qcd_int_4 ctr, ctr2;
   qcd_uint_4 dir;
   qcd_uint_4 lt_sink, t_sink, lx, ly, lz, s;
   qcd_real_8 zeta;
   qcd_real_8 theta[4] = {M_PI,0.0,0.0,0.0};    // antiperiodic b.c. in time

   qcd_uint_2 fl, pr, mu, nu, ku, lu, gu, ju, bu, cu, c1, c2, c3, cc1, cc2, c1p, c2p , c3p, nprojs, mom_0_done=0;
   qcd_geometry geo;
   qcd_gaugeField u, uAPE, uAPE2, u_ms;
   qcd_gaugeField *u_ptr, *uAPE_ptr, *utmp_ptr ;
   qcd_propagator prop[2], prop_pb[2];        
  
   qcd_vector vec, vec_mg;
   qcd_uint_2 P[4];
   qcd_uint_2 L[4];

   qcd_uint_8 i,j;         //lattice sites
   qcd_real_8 tmp;       //complex phase factor
   qcd_complex_16 tmp_c;

   /**** for threep ****/
   qcd_uint_2  *proj_type;
   qcd_gaugeField u_temp, u_shift_f, u_shift_b, u_prod_f, u_prod_b;
   qcd_propagator prop_temp, prop_shift_f[2], prop_shift_b[2];
   qcd_propagator prop_ms[2], (*prop_seq)[2] ;
   char (*prop_seq_name)[2][qcd_MAX_STRING_LENGTH], (*source_seq_name)[2][qcd_MAX_STRING_LENGTH];
   qcd_complex_16 *block_f, *block_b ;
   qcd_complex_16 (**threep_loc_f)[2], (**threep_loc_b)[2];
   qcd_complex_16 (***threep_f)[2], (***threep_b)[2];
   /***************/


   qcd_uint_4 k;
   char threep_name[qcd_MAX_STRING_LENGTH];
   char twop_mes_name[qcd_MAX_STRING_LENGTH];
   char twop_nucl_name[qcd_MAX_STRING_LENGTH];
   char tmp_string[qcd_MAX_STRING_LENGTH];
   char prop_name[2][qcd_MAX_STRING_LENGTH]; 
   char tmp_s[15];
	   
   int myid,numprocs, namelen;    
   char processor_name[MPI_MAX_PROCESSOR_NAME];

   qcd_complex_16 (**block_mes)[2], twop_mes_loc[2], twop_nucl_loc[16];
   qcd_complex_16 *(*block_nucl)[4][4];
   qcd_complex_16 (*twop_mes)[2];//pi+, pi-
   qcd_complex_16 (*twop_nucl)[2][16];// p, n
   qcd_complex_16 one_pm_ig5[2][4],g5[4]; //-for transformation purposes
  
   qcd_uint_2     cg5cg5_ind[16][4];
   qcd_complex_16 cg5cg5_val[16];
   FILE *pfile;

   MG4QCD_Init mg_init;
   MG4QCD_Parameters mg_params;
   MG4QCD_Status mg_status;

   inline void *replace_str(char *str, char *orig, char *rep)
   {
     static char temp[qcd_MAX_STRING_LENGTH];
     static char buffer[qcd_MAX_STRING_LENGTH];
     char *p;

     strcpy(temp, str);

     if(!(p = strstr(temp, orig)))  // Is 'orig' even in 'temp'?
       return temp;

     strncpy(buffer, temp, p-temp); // Copy characters from 'temp' start to 'orig' str
     buffer[p-temp] = '\0';

     sprintf(buffer + (p - temp), "%s%s", rep, p + strlen(orig));
     sprintf(str, "%s", buffer);    
   }

   inline int conf_index_fct(int t, int z, int y, int x, int mu) { //T Z Y X
     int pos = qcd_LEXIC(t,x,y,z,geo.lL);
     static int size_per_pos = 4*3*3*2; //link*SU(3)*complex
     int dir = (mu==1)?3:((mu==3)?1:mu); //swapping Z with X
     static int size_per_dir = 3*3*2;   

     return pos*size_per_pos + dir*size_per_dir;
   }

   inline int vector_index_fct(int t, int z, int y, int x) {
     int pos = qcd_LEXIC(t,x,y,z,geo.lL);
     static int size_per_pos = 4*3*2; //link*SU(3)*complex

     return pos*size_per_pos;
   }

   inline void smear_vect( qcd_vector *vect, qcd_gaugeField *u_s, qcd_uint_4 t_s ){
     for(int i=0; i<nsmear; i++)
       {
	 if(qcd_gaussIteration3d( vect, u_s, alpha, t_s))
	   {
	     fprintf(stderr,"process %i: Error while smearing!\n",geo.myid);
	     exit(EXIT_FAILURE);
	   }
       }
   }

   inline void print_mg_status() {
  
     if(myid==0) printf("Solving time %.2f sec (%.1f %% on coarse grid)\n", mg_status.time,
			100.*(mg_status.coarse_time/mg_status.time));
     if(myid==0) printf("Total iterations on fine grid %d\n", mg_status.iter_count);
     if(myid==0) printf("Total iterations on coarse grids %d\n", mg_status.coarse_iter_count);
     if (!mg_status.success && myid==0) printf("ERROR: not converged\n");
     if (!mg_status.success) exit(0);
   }	 

   inline int Cart_rank(MPI_Comm comm, const int coords[], int *rank) {
     *rank = (int) qcd_LEXIC((P[0]+coords[0])%P[0],(P[1]+coords[3])%P[1],(P[2]+coords[2])%P[2],(P[3]+coords[1])%P[3],P);
     return *rank;
   }
 
   inline int Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]) {
     qcd_uint_2 tmp_p[4];
     qcd_antilexic(tmp_p, rank, P);
     coords[0]=tmp_p[0];
     coords[1]=tmp_p[3];
     coords[2]=tmp_p[2];
     coords[3]=tmp_p[1];
     return 1;
   }

   //set up MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD,&numprocs);         // num. of processes taking part in the calculation
   MPI_Comm_rank(MPI_COMM_WORLD,&myid);             // each process gets its ID
   MPI_Get_processor_name(processor_name,&namelen); // 


   //////////////////// READ INPUT FILE /////////////////////////////////////////////
      
   if(argc!=2)
   {
      if(myid==0) fprintf(stderr,"No input file specified\n");
      exit(EXIT_FAILURE);
   }

   strcpy(param_name,argv[1]);
   if(myid==0)
   {
      i=0;
      printf("opening input file %s\n",param_name);
      params=qcd_getParams(param_name,&params_len);
      if(params==NULL)
      {
         i=1;
      }
   }
   MPI_Bcast(&i,1,MPI_INT, 0, MPI_COMM_WORLD);
   if(i==1) exit(EXIT_FAILURE);
   MPI_Bcast(&params_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(myid!=0) params = (char*) malloc(params_len*sizeof(char));
   MPI_Bcast(params, params_len, MPI_CHAR, 0, MPI_COMM_WORLD);
   
   sscanf(qcd_getParam("<processors_txyz>",params,params_len),"%hd %hd %hd %hd",&P[0], &P[1], &P[2], &P[3]);
   sscanf(qcd_getParam("<lattice_txyz>",params,params_len),"%hd %hd %hd %hd",&L[0], &L[1], &L[2], &L[3]);

   if(qcd_initGeometry(&geo, L, P, theta, myid, numprocs)) exit(EXIT_FAILURE);

   //Multigrid initialization parameters
   mg_init.comm_cart = MPI_COMM_WORLD;
   mg_init.Cart_rank=Cart_rank;
   mg_init.Cart_coords=Cart_coords;
   mg_init.global_lattice[0] = L[0];
   mg_init.global_lattice[1] = L[3];
   mg_init.global_lattice[2] = L[2];
   mg_init.global_lattice[3] = L[1];
   mg_init.procs[0] = P[0];
   mg_init.procs[1] = P[3];
   mg_init.procs[2] = P[2];
   mg_init.procs[3] = P[1];
   sscanf(qcd_getParam("<block_txyz>",params,params_len),"%d %d %d %d",&(mg_init.block_lattice[0]), 
	  &(mg_init.block_lattice[3]), &(mg_init.block_lattice[2]), &(mg_init.block_lattice[1]));
   if(myid==0) printf(" Global lattice: %i x %i x %i x %i\n", mg_init.global_lattice[0], mg_init.global_lattice[3],
		      mg_init.global_lattice[2], mg_init.global_lattice[1]);
   if(myid==0) printf(" Processors: %i x %i x %i x %i\n", mg_init.procs[0], mg_init.procs[3], mg_init.procs[2],
		      mg_init.procs[1]);
   if(myid==0) printf(" Block lattice: %i x %i x %i x %i\n", mg_init.block_lattice[0], mg_init.block_lattice[3],
		      mg_init.block_lattice[2], mg_init.block_lattice[1]);
   sscanf(qcd_getParam("<nlevels>",params,params_len),"%d", &(mg_init.number_of_levels));
   if(myid==0) printf(" Got number of levels: %d\n",mg_init.number_of_levels);
   mg_init.bc = 3;
#pragma omp parallel
   {
     mg_init.number_openmp_threads = omp_get_num_threads();
   }
   if(myid==0) printf(" Using number of threads: %d\n", mg_init.number_openmp_threads);

   sscanf(qcd_getParam("<kappa>",params,params_len),"%lf",&(mg_init.kappa));
   sscanf(qcd_getParam("<mu>",params,params_len),"%lf",&(mg_init.mu));
   sscanf(qcd_getParam("<csw>",params,params_len),"%lf",&(mg_init.csw));

   MG4QCD_init( &mg_init, &mg_params, &mg_status);

   //Multigrid parameters
   mg_params.conf_index_fct = conf_index_fct;
   mg_params.vector_index_fct = vector_index_fct;
   mg_params.setup_iterations[0] = 5;
   sscanf(qcd_getParam("<nvector>",params,params_len),"%d",&(mg_params.mg_basis_vectors[0]));
   if(myid==0) printf(" Got number of test vector: %d\n",mg_params.mg_basis_vectors[0]);
   mg_params.mg_basis_vectors[1] = max(28, mg_params.mg_basis_vectors[0]);
   if(myid==0) printf(" Using number of test vector on second level: %d\n",mg_params.mg_basis_vectors[1]);
   for(mu=0;mu<mg_params.number_of_levels; mu++)
     mg_params.mu_factor[mu]=1.;
   sscanf(qcd_getParam("<factor_cmu>",params,params_len),"%lf",&(mg_params.mu_factor[mg_params.number_of_levels-1]));
   if(myid==0) printf(" Got factor for coarsest mu: %f\n",mg_params.mu_factor[mg_params.number_of_levels-1]);
   mg_params.print = 1;

   MG4QCD_update_parameters( &mg_params, &mg_status );
   
   if(myid==0) printf(" Local lattice: %i x %i x %i x %i\n",geo.lL[0],geo.lL[1],geo.lL[2],geo.lL[3]);  
   
   sscanf(qcd_getParam("<alpha_gauss>",params,params_len),"%lf",&alpha);
   if(myid==0) printf(" Got alpha_gauss: %lf\n",alpha);
   sscanf(qcd_getParam("<nsmear_gauss>",params,params_len),"%d",&nsmear);
   if(myid==0) printf(" Got nsmear_gauss: %d\n",nsmear);
   sscanf(qcd_getParam("<alpha_APE>",params,params_len),"%lf",&alphaAPE);
   if(myid==0) printf(" Got alpha_APE: %lf\n",alphaAPE);
   sscanf(qcd_getParam("<nsmear_APE>",params,params_len),"%d",&nsmearAPE);
   if(myid==0) printf(" Got nsmear_APE: %d\n",nsmearAPE);   
   strcpy(gauge_name,qcd_getParam("<cfg_name>",params,params_len));
   if(myid==0) printf(" Got conf name: %s\n",gauge_name);

   //dir_loop, binary value for direction to loop on.
   //e.g. 1=x, 2=y, 3=xy, 4=z, 5=xz, 6=yz, 7=xyz 
   sscanf(qcd_getParam("<dir>",params,params_len),"%d",&dir_loop);
   if(myid==0) printf(" Got direction for phase factor: %d\n",dir_loop);

   //useful for loop over the momenta
   sscanf(qcd_getParam("<mom_min>",params,params_len),"%d",&mom_min);
   if(myid==0) printf(" Got momentum min: %d\n",mom_min);

   sscanf(qcd_getParam("<mom_max>",params,params_len),"%d",&mom_max);
   if(myid==0) printf(" Got momentum max: %d\n",mom_max);

   sscanf(qcd_getParam("<mom_step>",params,params_len),"%d",&mom_step);
   if(myid==0) printf(" Got momentum step: %d\n",mom_step);

   sscanf(qcd_getParam("<zeta>",params,params_len),"%lf",&zeta);
   if(myid==0) printf(" Got zeta: %.4f\n",zeta);

   mom[0]=0; //T

   sscanf(qcd_getParam("<source_pos_txyz>",params,params_len),"%d %d %d %d",&x_src[0], &x_src[1], &x_src[2], &x_src[3]);
   if(myid==0) printf(" Got source coords: %d %d %d %d\n",x_src[0],x_src[1],x_src[2],x_src[3]);

   sscanf(qcd_getParam("<t>",params,params_len),"%d %d",&t_start, &t_stop);
   if(myid==0) printf("Got insertion time slices: %d ... %d\n",t_start,t_stop);

#ifdef PION   
   //Prototype string with available replacement %mom, %dir, %zeta
   strcpy(twop_mes_name,qcd_getParam("<twop_mes_file>",params,params_len));
   if(myid==0) printf("Two point pions function #%d name: %s\n", i, twop_mes_name);
#endif      

#ifdef NUCLEON
   //Prototype string with available replacement %mom, %dir, %zeta, %nucl
   strcpy(twop_nucl_name,qcd_getParam("<twop_nucl_file>",params,params_len));
   if(myid==0) printf("Two point nucleons function #%d name: %s\n", i, twop_nucl_name);
#endif      

   /****** for threep *****/
   sscanf(qcd_getParam("<nprojectors>",params,params_len),"%hd",&nprojs);
   if(myid==0) printf(" Got number of projectors: %d\n", nprojs);

   sscanf(qcd_getParam("<t_sink>",params,params_len),"%d",&t_sink);
   if(myid==0) printf(" Got t position for sink smearing: %d\n",t_sink);
   
   strcpy(threep_name,qcd_getParam("<threep_file>",params,params_len));
   if(myid==0) printf("Three point function #%d name: %s\n", i, threep_name);

   proj_type = malloc(nprojs*sizeof(qcd_uint_2));
   prop_seq_name = malloc(nprojs*2*qcd_MAX_STRING_LENGTH*sizeof(char));
   source_seq_name = malloc(nprojs*2*qcd_MAX_STRING_LENGTH*sizeof(char));

   for(pr=0; pr<nprojs; pr++) {
     sprintf(tmp_string,"<projector_%d>",pr+1);
     sscanf(qcd_getParam(tmp_string,params,params_len),"%hd",&proj_type[pr]);
     if(myid==0) printf("Sequential source %d projector type: %d\n",pr, proj_type[pr]); //depends on the projector  
     for(dir=1; dir<4; dir++)
       if((dir_loop >> (dir-1))%2 == 1)
         if(proj_dir_gamma[proj_type[pr]][dir]==NULL){
           if(myid==0)
             printf("ERROR: gamma insertion not defined in projectors_gamma.h for proj. %d and dir %d\n",proj_type[pr],dir);                 
	   exit(EXIT_FAILURE);
         }
   }
   /************************/
   free(params);      
   ///////////////////////////////////////////////////////////////////////////////////////////////////

   if(myid==0) printf("\n\n Starting calculation. The software will compute...\n");

   /************** to clarify the output of the code *********/

   //set mom_0_done flag to 0
   //mom_0_done=0;

   for(dir=1; dir<4; dir++)  //loop on dir 1 to 3
     if( (dir_loop >> (dir-1))%2 == 1) //do it only if dir has been required
       for(mom_loop=mom_min; mom_loop<=mom_max; mom_loop=mom_loop+mom_step){


#ifdef DO_MOM_0_ONCE
	 if(mom_0_done && mom_loop==0)
	   continue;
	 else if (mom_loop==0)
	   mom_0_done=1;
#endif
	 for(mu=0; mu<4; mu++)
	   mom[mu]=0;
	 mom[dir]=mom_loop;

#ifdef SKIP_MOM_ZERO
	 if(mom_loop==0)
	   continue;
#endif


#ifdef PION   
	 strcpy(tmp_string, twop_mes_name);
	 sprintf(tmp_s, "%+d_%+d_%+d", mom[1], mom[2], mom[3]);
	 replace_str(tmp_string, "%mom", tmp_s);
	 replace_str(tmp_string, "%dir", ((dir==1)?"x":((dir==2)?"y":"z")));
	 sprintf(tmp_s, "%.4f", zeta);
	 replace_str(tmp_string, "%zeta", tmp_s);
	   
	 if(myid==0) printf(" twop for pion with dir %s mom %+d zeta %.4f will be saved in %s\n", 
			      ((dir==1)?"x":((dir==2)?"y":"z")), mom[dir], zeta, tmp_string);
#endif
      
#ifdef NUCLEON  
	 for(fl=0; fl<2; fl++) { 	 
	   strcpy(tmp_string, twop_nucl_name);
	   sprintf(tmp_s, "%+d_%+d_%+d", mom[1], mom[2], mom[3]);
	   replace_str(tmp_string, "%mom", tmp_s);
	   replace_str(tmp_string, "%dir", ((dir==1)?"x":((dir==2)?"y":"z")));
	   sprintf(tmp_s, "%.4f", zeta);
	   replace_str(tmp_string, "%zeta", tmp_s);
	   replace_str(tmp_string, "%nucl", ((fl==0)?"proton":"neutron"));
	     
	   if(myid==0) printf(" twop for %s with dir %s mom %+d zeta %.4f will be saved in %s\n", 
			      ((fl==0)?"proton":"neutron"),
			      ((dir==1)?"x":((dir==2)?"y":"z")), mom[dir], zeta, tmp_string);
	 }
#endif


#ifdef THREEP_SKIP_MOM_0
	 if(mom_loop != 0) {
#endif
	   /******* for threep *******/

	 if (myid==0) printf ("mom1=%+d mom2=%+d mom3=%+d\n", mom[1], mom[2], mom[3]);

#ifdef SKIP_MOM_ZERO
	 if(mom_loop==0)
	   continue;
#endif
	 
	 for(pr=0; pr<nprojs; pr++) {
	   strcpy(tmp_string, threep_name);
	   sprintf(tmp_s, "%+d_%+d_%+d", mom[1], mom[2], mom[3]);
	   replace_str(tmp_string, "%mom", tmp_s);
	   replace_str(tmp_string, "%dir", ((dir==1)?"x":((dir==2)?"y":"z")));                                                              
	   sprintf(tmp_s, "%d", proj_type[pr]);
	   replace_str(tmp_string, "%proj", tmp_s);
	   sprintf(tmp_s, "%.4f", zeta);
	   replace_str(tmp_string, "%zeta", tmp_s);

	   if(myid==0) printf(" threep with dir %s mom %+d proj %d will be saved in %s\n",
			      ((dir==1)?"x":((dir==2)?"y":"z")), mom[dir], proj_type[pr], tmp_string);
	 }
	 
#ifdef THREEP_SKIP_MOM_0
	 }
#endif
	 /*************/
      
       } //mom_loop  
   
   ///////////////////////// INITIALIZATING STRUCTURES ///////////////////////////////////////////////
   
   qcd_initGaugeField(&u,&geo);
   qcd_initGaugeField(&uAPE,&geo);
   qcd_initGaugeField(&uAPE2,&geo);
   qcd_initGaugeField(&u_ms,&geo);
   
   /**** for threep****/
   qcd_initGaugeField(&u_temp,&geo);
   qcd_initGaugeField(&u_shift_f,&geo);
   qcd_initGaugeField(&u_shift_b,&geo);
   qcd_initGaugeField(&u_prod_f,&geo);
   qcd_initGaugeField(&u_prod_b,&geo);

   prop_seq = malloc((nprojs)*2*sizeof(qcd_propagator));
   
   for(fl=0;fl<2;fl++){
     qcd_initPropagator(&(prop_ms[fl]), &geo);
     qcd_initPropagator(&(prop_shift_f[fl]), &geo);
     qcd_initPropagator(&(prop_shift_b[fl]), &geo);
     for(pr=0; pr<nprojs; pr++)
       qcd_initPropagator(&(prop_seq[pr][fl]), &geo);
   }
   qcd_initPropagator(&(prop_temp), &geo);

   block_f = malloc(geo.lV3*sizeof(qcd_complex_16));
   block_b = malloc(geo.lV3*sizeof(qcd_complex_16));

   threep_loc_f=malloc((nprojs)*sizeof(qcd_complex_16 *));
   for( pr=0; pr<nprojs; pr++)
     threep_loc_f[pr] = malloc((geo.L[1]/2)*2*sizeof(qcd_complex_16));

   threep_loc_b=malloc((nprojs)*sizeof(qcd_complex_16 *));
   for( pr=0; pr<nprojs; pr++)
     threep_loc_b[pr] = malloc((geo.L[1]/2)*2*sizeof(qcd_complex_16));

   threep_f = malloc((t_stop-t_start+1)*sizeof(qcd_complex_16**));
   threep_b = malloc((t_stop-t_start+1)*sizeof(qcd_complex_16**));

   for(t=t_start; t<=t_stop; t++) {
     threep_f[t-t_start] = malloc((nprojs)*sizeof(qcd_complex_16 *));
     for( pr=0; pr<nprojs; pr++)
       threep_f[t-t_start][pr] = malloc((geo.L[1]/2)*2*sizeof(qcd_complex_16));
     threep_b[t-t_start] = malloc((nprojs)*sizeof(qcd_complex_16 **));
     for( pr=0; pr<nprojs; pr++)
       threep_b[t-t_start][pr] = malloc((geo.L[1]/2)*2*sizeof(qcd_complex_16));
   }

   /***************/


   for(fl=0;fl<2;fl++){
     qcd_initPropagator(&(prop[fl]), &geo);
     qcd_initPropagator(&(prop_pb[fl]), &geo);  
   }

   qcd_initVector(&vec,&geo); 
   qcd_initVector(&vec_mg,&geo);

   twop_mes = malloc((t_stop-t_start+1)*2*sizeof(qcd_complex_16));
   twop_nucl = malloc((t_stop-t_start+1)*2*16*sizeof(qcd_complex_16));
   
   block_mes = malloc((t_stop-t_start+1)*sizeof(qcd_complex_16*));
   for(t=t_start; t<=t_stop; t++) 
     block_mes[t-t_start] = malloc((geo.lV3)*2*sizeof(qcd_complex_16));
   
   block_nucl = malloc((t_stop-t_start+1)*16*sizeof(qcd_complex_16*));
   for(t=t_start; t<=t_stop; t++) 
     for(mu=0;mu<4;mu++)
       for(nu=0;nu<4;nu++)
	 block_nucl[t-t_start][mu][nu] = malloc((geo.lV3)*sizeof(qcd_complex_16));

   // tabulate non-vanishing entries of C*gamma5 x \bar C*gamma5
   ctr = 0;
   for(mu=0;mu<4;mu++) 
     for(nu=0;nu<4;nu++)
       for(ku=0;ku<4;ku++)
	 for(lu=0;lu<4;lu++)
	   {
	     tmp_c = qcd_CMUL(qcd_CGAMMA[5][mu][nu],qcd_BAR_CGAMMA[5][ku][lu]);
	     if(qcd_NORM(tmp_c)>1e-3)
	       {
		 cg5cg5_val[ctr] = tmp_c;
		 cg5cg5_ind[ctr][0] = mu;
		 cg5cg5_ind[ctr][1] = nu;
		 cg5cg5_ind[ctr][2] = ku;
		 cg5cg5_ind[ctr][3] = lu;                                        
		 ctr++;
	       }
	   }

   tmp_c = (qcd_complex_16) {0.,1.};
   for(mu=0;mu<4;mu++){   
     one_pm_ig5[0][mu]  = qcd_CADD( qcd_ONE[mu][mu],qcd_CMUL(tmp_c,qcd_GAMMA[5][mu][mu]) );	   
     one_pm_ig5[1][mu] = qcd_CSUB( qcd_ONE[mu][mu],qcd_CMUL(tmp_c,qcd_GAMMA[5][mu][mu]) );
     g5[mu] = qcd_CSCALE(qcd_GAMMA[5][mu][mu],1.);
   }

   ///////////////////////// CONF AND SETUP //////////////////////////////////////////////
   
   if(qcd_getGaugeField(gauge_name,qcd_GF_LIME,&u))
     {
       fprintf(stderr,"process %i: Error reading gauge field!\n",myid);
       exit(EXIT_FAILURE);
     }
   
   if(myid==0) printf("gauge-field loaded\n");
   plaq = qcd_calculatePlaquette(&u);
   if(myid==0) printf("plaquette = %e\n",plaq);


   qcd_communicateGaugePM(&u);
   qcd_waitall(&geo); 
   qcd_copyGaugeField(&uAPE2, &u);
   qcd_communicateGaugePM(&uAPE2);
   qcd_waitall(&geo);
   u_ptr = &uAPE2;
   uAPE_ptr = &uAPE;   

   for(i=0; i<nsmearAPE; i++)
     {
       qcd_apeSmear3d(uAPE_ptr, u_ptr, alphaAPE);
       utmp_ptr=u_ptr; u_ptr=uAPE_ptr; uAPE_ptr=utmp_ptr;   
     }

   qcd_copyGaugeField(uAPE_ptr, u_ptr);
   qcd_communicateGaugePM(uAPE_ptr);
   qcd_waitall(&geo);
   
   // Multigrid set configuration
   MG4QCD_set_configuration( (double*) &(u.D[0][0][0][0].re), &mg_status);
   if(myid==0) printf("multigrid plaquette = %e\n",mg_status.info);
   
   //Multigrid setup
   if(myid==0) printf("Running setup\n");
   MG4QCD_setup( &mg_status );
   if(myid==0) printf("Setup time %.2f sec (%.1f %% on coarse grid)\n", mg_status.time,
		      100.*(mg_status.coarse_time/mg_status.time));

   if(myid==0) printf("gauge-field APE-smeared\n");
   plaq = qcd_calculatePlaquette(&uAPE);
   if(myid==0) printf("plaquette = %e\n",plaq); 

   for(i=0; i<4; i++)
     lx_src[i] = x_src[i]-geo.Pos[i]*geo.lL[i];  //source_pos in local lattice

   //set mom_0_done flag to 0
   //mom_0_done=0;

   for(dir=1; dir<4; dir++)  //loop on dir 1 to 3
     if( (dir_loop >> (dir-1))%2 == 1) //do it only if dir has been required
       for (mom_loop = mom_min; mom_loop <=mom_max; mom_loop=mom_loop+mom_step) {//BIG EXTERNAL LOOP

#ifdef SKIP_MOM_ZERO
	 if(mom_loop==0)
	   continue;
#endif


#ifdef DO_MOM_0_ONCE
	 if(mom_0_done && mom_loop==0)
	   continue;
	 else if (mom_loop==0)
	   mom_0_done=1;
#endif

	 for(mu=0; mu<4; mu++)
	   mom[mu]=0;
	 mom[dir]=mom_loop;
	 
	 if(myid==0) printf("dir=%d zeta=%.2f\n",dir,zeta);
	 
	 /********* momentum smearing ***********/
	 
	 qcd_copyGaugeField(&u_ms, &uAPE);
	 qcd_communicateGaugePM(uAPE_ptr);
	 qcd_waitall(&geo);
   	 for(mu=1; mu<4; mu++)
	   if ( mom[mu] != 0. ) {
	     tmp = 2.0*M_PI*((double)zeta * (double)mom[mu]/(double)geo.L[mu]);
	     tmp_c = (qcd_complex_16){cos(tmp),sin(tmp)};
#pragma omp parallel for private(j, c1, c2)
	     for(j=0;j<geo.lV;j++)
	       for(c1=0;c1<3;c1++)
		 for(c2=0;c2<3;c2++)
		   u_ms.D[j][mu][c1][c2]=qcd_CMUL(uAPE.D[j][mu][c1][c2], tmp_c);
	   }
    
	 //if(myid==0) printf("gauge-field momentum smeared\n");
	 qcd_communicateGaugePM(&u_ms);
	 qcd_waitall(&geo);
	 plaq = qcd_calculatePlaquette(&u_ms);
	 if(myid==0) printf("plaquette momentum smearing= %e\n",plaq);

	 
	 /**************************************/
	 
	 
	 ///////////////////////////////////////////////////////////////////////////////////////////////////
	 for(fl=0; fl<2; fl++) {
	   
	   ////////////////////////////////////// POINT SOURCE ///////////////////////////////////////////
	   if(myid==0) printf(" Creating %s propagator\n\n", (mg_params.mu>0)?"up":"down");
	   
	   for(nu=0; nu<4; nu++)
	     for(c2=0; c2<3; c2++) {
	       qcd_zeroVector(&vec); 
	       if( (lx_src[0]>=0) && (lx_src[0]<geo.lL[0]) && (lx_src[1]>=0) &&
		   (lx_src[1]<geo.lL[1]) && (lx_src[2]>=0) && (lx_src[2]<geo.lL[2]) &&
		   (lx_src[3]>=0) && (lx_src[3]<geo.lL[3]))
		 vec.D[qcd_LEXIC(lx_src[0],lx_src[1],lx_src[2],lx_src[3],geo.lL)][nu][c2].re=1.;
	       
	       smear_vect(&vec, &u_ms, x_src[0]);
	       
	       MG4QCD_solve( (double*) &(vec_mg.D[0][0][0].re), (double*) &(vec.D[0][0][0].re), 1.e-9, &mg_status );
	       print_mg_status();
	       
	       qcd_copyPropagatorVector(&(prop[fl]), &vec_mg, nu, c2);
	       
	       for(t=t_start; t<=t_stop; t++){
		 lt = ((t+x_src[0])%geo.L[0]) - geo.Pos[0]*geo.lL[0];       
#pragma omp parallel for private(j, i, mu, c1)
		 for(j=0;j<geo.lV3;j++) {
		   i =  lt + j*geo.lL[0];
		   for(mu=0;mu<4;mu++)
		     for(c1=0;c1<3;c1++)
		       vec_mg.D[i][mu][c1] = qcd_CSCALE(qcd_CMUL(vec_mg.D[i][mu][c1],
								 qcd_CMUL(one_pm_ig5[fl][mu],one_pm_ig5[fl][nu])), 0.5);
		 }
	       }
	       
	       qcd_copyPropagatorVector(&(prop_pb[fl]), &vec_mg, nu, c2);

	       /*** for threep ***/
	       qcd_copyVectorPropagator(&vec_mg, &(prop[fl]), nu, c2);
	       if(myid==0) printf("t_sink/geo.L=%d \n",t_sink % geo.L[0]);
	       smear_vect(&vec_mg, &u_ms, t_sink % geo.L[0] ); //(t_sink % geo.L[0])-added -smear the propagators on the sink side 
	       qcd_copyPropagatorVector(&(prop_ms[fl]), &vec_mg, nu, c2);
	       /********/
 
	       if(myid==0) printf("Vector smeared\n");
	       
	     }// end c2

#ifdef THREEP_SKIP_MOM_0
	   if(mom_loop != 0) {
#endif
	   /***** for threep ***/

           MG4QCD_get_parameters( &mg_params );
	   mg_params.mixed_precision=1;
           MG4QCD_update_parameters( &mg_params, &mg_status );

           for(pr=0; pr<nprojs; pr++) {
             qcd_zeroPropagator(&(prop_seq[pr][fl]));

             if(myid==0) printf(" Creating sequential source with Proj%i for %s.\n\n",
                                proj_type[pr], (fl==0)?"dd":"uu");
             lt_sink = (t_sink % geo.L[0]) - geo.Pos[0]*geo.lL[0]; // (t_sink % geo.L[0]) local t_sink                                       
             MPI_Barrier(MPI_COMM_WORLD);

	     if((lt_sink>=0) && (lt_sink<geo.lL[0]))
#pragma omp parallel for private(i, lx, ly, lz, j, cc1, c1, c2, c3, cc2, c1p, c2p, c3p, \
				 ctr2, mu, nu, ku, lu, bu, cu, tmp_c, gu, ju )
               for(i=0; i<geo.lV3; i++){
                 j = lt_sink + i*geo.lL[0];
                 for(cc1=0;cc1<6;cc1++) {
                   c1=qcd_EPS[cc1][0];
                   c2=qcd_EPS[cc1][1];
                   c3=qcd_EPS[cc1][2];
                   for(cc2=0;cc2<6;cc2++) {
                     c1p=qcd_EPS[cc2][0];
                     c2p=qcd_EPS[cc2][1];
                     c3p=qcd_EPS[cc2][2];
                     for(ctr2=0;ctr2<ctr;ctr2++) {
                       mu = cg5cg5_ind[ctr2][0];
                       nu = cg5cg5_ind[ctr2][1];
                       ku = cg5cg5_ind[ctr2][2];
                       lu = cg5cg5_ind[ctr2][3];
                       for(bu=0;bu<4;bu++)
                         for(cu=0;cu<4;cu++)
                           if(qcd_NORM(PROJECTOR[proj_type[pr]][bu][cu])>1e-3) {
			     if(fl==0) {//for d-insertion                                                                       
                               tmp_c = qcd_CMUL(qcd_CSCALE(PROJECTOR[proj_type[pr]][bu][cu], -qcd_SGN_EPS[cc1]*qcd_SGN_EPS[cc2]\
							   ), cg5cg5_val[ctr2]);
                               prop_seq[pr][fl].D[j][nu][lu][c3][c3p]=qcd_CADD(prop_seq[pr][fl].D[j][nu][lu][c3][c3p], qcd_CMUL\
									       (tmp_c, qcd_CADD(qcd_CMUL(prop_ms[0].D[j][mu][cu][c1][c1p], prop_ms[0].D[j][bu][ku][c2][c2p]),qcd_CMUL(prop_ms[0].D[j][mu][ku][\
																									      c1][c1p], prop_ms[0].D[j][bu][cu][c2][c2p]))));
                             }//end if fl == 0                                                                                  
                             else if(fl==1) { //for up-insertion                                                                
                               tmp_c = qcd_CMUL(qcd_CSCALE(PROJECTOR[proj_type[pr]][bu][cu], qcd_SGN_EPS[cc1]*qcd_SGN_EPS[cc2])\
						,cg5cg5_val[ctr2]);
                               for(gu=0; gu<4; gu++)
                                 for(ju=0; ju<4; ju++){
                                   if((mu==gu) && (cu==ju))
                                     prop_seq[pr][fl].D[j][gu][ju][c3][c3p]=qcd_CSUB(prop_seq[pr][fl].D[j][gu][ju][c3][c3p],qcd_CMUL(tmp_c, qcd_CMUL(prop_ms[1].D[j][nu][lu][c1][c1p], prop_ms[0].D[j][bu][ku][c2][c2p])));
                                   if((mu==gu) && (ku==ju))
                                     prop_seq[pr][fl].D[j][gu][ju][c3][c3p]=qcd_CSUB(prop_seq[pr][fl].D[j][gu][ju][c3][c3p],qcd_CMUL(tmp_c, qcd_CMUL(prop_ms[1].D[j][nu][lu][c1][c1p], prop_ms[0].D[j][bu][cu][c2][c2p])));
                                   if((bu==gu) && (cu==ju))
                                     prop_seq[pr][fl].D[j][gu][ju][c3][c3p]=qcd_CSUB(prop_seq[pr][fl].D[j][gu][ju][c3][c3p], qcd_CMUL(tmp_c, qcd_CMUL(prop_ms[1].D[j][nu][lu][c1][c1p], prop_ms[0].D[j][mu][ku][c2][c2p])));
                                   if((bu==gu) && (ku==ju))
                                     prop_seq[pr][fl].D[j][gu][ju][c3][c3p]=qcd_CSUB(prop_seq[pr][fl].D[j][gu][ju][c3][c3p],qcd_CMUL(tmp_c, qcd_CMUL(prop_ms[1].D[j][nu][lu][c1][c1p], prop_ms[0].D[j][mu][cu][c2][c2p])));
                                 }//end ju gu                                                                                   
                             }//end if fl==1                                                                                    
                           } //end norma-projector                                                                              
                     } //end ctr2                                                                                               
                   } //end cc2                                                                                                  
                 } //end cc1                                                                                                    
               }//end volume loop - parallelized

             qcd_gamma5Propagator(&(prop_seq[pr][fl]));

	     if((lt_sink>=0) && (lt_sink<geo.lL[0]))
#pragma omp parallel for private(i, lx, ly, lz, j, x, y, z, tmp, tmp_c, mu, nu, c1, c2 )
               for(i=0; i<geo.lV3; i++){
		 lx= i % geo.lL[1];
		 ly = ((i-lx)/geo.lL[1]) % geo.lL[2];
		 lz = ((i-lx)/geo.lL[1] - ly)/geo.lL[2];
		 //for(lx=0; lx<geo.lL[1]; lx++)
		 //for(ly=0; ly<geo.lL[2]; ly++)
		 //for(lz=0; lz<geo.lL[3]; lz++) {
		 j = qcd_LEXIC(lt_sink,lx,ly,lz,geo.lL);
		 x=lx+geo.Pos[1]*geo.lL[1]- x_src[1];
		 y=ly+geo.Pos[2]*geo.lL[2]- x_src[2];
		 z=lz+geo.Pos[3]*geo.lL[3]- x_src[3];
		 tmp = (((double) mom[1] * (double) x)/(double) geo.L[1] +
			((double) mom[2] * (double) y)/(double) geo.L[2] +
			((double) mom[3] * (double) z)/(double) geo.L[3])*2.0*M_PI;
		 tmp_c = (qcd_complex_16) {cos(tmp), -sin(tmp)};
		 for(mu=0;mu<4;mu++)
		   for(nu=0;nu<4;nu++)
		     for(c1=0;c1<3;c1++)
		       for(c2=0;c2<3;c2++)
			 prop_seq[pr][fl].D[j][mu][nu][c1][c2] = qcd_CMUL(prop_seq[pr][fl].D[j][mu][nu][c1][c2], tmp_c);
	       }//end volume i - parallelized
	     
	     qcd_conjPropagator(&(prop_seq[pr][fl]));

	     if(myid==0)
               printf(" Creating sequential propagator with Proj%i for %s.\n\n",proj_type[pr], (fl==0)?"dd":"uu");
	     for(nu=0; nu<4; nu++)   //to invert d-insertion
	       for(c2=0; c2<3; c2++) {
                 qcd_copyVectorPropagator(&vec, &(prop_seq[pr][fl]), nu, c2);
                 smear_vect(&vec, &u_ms, t_sink % geo.L[0] );// (t_sink % geo.L[0])                   
		 
                 MG4QCD_solve( (double*) &(vec_mg.D[0][0][0].re), (double*) &(vec.D[0][0][0].re), 1.e-9, &mg_status );
                 print_mg_status();
		 
                 qcd_copyPropagatorVector(&(prop_seq[pr][fl]), &vec_mg, nu, c2);
	       }
	   } //end projs
	   /**********/
#ifdef THREEP_SKIP_MOM_0
	   }
#endif

           MG4QCD_get_parameters( &mg_params );
	   mg_params.mixed_precision=2;
	   mg_params.mu*=-1.;   	
	   if(myid==0) printf("Running update\n");
	   MG4QCD_update_parameters( &mg_params, &mg_status );
	   if(myid==0) printf("Updating time %.2f sec\n", mg_status.time);	  
	   
	 }//end fl
   
	 //removing tmLQCD antipbc
	 if(mg_init.bc==3)
	   for(lt=0; lt<geo.lL[0]; lt++) {
	     t = lt + geo.Pos[0] * geo.lL[0];
	     tmp = theta[0] * (double) t/(double) geo.L[0];
	     tmp_c  = (qcd_complex_16) {cos(tmp), sin(tmp)};
	     for(fl=0; fl<2; fl++){
	       qcd_mulPropagatorC3d(&(prop[fl]), tmp_c, (t+x_src[0]) % geo.L[0]); //for threep
	       qcd_mulPropagatorC3d(&(prop_pb[fl]), tmp_c, (t+x_src[0]) % geo.L[0]);
	     }
	     /*** for threep***/
#ifdef THREEP_SKIP_MOM_0
	     if(mom_loop!=0){
#endif
	     // the backward-props get the complex-conjugated phases, so after the conjugation
	     // in the next step they will be correct                                                                                                 
	     tmp_c = (qcd_complex_16) {cos(M_PI*(t_sink-(t+x_src[0])+2*(t_sink-x_src[0]))/geo.L[0]),
				       -sin(M_PI*(t_sink-(t+x_src[0])+2*(t_sink-x_src[0]))/geo.L[0])};
	     for(fl=0; fl<2; fl++)
	       for(pr=0; pr<nprojs; pr++)
		 qcd_mulPropagatorC3d(&(prop_seq[pr][fl]), tmp_c, (t+x_src[0]) % geo.L[0]);
		 /*******/
#ifdef THREEP_SKIP_MOM_0
	     }
#endif
	   } //end lt


	 for(fl=0; fl<2; fl++)
	   for(mu=0;mu<4;mu++)
	     for(c1=0;c1<3;c1++){
	       
	       qcd_copyVectorPropagator(&vec,&(prop_pb[fl]),mu,c1);
	       for(i=0; i<nsmear; i++){
		 if(qcd_gaussIteration3dAll(&vec,&u_ms,alpha,i==0)){
		   fprintf(stderr,"process %i: Error while smearing!\n",geo.myid);
		   exit(EXIT_FAILURE);
		 }
	       }
	       qcd_copyPropagatorVector(&(prop_pb[fl]),&vec,mu,c1);
	     }
   
	 
	 ////////////////////////////////////// TWO POINT ///////////////////////////////////////////
	 
#ifdef PION   
	 
	 if(myid==0) printf(" Running pion\n");
	 
	 for(t=t_start; t<=t_stop; t++){
	   lt = ((t+x_src[0])%geo.L[0]) - geo.Pos[0]*geo.lL[0];
	   if(lt>=0 && lt<geo.lL[0]){  //inside the local lattice, otherwise nothing to calculate
#pragma omp parallel for private(i, j, fl, mu, nu, c1, c2)
	     for(i=0; i<geo.lV3; i++){
	       j =  lt + i*geo.lL[0];
	       for(fl=0; fl<2; fl++) {
		 block_mes[t-t_start][i][fl] = (qcd_complex_16) {0,0};
		 for(mu=0;mu<4;mu++)
		   for(nu=0;nu<4;nu++)
		     for(c1=0;c1<3;c1++)
		       for(c2=0;c2<3;c2++)
			 block_mes[t-t_start][i][fl] = qcd_CSUB( block_mes[t-t_start][i][fl],
								 qcd_CMUL(prop_pb[fl].D[j][mu][nu][c1][c2],
								 qcd_CONJ(prop_pb[fl].D[j][mu][nu][c1][c2])));
	       }//end fl
	     }// end i - parallelized
	   }//end if lt
	 }// end t
	   
	 for(t=t_start; t<=t_stop; t++){
	   lt = ((t+x_src[0])%geo.L[0]) - geo.Pos[0]*geo.lL[0];
	   if(lt>=0 && lt<geo.lL[0]){  //inside the local lattice, otherwise nothing to calculate
	     for(fl=0; fl<2; fl++) 
	       twop_mes_loc[fl] = (qcd_complex_16) {0,0};
#pragma omp parallel private(i, lx, ly, lz, x, y, z, tmp, tmp_c, fl)
	       {
		 qcd_complex_16 twop_mes_omp[2];
		 for(fl=0; fl<2; fl++)
		   twop_mes_omp[fl] = (qcd_complex_16) {0,0};
#pragma omp for 
		 for(i=0; i<geo.lV3; i++){
		   lx=i % geo.lL[1];
		   ly=((i-lx)/geo.lL[1])% geo.lL[2];
		   lz=(i-lx-geo.lL[1]*ly)/(geo.lL[1]*geo.lL[2]);
		   
		   x=lx + geo.Pos[1]*geo.lL[1] - x_src[1];
		   y=ly + geo.Pos[2]*geo.lL[2] - x_src[2];
		   z=lz + geo.Pos[3]*geo.lL[3] - x_src[3];
		   
		   tmp = (((double) mom[1]*(double) x)/(double) geo.L[1] + 
			  ((double) mom[2]*(double) y)/(double) geo.L[2] + 
			  ((double) mom[3]*(double) z)/(double) geo.L[3])*2.*M_PI;
		   
		   tmp_c = (qcd_complex_16) {cos(tmp), -sin(tmp)}; 
		   
		   for(fl=0; fl<2; fl++) 
		     twop_mes_omp[fl]  = qcd_CADD( twop_mes_omp[fl],  qcd_CMUL( block_mes[t-t_start][i][fl] ,tmp_c)); 	
		 }// end i - parallelized
#pragma omp critical
		 for(fl=0; fl<2; fl++)
		   twop_mes_loc[fl] = qcd_CADD( twop_mes_loc[fl], twop_mes_omp[fl]);
	       }
	       MPI_Allreduce(&(twop_mes_loc[0].re), &(twop_mes[t-t_start][0].re),
			     4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	   }//end if lt
	 }// loop over t
	   
	 if(myid==0) printf(" Done pions with mom %d %d %d. Results ...\n", mom[1], mom[2], mom[3]);
	 for(t=t_start; t<=t_stop; t++)     //write up and down parts of threeps
	   if(myid==0) 
	     printf( "  pion  %d %+e %+e %+e %+e \n", t, 
		     twop_mes[t-t_start][0].re, twop_mes[t-t_start][0].im,
		     twop_mes[t-t_start][1].re, twop_mes[t-t_start][1].im);
	 
	 if(myid==0) {
	   strcpy(tmp_string, twop_mes_name);
	     
	   sprintf(tmp_s, "%+d_%+d_%+d", mom[1], mom[2], mom[3]);
	   replace_str(tmp_string, "%mom", tmp_s);
#ifndef DO_MOM_0_ONCE
	   replace_str(tmp_string, "%dir", ((dir==1)?"x":((dir==2)?"y":"z")));
#else
	   replace_str(tmp_string, "%dir", ""));
#endif
	   sprintf(tmp_s, "%.4f", zeta);
	   replace_str(tmp_string, "%zeta", tmp_s);
	   printf("opening %s \n",tmp_string);
	   pfile=fopen(tmp_string,"w");
	   if(pfile==NULL) 
	     printf("failed to open %s for writing\n",threep_name[pr]); 
	   else 
	     for(t=t_start; t<=t_stop; t++)     //write up and down parts of threeps
	       if(myid==0) 
		 fprintf(pfile, "%d %+e %+e %+e %+e \n", t, 
			 twop_mes[t-t_start][0].re, twop_mes[t-t_start][0].im,
			 twop_mes[t-t_start][1].re, twop_mes[t-t_start][1].im);
	   fclose(pfile);	  
	 } //myid=0
#endif
      
#ifdef NUCLEON
	 for(fl=0; fl<2; fl++) {
	   
	   if(myid==0) printf(" Running %s\n", ((fl==0)?"proton":"neutron"));
	   
	   for(t=t_start; t<=t_stop; t++){
	     lt = ((t+x_src[0])%geo.L[0]) - geo.Pos[0]*geo.lL[0];
	     if(lt>=0 && lt<geo.lL[0]){  //inside the local lattice, otherwise nothing to calculate
	       
#pragma omp parallel for private(i, mu, nu)
	       for(i=0; i<geo.lV3; i++)   //set blocks to zero
		 for(mu=0;mu<4;mu++)
		   for(nu=0;nu<4;nu++)
		     block_nucl[t-t_start][mu][nu][i]= (qcd_complex_16) {0,0};
	       
	       qcd_contractions2pt(fl+1, block_nucl[t-t_start], &prop_pb[0], &prop_pb[1], NULL, NULL, &geo, lt);
	       
	     }//end lt inside local block condition
	   }//end t-loop      

	   if(myid==0) printf(" Running Fourier transformation \n");
	   if(myid==0) printf(" Done t = ");
	   
	   for(t=t_start; t<=t_stop; t++){
	     if(myid==0) printf("%i, ", t);
	     lt = ((t+x_src[0])%geo.L[0]) - geo.Pos[0]*geo.lL[0];
	     if(lt>=0 && lt<geo.lL[0]){  //inside the local lattice, otherwise nothing to calculate
	       for(k=0; k<16; k++)
		 twop_nucl_loc[k] = (qcd_complex_16) {0,0};
#pragma omp parallel private(i, k, lx, ly, lz, x, y, z, tmp, tmp_c, mu, nu)
	       {
		 qcd_complex_16 twop_nucl_omp[16];
		 for(k=0; k<16; k++)
		   twop_nucl_omp[k] = (qcd_complex_16) {0,0};
#pragma omp for
		 for(i=0; i<geo.lV3; i++) {	 
		   lx=i % geo.lL[1];
		   ly=((i-lx)/geo.lL[1])% geo.lL[2];
		   lz=(i-lx-geo.lL[1]*ly)/(geo.lL[1]*geo.lL[2]);
		     
		   x=lx+geo.Pos[1]*geo.lL[1] - x_src[1];
		   y=ly+geo.Pos[2]*geo.lL[2] - x_src[2];
		   z=lz+geo.Pos[3]*geo.lL[3] - x_src[3];
		   tmp = (((double) mom[1]*(double) x)/(double) geo.L[1] + 
			  ((double) mom[2]*(double) y)/(double) geo.L[2] +
			  ((double) mom[3]*(double) z)/(double) geo.L[3])*2.*M_PI;
		   tmp_c = (qcd_complex_16) {cos(tmp), -sin(tmp)}; 	
		   for(mu=0;mu<4;mu++)
		     for(nu=0;nu<4;nu++) 
		       twop_nucl_omp[mu*4+nu] = qcd_CADD(twop_nucl_omp[mu*4+nu], 
							 qcd_CMUL(block_nucl[t-t_start][mu][nu][i],tmp_c));
		 }//end i
#pragma omp critical
		 for(mu=0;mu<4;mu++)
		   for(nu=0;nu<4;nu++) 
		     twop_nucl_loc[mu*4+nu] = qcd_CADD(twop_nucl_loc[mu*4+nu], twop_nucl_omp[mu*4+nu]); 
		 
	       }
	       MPI_Allreduce(&(twop_nucl_loc[0].re), &(twop_nucl[t-t_start][fl][0].re),
			     32, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	     }//end lt inside local block condition
	   }//end t-loop      
	   
	   if(myid==0) printf("\n");
	   
	   if(myid==0) printf(" Done %s with momentum %d %d %d. Results...\n", 
			      ((fl==0)?"proton":"neutron"), mom[1], mom[2], mom[3]);
	   if(myid==0) 
	     for(t=t_start; t<=t_stop; t++)     //write up and down parts of threeps
	       for(mu=0;mu<4;mu++){
		 printf(" %s %d %d %+e %+e %+e %+e %+e %+e %+e %+e\n", ((fl==0)?"proton":"neutron"), t, mu,	
			twop_nucl[t-t_start][fl][4*mu].re, twop_nucl[t-t_start][fl][4*mu].im,
			twop_nucl[t-t_start][fl][4*mu+1].re, twop_nucl[t-t_start][fl][4*mu+1].im,
			twop_nucl[t-t_start][fl][4*mu+2].re, twop_nucl[t-t_start][fl][4*mu+2].im,
			twop_nucl[t-t_start][fl][4*mu+3].re, twop_nucl[t-t_start][fl][4*mu+3].im);
	       }//end mu
	     
	     
	   if(myid==0) {
	     strcpy(tmp_string, twop_nucl_name);
	       
	     sprintf(tmp_s, "%+d_%+d_%+d", mom[1], mom[2], mom[3]);
	     replace_str(tmp_string, "%mom", tmp_s);

#ifndef DO_MOM_0_ONCE
	     replace_str(tmp_string, "%dir", ((dir==1)?"x":((dir==2)?"y":"z")));
#else
	     replace_str(tmp_string, "%dir", "");
#endif
	     sprintf(tmp_s, "%.4f", zeta);
	     replace_str(tmp_string, "%zeta", tmp_s);
	     replace_str(tmp_string, "%nucl", ((fl==0)?"proton":"neutron"));
	     printf("opening %s \n",tmp_string);
	     pfile=fopen(tmp_string,"w");
	     if(pfile==NULL) 
	       printf("failed to open %s for writing\n",threep_name[pr]); 
	     else 
	       for(t=t_start; t<=t_stop; t++)     //write up and down parts of threeps
		 for(mu=0;mu<4;mu++){
		   fprintf(pfile, "%d %d %+e %+e %+e %+e %+e %+e %+e %+e\n", t, mu,	
			   twop_nucl[t-t_start][fl][4*mu].re, twop_nucl[t-t_start][fl][4*mu].im,
			   twop_nucl[t-t_start][fl][4*mu+1].re, twop_nucl[t-t_start][fl][4*mu+1].im,
			   twop_nucl[t-t_start][fl][4*mu+2].re, twop_nucl[t-t_start][fl][4*mu+2].im,
			   twop_nucl[t-t_start][fl][4*mu+3].re, twop_nucl[t-t_start][fl][4*mu+3].im);
		 }
	       fclose(pfile);	  
	   } //myid=0
	 }//end fl
#endif

	 //////////////////////////////////// THREE POINT ///////////////////////////////////////

#ifdef THREEP_SKIP_MOM_0
         if(mom_loop!=0){
#endif

	 for(fl=0; fl<2; fl++)
	   for(pr=0; pr<nprojs; pr++) {
	     qcd_conjPropagator(&(prop_seq[pr][fl]));
	     qcd_gamma5Propagator(&(prop_seq[pr][fl]));
	   }
	 qcd_communicatePropagatorPM(&(prop[0]));
	 qcd_waitall(&geo);
	 qcd_communicatePropagatorPM(&(prop[1]));
	 qcd_waitall(&geo);
	 
	 //t_stop -> t_sink-x_src[0]
	 if(nprojs>0) for(t=0; t<=t_sink-x_src[0]; t++) {
	     if(myid==0) printf("Running t=%d\n",t);

	     lt = ((t+x_src[0])%geo.L[0]) - geo.Pos[0]*geo.lL[0];
	     if(lt>=0 && lt<geo.lL[0]) {  //inside the local lattice, otherwise nothing to calculate                                                    

	       qcd_unitGaugeField(&u_prod_b);                  //equal to one  -  For the shift Backward                                                
	       qcd_copyGaugeField(&u_shift_b,&u);              //u_shift_b is a copy of u                                                               

	       qcd_unitGaugeField(&u_prod_f);                  //equal to one  -  For the shift Forward                                                 
	       qcd_copyGaugeField(&u_shift_f,&u);              //u_shift_f is a copy of u                                                               

	       for(fl=0; fl<2; fl++){
		 qcd_copyPropagatorPropagator(&prop_shift_b[fl],&(prop[fl]));   //prop_shift_b is a copy of prop                                        
		 qcd_copyPropagatorPropagator(&prop_shift_f[fl],&(prop[fl]));   //prop_shift_f is a copy of prop                                        
		 qcd_communicatePropagatorPM(&(prop_shift_b[fl]));
		 qcd_waitall(&geo); 
	       }

	       for(s=0;s<geo.L[1]/2;s++) {
		 for(fl=0; fl<2; fl++)
		   for(pr=0; pr<nprojs; pr++) {
		     threep_loc_f[pr][s][fl] = (qcd_complex_16) {0,0};
		     threep_loc_b[pr][s][fl] = (qcd_complex_16) {0,0};

#pragma omp parallel for private(j, i, mu, nu, ku, c1, c2, c3) 
		     for(j=0; j<geo.lV3; j++) { //loop over points                                                                                      
		       i=lt + j*geo.lL[0];                        //points having the same time                                                         
		       
		       block_f[j]=(qcd_complex_16){0,0};
		       block_b[j]=(qcd_complex_16){0,0};

		       qcd_complex_16 prod_f[4][4], prod_b[4][4];
		       
		       for(mu=0; mu<4; mu++)              //extern spinorial indices - Loop which does the calculation                                  
			 for(nu=0; nu<4; nu++) {
			   prod_f[mu][nu]=(qcd_complex_16){0,0};
			   prod_b[mu][nu]=(qcd_complex_16){0,0};
			   for(ku=0; ku<4; ku++)
			     for(c1=0; c1<3; c1++)
			       for(c2=0; c2<3; c2++)
				 for(c3=0; c3<3; c3++) {
				   prod_f[mu][nu] = qcd_CADD(prod_f[mu][nu],
							     qcd_CMUL(prop_seq[pr][fl].D[i][mu][ku][c1][c2],
								      qcd_CMUL(u_prod_f.D[i][dir][c1][c3],
									       prop_shift_f[(fl+1)%2].D[i][nu][ku][c3][c2])));
				   prod_b[mu][nu] = qcd_CADD(prod_b[mu][nu],
							     qcd_CMUL(prop_seq[pr][fl].D[i][mu][ku][c1][c2],
								      qcd_CMUL(u_prod_b.D[i][dir][c1][c3],
									       prop_shift_b[(fl+1)%2].D[i][nu][ku][c3][c2])));
				   
				 } //end ku c1 c2 c3  
			   
			   //accumulating mu and nu in block, multipling by gamma                                                                       
			   if(qcd_NORM(proj_dir_gamma[proj_type[pr]][dir][mu][nu])>1e-4) {
			     block_f[j]=qcd_CADD(block_f[j], qcd_CMUL(prod_f[mu][nu],
								      proj_dir_gamma[proj_type[pr]][dir][mu][nu]));
			     block_b[j]=qcd_CADD(block_b[j], qcd_CMUL(prod_b[mu][nu],
								      proj_dir_gamma[proj_type[pr]][dir][mu][nu]));
			   }
			 } //end mu nu                                                                                                                  
		     }//end j - parallelized
		     
		     for(j=0; j<geo.lV3; j++) { //loop over points                                                                                      
		       //accumulating j in threep_loc
		       threep_loc_f[pr][s][fl]=qcd_CADD(threep_loc_f[pr][s][fl], block_f[j]);
		       threep_loc_b[pr][s][fl]=qcd_CADD(threep_loc_b[pr][s][fl], block_b[j]);
		     }		       
		   }//end fl pr

		 if(s>0) {
		   qcd_shiftGaugeM( &u_temp, &u_shift_f, dir); //instead of P (l=l+)                                                                    
		   qcd_copyGaugeField( &u_shift_f, &u_temp);
		 }

#pragma omp parallel for private(j) 
		 for(j=0; j<geo.lV; j++) {
		   qcd_complex_16 tmp_3x3[3][3];
		   qcd_MUL3x3(tmp_3x3, u_prod_f.D[j][dir], u_shift_f.D[j][dir]); //u_shift_f is a copy of u, old x new                                  
		   qcd_copy3x3( u_prod_f.D[j][dir], tmp_3x3);
		 }

		 qcd_shiftGaugeP(&u_temp, &u_shift_b, dir);  //u_shift_b is a copy of u, l= l-                                                          
		 qcd_copyGaugeField(&u_shift_b, &u_temp);

#pragma omp parallel for private(j) 
		 for(j=0; j<geo.lV; j++) {
		   qcd_complex_16 tmp_3x3[3][3];
		   qcd_MULADJOINT3x3(tmp_3x3,u_prod_b.D[j][dir],u_shift_b.D[j][dir]);
		   qcd_copy3x3(u_prod_b.D[j][dir], tmp_3x3 );
		 }

		 for(fl=0; fl<2; fl++){
		   qcd_shiftPropagatorM(&prop_temp, &prop_shift_f[fl], dir ); //l=l+                                                                    
		   qcd_copyPropagatorPropagator( &prop_shift_f[fl], &prop_temp );

		   qcd_shiftPropagatorP( &prop_temp, &prop_shift_b[fl], dir); //l=l-                                                                    
		   qcd_copyPropagatorPropagator( &prop_shift_b[fl], &prop_temp);
		 }
	       } //end s 

	       for(pr=0; pr<nprojs; pr++)
		 for(s=0; s<geo.L[1]/2; s++) {
		   MPI_Allreduce(&(threep_loc_f[pr][s][0].re), &(threep_f[t-t_start][pr][s][0].re),
				 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		   MPI_Allreduce(&(threep_loc_b[pr][s][0].re), &(threep_b[t-t_start][pr][s][0].re),
				 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		 }
	     } //end lt inside local lattice                                                                                                            

	     if(myid==0) printf("time-slice-blocks calculated, t=%d \n",t);
	     if(myid==0) printf("Result:\n");

	     for(pr=0; pr<nprojs; pr++)
	       for(s=0; s<geo.L[1]/2; s++)
		 if(myid==0) printf("  Proj%d:  %d %d %+e %+e %+e %+e %+e %+e %+e %+e\n", proj_type[pr], t, s,
				    threep_f[t-t_start][pr][s][0].re, threep_f[t-t_start][pr][s][0].im,
				    threep_b[t-t_start][pr][s][0].re, threep_b[t-t_start][pr][s][0].im,
				    threep_f[t-t_start][pr][s][1].re, threep_f[t-t_start][pr][s][1].im,
				    threep_b[t-t_start][pr][s][1].re, threep_b[t-t_start][pr][s][1].im);

	   } //end t  

	 if(myid==0) printf("Writing results on files.\n");

	 if(myid==0)
	   for(pr=0; pr<nprojs; pr++) {
	     strcpy(tmp_string, threep_name);
	     sprintf(tmp_s, "%+d_%+d_%+d", mom[1], mom[2], mom[3]);
	     replace_str(tmp_string, "%mom", tmp_s);
	     replace_str(tmp_string, "%dir", ((dir==1)?"x":((dir==2)?"y":"z")));
	     sprintf(tmp_s, "%d", proj_type[pr]);
	     replace_str(tmp_string, "%proj", tmp_s);
	     sprintf(tmp_s, "%.4f", zeta);
             replace_str(tmp_string, "%zeta", tmp_s);
	     printf("opening %s \n",tmp_string);
	     pfile=fopen(tmp_string,"w");
	     if(pfile==NULL)
	       printf("failed to open %s for writing\n",threep_name[pr]);
	     else
	       for(t=t_start; t<=t_stop; t++) {    //write up and down parts of threeps                                                               
		 for(s=0; s<geo.L[1]/2; s++)
		   if(myid==0) fprintf(pfile, "%d %d %+e %+e %+e %+e %+e %+e %+e %+e\n", t, s,
				       threep_f[t-t_start][pr][s][0].re, threep_f[t-t_start][pr][s][0].im,
				       threep_b[t-t_start][pr][s][0].re, threep_b[t-t_start][pr][s][0].im,
				       threep_f[t-t_start][pr][s][1].re, threep_f[t-t_start][pr][s][1].im,
				       threep_b[t-t_start][pr][s][1].re, threep_b[t-t_start][pr][s][1].im);
	       }
	     fclose(pfile);
	   }//end pr 

	 ///////////////////////////////////////////////////////////////////////////////////////
#ifdef THREEP_SKIP_MOM_0
	 }
#endif

   }//for mom_loop    
   ////////////////////////////////////// CLEAN UP AND EXIT ///////////////////////////////////////////
   if(myid==0) printf("cleaning up...\n");
   for(fl=0;fl<2;fl++){
     qcd_destroyPropagator(&(prop[fl]));
     qcd_destroyPropagator(&(prop_pb[fl]));
   }

   /**** for threep *****/
   for(t=t_start; t<=t_stop; t++) {  //global threep                                                             
     for( pr=0; pr<nprojs; pr++){
       free(threep_f[t-t_start][pr]);
       free(threep_b[t-t_start][pr]);
     }
     free(threep_f[t-t_start]);
     free(threep_b[t-t_start]);
   }
   free(threep_f);
   free(threep_b);

   for( pr=0; pr<nprojs; pr++) {   //local threep                                                                
     free(threep_loc_f[pr]);
     free(threep_loc_b[pr]);
   }
   free(threep_loc_f);
   free(threep_loc_b);

   free(block_f);
   free(block_b);

   for(pr=0; pr<nprojs; pr++)
     for(fl=0;fl<2;fl++)
       qcd_destroyPropagator(&(prop_seq[pr][fl])); //seq-props                                                   

   free(prop_seq);

   for(fl=0;fl<2;fl++){
     qcd_destroyPropagator(&(prop_ms[fl]));
     qcd_destroyPropagator(&(prop_shift_f[fl]));
     qcd_destroyPropagator(&(prop_shift_b[fl]));
   }
   qcd_destroyPropagator(&(prop_temp));  //destroy propagators (threep) 
     
   qcd_destroyVector(&vec); 
   qcd_destroyVector(&vec_mg);

   
   qcd_destroyGaugeField(&u_shift_f);
   qcd_destroyGaugeField(&u_shift_b);
   qcd_destroyGaugeField(&u_prod_f);
   qcd_destroyGaugeField(&u_prod_b);
   qcd_destroyGaugeField(&u_ms);
   qcd_destroyGaugeField(&uAPE); 
   qcd_destroyGaugeField(&uAPE2);
   qcd_destroyGaugeField(&u);

   
   if(myid==0) printf("end of the program");
#ifdef MG4QCD
   MG4QCD_finalize();
#endif
   qcd_destroyGeometry(&geo);
   MPI_Finalize();
   return(EXIT_SUCCESS);
}//end main


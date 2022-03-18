#include <stdio.h>
#include <string>
#include <cmath>
#include <math.h>
#include <vector>
#include <cstring> 
#include <stdlib.h>
#define PI 3.1415926
void rdm_gauss(double& a1, double& rdmg)
{
    double PI = 0.0;
    double PI2 = 0.0;
    double asd = 0.0;
    double r01 = 0.0;
    double r02 = 0.0;
    double a1 = 0.0;
    double rq = 0.0;

    //PI = atan(1d0) * 4d0;
	PI2 = 2 * PI;
	rq = pow(PI, (8 + PI)) + PI;
  
	asd = a1;
	asd = asd * rq;
	asd = asd-floor(asd);  //The largest integer not greater than x
	r01 = sqrt( - 2 * log(asd));
	asd = asd * rq;
	asd = asd-floor(asd);
	r02 = PI2 * asd;
		
	rdmg = r01 * cos(r02);
	a1 = asd;

    return;
}


//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//       maxZernike
//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
int MAX_ZERNIKE(int maxZnkOrder, int maxZnkDim)
{
    int l = 0;
    int ni = 0;
    int m0 = 0;
    int m1 = 0;
    int m = 0;

    for(ni = 1; ni <= maxZnkOrder; ni++)
    {
        m0 = ni - 2 * ni / 2;
    	m1 = m0 + 2 * ni / 2;
        for(m = m0; m <= m1; m = m+2)
        {
            l = l + 1;
	        if(m != 0)
            {
                l = l + 1;
            }
        }
    }

    maxZnkDim = l;
    return maxZnkDim;
}


//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//      maxZernike
//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
int maxZernike(int nk)
{
    int l = 0;
    int ni = 0;
    int m0 = 0;
    int m1 = 0;
    int m = 0;

    for(ni = 1; ni <= nk; ni++)
    {
        m0 = ni - 2 * ni / 2;
    	m1 = m0 + 2 * ni / 2; //???
        for( m = m0; m <= m1 ; m = m+2)
        {
            l = l+1;
            if(m != 0)
            {
                l = l+1;
            }
        }
    }
    maxzernike = l;
	//write(11, *)nk, maxzernike ///????
    return maxzernike;

}


void nmlznk(int maxZnkOrder,int &maxZnkDim,int* &nznk,int* &mznk,int* &lznk)
{
	int maxZernike = 0;
	double fac1 = 0;
    double fac2 = 0;
    double fac3 = 0;
    double fac4 = 0;
	int l = 0;
    int ni = 0;
    int m0 = 0;
    int m1 = 0;
    int m = 0;
    int lz = 0;
    int j = 0;
    int nm = 0;
    int lznk_a(int l, int ni);
    for(ni = 1; ni <= maxZnkOrder; ni++)
    {
        m0 = ni - 2 * (ni / 2);
		m1 = m0 + 2 * (ni / 2);
        for(m = m0; m <= m1; m = m+2)
        {
            //l = l + 1; 
            if(l >= 0)
            {
                nznk(l) = ni;
				mznk(l) = m;
				lznk(l) = lznk_a(l, ni);
            }
            if(m != 0)
            {
                l = l + 1;
                if(l >= 1)
                {
                    nznk(l) = ni;
					mznk(l) = m;
					lznk(l) = lznk_a(l, ni,);
					nm = (ni - m) / 2;                    
                }
            }
            if(l >= maxZnkDim)
            {
                maxZnkDim = l;
                goto here;
            }
            l = l + 1; 
        }
    }
    here;
    // ?? format(4(1x, i4), 30(1x, f12.6))
    return;
}

//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//'直到maxznk阶的Zernike polynomial的nznk和mznk						   c
//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
		
void zernikeNumber(int maxZnkOrder,int &maxZnkDim,int* &nznk,int* &mznk,int* &lznk,double* &fc)
{
    int maxZernike;
	double fac1 = 0.0;
    double fac2 = 0.0; 
    double fac3 = 0.0;
    double fac4 = 0.0;
	int l = 0;
    int ni = 0;
    int m0 = 0;
    int m1 = 0;
    int m = 0;
    int lz = 0;
    int j = 0; 
    int nm = 0;
    int lznk_a(int l, int ni);
    void calc_fac(int ni - j, double &fac1);
    void calc_fac(int (ni + m) / 2 - j, double &fac3);
    void calc_fac(int nm - j, double &fac4);

    l = 0;
    for(ni = 1; ni <= maxZnkOrder; ni++)
    {
        m0 = ni - 2 * ni / 2;
		m1 = m0 + 2 * ni / 2;  
        for( m = m0; m <= m1; m = m+2)
        {
            //l = l + 1;
             if(l >= 0)
            {
                nznk(l) = ni;
				mznk(l) = m;
				lznk(l) = lznk_a(l, ni);
            }
            if(m != 0)
            {
                l = l + 1;
                if(l >= 1)
                {
                    nznk(l) = ni;
					mznk(l) = m;
					lznk(l) = lznk_a(l, ni,);
					nm = (ni - m) / 2; 

                    for(j = 0; j <= nm; j++)
                    {
						calc_fac(ni - j, fac1);
						calc_fac(j, fac2);
						calc_fac((ni + m) / 2 - j, fac3);
						calc_fac(nm - j, fac4);
                        fc(j, l) = ( - 1) ** j * fac1 / (fac2 * fac3 * fac4);
                    }                   
                }   
            }
            if(l >= maxZnkDim)
            {
                maxZnkDim = l;
                goto here;
            }
            l = l + 1; 
        }  
    }
    here;
    //format(4(1x, i4), 30(1x, f12.6));
    return;
}

//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//       lznk_a
//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
int lznk_a(int l, int ni)
{
    int lz = -1;
    if((ni / 2) / 2 != (ni / 2) * 0.5)
    {
       if(l / 2 == l * 0.5)
       {
            lz = 0; 
       }
        else
        {
            lz = 1;
        }
    }
    else
    {
        if(l / 2 == l * 0.5))
        {
            lz = 1;
        }
        else
        {
            lz = 0;
        }
    }
    return lz;
}

//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//       calc_fac
//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//'fac(n) = n!
void calc_fac(int ni, double &fac)
{
    int i = 0;
	if(ni < 2)
    {
        fac = 1;
        return;
    }
    for(i = 2; i <= ni; i++)
    {
        fac = fac * i;
    }
    return;
}
	
//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
//'      由公式计算多项式,zernike_formula
//'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

void zernike_formula(int maxZnkOrder, int minZnkDim, int maxZnkDim, int *nznk, int *mznk, int *lznk, double *fc, double* &pl, double x, double y)
{
    int m = 0;
    int l = 0;
    int jj = 0;

    double c1 = 0.0;
    double y2 = 0.0;
    double x2 = 0,0;
    double r = 0.0;
    double tm = 0.0;
    double af = 0.0;
    double phl = 0.0;
    double co = 0.0;
    double si = 0.0;

    double *cot;
    double *sit;
    double *rn;

    c1 = sqrt(2.0) / 2.0;
	y2 = y * y;
	x2 = x * x;
	r = sqrt(x2 + y2);
	af = atan(y, x);
	rn(0) = 1;

    for(m = 1; m <= maxZnkOrder; m++)
    {
        tm = m * af;
	    cot(m) = cos(tm);
	    sit(m) = sin(tm);
		rn(m) = rn(m - 1) * r;
    }
    pl(1) = 1.0;
    for(l = minZnkDim;l <= minZnkDim;l++)
    {
        pl(l) = 0;
        phl = 0;
        for(jj = 0; jj <= (nznk(l) - mznk(l)) / 2; jj++)
        {
            phl = phl + fc(jj, l) * rn(nznk(l) - 2 * jj);
        }
        if(mznk(l) == 0)
        {
            pl(l) = phl;
		    pl(l) = pl(l) * sqrt(nznk(l) + 1.0);
        }
        else
        {
            co = cot(mznk(l));
		    si = sit(mznk(l));
            if(lznk(l) == 0)
            {
                pl(l) = phl * co;
            }
            else
            {
                pl(l) = phl * si; 
            }
     		pl(l) = pl(l) * sqrt(2 * (nznk(l) + 1.));
            //pl(l) = pl(l) * c1           
        }
    }
    free(cot, sit, rn);
    return;
}


zernike_cg(int N, double *zer, double x, double y)
{
    if(N > 104)
    {
        //write(*, *)'   #SUB# zernike_cg   使用数组越界!'
    }
    r = x * x + y * y;
	c2 = x * x - y * y;
	s2 = 2 * x * y;
    for(int ii = 1; ii <= N; ii++)
    {
        if(ii == 1)zer(1) = 2.0 * x												
            if(ii == 2)zer(2) = 2.0 * y;												
            if(ii == 3)zer(3) = sqrt(3.) * (2 * r - 1);
            if(ii == 4)zer(4) = sqrt(6.) * s2;											
            if(ii == 5)zer(5) = sqrt(6.) * c2;											
            if(ii == 6)zer(6) = sqrt(8.) * y * (3 * r - 2;									
            if(ii == 7)zer(7) = sqrt(8.) * x * (3 * r - 2);									
            c3 = x * c2 - y * s2;
		    s3 = y * c2 + x * s2;
            if(ii == 8)zer(8) = sqrt(8.) * s3;											
            if(ii == 9)zer(9) = sqrt(8.) * c3;										
            if(ii == 10)zer(10) = sqrt(5.) * (6 * r ** 2 - 6 * r + 1);							
            if(ii == 11)zer(11) = sqrt(10.) * (4 * r - 3) * c2;								
            if(ii == 12)zer(12) = sqrt(10.) * (4 * r - 3) * s2;								
            c4 = c2 * c2 - s2 * s2;
		    s4 = 2 * s2 * c2;
            if(ii == 13)zer(13) = sqrt(10.) * c4;										
            if(ii == 14)zer(14) = sqrt(10.) * s4;									
            if(ii == 15)zer(15) = sqrt(12.) * (10 * r ** 2 - 12 * r + 3) * x;						
            if(ii == 16)zer(16) = sqrt(12.) * (10 * r ** 2 - 12 * r + 3) * y;						
            if(ii == 17)zer(17) = sqrt(12.) * c3 * (5 * r - 4);								
            if(ii == 18)zer(18) = sqrt(12.) * s3 * (5 * r - 4);								
            c5 = c2 * c3 - s2 * s3;
		    s5 = s2 * c3 + c2 * s3;
            if(ii == 19)zer(19) = sqrt(12.) * c5;										
            if(ii == 20)zer(20) = sqrt(12.) * s5;										
            if(ii == 21)zer(21) = sqrt(7.) * (20 * r ** 3 - 30 * r ** 2 + 12 * r - 1);					
            if(ii == 22)zer(22) = sqrt(14.) * (15 * r ** 2 - 20 * r + 6) * s2;						
            if(ii == 23)zer(23) = sqrt(14.) * (15 * r ** 2 - 20 * r + 6) * c2;						
            if(ii == 24)zer(24) = sqrt(14.) * (6 * r - 5) * s4;								
            if(ii == 25)zer(25) = sqrt(14.) * (6 * r - 5) * c4;									
            c6 = c3 * c3 - s3 * s3;
		    s6 = 2 * s3 * c3;
            if(ii == 26)zer(26) = sqrt(14.) * s6;											
            if(ii == 27)zer(27) = sqrt(14.) * c6;											
            if(ii == 28)zer(28) = 4.0 * (35 * r ** 3 - 60 * r ** 2 + 30 * r - 4) * y;					
            if(ii == 29)zer(29) = 4.0 * (35 * r ** 3 - 60 * r ** 2 + 30 * r - 4) * x;						
            if(ii == 30)zer(30) = 4.0 * (21 * r ** 2 - 30 * r + 10) * s3;							
            if(ii == 31)zer(31) = 4.0 * (21 * r ** 2 - 30 * r + 10) * c3;							
            if(ii == 32)zer(32) = 4.0 * (7 * r - 6) * s5;										
            if(ii == 33)zer(33) = 4.0 * (7 * r - 6) * c5;
            //??? if(ii .eq. 34)zer(34) = 4.0 * s7;												
            c7 = c4 * c3 - s4 * s3;
		    s7 = s4 * c3 + c4 * s3;
            if(ii == 35)zer(35) = 4.0 * c7;											
            if(ii == 36)zer(36) = 3.0 * (70 * r ** 4 - 140 * r ** 3 + 90 * r ** 2 - 20 * r + 1);			
            if(ii == 37)zer(37) = sqrt(18.) * (56 * r ** 3 - 105 * r ** 2 + 60 * r - 10) * c2;				
            if(ii == 38)zer(38) = sqrt(18.) * (56 * r ** 3 - 105 * r ** 2 + 60 * r - 10) * s2;				
            if(ii == 39)zer(39) = sqrt(18.) * (28 * r ** 2 - 42 * r + 15) * c4	;					
            if(ii == 40)zer(40) = sqrt(18.) * (28 * r ** 2 - 42 * r + 15) * s4;						
            if(ii == 41)zer(41) = sqrt(18.) * (8 * r - 7) * c6;									
            if(ii == 42)zer(42) = sqrt(18.) * (8 * r - 7) * s6;									
            c8 = c4 * c4 - s4 * s4;
            s8 = 2 * s4 * c4;
            if(ii == 43)zer(43) = sqrt(18.) * c8;											
            if(ii == 44)zer(44) = sqrt(18.) * s8;										
            if(ii == 45)zer(45) = sqrt(20.) *\
			  (126 * r ** 4 - 280 * r ** 3 + 210 * r ** 2 - 60 * r + 5) * x;	
            if(ii == 46)zer(46) = sqrt(20.) * \                                     
			  (126 * r ** 4 - 280 * r ** 3 + 210 * r ** 2 - 60 * r + 5) * y	;
            if(ii == 47)zer(47) = sqrt(20.) * (84 * r ** 3 - 168 * r ** 2 + 105 * r - 20) * c3;		
            if(ii == 48)zer(48) = sqrt(20.) * (84 * r ** 3 - 168 * r ** 2 + 105 * r - 20) * s3;			
            if(ii == 49)zer(49) = sqrt(20.) * (36 * r ** 2 - 56 * r + 21) * c5;						
            if(ii == 50)zer(50) = sqrt(20.) * (36 * r ** 2 - 56 * r + 21) * s5;						
            if(ii == 51)zer(51) = sqrt(20.) * (9 * r - 8) * c7;								
            if(ii == 52)zer(52) = sqrt(20.) * (9 * r - 8) * s7;								
            c9 = c5 * c4 - s5 * s4;
		    s9 = s5 * c4 + c5 * s4;
            if(ii == 53)zer(53) = sqrt(20.) * c9;											
            if(ii == 54)zer(54) = sqrt(20.) * s9;											
            if(ii == 55)zer(55) = sqrt(11.) *\                                      
			  (252 * r ** 5 - 630 * r ** 4 + 560 * r ** 3 - 210 * r ** 2 + 30 * r - 1);	
            if(ii == 56)zer(56) = sqrt(22.) * \                                   
			  (210 * r ** 4 - 504 * r ** 3 + 420 * r ** 2 - 140 * r + 15) * s2;		
            if(ii == 57)zer(57) = sqrt(22.) * \                                     
		      (210 * r ** 4 - 504 * r ** 3 + 420 * r ** 2 - 140 * r + 15) * c2;		
            if(ii == 58)zer(58) = sqrt(22.) * \                                  
     		  (120 * r ** 3 - 252 * r ** 2 + 168 * r - 35) * s4	;			
            if(ii == 59)zer(59) = sqrt(22.) *\                                     
     		  (120 * r ** 3 - 252 * r ** 2 + 168 * r - 35) * c4;				
            if(ii == 60)zer(60) = sqrt(22.) * (45 * r ** 2 - 72 * r + 28) * s6;							
            if(ii == 61)zer(61) = sqrt(22.) * (45 * r ** 2 - 72 * r + 28) * c6;							
            if(ii == 62)zer(62) = sqrt(22.) * (10 * r - 9) * s8	;							
            if(ii == 63)zer(63) = sqrt(22.) * (10 * r - 9) * c8	;								
            c10 = c5 * c5 - s5 * s5;
		    s10 = 2 * s5 * c5;
            if(ii == 64)zer(64) = sqrt(22.) * s10;											
            if(ii == 65)zer(65) = sqrt(22.) * c10;									
            if(ii == 66)zer(66) = sqrt(24.) * \                                     
			  (462 * r ** 5 - 1260 * r ** 4 + 1260 * r ** 3 - 560 * r ** 2 + 105 * r - 6) * x;
            if(ii == 67)zer(67) = sqrt(24.) *  \                                    
			  (462 * r ** 5 - 1260 * r ** 4 + 1260 * r ** 3 - 560 * r ** 2 + 105 * r - 6) * y;
            if(ii == 68)zer(68) = sqrt(24.) *  \                                    
			  (330 * r ** 4 - 840 * r ** 3 + 756 * r ** 2 - 280 * r + 35) * s3;		
            if(ii == 69)zer(69) = sqrt(24.) * \                                     
			  (330 * r ** 4 - 840 * r ** 3 + 756 * r ** 2 - 280 * r + 35) * c3;		
            if(ii == 70)zer(70) = sqrt(24.) * \                                     
			  (165 * r ** 3 - 360 * r ** 2 + 252 * r - 56) * s5	;			
            if(ii == 71)zer(71) = sqrt(24.) * \                                     
			  (165 * r ** 3 - 360 * r ** 2 + 252 * r - 56) * c5;				
            if(ii == 72)zer(72) = sqrt(24.) * (55 * r ** 2 - 90 * r + 36) * s7;							
            if(ii == 73)zer(73) = sqrt(24.) * (55 * r ** 2 - 90 * r + 36) * c7;							
            if(ii == 74)zer(74) = sqrt(24.) * (11 * r - 10) * s9;								
            if(ii == 75)zer(75) = sqrt(24.) * (11 * r - 10) * c9;								
            c11 = c6 * c5 - s6 * s5;
            s11 = s6 * c5 + c6 * s5;
            if(ii == 76)zer(76) = sqrt(24.) * s11;											
            if(ii == 77)zer(77) = sqrt(24.) * c11;											
            if(ii == 78)zer(78) = sqrt(13.) *  \                                    
			(924 * r ** 6 - 2772 * r ** 5 + 3150 * r ** 4 - 1680 * r ** 3 + 420 * r ** 2 - 42 * r + 1);
            if(ii == 79)zer(79) = sqrt(26.) *  \                                    
			  (792 * r ** 5 - 2310 * r ** 4 + 2520 * r ** 3 - 1260 * r ** 2 + 280 * r - 21) * c2;
            if(ii == 80)zer(80) = sqrt(26.) *  \                                    
			  (792 * r ** 5 - 2310 * r ** 4 + 2520 * r ** 3 - 1260 * r ** 2 + 280 * r - 21) * s2;
            if(ii == 81)zer(81) = sqrt(26.) *  \                                    
			  (495 * r ** 4 - 1320 * r ** 3 + 1260 * r ** 2 - 504 * r + 70) * c4;	
            if(ii == 82)zer(82) = sqrt(26.) *   \                                   
			  (495 * r ** 4 - 1320 * r ** 3 + 1260 * r ** 2 - 504 * r + 70) * s4;	
            if(ii == 83)zer(83) = sqrt(26.) *  \                                    
			  (220 * r ** 3 - 495 * r ** 2 + 360 * r - 84) * c6	;			
            if(ii == 84)zer(84) = sqrt(26.) * \                                     
			  (220 * r ** 3 - 495 * r ** 2 + 360 * r - 84) * s6	;			
            if(ii == 85)zer(85) = sqrt(26.) * (66 * r ** 2 - 110 * r + 45) * c8;						
            if(ii == 86)zer(86) = sqrt(26.) * (66 * r ** 2 - 110 * r + 45) * s8	;					
            if(ii == 87)zer(87) = sqrt(26.) * (12 * r - 11) * c10;								
            if(ii == 88)zer(88) = sqrt(26.) * (12 * r - 11) * s10;								
            c12 = c6 * c6 - s6 * s6;
		    s12 = 2 * s6 * c6;
            if(ii == 89)zer(89) = sqrt(26.) * c12;											
            if(ii == 90)zer(90) = sqrt(26.) * s12;										
            if(ii == 91)zer(91) = sqrt(28.) * \                                     
		(1716 * r ** 6 - 5544 * r ** 5 + 6930 * r ** 4 - 4200 * r ** 3 + 1260 * r ** 2 - 168 * r + 7) * x;
            if(ii == 92)zer(92) = sqrt(28.) * \                                    
		(1716 * r ** 6 - 5544 * r ** 5 + 6930 * r ** 4 - 4200 * r ** 3 + 1260 * r ** 2 - 168 * r + 7) * y;
            if(ii == 93)zer(93) = sqrt(28.) *   \                                   
			  (1287 * r ** 5 - 3960 * r ** 4 + 4620 * r ** 3 - 2520 * r ** 2 + 630 * r - 56) * c3;
            if(ii == 94)zer(94) = sqrt(28.) *  \                                    
			  (1287 * r ** 5 - 3960 * r ** 4 + 4620 * r ** 3 - 2520 * r ** 2 + 630 * r - 56) * s3;
            if(ii == 95)zer(95) = sqrt(28.) * \                                     
			  (715 * r ** 4 - 1980 * r ** 3 + 1980 * r ** 2 - 840 * r + 126) * c5;	
            if(ii == 96)zer(96) = sqrt(28.) *  \                                    
			  (715 * r ** 4 - 1980 * r ** 3 + 1980 * r ** 2 - 840 * r + 126) * s5;	
            if(ii == 97)zer(97) = sqrt(28.) * \                                     
			  (286 * r ** 3 - 660 * r ** 2 + 495 * r - 120) * c7;				
            if(ii == 98)zer(98) = sqrt(28.) *  \                                    
			  (286 * r ** 3 - 660 * r ** 2 + 495 * r - 120) * s7;				
            if(ii == 99)zer(99) = sqrt(28.) * (78 * r ** 2 - 132 * r + 55) * c9;						
            if(ii == 100)zer(100) = sqrt(28.) * (78 * r ** 2 - 132 * r + 55) * s9;					
            if(ii == 101)zer(101) = sqrt(28.) * (13 * r - 12) * c11;								
            if(ii == 102)zer(102) = sqrt(28.) * (13 * r - 12) * s11;								
            c13 = c7 * c6 - s7 * s6;
		    s13 = s7 * c6 + c7 * s6;
            if(ii == 103)zer(103) = sqrt(28.) * c13;										
            if(ii == 104)zer(104) = sqrt(28.) * s13;															
          
    }
      
   


}
#ifndef KOKKOS_MULTI_BSPLINE_H
#define KOKKOS_MULTI_BSPLINE_H
#include <Kokkos_Core.hpp>
#include <vector>
#include <assert.h>

// at first crack just use LayoutLeft
// later will want to experiment with a tiled layout with
// a static tile size (32, 64 etc)

// could imagine a specialization for complex types
// that split the coefficients into real and imaginary parts
namespace qmcplusplus
{

template<typename p>
void doEval_v(p x, p y, p z, Kokoks::View<p*> vals, Kokkos::View<p****> coefs,
	      Kokkos::View<double[3]> gridStarts, Kokkos::View<double[3]>delta_invs,
	      Kokkos::View<p[16]> A44, int blockSize = 32);

template<typename p>
void doEval_vgh(p x, p y, p z, Kokoks::View<p*> vals, Kokkos::View<p*[3]> grad,
		Kokkos::View<p*[6]> hess, Kokkos::View<p****> coefs,
		Kokkos::View<double[3]> gridStarts, Kokkos::View<double[3]>delta_invs,
		Kokkos::View<p[16]> A44, Kokkos::View<p[16], dA44,
		Kokkos::View<p[16]> d2A44, int blockSize = 32);



template<precision p, dims d>
struct multi_UBspline_base {
  // for bc_codes, 0 == periodic, 1 == deriv1,   2 == deriv2
  //               3 == flat    , 4 == natural,  5 == antiperiodic
  Kokkos::View<int[d]> left_bc_codes;
  Kokkos::View<int[d]> right_bc_codes;
  Kokkos::View<p[d]> left_values;
  Kokkos::View<p[d]> right_values;
  Kokoks::View<double[d]> gridStarts;
  Kokoks::View<double[d]> gridEnds;
  Kokkos::View<double[d]> deltas;
  Kokkos::View<double[d]> delta_invs;

protected:
  void initialize_base(std::vector<double> start_locs,
		       std::vector<double> end_locs,
		       std::vector<int> num_pts,
		       int boundary_condition_code) {
    assert(start_locs.size() == dims);
    assert(end_locs.size() == dims);
    
    spline.left_bc_codes = Kokkos::View<int[dims]>("left_bc_codes");
    spline.right_bc_codes = Kokkos::View<int[dims]>("right_bc_codes");
    spline.left_values = Kokkos::View<p[dims]>("left_values");
    spline.right_values = Kokkos::View<p[dims]>("right_values");
    spline.gridStarts = Kokkos::View<double[dims]>("gridStarts");
    spline.gridEnds = Kokkos::View<double[dims]>("gridEnds");
    spline.deltas = Kokkos::View<double[dims]>("deltas");
    spline.delta_invs = Kokkos::View<double[dims]>("delta_invs");
    
    auto lbcMirror = Kokkos::create_mirror_view(spline.left_bc_codes);
    auto rbcMirror = Kokkos::create_mirror_view(spline.right_bc_codes);
    auto gsMirror = Kokkos::create_mirror_view(spline.gridStarts);
    auto geMirror = Kokkos::create_mirror_view(spline.gridEnds);
    auto gdMirror = Kokkos::create_mirror_view(spline.deltas);
    auto gdinvMirror = Kokkos::create_mirror_view(spline.delta_invs);
    
    std::vector<int> nx;  
    for (int i = 0; i < dims; i++) {
      lbcMirror(i) = boundary_condition_code;
      rbcMirror(i) = boundary_condition_code;
      gsMirror(i) = start_loc[i];
      geMirror(i) = end_loc[i];
      
      nx.push_back(0);
      // if periodic or antiperiodic
      if (boundary_condition_code == 0 || boundary_condition_code == 5) {
	nx[i] = num_pts[i] + 3;
      } else {
	nx[i] = num_pts[i] + 2;
      }
      double delta = (end_locs[i] - start_locs[i]) / static_cast<double>(nx[i]-3);
      gdMirror(i) = delta;
      gdinvMirror(i) = 1.0 / delta;
    }
    
    Kokkos::deep_copy(spline.left_bc_codes, lbcMirror);
    Kokkos::deep_copy(spline.right_bc_codes, rbcMirror);
    Kokkos::deep_copy(spline.gridStarts, gsMirror);
    Kokkos::deep_copy(spline.gridEnds, geMirror);
    Kokkos::deep_copy(spline.deltas, gdMirror);
    Kokkos::deep_copy(spline.delta_invs, gdinvMirror);
  }
};  


template<precision p, int blocksize, int d>
struct multi_UBspline : public multi_UBspline_base<p,d> {};

template<precision p, int blocksize>
struct multi_UBspline<p, blocksize, 1> : public multi_UBspline_base<p,1> {
  using layout = Kokkos::LayoutRight;
  using coef_t = Kokkos::View<p**, layout>; // x, splines
  coef_t coef;

  using single_coef_t = decltype(Kokkos::subview(coef, Kokkos::ALL(), 0));
  using single_coef_mirror_t = single_coef_t::HostMirror;
  single_coef_t single_coef;
  single_coef_mirror_t single_coef_mirror;

  void initialize(std::vector<int>& dvec, std::vector<double>& start,
		  std::vector<double>& end, int bcCode, int numSpo) {
    initializeBase(start, end, dvec, 0);
    initializeCoefs(dvec, numSpo);
  }
  
  void setSingleCoef(int i) {
    assert(i >= 0 && i < coef.extent(1));
    single_coef = Kokkos::subview(coef, Kokkos::ALL(), i);
    single_coef_mirror = Kokkos::create_mirror_view(single_coef);
  }
  void pushCoefToDevice() {
    Kokkos::deep_copy(single_coef, single_coef_mirrror);
  }
 private:
  void initializeCoefs(std::vector<int>& dvec, int numSpo) {
    assert(dvec.size() == 1);
    std::vector<int> nx(dvec.begin(), dvec.end());
    
    auto lbcMirror = Kokkos::create_mirror_view(left_bc_codes);
    Kokkos::deep_copy(lbcMirror, left_bc_codes);
    for (int i = 0; i < 1; i++) {
      if (lbcMirror(i) == 0 || lbcMirror(i) == 5) {
	sz[i] += 3;
      } else {
	sz[i] += 2;
      }
    }
    coefs = coef_t("coefs", sz[0], numSpo);
  }

};

template<precision p, int blocksize>
struct multi_UBspline<p, blocksize, 2> : public multi_UBspline_base<p,2> {
  using layout = Kokkos::LayoutRight;
  using coef_t = Kokkos::View<p***, layout>; // x, y, splines
  coef_t coef;

  using single_coef_t = decltype(Kokkos::subview(coef, Kokkos::ALL(), Kokkos::ALL(), 0));
  using single_coef_mirror_t = single_coef_t::HostMirror;
  single_coef_t single_coef;
  single_coef_mirror_t single_coef_mirror;

  void initialize(std::vector<int>& dvec, std::vector<double>& start,
		  std::vector<double>& end, int bcCode, int numSpo) {
    initializeBase(start, end, dvec, 0);
    initializeCoefs(dvec, numSpo);
  }
    
  void setSingleCoef(int i) {
    assert(i >= 0 && i < coef.extent(2));
    single_coef = Kokkos::subview(coef, Kokkos::ALL(), Kokkos::ALL(), i);
    single_coef_mirror = Kokkos::create_mirror_view(single_coef);
  }

  void pushCoefToDevice() {
    Kokkos::deep_copy(single_coef, single_coef_mirrror);
  }
 private:
  void initializeCoefs(std::vector<int>& dvec, int numSpo) {
    assert(dvec.size() == 2);
    std::vector<int> nx(dvec.begin(), dvec.end());
    
    auto lbcMirror = Kokkos::create_mirror_view(left_bc_codes);
    Kokkos::deep_copy(lbcMirror, left_bc_codes);
    for (int i = 0; i < 2; i++) {
      if (lbcMirror(i) == 0 || lbcMirror(i) == 5) {
	sz[i] += 3;
      } else {
	sz[i] += 2;
      }
    }    
    coefs = coef_t("coefs", nx[0], nx[1], numSpo);
  }

};

template<precision p, int blocksize>
struct multi_UBspline<p, blocksize, 3> : public multi_UBspline_base<p,3> {
  using layout = Kokkos::LayoutRight;
  using coef_t = Kokkos::View<p****, layout>; // x, y, z, splines
  coef_t coef;

  using single_coef_t = decltype(Kokkos::subview(coef, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0));
  using single_coef_mirror_t = single_coef_t::HostMirror;
  single_coef_t single_coef;
  single_coef_mirror_t single_coef_mirror;

  Kokkos::View<p[16]> A44;
  Kokkos::View<p[16]> dA44;
  Kokkos::View<p[16]> d2A44;

  void initialize(std::vector<int>& dvec, std::vector<double>& start,
		  std::vector<double>& end, int bcCode, int numSpo) {
    initializeBase(start, end, dvec, 0);
    initializeCoefs(dvec, numSpo);
  }

  void setSingleCoef(int i) {
    assert(i >= 0 && i < coef.extent(3));
    single_coef = Kokkos::subview(coef, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), i);
    single_coef_mirror = Kokkos::create_mirror_view(single_coef);
  }

  void pushCoefToDevice() {
    Kokkos::deep_copy(single_coef, single_coef_mirrror);
  }

  void evaluate_v(p x, p y, p z, Kokkos::View<p*> vals) {
    assert(vals.extent(0) == coefs.extent(3));
    doEval_v(x, y, z, vals, coefs, gridStarts, delta_invs, A44, blocksize);
  }

  void evaluate_vgh(p x, p y, p z, Kokkos::View<p*> vals, Kokkos::View<p*[3]> grad,
		    Kokkos::View<p*[6]> hess) {
    assert(vals.extent(0) == coefs.extent(3));
    doEval_vgh(x, y, z, vals, grad, hess, coefs, gridStarts, delta_invs, A44, dA44, d2A44, blocksize);
  }

 private:
  void initializeCoefs(std::vector<int>& dvec, int numSpo) {
    assert(dvec.size() == 3);
    std::vector<int> nx(dvec.begin(), dvec.end());

    auto lbcMirror = Kokkos::create_mirror_view(left_bc_codes);
    Kokkos::deep_copy(lbcMirror, left_bc_codes);
    for (int i = 0; i < 3; i++) {
      if (lbcMirror(i) == 0 || lbcMirror(i) == 5) {
	sz[i] += 3;
      } else {
	sz[i] += 2;
      }
    }    
    coefs = coef_t("coefs", sz[0], sz[1], sz[2], numSpo);
    initializeAs();
  }

  void initializeAs() {
    A44 = Kokkos::View<p[16]>("A44");
    A44Mirror = Kokkos::create_mirror_view(A44);
    dA44 = Kokkos::View<p[16]>("dA44");
    dA44Mirror = Kokkos::create_mirror_view(dA44);
    d2A44 = Kokkos::View<p[16]>("d2A44");
    d2A44Mirror = Kokkos::create_mirror_view(d2A44);
    
    p ta[16] = {
      -1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0, 3.0 / 6.0, -6.0 / 6.0,
      0.0 / 6.0,  4.0 / 6.0, -3.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, 1.0 / 6.0,
      1.0 / 6.0,  0.0 / 6.0, 0.0 / 6.0,  0.0 / 6.0};
    p tda[16] = {
      0.0, -0.5, 1.0, -0.5, 0.0, 1.5, -2.0, 0.0,
      0.0, -1.5, 1.0, 0.5,  0.0, 0.5, 0.0,  0.0};
    p td2a[16] = {
      0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 3.0, -2.0,
      0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    
    for (int i = 0; i < 16; i++) {
      A44Mirror(i) = ta[i];
      dA44Mirror(i) = tda[i];
      d2A44Mirror(i) = td2a[i];
    }
    Kokkos::deep_copy(A44, A44Mirror);
    Kokkos::deep_copy(dA44, dA44Mirror);
    Kokkos::deep_copy(d2A44, d2A44Mirror);
  }
};

template<typename viewType, typename T>
KOKKOS_INLINE_FUNCTION void compute_prefactors(Kokkos::Array<T,4>& a, T tx, viewType A44) {
  a[0] = ((A44(0) * tx + A44(1)) * tx + A44(2)) * tx + A44(3);
  a[1] = ((A44(4) * tx + A44(5)) * tx + A44(6)) * tx + A44(7);
  a[2] = ((A44(8) * tx + A44(9)) * tx + A44(10)) * tx + A44(11);
  a[3] = ((A44(12) * tx + A44(13)) * tx + A44(14)) * tx + A44(15);
}

// would rather this be a member, but not sure how to 
template<typename viewType, typename T>
KOKKOS_INLINE_FUNCTION void compute_prefactors(Kokkos::Array<T,4>& a, Kokkos::Arrray<T,4>& da,
					       Kokkos::Array<T,4>& d2a, T tx,
					       viewType A44, viewType dA44, viewType d2A44) {
  a[0]   = ((A44(0) * tx + A44(1)) * tx + A44(2)) * tx + A44(3);
  a[1]   = ((A44(4) * tx + A44(5)) * tx + A44(6)) * tx + A44(7);
  a[2]   = ((A44(8) * tx + A44(9)) * tx + A44(10)) * tx + A44(11);
  a[3]   = ((A44(12) * tx + A44(13)) * tx + A44(14)) * tx + A44(15);
  da[0]  = ((dA44(0) * tx + dA44(1)) * tx + dA44(2)) * tx + dA44(3);
  da[1]  = ((dA44(4) * tx + dA44(5)) * tx + dA44(6)) * tx + dA44(7);
  da[2]  = ((dA44(8) * tx + dA44(9)) * tx + dA44(10)) * tx + dA44(11);
  da[3]  = ((dA44(12) * tx + dA44(13)) * tx + dA44(14)) * tx + dA44(15);
  d2a[0] = ((d2A44(0) * tx + d2A44(1)) * tx + d2A44(2)) * tx + d2A44(3);
  d2a[1] = ((d2A44(4) * tx + d2A44(5)) * tx + d2A44(6)) * tx + d2A44(7);
  d2a[2] = ((d2A44(8) * tx + d2A44(9)) * tx + d2A44(10)) * tx + d2A44(11);
  d2a[3] = ((d2A44(12) * tx + d2A44(13)) * tx + d2A44(14)) * tx + d2A44(15);
}

#define MYMAX(a, b) (a < b ? b : a)
#define MYMIN(a, b) (a > b ? b : a)
template<typename T>
KOKKOS_INLINE_FUNCTION void get(T x, T& dx, int& ind, int ng) {
  T ipart;
  dx  = std::modf(x, &ipart);
  ind = MYMIN(MYMAX(int(0), static_cast<int>(ipart)), ng);
}
#undef MYMAX
#undef MYMIN

template<typename p>
void doEval_v(p x, p y, p z, Kokoks::View<p*> vals, Kokkos::View<p****> coefs,
	      Kokkos::View<double[3]> gridStarts, Kokkos::View<double[3]>delta_invs,
	      Kokkos::View<p[16]> A44, int blockSize) {
  int numBlocks = coefs.extent(4) / blockSize;
  if (coefs.extent(4) % blockSize != 0) {
    numBlocks++;
  }

  Kokkos::Array<p,3> loc;
  loc[0] = x;
  loc[1] = y;
  loc[2] = z;
  
  Kokkos::TeamPolicy<> policy(numBlocks,1,32);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamPolicy<>::member_type member) {
      const int start = blockSize * member.league_rank();
      int end = start + blockSize;

      if (end > coefs.extent(4)) {
	end = coefs.extent(4);
      }
      const int num_splines = end-start;

      for (int i = 0; i < 3; i++) {
	loc[i] -= gridStarts(i);
      }
      Kokkos::Array<p,3> ts;
      Kokkos::Array<int,3> is;
      for (int i = 0; i < 3; i++) {
	get(loc[i] * delta_invs[i], ts[i], is[i], coefs.extent(i)-1);
      }

      Kokkos::Array<p,4> a,b,c;
      compute_prefactors(a, ts[0], A44);
      compute_prefactors(b, ts[1], A44);
      compute_prefactors(c, ts[2], A44);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, num_splines),
			   [&](const int& i) { vals(start+i) = p(); });

      for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	  const p pre00 = a[i] * b[j];
	  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, num_splines), [&](const int& n) {
	      vals[start+n] += pre00 * 
		(c[0] * coefs(is[0]+i,is[1]+j,is[2],start+n) +
		 c[1] * coefs(is[0]+i,is[1]+j,is[2]+1,start+n) +
		 c[2] * coefs(is[0]+i,is[1]+j,is[2]+2,start+n) +
		 c[3] * coefs(is[0]+i,is[1]+j,is[2]+3,start+n));
	    });
	}
      }
    });      
}

template<typename p>
void doEval_vgh(p x, p y, p z, Kokoks::View<p*> vals, Kokkos::View<p*[3]> grad,
		Kokkos::View<p*[6]> hess, Kokkos::View<p****> coefs,
		Kokkos::View<double[3]> gridStarts, Kokkos::View<double[3]>delta_invs,
		Kokkos::View<p[16]> A44, Kokkos::View<p[16], dA44,
		Kokkos::View<p[16]> d2A44, int blockSize = 32) {
  int numBlocks = coefs.extent(4) / blockSize;
  if (coefs.extent(4) % blockSize != 0) {
    numBlocks++;
  }
  
  Kokkos::Array<p,3> loc;
  loc[0] = x;
  loc[1] = y;
  loc[2] = z;
  
  Kokkos::TeamPolicy<> policy(numBlocks,1,32);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(TeamPolicy<>::member_type member) {
      const int start = blockSize * member.league_rank();
      int end = start + blockSize;
      
      if (end > coefs.extent(4)) {
	end = coefs.extent(4);
      }
      const int num_splines = end-start;

      for (int i = 0; i < 3; i++) {
	loc[i] -= gridStarts(i);
      }
      Kokkos::Array<p,3> ts;
      Kokkos::Array<int,3> is;
      for (int i = 0; i < 3; i++) {
	get(loc[i] * delta_invs[i], ts[i], is[i], coefs.extent(i)-1);
      }

      Kokkos::Array<p,4> a,b,c, da, db, dc, d2a, d2b, d2c;
      compute_prefactors(a, da, d2a, ts[0], A44, dA44, d2A44);
      compute_prefactors(b, db, d2b, ts[1], A44, dA44, d2A44);
      compute_prefactors(c, dc, d2c, ts[2], A44, dA44, d2A44);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, num_splines), [&](const int& i) {
	  vals(start+i) = p();
	  grad(start+i,0) = p();
	  grad(start+i,1) = p();
	  grad(start+i,2) = p();
	  hess(start+i,0) = p();
	  hess(start+i,1) = p();
	  hess(start+i,2) = p();
	  hess(start+i,3) = p();
	  hess(start+i,4) = p();
	  hess(start+i,5) = p();
	});

      for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	  const p pre20 = d2a(i) * b(j);
	  const p pre10 = da(i) * b(j);
	  const p pre00 = a(i) * b(j);
	  const p pre11 = da(i) * db(j);
	  const p pre01 = a(i) * db(j);
	  const p pre02 = a(i) * d2b(j);

	  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, num_splines), [&](const int& n) {
	      const p sum0 = c(0) * coefs(is(0)+i, is(1)+j, is(2), start+n) +
		c(1) * coefs(is(0)+i, is(1)+j, is(2)+1, start+n) +
		c(2) * coefs(is(0)+i, is(1)+j, is(2)+2, start+n) +
		c(3) * coefs(is(0)+i, is(1)+j, is(2)+3, start+n);
	      const p sum1 = dc(0) * coefs(is(0)+i, is(1)+j, is(2), start+n) +
		dc(1) * coefs(is(0)+i, is(1)+j, is(2)+1, start+n) +
		dc(2) * coefs(is(0)+i, is(1)+j, is(2)+2, start+n) +
		dc(3) * coefs(is(0)+i, is(1)+j, is(2)+3, start+n);
	      const p sum2 = d2c(0) * coefs(is(0)+i, is(1)+j, is(2), start+n) +
		d2c(1) * coefs(is(0)+i, is(1)+j, is(2)+1, start+n) +
		d2c(2) * coefs(is(0)+i, is(1)+j, is(2)+2, start+n) +
		d2c(3) * coefs(is(0)+i, is(1)+j, is(2)+3, start+n);
	      
	      hess(start+n,0) += pre20 * sum0;
	      hess(start+n,1) += pre11 * sum0;
	      hess(start+n,2) += pre10 * sum1;
	      hess(start+n,3) += pre02 * sum0;
	      hess(start+n,4) += pre01 * sum1;
	      hess(start+n,5) += pre00 * sum2;
	      grad(start+n,0) += pre10 * sum0;
	      grad(start+n,1) += pre01 * sum0;
	      grad(start+n,2) += pre00 * sum1;
	      vals(start+n) += pre00 * sum0;
	    });
	}
      }

      const p dxx = delta_invs(0) * delta_invs(0);
      const p dyy = delta_invs(1) * delta_invs(1);
      const p dzz = delta_invs(2) * delta_invs(2);
      const p dxy = delta_invs(0) * delta_invs(1);
      const p dxz = delta_invs(0) * delta_invs(2);
      const p dyz = delta_invs(1) * delta_invs(2);
      
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, num_splines), [&](const int& n) {
	  grad(start+n,0) *= delta_invs(0);
	  grad(start+n,1) *= delta_invs(1);
	  grad(start+n,2) *= delta_invs(2);
	  hess(start+n,0) *= dxx;
	  hess(start+n,1) *= dyy;
	  hess(start+n,2) *= dzz;
	  hess(start+n,3) *= dxy;
	  hess(start+n,4) *= dxz;
	  hess(start+n,5) *= dyz;
	});
    });
 }
    
};    

#endif
  
  

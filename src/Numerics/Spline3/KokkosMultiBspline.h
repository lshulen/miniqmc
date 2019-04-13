#ifndef KOKKOS_MULTI_BSPLINE_H
#define KOKKOS_MULTI_BSPLINE_H
#include <Kokkos_Core.hpp>

// at first crack just use LayoutLeft
// later will want to experiment with a tiled layout with
// a static tile size (32, 64 etc)

// would also be nice to template this on number of dimensions

// could imagine a specialization for complex types
// that split the coefficients into real and imaginary parts
template<precision p, dims d>
  struct multi_UBspline {};

template<precision p>
struct multi_UBspline<p, 1> {
  using coef_t = Kokkos::View<p**, Kokkos::LayoutRight>; // x, splines
  coef_t coef;
  // for bc_codes, 0 == periodic, 1 == deriv1,   2 == deriv2
  //               3 == flat    , 4 == natural,  5 == antiperiodic
  Kokkos::View<int[1]> left_bc_codes;
  Kokkos::View<int[1]> right_bc_codes;
  Kokkos::View<p[1]> left_values;
  Kokkos::View<p[1]> right_values;
  Kokoks::View<double[1]> gridStarts;
  Kokoks::View<double[1]> gridEnds;
  Kokkos::View<double[1]> deltas;
  Kokkos::View<double[1]> delta_invs;
  
  KOKKOS_INLINE_FUNCTION
  multi_UBspline() { ; }
  
  
  KOKKOS_INLINE_FUNCTION
  multi_UBspline* operator=(const multi_UBspline& rhs) {
    coef = rhs.coef;    
    left_bc_codes = rhs.left_bc_codes;
    right_bc_codes = rhs.right_bc_codes;
    left_values = rhs.left_values;
    right_values = rhs.right_values;
    gridStarts = rhs.gridStarts;
    gridEnds = rhs.gridEnds;
    deltas = rhs.deltas;
    delta_invs = rhs.delta_invs;
    return this;
  }
  
  multi_UBspline(const multi_UBspline&) = default;
};

template<precision p>
struct multi_UBspline<p, 2> {
  using coef_t = Kokkos::View<p***, Kokkos::LayoutRight>; // x, y, splines
  coef_t coef;
  // for bc_codes, 0 == periodic, 1 == deriv1,   2 == deriv2
  //               3 == flat    , 4 == natural,  5 == antiperiodic
  Kokkos::View<int[2]> left_bc_codes;
  Kokkos::View<int[2]> right_bc_codes;
  Kokkos::View<p[2]> left_values;
  Kokkos::View<p[2]> right_values;
  Kokoks::View<double[2]> gridStarts;
  Kokoks::View<double[2]> gridEnds;
  Kokkos::View<double[2]> deltas;
  Kokkos::View<double[2]> delta_invs;
  
  KOKKOS_INLINE_FUNCTION
  multi_UBspline() { ; }
  
  
  KOKKOS_INLINE_FUNCTION
  multi_UBspline* operator=(const multi_UBspline& rhs) {
    coef = rhs.coef;    
    left_bc_codes = rhs.left_bc_codes;
    right_bc_codes = rhs.right_bc_codes;
    left_values = rhs.left_values;
    right_values = rhs.right_values;
    gridStarts = rhs.gridStarts;
    gridEnds = rhs.gridEnds;
    deltas = rhs.deltas;
    delta_invs = rhs.delta_invs;
    return this;
  }
  
  multi_UBspline(const multi_UBspline&) = default;
};

template<precision p>
struct multi_UBspline<p, 3> {
  using coef_t = Kokkos::View<p****, Kokkos::LayoutRight>; // x, y, z, splines
  coef_t coef;
  // for bc_codes, 0 == periodic, 1 == deriv1,   2 == deriv2
  //               3 == flat    , 4 == natural,  5 == antiperiodic
  Kokkos::View<int[3]> left_bc_codes;
  Kokkos::View<int[3]> right_bc_codes;
  Kokkos::View<p[3]> left_values;
  Kokkos::View<p[3]> right_values;
  Kokoks::View<double[3]> gridStarts;
  Kokoks::View<double[3]> gridEnds;
  Kokkos::View<double[3]> deltas;
  Kokkos::View<double[3]> delta_invs;
  
  KOKKOS_INLINE_FUNCTION
  multi_UBspline() { ; }
  
  
  KOKKOS_INLINE_FUNCTION
  multi_UBspline* operator=(const multi_UBspline& rhs) {
    coef = rhs.coef;    
    left_bc_codes = rhs.left_bc_codes;
    right_bc_codes = rhs.right_bc_codes;
    left_values = rhs.left_values;
    right_values = rhs.right_values;
    gridStarts = rhs.gridStarts;
    gridEnds = rhs.gridEnds;
    deltas = rhs.deltas;
    delta_invs = rhs.delta_invs;
    return this;
  }
  
  multi_UBspline(const multi_UBspline&) = default;
};


template<precision p, int dims>
void initializeCoefMemory(multi_UBspline<p, dims>& spline , std::vector<int>& d, int numSpo);

template<precision p>
void initializeCoefMemory(multi_UBspline<p, 1>& spline, std::vector<int>& d, int numSpo) {
  spline.coefs = Kokkos::View<p**, Kokkos::LayoutRight>("coefs", d[0], numSpo);
}

template<precision p>
void initializeCoefMemory(multi_UBspline<p, 2>& spline, std::vector<int>& d, int numSpo) {
  spline.coefs = Kokkos::View<p***, Kokkos::LayoutRight>("coefs", d[0], d[1], numSpo);
}

template<precision p>
void initializeCoefMemory(multi_UBspline<p, 1>& spline, std::vector<int>& d, int numSpo) {
  spline.coefs = Kokkos::View<p****, Kokkos::LayoutRight>("coefs", d[0], d[1], d[2], numSpo);
}

// would be nice to template on number of dimensions
template<precision p, int dims>
void create_multi_UBspline(multi_UBspline<p, dims>& spline,
			   std::vector<double> start_locs,
			   std::vector<double> end_locs,
			   std::vector<int> num_pts,
			   int boundary_condition_code,
			   int num_spos) {
  spline.left_bc_codes = Kokkos::View<int[3]>("left_bc_codes");
  spline.right_bc_codes = Kokkos::View<int[3]>("right_bc_codes");
  spline.left_values = Kokkos::View<p[3]>("left_values");
  spline.right_values = Kokkos::View<p[3]>("right_values");
  spline.gridStarts = Kokkos::View<double[3]>("gridStarts");
  spline.gridEnds = Kokkos::View<double[3]>("gridEnds");
  spline.deltas = Kokkos::View<double[3]>("deltas");
  spline.delta_invs = Kokkos::View<double[3]>("delta_invs");

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

  initializeCoefMemory(spline, nx, num_spos);
}

template<typename T, dim d>
void setCoefsForOneOrbital(multi_UBspline& spline, int orbNum, T* orbCoefs) {
  auto devCoefs = 


#endif
  
  

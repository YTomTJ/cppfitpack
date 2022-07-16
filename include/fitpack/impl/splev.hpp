#pragma once
#include "./fitpack.h"

namespace fitpack {

    struct splev_param {
        /*  t : array
            the position of the knots
        */
        Eigen::VectorXd t = {};
        /*  c : array
            the b-spline coefficients
        */
        Eigen::MatrixXd c = {};
        /*  k : int
            giving the degree of s(x)
        */
        int k = -1; // -1=undefined
        /*  e : int
            if 0 the spline is extrapolated from the end spans
                for points not in the support,
            if 1 the spline evaluates to zero for those points,
            if 2 ier is set to 1 and the subroutine returns,
            if 3 the spline evaluates to the value of the
                nearest boundary point.
        */
        int e = 0;
        /*  nu : int,
            optional The order of derivative of the spline to compute
            (must be less than or equal to k)
        */
        int nu = 0;
        /*  verbose : int
            if 0 no output
            if 1 only output the error
            if 2 output all the infomation
        */
        int verbose = 0;
    };

    struct splev_result {
        Eigen::MatrixXd y = {};
    };

    splev_result splev(Eigen::VectorXd x, splev_param pp = splev_param())
    {
        static const std::map<int, std::string> errmsg = {
            { 0, "normal" },
            { 1, "argument out of bounds and e == 2." },
            { 10, "invalid input data (see restrictions)" },
        };

        // For N-D B-spline
        if (pp.c.cols() > 1) {
            splev_result sp;
            sp.y = Eigen::MatrixXd::Zero(x.rows(), pp.c.cols());
            for (int i = 0; i < pp.c.cols(); ++i) {
                splev_param _pp;
                _pp.t = pp.t;
                _pp.c = pp.c.col(i);
                _pp.k = pp.k;
                _pp.e = pp.e;
                _pp.nu = pp.nu;

                auto _sp = splev(x, _pp);
                sp.y.col(i) = _sp.y;
            }
            return sp;
        }

        int n = (int)pp.t.rows();
        int m = (int)x.rows();

        splev_result res;
        res.y = Eigen::VectorXd::Zero(m);
        std::vector<double> wrk(n, 0);

        int ier;
        if (pp.nu) {
            SPLDER(pp.t.data(), &n, pp.c.data(), &pp.k, &pp.nu, x.data(), res.y.data(), &m, &pp.e,
                wrk.data(), &ier);
        } else {
            SPLEV(pp.t.data(), &n, pp.c.data(), &pp.k, x.data(), res.y.data(), &m, &pp.e, &ier);
        }

        CHECK_ERROR(ier);
        return res;
    }

} // namespace fitpack

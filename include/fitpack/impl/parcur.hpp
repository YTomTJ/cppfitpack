#pragma once
#include "./fitpack.h"

namespace fitpack {

    struct parcur_param {
        /*  iopt : int, optional
            If iopt==0 (default), find t and c for a given smoothing factor, s.
            If iopt==1, find t and c for another value of the smoothing factor, s.
            There must have been a previous call with iopt=0 or iopt=1
            for the same set of data.
            If iopt=-1 find the weighted least square spline for a given set of
            knots, t.
        */
        int iopt = 0;
        /*  ipar : int, optional
            On entry ipar must specify whether (ipar=1) the user will supply
            the parameter values u(i),ub and ue or whether (ipar=0) these values
            are to be calculated by parcur. unchanged on exit.
        */
        int ipar = 0;
        /*  u : array_like, optional
            An array of parameter values. If not given, these values are
            calculated automatically as ``M = len(x[0])``, where
                v[0] = 0
                v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)
                u[i] = v[i] / v[M-1]
        */
        Eigen::VectorXd u = {};
        /*  ub, ue : int, optional
            The end-points of the parameters interval.  Defaults to
            u[0] and u[-1].
        */
        double ub = 0.0;
        double ue = 1.0;
        /*  k : int, optional
            Degree of the spline. Cubic splines are recommended.
            Even values of `k` should be avoided especially with a small s-value.
            ``1 <= k <= 5``, default is 3.
        */
        int k = 3;
        /*  s : float, optional
            A smoothing condition.  The amount of smoothness is determined by
            satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
            where g(x) is the smoothed interpolation of (x,y).  The user can
            use `s` to control the trade-off between closeness and smoothness
            of fit.  Larger `s` means more smoothing while smaller values of `s`
            indicate less smoothing. Recommended values of `s` depend on the
            weights, w.  If the weights represent the inverse of the
            standard-deviation of y, then a good `s` value should be found in
            the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
            data points in x, y, and w.
        */
        double s = -1; // -1=undefined
        /*  nest : int, optional
            An over-estimate of the total number of knots of the spline to
            help in determining the storage space.  By default nest=m/2.
            Always large enough is nest=m+k+1.
        */
        int nest = -1; // -1=undefined
        /*  w : array_like, optional
            Strictly positive rank-1 array of weights the same length as `x[0]`.
            The weights are used in computing the weighted least-squares spline
            fit. If the errors in the `x` values have standard-deviation given by
            the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
        */
        Eigen::VectorXd w = {};
        /*  verbose : int
            if 0 no output
            if 1 only output the error
            if 2 output all the infomation
        */
        int verbose = 0;
    };

    struct parcur_result {
        Eigen::VectorXd t = {};
        Eigen::MatrixXd c = {};
        Eigen::MatrixXd u = {};
        int k;
        double fp;
    };

    parcur_result parcur(Eigen::MatrixXd x, parcur_param pp = parcur_param())
    {
        static const std::map<int, std::string> errmsg = {
            { 0, "The spline has a residual sum of squares fp such that abs(fp-s)/s<=0.001" },
            { -1, "The spline is an interpolating spline (fp=0)" },
            { -2,
                "The spline is weighted least-squares polynomial of degree k.\nfp gives the "
                "upper bound fp0 for the smoothing factor s" },
            { 1,
                "The required storage space exceeds the available storage space.\nProbable "
                "causes,data (x,y) size is too small or smoothing parameter\ns is too small "
                "(fp>s)." },
            { 2,
                "A theoretically impossible result when finding a smoothing spline\nwith fp "
                "= s. Probable cause,s too small. (abs(fp-s)/s>0.001)" },
            { 3,
                "The maximal number of iterations (20) allowed for finding smoothing\nspline "
                "with fp=s has been reached. Probable cause,s too "
                "small.\n(abs(fp-s)/s>0.001)" },
            { 10, "Error on input data" },
        };

        int m = (int)x.rows(); // m > k
        int mx = (int)x.size();
        int idim = (int)x.cols();

        if (pp.w.size() <= 0) {
            pp.w = Eigen::VectorXd::Ones(m);
        }
        if (pp.u.size() <= 0) {
            pp.u = Eigen::VectorXd::Zero(m);
            pp.ipar = 0;
        }
        if (pp.s < 0) {
            pp.s = m - std::sqrt(2.0 * m);
        }
        if (pp.nest <= 0) {
            pp.nest = m + 2 * pp.k;
        }

        int nc = idim * pp.nest;
        int lwrk = m * (pp.k + 1) + pp.nest * (6 + idim + 3 * pp.k);

        std::vector<double> t(pp.nest, 0);
        std::vector<double> c(nc, 0);
        std::vector<double> wrk(lwrk, 0);
        std::vector<int> iwrk(pp.nest, 0);

        int n, ier;
        parcur_result res;
        x = x.transpose().eval(); // fitpack need row-major

        PARCUR(&pp.iopt, &pp.ipar, &idim, &m, pp.u.data(), &mx, x.data(), pp.w.data(), &pp.ub,
            &pp.ue, &pp.k, &pp.s, &pp.nest, &n, t.data(), &nc, c.data(), &res.fp, wrk.data(), &lwrk,
            iwrk.data(), &ier);

        t.resize(n);
        c.resize(idim * n);

        CHECK_ERROR(ier);

        res.c = Eigen::Map<Eigen::MatrixXd>(c.data(), c.size() / idim, idim);
        res.c = res.c(Eigen::seqN(0, res.c.rows() - pp.k - 1), Eigen::all).eval();
        res.t = Eigen::Map<Eigen::VectorXd>(t.data(), n);
        res.u = pp.u;
        res.k = pp.k;
        return res;
    }

} // namespace fitpack

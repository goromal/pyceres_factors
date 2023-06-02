import pytest
import numpy as np
from geometry import SO2, SE2, SO3, SE3
import PyCeres as ceres
import PyCeresFactors as factors
RSEED = 144440

class TestFactors:
    def test_so3factor(self):
        np.random.seed(RSEED)
        q = SO3.random()
        xhat = SO3.identity().array()
        problem = ceres.Problem()
        problem.AddParameterBlock(xhat, 4, factors.SO3Parameterization())
        problem.AddResidualBlock(factors.SO3Factor(q.array(), np.eye(3)), None, xhat)
        options = ceres.SolverOptions()
        options.max_num_iterations = 25
        options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False
        summary = ceres.Summary()
        ceres.Solve(options, problem, summary)
        assert np.allclose(xhat, q.array())

    def test_relse3factor(self):
        np.random.seed(RSEED)
        T0 = SE3.identity()
        wij = np.random.random(6)
        T1 = T0 + wij
        Tij = SE3.Exp(wij)
        x = [T0.array(), T1.array()]
        xhat = [SE3.identity().array(), SE3.identity().array()]
        problem = ceres.Problem()
        problem.AddParameterBlock(xhat[0], 7, factors.SE3Parameterization())
        problem.SetParameterBlockConstant(xhat[0])
        problem.AddParameterBlock(xhat[1], 7, factors.SE3Parameterization())
        problem.AddResidualBlock(factors.RelSE3Factor(Tij.array(), np.eye(6)), None, xhat[0], xhat[1])
        options = ceres.SolverOptions()
        options.max_num_iterations = 25
        options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False
        summary = ceres.Summary()
        ceres.Solve(options, problem, summary)
        assert np.allclose(x[0], xhat[0])
        assert np.allclose(x[1], xhat[1])

    def test_tsattfactor(self):
        np.random.seed(RSEED)
        Q = np.eye(3)
        qref = SO3.random()
        w = np.array([0.5,1.0,-2.0])
        dt_true = np.array([0.2])
        dt_hat = np.array([0.0])
        q = qref + (-dt_true * w)
        problem = ceres.Problem()
        problem.AddParameterBlock(dt_hat, 1)
        problem.AddResidualBlock(factors.TimeSyncAttFactor(qref.array(), q.array(), w, Q), None, dt_hat)
        options = ceres.SolverOptions()
        options.max_num_iterations = 25
        options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False
        summary = ceres.Summary()
        ceres.Solve(options, problem, summary)
        assert np.allclose(dt_true, dt_hat)

    def test_so3offsetfactor(self):
        np.random.seed(RSEED)
        Q = np.eye(3)
        q = SO3.random()
        q_off = SO3.random()
        q_ref = q * q_off
        q_off_hat = SO3.identity().array()
        problem = ceres.Problem()
        problem.AddParameterBlock(q_off_hat, 4, factors.SO3Parameterization())
        problem.AddResidualBlock(factors.SO3OffsetFactor(q_ref.array(), q.array(), Q),
                                 None, q_off_hat)
        options = ceres.SolverOptions()
        options.max_num_iterations = 25
        options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False
        summary = ceres.Summary()
        ceres.Solve(options, problem, summary)
        assert np.allclose(q_off_hat, q_off.array())

    def test_se3offsetfactor(self):
        np.random.seed(RSEED)
        Q = np.eye(6)
        T = SE3.random()
        T_off = SE3.random()
        T_ref = T * T_off
        T_off_hat = SE3.identity().array()
        problem = ceres.Problem()
        problem.AddParameterBlock(T_off_hat, 7, factors.SE3Parameterization())
        problem.AddResidualBlock(factors.SE3OffsetFactor(T_ref.array(), T.array(), Q),
                                 None, T_off_hat)
        options = ceres.SolverOptions()
        options.max_num_iterations = 25
        options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False
        summary = ceres.Summary()
        ceres.Solve(options, problem, summary)
        assert np.allclose(T_off_hat, T_off.array())

    def test_rangebearing2dfactor(self):
        np.random.seed(RSEED)
        sigma_d = 0.02
        sigma_theta = 0.02
        l = np.array([10.0, -5.0])
        x = SE2.identity()
        v = np.array([1.0, 0.0, 0.5])
        dt = 0.1
        R_err_hat = SO2.random().array()
        l_hat = np.random.random(2)

        problem = ceres.Problem()
        problem.AddParameterBlock(R_err_hat, 2, factors.SO2Parameterization())
        problem.AddParameterBlock(l_hat, 2)

        for _ in range(100):
            d = np.linalg.norm(x.t() - l)
            l_B = x.inverse() * l
            theta = SO2.fromTwoUnitVectors(np.array([1.,0.]), l_B / d).angle()
            p = x.t()
            phi = x.q().angle()
            problem.AddResidualBlock(factors.RangeBearing2DFactor(d, sigma_d, theta, sigma_theta, p, phi),
                                     None, l_hat, R_err_hat)
            x += v * dt

        options = ceres.SolverOptions()
        options.max_num_iterations = 25
        options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False
        summary = ceres.Summary()
        ceres.Solve(options, problem, summary)

        assert np.allclose(l_hat, l)
        assert abs(SO2.fromComplex(R_err_hat).angle()) < 1e-8

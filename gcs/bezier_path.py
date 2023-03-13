import numpy as np
import pydot
import time
from scipy.optimize import root_scalar

from pydrake.geometry.optimization import (
    HPolyhedron,
    Point,
)
from pydrake.math import (
    BsplineBasis,
    BsplineBasis_,
    KnotVectorType,
)
from pydrake.solvers import(
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    QuadraticCost,
    PerspectiveQuadraticCost,
)
from pydrake.symbolic import (
    DecomposeLinearExpressions,
    Expression,
    MakeMatrixContinuousVariable,
    MakeVectorContinuousVariable,
)
from pydrake.trajectories import (
    BsplineTrajectory,
    BsplineTrajectory_,
    Trajectory,
)

from gcs.base import BaseGCS

class BezierPathGCS(BaseGCS):
    def __init__(self, regions, order, continuity, edges=None, full_dim_overlap=False):
        BaseGCS.__init__(self, regions)

        self.order = order
        self.continuity = continuity
        assert continuity < order

        self.time_scaling_set = HPolyhedron.MakeBox([0], [1e3])

        # Formulate edge costs and constraints
        self.r_control = MakeMatrixContinuousVariable(
            self.dimension, order + 1, "r")
        self.h_control = MakeVectorContinuousVariable(1, "h")

        self.r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            self.r_control)

        for i, r in enumerate(self.regions):
            self.gcs.AddVertex(
                r.CartesianPower(order + 1).CartesianProduct(self.time_scaling_set),
                name = self.names[i] if not self.names is None else '')

        # Add edges to graph and apply costs/constraints
        if edges is None:
            if full_dim_overlap:
                edges = self.findEdgesViaFullDimensionOverlaps()
            else:
                edges = self.findEdgesViaOverlaps()

        # Continuity constraints
        self.contin_constraints = []
        for deriv in range(continuity + 1):
            r_deriv = self.r_trajectory.MakeDerivative(deriv)
            A_r = DecomposeLinearExpressions(
                r_deriv.control_points()[0], self.r_control[:, :deriv+1].flatten("F"))
            r_continuity = LinearEqualityConstraint(
                np.hstack((-A_r, A_r)), np.zeros(self.dimension))
            self.contin_constraints.append(r_continuity)

        vertices = self.gcs.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.gcs.AddEdge(u, v, f"({u.name()}, {v.name()})")

            for deriv, c_con in enumerate(self.contin_constraints):
                edge.AddConstraint(Binding[Constraint](
                        c_con,
                        np.append(u.x()[-self.dimension*(deriv+1)-1:-1],
                                  v.x()[:self.dimension*(deriv+1)])
                    ))


    def addTimeCost(self, weight):
        assert isinstance(weight, float) or isinstance(weight, int)

        time_cost = LinearCost(np.array([weight]), 0.0)
        for v in self.gcs.Vertices():
            if v == self.source or v == self.target:
                continue
            v.AddCost(Binding[Cost](time_cost, v.x()[-1:]))

    def addPathLengthCost(self, weight):
        if isinstance(weight, float) or isinstance(weight, int):
            weight_matrix = weight * np.eye(self.dimension)
        else:
            assert(len(weight) == self.dimension)
            weight_matrix = np.diag(weight)

        A_diff = np.hstack((-np.eye(self.dimension), np.eye(self.dimension)))
        length_cost = L2NormCost(weight_matrix @ A_diff, np.zeros(self.dimension))
        for v in self.gcs.Vertices():
            if v == self.source or v == self.target:
                continue
            for ii in range(self.order):
                v.AddCost(Binding[Cost](
                    length_cost, v.x()[ii * self.dimension:(ii+2) * self.dimension]))

    def addDerivativeRegularization(self, weight, order):
        assert isinstance(order, int) and 2 <= order <= self.order
        assert isinstance(weight, float) or isinstance(weight, int)

        r_deriv_points = self.r_trajectory.MakeDerivative(order).control_points()
        A_r = DecomposeLinearExpressions(
            r_deriv_points[0], self.r_control[:, :order+1].flatten("F"))
        Q_r = A_r.T.dot(A_r) * 2 * weight / (1 + self.order - order)
        r_cost = QuadraticCost(Q_r, np.zeros(Q_r.shape[0]), 0)

        for v in self.gcs.Vertices():
            if v == self.source or v == self.target:
                continue
            for ii in range(self.order - order + 1):
                v.AddCost(Binding[Cost](
                    r_cost, v.x()[ii * self.dimension:(ii+order+1) * self.dimension]))

    def addVelocityLimits(self, lower_bound, upper_bound):
        assert len(lower_bound) == self.dimension
        assert len(upper_bound) == self.dimension

        A_dq = np.hstack((-self.order * np.eye(self.dimension),
                        self.order * np.eye(self.dimension)))
        b_dt = np.array([[1.0/self.order]])
        lb = np.expand_dims(lower_bound, 1)
        ub = np.expand_dims(upper_bound, 1)

        A_constraint = np.vstack((np.hstack((A_dq, -ub * b_dt)),
                                np.hstack((-A_dq, lb * b_dt))))
        velocity_constraint = LinearConstraint(
            A_constraint, np.full(2*self.dimension, -np.inf), np.zeros(2*self.dimension))
        for v in self.gcs.Vertices():
            if v == self.source or v == self.target:
                continue
            for ii in range(self.order):
                v.AddConstraint(Binding[Constraint](
                    velocity_constraint,
                    np.concatenate((v.x()[ii * self.dimension:(ii+2) * self.dimension],
                                    v.x()[-1:]))))

    def addSourceTarget(self, source, target, edges=None, velocity=None, zero_deriv_boundary=None):
        source_edges, target_edges =  super().addSourceTarget(source, target, edges)

        # if velocity is not None:
        #     assert velocity.shape == (2, self.dimension)

        #     u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        #     u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        #     initial_velocity_error = np.squeeze(u_path_control[0]) - velocity[0] * np.squeeze(u_time_control[0])
        #     final_velocity_error = np.squeeze(u_path_control[-1]) - velocity[1] * np.squeeze(u_time_control[-1])
        #     initial_velocity_con = LinearEqualityConstraint(
        #         DecomposeLinearExpressions(initial_velocity_error, self.u_vars),
        #         np.zeros(self.dimension))
        #     final_velocity_con = LinearEqualityConstraint(
        #         DecomposeLinearExpressions(final_velocity_error, self.u_vars),
        #         np.zeros(self.dimension))

        if zero_deriv_boundary is not None:
            assert self.order > zero_deriv_boundary + 1

        for edge in source_edges:
            for jj in range(self.dimension):
                edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])

            # if velocity is not None:
            #     edge.AddConstraint(Binding[Constraint](initial_velocity_con, edge.xv()))
            if zero_deriv_boundary is not None:
                for deriv in range(1, zero_deriv_boundary+1):
                    path_control = self.r_trajectory.MakeDerivative(deriv).control_points()
                    A_initial = DecomposeLinearExpressions(
                        np.squeeze(path_control[0]), self.r_control[:, :deriv+1].flatten("F"))
                    i_con = LinearEqualityConstraint(A_initial, np.zeros(self.dimension))
                    edge.v().AddConstraint(Binding[Constraint](i_con, edge.xv()[:(deriv+1) * self.dimension]))

        for edge in target_edges:    
            for jj in range(self.dimension):
                edge.AddConstraint(
                    edge.xu()[-(self.dimension + self.order + 1) + jj] == edge.xv()[jj])

            # if velocity is not None:
            #     edge.AddConstraint(Binding[Constraint](final_velocity_con, edge.xu()))
            if zero_deriv_boundary is not None:
                for deriv in range(1, zero_deriv_boundary+1):
                    path_control = self.r_trajectory.MakeDerivative(deriv).control_points()
                    A_initial = DecomposeLinearExpressions(
                        np.squeeze(path_control[0]), self.r_control[:, :deriv+1].flatten("F"))
                    i_con = LinearEqualityConstraint(A_initial, np.zeros(self.dimension))
                    edge.u().AddConstraint(Binding[Constraint](i_con, edge.xu()[-(deriv+1) * self.dimension-1:-1]))


    def SolvePath(self, rounding=False, verbose=False, preprocessing=False):
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose)

        if best_path is None:
            return None, results_dict

        # Extract trajectory control points
        knots = np.zeros(self.order + 1)
        path_control_points = []
        time_control_points = []
        for edge in best_path:
            if edge.v() == self.target:
                knots = np.concatenate((knots, [knots[-1]]))
                path_control_points.append(best_result.GetSolution(edge.xv()))
                time_control_points.append(np.array([best_result.GetSolution(edge.xu())[-1]]))
                break
            edge_time = knots[-1] + 1.
            knots = np.concatenate((knots, np.full(self.order, edge_time)))
            edge_path_points = np.reshape(best_result.GetSolution(edge.xv())[:-(self.order + 1)],
                                             (self.dimension, self.order + 1), "F")
            edge_time_points = best_result.GetSolution(edge.xv())[-(self.order + 1):]
            for ii in range(self.order):
                path_control_points.append(edge_path_points[:, ii])
                time_control_points.append(np.array([edge_time_points[ii]]))

        offset = time_control_points[0].copy()
        for ii in range(len(time_control_points)):
            time_control_points[ii] -= offset

        path_control_points = np.array(path_control_points).T
        time_control_points = np.array(time_control_points).T

        path = BsplineTrajectory(BsplineBasis(self.order + 1, knots), path_control_points)
        time_traj = BsplineTrajectory(BsplineBasis(self.order + 1, knots), time_control_points)

        return BezierTrajectory(path, time_traj), results_dict

class BezierTrajectory:
    def __init__(self, path_traj, time_traj):
        assert path_traj.start_time() == time_traj.start_time()
        assert path_traj.end_time() == time_traj.end_time()
        self.path_traj = path_traj
        self.time_traj = time_traj
        self.start_s = path_traj.start_time()
        self.end_s = path_traj.end_time()

    def invert_time_traj(self, t):
        if t <= self.start_time():
            return self.start_s
        if t >= self.end_time():
            return self.end_s
        error = lambda s: self.time_traj.value(s)[0, 0] - t
        res = root_scalar(error, bracket=[self.start_s, self.end_s])
        return np.min([np.max([res.root, self.start_s]), self.end_s])

    def value(self, t):
        return self.path_traj.value(self.invert_time_traj(np.squeeze(t)))

    def vector_values(self, times):
        s = [self.invert_time_traj(t) for t in np.squeeze(times)]
        return self.path_traj.vector_values(s)

    def EvalDerivative(self, t, derivative_order=1):
        if derivative_order == 0:
            return self.value(t)
        elif derivative_order == 1:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            r_dot = self.path_traj.EvalDerivative(s, 1)
            return r_dot * s_dot
        elif derivative_order == 2:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            h_ddot = self.time_traj.EvalDerivative(s, 2)[0, 0]
            s_ddot = -h_ddot*(s_dot**3)
            r_dot = self.path_traj.EvalDerivative(s, 1)
            r_ddot = self.path_traj.EvalDerivative(s, 2)
            return r_ddot * s_dot * s_dot + r_dot * s_ddot
        else:
            raise ValueError()


    def start_time(self):
        return self.time_traj.value(self.start_s)[0, 0]

    def end_time(self):
        return self.time_traj.value(self.end_s)[0, 0]

    def rows(self):
        return self.path_traj.rows()

    def cols(self):
        return self.path_traj.cols()

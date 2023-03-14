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

        self.time_scaling_set = HPolyhedron.MakeBox([1e-3], [1e3])

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
                        np.concatenate((u.x()[-self.dimension*(deriv+1)-1:-1],
                                        v.x()[:self.dimension*(deriv+1)]))
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
                    edge.xu()[-(self.dimension + 1) + jj] == edge.xv()[jj])

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
        for edge in best_path:
            if edge.v() == self.target:
                knots = np.concatenate((knots, [knots[-1]]))
                path_control_points.append(best_result.GetSolution(edge.xv()))
                break
            edge_time = knots[-1] + best_result.GetSolution(edge.xv())[-1]
            knots = np.concatenate((knots, np.full(self.order, edge_time)))
            edge_path_points = np.reshape(best_result.GetSolution(edge.xv())[:-1],
                                             (self.dimension, self.order + 1), "F")
            for ii in range(self.order):
                path_control_points.append(edge_path_points[:, ii])

        path_control_points = np.array(path_control_points).T

        path = BsplineTrajectory(BsplineBasis(self.order + 1, knots), path_control_points)

        return path, results_dict

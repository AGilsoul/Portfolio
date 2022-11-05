import numpy as np
import math

i_hat = np.array([1, 0, 0])
j_hat = np.array([0, 1, 0])
k_hat = np.array([0, 0, 1])


# Vector method class
class Vector:
    # return dot product of two vectors
    @staticmethod
    def dot_product(u, v):
        return sum([u[x] * v[x] for x in range(len(u))])

    # return vector orthogonal to two given vectors
    @staticmethod
    def cross_product(u, v):
        return i_hat * (u[1] * v[2] - u[2] * v[1]) - j_hat * (u[0] * v[2] - u[2] * v[0]) + k_hat * (
                    u[0] * v[1] - u[1] * v[0])

    # return magnitude of a vector
    @staticmethod
    def vec_magnitude(u):
        return math.sqrt(sum([x ** 2 for x in u]))

    # return unit vector of length 1 from vector u
    @staticmethod
    def unit_vec(u):
        magnitude = Vector.vec_magnitude(u)
        return np.array([x / magnitude for x in u])

    # return array of vector and orthogonal projection of v onto u
    @staticmethod
    def vector_projection(u, v):
        c = Vector.dot_product(u, v) / (Vector.vec_magnitude(u) ** 2)
        w = c * u
        e = v - w
        return w, e

    # return angle between two vectors in radians
    @staticmethod
    def angle_dif(u, v):
        return math.acos(Vector.dot_product(u, v)/(Vector.vec_magnitude(u)*Vector.vec_magnitude(v)))

    @staticmethod
    # returns angle from z axis, and rotational angle about z axis
    def angle_offset3d(u):
        x = u[0]
        y = u[1]
        theta = Vector.angle_dif(u, k_hat)
        if x == 0:
            if y > 0:
                phi = math.pi / 2
            else:
                phi = -math.pi / 2
        else:
            phi = math.atan(y / x)

        return [theta, phi]

    @staticmethod
    def angle_offset2d(u):
        theta = (math.pi/2) - Vector.angle_dif(u, j_hat)
        return theta

    @staticmethod
    def to_string2d(u):
        return '{}i + {}j'.format(u[0], u[1])

    @staticmethod
    def to_string3d(u):
        return '{}i + {}j + {}k'.format(u[0], u[1], u[2])


# Quaternion method class
class Quaternion:
    # axis = axis of rotation
    # a = cos(theta/2)
    # b = sin(theta/2) * axis[x]*i_hat
    # c = sin(theta/2) * axis[y]*j_hat
    # d = sin(theta/2) * axis[z]*k_hat
    # quaternion_1 = a + b + c + d
    # p = point to rotate
    # quaternion_2 = a - b - c - d
    # rotated_point = quaternion_1 * p * quaternion_2
    @staticmethod
    def rotate_deg(p, axis, theta, verbose=False):
        rot_axis = Vector.unit_vec(axis)
        half_radians = (theta * np.pi)/360
        sin_theta_half = np.sin(half_radians)
        a = np.cos(half_radians)
        b = sin_theta_half * rot_axis[0]
        c = sin_theta_half * rot_axis[1]
        d = sin_theta_half * rot_axis[2]

        if verbose:
            print(
                'q1 * p * q2 =\n({}) * ({}) * ({})\n'.format(
                    Quaternion.to_string([a, b, c, d]),
                    Vector.to_string3d(p),
                    Quaternion.to_string([a, -b, -c, -d])
                )
            )

        return Quaternion.rotate(p, [a, b, c, d], unit=True)

    @staticmethod
    def rotate_rad(p, axis, theta, verbose=False):
        rot_axis = Vector.unit_vec(axis)
        half_radians = theta/2
        sin_theta_half = np.sin(half_radians)
        a = np.cos(half_radians)
        b = sin_theta_half * rot_axis[0]
        c = sin_theta_half * rot_axis[1]
        d = sin_theta_half * rot_axis[2]

        if verbose:
            print(
                '({}) * ({}) * ({}) = \n'.format(
                    Quaternion.to_string([a, b, c, d]),
                    Vector.to_string3d(p),
                    Quaternion.to_string([a, -b, -c, -d])
                )
            )

        return Quaternion.rotate(p, [a, b, c, d], unit=True)

    # rotates point p with given quaternion q
    @staticmethod
    def rotate(p, q, unit=False):
        quaternion = np.array(q, dtype=float)
        if not unit:
            mag = Vector.vec_magnitude(q)
            quaternion *= (1.0/mag)

        a = quaternion[0]
        b = quaternion[1]
        c = quaternion[2]
        d = quaternion[3]

        # calculate repeated values here to eliminate repeated calculations
        a_2 = a ** 2
        a_b = a * b
        a_c = a * c
        a_d = a * d

        b_2 = b ** 2
        b_c = b * c
        b_d = b * d

        c_2 = c ** 2
        c_d = c * d

        d_2 = d ** 2

        # calculate values for rotated point
        rotated_p = np.empty([3], dtype=float)
        rotated_p[0] = a_2 * p[0] + b_2 * p[0] - c_2 * p[0] - d_2 * p[0] + 2 * (
                    a_c * p[2] - a_d * p[1] + b_c * p[1] + b_d * p[2])
        rotated_p[1] = a_2 * p[1] - b_2 * p[1] + c_2 * p[1] - d_2 * p[1] + 2 * (
                    -a_b * p[2] + a_d * p[0] + b_c * p[0] + c_d * p[2])
        rotated_p[2] = a_2 * p[2] - b_2 * p[2] - c_2 * p[2] + d_2 * p[2] + 2 * (
                    a_b * p[1] - a_c * p[0] + b_d * p[0] + c_d * p[1])

        return rotated_p

    @staticmethod
    def rotate_to_axis(p, axis):
        unit_axis = Vector.unit_vec(axis)
        mag_p = Vector.vec_magnitude(p)
        return mag_p * unit_axis

    @staticmethod
    def euler_to_quaternion_deg(angles):
        rad_angles = np.array(angles)
        rad_angles *= np.pi / 180
        return Quaternion.euler_to_quaternion_rad(rad_angles)

    @staticmethod
    def euler_to_quaternion_rad(angles):
        phi = angles[0]
        theta = angles[1]
        psi = angles[2]

        cos_phi_2 = np.cos(phi/2)
        sin_phi_2 = np.sin(phi/2)
        cos_theta_2 = np.cos(theta/2)
        sin_theta_2 = np.sin(theta/2)
        cos_psi_2 = np.cos(psi/2)
        sin_psi_2 = np.sin(psi/2)

        a = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2
        b = sin_phi_2 * cos_theta_2 * cos_psi_2 - cos_phi_2 * sin_theta_2 * sin_psi_2
        c = cos_phi_2 * sin_theta_2 * cos_psi_2 + sin_phi_2 * cos_theta_2 * sin_psi_2
        d = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * sin_theta_2 * cos_psi_2

        return np.array([a, b, c, d])

    @staticmethod
    # converts quaternion to euler angles in radians
    def quaternion_to_euler_(q):
        phi = np.arctan2(q[2]*q[3] + q[0]*q[1], (1/2) - (q[1]**2 + q[2]**2))
        theta = np.arcsin(-2*(q[1]*q[3] - q[0]*q[2]))
        if abs(theta) == np.pi/2:
            print('gimbal-lock')
        psi = np.arctan2(q[1]*q[2] + q[0]*q[3], (1/2) - (q[2])**2 + q[3]**2)

        return np.array([phi, theta, psi])

    @staticmethod
    def to_string(q):
        return '{} + {}i + {}j + {}k'.format(q[0], q[1], q[2], q[3])
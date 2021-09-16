import tensorflow as tf
from tensorflow import keras
import numpy as np


class QSP(keras.layers.Layer):
    """Parameterized quantum signal processing layer.

    The `QSP` layer implements the quantum signal processing circuit with trainable QSP angles.

    The input of the layer is/are theta(s) where x = cos(theta), and w(x) is X rotation in the QSP sequence.

    The output is the real part of the upper left element in the resulting unitary that describes the whole sequence.
    This is Re[P(x)] in the representation of the QSP unitary from Gilyen et al.

    Input is of the form:
    [[theta1], [theta2], ... ]

    Output is of the form:
    [[P(x1)], [P(x2)], ...]

    The layer requires the desired polynomial degree of P(x)

    """

    def __init__(self, poly_deg=0, convention=0):
        """
        Params
        ------
        poly_deg: The desired degree of the polynomial in the QSP sequence.
            the layer will be parameterized with poly_deg + 1 trainable phi.
        """
        super(QSP, self).__init__()
        self.poly_deg = poly_deg
        self.convention = convention
        phi_init = tf.random_uniform_initializer(minval=0, maxval=np.pi)
        self.phis = tf.Variable(
            initial_value=phi_init(shape=(poly_deg + 1, 1), dtype=tf.float32),
            trainable=True,
        )


    def call(self, th):
        batch_dim = tf.gather(tf.shape(th), 0)

        # tiled up X rotations (input W(x))
        px = tf.constant([[0.0, 1], [1, 0]], dtype=tf.complex64)
        px = tf.expand_dims(px, axis=0)
        px = tf.repeat(px, [batch_dim], axis=0)

        rot_x_arg = tf.complex(real=0.0, imag=th)
        rot_x_arg = tf.expand_dims(rot_x_arg, axis=1)
        rot_x_arg = tf.tile(rot_x_arg, [1, 2, 2])

        wx = tf.linalg.expm(tf.multiply(px, rot_x_arg))

        # tiled up Z rotations
        pz = tf.constant([[1.0, 0], [0, -1]], dtype=tf.complex64)
        pz = tf.expand_dims(pz, axis=0)
        pz = tf.repeat(pz, [batch_dim], axis=0)

        z_rotations = []
        for k in range(self.poly_deg + 1):
            phi = self.phis[k]
            rot_z_arg = tf.complex(real=0.0, imag=phi)
            rot_z_arg = tf.expand_dims(rot_z_arg, axis=0)
            rot_z_arg = tf.expand_dims(rot_z_arg, axis=0)
            rot_z_arg = tf.tile(rot_z_arg, [batch_dim, 2, 2])

            rz = tf.linalg.expm(tf.multiply(pz, rot_z_arg))
            z_rotations.append(rz)

        u = z_rotations[0]
        for rz in z_rotations[1:]:
            u = tf.matmul(u, wx)
            u = tf.matmul(u, rz)

        if self.convention == 0:
            # the |0><0| convention; real(p(x)) and imag(p(x))
            return tf.math.real(u[:, 0, 0]), tf.math.imag(u[:, 0, 0])
        elif self.convention == 1:
            # the |+><+| convention; real(p(x)) and real(Q(x)*sqrt(1-x^2))
            return tf.math.real(u[:, 0, 0]), tf.math.imag(u[:, 0, 1])


def mean_deviation(y_true, y_pred):
  deviations = tf.abs(tf.subtract(y_true, y_pred))
  loss = tf.math.reduce_mean(deviations)
  return loss


def max_deviation(y_true, y_pred):
  deviations = tf.abs(tf.subtract(y_true, y_pred))
  loss = tf.math.reduce_max(deviations)
  return loss


def mean_deviation_squared(y_true, y_pred):
  deviations = tf.abs(tf.subtract(y_true, y_pred))
  loss = tf.math.reduce_mean(tf.square(deviations))
  return loss


def max_deviation_squared(y_true, y_pred):
  deviations = tf.abs(tf.subtract(y_true, y_pred))
  loss = tf.math.reduce_max(tf.square(deviations))
  return loss


def construct_qsp_model(poly_deg, convention, lr, mean_or_max, squared):
    """Helper function that compiles a QSP model with mean squared error and adam optimizer.

    Params
    ------
    poly_deg : int
        the desired degree of the polynomial in the QSP sequence.

    Returns
    -------
    Keras model
        a compiled keras model with trainable phis in a poly_deg QSP sequence.
    """
    theta_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="theta")
    qsp = QSP(poly_deg, convention)
    real_and_imag_parts = tf.cast(qsp(theta_input), dtype=tf.complex64)
    real_and_imag_parts_added = real_and_imag_parts[0]+1j*real_and_imag_parts[1]
    model = tf.keras.Model(inputs=theta_input, outputs=real_and_imag_parts_added)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # loss = tf.keras.losses.MeanSquaredError()
    # loss = tf.keras.losses.MeanAbsoluteError()

    if mean_or_max == 0:
        if squared == 0:
            loss = mean_deviation
        elif squared == 1:
            loss = mean_deviation_squared
    elif mean_or_max ==1:
        if squared == 0:
            loss = max_deviation
        elif squared == 1:
            loss = max_deviation_squared

    model.compile(optimizer=optimizer, loss=loss)
    model.convention = convention
    return model


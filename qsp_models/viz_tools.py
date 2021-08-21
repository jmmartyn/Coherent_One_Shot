# visualization tools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from . import QSPCircuit 

def plot_qsp_response(f_real, f_imag, model, convention):
	"""Plot the QSP response against the desired function response.
	
	Params
	------
	f_real : function float --> float
		the desired function to be implemented by the QSP sequence
	f_imag : function float --> float
		the desired function to be implemented by the QSP sequence
	model : Keras `Model` with `QSP` layer
		model trained to approximate f
	"""
	all_th = np.arange(0, np.pi, np.pi / 300)

	# construct circuit
	phis = model.trainable_weights[0].numpy()
	qsp_circuit = QSPCircuit(phis)
	qsp_circuit.svg()
	circuit_px = qsp_circuit.eval_px(all_th)
	circuit_qx = qsp_circuit.eval_qx(all_th)
	qsp_response = qsp_circuit.qsp_response(all_th)

	if convention == 0: 	# |0><0| convention
		df = pd.DataFrame({"x": np.cos(all_th), "Real[p(x)]": np.real(circuit_px),
						   "imag[p(x)]": np.imag(circuit_px), "desired Real[f(x)]": f_real(np.cos(all_th)),
						   "desired Imag[f(x)]": f_imag(np.cos(all_th))})
		df = df.melt("x", var_name="src", value_name="value")
		# ax = df.plot()
		# ax.axvline(-0.5, color="red", linestyle="--")
		# ax.axvline(0.5, color="red", linestyle="--")
		sns.lineplot(x="x", y="value", hue="src", data=df).set_title("QSP Response")
		plt.show()
	elif convention == 1: 	# |+><+| convention
		df = pd.DataFrame({"x": np.cos(all_th), "Real[p(x)]": np.real(circuit_px),
			"Real[q(x)]sqrt(1-x^2)": np.real(circuit_qx), "desired Real[f(x)]": f_real(np.cos(all_th)),
			"desired Imag[f(x)]": f_imag(np.cos(all_th))})
		df = df.melt("x", var_name="src", value_name="value")
		#ax = df.plot()
		#ax.axvline(-0.5, color="red", linestyle="--")
		#ax.axvline(0.5, color="red", linestyle="--")
		sns.lineplot(x="x", y="value", hue="src", data=df).set_title("QSP Response")
		plt.show()


def plot_loss(history):
	"""Plot the error of a trained QSP model. 
		
	Params
	------
	history : tensorflow `History` object
	"""
	plt.plot(history.history['loss'])
	plt.title("Learning QSP Angles")
	plt.xlabel("Iterations")
	plt.ylabel("Error")
	plt.show()
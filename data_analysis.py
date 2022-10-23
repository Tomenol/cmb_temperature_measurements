import numpy as np
import matplotlib.pyplot as plt
import scipy

from tqdm import tqdm
import pickle

import pathlib
from os import path
import pickle
__root_path__ = pathlib.Path(__file__).resolve().parent


def get_data(name):
	""" Returns the data as a np data structure from the specified data file. """
	if name == "lobe":
		data_path = "Lobe1_20221020.lvm"
	elif name == "scan":
		data_path = "Skydip1_20221020.lvm"
	elif name == "calibration":
		data_path = "Etal1_20221020.lvm"
	else:
		raise("ERROR : could not find specified data file : ", name)
		return None

	return np.genfromtxt(str(__root_path__ / "data" / data_path), dtype=["float", "float", "float", "float", "float", "float", "float", "str"], names=['raw','az_nc','el_nc', 'az', 'el', 'aref', 'a', 'time'])

def plot_lobe(calibration_function):
	data = get_data("lobe")

	# plot lobes
	fig = plt.figure()
	ax = fig.add_subplot(111)

	azimuth_ordered = np.unique(np.sort(data['az'][70:534]))
	gain_pattern = np.zeros(len(azimuth_ordered))

	for i in range(len(azimuth_ordered)):  
		gain_pattern[i] = np.mean(data['raw'][70:534][data['az'][70:534] == azimuth_ordered[i]])

	G = calibration_function(gain_pattern)
	G = (G - np.min(G))/(np.max(G) - np.min(G))

	azimuth_ordered -= 133

	ax.plot(azimuth_ordered, G, "-r")

	ax.set_xlabel("Azimuth [deg]")
	ax.set_ylabel("Normalized \n gain pattern [-]")

def plot_scan(fcal):
	data = get_data("scan")

	# plot whole raw signal
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot((data['raw']), "-r", label="Signal")
	ax.plot((data['a']), '-g', label="Sky")
	ax.plot((data['aref']), '-k', label="System noise")

	ax.set_ylabel("Raw signal [V]")
	ax.set_xlabel("Point [-]")

	ax.legend()

	# plot raw signal only within scan range
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.set_ylabel("Raw signal [V]")
	ax.set_xlabel("Point [-]")

	ax.plot((data['raw'][10:7280]), "-r", label="Signal")
	ax.plot((data['a'][10:7280]), '-g', label="Sky")
	ax.plot((data['aref'][10:7280]), '-k', label="System noise")

	# plot calibrated temperature signal
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.set_ylabel("Calibrated \n temperature [K]")
	ax.set_xlabel("Point [-]")

	ax.plot(fcal(data['raw'][10:7280]), "-b")


	# plot azimuth and elevation
	fig = plt.figure()
	ax = fig.add_subplot(211)

	ax.set_ylabel("azimuth [deg]")
	ax.tick_params(labelbottom=False)

	ax.plot(data['az'][10:7280][data['az'][10:7280]>25], "-b")

	ax = fig.add_subplot(212)
	ax.set_ylabel("Elevation [deg]")
	ax.set_xlabel("Point [-]")

	ax.plot(data['el'][10:7280][data['el'][10:7280] > 15], "-b")

def calibration_scan():
	""" uses the end of the scan to calibrate the instrument. """
	data = get_data("scan")

	# define calibration data
	v_T_max = (np.mean(data['raw'][7730:7777]))
	T_max = 292 # K

	v_T_min = 0.25*(np.mean(data['raw'][8278:8311]) 
		+ np.mean(data['raw'][8343:8388])
		+ np.mean(data['raw'][8431:8476])
		+ np.mean(data['raw'][8523:8573]))
	T_min = 77.5 # K

	# calibration function
	def fnc(signal):
		return (T_max - T_min)/(v_T_max - v_T_min)*(signal - v_T_min) + T_min

	# print data used for testing
	print("v_T_min : ", v_T_min)
	print("v_T_max : ", v_T_max)

	print(f"T_min ({T_min}) : ", fnc(v_T_min))
	print(f"T_max ({T_max}) : ", fnc(v_T_max))

	# plot calibration curve
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(np.linspace(-0.5, 3, 1000), fnc(np.linspace(-0.5, 3, 1000)))
	ax.scatter([v_T_min, v_T_max], [fnc(v_T_min), fnc(v_T_max)])

	return fnc

def get_system_response(plot=False):
	# get data
	data = get_data("calibration")

	# get min max temperature and signals
	v_T_max = 0.5*(np.mean(data['a'][55:150]) + np.mean(data['a'][320:356]))
	v_T_min = 0.25*(np.mean(data['a'][770:820]) + np.mean(data['a'][861:940]) + np.mean(data['a'][970:1000]) + np.mean(data['a'][1060:1240]))
	T_max = 292 # K
	T_min = 77.5 # K

	# compute slope and system noise
	response_slope = (v_T_max - v_T_min)/(T_max - T_min)
	T_noise = v_T_max/response_slope - T_max

	# compute calibration function for comparison
	def fnc(signal):
		return (T_max - T_min)/(v_T_max - v_T_min)*(signal - v_T_min) + T_min

	# print data for verification
	print("Response slope : ", response_slope)
	print("Noise temperature : ", T_noise)
	print("Noise temperature 2 : ", (T_max*v_T_min - T_min*v_T_max)/(v_T_max - v_T_min))

	print("Tmin = ", v_T_min/response_slope - T_noise)
	print("T_max = ", v_T_max/response_slope - T_noise)

	print('dT min 1 = ', (83.5)/(np.sqrt(1 * 1e9)))

	# plot calibrated temperature
	if plot == True:
		fig = plt.figure()
		ax1 = fig.add_subplot(211)

		ax1.plot(data['a'], '-g', label="Sky")
		ax1.plot(data['aref'], '-k', label="System noise")

		ax1.set_ylabel("Raw signal [V]")

		ax2 = fig.add_subplot(212)

		ax2.plot((data['a'])/response_slope - T_noise, '-g')
		ax2.plot(data['aref']/response_slope - T_noise, '-k')

		ax2.set_ylabel("Calibrated \n temperature [K]")
		ax2.set_xlabel("Point [-]")

		ax1.tick_params(labelbottom=False)

		ax1.legend()

def compute_calibration_errors():
	""" Computes the calibration errors function. """
	data = get_data("scan")

	# initial measurement uncertainties
	dT = 0.1 # K
	ds = 1e-3 # V

	# lower temperature
	v_T_max = (np.mean(data['raw'][7730:7777]))
	T_max = 292 # K

	v_T_min = 0.25*(np.mean(data['raw'][8278:8311]) 
		+ np.mean(data['raw'][8343:8388])
		+ np.mean(data['raw'][8431:8476])
		+ np.mean(data['raw'][8523:8573]))
	T_min = 77.5 # K

	# reference signal
	v = np.linspace(-3, 5, 100)

	# Differentiation method
	# compute individual variances due to errors of Tmin, Tmax, ...
	T_mean 	= (T_max - T_min)/(v_T_max - v_T_min)*(v - v_T_min) + T_min
	dT_max 	= ((v - v_T_min)/(v_T_max - v_T_min))**2*dT**2
	dT_min 	= ((v - v_T_max)/(v_T_max - v_T_min))**2*dT**2
	ds_max 	= ((v - v_T_min)/(v_T_max - v_T_min)**2*(T_max - T_min))**2*ds**2
	ds_min 	= ((v - v_T_max)/(v_T_max - v_T_min)**2*(T_max - T_min))**2*ds**2
	dv 		= ((T_max - T_min)/(v_T_max - v_T_min))**2*ds**2

	# compute std deviation using differentiation 
	sigma_T_direct = np.sqrt(dT_max + dT_min + ds_min + ds_max + dv)

	# MC simulation method
	T = 0
	N = 1e5 # number of steps


	sigma_T_mc = np.zeros(int(N))

	if not path.exists("data//mc_calibration_err.pickle"):
		print("Couldn't find data file mc_calibration_err, propagating uncertainties : ")
		
		q = tqdm(total=int(N))
		for i in range(int(N)):
			v_T_max_s = v_T_max + np.random.normal(0, ds)
			v_T_min_s = v_T_min + np.random.normal(0, ds)
			T_min_s = T_min + np.random.normal(0, dT)
			T_max_s = T_max + np.random.normal(0, dT)

			v_s = v + np.random.normal(0, ds)

			T += ((T_max_s - T_min_s)/(v_T_max_s - v_T_min_s)*(v_s - v_T_min_s) + T_min_s - T_mean)**2
			q.update(1)
			q.set_description(f"Calibration error computations - MC step {i}")

		q.close()

		# compute std deviation
		sigma_T_mc = np.sqrt(T/N)

		save_data(sigma_T_mc, "data//mc_calibration_err.pickle")
	else:
		# err_T_cmb2 = 0.756
		# T_err2 = np.array([0.24953701, 0.31273704, 0.22328032, 0.19658345, 0.40695419, 0.45139471,0.48034045, 0.42414219, 0.43971808, 0.6742288,  0.69401968, 0.74241193,0.47352462, 0.50923236, 0.56287675, 0.51286906, 0.48060466, 0.3435334,0.47686457, 0.49917934, 0.51802714, 0.49370046, 0.20776066, 0.2334345,0.37788319, 0.4225535 , 0.40809455, 0.30338645, 0.34261009, 0.4951771,0.51222118, 0.56071152, 0.61962796, 0.42713031, 0.42209044, 0.4082725,0.55648841, 0.55931063, 0.62743773, 0.33914109, 0.35124481])
		sigma_T_mc = load_data("data//mc_calibration_err.pickle")


 	# plot results
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(v, sigma_T_direct*2, "-r", label="Analytical")
	ax.plot(v, sigma_T_mc*2, ":k", label="MC simulation")

	ax.set_ylabel("Temperature \n error [K]")
	ax.set_xlabel("Signal [V]")
	ax.legend()

def propagate_uncertainty_temperature_calibration(i_start, i_end, min_elevation=50, mask=True):
	# get data
	data = get_data("scan")

	# measurement errors
	dT = 0.5 # K
	ds = 1e-3 # V
	d_el = 0.5 # deg

	# compute calibration measured temperatures, used as a mean value for the estimate of the real value of the temperature
	# max temperature
	v_T_max = (np.mean(data['raw'][7730:7777]))
	T_max = 292 # K

	# min temperature
	v_T_min = 0.25*(np.mean(data['raw'][8278:8311]) 
		+ np.mean(data['raw'][8343:8388])
		+ np.mean(data['raw'][8431:8476])
		+ np.mean(data['raw'][8523:8573]))
	T_min = 77.5 # K

	# reference signal used for comparison
	v = 1
	# compute mean temperature after calibration
	T_mean = (T_max - T_min)/(v_T_max - v_T_min)*(v - v_T_min) + T_min

	# compute individual variances due to errors of Tmin, Tmax, ...
	dT_max = ((v - v_T_min)/(v_T_max - v_T_min))**2*dT**2
	dT_min = ((v - v_T_max)/(v_T_max - v_T_min))**2*dT**2
	ds_max = ((v - v_T_min)/(v_T_max - v_T_min)**2*(T_max - T_min))**2*ds**2
	ds_min = ((v - v_T_max)/(v_T_max - v_T_min)**2*(T_max - T_min))**2*ds**2
	dv = ((T_max - T_min)/(v_T_max - v_T_min))**2*ds**2

	# compute total std devitation in temperature 
	delta_T1 = np.sqrt(dT_max + dT_min + ds_min + ds_max + dv)
	print("square errors temperature std deviation : ", delta_T1)

	T = 0
	N = 1e5
	for i in range(int(N)):
		v_T_max_s = v_T_max + np.random.normal(0, ds)
		v_T_min_s = v_T_min + np.random.normal(0, ds)
		T_min_s = T_min + np.random.normal(0, dT)
		T_max_s = T_max + np.random.normal(0, dT)

		v_s = v + np.random.normal(0, ds)

		T += ((T_max_s - T_min_s)/(v_T_max_s - v_T_min_s)*(v_s - v_T_min_s) + T_min_s - T_mean)**2

	delta_T2 = np.sqrt(T/N)
	print("MC error  : ", delta_T2)

	# estimate uncertainties measurements 
	data = get_data("scan")

	# get start and end point 
	signal = data['raw'][i_start:i_end]
	el = data['el'][i_start:i_end]

	if np.size(mask) > 1:
		mask = mask[0:i_end-i_start]

	N = 1e5

	T_CMB = np.zeros(int(N))
	var_intersept = 0

	mask_tot = np.logical_and(el >= min_elevation, mask)

	el_ordered = np.unique(np.sort(el[mask_tot]))
	T_mean = np.zeros(len(el_ordered))
	T_sig = np.zeros(len(el_ordered))
	k = 0

	q = tqdm(total=int(N))

	while k < int(N):
		success = 0

		v_T_max_s = v_T_max + np.random.normal(0, ds)
		v_T_min_s = v_T_min + np.random.normal(0, ds)
		T_min_s = T_min + np.random.normal(0, dT)
		T_max_s = T_max + np.random.normal(0, dT)

		el_s = el + np.random.normal(0, d_el, len(el))

		signal_err_s = signal + np.random.normal(0, ds, len(signal))
		signal_err_s = (T_max_s - T_min_s)/(v_T_max_s - v_T_min_s)*(signal_err_s - v_T_min_s) + T_min_s

		mask_tot = np.logical_and(el_s >= min_elevation, mask)
		air_thickness = 1/np.cos((90 - el_s)*np.pi/180.0)

		res = scipy.stats.linregress(air_thickness[mask_tot], y=signal_err_s[mask_tot], alternative='two-sided')

		# compute error bars 
		for i in range(len(el_ordered)):
			if i == 0:
				msk = np.logical_and(el_s < el_ordered[i] + 0.5, mask_tot)
			elif i == len(el_ordered) - 1:
				msk = np.logical_and(el_s >= el_ordered[i] - 0.5, mask_tot)
			else:
				msk = np.logical_and(np.logical_and(el_s >= el_ordered[i] - 0.5, el_s < el_ordered[i] + 0.5), mask_tot)
			
			# discard step if there isn't more than one measurement 
			if(np.size(np.where(msk)) > 1):
				T_mean[i] += np.mean(signal_err_s[msk])
				T_sig[i] += np.var(signal_err_s[msk])

				success = 1
			else:
				success = 0
				break

		# if the step is valid, add to MC data
		if success == 1:
			T_CMB[k] = res.intercept
			var_intersept += res.intercept_stderr**2

			q.update(1)
			q.set_description(f'Instrumental error estimation : MC Sample n={k}')

			k += 1

	q.close()

	T_cmb_std = np.sqrt(np.var(T_CMB) + var_intersept/N)

	print("sigma T_cmb : ", T_cmb_std)
	print("mean : ", list(T_mean/N))
	print("var : ", list(np.sqrt(T_sig/N)))

	return T_cmb_std, np.sqrt(T_sig/N)


def data_analysis(cal_function, i_start, i_end, min_elevation=70, mask=True, T_err=None):
	# get data
	data = get_data("scan")

	# get start and end point 
	signal = cal_function(data['raw'][i_start:i_end])
	el = data['el'][i_start:i_end]

	if np.size(mask) > 1:
		mask = mask[0:i_end-i_start]

	mask_tot = np.logical_and(el >= min_elevation, mask)
	air_thickness = 1/np.cos((90 - el)*np.pi/180.0)

	# perform linear regression
	res = scipy.stats.linregress(air_thickness[mask_tot], y=signal[mask_tot], alternative='two-sided')

	# print temperature results
	print("results T_CMB : ", res.intercept, " (stat error : ", res.intercept_stderr, ")")

	# split arrays if a data mask is used
	indices = np.nonzero(mask_tot[1:] != mask_tot[:-1])[0] + 1
	n_arr = np.split(np.arange(0, len(signal)), indices)
	n_arr = n_arr[0::2] if mask_tot[0] else n_arr[1::2]

	b = np.split(signal, indices)
	b = b[0::2] if mask_tot[0] else b[1::2]

	# plot data and statistical analysis results 
	fig = plt.figure()

	fig.suptitle("az = " + str(data["az"][i_start + 1000]) + " deg")
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	ax1.plot(signal, "forestgreen")
	for i in range(len(b)):
		ax1.plot(n_arr[i], b[i], color="darkorange")

	ax1.set_xlabel("Points [-]")
	ax1.set_ylabel("Temperature [K]")

	# compute mean of all data points for eacjh elevation
	el_ordered = np.unique(np.sort(el[mask_tot]))
	T_mean = np.zeros(len(el_ordered))
	T_sig = np.zeros(len(el_ordered))

	for i in range(len(el_ordered)):  
		msk = np.logical_and(el == el_ordered[i], mask_tot)

		T_mean[i] = np.mean(signal[msk])
		T_sig[i] = np.sqrt(np.var(signal[msk]))

	ax2.plot([0.8, 1.3], [res.intercept +0.8*res.slope, res.intercept + 1.3*res.slope], "-", color="steelblue", label="Best Fit")

	# add errors 
	if T_err is not None:
		T_err = T_err[(min_elevation-50):]

		ax2.errorbar(1/np.cos((90 - el_ordered)*np.pi/180.0), T_mean, xerr=np.sin((90 - el_ordered)*np.pi/180.0)/np.cos((90 - el_ordered)*np.pi/180.0)**2*0.5*np.pi/180.0, yerr=3*T_err, fmt=".k", markeredgecolor="black", mew=.5, ms=10, mfc="red", elinewidth=.75, capsize=0, label="Measurements")
	else:
		ax2.errorbar(1/np.cos((90 - el_ordered)*np.pi/180.0), T_mean, xerr=np.sin((90 - el_ordered)*np.pi/180.0)/np.cos((90 - el_ordered)*np.pi/180.0)**2*0.5*np.pi/180.0, fmt=".k", markeredgecolor="black", mfc="red", mew=.5, elinewidth=.75, capsize=0, ms=10, label="Measurements")

	ax2.legend()
	ax2.set_xlabel("Air mass $A(z)$ [-]")
	ax2.set_ylabel("Temperature [K]")

	return res.intercept


def calibrate():
	""" Compute calibration function. """
	data = get_data("calibration")

	# lower temperature
	v_T_max = 0.5*(np.mean(data['raw'][55:150]) + np.mean(data['raw'][320:356]))
	T_max = 293 # K

	v_T_min = 0.25*(np.mean(data['raw'][770:820]) 
		+ np.mean(data['raw'][861:940])
		+ np.mean(data['raw'][970:1000])
		+ np.mean(data['raw'][1060:1240]))
	T_min = 77 # K

	# calibration function
	def fnc(signal):
		return (T_max - T_min)/(v_T_max - v_T_min)*(signal - v_T_min) + T_min

	# print data for verification
	print("Verification : ")
	print("A : ",  (T_max - T_min)/(v_T_max - v_T_min))
	print("B : ", T_min)
	print("v0 : ", v_T_min)

	print("v_T_min : ", v_T_min)
	print("v_T_max : ", v_T_max)

	print(f"T_min ({T_min}) : ", fnc(v_T_min))
	print(f"T_max ({T_max}) : ", fnc(v_T_max))


	# plot calibration function
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(np.linspace(-0.5, 3, 1000), fnc(np.linspace(-0.5, 3, 1000)), "-r")
	ax.plot([v_T_min, v_T_max], [fnc(v_T_min), fnc(v_T_max)], "+b")

	ax.set_ylabel("Temperature [K]")
	ax.set_xlabel("Signal [V]")

	return fnc

def main():
	""" Main data analysis loop. """
	# General variables
	N = len(get_data("scan"))
	
	# get_system_response(plot=True)
	print("\nCalibrating the instrument : ")
	cal_function = calibrate()
	print("Done.")

	print("\nPlotting the main lobe estimation : ")
	plot_lobe(cal_function)	
	print("Done.")

	print("\nPlotting scan : ")
	plot_scan(cal_function)
	print("Done.")

	print("\nPropagating calibration errors : ")
	compute_calibration_errors()
	print("Done.")

	# -----------------------------------------------------------------------------
	# 								DATA ANALYSIS
	# -----------------------------------------------------------------------------
	print("\nMeasurements :")

	inds = [[100, 1850], [1860, 3660], [3676, 5433], [5470, 7224]]
	
	# -----------------------------------------------------------------------------
	# 								First scan (DISCARDED)
	# -----------------------------------------------------------------------------
	# mask1 = np.full(N, True)
	# mask1[572:700] = False

	# data_analysis(cal_function, *inds[0], min_elevation=55, mask=mask1)



	# -----------------------------------------------------------------------------
	# 								Second scan
	# -----------------------------------------------------------------------------

	# results from the propagate_uncertainty_temperature_calibration function :
	# propagate_uncertainty_temperature_calibration(*inds[1], min_elevation=50, mask=True)
	if not path.exists("data//err_T_cmb2.pickle"):
		print("Couldn't find data file err_T_cmb2, propagating uncertainties : ")
		err_T_cmb2, T_err2 = propagate_uncertainty_temperature_calibration(*inds[1], min_elevation=50)
		save_data([err_T_cmb2, T_err2], "data//err_T_cmb2.pickle")
	else:
		# err_T_cmb2 = 0.756
		# T_err2 = np.array([0.24953701, 0.31273704, 0.22328032, 0.19658345, 0.40695419, 0.45139471,0.48034045, 0.42414219, 0.43971808, 0.6742288,  0.69401968, 0.74241193,0.47352462, 0.50923236, 0.56287675, 0.51286906, 0.48060466, 0.3435334,0.47686457, 0.49917934, 0.51802714, 0.49370046, 0.20776066, 0.2334345,0.37788319, 0.4225535 , 0.40809455, 0.30338645, 0.34261009, 0.4951771,0.51222118, 0.56071152, 0.61962796, 0.42713031, 0.42209044, 0.4082725,0.55648841, 0.55931063, 0.62743773, 0.33914109, 0.35124481])
		err_T_cmb2, T_err2 = load_data("data//err_T_cmb2.pickle")

	# data analysis
	T_cmb2 = data_analysis(cal_function, *inds[1], min_elevation=50, T_err=T_err2)	



	# -----------------------------------------------------------------------------
	# 								Third scan
	# -----------------------------------------------------------------------------
	mask2 = np.full(N, True)
	mask2[790:870] = False

	# results from the propagate_uncertainty_temperature_calibration function :
	# propagate_uncertainty_temperature_calibration(inds[2][0], inds[2][1], min_elevation=50, mask=mask2)
	if not path.exists("data//err_T_cmb3.pickle"):
		print("Couldn't find data file err_T_cmb3, propagating uncertainties : ")

		err_T_cmb3, T_err3 = propagate_uncertainty_temperature_calibration(*inds[2], min_elevation=50, mask=mask2)
		save_data([err_T_cmb3, T_err3], "data//err_T_cmb3.pickle")
	else:
		# err_T_cmb3 = 0.786
		# T_err3 = np.array([0.36437024773488563, 0.35711858736309726, 0.3805492474951069, 0.3292077979343326, 0.36526719621407777, 0.3550057549342811, 0.4247805417443455, 0.3284789516425143, 0.47572838703157194, 0.42128744828864745, 0.4386408622863879, 0.4200763487172806, 0.34475789469905405, 0.36531653570458766, 0.3925840164896263, 0.38199551764272255, 0.40357951073520965, 0.3533211140379978, 0.3904537932935995, 0.6623558797775972, 0.7120402169314204, 0.7143985761872609, 0.45045492936277387, 0.4389639559518251, 0.659923513579823, 0.5735447571052207, 0.5059042687965163, 0.3989035372491346, 0.569982159176515, 1.0501844182815758, 1.0637697681023905, 1.0900179578768867, 0.34672200747253745, 0.3158391931109838, 0.34242465050930804, 0.47914989499484467, 0.5002271812911171, 0.5436902659343428, 0.3971266033277998, 0.37285297942901097, 0.3461997655640862])
		err_T_cmb3, T_err3 = load_data("data//err_T_cmb3.pickle")

	# data analysis
	T_cmb3 = data_analysis(cal_function, inds[2][0], inds[2][1], min_elevation=50, mask=mask2, T_err=T_err3)	



	# -----------------------------------------------------------------------------
	# 								Fourth scan
	# -----------------------------------------------------------------------------

	# results from the propagate_uncertainty_temperature_calibration function :
	if not path.exists("data//err_T_cmb4.pickle"):
		print("Couldn't find data file err_T_cmb4, propagating uncertainties : ")

		err_T_cmb4, T_err4 = propagate_uncertainty_temperature_calibration(*inds[3], min_elevation=50)
		save_data([err_T_cmb4, T_err4], "data//err_T_cmb4.pickle")
	else:
		# err_T_cmb4 = 0.775
		# T_err4 = np.array([0.604857498985673, 0.7138288550189172, 0.7966567138954004, 0.7247643405726972, 1.1402977680350888, 0.8393219832292129, 0.9723468021813229, 0.906725016315721, 0.8561618832901373, 0.8349797218918351, 0.8630136653286884, 0.8428654330475047, 0.7235724110631478, 0.5113329867408285, 0.6872586739073179, 0.4275591548769679, 0.4919186291302827, 0.42921808639645254, 0.36737404974093735, 0.5258294008372986, 0.422494746225907, 0.40078953787372407, 0.40006705422424393, 0.3941832848150002, 0.4800804861875626, 0.41991983782576053, 0.40964504919989275, 0.6580131693299073, 0.7872946740465391, 0.6676877484182941, 0.7036490768515036, 0.6611298278355041, 0.32710662494473325, 0.41351311858453127, 0.368838003260891, 0.28917767558239177, 0.24834714653962078, 0.2966170799615092, 0.35369300248066554, 0.3077646743144729, 0.26247084894128675])
		err_T_cmb4, T_err4 = load_data("data//err_T_cmb4.pickle")

	# data analysis
	T_cmb4 = data_analysis(cal_function, *inds[3], min_elevation=50, T_err=T_err4)	
	
	print("Done.")

	# -----------------------------------------------------------------------------
	# 								  Results 
	# -----------------------------------------------------------------------------

	print("CMB temperature : ", (T_cmb2 + T_cmb3 + T_cmb4)/3)
	print("CMB temperature error : ", 2*np.sqrt((err_T_cmb2**2 + err_T_cmb3**2 + err_T_cmb4**2)/3))

	plt.show()

def save_data(data, path):
	file = open(path, 'wb')
	pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(path):
	file = open(path, 'rb')
	return pickle.load(file)

if __name__ == "__main__":
	main()

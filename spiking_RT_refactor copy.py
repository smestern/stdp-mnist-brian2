'''
Created on 15.12.2014

@author: Peter U. Diehl
'''


import numpy as np
import os.path
import pickle as pickle
import brian2 as b
from struct import unpack
from brian2 import *
from ni_interface.ni_brian2 import *
import time

defaultclock.dt = 0.1*ms

set_device('cpp_standalone', build_on_run=False)
#device = get_device()

# specify the location of the MNIST data
MNIST_data_path = '/home/smestern/brian2_SDTP/stdp-mnist-brian2/MNIST/'

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------

def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_matrix_from_file(fileName, n_src, n_tgt):
    readout = np.load(fileName)
    print( readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def save_connections():
    print( 'save connections')
    conn = connections_xeae
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save(data_path + 'weights/XeAe', connListSparse)

def save_theta():
    print( 'save theta')
    np.save(data_path + 'weights/theta_A', neuron_groups_ae.theta)

def normalize_weights():
    len_source = len(connections_xeae.source)
    len_target = len(connections_xeae.target)
    connection = np.zeros((len_source, len_target))
    connection[connections_xeae.i, connections_xeae.j] = connections_xeae.w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = 78./colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    connections_xeae.w = temp_conn[connections_xeae.i, connections_xeae.j]


def run_network(device, i):
    if i != 0: #we do not need to propagate the weights and theta on the first run
        #get all possible inputs

        device.run(device.project_dir, run_args={input_groups_xe.rates: rate * Hz, 
        neuron_groups_ae.theta: neuron_groups_ae.theta, #neuron_groups_ai.theta: neuron_groups_ai.theta, #propagate the thresholds
        connections_xeae.w: connections_xeae.w,
        neuron_groups_ae.v: neuron_groups_ae.v, neuron_groups_ai.v: neuron_groups_ai.v, #propagate the membrane potentials
        neuron_groups_ae.ge : neuron_groups_ae.ge, neuron_groups_ai.ge : neuron_groups_ai.ge, #propagate the synaptic conductances
        })
    else:
        device.run(device.project_dir, run_args={input_groups_xe.rates: rate * Hz})
    return


def plot_state(state_monitor, spike_monitor, j):
    #plot I total and v
    spike_dict = spike_monitor.spike_trains()
    figure(figsize=(10, 4), num=0)
    clf()
    subplot(211)
    #plot(state_monitor.t / ms, state_monitor.I_total[0] / nA)
    
    xlabel('Time (ms)')
    ylabel('I in (nA)')
    subplot(212)
    plot(state_monitor.t / ms, state_monitor.v[0] / mV)
    scatter(spike_dict[0] / ms, np.ones(len(spike_dict[0])) * -50, color='r')
    xlabel('Time (ms)')
    ylabel('v (mV)')
    title(f"Neuron {j}")
    pause(0.01)



#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------
DYN_CLAMP = False
test_mode = True

np.random.seed(0)
data_path = ''
'''
if test_mode:
    num_examples = 10000
else:
    num_examples = 60000 * 3
'''
num_examples =  600 * 1 # 추가
ending    = '' # 추가
n_output  = 10 # 추가
n_e =10
n_i = n_e
n_input = 784

input_intensity = 2.
start_input_intensity = input_intensity

previous_spike_count = np.zeros(n_e)
input_numbers = [0] * num_examples

if test_mode:
        result_monitor = np.zeros((num_examples,n_e))


single_example_time =   0.35 * b.second
resting_time = 0.25 * b.second
runtime = num_examples * (single_example_time + resting_time)

use_testing_set       = True # add

v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = 'v=-45.*mV'
v_thresh_i = 'v>-40.*mV'
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms


tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b.mV
v_thresh_e = '(v>(theta - offset + -52.*mV + thresh_offset_RT)) and (timer>refrac_e)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        I_total = clip(I_synE + I_synI, -250*pA, 250*pA)                                : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        thresh_offset_RT : volt
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'


neuron_groups_ae = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups_ai = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')
#------------------------------------------------------------------------------
# attach the dynamic clamp'd neuron to the network
#------------------------------------------------------------------------------
if DYN_CLAMP:
    dyn_clamp_neuron, group = attach_neuron(neuron_groups_ae, 0, 'v', 'I_total', dt=defaultclock.dt)
    dyn_clamp_neuron.thresh_offset_RT = 20*b.mV
    dyn_clamp_neuron.v = -65. * b.mV
#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
print( 'create neuron group A')

neuron_groups_ae.v = v_rest_e - 40. * b.mV
neuron_groups_ai.v = v_rest_i - 40. * b.mV
if test_mode:
    neuron_groups_ae.theta = np.load(data_path + 'weights/theta_A.npy') * b.volt
else:
    neuron_groups_ae.theta = np.ones((n_e)) * 20.0*b.mV

print( 'create recurrent connections')
weightMatrix = get_matrix_from_file(data_path + 'random/AeAi.npy', n_e, n_i)
connections_aeai = b.Synapses(neuron_groups_ae, neuron_groups_ai, model='w : 1', on_pre='ge_post += w')
#here we need to use the all-to-all connection, however we want to generate this using numpy. 
Ai_idxs = np.arange(n_i)
Ae_idxs = np.arange(n_e)
#generate all to all conectivity
AeAi_idxs = np.array(np.meshgrid(Ae_idxs, Ai_idxs)).T.reshape(-1,2)

connections_aeai.connect(i=AeAi_idxs[:,0], j=AeAi_idxs[:,1]) # all-to-all connection
connections_aeai.w = weightMatrix[AeAi_idxs[:,0], AeAi_idxs[:,1]]

weightMatrix = get_matrix_from_file(data_path + 'random/AiAe.npy', n_i, n_e)
connections_aiae = b.Synapses(neuron_groups_ai, neuron_groups_ae, model='w : 1', on_pre='gi_post += w')
#generate all to all conectivity
AiAe_idxs = np.array(np.meshgrid(Ai_idxs, Ae_idxs)).T.reshape(-1,2)
connections_aiae.connect(i=AiAe_idxs[:,0], j=AiAe_idxs[:,1]) # all-to-all connection
connections_aiae.w = weightMatrix[AiAe_idxs[:,0], AiAe_idxs[:,1]]

print( 'create monitors for Ae')
spike_counters_ae = b.SpikeMonitor(neuron_groups_ae)
state_monitors_ae = b.StateMonitor(neuron_groups_ae, ['v', 'ge', 'gi'], record=[0,1])
#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
input_groups_xe = b.NeuronGroup(n_input, model='rates:Hz (constant)', threshold='rand()<rates*dt')
input_groups_xe.rates = np.full((n_input), 0.0, dtype=np.float32) * b.Hz

print( 'create connections between X and A')
if test_mode:
    weightMatrix = get_matrix_from_file(data_path + 'weights/XeAe.npy', n_input, n_e)
else:
    weightMatrix = get_matrix_from_file(data_path + 'random/XeAe.npy', n_input, n_e)
model = 'w : 1'
pre = 'ge_post += w'
post = ''

if not test_mode:
    print( 'create STDP for connection XeAe')
    model += eqs_stdp_ee
    pre += '; ' + eqs_stdp_pre_ee
    post = eqs_stdp_post_ee

connections_xeae = b.Synapses(input_groups_xe, neuron_groups_ae,
                                            model=model, on_pre=pre, on_post=post)
minDelay = 0*b.ms
maxDelay = 10*b.ms
deltaDelay = maxDelay - minDelay

Xe_idxs = np.arange(n_input)
Ae_idxs = np.arange(n_e)
#generate all to all conectivity
XeAe_idxs = np.array(np.meshgrid(Xe_idxs, Ae_idxs)).T.reshape(-1,2)
connections_xeae.connect(i=XeAe_idxs[:,0], j=XeAe_idxs[:,1]) # all-to-all connection
#we need to pregenerate the delays
delays = np.random.rand(n_input, n_e) * deltaDelay + minDelay

#connections_xeae.connect(True) # all-to-all connection
connections_xeae.delay = delays
connections_xeae.w = weightMatrix[XeAe_idxs[:,0], XeAe_idxs[:,1]]

#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

# net = Network()
# for obj_list in [neuron_groups, input_groups, connections, spike_counters, state_monitors]:
#     for key in obj_list:
#         net.add(obj_list[key])
if DYN_CLAMP:
    device = init_neuron_device(device)
zero_arr = np.zeros(n_input)
#param_rate = initialize_parameter(input_groups_xe.rates, zero_arr )
run(single_example_time, report='text')

device.build(compile=True, run=False, debug=False)
#device.run(device.project_dir, with_output=False, run_args=[])




#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
start_time = time.time()
#get the network run duration
#network = list(device.networks)[0]

numbers_to_obs = np.arange(4).tolist()

i = 0 #number of times loop has run, mostly for checking the iter
j = 0 #j is example number
while j < (int(num_examples)):
    if test_mode and (testing['y'][j%60000][0] not in numbers_to_obs):
        
        j += 1
        continue
    elif not test_mode and (training['y'][j%10000][0] not in numbers_to_obs):
        j += 1
        continue
    if test_mode:
        rate = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        if i > 0:
            normalize_weights()
        rate = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    #set_the new rates, propagate the weights and theta
    run_network(device, i)
    plot_state(state_monitors_ae, spike_counters_ae, j)
    current_spike_count = np.asarray(spike_counters_ae.count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters_ae.count[:])
    if np.sum(current_spike_count) < 1:
        input_intensity += 1
        rate = zero_arr
        run_network(device, i)
    else:
        if test_mode:
            result_monitor[j,:] = current_spike_count
            input_numbers[j] = testing['y'][j%10000][0]
        if j % 100 == 0 and j > 0:
            print( 'runs done:', j, 'of', int(num_examples))
        rate = zero_arr
        input_intensity = start_input_intensity
        #set_parameter_value(param_rate, zero_arr )
        run_network(device, i)
        j += 1

        
    i+=1

print( 'total time:', (time.time() - start_time)/60)

#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print( 'save results')
if not test_mode:
    save_theta()
    save_connections()
else:
    np.save(data_path + './activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + './activity/inputNumbers' + str(num_examples), input_numbers)
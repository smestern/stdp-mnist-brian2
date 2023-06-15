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
import time
import matplotlib.pyplot as plt


prefs.codegen.target = 'cython'

# specify the location of the MNIST data
MNIST_data_path = './MNIST/'

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
    conn = connections['XeAe']
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save(data_path + 'weights/XeAe', connListSparse)
    conn = connections['AeAe']
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save(data_path + 'weights/AeAe', connListSparse)

def save_theta():
    print( 'save theta')
    np.save(data_path + 'weights/theta_A', neuron_groups['Ae'].theta)

def normalize_weights(connection_name='XeAe', mute_neurons=[]):
    len_source = len(connections[connection_name].source)
    len_target = len(connections[connection_name].target)
    connection = np.zeros((len_source, len_target))
    connection[connections[connection_name].i, connections[connection_name].j] = connections[connection_name].w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = (len_source*(0.1))/colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    for nrn in mute_neurons:
        temp_conn[:,nrn] = 0.0
    connections[connection_name].w = temp_conn[connections[connection_name].i, connections[connection_name].j]

def plot_net(state_monitor):
    plt.figure(1)
    plt.clf()
    fig, ax = plt.subplots(3,1, num=1)
    ax[0].plot(state_monitor.t/b.second, state_monitor.v[0]/b.mV)
    plt.xlabel('Time (s)')
    plt.ylabel('v (mV)')
    ax[1].plot(state_monitor.t/b.second, state_monitor.ge[0], alpha=0.7,)
    plt.pause(0.5)


#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)


#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = False
EE_SDTP = True
XE_SDTP = True
np.random.seed(0)
data_path = ''
'''
if test_mode:
    num_examples = 10000
else:
    num_examples = 60000 * 3
'''
num_examples =  600 * 1 
ending    = '' 


n_input = 784
n_e = 40
n_i = n_e

single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)

use_testing_set       = True # add

v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = 'v=-45.*mV'
v_thresh_i = 'v>-40.*mV'
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

input_intensity = 3.
start_input_intensity = input_intensity

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
v_thresh_e = '(v>(theta - offset + -52.*mV)) and (timer>refrac_e)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
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

neuron_groups = {}
input_groups = {}
connections = {}
spike_counters = {}
state_monitors = {}
if test_mode:
    result_monitor = np.zeros((num_examples,n_e))

neuron_groups['Ae'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups['Ai'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')


#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
print( 'create neuron group A')

neuron_groups['Ae'].v = v_rest_e - 40. * b.mV
neuron_groups['Ai'].v = v_rest_i - 40. * b.mV
if test_mode:
    neuron_groups['Ae'].theta = np.load(data_path + 'weights/theta_A.npy') * b.volt
else:
    neuron_groups['Ae'].theta = np.ones((n_e)) * 20.0*b.mV

print( 'create recurrent connections')
weightMatrix = get_matrix_from_file(data_path + 'random/AeAi.npy', n_e, n_i)
connections['AeAi'] = b.Synapses(neuron_groups['Ae'], neuron_groups['Ai'], model='w : 1', on_pre='ge_post += w')
connections['AeAi'].connect(True) # all-to-all connection
connections['AeAi'].w = weightMatrix[connections['AeAi'].i, connections['AeAi'].j]

weightMatrix = get_matrix_from_file(data_path + 'random/AiAe.npy', n_i, n_e)
connections['AiAe'] = b.Synapses(neuron_groups['Ai'], neuron_groups['Ae'], model='w : 1', on_pre='gi_post += w')
connections['AiAe'].connect(True) # all-to-all connection
connections['AiAe'].w = weightMatrix[connections['AiAe'].i, connections['AiAe'].j]


if test_mode:
    weightMatrix = get_matrix_from_file(data_path + 'weights/AeAe.npy', n_e, n_e)
else:
    weightMatrix = get_matrix_from_file(data_path + 'random/AeAe.npy', n_e, n_e)
model = 'w : 1'
pre = 'ge_post += w'
post = ''

if not test_mode and EE_SDTP:
    print( 'create STDP for connection AeAe')
    model += eqs_stdp_ee
    pre += '; ' + eqs_stdp_pre_ee
    post = eqs_stdp_post_ee

connections['AeAe'] = b.Synapses(neuron_groups['Ae'], neuron_groups['Ae'],
                                            model=model, on_pre=pre, on_post=post)
minDelay = 0*b.ms
maxDelay = 10*b.ms
deltaDelay = maxDelay - minDelay

connections['AeAe'].connect(True) # all-to-all connection
connections['AeAe'].delay = 'minDelay + rand() * deltaDelay'
connections['AeAe'].w = weightMatrix[connections['AeAe'].i, connections['AeAe'].j]


print( 'create monitors for Ae')
spike_counters['Ae'] = b.SpikeMonitor(neuron_groups['Ae'])
state_monitors['Ae'] = b.StateMonitor(neuron_groups['Ae'], ['v', 'ge', 'theta'], record=range(5))

#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
input_groups['Xe'] = b.PoissonGroup(n_input, 0*Hz)

print( 'create connections between X and A')
if test_mode:
    weightMatrix = get_matrix_from_file(data_path + 'weights/XeAe.npy', n_input, n_e)
else:
    weightMatrix = get_matrix_from_file(data_path + 'random/XeAe.npy', n_input, n_e)
model = 'w : 1'
pre = 'ge_post += w'
post = ''

if not test_mode and XE_SDTP:
    print( 'create STDP for connection XeAe')
    model += eqs_stdp_ee
    pre += '; ' + eqs_stdp_pre_ee
    post = eqs_stdp_post_ee

connections['XeAe'] = b.Synapses(input_groups['Xe'], neuron_groups['Ae'],
                                            model=model, on_pre=pre, on_post=post)
minDelay = 0*b.ms
maxDelay = 10*b.ms
deltaDelay = maxDelay - minDelay

connections['XeAe'].connect(True) # all-to-all connection
connections['XeAe'].delay = 'minDelay + rand() * deltaDelay'
connections['XeAe'].w = weightMatrix[connections['XeAe'].i, connections['XeAe'].j]


#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

net = Network()
for obj_list in [neuron_groups, input_groups, connections, spike_counters, state_monitors]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(10)
input_numbers = [0] * num_examples
input_groups['Xe'].rates = 0 * Hz
net.run(0*second)
j = 0
np.random.seed(42)
numbers_to_obs = np.arange(4).tolist()


start_time = time.time()
while j < (int(num_examples)):
    if test_mode and (testing['y'][j%60000][0] not in numbers_to_obs):
        j += 1
        continue
    elif not test_mode and (training['y'][j%10000][0] not in numbers_to_obs):
        j += 1
        continue

    if input_intensity > 400:
        j+=1
        input_intensity = start_input_intensity
        print(f"skipped {j}th example")
        continue

    if test_mode:
        rate = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        if XE_SDTP:
            normalize_weights()
        elif EE_SDTP:
            normalize_weights('AeAe')
        rate = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    input_groups['Xe'].rates = rate * Hz
    net.run(single_example_time)
    plot_net(state_monitors['Ae'])
    current_spike_count = np.asarray(spike_counters['Ae'].count[:10]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:10])
    print(f"Running sample {j} with target {training['y'][j%10000][0]} and and found spikes {current_spike_count}", end="\r")
    if np.sum(current_spike_count) < 2:
        input_intensity += 1
        input_groups['Xe'].rates = 0 * Hz
        net.run(resting_time)
    else:
        if test_mode:
            result_monitor[j,:] = current_spike_count
            input_numbers[j] = testing['y'][j%10000][0]
        if j % 100 == 0 and j > 0:
            print( 'runs done:', j, 'of', int(num_examples))
        input_groups['Xe'].rates = 0 * Hz
        net.run(resting_time)
        input_intensity = start_input_intensity
        j += 1


    

    

print( 'simulation time:', time.time() - start_time)
#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print( 'save results')
if not test_mode:
    save_theta()
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
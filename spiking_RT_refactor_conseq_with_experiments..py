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
from sklearn.model_selection import train_test_split
import time
import copy as defaultcopy
import shutil
from dataset_handler import mnistDataHandler as mnistDataHandler
#set the numpy seed
np.random.seed(42)
defaultclock.dt = 0.1*ms



set_device('cpp_standalone', build_on_run=False)
#device = get_device()
# specify the location of the MNIST data
MNIST_data_path = './MNIST/'

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------

def find_trial_num(path):
    #find the trial number
    trial_num = 0
    while os.path.exists(path + 'EXP_1_trial_' + str(trial_num)):
        trial_num += 1
    return trial_num

def find_or_make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_matrix_from_file(fileName, n_src, n_tgt):
    readout = np.load(fileName)
    print( readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def make_exp_paths(EXP_path, neuron_num, trial_num, TEST_MODE=False):
    #make a new folder for the experiment should be /date/neuron_num/EXP_1_trial_num
    base_path = defaultcopy.copy(EXP_path)
    EXP_path = EXP_path + str(neuron_num) + '/'
    EXP_path = find_or_make_path(EXP_path)
    #determine the trial number
    if TEST_MODE:
        trial_num = find_trial_num(EXP_path) - 1
    else:
        trial_num = find_trial_num(EXP_path)
    #make the trial folder
    EXP_path = EXP_path + 'EXP_1_trial_' + str(trial_num) + '/'
    EXP_path = find_or_make_path(EXP_path)
    #copy the weights file to the trial folder, for reference
    if os.path.exists(os.path.abspath(EXP_path+'random/'))==False:
        
        shutil.copytree(os.path.abspath(base_path+'random/'), EXP_path + '/random/')
    find_or_make_path(EXP_path + '/activity/')
    find_or_make_path(EXP_path + '/weights/')
    return EXP_path


#GLOBALS
test_mode = False
DYN_CLAMP = False
BLANKED = False
CLEAR_OUTPUT = True
NUM_EXAMPLES=  300
Nneurons = 400

#settings modified by the experiments
SDTP = False
E_TO_E = True
EE_SDTP =True
MUTE_NEURONS = []


#EXPERIMENTS
EXP_1 = False #No EE with STDP
EXP_2 = False
EXP_3 = False #E TO E with STDP on all connections
EXP_4 = True #E TO E with STDP on E TO E connections only




EXP_1_path = f'./no_EE_w_STDP_{Nneurons}/'
EXP_2_path = f'./E_TO_E_no_STDP_{Nneurons}/'
EXP_3_path = f'./E_TO_E_ALL_STDP_{Nneurons}/'
EXP_4_path = f'./E_TO_E_no_XE_{Nneurons}/'

TEST_MODE = test_mode

neuron_num = 'DEFAULT'
trial_num = 0

if EXP_1:
    EXP_1_path = make_exp_paths(EXP_1_path, neuron_num, trial_num, TEST_MODE)
    data_path= EXP_1
    E_TO_E = False
    SDTP = True
elif EXP_2:
    EXP_2_path = make_exp_paths(EXP_2_path, neuron_num, trial_num, TEST_MODE)
    
elif EXP_3:
    EXP_3_path = make_exp_paths(EXP_3_path, neuron_num, trial_num, TEST_MODE)
    SDTP = True
    E_TO_E = True
    EE_SDTP = True
    data_path = EXP_3_path
    
elif EXP_4:
    EXP_4_path = make_exp_paths(EXP_4_path, neuron_num, trial_num, TEST_MODE)
    data_path = EXP_4_path
    SDTP=False
    MUTE_NEURONS=np.arange(10).tolist()



 
#ACTUAL NETWORK CODE. RUNNING THIS IN A FUNCTION BREAKS IT SO IT IS GLOBAL LMAO
#DONT TOUCH THIS



seed(42)
def save_connections():
    print( 'save connections')
    conn = connections_xeae
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save(data_path + 'weights/XeAe', connListSparse)
    if E_TO_E:
        conn = connections_aeae
        connListSparse = list(zip(conn.i, conn.j, conn.w))
        np.save(data_path + 'weights/AeAe', connListSparse)

def save_theta():
    print( 'save theta')
    np.save(data_path + 'weights/theta_A', neuron_groups_ae.theta)

def normalize_weights_xe(muteNeurons=None):
    len_source = len(connections_xeae.source)
    len_target = len(connections_xeae.target)
    connection = np.zeros((len_source, len_target))
    connection[connections_xeae.i, connections_xeae.j] = connections_xeae.w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    if muteNeurons is not None: 
        for i in range(len(muteNeurons)):
            colSums[muteNeurons[i]] = 1
    colFactors = 78./colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    if muteNeurons is not None:
        for i in range(len(muteNeurons)):
            temp_conn[muteNeurons[i],:] = 0
    connections_xeae.w = temp_conn[connections_xeae.i, connections_xeae.j]

def normalize_weights_ae(muteNeurons=None, muteNeurons_ae_sf=(7. * (784/10))):
    len_source = len(connections_aeae.source)
    len_target = len(connections_aeae.target)
    connection = np.zeros((len_source, len_target))
    connection[connections_aeae.i, connections_aeae.j] = connections_aeae.w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    if muteNeurons is not None:
        Non_mute = np.setdiff1d(np.arange(n_e), muteNeurons)
        colFactors = np.copy(colSums)
        colFactors[Non_mute ] = (1. * (784/len_source))/colSums[Non_mute ]
        colFactors[muteNeurons] = muteNeurons_ae_sf/colSums[muteNeurons]
    else:
        colFactors = 78./colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    connections_aeae.w = temp_conn[connections_aeae.i, connections_aeae.j]


def run_network(device, i):
    if i != 0: #we do not need to propagate the weights and theta on the first run
        #get all possible inputs
        run_args = {}
        for key, val in neuron_groups_ae.get_states().items():
            if key == 'dt' or key == 't' or key == 'N' or key == 't_in_timesteps' or key == 'i' or key=='lastspike':
                continue
            run_args[getattr(neuron_groups_ae, key)] = val
        for key, val in neuron_groups_ai.get_states().items():
            if key == 'dt' or key == 't' or key == 'N' or key == 't_in_timesteps' or key == 'i'or key=='lastspike':
                continue
            run_args[getattr(neuron_groups_ai, key)] = val
        for key, val in connections_xeae.get_states().items():
            if key == 'post1' or key == 'post2' or key == 'post2before' or key=='w':
                
                run_args[getattr(connections_xeae, key)] = val
        if E_TO_E:
            for key, val in connections_aeae.get_states().items():
                if key == 'post1' or key == 'post2' or key == 'post2before' or key=='w':
                    
                    run_args[getattr(connections_aeae, key)] = val
    
        
        run_args.update({input_groups_xe.rates: rate * Hz})
        
        if DYN_CLAMP:
            run_args.update({dyn_clamp_neuron.v: -70*mV})

        device.run(device.project_dir, run_args=run_args)
    else:
        device.run(device.project_dir, run_args={input_groups_xe.rates: rate * Hz})
    return


def plot_state(state_monitor, spike_monitor, j):
    #plot I total and v
    spike_dict = spike_monitor.spike_trains()
    figure(figsize=(10, 5), num=0)
    clf()
    subplot(311)
    title(f"iter {j}")
    scatter( spike_monitor.t[spike_monitor.i<=10] / ms,spike_monitor.i[spike_monitor.i<=10], color='k', marker='.')
    ylim(0, 11)
    xlim(0, 350)
    subplot(312)
    plot(state_monitor.t / ms, state_monitor.I_total[0] / pA)
    twinx(
    )   
    plot(state_monitor.t / ms, state_monitor.theta[0] / mV, color='r')
    xlabel('Time (ms)')
    ylabel('I in (pA)')
    subplot(313)
    plot(state_monitor.t / ms, state_monitor.v[0] / mV)
    scatter(spike_dict[0] / ms, np.ones(len(spike_dict[0])) * -50, color='r')
    xlabel('Time (ms)')
    ylabel('v (mV)')
    
    if j == 0:
     show(block=False)



#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------

np.random.seed(0)
ending    = '' # 추가
n_output  = 10 # 추가
n_e = Nneurons
n_i = n_e
n_input = 784

input_intensity = 4.
start_input_intensity = input_intensity

previous_spike_count = np.zeros(n_e)
input_numbers = [0] * NUM_EXAMPLES

if test_mode:
    result_monitor = np.zeros((NUM_EXAMPLES,10))


single_example_time =   0.35 * b.second
resting_time = 0.25 * b.second
runtime = NUM_EXAMPLES* (single_example_time + resting_time)

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
v_thresh_e = '(v>((theta - offset + -52.*mV)*(1-dyn_clamp) + thresh_offset_RT)) and (timer>refrac_e)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        I_total = clip(I_synE + I_synI, -999*pA, 999*pA)          : amp 
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        thresh_offset_RT : volt
        dyn_clamp : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = (0.1 * (1-dyn_clamp)) + (1 * dyn_clamp) : second'

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
neuron_groups_ae.dyn_clamp = 0
neuron_groups_ai = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')
#------------------------------------------------------------------------------
# attach the dynamic clamp'd neuron to the network
#------------------------------------------------------------------------------
if DYN_CLAMP:
    dyn_clamp_neuron, group = attach_neuron(neuron_groups_ae, 0, 'v', 'I_total', dt=defaultclock.dt)
    dyn_clamp_neuron.dyn_clamp = 1
    dyn_clamp_neuron.thresh_offset_RT = -20*b.mV
    dyn_clamp_neuron.timer = 0*b.ms
    #dyn_clamp_neuron.refrac_e = 0*b.ms
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

# === Generate AeAe connections ===
if E_TO_E:
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

    connections_aeae = b.Synapses(neuron_groups_ae, neuron_groups_ae,
                                                model=model, on_pre=pre, on_post=post)
    minDelay = 0*b.ms
    maxDelay = 10*b.ms
    deltaDelay = maxDelay - minDelay

    connect_aeae = np.array(np.meshgrid(np.arange(n_e), np.arange(n_e))).T.reshape(-1,2)
    connections_aeae.connect(i=connect_aeae[:,0], j=connect_aeae[:,1]) # all-to-all connection
    connections_aeae.delay = 'minDelay + rand() * deltaDelay'
    connections_aeae.w = weightMatrix[connect_aeae[:,0], connect_aeae[:,1]]





print( 'create monitors for Ae')
spike_counters_ae = b.SpikeMonitor(neuron_groups_ae)
state_monitors_ae = b.StateMonitor(neuron_groups_ae, ['v', 'ge', 'theta', 'I_total'], record=[0,1])
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

if not test_mode and SDTP:
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


if DYN_CLAMP:
    device = init_neuron_device(device, scalefactor_out=2.5)
zero_arr = np.zeros(n_input)
#param_rate = initialize_parameter(input_groups_xe.rates, zero_arr )
run(single_example_time, report='text')

device.build(compile=True, run=False,  debug=False)

#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------

numbers_to_obs = np.arange(4).tolist()
training, testing = mnistDataHandler(N=NUM_EXAMPLES, labels=numbers_to_obs, MNIST_data_path=MNIST_data_path).load_data()

#------------------------------------------------------------------------------
# Run the network
#------------------------------------------------------------------------------
start_time = time.time()

spike_times = []


i = 0 #number of times loop has run, mostly for checking the iter
j = 0 #j is example number
k = 0 #k is number of training examples shown

#make sure the network enforces some spikes for the first cell
#every X examples we want at least one spike
#this is to make sure the network doesn't get stuck in a rut
#where it doesn't have any spikes
n0_spikes = 0
bool_pass_n0 = False
n0_Threshold = None


while k < (int(NUM_EXAMPLES)):
    bool_pass_n0 = True
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
            if SDTP:
                normalize_weights_xe(muteNeurons=MUTE_NEURONS)
            if E_TO_E and EE_SDTP:
                normalize_weights_ae(muteNeurons=MUTE_NEURONS)
        rate = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    #set_the new rates, propagate the weights and theta
    run_network(device, i)
    plot_state(state_monitors_ae, spike_counters_ae, k)
    current_spike_count = np.asarray(spike_counters_ae.count[:10]) #- previous_spike_count
    previous_spike_count = np.copy(spike_counters_ae.count[:10])

    #create conditonal for n0 spikes
    if n0_Threshold is not None:
        if current_spike_count[0] >=  1 and n0_spikes < n0_Threshold:
            bool_pass_n0    = True
        if current_spike_count[0] < 1 and n0_Threshold >= n0_Threshold:
            bool_pass_n0    = False
    else:
        bool_pass_n0

    
    if np.sum(current_spike_count) <1 or (np.sum(current_spike_count) >= 3 and bool_pass_n0==False):
        input_intensity += 1
        rate = zero_arr
        run_network(device, i)
    elif np.sum(current_spike_count) >= 1 and bool_pass_n0:
        if test_mode:
            result_monitor[k,:] = current_spike_count
            input_numbers[k] = testing['y'][j%10000][0]
            spike_times.append(spike_counters_ae.t[spike_counters_ae.i==0])
        else:
            #during training dump the weight matrix with_i
            if not test_mode and j % 10 == 0:
                np.save(data_path + 'activity/weights_ae_' + str(j), connections_aeae.w)
        if k % 100 == 0 and k > 0:
            print( 'runs done:', j, 'of', int(NUM_EXAMPLES)) 
        rate = zero_arr
        input_intensity = start_input_intensity
        #set_parameter_value(param_rate, zero_arr )
        run_network(device, i)
        j += 1
        k += 1
        if current_spike_count[0] >=1 :
            n0_spikes =0

        else:
            n0_spikes += 1
    i += 1
    if input_intensity > 100:
        print( 'input intensity too high, network is probably saturated')
        j+=1
        input_intensity = start_input_intensity
        k+=1

    if (time.time() - start_time)/60 > 60:
        print( 'runs done:', j, 'of', int(NUM_EXAMPLES)) 
        break

print( 'total time:', (time.time() - start_time)/60)

#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print( 'save results')
if not test_mode:
    save_theta()
    save_connections()
else:
    np.save(data_path + './activity/resultPopVecs' + str(NUM_EXAMPLES), result_monitor)
    np.save(data_path + './activity/inputNumbers' + str(NUM_EXAMPLES), input_numbers)
    np.save(data_path + './activity/spikeTimes' + str(NUM_EXAMPLES), spike_times)

if CLEAR_OUTPUT:
        #force clear the folder /output
        shutil.rmtree('./output')


    
        



